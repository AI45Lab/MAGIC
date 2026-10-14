# Qwen3.6-27B 与 Ministral 模型适配

[English](README.md)

本模块为 MAGIC 提供 **Qwen3.6-27B** 和 **Ministral-3-14B-Reasoning-2512** 的模型级兼容工具，包含模型加载、SFT token 对齐、checkpoint 转换、原生 reasoning 序列化，以及必要的运行时补丁。调用方可将这些工具接入训练或服务入口。

## 支持范围

| 组件 | Qwen3.6-27B | Ministral-3-14B-Reasoning-2512 |
|---|---|---|
| 模型加载 | 从多模态 checkpoint 中加载语言模型 | 加载原生 Mistral3 外层模型，保留独立输出 head |
| SFT 模板 | 显式 ChatML，保留 assistant 目标文本 | 使用原生 tokenizer 模板，检查 prompt 前缀对齐 |
| checkpoint 转换 | 生成纯文本视图和 vLLM 服务视图 | 规范嵌套参数名称与 LM head 别名 |
| 运行时兼容 | vLLM 配置与权重映射、多轴 FlashAttention position IDs | 纯文本 Pixtral 权重同步、RoPE/YaRN 配置适配 |
| reasoning 表示 | 保留 MAGIC 的原有尖括号标签 | 转换为 `[THINK]...[/THINK]answer`，支持结构化 system 消息和可见回答解析 |

代码中的 `qwen3_5`、`qwen3_5_text` 和 `Qwen3_5ForCausalLM` 是 Qwen3.6 适配使用的运行时架构接口；兼容性检查依据这些架构字段。

## 环境要求

运行时补丁针对 **vLLM 0.18.0**、Transformers **5.x** 模型 API，以及 VERL 的
[`5ba5f2f51281522150a6c1c87e3f841d1d220042`](https://github.com/volcengine/verl/tree/5ba5f2f51281522150a6c1c87e3f841d1d220042)
版本（`0.10.0.dev0`）。应用这些补丁时，应使用单独安装的现代 VERL，而不是仓库内的旧版 `src/verl`。

vLLM 0.18.0 的依赖元数据要求 `transformers<5`，而本适配需要 Transformers 5 的模型 API，因此需要明确配置兼容的隔离环境；仅通过常规依赖解析安装 vLLM 并不足够。补丁面向指定源码版本，不能直接视为适用于任意新版本。

CPU 测试需要 Python 3.12、PyTorch、Transformers 5.3.0 和 safetensors。实际模型加载还需要 CUDA 及对应模型依赖。

## 模型加载与 SFT token 对齐

```python
from magic_model_support.sft import (
    load_qwen_text_model,
    load_ministral_model,
    tokenize_sft_record,
)

# 按模型选择加载器，local_rank 指定 CUDA 设备。
model, tokenizer, config = load_qwen_text_model("/models/Qwen3.6-27B", local_rank=0)
# model, tokenizer, config = load_ministral_model(
#     "/models/Ministral-3-14B-Reasoning-2512", local_rank=0
# )

record = {
    "system": "You are a helpful assistant.",
    "instruction": "Explain the example.",
    "input": "Example text",
    "output": "<think>Explanation</think><answer>Answer</answer>",
}
prompt_ids, input_ids = tokenize_sft_record(record, tokenizer, model_family="qwen")
labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids):]
```

Ministral 使用 `model_family="ministral"` 和已转换的原生 reasoning 目标；可通过 `template_kwargs` 传入模板选项。如果 prompt token 不是完整对话 token 的前缀，适配器会报错，避免 assistant loss mask 错位。加载器返回位于指定 CUDA 设备上的模型；并行方式、优化器及 checkpoint 保存由调用方管理。

## checkpoint 与模板转换

导出 Qwen 语言模型权重，无需另一个 checkpoint 作为参考：

```bash
python -m magic_model_support.scripts.prepare_qwen35_text_only_view \
  --source /models/Qwen3.6-27B \
  --output /models/qwen-text-view

python -m magic_model_support.scripts.prepare_qwen35_vllm_service_view \
  --text-model /models/qwen-text-view \
  --multimodal-config-source /models/Qwen3.6-27B \
  --output /models/qwen-service-view
```

纯文本导出器将重命名后的语言模型 tensor 写入新的分片，保留源文件，因此需要为导出的文本权重预留磁盘空间。服务视图通过链接复用导出的文件，后者需保持可访问。

Ministral 的 SFT 转换器读取一个 JSON 数组文件，记录包含 `instruction`、`input`、`system` 和 `output` 字段：

```bash
python -m magic_model_support.scripts.prepare_ministral_reasoning_sft \
  --input /data/sft-input.json \
  --output /data/sft-ministral.json \
  --system-prompt-file /models/Ministral-3-14B-Reasoning-2512/SYSTEM_PROMPT.txt
```

转换器保留源文件，将规范的尖括号标签转换为原生 reasoning 标记，加入指定的官方 system prompt，遇到不符合格式的记录会在写出前报错。转换保持样本数量和顺序，不决定样本筛选或数据划分。

对导出的 Ministral 模型目录执行以下命令，可规范合并后的权重名称。该操作会原地更新指定目录，目前支持单个 `model.safetensors` 文件：

```bash
python -m magic_model_support.compat.normalize_ministral3_merged \
  --model-dir /models/ministral-export
```

已有的独立 LM head 会被保留；缺少或存在相互冲突的 head 时会报错，不会自动以输入 embedding 替代输出 head。

对于使用 mistral-common 的 API，`magic_model_support.reasoning.mistral_reasoning_api_messages` 可将 system 消息中的 reasoning 示例编码为结构化内容。对于模型的原始原生输出，`parse_native_reasoning_response` 返回 `(reasoning, visible_answer, issue)`，未闭合的 reasoning 不会被当成可见回答。这两个工具只负责格式处理。

## 运行时补丁接入

在独立安装或复制出的包目录中操作，并在启用运行时 hook 前获取包路径：

```bash
VERL_DIR=$(python -c 'import importlib.util; print(next(iter(importlib.util.find_spec("verl").submodule_search_locations)))')
VLLM_DIR=$(python -c 'import importlib.util; print(next(iter(importlib.util.find_spec("vllm").submodule_search_locations)))')
PATCH_DIR="$PWD/magic_model_support/compat/patches"

# 两种模型共用：降低 padded log-probability 和 entropy 的中间显存占用。
patch --forward --batch "$VERL_DIR/workers/engine/fsdp/transformer_impl.py" \
  < "$PATCH_DIR/verl010_no_rmpad_logprob.patch"
```

Qwen 接入：

```bash
patch --forward --batch "$VERL_DIR/workers/engine/fsdp/transformer_impl.py" \
  < "$PATCH_DIR/verl010_qwen35_response_logits.patch"
patch --forward --batch "$VERL_DIR/workers/rollout/vllm_rollout/vllm_async_server.py" \
  < "$PATCH_DIR/verl010_qwen35_text_renderer.patch"

export PYTHONPATH="$PWD/magic_model_support/runtime/qwen:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MAGIC_QWEN35_OUTER_CONFIG_PATH=/models/Qwen3.6-27B/config.json
export MAGIC_QWEN35_FLASH_ATTN_POSIDS_PATCH=1
export MAGIC_RESPONSE_ONLY_LOGITS=1
```

renderer 补丁在 GPU 分配后准备纯文本模型配置。若独立 vLLM 服务通过多模态外层模型加载纯文本权重，可设置 `MAGIC_QWEN35_SFT_PREFIX_ADAPTER=1`。response-only logits 补丁需要 padded、非 fused 执行路径（`use_remove_padding=false`），支持 policy log probabilities 和 entropy 输出。

Ministral 接入：

```bash
patch --forward --batch "$VLLM_DIR/model_executor/models/mistral.py" \
  < "$PATCH_DIR/vllm018_mistral_llama4_rope_parameters.patch"
python -m magic_model_support.compat.patch_vllm018_yarn_mscale \
  --target "$VLLM_DIR/model_executor/layers/rotary_embedding/__init__.py"

export PYTHONPATH="$PWD/magic_model_support/runtime/ministral:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MAGIC_MINISTRAL_TEXT_ONLY_SYNC_PATCH=1
```

按模型选择对应 hook 目录。Ministral hook 处理纯文本权重同步和 padded entropy 计算，vLLM 模型需使用 language-model-only 路径。源码补丁应对匹配版本应用一次；YaRN patcher 也能识别已应用的补丁。

## 相对 MAGIC 原有模型接口的变化

改动集中在模型加载与配置、token 序列化和 checkpoint 参数名称转换。Ministral 在 tokenizer 边界使用原生 reasoning 表示；Qwen 使用纯文本模型视图和对应的 vLLM 配置。显存辅助函数减少中间词表 logits 的分配，同时保留对应的 log-probability 和 entropy 计算。

MAGIC 现有 trainer 和 reward 实现保持不变。这些工具提供模型接入点，不会自动让原有启动脚本支持两个新模型。

## 测试

```bash
python -m unittest discover -s magic_model_support/tests -t . -v
```

测试覆盖模板对齐、原生 reasoning 序列化、checkpoint 视图、LM head 保留、权重名称映射，以及显存辅助函数的数值和梯度。Qwen 导出测试真实加载一个小型 Hugging Face 模型并检查权重完全一致；Ministral 加载器调用使用模拟测试。CPU 单元测试验证适配逻辑，不代表完整 CUDA/分布式执行验证。
