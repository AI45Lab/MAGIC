# Qwen3.6-27B 与 Ministral 模型支持

[English](README.md) · [训练入口说明](WORKFLOWS.zh-CN.md)

本模块为 Qwen3.6-27B 和 Ministral-3-14B-Reasoning-2512 提供模型加载、checkpoint 视图、运行时兼容，以及 SFT/MAGIC 训练入口。必需的环境变量与启动命令见训练入口说明。

## 组成

- `fsdp_sft.py`：原生 FSDP SFT 循环、模型加载、token 对齐和 HF 导出。
- `rl/agent_loop.py`、`rl/contracts.py`：当前角色与冻结对手的 rollout 和奖励。
- `scripts/run_qwen_sft.sh`、`scripts/run_ministral_sft.sh`：接收已准备数据的 SFT 入口。
- `scripts/run_qwen_rl.sh`：Qwen controller/service 双主机 RL 入口。
- `scripts/run_ministral_rl.sh`：Ministral 本机 GPU 分区 RL 入口。
- `scripts/serve_magic_opponent.sh`、`scripts/serve_qwen3guard.sh`：训练所需服务。
- `compat/`：权重、RoPE、显存相关兼容工具。
- `runtime/`：启动器使用的可选 Python 启动 hook。

上游原有 separated trainer 保持不变。这些入口使用指定的现代 VERL 运行栈和调用方准备的数据，不包含独立 benchmark 评测入口。

## 模型接口

Qwen3.6 使用 `qwen3_5` / `qwen3_5_text` 运行时接口。SFT loader 提取语言模型，保留显式 ChatML assistant 目标。Ministral 使用原生 Mistral3 外层模型、tokenizer 模板和独立输出 head。准备好的 Ministral Reasoning SFT 目标采用 `[THINK]...[/THINK]answer` 并包含官方 reasoning system prompt。

Rollout 对 Ministral 分离原生 reasoning 与可见输出。逐轨迹行为和可配置 checkpoint 内容见[训练说明](WORKFLOWS.zh-CN.md)。

## Qwen checkpoint 视图

基座文本视图会参照文本 checkpoint 校验参数名和形状元数据，通过链接复用源分片：

```bash
python -m magic_model_support.scripts.prepare_qwen35_text_only_view \
  --source /models/Qwen3.6-27B \
  --reference-text-model /models/qwen-attacker/hf_model \
  --output /models/qwen-text-view

python -m magic_model_support.scripts.prepare_qwen35_vllm_service_view \
  --text-model /models/qwen-text-view \
  --multimodal-config-source /models/Qwen3.6-27B \
  --output /models/qwen-service-view
```

Qwen RL 入口自动调用这些工具，运行期间需保持链接源模型可访问。

## Ministral checkpoint 规范化

工具原地规范单文件 merged checkpoint，保留已有的独立 LM head，不从输入 embedding 人工构造输出 head：

```bash
python -m magic_model_support.compat.normalize_ministral3_merged \
  --model-dir /models/ministral-export
```

Ministral RL 入口在合并后自动调用，并在使用冻结对手前检查 head 和 embedding 的布局。

## 运行栈范围

使用针对 vLLM 0.18.0、指定 VERL 开发版本和 Transformers 5 模型 API 的独立运行环境。启动器会应用必要补丁，不面向任意未来后端版本。vLLM 包声明的 `transformers<5` 上限与模型 API 要求冲突，需要显式配置兼容环境，详见[运行环境](WORKFLOWS.zh-CN.md#运行环境)。

Qwen 路径包含文本配置渲染、权重前缀、多轴 FlashAttention position IDs 和 response-only logits 处理。Ministral 路径包含纯文本权重同步与 RoPE/YaRN 兼容。共用工具减少 entropy 和 padded log probability 计算中的中间词表 tensor 分配。
