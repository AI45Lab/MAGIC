# 训练、推理服务与评测

[English](WORKFLOWS.md)

以下入口支持 Qwen3.6-27B 和 Ministral-3-14B-Reasoning-2512。请从仓库根目录运行，修改 JSON 示例中的模型路径、数据路径、输出位置与 GPU 资源。示例资源分配可以调整；实际显存需求取决于上下文长度和 batch 设置。

## 环境安装

使用 Linux、CUDA、Python 3.12 和 PyTorch 2.10.0，并为现代运行栈建立独立环境：

```bash
python -m pip install vllm==0.18.0
python -m pip install transformers==5.5.3 \
  'verl @ git+https://github.com/volcengine/verl.git@5ba5f2f51281522150a6c1c87e3f841d1d220042' \
  'ray[default]==2.54.1' accelerate httpx pyarrow
python -m pip install flash-attn==2.8.3 --no-build-isolation

python -m magic_model_support.setup_runtime --model-family qwen
# Ministral 使用：
# python -m magic_model_support.setup_runtime --model-family ministral
```

Transformers 5.5.3 提供此入口所需的模型 API，但超出 vLLM 0.18.0 声明的 `transformers<5` 依赖上限，因此 `pip check` 会报告该元数据冲突。兼容补丁针对上述版本，不应换成仓库内的旧版 VERL，或直接升级后端而继续套用补丁。`setup_runtime` 会修改当前独立环境内的后端源码；同一环境可依次安装两个模型的补丁。

## SFT

Qwen 先导出语言模型：

```bash
python -m magic_model_support.scripts.prepare_qwen35_text_only_view \
  --source /models/Qwen3.6-27B --output /models/qwen-text
```

Ministral Reasoning 先转换选定的 SFT 文件；独立的验证文件也按相同方式转换：

```bash
python -m magic_model_support.scripts.prepare_ministral_reasoning_sft \
  --input /data/sft-source.json --output /data/sft-train.json \
  --system-prompt-file /models/Ministral-3-14B-Reasoning-2512/SYSTEM_PROMPT.txt
```

输入为 JSON 数组，记录包含 `instruction`、`input`、`system` 和 `output`。转换器只改变序列化格式。SFT 入口遇到超长记录时会报错，需明确调整 `max_length` 或输入数据。

修改 `examples/qwen-sft.json` 或 `examples/ministral-sft.json` 后启动：

```bash
torchrun --standalone --nproc-per-node=4 \
  -m magic_model_support.train_sft --config magic_model_support/examples/qwen-sft.json
# Ministral 改用 ministral-sft.json。
```

示例采用 FSDP 全分片。GPU 数、micro-batch、梯度累积、精度和保存频率通过 Hugging Face `training_args` 配置。单卡可移除 `fsdp`、`fsdp_config` 字段并使用 `python` 启动。需要恢复时，将 `resume_from_checkpoint` 设置为 Trainer checkpoint 目录。

最终 HF 导出位于 `<output_dir>/hf_model`；Trainer checkpoint 还保存优化器与学习率调度器状态。SFT 示例的学习率和 epoch 数参考仓库已有的全参数 SFT 示例，调用方可以调整。

## MAGIC RL

可以直接使用上游格式的 parquet，也可转换 JSONL：每条记录提供 `attacker_prompt`（完整改写任务）和 `seed_prompt`（原始问题）。

```bash
python -m magic_model_support.prepare_rl_data \
  --input /data/rl-train.jsonl --output /data/train.parquet
```

单独转换调用方选择的验证数据。转换器不抽样、不划分数据。直接使用 parquet 时，需包含 `prompt`、`data_source`、`reward_model` 和 `extra_info.raw_prompt`。

在独立的 Guard 主机上启动 Qwen3Guard，或提供已有的 OpenAI 兼容接口：

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve /models/Qwen3Guard-Gen-8B \
  --served-model-name Qwen3Guard-Gen-8B --port 8000
```

修改 `examples/qwen-rl.json` 或 `examples/ministral-rl.json`：设置 `guard.base_url` 和 `guard.model`；若接口需要鉴权，将密钥放入 `guard.api_key_env` 指定的环境变量。无需鉴权时移除 `api_key_env`。Guard 资源应与 active/opponent 使用的 GPU 分开。

```bash
python -m magic_model_support.train_rl --config magic_model_support/examples/qwen-rl.json
# 恢复同一个任务：
python -m magic_model_support.train_rl --config magic_model_support/examples/qwen-rl.json --resume
```

该入口管理单个本地节点，active policy 和 frozen opponent 使用互不重叠的 GPU 集合。流程依次启动冻结对手服务、执行当前角色的 VERL 更新、导出更新后的模型，再切换角色。程序只终止自己创建的子进程。

`state.json` 指向最新导出的两个模型。转换完成后才原子发布导出目录；若在导出阶段中断，可恢复并继续导出，不重复已保存的更新区块。已有 checkpoint 会被保留。

默认算法参数来自 `scripts/rl/separated/grpo_public.sh`：defender 先更新，每个区块 15 步，合计 300 步；GRPO 每组 4 个样本，batch 64，mini-batch 32，学习率 `1e-6`，不使用 KL 项。harm、refusal、format 权重分别为 1、0.5、1，与公开脚本默认值一致。这些字段都可以在 JSON 中显式设置。

`micro_batch_size_per_gpu` 必须由调用方指定。`verl_overrides` 接受后端执行参数的 Hydra override；模型路径、GRPO、checkpoint 状态和角色调度由入口统一管理，优先于这些 override。

两个角色均保存并恢复 model、optimizer 和 extra 状态。现代 AgentLoop 使用公开的单轮交互 prompt 和可见改写内容，Ministral 使用原生 reasoning 序列化。无效 attacker 输出的任务奖励在 GRPO 归一化前按公开的组内有效样本均值处理，格式惩罚单独保留。未完成的 active 输出采用公开 reward manager 的 reward mask 处理；缺少可见回答时不计算任务得分；不执行输出重采样。Guard 格式错误和网络请求错误会中止当前区块。

这是现代 VERL 执行入口，原有 `separated_trainer` 保持不变。模型补丁和原生格式细节见[模型适配说明](README.zh-CN.md)。

## 推理服务

修改对应模型的 service 示例，将 `model_path` 设置为基座模型、SFT 导出或 RL 导出路径：

```bash
python -m magic_model_support.serve --config magic_model_support/examples/qwen-serve.json
python -m magic_model_support.serve --config magic_model_support/examples/ministral-serve.json
```

Qwen 服务会自动创建临时服务视图。Ministral 使用 `--tokenizer-mode mistral` 和原生 reasoning parser；模型目录或显式 `tokenizer_path` 中需有 `tekken.json`。训练导出器会保留源 checkpoint 中已有的该文件。

需要不经过 reasoning parser 的原始 completion 时，使用 `--raw`；RL 对手服务采用此模式。服务位置和显存由 `cuda_visible_devices`、`tensor_parallel_size`、`max_model_len` 和 `gpu_memory_utilization` 控制。

## 评测

评测输入是 JSONL，每条记录具有唯一 `id`，并提供 `prompt` 或 OpenAI 格式的 `messages`。可选的 `reference` 启用精确字符串匹配准确率。Ministral 需设置官方 `system_prompt_file` 和 `tokenizer_mode="mistral"`，以正确序列化 system prompt 中的 reasoning 示例。

```bash
python -m magic_model_support.evaluate --config magic_model_support/examples/qwen-eval.json
# Ministral 改用 ministral-eval.json。
```

`generation` 中的参数直接传给生成接口。逐条结果记录回答、独立 reasoning、结束原因和有效性。只有非空回答且 `finish_reason="stop"` 时才进入评分。相邻的 `.metrics.json` 记录 total、valid、invalid、coverage，以及有参考答案的有效样本上的精确匹配准确率。输出需使用新文件名。

若需要 API judge，在配置中增加 `judge` 对象，包含 `endpoint`、`generation` 和 `prompt_file`。调用方的模板可以使用 `{prompt}`、`{response}`、`{reference}`，并要求 judge 返回带有有限数值 `score` 的 JSON 对象。评测器保存判断结果并汇总有效回答的平均分；评分标准和量表由调用方定义。仓库已有的 benchmark 专用评测器也可连接所启动的 OpenAI 接口。
