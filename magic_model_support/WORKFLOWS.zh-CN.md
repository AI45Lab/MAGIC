# Qwen 与 Ministral 训练入口

[English](WORKFLOWS.md)

SFT 使用原生 PyTorch FSDP，MAGIC 交替更新使用现代 VERL。请使用独占的 Linux/CUDA 主机：RL 脚本会重启本机 Ray 进程。命令从仓库根目录执行，路径使用绝对路径，模型和数据由调用方准备。参数使用导出的环境变量（`export NAME=value`），或作为启动命令前的环境变量赋值。

## 运行环境

兼容层面向 vLLM 0.18.0、PyTorch 2.10、Transformers 5 模型 API，以及 VERL 的 `5ba5f2f51281522150a6c1c87e3f841d1d220042` 版本（`0.10.0.dev0`）。使用相应的独立环境，不要把仓库内的 `src/verl` 加入导入路径。

Transformers 5 超出 vLLM 0.18.0 声明的 `transformers<5` 上限，因此需要配套兼容补丁，不能只依赖包管理器解析依赖。Qwen 入口复制 `VERL_SNAPSHOT_SOURCE` 后在副本上应用补丁；该变量可指向已安装的 VERL 包目录或兼容源码快照。Ministral 入口在其独立环境内修改后端源码。重复应用时允许已安装的补丁，不匹配的源码会报错。

## SFT

输入为已经准备好的 JSON 数组，记录包含 `instruction`、`input`、`system` 和 `output`。

- Qwen 的 assistant 目标使用 `<think>...</think><answer>...</answer>`。
- Ministral 使用 `[THINK]...[/THINK]answer`，准备好的记录中应含官方 reasoning system prompt。

入口不生成、划分或筛选记录。运行前设置：

| 参数 | 含义 |
|---|---|
| `MODEL_PATH` | 原始模型 checkpoint |
| `TRAIN_FILE`、`VALIDATION_FILE` | 已准备好的 SFT JSON 文件 |
| `OUTPUT_DIR` | checkpoint 输出目录 |
| `NPROC_PER_NODE` | 本机训练进程数 |
| `CUTOFF_LEN` | token 截断长度，超长记录按此截断 |
| `GRADIENT_ACCUMULATION_STEPS` | 梯度累积次数 |
| `SAVE_STEPS`、`EVAL_STEPS` | 模型保存和 validation loss 计算间隔 |
| `SEED` | 训练随机种子 |

```bash
bash magic_model_support/scripts/run_qwen_sft.sh
# 或：
bash magic_model_support/scripts/run_ministral_sft.sh
```

可选参数包括 `EPOCHS`、`MICRO_BATCH_SIZE`、`LEARNING_RATE`、`WEIGHT_DECAY`、`WARMUP_RATIO`、`SAVE_TOTAL_LIMIT` 和 `CPU_OFFLOAD`。结果位于 `checkpoint-<step>/hf_model`，`latest_checkpoint.txt` 指向最新导出。两个模型使用同一套 FSDP 前向、loss、优化器、validation loss 和模型导出循环，通过各自的 loader 加载模型。

## RL 通用参数

`TRAIN_FILE` 和 `VAL_FILE` 指向已准备好的 MAGIC 格式 parquet，`GUARD_MODEL` 指向本地 Qwen3Guard checkpoint，`RUN_ROOT` 为运行输出目录，`RUN_ID` 使用唯一名称。入口不要求固定的数据集大小或 benchmark 目录结构。

角色以 defender 开始交替更新，`TOTAL_MAGIC_STEPS` 必须能被 `2 * SWITCH_FREQUENCY` 整除。显式提供 `FORMAT_REWARD_VALUE`、`DATA_SEED` 和 `ROLLOUT_SEED`。根据实际 GPU 设置 micro-batch 和服务显存比例。

通过 `CHECKPOINT_SAVE_CONTENTS`、`CHECKPOINT_LOAD_CONTENTS` 选择 VERL 保存/加载内容，值为 JSON 列表，例如 `["model"]` 或 `["model","optimizer","extra"]`。加载项需与 checkpoint 实际包含的状态匹配。Qwen 未显式指定加载项时，会检测是否存在 optimizer 和 extra-state 文件。

## Qwen RL

在两台独占主机运行同一脚本，共享代码、checkpoint 和 `RUN_ROOT`。一端取得 controller 角色，另一端运行冻结对手及 Guard。两端需能访问共享目录和服务端口。

除通用参数外，还需设置：

- `BASE_MULTIMODAL_MODEL`：原始 Qwen checkpoint。
- `SFT_CHECKPOINT`：包含 `hf_model/` 的 SFT checkpoint 目录。
- `VERL_SNAPSHOT_SOURCE`：兼容的 VERL 包或源码目录。
- `CUDA_VISIBLE_DEVICES`、`ACTIVE_GPUS`、`OPPONENT_TENSOR_PARALLEL_SIZE`。
- `PPO_MICRO_BATCH_SIZE_PER_GPU`、`PPO_MAX_TOKEN_LEN_PER_GPU`。
- `GUARD_GPU_MEMORY_UTILIZATION`、`OPPONENT_GPU_MEMORY_UTILIZATION`。

```bash
bash magic_model_support/scripts/run_qwen_rl.sh
```

controller 创建基座文本视图，通过共享协调目录请求对手服务，执行一个角色区块，合并 checkpoint，再切换对手。完成区块后需要重新启动时，使用 `RESUME_RUN_ROOT`、`RESUME_MAGIC_STEP`，并为新任务提供新的 `RUN_ROOT` 和协调目录。每个角色保留最新 merged 模型，raw checkpoint 按入口配置保留。模型视图链接的源文件需一直可访问。

## Ministral RL

在独占主机上为 active policy 和 frozen opponent 分配不同的 GPU 集合。显式分配的显存比例允许时，Guard 可以与服务 GPU 共置。

除通用参数外，还需设置：

- `BASE_SOURCE_MODEL`：原始 Ministral Reasoning checkpoint。
- `SFT_CHECKPOINT_OVERRIDE`：包含 `hf_model/` 的 SFT checkpoint；也可使用带有 `latest_checkpoint.txt` 的 `SFT_ROOT`。
- `REASONING_SYSTEM_PROMPT_FILE`：默认取基座目录的 `SYSTEM_PROMPT.txt`。
- `EXPECTED_CUDA_VISIBLE_DEVICES` 及与之相同的 `CUDA_VISIBLE_DEVICES`。
- `GUARD_CUDA_VISIBLE_DEVICES`、`OPPONENT_CUDA_VISIBLE_DEVICES`、`ACTIVE_CUDA_VISIBLE_DEVICES`、`ACTIVE_GPUS`、`OPPONENT_TENSOR_PARALLEL_SIZE`。
- `GUARD_GPU_MEMORY_UTILIZATION`、`OPPONENT_GPU_MEMORY_UTILIZATION`。
- `PPO_MICRO_BATCH_SIZE_PER_GPU`、`MAX_PROMPT_LENGTH`、`MAX_RESPONSE_LENGTH`。
- `OPPONENT_MAX_RESAMPLES`：对手没有可见回答时，最多额外尝试的次数。

```bash
bash magic_model_support/scripts/run_ministral_rl.sh
```

脚本依次启动 Guard 和冻结对手、等待服务就绪、执行当前角色、合并并规范 checkpoint，再切换角色。同一任务在完成的角色边界恢复时，设置 `RESUME_AFTER_MAGIC_STEP`。attacker 保留最新产物；defender 产物保留，不执行基于 benchmark 的选优。导出规范工具使用单个 `model.safetensors` 文件，保留独立 LM head。

## Rollout 行为

当前训练角色的 token 和 log probability 由 VERL rollout engine 生成；冻结对手和 Guard 通过 OpenAI 兼容服务调用。Qwen 转交完整解码输出；Ministral 先分离原生 reasoning 和可见回答，再转交或评分。

对手按指定上限重采样；仍无可见回答时，原始 reward 为 0，该分数仍参与 GRPO 归一化。奖励按单条轨迹计算；这个 AgentLoop 不实现上游 reward manager 中单独的组内无效 attacker 修正。

`serve_magic_opponent.sh` 和 `serve_qwen3guard.sh` 提供训练入口所需的服务启动和就绪协议。本模块不提供独立 benchmark 或 API-judge 评测命令。
