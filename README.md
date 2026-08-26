<!-- # MAGIC: Multi-Agent Adversarial Game for Robust LLM Safety -->

<p align="center" style="margin-bottom: 0px;">
  <img src="assets/magic_logo.png" alt="MAGIC" height="140">
</p>

<p align="center" style="font-size: 18px; margin-top: 0;">
  Multi-Agent Adversarial Game for Robust LLM Safety
</p>

<p align="center">
  <a href="https://huggingface.co/XiaoyuWen/MAGIC"><img src="https://img.shields.io/badge/🌐_huggingface-MAGIC-yellow" alt="Page"></a>
  <a href="https://github.com/vllm-project/vllm"><img src="https://img.shields.io/badge/vLLM-0.8.5-5C2D91?style=flat-square" alt="vLLM0.8.5"></a>
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-green.svg"></a>
  <a href="assets/MAGIC_arxiv.pdf"><img src="https://img.shields.io/badge/Paper-PDF-red" alt="Paper"></a>
</p>

# 🔥 News
[Aug. 2026] We added the attacker SFT model weights to Hugging Face: [Llama3.1-8B-Instruct](https://huggingface.co/XiaoyuWen/MAGIC-attacker-sft-Llama3.1-8B-Instruct), [Qwen2.5-7B-Instruct](https://huggingface.co/XiaoyuWen/MAGIC-attacker-sft-Qwen2.5-7B-Instruct), and [Qwen2.5-14B-Instruct](https://huggingface.co/XiaoyuWen/MAGIC-attacker-sft-Qwen2.5-14B-Instruct).

[May. 2026] Our paper have been accepeted by ICML2026!

# 🤗 Overview
<!-- ![MAGIC](./assets/MAGIC.png) -->
<p align="center">
  <img src="./assets/MAGIC.png" alt="MAGIC" width="800">
</p>

	✨ Key Contributions:

	•	Asymmetric attacker–defender game
		· Sequential (attacker → defender), not symmetric self-play
		· Avoids gradient conflict in shared-parameter setups
	•	Co-evolving online training
		· Attacker continuously discovers long-tail vulnerabilities
		· Defender learns pointwise-safe responses
	•	Attack Pool Benchmark with CoT
		· 20 rewriting strategies
		· Solves cold-start for automated red-teaming
	•	Strong empirical results
		· Lower ASR across HarmBench, WildGuardTest, DAN, X-Teaming
		· Minimal degradation on general capabilities
	
	🧠 Co-evolving attacker & defender
	🎯 Subgame Perfect Nash Equilibrium (SPNE) formulation
	🔄 Online MARL with GRPO
	🧪 Strong single-turn & multi-turn safety improvements

	We build our training pipeline on: https://github.com/ziyuwan/ReMA-public

# 📒 Method

![MAGIC](./assets/magic_methods.png)
	
	MAGIC operates in two phases:

	Phase 1: Offensive Capability Initialization
		•	SFT attacker on CoT-enriched Attack Pool
		•	Enables reasoning-based adversarial rewriting

	Phase 2: Iterative Co-evolution
		•	Alternate optimization:
		•	Fix attacker → train defender (GRPO)
		•	Fix defender → train attacker (GRPO)
		•	Approximates bilevel best-response dynamics

# 🚀 Getting Started
we recommend using cuda12.4 + torch2.6 + python3.10.
```bash
conda create -n magic python=3.10.0
conda activate magic
# install flash-attn (faster)
# wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
# pip install flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl --no-build-isolation

# install flash-attn
pip install flash-attn==2.7.4.post1 --no-build-isolation

# for sft
cd src/360-LLaMA-Factory
pip install -e .

# for RL
cd src/verl
pip install -e .

# update libs
pip install -r requirements.txt
```

# ✈️ Training & Evalution

## How to run scripts
To run the training, you can see `scripts/rl/separated/grpo_public.sh` as an example.
```bash
bash scripts/rl/separated/grpo_public.sh

# NOTION!!!(add following sentenses into bash file at least)
export WANDB_API_KEY=<API_KEY>
export WILDGUARD_API_ENDPOINT="http://<API URL>/v1"
export WORKSPACE=<ROOT_PATH>
export CHECKPOINT_DIR=<ROOT_PATH>/MAGIC/checkpoints
export MODEL_DEFENDER_BASE=$WORKSPACE/models/Qwen2.5-7B-Instruct
export MODEL_ATTACKER_SFT=$WORKSPACE/models/Qwen2.5-7B-Instruct

```

key arguments:
- Rollout:
    - `data.max_prompt_length`: max length of prompt, **for each turn's input**
    - `data.max_response_length`: max length of response **per turn**
    - `actor_rollout_ref.rollout.stop_when_truncated=True`: whether to stop when truncated, i.e. last agent's response is stopped by length
	- `actor_rollout_ref.rollout.max_num_turns`: control single-turn or multi-turn conversations, 1 for single-turn; 3 for multi-turn

- Training:
    - `algorithm.switch_agent.model_paths=[${Attacker_model_path},${Defender_model_path}]`: model paths of two agents (the order follows `algorithm.switch_agent.agent_roles`)
    - `algorithm.switch_agent.train_roles=[attacker,defender]`: Control whether to use defender-only, attacker-only, or MAGIC
	- `algorithm.switch_agent.freq=15`: switch agent frequency
	- `SWITCH_ARGS`: `algorithm.switch_agent.mode=ratio` for fixed step update; `algorithm.switch_agent.mode=metric` for ASR-aware

- Reward Judge:
    - `+reward_model.format_reward_roles=[attacker]`: Whether to enable the format reward and which agent uses it.
    - `REWARD_HARM \ REWARD_REFUSAL \ REWARD_LABEL`: control the reward values
    - `USE_Q_SAFE_LABEL_FOR_REFUSAL`: 1 for using attcker_prompt label to judge refusal reward; 0 for using raw_prompt label to judeg
	- `REWARD_MODEL_USE_ANSWER_TAGS`: 1 for extracting the content within the defender’s <answer> tags for safety evaluation; 0 for using the defender’s entire response directly.
	- `SAFETY_SCORE_MODE`: `classifer` for qwen3guard or wildguard; `rule_api` for gpt-4o (**Not yet integrated for testing**)
	- `Other reward`: `DEFENDER_QUALITY_API_ENDPOINT` and `REVISION_SCORE_ENDPOINT` **Not yet integrated for testing**

see `src/verl/verl/separated_trainer/config/ppo_trainer.yaml` for more details.

## How to evalute
```bash
Looking for ./eval/README.md
```

# 📄 Paper & Citation
```bibtex
@article{wen2026magic,
	title={MAGIC: A Co-Evolving Attacker-Defender Adversarial Game for Robust LLM Safety},
	author={Wen, Xiaoyu and He, Zhida and Qi, Han and Wan, Ziyu and Wen, Ying and Zheng, Tianhang and Xu, Xingcheng and Lu, Chaochao and Zhang, Qiaosheng},
	journal={arXiv preprint arxiv:2602.01539},
	year={2026}
}

@article{wan2025rema,
	title={Rema: Learning to meta-think for llms with multi-agent reinforcement learning},
	author={Wan, Ziyu and Li, Yunxiang and Wen, Xiaoyu and Song, Yan and Wang, Hanjing and Yang, Linyi and Schmidt, Mark and Wang, Jun and Zhang, Weinan and Hu, Shuyue and others},
	journal={arXiv preprint arXiv:2503.09501},
	year={2025}
}
