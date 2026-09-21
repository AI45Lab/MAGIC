# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
from typing import Dict, List, Optional
import copy
import os
from collections import deque

from tqdm import tqdm
from verl import DataProto
from verl.utils.reward_score.game import (
    RewardScores,
    format_reward_func,
    extract_answer,
    visible_attack,
    extract_think,
    # revision_reward_func,
    compute_reward_bundle,
    USE_ANSWER_TAGS_FOR_REWARD_MODEL,
)
import torch
from pebble import ProcessPool
from concurrent.futures import TimeoutError, as_completed
from math_verify.errors import TimeoutException

def compute_score_fn(compute_score, params):
    data_source, attacker_prompt, defender_response, extra_info = params
    return compute_score(data_source, attacker_prompt, defender_response, extra_info)


def compute_format_r(data_source, role, response_str):
    if data_source == "game":
        return format_reward_func(response_str)
    else:
        raise ValueError(f'Unknown {data_source=} for format reward.')

def _filter_valid_history(history: List[Dict[str, str]], agent_roles: List[str]) -> List[Dict[str, str]]:
    return [msg for msg in history if msg.get('role') in agent_roles]


def _get_last_role_message(history: List[Dict[str, str]], role: str) -> Optional[Dict[str, str]]:
    for msg in reversed(history):
        if msg.get('role') == role:
            return msg
    return None


def _get_visible_answer(msg: Optional[Dict[str, str]], *, answer_only: bool) -> str:
    if msg is None:
        return ""
    content = (msg.get('content', '') or '').strip()
    if not answer_only:
        return content
    return (msg.get('parsed_answer') or visible_attack(content)).strip()


def _normalize_format_reward_roles(format_reward_roles, train_roles):
    if format_reward_roles is None:
        return [role for role in train_roles if role != 'defender']
    if isinstance(format_reward_roles, str):
        return [role.strip() for role in format_reward_roles.split(',') if role.strip()]
    return list(format_reward_roles)


def _ensure_reward_scores(raw_score) -> RewardScores:
    """Normalize various compute_score outputs into RewardScores."""

    if isinstance(raw_score, RewardScores):
        return raw_score
    if isinstance(raw_score, dict):
        return RewardScores(
            safety=float(raw_score.get('safety', 0.0)),
            revision=float(raw_score.get('revision', 0.0)),
            defender_quality=float(raw_score.get('defender_quality', 0.0)),
            label_reward=float(raw_score.get('label_reward', 0.0)),
            reward_harm=float(raw_score.get('reward_harm', 0.0)),
            reward_refusal=float(raw_score.get('reward_refusal', 0.0)),
            question_is_harmful=raw_score.get('question_is_harmful'),
            defender_refused=raw_score.get('defender_refused'),
            request_safety_label=raw_score.get('request_safety_label'),
        )
    if isinstance(raw_score, (tuple, list)):
        # allow (safety, revision, defender_quality)
        values = list(raw_score) + [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, None, None, None, None]
        return RewardScores(
            safety=float(values[0]),
            revision=float(values[1]),
            defender_quality=float(values[2]),
            label_reward=float(values[3]),
            reward_harm=float(values[4]),
            reward_refusal=float(values[5]),
            question_is_harmful=values[6],
            defender_refused=values[7],
            request_safety_label=values[8],
        )
    return RewardScores(safety=float(raw_score))


class GameRewardManager:
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or compute_reward_bundle

    def verify(self, data):
        scores = []
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch['data_source']

            extra_info = data_item.non_tensor_batch.get('extra_info', None)

            history = data_item.non_tensor_batch.get('history', [])
            valid_history = _filter_valid_history(history, data.meta_info['agent_roles'])
            attacker_msg = _get_last_role_message(valid_history, 'attacker')
            defender_msg = _get_last_role_message(valid_history, 'defender')
            attacker_prompt = _get_visible_answer(attacker_msg, answer_only=True) or (extra_info or {}).get('seed_prompt', prompt_str)
            defender_answer = _get_visible_answer(defender_msg, answer_only=USE_ANSWER_TAGS_FOR_REWARD_MODEL) or response_str
            score = _ensure_reward_scores(self.compute_score(
                data_source=data_source,
                attacker_prompt=attacker_prompt,
                defender_response=defender_answer,
                extra_info=extra_info,
            ))
            scores.append(score.safety)
        data.batch['acc'] = torch.tensor(scores, dtype=torch.float32, device=prompt_ids.device)
        return scores

    def __call__(self, data: DataProto)-> Dict[str, torch.Tensor]:
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']
        
        batch_size = len(data)
        max_num_turns = data.meta_info['max_num_turns']

        
        agent_roles = data.meta_info['agent_roles']
        reward_tensor_map = {
            f'{role}_turn_level_reward': torch.zeros(batch_size, max_num_turns, dtype=torch.float32) for role in agent_roles
        }
        role_format_rewards = {
            role: torch.zeros(batch_size, dtype=torch.float32) for role in agent_roles
        }
        
        already_print_data_sources = {}

        attacker_inputs = []
        defender_inputs = []
        sample_extra_infos = []
        params = []
        for i in range(len(data)):
            data_item = data[i]
            history = data_item.non_tensor_batch['history']
            valid_history = _filter_valid_history(history, agent_roles)
            attacker_msg = _get_last_role_message(valid_history, 'attacker')
            defender_msg = _get_last_role_message(valid_history, 'defender')

            attacker_prompt = _get_visible_answer(attacker_msg, answer_only=True)
            defender_answer = _get_visible_answer(defender_msg, answer_only=USE_ANSWER_TAGS_FOR_REWARD_MODEL)

            extra_info = copy.deepcopy(data_item.non_tensor_batch.get('extra_info', {}))
            seed_prompt = data_item.non_tensor_batch.get('seed_prompt') or extra_info.get('raw_prompt')
            if seed_prompt:
                extra_info.setdefault('seed_prompt', seed_prompt)

            if not attacker_prompt:
                attacker_prompt = extra_info.get('seed_prompt', '')
            if not defender_answer:
                defender_answer = data_item.non_tensor_batch.get('response', '')

            attacker_inputs.append(attacker_prompt)
            defender_inputs.append(defender_answer)
            sample_extra_infos.append(extra_info)

            params.append((
                data_item.non_tensor_batch['data_source'],
                attacker_prompt,
                defender_answer,
                extra_info,
            ))

        timeout_s = float(os.environ.get("REWARD_SCORE_TIMEOUT_S", "20"))
        max_retries = 3
        score_bundles = [None] * len(params)
        max_workers = max(1, int(os.environ.get("REWARD_SCORE_MAX_WORKERS", "1")))
        max_workers = min(max_workers, len(params)) if params else 1
        with ProcessPool(max_workers=max_workers) as pool:
            with tqdm(total=len(data), desc="Computing scores") as pbar:
                pending = deque(range(len(params)))
                attempts = [0] * len(params)
                inflight = {}

                def _schedule(idx):
                    attempts[idx] += 1
                    future = pool.schedule(
                        partial(compute_score_fn, self.compute_score),
                        args=(params[idx],),
                        timeout=timeout_s,
                    )
                    inflight[future] = idx

                while pending or inflight:
                    while pending and len(inflight) < max_workers:
                        _schedule(pending.popleft())

                    for future in as_completed(list(inflight.keys())):
                        idx = inflight.pop(future)
                        try:
                            result = future.result()
                            score_bundles[idx] = _ensure_reward_scores(result)
                            pbar.update(1)
                        except TimeoutError:
                            print(f'Time Out (attempt {attempts[idx]}/{max_retries})')
                            future.cancel()
                            if attempts[idx] < max_retries:
                                pending.append(idx)
                            else:
                                score_bundles[idx] = RewardScores()
                                pbar.update(1)
                        except TimeoutException:
                            print(f'Internal timeout (attempt {attempts[idx]}/{max_retries})')
                            future.cancel()
                            if attempts[idx] < max_retries:
                                pending.append(idx)
                            else:
                                score_bundles[idx] = RewardScores()
                                pbar.update(1)
                        except Exception as e:
                            future.cancel()
                            print(f"Error: {e}")
                            raise e
                        break
        
        def _flag_to_float(flag: Optional[bool]) -> float:
            if flag is None:
                return -1.0
            return 1.0 if flag else 0.0

        assert len(score_bundles) == len(data)
        accuracy = torch.tensor([score.safety for score in score_bundles], dtype=torch.float32)
        reward_tensor_map['acc'] = accuracy
        reward_tensor_map['reward_harm'] = torch.tensor(
            [score.reward_harm for score in score_bundles],
            dtype=torch.float32,
        )
        reward_tensor_map['reward_refusal'] = torch.tensor(
            [score.reward_refusal for score in score_bundles],
            dtype=torch.float32,
        )
        reward_tensor_map['question_is_harmful'] = torch.tensor(
            [_flag_to_float(score.question_is_harmful) for score in score_bundles],
            dtype=torch.float32,
        )
        reward_tensor_map['defender_refused'] = torch.tensor(
            [_flag_to_float(score.defender_refused) for score in score_bundles],
            dtype=torch.float32,
        )
        reward_tensor_map['revision_scores'] = torch.tensor(
            [score.revision for score in score_bundles],
            dtype=torch.float32,
        )
        reward_tensor_map['defender_quality_scores'] = torch.tensor(
            [score.defender_quality for score in score_bundles],
            dtype=torch.float32,
        )
        reward_tensor_map['label_scores'] = torch.tensor(
            [score.label_reward for score in score_bundles],
            dtype=torch.float32,
        )
        format_rewards = torch.zeros(len(data), dtype=torch.float32)
        for i_bsz in range(len(data)):
            data_item = data[i_bsz]
            valid_history = _filter_valid_history(data_item.non_tensor_batch['history'], agent_roles)
            attacker_prompt = attacker_inputs[i_bsz]
            defender_answer = defender_inputs[i_bsz]
            sample_extra = sample_extra_infos[i_bsz]
            data_source = data_item.non_tensor_batch['data_source']
            score_components = score_bundles[i_bsz]
            base_score = score_components.safety
            revision_score = score_components.revision
            defender_quality_score = score_components.defender_quality
            label_reward = score_components.label_reward
            num_turns = data_item.non_tensor_batch['num_turns']
            format_bonus = 0.0
            last_attacker_msg = _get_last_role_message(valid_history, 'attacker')
            attacker_format_invalid = (
                data_source == 'game'
                and last_attacker_msg is not None
                and compute_format_r(data_source, 'attacker', last_attacker_msg['content']) < 0
            )

            train_roles = data_item.meta_info.get('train_roles', agent_roles)
            format_reward_roles = _normalize_format_reward_roles(
                data_item.meta_info.get('format_reward_roles'),
                train_roles,
            )
            for role in agent_roles:
                turn_finished = data_item.batch[f'{role}_turn_finished'].item()
                if role == 'attacker':
                    role_score = -base_score + revision_score + label_reward
                    if attacker_format_invalid:
                        role_score = 0.0
                else:
                    role_score = base_score + defender_quality_score
                if data_item.meta_info['mask_unfinished_reward']:
                    role_score = role_score if turn_finished not in (2, 3) else 0.0

                if (turn_finished not in (2, 3) and data_item.meta_info['use_format_reward']
                        and role in train_roles and role in format_reward_roles):
                    last_role_msg = _get_last_role_message(valid_history, role)
                    if last_role_msg is not None:
                        format_r = compute_format_r(data_source, role, last_role_msg['content'])
                        role_score += format_r
                        format_bonus += format_r
                        role_format_rewards[role][i_bsz] = format_r
                        # if role == 'attacker':
                        #     role_score += revision_reward_func(attacker_prompt, sample_extra)

                reward_tensor_map[f'{role}_turn_level_reward'][i_bsz, num_turns - 1] = role_score
            format_rewards[i_bsz] = format_bonus

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                prompt_str = data_item.non_tensor_batch['question']
                ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
                history = valid_history[:num_turns * len(agent_roles)]
                already_print_data_sources[data_source] += 1
                print('[question]', prompt_str)
                print('[ground_truth]', ground_truth)
                print('[answer]', defender_answer)
                print('[score_components]', score_components)
                print('[history]', history)

        reward_tensor_map['format_reward'] = format_rewards
        for role in agent_roles:
            reward_tensor_map[f'{role}_format_reward'] = role_format_rewards[role]
        # Return both reward tensors in a dictionary
        return reward_tensor_map
