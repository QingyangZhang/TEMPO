import asyncio
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Any, Optional
import psutil
import torch
from collections import defaultdict

from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager
from verl.utils.reward_score.rm_p1 import compute_score_p1


async def single_compute_score(evaluation_func, completion, reference, task_extra_info, executor, timeout=300.0):
    loop = asyncio.get_running_loop()
    
    standard_template = {
        "score": 0.0,
        "point": 0.0,
        "acc": False,
        "extracted_gt": "n/a",
        "extracted_pred": "n/a",
        "scored_by": "unknown",
        "score_noxverify": 0.0,
        "point_noxverify": 0.0,
        "llm_judge_responses": "n/a",
    }

    try:
        kwargs = {
            'model_output': completion,
            'label': reference,
            'points': task_extra_info.get('points', [1.0]),
            'question': task_extra_info.get('question', ""),
            'use_xverify': task_extra_info.get('use_xverify', True),
            'model_port': task_extra_info.get('model_port', 34812),
            'marking': task_extra_info.get('marking', None),
            'marking_mode': "total_score",
            'skip_xverify_threshold': 1.0
        }
        
        future = loop.run_in_executor(executor, partial(evaluation_func, **kwargs))
        result = await asyncio.wait_for(future, timeout=timeout)
        
        final_res = standard_template.copy()
        final_res.update(result)
        return final_res

    except asyncio.TimeoutError:
        res = standard_template.copy()
        res["scored_by"] = "timeout"
        return res
    except Exception as e:
        res = standard_template.copy()
        res["scored_by"] = f"error: {str(e)[:100]}"
        return res

async def parallel_compute_score_async(
    evaluation_func, completions, references, extra_infos, num_processes=8
):
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        try:
            tasks_async = [
                single_compute_score(evaluation_func, c, r, ei, executor, timeout=300.0)
                for c, r, ei in zip(completions, references, extra_infos)
            ]
            results = await asyncio.gather(*tasks_async, return_exceptions=False)
        finally:
            for pid, proc in executor._processes.items():
                try:
                    p = psutil.Process(pid)
                    p.terminate()
                except:
                    pass
    return results

def run_reward_scoring(evaluation_func, completions, references, extra_infos, num_processes=64):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(
            parallel_compute_score_async(evaluation_func, completions, references, extra_infos, num_processes)
        )
    finally:
        loop.close()


@register("physics_prime")
class PhysicsPrimeRewardManager(AbstractRewardManager):
    def __init__(self, tokenizer, num_examine, compute_score=None, model_port=34812, reward_fn_key="data_source", use_xverify=True) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.model_port = model_port
        self.use_xverify = use_xverify
        self.compute_score_fn = compute_score_p1

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        prompt_ids = data.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        response_ids = data.batch["responses"]
        
        valid_response_lengths = data.batch["attention_mask"][:, prompt_length:].sum(dim=-1).int()
        
        sequences_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        ground_truths = [data_item.non_tensor_batch["reward_model"]["ground_truth"] for data_item in data]
        data_sources = data.non_tensor_batch.get("data_source", ["PhysicsDataset"] * len(data))
        
        extra_infos_for_scoring = []
        for i in range(len(data)):
            item_extra = data[i].non_tensor_batch.get("extra_info", {})
            scoring_cfg = {
                "question": item_extra.get("question", ""),
                "marking": item_extra.get("marking", None),
                "points": item_extra.get("points", [1.0]),
                "use_xverify": False, # data.meta_info.get("validate", False),
                "model_port": self.model_port,
            }
            extra_infos_for_scoring.append(scoring_cfg)

        try:
            results = run_reward_scoring(
                self.compute_score_fn,
                completions=sequences_str,
                references=ground_truths,
                extra_infos=extra_infos_for_scoring,
                num_processes=128
            )
        except Exception as e:
            print(f"[Error] Parallel scoring failed: {e}")
            fallback_item = {
                "score": 0.0,
                "point": 0.0,
                "acc": False,
                "scored_by": f"error: {type(e).__name__}",
                "score_noxverify": 0.0,
                "point_noxverify": 0.0,
                "llm_judge_responses": "n/a",
            }
            results = [fallback_item.copy() for _ in range(len(data))]

        reward_tensor = torch.zeros_like(response_ids, dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        
        already_print_data_sources = {}

        keys_to_collect = ["score", "point", "acc", "score_noxverify", "point_noxverify"]

        for i in range(len(data)):
            res = results[i]
            
            reward = float(res.get("score", 0.0))
            v_len = valid_response_lengths[i].item()
            if v_len > 0:
                reward_tensor[i, v_len - 1] = reward

            for key in keys_to_collect:
                reward_extra_info[key].append(res.get(key, 0.0))

            ds = data_sources[i]
            if ds not in already_print_data_sources:
                already_print_data_sources[ds] = 0
            
            if already_print_data_sources[ds] < self.num_examine:
                already_print_data_sources[ds] += 1
                print(f"\n--- [Examine Item {already_print_data_sources[ds]} | Source: {ds}] ---")
                print(f"[Response]: {sequences_str[i]}...")
                print("[ground_truth]", ground_truths[i])
                print(f"[Reward]: {reward} | Scored By: {res.get('scored_by', 'unknown')}")

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": dict(reward_extra_info)}
        else:
            return reward_tensor