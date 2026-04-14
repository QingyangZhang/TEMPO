import torch
from collections import defaultdict
from typing import Any
from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager
from verl.utils.reward_score.rm_physics import compute_score_p1

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
        
        prompts_str = self.tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
        responses_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        ground_truths = [data_item.non_tensor_batch["reward_model"]["ground_truth"] for data_item in data]
        data_sources = data.non_tensor_batch.get("data_source", ["PhysicsDataset"] * len(data))

        reward_tensor = torch.zeros_like(response_ids, dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        already_print_data_sources = {}
        
        keys_to_collect = ["score", "point", "acc", "score_noxverify", "point_noxverify"]
        
        standard_template = {
            "score": 0.0,
            "point": 0.0,
            "acc": False,
            "extracted_gt": "n/a",
            "extracted_pred": "n/a",
            "scored_by": "unknown",
            "score_noxverify": 0.0,
            "point_noxverify": 0.0,
        }

        for i in range(len(data)):
            item_extra = data[i].non_tensor_batch.get("extra_info", {})
            
            kwargs = {
                'model_output': responses_str[i],
                'label': ground_truths[i],
                'points': item_extra.get('points', [1.0]),
                'question': item_extra.get('question', ""),
                'use_xverify': data.meta_info.get("validate", False),
                'model_port': self.model_port,
                'marking': item_extra.get('marking', None),
                'marking_mode': "total_score",
                'skip_xverify_threshold': 1.0
            }

            try:
                raw_res = self.compute_score_fn(**kwargs)
                res = standard_template.copy()
                res.update(raw_res)
            except Exception as e:
                print(f"[Error] Scoring failed for item {i}: {e}")
                res = standard_template.copy()
                res["scored_by"] = f"error: {str(e)[:50]}"

            reward_val = float(res.get("score", 0.0))
            v_len = valid_response_lengths[i].item()
            if v_len > 0:
                reward_tensor[i, v_len - 1] = reward_val

            for key in keys_to_collect:
                reward_extra_info[key].append(res.get(key, 0.0))

            ds = data_sources[i]
            if ds not in already_print_data_sources:
                already_print_data_sources[ds] = 0
            
            if already_print_data_sources[ds] < self.num_examine:
                already_print_data_sources[ds] += 1
                print(f"\n--- [Examine Item {already_print_data_sources[ds]} | Source: {ds}] ---")
                print(f"[Prompt]: {prompts_str[i]}")
                print(f"[Response]: {responses_str[i]}")
                print(f"[Extracted GT]: {res.get('extracted_gt')}")
                print(f"[Extracted Pred]: {res.get('extracted_pred')}")
                print(f"[Reward/Score]: {reward_val} | [Point]: {res.get('point')} | [Acc]: {res.get('acc')}")
                print(f"[Scored By]: {res.get('scored_by')}")
                print("-" * 50)

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": dict(reward_extra_info)}
        else:
            return reward_tensor