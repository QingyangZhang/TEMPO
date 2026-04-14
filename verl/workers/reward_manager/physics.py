import torch
from collections import defaultdict
from typing import Any
from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager

from verl.utils.reward_score.rm_physics import compute_score_p1, Model_args 

@register("physics")
class PhysicsRewardManager(AbstractRewardManager):
    """
    
    
    
    
    """

    def __init__(self, tokenizer, num_examine, compute_score=None, model_port=34812, reward_fn_key="data_source", use_xverify=True) -> None:
        """
        Args:
            
            
            
            
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.model_port = model_port
        self.use_xverify = use_xverify

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """Main entry point for computing rewards."""
        
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]
            
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_mask = data_item.batch["attention_mask"][:prompt_length]
            valid_prompt_ids = prompt_ids[valid_prompt_mask == 1]

            response_ids = data_item.batch["responses"]
            valid_response_mask = data_item.batch["attention_mask"][prompt_length:]
            valid_response_length = int(valid_response_mask.sum().item())
            valid_response_ids = response_ids[:valid_response_length]

            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=False)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=False)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch.get("data_source", "PhysicsDataset")
            
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            question = extra_info.get("question", prompt_str)
            marking = extra_info.get("marking", None)
            points = extra_info.get("points", [1.0])
            
            grading_result = compute_score_p1(
                model_output=response_str,
                label=ground_truth,
                points=points,
                question=question,
                use_xverify=False,
                model_port=self.model_port,
                marking=marking,
                marking_mode="total_score",
                skip_xverify_threshold=1.0
            )

            reward = 1.0 if float(grading_result["score"]) > 0.5 else -1.0
            
            for key, value in grading_result.items():
                if key in ["score", "point", "acc", "score_noxverify", "point_noxverify"]:
                    reward_extra_info[key].append(value)

            if valid_response_length > 0:
                reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(grading_result, dict):
                    for key, value in grading_result.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", grading_result["score"])

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor