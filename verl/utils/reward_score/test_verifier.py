import json
from tqdm import tqdm
from rm_p1 import compute_score_p1

def test_ground_truth_extraction(jsonl_path):
    invalid_samples = []
    total_count = 0
    
    print(f"Testing dataset: {jsonl_path}")
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            if not line.strip():
                continue
            
            total_count += 1
            data = json.loads(line)
            
            ground_truth = data.get("reward_model", {}).get("ground_truth", [])
            points = data.get("extra_info", {}).get("points", [1.0])
            question = data.get("prompt", [{}])[0].get("content", "")
            
            mock_model_output = "The answer is \\boxed{0}" 
            
            try:
                result = compute_score_p1(
                    model_output=mock_model_output,
                    label=ground_truth,
                    points=points,
                    question=question,
                    use_xverify=False
                )
                
                ext_gt = result.get("extracted_gt", "")
                
                if not ext_gt or ext_gt.strip() == "" or ext_gt == "invalid":
                    invalid_samples.append({
                        "index": data.get("index", "unknown"),
                        "raw_gt": ground_truth,
                        "extracted_gt": ext_gt,
                        "scored_by": result.get("scored_by")
                    })
                    
            except Exception as e:
                invalid_samples.append({
                    "index": data.get("index", "unknown"),
                    "raw_gt": ground_truth,
                    "error": str(e)
                })

    print("\n" + "="*50)
    print(f"Done. Total samples: {total_count}")
    print(f"Invalid samples: {len(invalid_samples)}")
    print("="*50)

    if invalid_samples:
        print("Done.")
        for res in invalid_samples[:20]:
            print(f"- [Index {res['index']}]: Raw GT: {res['raw_gt']} | Extracted: {res.get('extracted_gt', 'ERROR')}")
        
        with open("failed_gt_extraction.json", "w", encoding="utf-8") as f:
            json.dump(invalid_samples, f, indent=4, ensure_ascii=False)
        print("Done.")
    else:
        print("Done.")

if __name__ == "__main__":
    target_jsonl = "path/to/resource" 
    test_ground_truth_extraction(target_jsonl)