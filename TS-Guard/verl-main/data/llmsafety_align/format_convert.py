import json
import random
import pandas as pd

input_path = "STAIR_SFT_SaROcot_v2.json"

with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

new_data = []
i = 0
for item in data:
    new_item = {
        "data_source": "llmsafety_align",
        "prompt": [
            {
                "role": "user",
                "content": item["instruction"]
            }
        ],
        "ability": "llmsafety_align",
        "reward_model": {
            "ground_truth": "",
            "style": "llm-as-a-judge",
        },
        "extra_info": {
            "answer": item["output"],
            "index": i,
            "user_request": item["instruction"],
            "split": "train"
        }
    }
    new_data.append(new_item)
    i += 1


random.shuffle(new_data)

output_path = "train.json"
eval_path = "eval.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(new_data[:2000], f, indent=4, ensure_ascii=False)

with open(eval_path, "w", encoding="utf-8") as f:
    json.dump(new_data[2000:3000], f, indent=4, ensure_ascii=False)

print("Train and eval data prepared.")

train_df = pd.read_json(output_path)
train_df.to_parquet('train.parquet', engine='pyarrow')

eval_df = pd.read_json(eval_path)
eval_df.to_parquet('eval.parquet', engine='pyarrow')
print("Parquet files saved.")