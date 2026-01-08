import json
import torch

def write_rollout_log(prompts, responses):

    if isinstance(prompts, torch.Tensor):
        prompts = prompts.tolist()
    if isinstance(responses, torch.Tensor):
        responses = responses.tolist()

    item = {
        "prompts": prompts,
        "responses": responses
    }

    # 保存为 JSON 文件
    with open('output.json', 'a') as f:
        json.dump(item, f, ensure_ascii=False, indent=4)