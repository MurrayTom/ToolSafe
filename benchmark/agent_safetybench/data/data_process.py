import json

with open('agent_align_data_v3.json', 'r') as file:
    data = json.load(file)

harmful_list = []
for i in range(len(data)):
    if data[i]["category"] == "benign":
        continue
    else:
        data[i]["instruction"] = data[i]["messages"][0]["content"]
        harmful_list.append(data[i])


with open('agent_align_data_v3_harmful.json', 'w') as file:
    json.dump(harmful_list, file, indent=4)