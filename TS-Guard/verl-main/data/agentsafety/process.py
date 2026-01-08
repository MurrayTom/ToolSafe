import json
import random
import pandas as pd

with open("train_agentalign-benign.json", 'r') as file:
    data = json.load(file)

data_selected = random.sample(data, 150)

with open('train_agentalign-benign-150.json', 'w') as val_file:
    json.dump(data_selected, val_file, indent=4)


# 读取 JSON 文件
df = pd.read_json('train_agentalign-benign-150.json')

# 将 DataFrame 写入 Parquet 文件
df.to_parquet('train_agentalign-benign-150.parquet', engine='pyarrow')