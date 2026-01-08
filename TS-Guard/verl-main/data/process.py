import pandas as pd

# 读取 Parquet 文件
df = pd.read_parquet('./agentsafety/test.parquet')

# 将 DataFrame 保存为 JSON 文件
df.to_json('./agentsafety/test.json', orient='records', lines=True)

print("转换完成！")