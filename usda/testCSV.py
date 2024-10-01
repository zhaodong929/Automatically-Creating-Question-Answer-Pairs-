import pandas as pd

# 读取 CSV 文件
df = pd.read_csv("finally_results_cleaned_2.csv")

# 从数据中随机抽取 300 条数据
sampled_df = df.sample(n=5)

# 将抽取的数据保存到一个新的 CSV 文件中
sampled_df.to_csv('sampled_data2.csv', index=False)

print("随机抽取的100条数据已保存到 sampled_data2.csv 文件中。")
