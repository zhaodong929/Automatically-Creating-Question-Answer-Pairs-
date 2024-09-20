import pandas as pd

# 定义CSV文件的路径列表
csv_files = ['results1.csv', 'results4.csv']

# 读取并合并所有CSV文件
dataframes = []
for file in csv_files:
    try:
        # 尝试使用 utf-8 编码读取文件
        df = pd.read_csv(file, encoding='utf-8')
    except UnicodeDecodeError:
        # 如果 utf-8 编码失败，使用 ISO-8859-1 或 latin1 编码
        print(f"Warning: Unable to read {file} with utf-8 encoding, trying ISO-8859-1.")
        df = pd.read_csv(file, encoding='ISO-8859-1')  # 或者使用 'latin1' 编码

    dataframes.append(df)

# 合并所有DataFrame
merged_df = pd.concat(dataframes, ignore_index=True)

# 保存合并后的结果到一个新的CSV文件
merged_df.to_csv('merged_output.csv', index=False, encoding='utf-8')

print("CSV file successfully merged and saved as 'merged_output.csv'")
