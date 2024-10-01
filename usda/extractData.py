import json
import os
import random

# 定义存储提取数据的列表
all_data = []

# 设置outcome文件夹路径
json_dir = "output//overall"  # 修改为你的outcome文件夹路径

# 遍历outcome文件夹中的所有json文件
for filename in os.listdir(json_dir):
    if filename.endswith(".json"):
        filepath = os.path.join(json_dir, filename)

        # 检查文件是否为空
        if os.path.getsize(filepath) == 0:
            print(f"文件 {filename} 是空的，跳过此文件。")
            continue

        try:
            # 逐行读取并检查文件内容
            with open(filepath, 'r', encoding='utf-8') as f:
                file_content = f.read().strip()
                if not file_content:
                    print(f"文件 {filename} 的内容为空或无效，跳过此文件。")
                    continue

                # 解析 JSON
                data = json.loads(file_content)

                # 确保是列表格式
                if not isinstance(data, list):
                    print(f"文件 {filename} 格式不正确（不是列表），跳过此文件。")
                    continue

            # 随机抽取200条数据
            extracted_data = random.sample(data, 200)

            # 将提取的数据添加到总列表中
            all_data.extend(extracted_data)

        except json.JSONDecodeError as e:
            print(f"文件 {filename} 读取失败: {e}")
        except Exception as e:
            print(f"处理文件 {filename} 时出现其他错误: {e}")

# 将合并的数据保存为新的json文件
total_data_len = len(all_data)
split_size = total_data_len // 5

# 分成五个部分保存
for i in range(5):
    start_index = i * split_size
    # 最后一份文件包含所有剩下的数据，以避免数据丢失
    if i == 4:
        part_data = all_data[start_index:]
    else:
        part_data = all_data[start_index:start_index + split_size]

    with open(f"{i+1}.json", 'w', encoding='utf-8') as f:
        json.dump(part_data, f, ensure_ascii=False, indent=4)

print("数据随机提取、合并并平分完成！")
