import csv
import re

# 读取CSV文件并清理question内容
def clean_question(question: str) -> str:
    """
    清理question字段，移除从'Here is'开始到下一个冒号之间的句子
    """
    # 匹配从 'Here is' 到下一个冒号的文本，并将其替换为空字符串
    cleaned_question = re.sub(r"Here\s*(?:is|[''‘]s).*?:", "", question, flags=re.DOTALL)


    # 返回清理后的内容
    return cleaned_question.strip()

# 读取CSV文件并清理内容
def clean_csv_question(input_csv: str, output_csv: str):
    with open(input_csv, 'r', encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file)

        # 打开输出CSV文件用于写入
        with open(output_csv, 'w', newline='', encoding='utf-8') as output_file:
            # 获取CSV的字段名
            fieldnames = reader.fieldnames
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)

            # 写入标题行
            writer.writeheader()

            # 遍历CSV文件的每一行
            for row in reader:
                # 清理question字段
                row['question'] = clean_question(row['question'])

                # 写入清理后的行到输出文件
                writer.writerow(row)

    print(f"清理后的CSV文件已保存为: {output_csv}")

# 使用该函数清理CSV文件
input_csv = 'results.csv'
output_csv = 'results4.csv'
clean_csv_question(input_csv, output_csv)
