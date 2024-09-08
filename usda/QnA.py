import os
import json
from transformers import pipeline

# Step 1: 读取 JSON 文件内容
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# Step 2: 初始化问题生成模型
def initialize_question_generator():
    # 使用 T5 模型，专用于生成问答对
    return pipeline("text2text-generation", model="valhalla/t5-small-qg-hl")

# Step 3: 生成问答对
def generate_qna_pairs(content_list, question_generator):
    qna_pairs = []

    for entry in content_list:
        content = entry.get('content', '')  # 获取段落内容

        if not content:  # 如果没有内容，则跳过
            continue

        highlighted_text = f"<hl> {content} <hl>"  # 在段落两边添加标记，供模型生成问题

        # 使用模型生成问题
        questions = question_generator(highlighted_text, max_length=64, do_sample=False)

        # 将生成的问题与原内容配对
        for question in questions:
            qna_pairs.append({
                "question": question['generated_text'],
                "answer": content
            })

    return qna_pairs

# Step 4: 保存问答对到 JSON 文件
def save_qna_to_json(qna_pairs, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(qna_pairs, file, ensure_ascii=False, indent=4)

# 主函数，执行所有步骤
def main():
    # 输出当前工作目录
    print(f"Current working directory: {os.getcwd()}")

    # 切换到正确的目录，如果需要
    # os.chdir('D:/159333/Automatically-Creating-Question-Answer-Pairs-/usda')

    # 读取 JSON 数据，路径指向 output 文件夹中的 Forestry.json
    # 使用绝对路径读取 JSON 文件
    content_list = load_json('D:/159333/Automatically-Creating-Question-Answer-Pairs-/usda/output/Forestry.json')


# 初始化问题生成器
    question_generator = initialize_question_generator()

    # 生成问答对
    qna_pairs = generate_qna_pairs(content_list, question_generator)

    # 输出生成的问答对
    for qna in qna_pairs:
        print(f"Question: {qna['question']}")
        print(f"Answer: {qna['answer']}")
        print('-' * 50)

    # 保存问答对到文件，保存路径在与脚本同一级
    save_qna_to_json(qna_pairs, 'Forestry_QnA.json')  # 保存结果到新的 JSON 文件中

if __name__ == "__main__":
    main()
