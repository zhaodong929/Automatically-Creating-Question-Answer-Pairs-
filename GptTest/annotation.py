import textwrap
import openai
import json

# 设置 OpenAI 的 API 密钥以进行 API 调用
openai.api_key = 'sk-isurHnUU_TOvoXiulIQqomzhvAlvIVpbEKLYQdovxtT3BlbkFJBho-Ytsq-eZYdseLBsaPYO_OWZFSdRA7xMFei6RAMA'

# 定义函数以根据文本生成问答对
def generate_qna_pairs(text):
    # 调用 OpenAI 的 Chat Completion 接口生成问答对
    response = openai.chat.completions.create(
        model="gpt-4",  # 使用 GPT-4 模型
        messages=[
            {"role": "system", "content": "You are an assistant that helps generate question and answer pairs."},  # 系统角色设定
            {"role": "user", "content": f"Generate several relevant question-answer pairs from the following text:\n\n{text}"}  # 传递的用户输入
        ],
        max_tokens=500,  # 最大生成的 tokens 数
        temperature=0.7  # 控制生成内容的多样性
    )
    # 返回生成的问答对内容
    return response.choices[0].message.content

# 定义函数，将文本块按指定最大长度切分
def split_text(text, max_length=1000):
    return textwrap.wrap(text, max_length)

# 打开文件并读取文件内容
with open('animals.txt', 'r', encoding='utf-8') as file:
    content = file.read()

# 将文件内容分成多个块
chunks = split_text(content)

# 初始化空列表以存储生成的问答对
qna_pairs = []

# 遍历每个文本块并生成对应的问答对
for chunk in chunks:
    qna_pairs.append(generate_qna_pairs(chunk))

# 将问答对存储为 JSON 格式
qna_json = {"qna_pairs": qna_pairs}

# 将结果写入文件，确保编码为 UTF-8 且格式美观
with open('qna_output.json', 'w', encoding='utf-8') as outfile:
    json.dump(qna_json, outfile, ensure_ascii=False, indent=4)
