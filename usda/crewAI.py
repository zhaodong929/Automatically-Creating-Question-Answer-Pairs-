import os
import json
from crewai import Agent, Task, Crew, Process
import spacy

# 设置 OpenAI API Key
os.environ["OPENAI_API_KEY"] = " "
os.environ["OPENAI_MODEL_NAME"] = "gpt-3.5-turbo"

# 加载 NLP 模型，例如 spaCy
nlp = spacy.load("en_core_web_sm")

# 自定义函数读取 JSON 数据
def read_data_from_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded data from {file_path}")
    return data

# 自定义函数生成问答对
def generate_qna_pairs(text):
    doc = nlp(text)
    # 简单问答对生成示例
    qna_pairs = [{"question": "What is this text about?", "answer": doc.text}]
    return qna_pairs

# 创建 Data Management Agent
storage_agent = Agent(
    role="Structured Data Manager",
    goal="Read and manage structured data for NLP processing.",
    verbose=True,
    backstory=(
        "You are responsible for managing data efficiently for further processing. "
        "With your organizational skills, you excel at ensuring data is ready for analysis and further use."
    )
)

# 读取并处理 JSON 文件的 Task
data_management_task = Task(
    description=(
        "Your task is to read structured data from a stored JSON file. "
        "The data should be prepared and cleaned to be suitable for NLP processing, ensuring that it's structured properly."
    ),
    expected_output="Data loaded and prepared for NLP processing.",
    tools=[],  # 使用 Python 内置方法，不需要额外工具
    agent=storage_agent,
    async_execution=False  # 修改为同步执行
)

# 创建 QnA Generation Agent
qna_agent = Agent(
    role="Domain-Specific QnA Expert",
    goal="Generate question-and-answer pairs from the structured data using NLP techniques.",
    verbose=True,
    backstory=(
        "As a QnA generation expert, your role is to transform structured data into useful questions and answers. "
        "You leverage advanced NLP techniques to understand the content and create relevant QnA pairs that help in training conversational models."
    )
)

# QnA 生成的 Task
qna_task = Task(
    description=(
        "Utilize NLP models to analyze the structured data and generate meaningful QnA pairs. "
        "Ensure that the questions and answers are contextually accurate, reflecting the specific domain knowledge."
    ),
    expected_output="A structured list of QnA pairs, properly formatted and validated for further use.",
    tools=[],  # 使用 Python NLP 库处理文本
    agent=qna_agent,
    async_execution=True  # 保持为异步执行
)

# 组建 Crew 并执行任务
agricultural_crew = Crew(
    agents=[storage_agent, qna_agent],
    tasks=[data_management_task, qna_task],
    process=Process.sequential,  # 顺序执行任务，确保依赖关系
    verbose=True
)

# 示例：读取现有的 JSON 数据并生成问答对
json_file_path = "output/Forestry.json"  # 您提供的 JSON 文件路径
data = read_data_from_json(json_file_path)

# 运行 NLP 生成 QnA 对
for item in data:
    text = item.get("content", "")  # 假设每个 item 有一个 "content" 字段
    qna_pairs = generate_qna_pairs(text)
    print(f"Generated QnA Pairs: {qna_pairs}")

# 启动 Crew 执行
result = agricultural_crew.kickoff()
print(result)
