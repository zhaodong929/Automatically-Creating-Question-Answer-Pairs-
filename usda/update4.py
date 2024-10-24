import json
import csv
import subprocess
import re
from typing import List
from keybert import KeyBERT

# 使用 KeyBERT 提取关键词
kw_model = KeyBERT(model='bert-base-uncased')

# 加载JSON文件时指定编码格式为UTF-8
with open('output/test2/4.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 数据预处理函数，清理内容
def preprocess_content(content: str) -> str:
    return content.replace("\n", " ").strip()

processed_data = [preprocess_content(item['content']) for item in data]

# 通用的冗余清理函数
def clean_generated_text(text: str) -> str:
    """清理生成的文本，移除各种形式的冗余描述"""
    # 匹配所有可能的冗余前缀和句式
    text = re.sub(r"Here['’]s a specific.*?:", "", text, flags=re.DOTALL)
    text = re.sub(r"Here is a specific.*?:", "", text, flags=re.DOTALL)
    text = re.sub(r"Based on the provided content.*?:", "", text, flags=re.DOTALL)
    text = re.sub(r"As an .*? at .*?,", "", text, flags=re.DOTALL)  # 处理类似职位背景的描述
    text = re.sub(r"I would like to offer a.*?:", "", text, flags=re.DOTALL)
    text = re.sub(r"This question.*?\. ", "", text, flags=re.DOTALL)
    text = re.sub(r"Here is a clear and specific question related to agriculture:.*?\. ", "", text, flags=re.DOTALL)
    text = re.sub(r"Here is a clear and specific question related to agriculture based on the provided content:.*?\. ", "", text, flags=re.DOTALL)
    text = re.sub(r"Here is a clear and specific question based on the obtained keywords:.*?\. ", "", text, flags=re.DOTALL)
    text = re.sub(r"(In light of the following content|To provide further clarity).*?:", "", text, flags=re.DOTALL)
    text = re.sub(r"(According to the text|Based on the content|Provided in the text).*?:", "", text, flags=re.DOTALL)
    text = text.replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# 使用 BERT 提取关键词
def extract_keywords(content: str, num_keywords: int = 5) -> list:
    """通过 BERT 提取内容的关键词"""
    keywords = kw_model.extract_keywords(content, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=num_keywords)
    # keywords 格式为[(keyword, score), ...]，我们只提取关键词
    return [kw[0] for kw in keywords]

# 使用 llama3.1 生成问题
def generate_question(content: str) -> str:
    keywords = extract_keywords(content)
    prompt = (
        f"Analyze the following content and extract the keywords related to agriculture"
        f"Generate a clear and specific question based on obtained keywords. "
        f"Keywords do not need to be shown."
        f"Just generate the question, other cohesive statement is not required."
        f"The question must be related to agriculture."
        f"The question can be answered according to the provided content. but the basis does not need to be shown"
        f"Ensure the question is directly related to the provided content. Keep it brief and helpful:\n\n"
        f"{', '.join(keywords)}.\n\n{content}"
    )
    # 调用本地 llama3.1 模型生成问题
    result = subprocess.run(
        ["ollama", "run", "llama3.1"],  # 使用 llama3.1 模型
        input=prompt,
        capture_output=True, text=True,
        encoding='utf-8'
    )

    if result.returncode == 0:
        question = result.stdout.strip()
        return clean_generated_text(question)
    else:
        print(f"Error generating question: {result.stderr}")
        return "Error generating question."

# 使用 llama3.1 生成答案
def generate_answer(content: str, question: str) -> str:
    prompt = (
        f"Provide a concise and accurate answer to the following question, based on the provided content. "
        f"Just generate the answer, no pleasantries necessary."
        f"The answer must be directly related to the content. "
        f"If there is no answer according to the content, you can give me an answer by searching website."
        f"The answer must be the solution to deal with the question."
        f"The answer can be found in the provided content:\n\n"
        f"Content: {content}\n\nQuestion: {question}"
    )
    result = subprocess.run(
        ["ollama", "run", "llama3.1"],  # 调用 llama3.1 生成答案
        input=prompt,
        capture_output=True, text=True,
        encoding='utf-8'
    )

    if result.returncode == 0:
        answer = result.stdout.strip()
        clean_answer = clean_generated_text(answer)
        if "no information" in clean_answer.lower() or "none" in clean_answer.lower():
            return None
        return clean_answer
    else:
        print(f"Error generating answer: {result.stderr}")
        return None


# 定义Agent类
class Agent:
    def __init__(self, role: str, goal: str, backstory: str, verbose: bool = False):
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.verbose = verbose

    def process(self, content: str, question: str = None) -> str:
        if self.verbose:
            print(f"{self.role} is processing the content...")

        if self.role == "Question Generator":
            return generate_question(content)

        elif self.role == "Answer Generator":
            return generate_answer(content, question) if question else "No question to answer."

        else:
            return "Unknown role."

# 定义Task类
class Task:
    def __init__(self, description: str, expected_output: str, agent: Agent):
        self.description = description
        self.expected_output = expected_output
        self.agent = agent

    def execute(self, content: str, question: str = None) -> str:
        return self.agent.process(content, question)

# 定义Crew类
class Crew:
    def __init__(self, agents: List[Agent], tasks: List[Task], verbose: bool = False):
        self.agents = agents
        self.tasks = tasks
        self.verbose = verbose

    def kickoff(self, inputs: dict) -> dict:
        content = inputs["content"]
        question = None
        qa_pairs = []

        # 依次执行每个任务，保持递进关系
        for task in self.tasks:
            if task.agent.role == "Question Generator":
                question = task.execute(content)
            elif task.agent.role == "Answer Generator":
                answer = task.execute(content, question)
                if answer:
                    qa_pairs.append({
                        "content": content,
                        "question": question,
                        "answer": answer
                    })

        return qa_pairs

# 定义Agent
question_generator = Agent(
    role="Question Generator",
    goal="Generate important relevant questions from extracted content.",
    backstory="Generates clear and relevant questions that focus on contemporary agricultural development.",
    verbose=True
)

answer_generator = Agent(
    role="Answer Generator",
    goal="Generate accurate answers based on the content and questions provided.",
    backstory="Generates answers that are directly relevant to the questions and are based on the dataset content.",
    verbose=True
)

# 定义任务
question_generation_task = Task(
    description="Generate relevant questions based on the extracted content.",
    expected_output="the questions relevant to the content.",
    agent=question_generator
)

answer_generation_task = Task(
    description="Generate accurate answers based on the questions and content provided.",
    expected_output="the correct answers corresponding to the generated questions.",
    agent=answer_generator
)

# 创建Crew
qa_crew = Crew(
    agents=[question_generator, answer_generator],
    tasks=[question_generation_task, answer_generation_task],
    verbose=True
)

# 生成每个内容的问答对，并将结果保存为JSON文件
qa_results2 = []
try:
    for content in processed_data:
        inputs = {
            "content": content,
        }

        # 执行crew生成问答对
        qa_pairs = qa_crew.kickoff(inputs=inputs)
        qa_results2.extend(qa_pairs)

except KeyboardInterrupt:
    print("Process was manually stopped.")

finally:
    # 无论是否中断，保存已生成的问答对
    with open('output/outcome/demo4_2.json', 'w', encoding='utf-8') as f:
        json.dump(qa_results2, f, ensure_ascii=False, indent=4)
    print("Question and answer pairs are generated and saved to a file 'demo1.json'")

    # 读取qa_results2.json并转换为CSV格式
    with open('output/outcome/demo4_2.json', 'r', encoding='utf-8') as json_file:
        qa_data = json.load(json_file)

    # 写入CSV文件
    with open('output/outcome/demo4_2.csv', 'w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        # 写入标题行
        csv_writer.writerow(['content', 'question', 'answer'])

        # 写入每一行数据
        for entry in qa_data:
            csv_writer.writerow([entry['content'], entry['question'], entry['answer']])

    print("CSV file 'demo4_2.csv' created")