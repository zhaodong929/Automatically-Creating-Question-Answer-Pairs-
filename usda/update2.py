import json
import os
import subprocess  # 使用 subprocess 调用系统命令
from typing import List

# 加载JSON文件时指定编码格式为UTF-8
with open('Animals.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 数据预处理函数，清理内容
def preprocess_content(content: str) -> str:
    # 去掉不必要的换行符和多余的空格
    return content.replace("\n", " ").strip()

# 将所有的内容分类整理
processed_data = [preprocess_content(item['content']) for item in data]

# 使用 Ollama 本地模型生成问题，确保问题与内容密切相关
def generate_question(content: str) -> str:
    prompt = f"Based on the following content, generate a specific and focused question that could be asked to test understanding or gather further information:\n\n{content}\n\nThe question should focus on the main ideas or key concepts related to animal production, livestock health, risk management, or financial support."
    
    result = subprocess.run(
        ["ollama", "run", "llama3.1"],
        input=prompt,
        capture_output=True, text=True,
        encoding='utf-8'
    )
    
    if result.returncode == 0:
        return result.stdout.strip()
    else:
        print(f"Error generating question: {result.stderr}")
        return "Error generating question."

# 使用 Ollama 本地模型生成答案，确保答案针对问题，并符合内容描述
def generate_answer(content: str, question: str) -> str:
    prompt = f"Based on the following content, provide a detailed and accurate answer to the question:\n\nContent: {content}\n\nQuestion: {question}\n\nEnsure the answer is relevant to the key concepts of the content and fully addresses the question."
    
    result = subprocess.run(
        ["ollama", "run", "llama3.1"],
        input=prompt,
        capture_output=True, text=True,
        encoding='utf-8'
    )
    
    if result.returncode == 0:
        return result.stdout.strip()
    else:
        print(f"Error generating answer: {result.stderr}")
        return "Error generating answer."

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

        if self.role == "Content Extractor":
            # 提取相关内容
            return content

        elif self.role == "Question Generator":
            # 使用本地模型生成问题
            return generate_question(content)

        elif self.role == "Answer Generator":
            # 根据生成的问题生成答案
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
        intermediate_results = {}

        # 依次执行每个任务，保持递进关系
        for i, task in enumerate(self.tasks):
            if task.agent.role == "Question Generator":
                # 生成问题
                question = task.execute(content)
                intermediate_results[f"task_{i + 1}_question"] = question
            elif task.agent.role == "Answer Generator":
                # 生成答案，基于生成的问题
                answer = task.execute(content, question)
                intermediate_results[f"task_{i + 1}_answer"] = answer
            else:
                # 提取内容或执行其他任务
                result = task.execute(content)
                intermediate_results[f"task_{i + 1}_result"] = result

        return intermediate_results

# 定义三个Agent
content_extractor = Agent(
    role="Content Extractor",
    goal="Extract relevant content from agricultural reports for QnA generation.",
    backstory="Extracts important information from the dataset.",
    verbose=True
)

question_generator = Agent(
    role="Question Generator",
    goal="Generate questions from extracted content.",
    backstory="Generates clear and relevant questions that focus on key aspects of animal production and management.",
    verbose=True
)

answer_generator = Agent(
    role="Answer Generator",
    goal="Generate accurate answers based on the content and questions provided.",
    backstory="Generates answers that are directly relevant to the questions and are based on the dataset content.",
    verbose=True
)

# 定义任务
content_extraction_task = Task(
    description="Extract relevant content from the JSON data files.",
    expected_output="A section of content from the dataset for QnA generation.",
    agent=content_extractor
)

question_generation_task = Task(
    description="Generate relevant questions based on the extracted content.",
    expected_output="A list of questions relevant to the content.",
    agent=question_generator
)

answer_generation_task = Task(
    description="Generate accurate answers based on the questions and content provided.",
    expected_output="A list of correct answers corresponding to the generated questions.",
    agent=answer_generator
)

# 创建Crew
qa_crew = Crew(
    agents=[content_extractor, question_generator, answer_generator],
    tasks=[content_extraction_task, question_generation_task, answer_generation_task],
    verbose=True
)

# 生成每个内容的问答对，并将结果保存为JSON文件
qa_results2 = []
try:
    for content in processed_data:
        inputs = {
            "content": content,
            "questions": None,
            "answers": None
        }

        # 执行crew生成问答对
        result = qa_crew.kickoff(inputs=inputs)
        qa_results2.append(result)

except KeyboardInterrupt:
    print("Process was manually stopped.")

finally:
    # 无论是否中断，保存已生成的问答对
    with open('qa_results2.json', 'w', encoding='utf-8') as f:
        json.dump(qa_results2, f, ensure_ascii=False, indent=4)
    print("问答对已生成并保存到文件 'qa_results2.json'")
