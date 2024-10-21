import json
import csv
import subprocess
import re
from typing import List
from transformers import pipeline
from keybert import KeyBERT

# Use KeyBERT to extract keywords
kw_model = KeyBERT(model='bert-base-uncased')

# Load JSON file with UTF-8 encoding
with open('new1.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Data preprocessing function to clean content
def preprocess_content(content: str) -> str:
    return content.replace("\n", " ").strip()

processed_data = [preprocess_content(item['content']) for item in data]

# Clean redundant descriptions
def clean_generated_text(text: str) -> str:
    text = re.sub(r"Here['’]s a specific.*?:", "", text, flags=re.DOTALL)
    text = re.sub(r"Here's a concise question related to the content.*?:", "", text, flags=re.DOTALL)  
    text = re.sub(r"Based on the provided content.*?:", "", text, flags=re.DOTALL)
    text = re.sub(r"I would like to offer a.*?:", "", text, flags=re.DOTALL)
    text = re.sub(r"This question.*?\. ", "", text, flags=re.DOTALL)
    text = text.replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# Use BERT to extract keywords
def extract_keywords(content: str, num_keywords: int = 5) -> list:
    """Extract keywords from content using BERT"""
    keywords = kw_model.extract_keywords(content, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=num_keywords)
    # Keywords format is [(keyword, score), ...], we only extract keywords
    return [kw[0] for kw in keywords]

# Generate questions using llama3.1
def generate_question(content: str) -> str:
    # Extract keywords
    keywords = extract_keywords(content)
    
    # Generate question prompt using keywords
    prompt = f"Generate a concise question related to the following content, focusing on the key concepts: {', '.join(keywords)}.\n\n{content}"

    # Run llama3.1 model to generate questions
    result = subprocess.run(
        ["ollama", "run", "llama3.1"],  # Use llama3.1 model
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

# Generate answers using llama3.1
def generate_answer(content: str, question: str) -> str:
    prompt = f"Provide a clear answer to the following question based on the content provided. The answer should be concise and directly relevant to the content:\n\nContent: {content}\n\nQuestion: {question}"

    result = subprocess.run(
        ["ollama", "run", "llama3.1"],  # Run llama3.1 to generate answer
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

# Define Agent class
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

# Define Task class
class Task:
    def __init__(self, description: str, expected_output: str, agent: Agent):
        self.description = description
        self.expected_output = expected_output
        self.agent = agent

    def execute(self, content: str, question: str = None) -> str:
        return self.agent.process(content, question)

# Define Crew class
class Crew:
    def __init__(self, agents: List[Agent], tasks: List[Task], verbose: bool = False):
        self.agents = agents
        self.tasks = tasks
        self.verbose = verbose

    def kickoff(self, inputs: dict) -> dict:
        content = inputs["content"]
        question = None
        qa_pairs = []

        # Execute each task sequentially, maintaining progress
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

# Define Agents
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

# Define Tasks
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

# Create Crew
qa_crew = Crew(
    agents=[question_generator, answer_generator],
    tasks=[question_generation_task, answer_generation_task],
    verbose=True
)

# Generate question-answer pairs for each content and save the results as a JSON file
qa_results2 = []
try:
    for content in processed_data:  
        inputs = {
            "content": content,
        }

        # Execute crew to generate question-answer pairs
        qa_pairs = qa_crew.kickoff(inputs=inputs)
        qa_results2.extend(qa_pairs)

except KeyboardInterrupt:
    print("Process was manually stopped.")

finally:
    # Save generated question-answer pairs regardless of interruption
    with open('newResults.json', 'w', encoding='utf-8') as f:
        json.dump(qa_results2, f, ensure_ascii=False, indent=4)
    print("Question and answer pairs are generated and saved to a file 'newResults.json'")

    # Read qa_results2.json and convert to CSV format
    with open('newResults.json', 'r', encoding='utf-8') as json_file:
        qa_data = json.load(json_file)

    # Write to CSV file
    with open('newResults.csv', 'w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        # Write header row
        csv_writer.writerow(['content', 'question', 'answer'])

        # Write each row of data
        for entry in qa_data:
            csv_writer.writerow([entry['content'], entry['question'], entry['answer']])

    print("CSV file 'newResults.csv' created")
