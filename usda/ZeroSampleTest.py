import pandas as pd
import subprocess
from bert_score import score as bert_score
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import re
import numpy as np

# Step 1: 从 CSV 文件中加载测试集
test_set_file = 'test4.csv'
df = pd.read_csv(test_set_file)

# 定义清理生成文本的函数
def clean_generated_text(text: str) -> str:
    text = re.sub(r"Here['’]s a specific.*?:", "", text, flags=re.DOTALL)
    text = re.sub(r"Here's a concise question related to the content.*?:", "", text, flags=re.DOTALL)
    text = re.sub(r"Based on the provided content.*?:", "", text, flags=re.DOTALL)
    text = re.sub(r"I would like to offer a.*?:", "", text, flags=re.DOTALL)
    text = re.sub(r"This question.*?\. ", "", text, flags=re.DOTALL)
    text = text.replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# 使用 llama3.1 模型生成答案
def generate_llm_answer(question: str) -> str:
    prompt = f"Provide a clear answer to the following question:\n\nQuestion: {question}"
    result = subprocess.run(
        ["ollama", "run", "llama3.1"],  # 请确保您的 llama3.1 模型可以通过命令行运行
        input=prompt,
        capture_output=True, text=True, encoding='utf-8'
    )

    if result.returncode == 0:
        answer = result.stdout.strip()
        return clean_generated_text(answer)
    else:
        print(f"Error generating answer: {result.stderr}")
        return None

# 计算评估指标的函数
def calculate_evaluation_metrics(reference: str, hypothesis: str) -> dict:
    # 确保 reference 和 hypothesis 是字符串类型，且不是空值
    if not isinstance(reference, str) or pd.isna(reference):
        reference = ""
    if not isinstance(hypothesis, str) or pd.isna(hypothesis):
        hypothesis = ""

    # 计算 BLEU 分数
    bleu = sentence_bleu([reference.split()], hypothesis.split()) if reference and hypothesis else 0.0

    # 计算 ROUGE 分数
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(reference, hypothesis)

    # 计算 BERTScore
    P, R, F1 = bert_score([hypothesis], [reference], lang="en", verbose=False)
    bertscore = F1.mean().item()

    return {
        "BLEU": bleu,
        "ROUGE-1": rouge_scores['rouge1'].fmeasure,
        "ROUGE-2": rouge_scores['rouge2'].fmeasure,
        "ROUGE-L": rouge_scores['rougeL'].fmeasure,
        "BERTScore": bertscore
    }

# Step 2: 使用 LLM 生成答案并与测试集答案进行对比
results = []
for index, row in df.iterrows():
    content = row['content']  # 文本内容
    question = row['question']  # 问题
    true_answer = row['answer']  # 测试集中的答案

    # 如果 true_answer 是空的或非字符串，跳过该条目
    if pd.isna(true_answer) or not isinstance(true_answer, str):
        continue

    # 使用 LLM 生成答案
    generated_answer = generate_llm_answer(question)

    # 如果模型未生成答案，跳过此条目
    if not generated_answer:
        continue

    # 计算评估指标
    metrics = calculate_evaluation_metrics(true_answer, generated_answer)

    # 存储结果
    results.append({
        "Content": content,
        "Question": question,
        "True Answer": true_answer,
        "Generated Answer": generated_answer,
        "BLEU": metrics["BLEU"],
        "ROUGE-1": metrics["ROUGE-1"],
        "ROUGE-2": metrics["ROUGE-2"],
        "ROUGE-L": metrics["ROUGE-L"],
        "BERTScore": metrics["BERTScore"]
    })

# Step 3: 保存评估结果到一个新的 CSV 文件
evaluation_results_df = pd.DataFrame(results)
evaluation_results_df.to_csv('evaluation_TestResults4.csv', index=False, encoding='utf-8')
print("Evaluation results saved to 'evaluation_TestResults.csv'.")
