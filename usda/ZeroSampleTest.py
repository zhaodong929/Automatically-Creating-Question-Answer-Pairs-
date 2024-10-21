import pandas as pd
import subprocess
from bert_score import score as bert_score
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import re

# Step 1: Load the test set from a CSV file
test_set_file = 'test1.csv' 
df = pd.read_csv(test_set_file)

# Define a function to clean generated text
def clean_generated_text(text: str) -> str:
    text = re.sub(r"Here['’]s a specific.*?:", "", text, flags=re.DOTALL)
    text = re.sub(r"Here's a concise question related to the content.*?:", "", text, flags=re.DOTALL)  
    text = re.sub(r"Based on the provided content.*?:", "", text, flags=re.DOTALL)
    text = re.sub(r"I would like to offer a.*?:", "", text, flags=re.DOTALL)
    text = re.sub(r"This question.*?\. ", "", text, flags=re.DOTALL)
    text = text.replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# Generate an answer using the llama3.1 model
def generate_llm_answer(question: str) -> str:
    prompt = f"Provide a clear answer to the following question:\n\nQuestion: {question}"
    result = subprocess.run(
        ["ollama", "run", "llama3.1"],  # Ensure that your llama3.1 model can be run from the command line
        input=prompt,
        capture_output=True, text=True, encoding='utf-8'
    )

    if result.returncode == 0:
        answer = result.stdout.strip()
        return clean_generated_text(answer)
    else:
        print(f"Error generating answer: {result.stderr}")
        return None

# Function to calculate evaluation metrics
def calculate_evaluation_metrics(reference: str, hypothesis: str) -> dict:
    # Calculate BLEU score
    bleu = sentence_bleu([reference.split()], hypothesis.split())

    # Calculate ROUGE score
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(reference, hypothesis)

    # Calculate BERTScore
    P, R, F1 = bert_score([hypothesis], [reference], lang="en", verbose=False)
    bertscore = F1.mean().item()

    return {
        "BLEU": bleu,
        "ROUGE-1": rouge_scores['rouge1'].fmeasure,
        "ROUGE-2": rouge_scores['rouge2'].fmeasure,
        "ROUGE-L": rouge_scores['rougeL'].fmeasure,
        "BERTScore": bertscore
    }

# Step 2: Generate answers using LLM and compare with test set answers
results = []
for index, row in df.iterrows():
    content = row['content']  # Content text
    question = row['question']  # Question
    true_answer = row['answer']  # Answer from the test set

    # Generate answer using LLM
    generated_answer = generate_llm_answer(question)

    # Skip entry if no answer is generated
    if not generated_answer:
        continue

    # Calculate evaluation metrics
    metrics = calculate_evaluation_metrics(true_answer, generated_answer)

    # Store results
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

# Step 3: Save evaluation results to a new CSV file
evaluation_results_df = pd.DataFrame(results)
evaluation_results_df.to_csv('evaluation_TestResults1.csv', index=False, encoding='utf-8')
print("Evaluation results saved to 'evaluation_TestResults1.csv'.")
