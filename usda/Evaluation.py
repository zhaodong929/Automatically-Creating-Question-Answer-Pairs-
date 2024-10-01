import pandas as pd
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import re
from keybert import KeyBERT

# 加载BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
kw_model = KeyBERT(model='bert-base-uncased')

# 清理文本的函数
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\W+', ' ', text)
    return text

# 计算文本相似度 (Relevance)
def calculate_similarity(text1, text2):
    # 对文本进行分词和编码
    inputs1 = tokenizer(text1, return_tensors='pt', max_length=512, truncation=True, padding=True)
    inputs2 = tokenizer(text2, return_tensors='pt', max_length=512, truncation=True, padding=True)
    
    # 通过 BERT 模型计算嵌入
    outputs1 = model(**inputs1)
    outputs2 = model(**inputs2)
    
    # 取出最后一层隐藏状态
    embeddings1 = outputs1.last_hidden_state.mean(dim=1).detach().numpy()
    embeddings2 = outputs2.last_hidden_state.mean(dim=1).detach().numpy()
    
    # 计算余弦相似度
    similarity = cosine_similarity(embeddings1, embeddings2)
    return similarity[0][0]

# 计算简洁性 (Conciseness)
def calculate_conciseness(text):
    # 计算文本的长度
    length = len(str(text).split())
    return length

# 使用 BERT 提取关键词
def extract_keywords(content: str, num_keywords: int = 5) -> list:
    """通过 BERT 提取内容的关键词"""
    keywords = kw_model.extract_keywords(content, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=num_keywords)
    return [kw[0] for kw in keywords]

# 计算覆盖度 (Coverage) - 基于关键词
def calculate_coverage(text, key_points):
    matched_keywords = 0
    total_keywords = len(key_points)
    
    for word in key_points:
        if word in text:
            matched_keywords += 1
    
    coverage = matched_keywords / total_keywords if total_keywords > 0 else 0
    return coverage

# 计算一致性 (Consistency)
def calculate_consistency(text, answer):
    consistency_score = calculate_similarity(text, answer)
    return consistency_score

# 评估问题和答案的质量
def evaluate_question_answer(row):
    text = clean_text(row['content'])
    question = clean_text(row['question'])
    answer = clean_text(row['answer'])

    # 1. 提取关键词
    key_points = extract_keywords(row['content'])

    # 2. 计算关联度
    relevance_score_question = calculate_similarity(text, question)
    relevance_score_answer = calculate_similarity(text, answer)
    
    # 3. 计算简洁性
    conciseness_score_question = calculate_conciseness(row['question'])
    conciseness_score_answer = calculate_conciseness(row['answer'])
    
    # 4. 计算覆盖度
    coverage_score_answer = calculate_coverage(answer, key_points)
    


    return {
        "relevance_question": relevance_score_question,
        "relevance_answer": relevance_score_answer,
        "conciseness_question": conciseness_score_question,
        "conciseness_answer": conciseness_score_answer,
        "coverage_answer": coverage_score_answer,
       
    }

# 处理CSV文件并计算每一行和整体的综合指标
def process_file(file_path):
    df = pd.read_csv(file_path)

    # 对每一行进行评估
    evaluation_results = df.apply(evaluate_question_answer, axis=1)

    # 将评估结果加入到DataFrame（只保留指标部分）
    evaluation_df = pd.DataFrame(evaluation_results.tolist())

    # 计算综合平均值
    avg_relevance_question = evaluation_df['relevance_question'].mean()
    avg_relevance_answer = evaluation_df['relevance_answer'].mean()
    avg_conciseness_question = evaluation_df['conciseness_question'].mean()
    avg_conciseness_answer = evaluation_df['conciseness_answer'].mean()
    avg_coverage_answer = evaluation_df['coverage_answer'].mean()


    # 添加平均指标作为最后一行
    avg_row = {
        'relevance_question': avg_relevance_question,
        'relevance_answer': avg_relevance_answer,
        'conciseness_question': avg_conciseness_question,
        'conciseness_answer': avg_conciseness_answer,
        'coverage_answer': avg_coverage_answer,
       
    }

    evaluation_df = evaluation_df.append(avg_row, ignore_index=True)

    return evaluation_df

# 示例: 调用 process_file 函数，提供 CSV 文件路径
file_path = 'output/everyone'  # 替换为实际文件路径
evaluation_df = process_file(file_path)

# 添加序号列
evaluation_df.insert(0, 'ID', range(1, len(evaluation_df) + 1))

# 将评估结果保存为CSV文件
evaluation_df.to_csv('finally_evaluation_results.csv', index=False)

# 打印综合平均指标
print("综合平均指标已添加到最后一行并保存到 evaluation_results.csv 文件中。")
