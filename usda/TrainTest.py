import torch
import pandas as pd
from transformers import LlamaTokenizer, LlamaForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from transformers import DataCollatorForSeq2Seq
from evaluate import load
from bert_score import score as bert_score
import nltk
from nltk.translate.bleu_score import sentence_bleu
import signal
import sys

# 加载数据集
train_df = pd.read_csv('finally_results_cleaned_2.csv')
valid_df = pd.read_csv('sampled_data.csv')
test_df = pd.read_csv('merged_1000test_output.csv')

# 初始化模型和tokenizer
tokenizer = LlamaTokenizer.from_pretrained('huggingface/meta-llama/Llama-3.1-8B')
model = LlamaForCausalLM.from_pretrained('huggingface/meta-llama/Llama-3.1-8B')

# 加载PEFT微调模块
lora_config = LoraConfig(
    r=16,            # LoRA的秩
    lora_alpha=32,   # LoRA的alpha参数
    target_modules=["q_proj", "v_proj"],  # 只在某些模块应用LoRA
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# 将模型转换为PEFT模型，使用LoRA进行微调
model = get_peft_model(model, lora_config)

# 打印模型的可训练参数
model.print_trainable_parameters()

# Ctrl+C 终止处理函数
def signal_handler(sig, frame):
    print("\nTraining paused by Ctrl+C!")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# 微调函数
def fine_tune_model(model, tokenizer, train_df, valid_df, output_dir='./output', epochs=3, batch_size=8, learning_rate=5e-5):
    """
    使用PEFT框架对LLaMA 3.1模型进行微调，使用LoRA进行参数高效微调
    """
    # Step 1: 准备数据集
    train_dataset = Dataset.from_pandas(train_df)
    valid_dataset = Dataset.from_pandas(valid_df)

    # 数据预处理，将question转化为模型输入
    def preprocess_function(examples):
        inputs = examples['question']
        outputs = examples['answer']
        model_inputs = tokenizer(inputs, text_target=outputs, max_length=512, truncation=True)
        return model_inputs

    train_dataset = train_dataset.map(preprocess_function, batched=True)
    valid_dataset = valid_dataset.map(preprocess_function, batched=True)

    # Step 2: 设置TrainingArguments
    training_args = TrainingArguments(
        output_dir=output_dir,                 # 输出文件夹
        num_train_epochs=epochs,               # 微调的轮数
        per_device_train_batch_size=batch_size,# 训练时的batch大小
        per_device_eval_batch_size=batch_size, # 验证时的batch大小
        evaluation_strategy="epoch",           # 每轮评估一次
        learning_rate=learning_rate,           # 学习率
        weight_decay=0.01,                     # 权重衰减
        logging_dir=f'{output_dir}/logs',      # 日志存放的文件夹
        logging_steps=10,                      # 每10步记录一次日志
        save_total_limit=3,                    # 保存模型的最大数量
        load_best_model_at_end=True,           # 在结束时加载最优模型
        metric_for_best_model="eval_loss",     # 用于选择最佳模型的评估指标
    )

    # Step 3: 使用DataCollator处理动态padding
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Step 4: 定义评估指标（如果需要自定义）
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = torch.argmax(predictions, dim=-1)
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = torch.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # 计算BLEU、ROUGE等评估指标
        bleu_score = sentence_bleu([decoded_labels], decoded_preds)
        rouge_metric = load("rouge")  # 使用 evaluate.load 替代 load_metric
        rouge_scores = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)

        return {
            'bleu': bleu_score,
            'rouge1': rouge_scores['rouge1'].mid.fmeasure,
            'rouge2': rouge_scores['rouge2'].mid.fmeasure,
            'rougeL': rouge_scores['rougeL'].mid.fmeasure
        }

    # Step 5: 初始化Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # Step 6: 开始训练
    trainer.train()

    # 保存微调后的模型
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

# 开始模型微调
fine_tune_model(model, tokenizer, train_df, valid_df)

# 推理生成新的回答
def generate_answer(model, tokenizer, question):
    inputs = tokenizer(question, return_tensors="pt").input_ids
    outputs = model.generate(inputs, max_length=100)
    generated_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_answer

# 计算评估指标
def compute_metrics(generated_answer, reference_answer):
    reference = [reference_answer.split()]
    candidate = generated_answer.split()
    bleu_score = sentence_bleu(reference, candidate)

    rouge_metric = load("rouge")  # 替换 load_metric
    rouge_scores = rouge_metric.compute(predictions=[generated_answer], references=[reference_answer])

    P, R, F1 = bert_score([generated_answer], [reference_answer], lang='en')
    bert_score_f1 = F1.mean().item()

    metrics = {
        "BLEU": bleu_score,
        "ROUGE-1": rouge_scores["rouge1"].mid.fmeasure,
        "ROUGE-2": rouge_scores["rouge2"].mid.fmeasure,
        "ROUGE-L": rouge_scores["rougeL"].mid.fmeasure,
        "BERTScore": bert_score_f1
    }
    return metrics

# 为每个问题生成回答并计算评估指标
results = []
for index, row in test_df.iterrows():
    question = row['question']
    reference_answer = row['answer']

    # 生成回答
    generated_answer = generate_answer(model, tokenizer, question)

    # 计算评估指标
    metrics = compute_metrics(generated_answer, reference_answer)

    # 将结果添加到列表
    results.append({
        "question": question,
        "reference_answer": reference_answer,
        "generated_answer": generated_answer,
        "BLEU": metrics['BLEU'],
        "ROUGE-1": metrics['ROUGE-1'],
        "ROUGE-2": metrics['ROUGE-2'],
        "ROUGE-L": metrics['ROUGE-L'],
        "BERTScore": metrics['BERTScore']
    })

# 生成最终的CSV文件
final_df = pd.DataFrame(results)
final_df.to_csv('final_output_with_metrics.csv', index=False)
print("评估结果已保存到 'final_output_with_metrics.csv'.")
