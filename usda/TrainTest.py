import pandas as pd
import subprocess
from bert_score import score as bert_score
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import re

# Step 1: Install necessary dependencies
%%capture 
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps xformers "trl<0.9.0" peft accelerate bitsandbytes

# Step 2: Load the model
from unsloth import FastLanguageModel
import torch
max_seq_length = 2048
dtype = None
load_in_4bit = True
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# Step 3: Prepare fine-tuning dataset
# Dataset should include text (context), question, and answer format
alpaca_prompt = """Below is a context followed by a question. Provide a clear and accurate response.

### Context:
{}
### Question:
{}
### Response:
{}"""
EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN

# Define a function to format dataset into model training format
def formatting_prompts_func(examples):
    contexts = examples["text"]  # Contexts
    questions = examples["question"]  # Questions
    answers = examples["answer"]  # Answers
    texts = []
    for context, question, answer in zip(contexts, questions, answers):
        # Must add EOS_TOKEN to prevent infinite generation
        text = alpaca_prompt.format(context, question, answer) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts }

# Load custom dataset which should include "text", "question", and "answer"
from datasets import load_dataset
dataset = load_dataset("TrainTest", split = "train")  # Replace with your dataset
dataset = dataset.map(formatting_prompts_func, batched=True)

# Step 3.1: Split dataset into training set (2500 samples) and validation set (300 samples) using interval sampling
validation_indices = list(range(0, 2800, 9))  # Every 9th sample as validation set, total 300 samples
train_indices = [i for i in range(2800) if i not in validation_indices]  # Remaining 2500 as training set

train_dataset = dataset.select(train_indices)  # Training set
valid_dataset = dataset.select(validation_indices)  # Validation set

# Step 4: Set training parameters
from trl import SFTTrainer
from transformers import TrainingArguments

# Adjust model parameters for question-answering task
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA adjustment
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0.05,  # Small dropout to reduce overfitting risk
    bias="none",
    use_gradient_checkpointing="unsloth",  # Use checkpointing to support long contexts
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,  # Training set
    eval_dataset=valid_dataset,   # Validation set
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,  # Parallel processing
    packing=False,  # Suitable for short sequences, increases training efficiency
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=80,  # Increase steps to improve model convergence
        learning_rate=1e-4,  # Adjust learning rate to prevent gradient explosion
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,  # Log every 10 steps
        evaluation_strategy="steps",  # Validate during training steps
        eval_steps=20,  # Validate every 20 steps
        save_steps=20,  # Save model every 20 steps
        save_total_limit=3,  # Save up to 3 checkpoints
        load_best_model_at_end=True,  # Load best model
        optim="adamw_8bit",  # Use 8-bit optimizer to reduce memory usage
        weight_decay=0.01,
        lr_scheduler_type="cosine",  # Use cosine learning rate scheduling
        seed=3407,
        output_dir="outputs",
    ),
)

# Step 5: Start training
trainer_stats = trainer.train()

# Step 6: Test the fine-tuned model
FastLanguageModel.for_inference(model)
inputs = tokenizer(
[
    alpaca_prompt.format(
        "Sample context",  # Example context
        "Sample question",  # Example question
        ""  # Empty output since we want to generate an answer
    )
], return_tensors="pt").to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=128)

# Step 7: Save the LoRA fine-tuned model
model.save_pretrained("lora_model")  # Save model
tokenizer.save_pretrained("lora_model")  # Save tokenizer

# Step 9: Merge model and quantize to 4-bit gguf format
model.save_pretrained_gguf("model", tokenizer, quantization_method="q4_k_m")

# Step 10: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')
# Save to Google Drive
import shutil
source_file = '/content/model-unsloth.Q4_K_M.gguf'
destination_dir = '/content/drive/MyDrive/Llama3'
destination_file = f'{destination_dir}/model-unsloth.Q4_K_M.gguf'
shutil.copy(source_file, destination_file)
