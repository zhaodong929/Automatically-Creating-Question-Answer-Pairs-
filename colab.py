%%capture
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps xformers "trl<0.9.0" peft accelerate bitsandbytes

#2加载模型
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

#3准备微调数据集
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}
### Input:
{}
### Response:
{}"""
EOS_TOKEN = tokenizer.eos_token # 必须添加 EOS_TOKEN
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # 必须添加EOS_TOKEN，否则无限生成
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass

from datasets import load_dataset
dataset = load_dataset("Cutewang/USDA-llama3.1-8b", split = "train")
dataset = dataset.map(formatting_prompts_func, batched = True,)

#4设置训练参数
from trl import SFTTrainer
from transformers import TrainingArguments

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, #  建议 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth", # 检查点，长上下文度
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # 可以让短序列的训练速度提高5倍。
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,  # 微调步数
        learning_rate = 2e-4, # 学习率
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

#5开始训练
trainer_stats = trainer.train()

#6测试微调后的模型
FastLanguageModel.for_inference(model)
inputs = tokenizer(
    [
        alpaca_prompt.format(
            "", # instruction
            "", # input
            "", # output
        )
    ], return_tensors = "pt").to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128)


#7保存LoRA模型
model.save_pretrained("lora_model") # Local saving
tokenizer.save_pretrained("lora_model")#（2024/06/16修改，官方库变化）

#9合并模型并量化成4位gguf保存 （目前colab有空间限制，可能会转换失败）
model.save_pretrained_gguf("model", tokenizer, quantization_method = "q4_k_m")

#10挂载google drive
from google.colab import drive
drive.mount('/content/drive')
#保存到Google drive
import shutil
source_file = '/content/model-unsloth.Q4_K_M.gguf'
destination_dir = '/content/drive/MyDrive/Llama3'
destination_file = f'{destination_dir}/model-unsloth.Q4_K_M.gguf'
shutil.copy(source_file, destination_file)