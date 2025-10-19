from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import torch
import os
from dotenv import load_dotenv

load_dotenv()

print("HF_HOME =", os.getenv("HF_HOME"))
print("TRANSFORMERS_CACHE =", os.getenv("TRANSFORMERS_CACHE"))
print("OUTPUT_DIR =", os.getenv("OUTPUT_DIR"))

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"\n Используем устройство: {device.upper()}")


model_name = "openai-community/gpt2-medium"

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto" if device == "cuda" else None,  # распределение по GPU
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    cache_dir=os.getenv("TRANSFORMERS_CACHE"),
)

model.to(device)

lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["c_attn", "c_proj"],  # правильные модули для GPT-2
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

dataset_path = os.path.join(os.path.dirname(__file__), "../../dataset/kaggle_formdated_hr_dataset.json")
print(f"\nЗагружаем датасет: {dataset_path}")

dataset = load_dataset("json", data_files=dataset_path, split="train")

def preprocess(examples):
    texts = []
    for msgs in examples["messages"]:
        user_msg = next((m["content"] for m in msgs if m["role"] == "user"), None)
        assistant_msg = next((m["content"] for m in msgs if m["role"] == "assistant"), None)
        if user_msg and assistant_msg:
            text = f"<|user|> {user_msg}\n<|assistant|> {assistant_msg}"
            texts.append(text)

    return tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=512,
    )

dataset = dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)
subset = dataset.select(range(int(len(dataset) * 0.05)))  # можно уменьшить на тесте

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=1,
    output_dir="./outputs/hr-gpt-05",
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    fp16=True if device == "cuda" else False,
    bf16=False,
    optim="adamw_torch",
    report_to="none",
    dataloader_num_workers=2,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=subset,
    data_collator=data_collator,
    args=training_args,
)

print("\nЗапускаем обучение...\n")
trainer.train()

save_dir = "./models/hr-gpt-05"
os.makedirs(save_dir, exist_ok=True)
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

print(f"\nМодель сохранена в {save_dir}")
