import os
import torch
import numpy as np
from dotenv import load_dotenv
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model
from evaluate import load as load_metric

load_dotenv()

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Используем устройство:", device)

model_name = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,   
)

# Включаем градиентные контрольные точки
model.gradient_checkpointing_enable()

# LoRA конфиг
lora_config = LoraConfig(
    r=32,
    lora_alpha=16,
    target_modules=["query", "value"],  # для BERT
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_CLS",
)

model = get_peft_model(model, lora_config)


# -----------------------------
# Загружаем датасет
# -----------------------------

dataset_path = "../../dataset/kaggle_formdated_hr_dataset.json"
dataset = load_dataset("json", data_files=dataset_path, split="train")

print("Всего примеров:", len(dataset))

# -----------------------------
# Препроцессинг
# -----------------------------

def preprocess(examples):
    texts = []
    labels = []

    for msgs in examples["messages"]:
        user_msg = next((m["content"] for m in msgs if m["role"] == "user"), None)
        assistant_msg = next((m["content"] for m in msgs if m["role"] == "assistant"), None)

        if user_msg and assistant_msg:
            text = f"HR Question: {user_msg}\nCandidate Answer: {assistant_msg}"
            texts.append(text)
            labels.append(1)  # все примеры считаем "корректными" (можно изменить)

    enc = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=256,
    )
    enc["labels"] = labels
    return enc


dataset = dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)

subset_size = int(len(dataset) * 0.05)
subset = dataset.select(range(subset_size))

eval_set = subset.select(range(100))

collator = DataCollatorWithPadding(tokenizer)


# -----------------------------
# Метрики
# -----------------------------


metric = load_metric("accuracy")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    preds = np.argmax(preds, axis=1)
    return metric.compute(predictions=preds, references=labels)


# -----------------------------
# Training
# -----------------------------

training_args = TrainingArguments(
    output_dir="./outputs/hr-bert-lora",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=2e-5,
    num_train_epochs=3,
    warmup_ratio=0.1,
    logging_steps=20,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    report_to="none",
)

trainer = Trainer(
    model=model,
    train_dataset=subset,
    eval_dataset=eval_set,
    data_collator=collator,
    compute_metrics=compute_metrics,
    args=training_args,
)

trainer.train()

save_dir = "./models/bert-lora"
os.makedirs(save_dir, exist_ok=True)
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

print(f"\nМодель сохранена в {save_dir}")
