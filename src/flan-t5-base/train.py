import os
import torch
import numpy as np
from dotenv import load_dotenv
from evaluate import load as load_metric
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

load_dotenv()

print("HF_HOME =", os.getenv("HF_HOME"))
print("TRANSFORMERS_CACHE =", os.getenv("TRANSFORMERS_CACHE"))
print("OUTPUT_DIR =", os.getenv("OUTPUT_DIR"))

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Используем устройство: {device}")

model_name = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    cache_dir=os.getenv("TRANSFORMERS_CACHE"),
)

model.gradient_checkpointing_enable()


lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q", "v"],  # для T5 корректно
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_2_SEQ_LM",
)

model = get_peft_model(model, lora_config)


dataset_path = "../../dataset/kaggle_formdated_hr_dataset.json"
dataset = load_dataset("json", data_files=dataset_path, split="train")
print(f"Всего примеров в датасете: {len(dataset):,}")

max_source_length = 256
max_target_length = 256

def preprocess(examples):
    inputs, targets = [], []
    for msgs in examples["messages"]:
        user_msg = next((m["content"] for m in msgs if m["role"] == "user"), None)
        assistant_msg = next((m["content"] for m in msgs if m["role"] == "assistant"), None)
        if user_msg and assistant_msg:
            prompt = f"Answer this HR interview question professionally:\nQuestion: {user_msg}\nAnswer:"
            inputs.append(prompt)
            targets.append(assistant_msg)

    model_inputs = tokenizer(
        inputs, truncation=True, padding="max_length", max_length=max_source_length
    )
    labels = tokenizer(
        targets, truncation=True, padding="max_length", max_length=max_target_length
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

dataset = dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)

subset_size = int(len(dataset) * 0.05)
subset = dataset.select(range(subset_size))
print(f"Используем {len(subset)} примеров из {len(dataset)} (5%)")


data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

metric = load_metric("rouge")

def compute_metrics(eval_preds):
    preds, labels = eval_preds

    if isinstance(preds, tuple):
        preds = preds[0]

    preds = np.array(preds)
    labels = np.array(labels)

    if preds.ndim > 2:
        preds = np.argmax(preds, axis=-1)
    elif preds.dtype != np.int64 and preds.dtype != np.int32:
        preds = preds.astype(np.int64)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    processed_result = {}
    for k, v in result.items():
        if hasattr(v, "mid"):
            processed_result[k] = round(v.mid.fmeasure * 100, 2)
        else:
            processed_result[k] = round(float(v) * 100, 2)

    processed_result["gen_len"] = np.mean([len(p.split()) for p in decoded_preds])

    return processed_result

training_args = TrainingArguments(
    output_dir="./outputs/hr-flan-t5-lora",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    num_train_epochs=3,
    logging_steps=20,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    report_to="none",
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    max_grad_norm=1.0,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=subset,
    eval_dataset=subset.select(range(100)),
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    args=training_args,
)

trainer.train()

save_dir = "./models/flan-t5-lora"
os.makedirs(save_dir, exist_ok=True)
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print(f"\nМодель сохранена в {save_dir}")
