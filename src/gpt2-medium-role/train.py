import os
import torch
from datasets import load_dataset
from dotenv import load_dotenv
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

load_dotenv()

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using:", device)

model_name = "openai-community/gpt2-medium"

# --- tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(model_name)

special_tokens = {
    "additional_special_tokens": ["<|user|>", "<|assistant|>", "<|end|>"]
}
tokenizer.add_special_tokens(special_tokens)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# --- model ---
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
)

model.resize_token_embeddings(len(tokenizer))
model.to(device)

# --- LoRA ---
lora_config = LoraConfig(
    r=32,
    lora_alpha=16,
    target_modules=["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

# --- dataset ---
dataset_path = "../../dataset/kaggle_formdated_hr_dataset.json"
dataset = load_dataset("json", data_files=dataset_path, split="train")

def format_dialog(user, assistant):
    return (
        f"<|user|> {user}\n"
        f"<|assistant|> {assistant}<|end|>"
    )

def preprocess(examples):
    texts = []

    for msgs in examples["messages"]:
        user_msg = next((m["content"] for m in msgs if m["role"] == "user"), None)
        assistant_msg = next((m["content"] for m in msgs if m["role"] == "assistant"), None)

        if not (user_msg and assistant_msg):
            continue

        text = format_dialog(user_msg, assistant_msg)
        texts.append(text)

    tokens = tokenizer(
        texts,
        max_length=512,
        truncation=True,
        padding="max_length",
    )
    return tokens

dataset = dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)
dataset = dataset.select(range(int(len(dataset) * 0.5)))

collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# --- training ---
args = TrainingArguments(
    output_dir="./outputs/gpt2-hr",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=2,
    fp16=(device == "cuda"),
    save_strategy="epoch",
    save_total_limit=2,
    logging_steps=20,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=args,
    data_collator=collator,
)

trainer.train()

save_dir = "./models/gpt2-hr-lora"
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

print("Model saved to", save_dir)
