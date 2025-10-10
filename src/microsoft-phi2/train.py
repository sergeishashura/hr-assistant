from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import torch, os
from dotenv import load_dotenv

load_dotenv()

model_name = "microsoft/phi-2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Используем устройство: {device}")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16 if device != "cpu" else torch.float32,
    cache_dir=os.getenv("TRANSFORMERS_CACHE")
)

lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

dataset = load_dataset("json", data_files="./data/kaggle_formdated_hr_dataset.json", split="train")

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
subset = dataset.select(range(int(len(dataset) * 0.05)))

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = SFTTrainer(
    model=model,
    train_dataset=subset,
    data_collator=data_collator,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=1,
        output_dir="./outputs/hr-gpt-05",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        fp16=True if device == "cuda" else False,
    ),
)

trainer.train()

save_dir = "./models/hr-gpt-05"
os.makedirs(save_dir, exist_ok=True)
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print(f"Модель сохранена в {save_dir}")
