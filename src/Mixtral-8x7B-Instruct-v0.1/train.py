from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM,AutoModelForCausalLM, AutoTokenizer, TrainingArguments, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from dotenv import load_dotenv
import torch
import os


load_dotenv()

print("HF_HOME =", os.getenv("HF_HOME"))
print("TRANSFORMERS_CACHE =", os.getenv("TRANSFORMERS_CACHE"))
print("OUTPUT_DIR =", os.getenv("OUTPUT_DIR"))

model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model_name)

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
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
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
            texts.append(f"User: {user_msg}\nAssistant: {assistant_msg}")

    model_inputs = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=512
    )
    return model_inputs


dataset = dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)

subset_size = int(len(dataset) * 0.05)
subset = dataset.select(range(subset_size))
print(f"Используем {len(subset)} примеров из {len(dataset)} (30%)")


data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = SFTTrainer(
    model=model,
    train_dataset=subset,
    data_collator=data_collator,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=1,
        output_dir="./outputs/Mixtral-8x7B-Instruct-v0.1-05",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
    ),
)


trainer.train()


save_dir = "./models/Mixtral-8x7B-Instruct-v0.1-05"
os.makedirs(save_dir, exist_ok=True)
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

print(f"Модель сохранена в {save_dir}")
