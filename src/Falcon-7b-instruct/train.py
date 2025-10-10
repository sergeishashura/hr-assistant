from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from dotenv import load_dotenv
import torch
import os

load_dotenv()

print("HF_HOME =", os.getenv("HF_HOME"))
print("TRANSFORMERS_CACHE =", os.getenv("TRANSFORMERS_CACHE"))
print("OUTPUT_DIR =", os.getenv("OUTPUT_DIR"))

model_name = "tiiuae/falcon-7b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float32 if device == "cpu" else torch.float16,
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


dataset = load_dataset("json", data_files="hr_dataset.jsonl", split="train")


subset = dataset.select(range(int(len(dataset) * 0.3)))
print(f"Используем {len(subset)} примеров из {len(dataset)}")


trainer = SFTTrainer(
    model=model,
    train_dataset=subset,
    tokenizer=tokenizer,
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=1,
        output_dir="./outputs/hr-falcon-30",
        logging_steps=10,
        save_strategy="epoch",
    ),
)

trainer.train()


save_dir = "./models/Falcon-7b-instruct-30"
os.makedirs(save_dir, exist_ok=True)
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

print(f"Модель сохранена в {save_dir}")
