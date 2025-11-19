import json
import os

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from dotenv import load_dotenv
import torch


load_dotenv()

print("HF_HOME =", os.getenv("HF_HOME"))
print("TRANSFORMERS_CACHE =", os.getenv("TRANSFORMERS_CACHE"))
print("OUTPUT_DIR =", os.getenv("OUTPUT_DIR"))


def run_gpt_inference(model, tokenizer, questions, device, model_label):
    results = []

    stop_token_id = tokenizer.convert_tokens_to_ids("<|end|>")

    for q in questions:

        prompt = f"<|user|> {q['question']}\n<|assistant|>"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.4,
                do_sample=False,
                eos_token_id=stop_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )

        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

        answer = answer.split("<|assistant|>")[-1].strip()

        results.append(
            {"question": q["question"], "model": model_label, "answer": answer}
        )

    return results


def test_gpt_model_pair(
    base_model,
    peft_model,
    questions_file="data/test_questions/test_questions.json",
):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    print(f"\nИспользуем устройство: {device}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    questions_path = os.path.join(project_root, questions_file)

    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    print("\nЗагружаем кастомный токенизатор LoRA...")
    tokenizer = AutoTokenizer.from_pretrained(
        peft_model,  # путь к gpt2-hr-lora
        cache_dir=os.getenv("TRANSFORMERS_CACHE"),
    )
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<|user|>", "<|assistant|>", "<|end|>"]}
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\nЗагружаем базовую модель GPT-2 medium...")
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float32,
        cache_dir=os.getenv("TRANSFORMERS_CACHE"),
    )

    base.resize_token_embeddings(len(tokenizer))
    base.to(device)

    base_output_dir = os.getenv("OUTPUT_DIR", "../outputs")
    model_name_safe = base_model.split("/")[-1]
    model_output_dir = os.path.join(base_output_dir, model_name_safe)
    os.makedirs(model_output_dir, exist_ok=True)

    print("\nЗагружаем fine-tuned (LoRA) модель...")
    finetuned = PeftModel.from_pretrained(
        base,
        peft_model,
    )
    finetuned.to(device)

    print("\nЗапуск inference LoRA модели...")
    finetuned_results = run_gpt_inference(
        finetuned, tokenizer, questions, device, "fine-tuned"
    )

    finetuned_out_path = os.path.join(model_output_dir, "results_finetuned.json")
    with open(finetuned_out_path, "w", encoding="utf-8") as f:
        json.dump(finetuned_results, f, indent=2, ensure_ascii=False)
    print(f"Результаты fine-tuned модели сохранены в {finetuned_out_path}")


if __name__ == "__main__":


    test_gpt_model_pair(
        base_model="openai-community/gpt2-medium",
        peft_model="../models/gpt-2-medium-role/gpt2-hr-lora",
        questions_file="data/test_questions/test_questions.json",
    )
