import json
import os

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel
from dotenv import load_dotenv
import torch


load_dotenv()

print("HF_HOME =", os.getenv("HF_HOME"))
print("TRANSFORMERS_CACHE =", os.getenv("TRANSFORMERS_CACHE"))
print("OUTPUT_DIR =", os.getenv("OUTPUT_DIR"))


def run_inference(model, tokenizer, questions, device, model_label):
    results = []
    for q in questions:
        prompt = q["question"]
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            temperature=None,
            top_p=None,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        results.append({"question": prompt, "model": model_label, "answer": answer})
    return results


def test_t5_model_pair(
    base_model,
    peft_model,
    questions_file="data/test_questions/test_questions.json",
):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))

    questions_path = os.path.join(project_root, questions_file)


    tokenizer = AutoTokenizer.from_pretrained(
        base_model, cache_dir=os.getenv("TRANSFORMERS_CACHE")
    )


    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    base_output_dir = os.getenv("OUTPUT_DIR", "../outputs")
    model_name_safe = base_model.split("/")[-1]
    model_output_dir = os.path.join(base_output_dir, model_name_safe)
    os.makedirs(model_output_dir, exist_ok=True)

    base = AutoModelForSeq2SeqLM.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype=torch.float32,
        cache_dir=os.getenv("TRANSFORMERS_CACHE"),
    )
    base_results = run_inference(base, tokenizer, questions, device, "base")

    base_out_path = os.path.join(model_output_dir, "results_base.json")
    with open(base_out_path, "w", encoding="utf-8") as f:
        json.dump(base_results, f, indent=2, ensure_ascii=False)
    print(f"Результаты базовой модели сохранены в {base_out_path}")



    finetuned = PeftModel.from_pretrained(base, peft_model)
    finetuned_results = run_inference(
        finetuned, tokenizer, questions, device, "fine-tuned"
    )

    finetuned_out_path = os.path.join(model_output_dir, "results_finetuned.json")
    with open(finetuned_out_path, "w", encoding="utf-8") as f:
        json.dump(finetuned_results, f, indent=2, ensure_ascii=False)
    print(f"Результаты fine-tuned модели сохранены в {finetuned_out_path}")


if __name__ == "__main__":
    test_t5_model_pair(
        base_model="google/flan-t5-base",
        peft_model="../models/flan-t5-05",
        questions_file="data/test_questions/test_questions.json",
    )
