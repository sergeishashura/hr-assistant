from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from dotenv import load_dotenv
import torch
import json
import os


load_dotenv()

print("HF_HOME =", os.getenv("HF_HOME"))
print("TRANSFORMERS_CACHE =", os.getenv("TRANSFORMERS_CACHE"))
print("OUTPUT_DIR =", os.getenv("OUTPUT_DIR"))

def test_model(base_model, peft_model=None, questions_file="data/test_questions/test_questions.json", output_file="results.json"):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        cache_dir=os.getenv("TRANSFORMERS_CACHE")
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype=torch.float32,
        cache_dir=os.getenv("TRANSFORMERS_CACHE")
    )

    if peft_model:
        model = PeftModel.from_pretrained(model, peft_model)

    with open(questions_file, "r") as f:
        questions = json.load(f)

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
          pad_token_id=tokenizer.eos_token_id
          )
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

        results.append({
            "question": prompt,
            "model": "base" if peft_model is None else "fine-tuned",
            "answer": answer
        })

    os.makedirs(os.getenv("OUTPUT_DIR", "./outputs"), exist_ok=True)
    out_path = os.path.join(os.getenv("OUTPUT_DIR", "./outputs"), output_file)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✅ Результаты сохранены в {out_path}")

if __name__ == "__main__":
    # Тест до обучения
    # test_model(
    #     base_model="tiiuae/falcon-7b-instruct",
    #     peft_model=None,
    #     questions_file="data/test_questions/test_questions.json",
    #     output_file="results_base.json"
    # )

    # Тест после обучения (разкомментируй, когда появятся чекпоинты)
    # test_model(
    #     base_model="tiiuae/falcon-7b-instruct",
    #     peft_model="./hr-falcon-lora-30",
    #     questions_file="data/test_questions/test_questions.json",
    #     output_file="results_finetuned_30.json"
    # )

    test_model(
        base_model="microsoft/phi-2",
        peft_model="models/hr-gpt-30",
        questions_file="data/test_questions/test_questions.json",
        is_causal=True
    )
