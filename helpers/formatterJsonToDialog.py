import json

with open("data/drafts/hr_interview_questions_dataset.json", "r") as f:
    raw = json.load(f)

dialogs = []
for item in raw:
    dialog = {
        "messages": [
            {"role": "system", "content": f"You are an HR interviewer. Category: {item['category']}, Role: {item['role']}, Experience: {item['experience']}, Difficulty: {item['difficulty']}."},
            {"role": "user", "content": item["question"]},
            {"role": "assistant", "content": item["ideal_answer"]}
        ]
    }
    dialogs.append(dialog)

with open("data/kaggle_formdated_hr_dataset.json", "w") as f:
    json.dump(dialogs, f, indent=2, ensure_ascii=False)
