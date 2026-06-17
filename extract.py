import json

all_text = ""

with open("chatAlpaca_chat.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        obj = json.loads(line)

        # varje rad är {"chat": [...]}
        for msg in obj["chat"]:
            print("loading:", msg["content"])
            all_text += msg["content"] + "\n"

with open("english_training_data.txt", "w", encoding="utf-8") as f:
    f.write(all_text)
