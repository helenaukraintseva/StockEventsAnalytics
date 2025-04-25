import os
import json
import re

def sanitize_title(title):
    return re.sub(r"[\\/*?\"<>|]", "_", title)

def txt_to_rag_format(txt_folder="wiki", output_file="rag_data.jsonl"):
    files = [f for f in os.listdir(txt_folder) if f.endswith(".txt")]
    with open(output_file, "w", encoding="utf-8") as out:
        for filename in files:
            path = os.path.join(txt_folder, filename)
            with open(path, "r", encoding="utf-8") as file:
                content = file.read()
            title = os.path.splitext(filename)[0]
            record = {
                "title": title,
                "content": content,
                "source": filename
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"✅ Сохранено в {output_file}: {len(files)} документов")

# Пример вызова
if __name__ == "__main__":
    txt_to_rag_format("wiki", "rag_data.jsonl")
