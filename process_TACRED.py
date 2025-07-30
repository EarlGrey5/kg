import os
import json

ori_path = "./data/Re-TACRED/ori_data"
save_path = "./data/Re-TACRED/ner_data"
os.makedirs(save_path, exist_ok=True)

data_files = ["train.json", "dev.json", "test.json"]

entity_types = set()

def convert_to_bio(tokens, ner_tags):
    bio_labels = []
    prev_tag = "O"
    for i, tag in enumerate(ner_tags):
        if tag == "O":
            bio_labels.append("O")
        elif i == 0 or tag != ner_tags[i - 1]:
            bio_labels.append("B-" + tag)
        else:
            bio_labels.append("I-" + tag)
    return bio_labels

def process_file(filename):
    data = []
    with open(os.path.join(ori_path, filename), 'r', encoding='utf-8') as f:
        items = json.load(f)
        for item in items:
            tokens = item['token']
            ner_tags = item['stanford_ner']
            bio_labels = convert_to_bio(tokens, ner_tags)
            data.append({
                "id": item['id'],
                "text": tokens,
                "labels": bio_labels
            })
            entity_types.update([tag for tag in ner_tags if tag != "O"])
    return data

# 处理所有文件
for file in data_files:
    processed_data = process_file(file)
    save_file = os.path.join(save_path, file.replace(".json", ".txt"))
    with open(save_file, 'w', encoding='utf-8') as f:
        for item in processed_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

# 写入labels.txt
bio_labels = ["B-" + t for t in sorted(entity_types)] + ["I-" + t for t in sorted(entity_types)] + ["O"]
with open(os.path.join(save_path, "labels.txt"), 'w', encoding='utf-8') as f:
    for label in bio_labels:
        f.write(label + '\n')

print("预处理完成，数据保存于：", save_path)

