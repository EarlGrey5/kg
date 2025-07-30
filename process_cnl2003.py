import os
import json

base_dir = "data/conll2003"
ori_dir = os.path.join(base_dir, "ori_data")
ner_dir = os.path.join(base_dir, "ner_data")
os.makedirs(ner_dir, exist_ok=True)

split_map = {
    "train.txt": "train",
    "valid.txt": "validation",
    "test.txt": "test"
}


def convert_file(filename, split_name):
    file_path = os.path.join(ori_dir, filename)
    output_path = os.path.join(ner_dir, f"{split_name}.txt")

    dataset = []
    tokens = []
    labels = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    sample_id = f"{split_name.upper()}{str(len(dataset)).zfill(5)}"
                    dataset.append({
                        "id": sample_id,
                        "text": tokens,
                        "labels": labels
                    })
                    tokens, labels = [], []
                continue
            if line.startswith("-DOCSTART-"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            token = parts[0]
            label = parts[-1]
            tokens.append(token)
            labels.append(label)


    if tokens:
        sample_id = f"{split_name.upper()}{str(len(dataset)).zfill(5)}"
        dataset.append({
            "id": sample_id,
            "text": tokens,
            "labels": labels
        })


    with open(output_path, "w", encoding="utf-8") as fout:
        for item in dataset:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"{filename} → {output_path}，共 {len(dataset)} 条样本")



for filename, split in split_map.items():
    convert_file(filename, split)
