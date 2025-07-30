import json
import os
from tqdm import tqdm

def process_data(input_file, output_txt_file, output_label_file):
    os.makedirs(os.path.dirname(output_txt_file), exist_ok=True)

    relations_set = set()
    simplified_data = []

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in tqdm(data, desc="Processing"):
        # 选取需要保留的字段
        simplified_item = {
            "id": item["id"],
            "relation": item["relation"],
            "token": item["token"],
            "subj_start": item["subj_start"],
            "subj_end": item["subj_end"],
            "obj_start": item["obj_start"],
            "obj_end": item["obj_end"],
            "subj_type": item["subj_type"],
            "obj_type": item["obj_type"],
            "stanford_head": item["stanford_head"],
            "stanford_deprel": item["stanford_deprel"]
        }
        simplified_data.append(simplified_item)
        relations_set.add(item["relation"])

    # 写入 txt 文件，每行一个 JSON
    with open(output_txt_file, "w", encoding="utf-8") as out_f:
        for item in simplified_data:
            out_f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 写入 labels.txt
    with open(output_label_file, "w", encoding="utf-8") as label_f:
        for rel in sorted(relations_set):
            label_f.write(rel + "\n")

    print(f"Processed {len(simplified_data)} examples.")
    print(f"Extracted {len(relations_set)} unique relation labels.")

if __name__ == "__main__":
    process_data(
        input_file="./data/Re-TACRED/ori_data/train.json",
        output_txt_file="./data/Re-TACRED/re_data/train.txt",
        output_label_file="./data/Re-TACRED/re_data/labels.txt"
    )

