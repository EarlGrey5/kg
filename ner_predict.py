import os
import json
import torch
from transformers import BertTokenizer
from ner_model import BertNer
from config import NerConfig
from data_loader import NerDataset
import re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = NerConfig(data_name="Re-TACRED")
tokenizer = BertTokenizer.from_pretrained(config.bert_dir)

# 初始化模型
model = BertNer(config)
model_path = os.path.join(config.output_dir, "pytorch_model_ner.bin")
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

id2label = config.id2label

pending_path = os.path.join(config.data_path, "pending.txt")
with open(pending_path, "r", encoding="utf-8") as f:
    text = f.read()


sentences = re.split(r'[。.\n]', text)
sentences = [s.strip() for s in sentences if s.strip()]

# 构造输入数据
all_results = {}

for sentence in sentences:
    words = sentence.split()
    if len(words) == 0:
        continue

    sample = [{
        "text": words,
        "labels": ["O"] * len(words)
    }]

    dataset = NerDataset(sample, config, tokenizer)
    batch = dataset[0]

    input_ids = batch["input_ids"].unsqueeze(0).to(device)
    attention_mask = batch["attention_mask"].unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        pred_ids = output.logits[0]  # [seq_len]

    pred_labels = [id2label[i] for i in pred_ids]

    # 对预测结果进行解码为实体
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    words = sample[0]["text"]
    pred_labels = pred_labels[1:len(words)+1]

    entities = {}
    current_entity = ""
    current_type = ""

    for word, label in zip(words, pred_labels):
        if label.startswith("B-"):
            if current_entity and current_type:
                entities.setdefault(current_type, []).append(current_entity)
            current_type = label[2:]
            current_entity = word
        elif label.startswith("I-") and current_type == label[2:]:
            current_entity += " " + word
        else:
            if current_entity and current_type:
                entities.setdefault(current_type, []).append(current_entity)
            current_entity = ""
            current_type = ""

    if current_entity and current_type:
        entities.setdefault(current_type, []).append(current_entity)

    for etype, items in entities.items():
        all_results.setdefault(etype, []).extend(items)

# 去重输出
print("识别结果：")
for etype, ents in all_results.items():
    unique_ents = list(set(ents))
    print(f"{etype}：{', '.join(unique_ents)}")
