# -*- coding: utf-8 -*-
import os
import json
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from transformers import BertTokenizer

from ner_model import BertNer
from data_loader import NerDataset
from config import NerConfig


def evaluate(model, test_loader, id2label, device):
    model.eval()
    preds = []
    trues = []

    with torch.no_grad():
        for step, batch_data in enumerate(test_loader):
            for key, value in batch_data.items():
                batch_data[key] = value.to(device)
            input_ids = batch_data["input_ids"]
            attention_mask = batch_data["attention_mask"]
            labels = batch_data["labels"]

            output = model(input_ids, attention_mask, labels)
            logits = output.logits  # CRF解码后的预测标签序列

            attention_mask = attention_mask.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()

            batch_size = input_ids.size(0)
            for i in range(batch_size):
                length = sum(attention_mask[i])
                logit = logits[i][1:length]  # 去掉CLS
                logit = [id2label[j] for j in logit]
                label = labels[i][1:length]
                label = [id2label[j] for j in label]
                preds.extend(logit)
                trues.extend(label)

    report = classification_report(trues, preds, digits=4)
    return report


def predict_sentence(model, tokenizer, sentence, label2id, id2label, device, max_seq_len=128):
    """对单句预测命名实体"""
    model.eval()

    # 分词 + 编码
    tokens = tokenizer.tokenize(sentence)
    if len(tokens) > max_seq_len - 2:
        tokens = tokens[: max_seq_len - 2]

    input_tokens = ["[CLS]"] + tokens + ["[SEP]"]
    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
    attention_mask = [1] * len(input_ids)
    label_ids = [0] * len(input_ids)  # 占位

    # 转Tensor
    input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
    attention_mask = torch.tensor([attention_mask], dtype=torch.long).to(device)
    label_ids = torch.tensor([label_ids], dtype=torch.long).to(device)

    with torch.no_grad():
        output = model(input_ids, attention_mask, labels=None)
        preds = output.logits[0]

    # 还原预测标签
    pred_labels = [id2label[p] for p in preds[1:-1]]  # 去掉[CLS]和[SEP]
    return list(zip(tokens, pred_labels))


def main(data_name="Re-TACRED"):
    args = NerConfig(data_name)
    tokenizer = BertTokenizer.from_pretrained(args.bert_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载test数据
    with open(os.path.join(args.data_path, "test.txt"), "r", encoding="utf-8") as fp:
        test_data = [json.loads(d) for d in fp.read().split("\n") if d.strip()]

    test_dataset = NerDataset(test_data, args, tokenizer)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.dev_batch_size, num_workers=2)

    # 加载模型
    model_path = os.path.join(args.output_dir, "pytorch_model_ner.bin")
    model = BertNer(args)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # 评估
    report = evaluate(model, test_loader, args.id2label, device)
    print(" Test Evaluation ")
    print(report)
    with open(os.path.join(args.output_dir, "ner_eval_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    # 示例预测
    example_sentence = "Apple is looking at buying U.K. startup for $1 billion. Tim Cook is the CEO of Apple.West Indian all-rounder Phil Simmons took four for 38 on Friday as Leicestershire beat Somerset by an innings and 39 runs"
    pred_result = predict_sentence(model, tokenizer, example_sentence, args.label2id, args.id2label, device)
    print(" Example Prediction ")
    print(pred_result)


if __name__ == "__main__":
    main()
