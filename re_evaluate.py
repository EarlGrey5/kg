import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from config import ReConfig
from data_loader import create_dataloader
from re_model import BertGATRelationModel
from tqdm import tqdm

def evaluate():
    # 加载配置
    config = ReConfig()

    # 创建模型
    model = BertGATRelationModel(
        bert_path=config.bert_path,
        gat_hidden_dim=config.gat_hidden_dim,
        gat_heads=config.gat_heads,
        dropout=config.dropout,
        num_relations=len(config.label2id)
    ).to(config.device)

    # 加载已训练模型权重
    model.load_state_dict(torch.load(config.save_path, map_location=config.device))
    model.eval()

    # 加载测试集
    test_loader = create_dataloader(
        path=config.test_file,
        tokenizer=config.tokenizer,
        label_map=config.label2id,
        batch_size=config.dev_batch_size,
        max_length=config.max_seq_len,
        shuffle=False
    )

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)
            subj_pos = batch["subj_pos"].to(config.device)
            obj_pos = batch["obj_pos"].to(config.device)
            edge_index = batch["edge_index"].to(config.device)
            labels = batch["labels"].to(config.device)

            logits = model(input_ids, attention_mask, subj_pos, obj_pos, edge_index)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    # 输出分类report
    id2label = config.id2label
    label_ids = list(range(len(id2label)))
    target_names = [id2label[i] for i in label_ids]

    report = classification_report(
        all_labels,
        all_preds,
        labels=label_ids,
        target_names=target_names,
        digits=4
    )
    print("\n Evaluation Report ")
    print(report)
    with open(os.path.join(config.output_dir, "re_eval_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)


if __name__ == "__main__":
    evaluate()
