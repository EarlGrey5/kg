import torch
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

class NerDataset(Dataset):
    def __init__(self, data, args, tokenizer):
        self.data = data
        self.args = args
        self.tokenizer = tokenizer
        self.label2id = args.label2id
        self.max_seq_len = args.max_seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        text = self.data[item]["text"]
        labels = self.data[item]["labels"]
        if len(text) > self.max_seq_len - 2:
            text = text[:self.max_seq_len - 2]
            labels = labels[:self.max_seq_len - 2]
        tmp_input_ids = self.tokenizer.convert_tokens_to_ids(["[CLS]"] + text + ["[SEP]"])
        attention_mask = [1] * len(tmp_input_ids)
        input_ids = tmp_input_ids + [0] * (self.max_seq_len - len(tmp_input_ids))
        attention_mask = attention_mask + [0] * (self.max_seq_len - len(tmp_input_ids))
        labels = [self.label2id[label] for label in labels]
        labels = [0] + labels + [0] + [0] * (self.max_seq_len - len(tmp_input_ids))

        input_ids = torch.tensor(np.array(input_ids))
        attention_mask = torch.tensor(np.array(attention_mask))
        labels = torch.tensor(np.array(labels))

        data = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        return data

class ReDataset(Dataset):
    def __init__(self, file_path, label_map, tokenizer: BertTokenizer, max_length=512):
        self.samples = []
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.max_length = max_length

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                example = json.loads(line)
                self.samples.append(example)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        tokens = item["token"]
        subj_start = item["subj_start"]
        obj_start = item["obj_start"]
        relation = item["relation"]

        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # 主语和宾语实体起始索引映射到WordPiece后的索引
        word_ids = encoding.word_ids()
        try:
            subj_token_index = word_ids.index(subj_start)
            obj_token_index = word_ids.index(obj_start)
        except ValueError:
            subj_token_index = 0
            obj_token_index = 0

        label_id = self.label_map.get(relation, self.label_map["no_relation"])

        # 构造依存图
        head_indices = item["stanford_head"]
        head_indices = head_indices[:len(tokens)]  # 截断
        edge_index = build_edge_index(head_indices)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "subj_pos": torch.tensor(subj_token_index, dtype=torch.long),
            "obj_pos": torch.tensor(obj_token_index, dtype=torch.long),
            "edge_index": edge_index,
            "label": torch.tensor(label_id, dtype=torch.long)
        }

def load_label_map(label_file):
    with open(label_file, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f if line.strip()]
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


def build_edge_index(heads):
    """
    构造 PyG GAT 所需的 edge_index（[2, num_edges]）
    heads 是一个长度为 L 的列表，代表每个 token 的依存父节点索引（1-based）
    返回：torch.LongTensor，形状 [2, num_edges]
    """
    edge_index = []
    for idx, head in enumerate(heads):
        if head == 0:
            continue  # skip root connection
        edge_index.append((head - 1, idx))  # 父→子
        edge_index.append((idx, head - 1))  # 子→父
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # [2, E]
    return edge_index

def create_dataloader(path, tokenizer, label_map, batch_size, max_length, shuffle=True):
    dataset = ReDataset(path, label_map, tokenizer, max_length=max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


def collate_fn(batch):
    # 动态 batch 构造 padding，适配 edge_index 图
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    subj_pos = torch.stack([b["subj_pos"] for b in batch])
    obj_pos = torch.stack([b["obj_pos"] for b in batch])
    labels = torch.stack([b["label"] for b in batch])

    # 合并所有句子的图到一个 batched 图
    edge_indices = []
    base = 0
    max_len = input_ids.size(1)
    for i, b in enumerate(batch):
        ei = b["edge_index"] + base
        edge_indices.append(ei)
        base += max_len  # 每个句子最大长度视为节点数量
    edge_index = torch.cat(edge_indices, dim=1)  # [2, total_edges]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "subj_pos": subj_pos,
        "obj_pos": obj_pos,
        "edge_index": edge_index,
        "labels": labels
    }
