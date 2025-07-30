import os
import torch
from transformers import BertTokenizerFast

class CommonConfig:
    bert_dir = "./model_hub/bert-base-NER"
    output_dir = "./checkpoint/"
    data_dir = "./data/"


class NerConfig:
    def __init__(self, data_name):
        cf = CommonConfig()
        self.bert_dir = cf.bert_dir
        self.output_dir = os.path.join(cf.output_dir, data_name)
        self.data_dir = cf.data_dir
        self.data_path = os.path.join(self.data_dir, data_name, "ner_data")

        # 创建保存模型的目录
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)


        label_path = os.path.join(self.data_path, "labels.txt")
        with open(label_path, "r", encoding="utf-8-sig") as fp:
            lines = fp.readlines()
            self.labels = [line.strip() for line in lines if line.strip()]

        print("原始标签标签:", self.labels)


        self.bio_labels = ["O"]
        for label in self.labels:
            self.bio_labels.append(f"B-{label}")
            self.bio_labels.append(f"I-{label}")

        print("BIO标签列表:", self.bio_labels)


        self.num_labels = len(self.bio_labels)
        self.label2id = {label: i for i, label in enumerate(self.bio_labels)}
        self.id2label = {i: label for i, label in enumerate(self.bio_labels)}

        print("label2id 映射:", self.label2id)


        self.max_seq_len = 128
        self.epochs = 5
        self.train_batch_size = 12
        self.dev_batch_size = 12
        self.bert_learning_rate = 3e-5
        self.crf_learning_rate = 3e-3
        self.adam_epsilon = 1e-8
        self.weight_decay = 0.01
        self.warmup_proportion = 0.01
        self.save_step = 500


class ReConfig:
    def __init__(self):

        self.bert_path = "./model_hub/bert-base-NER"
        self.label_path = "./data/Re-TACRED/re_data/labels.txt"
        self.train_file = "./data/Re-TACRED/re_data/train.txt"
        self.dev_file = "./data/Re-TACRED/re_data/dev.txt"
        self.test_file = "./data/Re-TACRED/re_data/test.txt"
        self.save_path = "./checkpoint/relation_Re-TACRED/relation_model.pt"
        self.output_dir = "./checkpoint/relation_Re-TACRED/"

        self.max_seq_len = 128
        self.epochs = 5
        self.train_batch_size = 16
        self.dev_batch_size = 32
        self.bert_learning_rate = 2e-5

        self.adam_epsilon = 1e-8
        self.weight_decay = 0.01
        self.warmup_proportion = 0.1
        self.save_step = 500

        self.gat_hidden_dim = 128
        self.gat_heads = 4
        self.dropout = 0.1

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.label2id, self.id2label = self.load_labels()

        self.tokenizer = BertTokenizerFast.from_pretrained(self.bert_path)

    def load_labels(self):
        label2id = {}
        id2label = {}
        with open(self.label_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                label = line.strip()
                label2id[label] = idx
                id2label[idx] = label
        return label2id, id2label
