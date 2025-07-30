import torch
import torch.nn as nn
from transformers import BertModel
from torch_geometric.nn import GATConv


class BertGATRelationModel(nn.Module):
    def __init__(self, bert_path, gat_hidden_dim=128, gat_heads=4, dropout=0.1, num_relations=41):
        super(BertGATRelationModel, self).__init__()
        # BERT 模型
        self.bert = BertModel.from_pretrained(bert_path)
        self.bert_hidden_size = self.bert.config.hidden_size

        # GAT 层：输入维度=bert hidden，输出维度=GAT hidden
        self.gat = GATConv(self.bert_hidden_size, gat_hidden_dim, heads=gat_heads, dropout=dropout, concat=True)
        self.gat_output_dim = gat_hidden_dim * gat_heads

        # 分类器输入 = [subj_repr, obj_repr, cls_token, gat_mean]
        self.classifier_input_dim = self.bert_hidden_size * 3 + self.gat_output_dim
        self.classifier = nn.Sequential(
            nn.Linear(self.classifier_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_relations)
        )

    def forward(self, input_ids, attention_mask, subj_pos, obj_pos, edge_index):
        """
        input_ids: [B, L]
        attention_mask: [B, L]
        subj_pos: [B] 每个样本主语token起始位置
        obj_pos: [B] 每个样本宾语token起始位置
        edge_index: [2, E] 图的边
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state       # [B, L, H]
        cls_output = outputs.pooler_output                # [B, H]

        B, L, H = sequence_output.size()

        # Flatten batch to feed into GAT
        flat_sequence = sequence_output.view(-1, H)       # [B*L, H]
        gat_output = self.gat(flat_sequence, edge_index)  # [B*L, H']

        # 还原为 batch 结构，取每个句子的GAT特征平均值
        gat_output = gat_output.view(B, L, -1)            # [B, L, H']
        gat_mean = gat_output.mean(dim=1)                 # [B, H']

        # 获取主/宾首token
        subj_repr = sequence_output[torch.arange(B), subj_pos]  # [B, H]
        obj_repr = sequence_output[torch.arange(B), obj_pos]    # [B, H]

        # 拼接所有特征
        concat = torch.cat([subj_repr, obj_repr, cls_output, gat_mean], dim=1)  # [B, H*3 + H']

        logits = self.classifier(concat)  # [B, num_relations]
        return logits
