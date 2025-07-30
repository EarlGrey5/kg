import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import BertModel, BertConfig


class ModelOutput:
    def __init__(self, logits, labels=None, loss=None):
        self.logits = logits
        self.labels = labels
        self.loss = loss

class BertNer(nn.Module):
    def __init__(self, args):
        super(BertNer, self).__init__()
        self.bert = BertModel.from_pretrained(args.bert_dir)
        self.bert_config = BertConfig.from_pretrained(args.bert_dir)
        self.hidden_size = self.bert_config.hidden_size

        # BiLSTM层
        self.lstm_hidden = 128
        self.num_layers = 2
        self.bilstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.lstm_hidden,
            num_layers=self.num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.1
        )

        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(self.lstm_hidden * 2, args.num_labels)
        self.crf = CRF(args.num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        # BERT encoder
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_output.last_hidden_state  # [batch, seq_len, hidden]
        sequence_output, _ = self.bilstm(sequence_output)  # LSTM output

        sequence_output = self.dropout(sequence_output)
        emissions = self.linear(sequence_output)  # [batch, seq_len, num_labels]

        # Predict path
        logits = self.crf.decode(emissions, mask=attention_mask.bool())

        # Loss calculation
        loss = None
        if labels is not None:
            loss = -self.crf(emissions, labels, mask=attention_mask.bool(), reduction='mean')

        return ModelOutput(logits=logits, labels=labels, loss=loss)
