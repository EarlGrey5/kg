import os
import torch
from torch import nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from config import ReConfig
from data_loader import create_dataloader
from re_model import BertGATRelationModel

def train():
    config = ReConfig()
    device = config.device

    # 模型
    model = BertGATRelationModel(
        bert_path=config.bert_path,
        gat_hidden_dim=config.gat_hidden_dim,
        gat_heads=config.gat_heads,
        dropout=config.dropout,
        num_relations=len(config.label2id)
    ).to(device)

    if os.path.exists(config.save_path):
        print(f"从断点加载模型: {config.save_path}")
        model.load_state_dict(torch.load(config.save_path, map_location=config.device))

    # 数据
    train_loader = create_dataloader(
        config.train_file,
        config.tokenizer,
        config.label2id,
        config.train_batch_size,
        config.max_seq_len,
        shuffle=True
    )

    # 优化器
    optimizer = AdamW(model.parameters(), lr=config.bert_learning_rate,
                      eps=config.adam_epsilon, weight_decay=config.weight_decay)

    total_steps = len(train_loader) * config.epochs
    warmup_steps = int(total_steps * config.warmup_proportion)

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)

    # 损失函数
    loss_fn = nn.CrossEntropyLoss()

    best_loss = float("inf")

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        print(f"\n Epoch {epoch+1}/{config.epochs}")

        for step, batch in enumerate(tqdm(train_loader, desc="Training")):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            subj_pos = batch["subj_pos"].to(device)
            obj_pos = batch["obj_pos"].to(device)
            edge_index = batch["edge_index"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                subj_pos=subj_pos,
                obj_pos=obj_pos,
                edge_index=edge_index
            )

            loss = loss_fn(outputs, labels)
            loss.backward()

            total_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if config.save_step and step % config.save_step == 0 and step != 0:
                print(f"Step {step}: Current loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} finished, avg loss: {avg_loss:.4f}")

        # 保存模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), config.save_path)
            print(f"Best model saved to {config.save_path}")

    print(" Training complete.")

if __name__ == "__main__":
    train()
