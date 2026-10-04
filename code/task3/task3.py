import sys
import os
sys.path.append(r"D:\NLP\HW1\code")
from utils import split_nyt, confusion_matrix_fig
#使用国内镜像下载huggingface的模型
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from transformers import BertTokenizer, BertForSequenceClassification, set_seed
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
#数据集划分
train_df, val_df, test_df, df = split_nyt()

#标签编号双映射
label_list = sorted(train_df["label"].unique())
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for label, i in label2id.items()}

model_name = "google-bert/bert-base-uncased"
set_seed(42)
#加载和该 BERT 配套的词表和切分规则
tokenizer = BertTokenizer.from_pretrained(model_name)
#下载和加载含预训练参数BERT-base-uncased，接3类分类头
model = BertForSequenceClassification.from_pretrained(model_name,num_labels=len(label_list),id2label=id2label,label2id=label2id,)
print("\n-----预训练 BERT-----")
print("模型：", model_name)
print("类别：", label2id)
print("参数量：", sum(p.numel() for p in model.parameters()))

#BERT Tokenization
MAX_LEN = 64
def encode_texts(texts):
    return tokenizer(list(texts),max_length=MAX_LEN,truncation=True,padding="max_length",return_tensors="pt",)
train_enc = encode_texts(train_df["text"])
val_enc = encode_texts(val_df["text"])
test_enc = encode_texts(test_df["text"])
print("\n-----BERT Tokenization-----")
print("训练集input_ids形状：", tuple(train_enc["input_ids"].shape))
print("验证集input_ids形状：", tuple(val_enc["input_ids"].shape))
print("测试集input_ids形状：", tuple(test_enc["input_ids"].shape))


#Fine-tuning：只在NYT训练集上训3个epoch；验证集仅作每个 epoch 的观察
#NYT数据集类
class NYTDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
#标签转为数字
y_train = train_df["label"].map(label2id).tolist()
y_val = val_df["label"].map(label2id).tolist()
y_test = test_df["label"].map(label2id).tolist()
#编码后的文本分装成数据集
train_ds = NYTDataset(train_enc, y_train)
val_ds = NYTDataset(val_enc, y_val)
test_ds = NYTDataset(test_enc, y_test)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }

train_args = {
    "output_dir": r"D:\NLP\HW1\results\task3\bert_runs",
    "num_train_epochs": 3,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 16,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "logging_strategy": "epoch",
    "save_strategy": "no",
    "seed": 42,
    "report_to": [],
}
os.makedirs(train_args["output_dir"], exist_ok=True)
try:
    training_args = TrainingArguments(eval_strategy="epoch", **train_args)
except TypeError:
    training_args = TrainingArguments(evaluation_strategy="epoch", **train_args)

trainer = Trainer(model=model,args=training_args,train_dataset=train_ds,eval_dataset=val_ds,compute_metrics=compute_metrics,)

print("\n-----3 epoch Fine-tuning-----")
trainer.train()

train_epochs, train_losses = [], []
val_epochs, val_accs, val_f1s = [], [], []
for rec in trainer.state.log_history:
    if "loss" in rec and "eval_loss" not in rec:
        train_epochs.append(rec["epoch"])
        train_losses.append(rec["loss"])
    if "eval_accuracy" in rec:
        val_epochs.append(rec["epoch"])
        val_accs.append(rec["eval_accuracy"])
        val_f1s.append(rec["eval_macro_f1"])
print("Train loss：", list(zip(train_epochs, train_losses)))
print("Val Acc / Macro-F1：", list(zip(val_epochs, val_accs, val_f1s)))

#训练曲线图
save_path = r"D:\NLP\HW1\results\task3\task3_bert_train_curve.png"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
axes[0].plot(train_epochs, train_losses, marker="o")
axes[0].set_xlabel("epoch")
axes[0].set_ylabel("训练损失")
axes[0].grid(True, linestyle="--", alpha=0.4)
axes[1].plot(val_epochs, val_accs, marker="o", label="验证集 Accuracy")
axes[1].plot(val_epochs, val_f1s, marker="s", label="验证集 Macro-F1")
axes[1].set_xlabel("epoch")
axes[1].set_ylabel("指标")
axes[1].legend()
axes[1].grid(True, linestyle="--", alpha=0.4)
fig.tight_layout()
fig.savefig(save_path, dpi=150)
plt.close(fig)

print("\n-----最后一个epoch之后的验证集-----")
val_metrics = trainer.evaluate(val_ds)
print(val_metrics)

print("\n-----BERT NYT TEST-----")
test_out = trainer.predict(test_ds)
test_preds = np.argmax(test_out.predictions, axis=-1)
test_acc = accuracy_score(y_test, test_preds)
test_f1 = f1_score(y_test, test_preds, average="macro")
print("Accuracy ：", test_acc)
print("Macro-F1：", test_f1)

#保留测试集结果分析badcase
import pandas as pd
y_true_name = [id2label[int(i)] for i in y_test]
y_pred_name = [id2label[int(i)] for i in test_preds]
pred_path = r"D:\NLP\HW1\results\task3\task3_bert_test_preds.csv"
os.makedirs(os.path.dirname(pred_path), exist_ok=True)
pd.DataFrame({
    "text": test_df["text"].tolist(),
    "y_true": y_true_name,
    "y_pred": y_pred_name,
}).to_csv(pred_path, index=False, encoding="utf-8-sig")
#保存混淆矩阵图
confusion_matrix_fig(y_true_name, y_pred_name, r"D:\NLP\HW1\results\task3\task3_bert_confusion.png")
