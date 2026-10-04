import sys
sys.path.append(r"D:\NLP\HW1\code")
from utils import split_nyt, tokenize_clean, confusion_matrix_fig, learning_curve_fig, w2v_loss_fig
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
#数据集划分
train_df, val_df, test_df, df = split_nyt()

#分词+小写+去停用词+去纯标点+去数字
train_df["tokens"] = train_df["text"].apply(tokenize_clean)
val_df["tokens"] = val_df["text"].apply(tokenize_clean)
test_df["tokens"] = test_df["text"].apply(tokenize_clean)

#用AG News的text训练100维Word2Vec
ag_path = r"D:\NLP\HW1\HW-1-dataset\ag.csv"
ag_df = pd.read_csv(ag_path)
ag_sents = [tokenize_clean(text) for text in ag_df["text"]]

class EpochLoss(CallbackAny2Vec):
    def __init__(self):
        self.losses = []
        self.prev = 0.0
    def on_epoch_end(self, model):
        cur = model.get_latest_training_loss()
        self.losses.append(cur - self.prev)
        self.prev = cur

loss_cb = EpochLoss()
w2v_model = Word2Vec(
    sentences=ag_sents,
    vector_size=100,
    window=5,
    min_count=5,
    workers=4,
    seed=42,
    epochs=5,
    compute_loss=True,
    callbacks=[loss_cb],
)
w2v = w2v_model.wv  #训好的词向量字典 单词-100维向量
print("Word2Vec epoch 损失：", loss_cb.losses)
w2v_loss_fig(list(range(1, len(loss_cb.losses) + 1)),loss_cb.losses,r"D:\NLP\HW1\results\task2\task2_w2v_ag_loss.png",)

#对有效词求平均Word2Vec得到文档向量
def w2v_vec(tokens, w2v, dim=100):
    vecs = [w2v[w] for w in tokens if w in w2v]
    if len(vecs) == 0:
        return np.zeros(dim, dtype=np.float32)
    return np.mean(vecs, axis=0)
X_train = np.array([w2v_vec(t, w2v) for t in train_df["tokens"]])
X_val = np.array([w2v_vec(t, w2v) for t in val_df["tokens"]])
X_test = np.array([w2v_vec(t, w2v) for t in test_df["tokens"]])

#训练Logistic Regression：训练集拟合，用验证集调正则强度C
y_train = train_df["label"].tolist()
y_val = val_df["label"].tolist()

C_list = [0.01, 0.1, 1.0, 10.0, 100.0]
best_C = None
best_f1 = -1
best_clf = None


for C in C_list:
    clf = LogisticRegression(C=C, max_iter=1000, solver="saga", random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average="macro")
    print("C =", C, "  Val Accuracy =", acc, "  Val Macro-F1 =", f1)
    if f1 > best_f1:
        best_f1 = f1
        best_C = C
        best_clf = clf

print("\n验证集上最优 C：", best_C)
print("对应 Val Macro-F1：", best_f1)

#在NYT Test上评价
y_test = test_df["label"].tolist()
y_test_pred = best_clf.predict(X_test)
test_acc = accuracy_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred, average="macro")

print("Accuracy ：", test_acc)
print("Macro-F1：", test_f1)

#保留测试集结果分析badcase
pred_path = r"D:\NLP\HW1\results\task2\task2_w2v_ag_test_preds.csv"
pd.DataFrame({
    "text": test_df["text"].tolist(),
    "y_true": y_test,
    "y_pred": y_test_pred,
}).to_csv(pred_path, index=False, encoding="utf-8-sig")
#保存混淆矩阵图
confusion_matrix_fig(y_test, y_test_pred, r"D:\NLP\HW1\results\task2\task2_w2v_ag_confusion.png")

#学习曲线：固定最优C，按训练集规模变化画Train/Val Macro-F1（分类，不是Word2Vec损失）
ratios = [0.2, 0.4, 0.6, 0.8, 1.0]
train_sizes = []
train_f1s = []
val_f1s = []
y_train_arr = np.array(y_train)
n_train = len(y_train_arr)
for ratio in ratios:
    if ratio >= 1.0:
        idx = np.arange(n_train)
    else:
        idx, _ = train_test_split(np.arange(n_train), train_size=ratio, random_state=42, stratify=y_train_arr)
    X_sub = X_train[idx]
    y_sub = y_train_arr[idx]
    clf = LogisticRegression(C=best_C, max_iter=1000, solver="saga", random_state=42)
    clf.fit(X_sub, y_sub)
    train_f1 = f1_score(y_sub, clf.predict(X_sub), average="macro")
    val_f1 = f1_score(y_val, clf.predict(X_val), average="macro")
    train_sizes.append(len(idx))
    train_f1s.append(train_f1)
    val_f1s.append(val_f1)
    print("训练样本数 =", len(idx), "  Train Macro-F1 =", train_f1, "  Val Macro-F1 =", val_f1)

lc_path = r"D:\NLP\HW1\results\task2\task2_w2v_ag_learning_curve.png"
learning_curve_fig(train_sizes, train_f1s, val_f1s, lc_path)



