import sys
sys.path.append(r"D:\NLP\HW1\code")
from utils import split_nyt, tokenize_clean, confusion_matrix_fig, learning_curve_fig
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
#数据集划分
train_df, val_df, test_df, df = split_nyt()
print("\n比例：")
print("Training   :", len(train_df) / len(df))
print("Validation :", len(val_df) / len(df))
print("Test       :", len(test_df) / len(df))

#分词+小写+去停用词+去纯标点+去数字
train_df["tokens"] = train_df["text"].apply(tokenize_clean)
val_df["tokens"] = val_df["text"].apply(tokenize_clean)
test_df["tokens"] = test_df["text"].apply(tokenize_clean)

#用训练集建词表
vocab = set()
for tokens in train_df["tokens"]:
    vocab.update(tokens)
vocab = sorted(vocab)#集合变list
word_idx = {word: i for i, word in enumerate(vocab)}
print("词表大小 |V|：", len(vocab))

#将一篇文档变为向量
def binary_bow(tokens, word_idx):
    vector = [0] * len(word_idx)
    for word in tokens:
        if word in word_idx:
            vector[word_idx[word]] = 1
    return vector
#词转为向量 每行是一个向量
X_train = train_df["tokens"].apply(lambda x: binary_bow(x, word_idx)).tolist()
X_val = val_df["tokens"].apply(lambda x: binary_bow(x, word_idx)).tolist()
X_test = test_df["tokens"].apply(lambda x: binary_bow(x, word_idx)).tolist()

#训练Logistic Regression：训练集拟合，用验证集调正则强度C
y_train = train_df["label"].tolist()
y_val = val_df["label"].tolist()
#词向量矩阵转稀疏矩阵
X_train_sp = csr_matrix(np.array(X_train, dtype=np.int8))
X_val_sp = csr_matrix(np.array(X_val, dtype=np.int8))

C_list = [0.01, 0.1, 1.0, 10.0, 100.0]
best_C = None
best_f1 = -1
best_clf = None

#验证集调参
for C in C_list:
    clf = LogisticRegression(C=C, max_iter=1000, solver="saga", random_state=42)
    clf.fit(X_train_sp, y_train)
    y_pred = clf.predict(X_val_sp)
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
X_test_sp = csr_matrix(np.array(X_test, dtype=np.int8))
y_test_pred = best_clf.predict(X_test_sp)
test_acc = accuracy_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred, average="macro")

print("\n---测试集指标---")
print("Accuracy ：", test_acc)
print("Macro-F1：", test_f1)

#保留测试集结果分析badcase
pred_path = r"D:\NLP\HW1\results\task1\task1_binary_test_preds.csv"
pd.DataFrame({
    "text": test_df["text"].tolist(),
    "y_true": y_test,
    "y_pred": y_test_pred,
}).to_csv(pred_path, index=False, encoding="utf-8-sig")
#保存混淆矩阵图
confusion_matrix_fig(y_test,y_test_pred,r"D:\NLP\HW1\results\task1\task1_binary_confusion.png")

#学习曲线：固定最优C，按训练集规模变化画Train/Val Macro-F1
from sklearn.model_selection import train_test_split
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
        idx, _ = train_test_split(np.arange(n_train),train_size=ratio,random_state=42,stratify=y_train_arr,)
    X_sub = X_train_sp[idx]
    y_sub = y_train_arr[idx]
    clf = LogisticRegression(C=best_C, max_iter=1000, solver="saga", random_state=42)
    clf.fit(X_sub, y_sub)
    train_f1 = f1_score(y_sub, clf.predict(X_sub), average="macro")
    val_f1 = f1_score(y_val, clf.predict(X_val_sp), average="macro")
    train_sizes.append(len(idx))
    train_f1s.append(train_f1)
    val_f1s.append(val_f1)
    print("训练样本数 =", len(idx), "  Train Macro-F1 =", train_f1, "  Val Macro-F1 =", val_f1)

lc_path = r"D:\NLP\HW1\results\task1\task1_binary_learning_curve.png"
learning_curve_fig(train_sizes,train_f1s,val_f1s,lc_path)



