import os
import string
import nltk
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
try:
    nltk.pathsec.ALLOW_PROXIED_FETCH = True
except Exception:
    pass

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
#显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False 
STOPWORDS = set(stopwords.words("english"))
NYT_PATH = r"D:\NLP\HW1\HW-1-dataset\nyt.csv"

#NYT数据集8:1:1划分、分层抽样、task1-3共用
def split_nyt(file_path=NYT_PATH, random_state=42):
    df = pd.read_csv(file_path)
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=random_state, stratify=df["label"])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=random_state, stratify=temp_df["label"])
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    return train_df, val_df, test_df, df

#word_tokenize后小写，去掉停用词、纯标点、不含字母的token
def tokenize_clean(text):
    tokens = [tok.lower() for tok in word_tokenize(str(text))]
    cleaned = []
    for tok in tokens:
        if tok in STOPWORDS:
            continue
        if all(ch in string.punctuation for ch in tok):
            continue
        if not any(ch.isalpha() for ch in tok):
            continue
        cleaned.append(tok)
    return cleaned

def _ensure_dir(save_path):
    folder = os.path.dirname(os.path.abspath(save_path))
    if folder:
        os.makedirs(folder, exist_ok=True)

#绘制混淆矩阵图
def confusion_matrix_fig(y_true, y_pred, save_path):
    labels = sorted(set(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

#task1学习曲线：横轴训练样本量，纵轴Train/Val的Macro-F1
def learning_curve_fig(train_sizes, train_scores, val_scores, save_path):
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(train_sizes, train_scores, marker="o", label="训练集 Macro-F1")
    ax.plot(train_sizes, val_scores, marker="s", label="验证集 Macro-F1")
    ax.set_xlabel("训练样本量")
    ax.set_ylabel("Macro-F1")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

#Word2Vec：横轴epoch，纵轴该轮训练损失（词向量目标，不是分类损失）
def w2v_loss_fig(epochs, losses, save_path):
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(epochs, losses, marker="o")
    ax.set_xlabel("epoch")
    ax.set_ylabel("Word2Vec 训练损失")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
