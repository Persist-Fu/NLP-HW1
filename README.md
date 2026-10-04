# NLP HW1 代码运行说明


## 运行前改路径
数据集文件（NYT、AG News）、GloVe、结果路径都写在脚本里。运行前，需要在代码中改成正确的文件路径，目前代码中路径是笔者在本机运行时的路径。

需要三份数据文件：`nyt.csv`（列名 `text`、`label`）、`ag.csv`（列名 `text`）、以及 100 维 GloVe（`glove.6B.100d.txt`，从 http://nlp.stanford.edu/data/glove.6B.zip 下载）。

| 文件 | 修改位置改 |
|---|---|
| NYT 数据 | `code/utils.py` 里的 `NYT_PATH`（Task1/2/3 共用） |
| GloVe 词向量 | `code/task2/task2-1.py` 里的 `glove_path` |
| AG News 数据 | `code/task2/task2-2.py` 里的 `ag_path` |


各脚本开头的 `sys.path.append(...)` 需要改成实际运行时的 `code` 目录；预测结果和图片的保存路径在各 `task*.py` 里，按需改到本地已有的文件夹。

## 环境

Python 3.12，需安装：`numpy pandas scikit-learn scipy nltk gensim matplotlib transformers torch`

首次运行会自动下载 NLTK 的 `punkt`、`stopwords`。Task3 会从镜像下载 `google-bert/bert-base-uncased`。

## 运行

路径改好后执行：

```bash
python code/task1/task1-1.py
python code/task1/task1-2.py
python code/task2/task2-1.py
python code/task2/task2-2.py
python code/task2/task2-3.py
python code/task3/task3.py
```

| 脚本 | 作用 |
|---|---|
| `task1-1.py` | Binary Bag of Words + LR |
| `task1-2.py` | Word Frequency + LR |
| `task2-1.py` | 预训练 GloVe 平均池化 + LR |
| `task2-2.py` | 在 AG News 上训 Word2Vec + LR |
| `task2-3.py` | 在 NYT 训练集上训 Word2Vec + LR |
| `task3.py` | 微调 BERT（max_len=64，3 个 epoch） |

各图和预测结果 csv 会写到各脚本里设置的保存路径。
