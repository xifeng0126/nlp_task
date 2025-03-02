from collections import Counter
from string import punctuation
import numpy as np
import pandas as pd
import torch
from nltk import PorterStemmer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from task1_Sentiment_Analysis.Config import Config


def pre_process_data(data):
    reviews = []
    stop_words = set(stopwords.words('english'))  # 停用词
    ps = PorterStemmer()  # 词干提取
    for p in tqdm(data['Phrase']):
        if not isinstance(p, str):
            # 如果不是字符串，将其转换成字符串
            print("Not a string: ", p)
            p = str(p)
            reviews.append(p)
            continue
        p = p.lower()
        # 去掉标点符号和空字符
        p = ''.join([c for c in p if c not in punctuation])
        reviews_split = p.split()
        reviews_wo_stopwords = [word for word in reviews_split if not word in stop_words]
        reviews_stemm = [ps.stem(word) for word in reviews_wo_stopwords]  # 词干提取, 例如running -> run
        p = ' '.join(reviews_stemm)
        reviews.append(p)
    return reviews


# 构建词典
def encode_words(data_pp):
    words = []
    for p in data_pp:
        words.extend(p.split())  # 将每个句子分割成单词
    counter = Counter(words)
    vocab = sorted(counter, key=counter.get, reverse=True)  # 按照单词出现的次数排序
    vocab_to_int = {word: i for i, word in enumerate(vocab, 1)}  # 从1开始编号
    return vocab_to_int


def encode_data(data, vocab_to_int):
    reviews_int = []
    for p in data:
        reviews_int.append([vocab_to_int[word] for word in p.split()])  # 将单词转换成数字
    return reviews_int


# encode_voc = encode_words(train_data_pp + test_data_pp)
# train_data_int = encode_data(train_data_pp, encode_voc)
# test_data_int = encode_data(test_data_pp, encode_voc)
# print("Example of encoded train data: ", train_data_int[0])
# print("Example of encoded test data: ", test_data_int[0])


# 把目标向量转换成one-hot编码
def to_categorical(y, num_classes):
    return np.eye(num_classes, dtype='uint8')[y]


# y_target = to_categorical(train_data['Sentiment'], Config.n_classes)  # 将评分转换成one-hot编码
#
# train_reviews_lens = Counter([len(x) for x in train_data_int])
#
# # 从训练集中删除长度为0的评论
# non_zero_idx = [ii for ii, review in enumerate(train_data_int) if len(review) != 0]
# train_data_int = [train_data_int[ii] for ii in non_zero_idx]
# y_target = np.array([y_target[ii] for ii in non_zero_idx])


# 填充或截断评论到固定序列长度
def pad_features(reviews_int, seq_length):
    features = np.zeros((len(reviews_int), seq_length), dtype=int)
    for i, row in enumerate(reviews_int):
        try:
            features[i, -len(row):] = np.array(row)[:seq_length]
        except ValueError:
            continue
    return features

def load_data(train_path, test_path):
    # 加载数据并返回处理后的数据集
    train_data = pd.read_csv(train_path, sep='\t')
    test_data = pd.read_csv(test_path, sep='\t')

    train_data_pp = pre_process_data(train_data)
    test_data_pp = pre_process_data(test_data)

    encode_voc = encode_words(train_data_pp + test_data_pp)
    train_data_int = encode_data(train_data_pp, encode_voc)
    test_data_int = encode_data(test_data_pp, encode_voc)

    y_target = to_categorical(train_data['Sentiment'], Config.n_classes)

    # 记录测试集中长度为0的评论
    test_zero_idx = [test_data.iloc[ii]['PhraseId'] for ii, review in enumerate(test_data_int) if len(review) == 0]

    # 删除长度为0的评论
    non_zero_idx = [ii for ii, review in enumerate(train_data_int) if len(review) != 0]
    train_data_int = [train_data_int[ii] for ii in non_zero_idx]
    y_target = np.array([y_target[ii] for ii in non_zero_idx])

    train_features = pad_features(train_data_int, max(Counter([len(x) for x in train_data_int])))
    X_test = pad_features(test_data_int, max(Counter([len(x) for x in train_data_int])))

    # 分割数据集
    X_train, X_val, y_train, y_val = train_test_split(train_features, y_target, test_size=0.2)

    # 调整数据集大小
    train_size = X_train.shape[0] - X_train.shape[0] % Config.batch_size
    val_size = X_val.shape[0] - X_val.shape[0] % Config.batch_size
    X_train = X_train[:train_size]
    X_val = X_val[:val_size]
    y_train = y_train[:train_size]
    y_val = y_val[:val_size]

    # 提取测试数据的唯一标识符
    ids_test = np.array([t['PhraseId'] for ii, t in test_data.iterrows()])

    # 创建 DataLoader
    train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    valid_data = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_data = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(ids_test))

    train_loader = DataLoader(train_data, shuffle=True, batch_size=Config.batch_size)
    valid_loader = DataLoader(valid_data, shuffle=True, batch_size=Config.batch_size)
    test_loader = DataLoader(test_data, batch_size=Config.test_batch_size)

    return train_loader, valid_loader, test_loader, encode_voc, test_zero_idx