from collections import Counter
from string import punctuation
import numpy as np
import pandas as pd
import torch
from nltk import PorterStemmer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import os
import torchtext

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




# 把目标向量转换成one-hot编码
def to_categorical(y, num_classes):
    return np.eye(num_classes, dtype='uint8')[y]




# 填充或截断评论到固定序列长度
def pad_features(reviews_int, seq_length):
    features = np.zeros((len(reviews_int), seq_length), dtype=int)
    for i, row in enumerate(reviews_int):
        try:
            features[i, -len(row):] = np.array(row)[:seq_length]
        except ValueError:
            continue
    return features


def load_glove_embeddings(word_to_idx, embedding_dim=100):
    """
    加载GloVe预训练词向量
    """
    # 下载GloVe词向量（如果尚未下载）
    glove_path = '.vector_cache'
    if not os.path.exists(glove_path):
        os.makedirs(glove_path)
    
    # 使用torchtext加载GloVe词向量
    glove = torchtext.vocab.GloVe(name='6B', dim=embedding_dim, cache=glove_path)
    
    # 初始化嵌入矩阵
    embeddings = torch.zeros((len(word_to_idx) + 1, embedding_dim))
    
    # 填充嵌入矩阵
    for word, idx in word_to_idx.items():
        if word in glove.stoi:
            embeddings[idx] = glove.vectors[glove.stoi[word]]
        else:
            # 对于不在GloVe中的词，使用随机初始化
            embeddings[idx] = torch.randn(embedding_dim)
    
    return embeddings


def load_data(train_path, test_path, batch_size=50, test_batch_size=500, val_size=0.1, max_seq_len=200, embedding_dim=100):
    """
    加载并预处理数据
    """
    # 读取训练数据
    train_df = pd.read_csv(train_path, sep='\t')
    
    # 读取测试数据
    test_df = pd.read_csv(test_path, sep='\t')
    
    # 构建词汇表
    word_counter = Counter()
    for phrase in train_df['Phrase'].values:
        word_counter.update(phrase.lower().split())
    
    # 保留出现频率最高的词
    vocab_size = 10000
    common_words = [word for word, count in word_counter.most_common(vocab_size-1)]
    
    # 创建词到索引的映射
    word_to_idx = {word: i+1 for i, word in enumerate(common_words)}
    
    # 处理训练数据
    X_train = []
    y_train = []
    for phrase, sentiment in zip(train_df['Phrase'].values, train_df['Sentiment'].values):
        # 将短语转换为索引序列
        indices = [word_to_idx.get(word.lower(), 0) for word in phrase.split()]
        # 填充或截断序列
        if len(indices) < max_seq_len:
            indices = indices + [0] * (max_seq_len - len(indices))
        else:
            indices = indices[:max_seq_len]
        X_train.append(indices)
        
        # 将情感标签转换为one-hot编码
        sentiment_one_hot = [0] * 5
        sentiment_one_hot[sentiment] = 1
        y_train.append(sentiment_one_hot)
    
    X_train = torch.tensor(X_train, dtype=torch.long)
    y_train = torch.tensor(y_train, dtype=torch.float)
    
    # 处理测试数据
    X_test = []
    test_ids = []
    test_zero_idx = []
    
    for phrase_id, phrase in zip(test_df['PhraseId'].values, test_df['Phrase'].values):
        # 将短语转换为索引序列
        indices = [word_to_idx.get(word.lower(), 0) for word in phrase.split()]
        # 填充或截断序列
        if len(indices) < max_seq_len:
            indices = indices + [0] * (max_seq_len - len(indices))
        else:
            indices = indices[:max_seq_len]
        X_test.append(indices)
        test_ids.append(phrase_id)
        
        # 记录空短语的ID
        if len(phrase.strip()) == 0:
            test_zero_idx.append(phrase_id)
    
    X_test = torch.tensor(X_test, dtype=torch.long)
    test_ids = torch.tensor(test_ids, dtype=torch.long)
    
    # 创建测试数据加载器
    test_data = TensorDataset(X_test, test_ids)
    test_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False)
    
    # 加载GloVe词向量
    pretrained_embeddings = load_glove_embeddings(word_to_idx, embedding_dim)
    
    return X_train, y_train, test_loader, word_to_idx, test_zero_idx, pretrained_embeddings


def create_kfold_loaders(X, y, n_splits=5, batch_size=50):
    """
    创建K折交叉验证的数据加载器
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_loaders = []
    
    for train_idx, val_idx in kf.split(X):
        # 分割数据
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # 创建数据集
        train_data = TensorDataset(X_train_fold, y_train_fold)
        val_data = TensorDataset(X_val_fold, y_val_fold)
        
        # 创建数据加载器
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        
        fold_loaders.append((train_loader, val_loader))
    
    return fold_loaders