import torch


class Config():
    # 数据集
    train_path = "sentiment_analysis_on_movie_reviews\\train\\train.tsv"
    test_path = "sentiment_analysis_on_movie_reviews\\test\\test.tsv"
    pad_inputs = 0  # padding的值
    # 训练
    batch_size = 12
    epochs = 30
    lr = 0.001
    clip_value = 5
    eval_epoch = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 模型
    output_size = 1
    hidden_dim = 256
    embedding_dim = 300
    n_layers = 2  # LSTM的层数
    n_classes = 5  # 评分的类别数
    dropout = 0.5  # dropout的概率，防止过拟合
    # 测试
    test_batch_size = 4
    model_path = '.\\SentimentRNN.pt'
