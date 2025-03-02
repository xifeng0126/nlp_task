import torch


class Config():
    # 数据集
    train_path = "sentiment_analysis_on_movie_reviews\\train\\train.tsv"
    test_path = "sentiment_analysis_on_movie_reviews\\test\\test.tsv"
    pad_inputs = 0  # padding的值
    # 训练
    batch_size = 64
    epochs = 50
    lr = 0.001
    clip_value = 5
    eval_epoch = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 模型
    output_size = 5
    hidden_dim = 256
    embedding_dim = 100
    n_layers = 2  # LSTM的层数
    n_classes = 5  # 评分的类别数
    dropout = 0.5  # dropout的概率，防止过拟合
    # 测试
    test_batch_size = 500
    model_path = 'task1_Sentiment_Analysis/model.pth'
    
    # 数据处理参数
    val_size = 0.1
    max_seq_len = 200
    
    # K折交叉验证参数
    use_kfold = True
    k_folds = 5
    
    # 早停法参数
    patience = 5
    
    # 预训练词向量参数
    freeze_embeddings = False  # 是否冻结预训练词向量
    
    # 最终模型训练参数
    final_epochs = 10
    
    # 权重衰减参数
    weight_decay = 1e-5
