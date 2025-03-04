import torch

class Config:
    # 设备配置
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 数据处理相关
    MAX_LENGTH = 10
    SOS_TOKEN = 0
    EOS_TOKEN = 1
    
    # 模型相关
    HIDDEN_SIZE = 128
    DROPOUT = 0.1
    
    # 训练相关
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    N_EPOCHS = 60
    PRINT_EVERY = 5
    PLOT_EVERY = 5
    
    # 英语前缀过滤
    ENG_PREFIXES = (
        "i am ", "i m ",
        "he is", "he s ",
        "she is", "she s ",
        "you are", "you re ",
        "we are", "we re ",
        "they are", "they re "
    ) 