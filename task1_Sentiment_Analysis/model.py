from torch import nn


class SentimentRNN(nn.Module):
    """
    用于情感分析的RNN模型。
    """

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob):
        """
        初始化模型，设置层。
        """
        super(SentimentRNN, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        # 嵌入层，将每个单词索引映射到其稠密向量表示。
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # LSTM层，处理输入序列的单词嵌入并生成隐藏状态序列。
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            n_layers,
                            dropout=0.5,
                            batch_first=True)

        # Dropout层，随机将输入的元素置为零以防止过拟合。
        self.dropout = nn.Dropout(p=drop_prob)

        # 线性层和sigmoid层
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sig = nn.Sigmoid()

    def forward(self, x, hidden):
        """
        执行模型的前向传播。
        """
        # 计算输入序列的单词嵌入。
        batch_size = x.size(0)
        embeds = self.embedding(x)

        # 通过LSTM层传递嵌入以获取LSTM输出和更新的隐藏状态。
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        # 对重塑的LSTM输出应用dropout。
        out = self.dropout(lstm_out)

        # 通过全连接层传递输出。
        out = self.fc(out)

        # 应用sigmoid激活函数将输出压缩到0和1之间。
        out = self.sig(out)
        out = out.view(batch_size, -1)

        # 从每个序列中提取最后五个元素
        out = out[:, -5:]
        return out, hidden

    def init_hidden(self, batch_size, device):
        """
        初始化隐藏状态。
        """
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))

        return hidden