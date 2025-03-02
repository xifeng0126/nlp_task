import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch import nn, optim
from task1_Sentiment_Analysis.model import SentimentRNN
from task1_Sentiment_Analysis.Config import Config
from task1_Sentiment_Analysis.data_process import load_data


def train_loop(model, optimizer, criterion, train_loader, clip_value, device, batch_size=Config.batch_size):
    """
    训练模型
    """
    running_loss = 0
    model.train()

    # 返回LSTM层的初始隐藏状态
    h = model.init_hidden(batch_size, device)

    for seq, targets in train_loader:
        # 将数据移动到设备
        seq = seq.to(device)
        targets = targets.to(device)

        # 将隐藏状态元组h的元素转换为与输入数据相同设备的张量
        h = tuple([each.data for each in h])

        # 进行模型的前向传播
        out, h = model.forward(seq, h)

        # 计算预测输出与目标值之间的损失
        loss = criterion(out, targets.float())
        running_loss += loss.item() * seq.shape[0]

        # 重置模型参数的梯度
        optimizer.zero_grad()
        # 计算损失相对于模型参数的梯度
        loss.backward()
        if clip_value:
            # 防止梯度爆炸
            nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        # 更新模型参数
        optimizer.step()

    running_loss /= len(train_loader.sampler)
    return running_loss

# 将预测结果（概率）转换为one-hot编码
def get_prediction(t):
    max_indices = torch.argmax(t, dim=1)
    new = torch.zeros_like(t)
    new[torch.arange(t.shape[0]), max_indices] = 1
    return new


def eval_loop(model, criterion, eval_loader, device, batch_size=Config.batch_size, ignore_index=None):
    """
    评估模型
    """

    # 返回LSTM层的初始隐藏状态
    val_h = model.init_hidden(batch_size, device)
    val_loss = 0
    model.eval()
    accuracy = []
    for seq, targets in eval_loader:
        # 将隐藏状态元组val_h的元素转换为与输入数据相同设备的张量
        val_h = tuple([each.data for each in val_h])

        # 将数据移动到设备
        seq = seq.to(device)
        targets = targets.to(device)

        # 进行模型的前向传播
        out, val_h = model(seq, val_h)

        # 计算损失
        loss = criterion(out, targets.float())
        val_loss += loss.item() * seq.shape[0]

        # 转换模型输出
        predicted = get_prediction(out).flatten().cpu().numpy()
        labels = targets.view(-1).cpu().numpy()

        # 计算预测值与目标值之间的准确率
        accuracy.append(accuracy_score(labels, predicted))

    acc = sum(accuracy) / len(accuracy)
    val_loss /= len(eval_loader.sampler)
    return {'accuracy': acc,
            'loss': val_loss}


def train_model(model, optimizer, criterion, train_loader, valid_loader,
          eval_every, num_epochs, clip_value,
          ignore_index=None,
          device=Config.device,
          valid_loss_min=np.inf):
    for e in range(num_epochs):
        # 训练一个epoch
        train_loss = train_loop(model, optimizer, criterion, train_loader, clip_value, device)

        if (e + 1) % eval_every == 0:

            # 在验证集上评估
            metrics = eval_loop(model, criterion, valid_loader, device)

            # 显示进度
            print_string = f'Epoch: {e + 1} '
            print_string += f'TrainLoss: {train_loss:.5f} '
            print_string += f'ValidLoss: {metrics["loss"]:.5f} '
            print_string += f'ACC: {metrics["accuracy"]:.5f} '
            print(print_string)

            # 更优时保存模型
            if metrics["loss"] <= valid_loss_min:
                torch.save(model.state_dict(), Config.model_path)
                valid_loss_min = metrics["loss"]

def train():
    # 加载数据
    train_loader, valid_loader, _, encode_voc, _ = load_data(Config.train_path, Config.test_path)

    # 初始化模型
    vocab_size = len(encode_voc) + 1
    model = SentimentRNN(vocab_size, Config.output_size, Config.embedding_dim, Config.hidden_dim, Config.n_layers, Config.dropout)
    model = model.to(Config.device)

    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=Config.lr)
    criterion = nn.BCELoss()

    # 训练模型
    train_model(model, optimizer, criterion, train_loader, valid_loader, Config.eval_epoch, Config.epochs, Config.clip_value)

if __name__ == '__main__':
    train()
