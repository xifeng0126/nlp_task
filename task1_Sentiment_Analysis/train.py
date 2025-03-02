import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch import nn, optim
from task1_Sentiment_Analysis.model import SentimentRNN
from task1_Sentiment_Analysis.Config import Config
from task1_Sentiment_Analysis.data_process import load_data, create_kfold_loaders


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

        # 创建新的隐藏状态，分离之前的计算历史
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

    running_loss /= len(train_loader.dataset)
    return running_loss

# 将预测结果（概率）转换为one-hot编码
def get_prediction(t):
    max_indices = torch.argmax(t, dim=1)
    new = torch.zeros_like(t)
    new[torch.arange(t.shape[0]), max_indices] = 1
    return new


def eval_loop(model, criterion, eval_loader, device, batch_size=Config.batch_size):
    """
    评估模型
    """
    # 返回LSTM层的初始隐藏状态
    val_h = model.init_hidden(batch_size, device)
    val_loss = 0
    model.eval()
    accuracy = []
    
    with torch.no_grad():
        for seq, targets in eval_loader:
            # 创建新的隐藏状态
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
    val_loss /= len(eval_loader.dataset)
    return {'accuracy': acc, 'loss': val_loss}


def train_model_with_early_stopping(model, optimizer, criterion, train_loader, valid_loader,
                                   patience, num_epochs, clip_value,
                                   device=Config.device):
    """
    使用早停法训练模型
    """
    # 初始化变量
    valid_loss_min = np.inf
    best_accuracy = 0
    counter = 0
    
    # 存储训练历史
    history = {
        'train_loss': [],
        'valid_loss': [],
        'valid_accuracy': []
    }
    
    for e in range(num_epochs):
        # 训练一个epoch
        train_loss = train_loop(model, optimizer, criterion, train_loader, clip_value, device)
        
        # 在验证集上评估
        metrics = eval_loop(model, criterion, valid_loader, device)
        valid_loss = metrics["loss"]
        valid_accuracy = metrics["accuracy"]
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['valid_loss'].append(valid_loss)
        history['valid_accuracy'].append(valid_accuracy)
        
        # 显示进度
        print_string = f'Epoch: {e + 1}/{num_epochs} '
        print_string += f'TrainLoss: {train_loss:.5f} '
        print_string += f'ValidLoss: {valid_loss:.5f} '
        print_string += f'ACC: {valid_accuracy:.5f} '
        print(print_string)
        
        # 如果验证损失减小，保存模型
        if valid_loss < valid_loss_min:
            print(f'验证损失减小 ({valid_loss_min:.6f} --> {valid_loss:.6f}). 保存模型...')
            torch.save(model.state_dict(), Config.model_path)
            valid_loss_min = valid_loss
            best_accuracy = valid_accuracy
            counter = 0
        else:
            counter += 1
            print(f'早停计数器: {counter}/{patience}')
            
            # 如果连续patience个epoch验证损失没有减小，停止训练
            if counter >= patience:
                print('早停触发。停止训练。')
                break
    
    return best_accuracy, history


def train_with_kfold():
    """
    使用K折交叉验证训练模型
    """
    # 加载数据
    X_train, y_train, test_loader, word_to_idx, test_zero_idx, pretrained_embeddings = load_data(
        Config.train_path, Config.test_path, embedding_dim=Config.embedding_dim)
    
    # 创建K折数据加载器
    fold_loaders = create_kfold_loaders(X_train, y_train, n_splits=Config.k_folds, batch_size=Config.batch_size)
    
    # 存储每个折的最佳准确率
    fold_accuracies = []
    
    # 对每个折进行训练
    for fold, (train_loader, valid_loader) in enumerate(fold_loaders):
        print(f'开始训练第 {fold+1}/{Config.k_folds} 折')
        
        # 初始化模型
        vocab_size = len(word_to_idx) + 1
        model = SentimentRNN(
            vocab_size, 
            Config.output_size, 
            Config.embedding_dim, 
            Config.hidden_dim, 
            Config.n_layers, 
            Config.dropout,
            pretrained_embeddings=pretrained_embeddings,
            freeze_embeddings=Config.freeze_embeddings
        )
        model = model.to(Config.device)
        
        # 定义优化器和损失函数
        optimizer = optim.Adam(model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay)
        criterion = nn.BCELoss()
        
        # 使用早停法训练模型
        best_accuracy, _ = train_model_with_early_stopping(
            model, 
            optimizer, 
            criterion, 
            train_loader, 
            valid_loader,
            Config.patience, 
            Config.epochs, 
            Config.clip_value,
            Config.device
        )
        
        fold_accuracies.append(best_accuracy)
        print(f'第 {fold+1} 折最佳准确率: {best_accuracy:.5f}')
    
    # 输出所有折的平均准确率
    mean_accuracy = sum(fold_accuracies) / len(fold_accuracies)
    print(f'K折交叉验证平均准确率: {mean_accuracy:.5f}')
    
    # 使用全部训练数据训练最终模型
    print('使用全部训练数据训练最终模型...')
    
    # 创建全部数据的加载器
    full_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    full_loader = torch.utils.data.DataLoader(full_dataset, batch_size=Config.batch_size, shuffle=True)
    
    # 初始化最终模型
    final_model = SentimentRNN(
        vocab_size, 
        Config.output_size, 
        Config.embedding_dim, 
        Config.hidden_dim, 
        Config.n_layers, 
        Config.dropout,
        pretrained_embeddings=pretrained_embeddings,
        freeze_embeddings=Config.freeze_embeddings
    )
    final_model = final_model.to(Config.device)
    
    # 定义优化器和损失函数
    final_optimizer = optim.Adam(final_model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay)
    final_criterion = nn.BCELoss()
    
    # 训练最终模型
    for e in range(Config.final_epochs):
        train_loss = train_loop(final_model, final_optimizer, final_criterion, full_loader, Config.clip_value, Config.device)
        print(f'Epoch: {e + 1}/{Config.final_epochs} TrainLoss: {train_loss:.5f}')
    
    # 保存最终模型
    torch.save(final_model.state_dict(), Config.model_path)
    print('最终模型已保存')


def train():
    """
    传统训练方法（不使用K折交叉验证）
    """
    # 加载数据
    X_train, y_train, test_loader, word_to_idx, test_zero_idx, pretrained_embeddings = load_data(
        Config.train_path, Config.test_path, embedding_dim=Config.embedding_dim)
    
    # 分割训练集和验证集
    dataset_size = len(X_train)
    indices = list(range(dataset_size))
    split = int(np.floor(Config.val_size * dataset_size))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    
    # 创建数据加载器
    train_dataset = torch.utils.data.TensorDataset(X_train[train_indices], y_train[train_indices])
    valid_dataset = torch.utils.data.TensorDataset(X_train[val_indices], y_train[val_indices])
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=Config.batch_size, shuffle=False)
    
    # 初始化模型
    vocab_size = len(word_to_idx) + 1
    model = SentimentRNN(
        vocab_size, 
        Config.output_size, 
        Config.embedding_dim, 
        Config.hidden_dim, 
        Config.n_layers, 
        Config.dropout,
        pretrained_embeddings=pretrained_embeddings,
        freeze_embeddings=Config.freeze_embeddings
    )
    model = model.to(Config.device)
    
    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay)
    criterion = nn.BCELoss()
    
    # 使用早停法训练模型
    train_model_with_early_stopping(
        model, 
        optimizer, 
        criterion, 
        train_loader, 
        valid_loader,
        Config.patience, 
        Config.epochs, 
        Config.clip_value,
        Config.device
    )


if __name__ == '__main__':
    if Config.use_kfold:
        train_with_kfold()
    else:
        train()
