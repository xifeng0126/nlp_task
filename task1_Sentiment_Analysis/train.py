import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch import nn, optim
from task1_Sentiment_Analysis.model import SentimentRNN
from task1_Sentiment_Analysis.Config import Config
from task1_Sentiment_Analysis.data_process import load_data


def train_loop(model, optimizer, criterion, train_loader, clip_value, device, batch_size=Config.batch_size):
    """
    Train model
    """
    running_loss = 0
    model.train()

    # returns the initial hidden state for the LSTM layers
    h = model.init_hidden(batch_size, device)

    for seq, targets in train_loader:
        # move data to device
        seq = seq.to(device)
        targets = targets.to(device)

        # convert the elements of the hidden state tuple h to tensors with the same device as the input data.
        h = tuple([each.data for each in h])

        # perform a forward pass through the model.
        # returns the model's output (out) and the updated hidden state (h).
        out, h = model.forward(seq, h)

        # calculate the loss between the predicted output and the target values
        loss = criterion(out, targets.float())
        running_loss += loss.item() * seq.shape[0]

        # reset the gradients of the model's parameters
        optimizer.zero_grad()

        # compute the gradients of the loss with respect to the model's parameters
        loss.backward()
        if clip_value:
            # clip the gradients to prevent them from exploding
            nn.utils.clip_grad_norm_(model.parameters(), clip_value)

        # update the model's parameters
        optimizer.step()
    running_loss /= len(train_loader.sampler)
    return running_loss


def get_prediction(t):
    max_indices = torch.argmax(t, dim=1)
    new = torch.zeros_like(t)
    new[torch.arange(t.shape[0]), max_indices] = 1
    return new


def eval_loop(model, criterion, eval_loader, device, batch_size=Config.batch_size, ignore_index=None):
    """
    Evaluate model
    """

    # returns the initial hidden state for the LSTM layers
    val_h = model.init_hidden(batch_size, device)
    val_loss = 0
    model.eval()
    accuracy = []
    for seq, targets in eval_loader:
        # convert the elements of the hidden state tuple val_h to tensors with the same device as the input data.
        val_h = tuple([each.data for each in val_h])

        # move data to device
        seq = seq.to(device)
        targets = targets.to(device)

        # perform a forward pass through the model.
        # returns the model's output (out) and the updated hidden state (val_h).
        out, val_h = model(seq, val_h)

        # calculate the loss
        loss = criterion(out, targets.float())
        val_loss += loss.item() * seq.shape[0]

        # convert the model's output
        predicted = get_prediction(out).flatten().cpu().numpy()
        labels = targets.view(-1).cpu().numpy()

        # calculate the accuracy score between the predicted and target values
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
        # train for epoch
        train_loss = train_loop(model, optimizer, criterion, train_loader, clip_value, device)

        if (e + 1) % eval_every == 0:

            # evaluate on validation set
            metrics = eval_loop(model, criterion, valid_loader, device)

            # show progress
            print_string = f'Epoch: {e + 1} '
            print_string += f'TrainLoss: {train_loss:.5f} '
            print_string += f'ValidLoss: {metrics["loss"]:.5f} '
            print_string += f'ACC: {metrics["accuracy"]:.5f} '
            print(print_string)

            # save the model
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
