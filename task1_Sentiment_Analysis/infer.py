import pandas as pd
import torch
from task1_Sentiment_Analysis.Config import Config
from task1_Sentiment_Analysis.data_process import load_data
from task1_Sentiment_Analysis.train import get_prediction
from task1_Sentiment_Analysis.train import SentimentRNN


@torch.no_grad()
def prediction(model, test_loader, test_zero_idx, device=Config.device, batch_size=Config.test_batch_size):
    df = pd.DataFrame({'PhraseId': pd.Series(dtype='int'),
                       'Sentiment': pd.Series(dtype='int')})
    test_h = model.init_hidden(batch_size, device)
    model.eval()
    for seq, id_ in test_loader:
        test_h = tuple([each.data for each in test_h])
        seq = seq.to(device)
        out, test_h = model(seq, test_h)
        out = get_prediction(out)
        for ii, row in zip(id_, out):
            if ii in test_zero_idx:
                predicted = 2
            else:
                predicted = int(torch.argmax(row))
            subm = {'PhraseId': int(ii),
                    'Sentiment': predicted}
            df = pd.concat([df, pd.DataFrame([subm])], ignore_index=True)
    return df


def infer():
    # 加载数据
    _, _, test_loader, encode_voc, test_zero_idx = load_data(Config.train_path, Config.test_path)

    # 加载模型
    vocab_size = len(encode_voc) + 1
    model = SentimentRNN(vocab_size, Config.output_size, Config.embedding_dim, Config.hidden_dim, Config.n_layers,
                         Config.dropout)
    model.load_state_dict(torch.load(Config.model_path))
    model = model.to(Config.device)

    # 进行推断
    predictions = prediction(model, test_loader, test_zero_idx)
    predictions.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    infer()
