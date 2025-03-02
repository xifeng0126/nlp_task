from torch import nn


class SentimentRNN(nn.Module):
    """
    The RNN model that will be used to perform Sentiment analysis.
    """

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob):
        """
        Initialize the model by setting up the layers.
        Arguments:
        vocab_size - The size of the vocabulary, i.e., the total number of unique words in the input data.
        output_size - The size of the output, which is usually set to 1 for binary classification tasks like sentiment analysis.
        embedding_dim - The dimensionality of the word embeddings. Each word in the input data will be represented by a dense vector of this dimension.
        hidden_dim - The number of units in the hidden state of the LSTM layer.
        n_layers - The number of layers in the LSTM.
        drop_prob - The probability of dropout, which is a regularization technique used to prevent overfitting.
        """
        super(SentimentRNN, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        # an embedding layer that maps each word index to its dense vector representation.
        # this layer is used to learn word embeddings during training.
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # an LSTM layer that processes the input sequence of word embeddings
        # and produces a sequence of hidden states.
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            n_layers,
                            dropout=0.5,
                            batch_first=True)

        # a dropout layer that randomly sets elements of the input to zero
        # with probability drop_prob.
        # this layer helps in preventing overfitting.
        self.dropout = nn.Dropout(p=drop_prob)

        # linear and sigmoid layers
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sig = nn.Sigmoid()

    def forward(self, x, hidden):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        # compute the word embeddings for the input sequence.
        batch_size = x.size(0)
        embeds = self.embedding(x)

        # pass the embeddings through the LSTM layer to get the LSTM outputs and the updated hidden state.
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        # apply dropout to the reshaped LSTM outputs.
        out = self.dropout(lstm_out)

        # pass the output through the fully connected layer.
        out = self.fc(out)

        # apply the sigmoid activation function to squash the output between 0 and 1.
        out = self.sig(out)
        out = out.view(batch_size, -1)

        # extract the last five elements from each sequence in the batch
        out = out[:, -5:]
        return out, hidden

    def init_hidden(self, batch_size, device):
        """
        Initializes hidden state
        """
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))

        return hidden