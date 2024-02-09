from torch import nn
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)


class SentimentRNN(nn.Module):
    def __init__(self, num_layers, vocab_size, hidden_dim, embedding_dim):
        super(SentimentRNN, self).__init__()

        self.output_dim = 1  # probability of review being positive
        self.hidden_dim = hidden_dim

        self.num_layers = num_layers
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=self.hidden_dim,
                            num_layers=self.num_layers, batch_first=True)

        self.dropout = nn.Dropout(0.25)

        self.fc = nn.Linear(self.hidden_dim, self.output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden):
        batch_size = x.size(0)
        embeddings = self.embedding(x)

        lstm_output, hidden = self.lstm(embeddings, hidden)

        lstm_output = lstm_output.view(-1, self.hidden_dim)

        out = self.dropout(lstm_output)
        out = self.fc(out)

        out = self.sig(out)
        out = out.view(batch_size, -1)

        out = out[:, -1]  # get last batch of labels

        return out

    def init_hidden(self, batch_size):

        h0 = torch.zeros((self.num_layers, batch_size, self.hidden_dim)).to(device)
        c0 = torch.zeros((self.num_layers, batch_size, self.hidden_dim)).to(device)

        return h0, c0
