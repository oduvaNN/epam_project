from torch import nn
import torch
from src.train.model import SentimentRNN
from torch.utils.data import DataLoader
import numpy as np
import os
from pathlib import Path
from datetime import datetime
from src.train.load_data import create_dataloaders


EXPERIMENT_NAME = f"sentimentalRNN_1"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

parent = Path(__file__).parent
model_path = os.path.join(parent, "../../outputs", EXPERIMENT_NAME)
Path(model_path).mkdir(parents=True, exist_ok=True)

data_path = os.path.join(parent, "../../data/processed")
train_path = os.path.join(data_path, "train.pkl")
test_path = os.path.join(data_path, 'test.pkl')

class TrainPipeline:
    def __init__(self, num_layers, vocab_size, embedding_dim, hidden_dim, lr, batch_size):
        self.model = SentimentRNN(num_layers, vocab_size, hidden_dim, embedding_dim)

        self.model.to(device)
        self.batch_size = batch_size
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.train_loss = []
        self.eval_loss = []
        self.train_acc = []
        self.eval_acc = []

        self.best_loss = torch.inf

    @staticmethod
    def accuracy(pred, label):
        pred = torch.round(pred.squeeze())
        return torch.sum(pred == label.squeeze()).item()

    def train(self, epochs, train_loader: DataLoader, test_loader: DataLoader):

        for epoch in range(epochs):
            print(f"Epoch {epoch}")
            tr_ls = []  # train losses for batches in this epoch
            ev_ls = []  # same but eval loss
            tr_acc = 0.0
            self.model.train()
            h = self.model.init_hidden(self.batch_size)

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                h = tuple([each.data for each in h])

                self.model.zero_grad()
                out, h = self.model(inputs, h)

                loss = self.criterion(out.squeeze(), labels.float())

                loss.backward()
                tr_ls.append(loss.item())

                accuracy = TrainPipeline.accuracy(out, labels)
                tr_acc += accuracy / len(train_loader.dataset)

                nn.utils.clip_grad_norm(self.model.parameters(), 5)
                self.optimizer.step()

            eval_h = self.model.init_hidden(self.batch_size)
            ev_acc = 0.0
            self.model.eval()

            for inputs, labels in test_loader:
                eval_h = tuple([each.data for each in eval_h])

                inputs, labels = inputs.to(device), labels.to(device)

                out, eval_h = self.model(inputs, eval_h)
                eval_loss = self.criterion(out.squeeze(), labels.float())

                ev_ls.append(eval_loss)
                accuracy = TrainPipeline.accuracy(out, labels)
                ev_acc += accuracy / len(test_loader.dataset)

            ep_tr_ls = np.mean(tr_ls)
            ep_ev_ls = np.mean(ev_ls)
            self.train_loss.append(ep_tr_ls)
            self.eval_loss.append(ep_ev_ls)
            self.train_acc.append(tr_acc)
            self.eval_acc.append(ev_acc)

            print(f"Train loss: {ep_tr_ls}  Evaluation loss: {ep_ev_ls}")
            print(f"Train accuracy: {tr_acc}    Evaluation accuracy: {ev_acc}")

            if ep_ev_ls <= self.best_loss:
                torch.save(self.model.state_dict(), os.path.join(model_path, 'best_model.pt'))

                self.best_loss = ep_ev_ls

        print('='*25)


if __name__ == '__main__':
    print(torch.cuda.is_available())
    batch_size = 50
    train_loader, test_loader, vocab_size = create_dataloaders(train_path, test_path, batch_size)

    pipeline = TrainPipeline(2, vocab_size, 64, 256, 0.001, batch_size)
    pipeline.train(5, train_loader, test_loader)
