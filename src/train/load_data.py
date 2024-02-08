from torch import nn
import torch
import numpy as np
import pandas as pd
import os
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from torch.utils.data import TensorDataset, DataLoader

parent = Path(__file__).parent
data_path = os.path.join(parent, "../../data/processed")
train_path = os.path.join(data_path, "train.pkl")
test_path = os.path.join(data_path, 'test.pkl')


def create_dataloaders(train_path, test_path, batch_size):
    train_data = pd.read_pickle(train_path)
    test_data = pd.read_pickle(test_path)

    train_data['sentiment'] = train_data['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
    test_data['sentiment'] = test_data['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

    vectorizer = TfidfVectorizer(analyzer='word', tokenizer=lambda x: x, preprocessor=lambda x: x, token_pattern=None)
    vectorizer.fit(train_data['final_review'].values)

    train_x = vectorizer.transform(train_data['final_review']).toarray()
    test_x = vectorizer.transform(test_data['final_review']).toarray()
    train_y = train_data['sentiment'].values
    test_y = train_data['sentiment'].values

    train_dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    test_dataset = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)

    return train_loader, test_loader


if __name__ == '__main__':
    train_loader, test_loader = create_dataloaders(train_path, test_path, 50)
    iterable = iter(train_loader)
    sample_x = next(iterable)
    print(sample_x)
    print(sample_x.shape)
