import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from .data_processing import preprocessing_for_bert

def data_sampler(text, labels, feature, tokenizer, batch_size=16, max_len=512, shuffle=True):
    inputs, masks = preprocessing_for_bert(text, tokenizer, max_len)
    labels = torch.tensor(labels)
    feature = torch.tensor(feature)
    data = TensorDataset(inputs, masks, labels, feature)
    if shuffle:
        data_sampler = DataLoader(data, sampler=RandomSampler(data), batch_size=batch_size)
    else:
        data_sampler = DataLoader(data, sampler=SequentialSampler(data), batch_size=batch_size)

    return data_sampler
