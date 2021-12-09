# Sources:
# https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html
# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
# https://github.com/pytorch/tutorials/blob/master/beginner_source/transformer_tutorial.py


import math
import time
from enum import Enum

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchtext.data.functional import to_map_style_dataset
from torchtext.datasets import IMDB
from torchtext.vocab import build_vocab_from_iterator


def tokenize(line):
    return line.split()


def yield_tokens(train_iter):
    for _, line in train_iter:
        yield tokenize(line)


def create_vocabulary(train_iter):
    vocab = build_vocab_from_iterator(yield_tokens(
        train_iter), min_freq=5, specials=["<unk>", "<pad>"])
    vocab.set_default_index(vocab["<unk>"])
    return vocab


def map_text_to_int(text):
    return vocab(tokenize(text))


class Labels(Enum):
    NEGATIVE = 0
    POSITIVE = 1


TEXT_LABELS_MAP = {
    'neg': Labels.NEGATIVE,
    'pos': Labels.POSITIVE
}


def collate_batch(batch):
    labels, texts = [], []
    for label, text in batch:
        labels.append(TEXT_LABELS_MAP[label].value)
        texts.append(torch.tensor(
            map_text_to_int(text[:512]), dtype=torch.int64))

    labels_tensor = torch.tensor(labels, dtype=torch.long)
    texts_tensor = nn.utils.rnn.pad_sequence(texts, padding_value=1)

    return labels_tensor.to(device), texts_tensor.to(device)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pass through all training data to create a vocabulary.
train_iter = IMDB(split='train')
vocab = create_vocabulary(train_iter)

# Prepare iterators
train_iter = IMDB(split="train")
test_iter = IMDB(split="test")
train_dataset = to_map_style_dataset(train_iter)
train_size = int(0.95 * len(train_dataset))
validation_size = len(train_dataset) - train_size
train_dataset, validation_dataset = torch.utils.data.random_split(
    train_dataset, [train_size, validation_size], generator=torch.Generator().manual_seed(42))
test_dataset = to_map_style_dataset(test_iter)
train_dataloader = DataLoader(
    train_dataset, batch_size=32, shuffle=True, collate_fn=collate_batch)
validation_dataloader = DataLoader(
    validation_dataset, batch_size=32, collate_fn=collate_batch)
test_dataloader = DataLoader(
    test_dataset, batch_size=32, collate_fn=collate_batch)


class EncoderClassifier(nn.Module):

    def __init__(self, ntoken, d_model, dropout, nhead, nhid, nlayers) -> None:
        super().__init__()

        self.src_mask = None
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model, nhead, nhid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, d_model)
        self.embedding_size = d_model
        self.decoder = nn.Linear(d_model, 2)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src):
        source_mask = (src == 1).cpu().reshape(src.shape[1], src.shape[0])
        src = self.encoder(src) * math.sqrt(self.embedding_size)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(
            src, src_key_padding_mask=source_mask)
        output, _ = torch.max(output, dim=0)
        # output = torch.mean(output, dim=0)
        output = self.decoder(output)
        return output


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    # max length can be passed in as a param of longest input sequence
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


d_model = 512
dropout = 0.2
num_of_attention_heads = 8
hidden_units = 512
num_of_encoder_components = 1

model = EncoderClassifier(len(vocab), d_model, dropout, num_of_attention_heads,
                          hidden_units, num_of_encoder_components).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)


def train(dataloader):
    model.train()
    total_acc, total_count = 0, 0
    start_time = time.time()

    for idx, (label, text) in enumerate(dataloader):
        optimizer.zero_grad()
        predicted_label = model(text)
        loss = criterion(predicted_label, label)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % 50 == 0:
            # evaluate(test_dataloader)
            elapsed = time.time() - start_time
            print('| epoch | {:5d}/{:5d} batches '
                  '| accuracy {:8.3f}'.format(idx, len(dataloader),
                                              total_acc/total_count))
            total_acc, total_count = 0, 0
            start_time = time.time()


def evaluate(model, dataloader):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (label, text) in enumerate(dataloader):
            predicted_label = model(text)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    print(
        f'Accuracy is {total_acc/total_count}. Correct predictions: {total_acc}, Total count: {total_count}')
    return total_acc/total_count


max_accu = 0
for epoch in range(10):
    epoch_start_time = time.time()
    train(train_dataloader)
    accu_val = evaluate(model, validation_dataloader)
    if accu_val > max_accu:
        best_model_path = f'/home/mleginus/ml/transformers/models/transformer_encoder_{epoch}_{accu_val}.model'
        torch.save(model, best_model_path)
        print(f"Model has been saved: {best_model_path}")
        max_accu = accu_val

    print('-' * 59)
    print('| end of epoch {:3d} | time: {:5.2f}s | '
          'valid accuracy {:8.3f} '.format(epoch,
                                           time.time() - epoch_start_time,
                                           accu_val))

loaded_model = torch.load(best_model_path)
accu_val = evaluate(loaded_model, test_dataloader)
