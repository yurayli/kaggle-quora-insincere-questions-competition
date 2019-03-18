import os, time, json, re
import itertools, argparse, pickle, random

import numpy as np
import pandas as pd
import nltk
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import f1_score, roc_curve, precision_recall_curve

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, StepLR, MultiStepLR, ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, sampler

misspell_dict = {"aren't" : "are not", "can't" : "cannot", "couldn't" : "could not",
    "didn't" : "did not", "doesn't" : "does not", "don't" : "do not", "hadn't" : "had not",
    "hasn't" : "has not", "haven't" : "have not", "he'd" : "he would", "he'll" : "he will",
    "he's" : "he is", "i'd" : "I would", "i'd" : "I had", "i'll" : "I will", "i'm" : "I am",
    "isn't" : "is not", "it's" : "it is", "it'll":"it will", "i've" : "I have", "let's" : "let us",
    "mightn't" : "might not", "mustn't" : "must not", "shan't" : "shall not", "she'd" : "she would",
    "she'll" : "she will", "she's" : "she is", "shouldn't" : "should not", "that's" : "that is",
    "there's" : "there is", "they'd" : "they would", "they'll" : "they will", "they're" : "they are",
    "they've" : "they have", "we'd" : "we would", "we're" : "we are", "weren't" : "were not",
    "we've" : "we have", "what'll" : "what will", "what're" : "what are", "what's" : "what is",
    "what've" : "what have", "where's" : "where is", "who'd" : "who would", "who'll" : "who will",
    "who're" : "who are", "who's" : "who is", "who've" : "who have", "won't" : "will not",
    "wouldn't" : "would not", "you'd" : "you would", "you'll" : "you will", "you're" : "you are",
    "you've" : "you have", "'re": " are", "wasn't": "was not", "we'll":" will", "didn't": "did not",
    "tryin'":"trying"}


puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£',
 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…',
 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─',
 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞',
 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]

def clean_punc(x):
    x = str(x)
    for punct in puncts:
        x = x.replace(punct, f' {punct} ')
    return x

def clean_numbers(x):
    x = re.sub('[0-9]{5,}', '#####', x)
    x = re.sub('[0-9]{4}', '####', x)
    x = re.sub('[0-9]{3}', '###', x)
    x = re.sub('[0-9]{2}', '##', x)
    return x

# def clean_text(raw, correct=False):
#     text = re.sub(r'[^\w\s\'\*]+', '', raw.strip())
#     misspell_re = re.compile('(%s)' % '|'.join(misspell_dict.keys()))
#     def replace(match):
#         return misspell_dict[match.group(0)]
#     return misspell_re.sub(replace, text) if correct else text
def clean_text(raw):
    return clean_numbers(clean_punc(raw.strip().lower()))


def word_idx_map(raw_questions, vocab_size):
    texts = []
    for q in raw_questions:
        texts.append(q.split())
    word_freq = nltk.FreqDist(itertools.chain(*texts))
    vocab_freq = word_freq.most_common(vocab_size-2)
    idx_to_word = ['<pad>'] + [word for word, cnt in vocab_freq] + ['<unk>']
    word_to_idx = {word:idx for idx, word in enumerate(idx_to_word)}

    return idx_to_word, word_to_idx


def tokenize(questions, word_to_idx, maxlen):
    '''
    Tokenize and numerize the question sequences
    Inputs:
    - questions: pandas series with quora questions
    - word_to_idx: mapping from word to index
    - maxlen: max length of each sequence of tokens

    Returns:
    - tokens: array of shape (data_size, maxlen)
    '''

    tokens = []
    for q in questions.tolist():
        token = [(lambda x: word_to_idx[x] if x in word_to_idx else word_to_idx['<unk>'])(w) \
                 for w in q.split()]
        if len(token) > maxlen:
            token = token[-maxlen:]
        else:
            token = [0] * (maxlen-len(token)) + token
        tokens.append(token)
    return np.array(tokens).astype('int32')


# Load pre-trained word vector
def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')

def get_embedding(embedding_file, embedding_dim, word_to_idx, vocab_size):
    with open(embedding_file, encoding="utf8", errors='ignore') as f:
        embeddings_index = dict(get_coefs(*o.split(' ')) for o in f if len(o)>100)
    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    nb_words = min(vocab_size, len(word_to_idx))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embedding_dim))
    for word, i in word_to_idx.items():
        if i >= vocab_size: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    return embedding_matrix


class Quora_sincerity(Dataset):

    def __init__(self, tokenized_questions, targets=None):
        self.questions = tokenized_questions
        self.targets = targets[:,None] if targets is not None else targets

    def __getitem__(self, index):
        question = self.questions[index]
        if self.targets is not None:
            target = self.targets[index]
            return torch.LongTensor(question), torch.FloatTensor(target)
        else:
            return torch.LongTensor(question)

    def __len__(self):
        return len(self.questions)


def prepare_loader(x, y=None, batch_size=1024, train=True):
    data_set = Quora_sincerity(x, y)
    if train:
        return DataLoader(data_set, batch_size=batch_size, shuffle=True)
    else:
        return DataLoader(data_set, batch_size=batch_size)


# plot log of loss and f1 during training
def plot_history(history, fname):
    f1s, val_f1s, losses, val_losses = history
    epochs = range(len(f1s))
    fig = plt.figure(figsize=(14,5))
    ax1 = fig.add_subplot(1,2,1)
    ax1.plot(epochs, f1s, '-o')
    ax1.plot(epochs, val_f1s, '-o')
    #ax1.set_ylim(0.6, 0.8)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('F1')
    ax1.legend(['train', 'val'], loc='lower right')
    ax2 = fig.add_subplot(1,2,2)
    ax2.plot(epochs, losses, '-o')
    ax2.plot(epochs, val_losses, '-o')
    #ax2.set_ylim(bottom=-0.1)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend(['train', 'val'], loc='upper right')
    fig.savefig(fname)
