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

USE_GPU = True
dtype = torch.float32
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

SEED = 2019
path = '../input/'
output_path = './'
EMBEDDING_FILE_GV = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
EMBEDDING_FILE_PR = '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
EMBEDDING_FILE_FT = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'

parser = argparse.ArgumentParser()
parser.add_argument('--maxlen', type=int, default=40,
                    help='maximum length of a question sentence')
parser.add_argument('--vocab-size', type=int, default=120000)
parser.add_argument('--n-splits', type=int, default=5,
                    help='splits of n-fold cross validation')
parser.add_argument('--batch-size', type=int, default=512,
                    help='batch size during training')
parser.add_argument('--lr', type=float, default=0.003)
parser.add_argument('--epochs', type=int, default=4,
                    help='number of training epochs')
args = parser.parse_args()


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

def clean_text(raw):
    return clean_numbers(clean_punc(raw.lower()))


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
        loader = DataLoader(data_set, batch_size=batch_size, shuffle=True)
    else:
        loader = DataLoader(data_set, batch_size=batch_size)
    return loader


# evaluation metric
def f1_threshold(y_true, preds):
    best_f1 = 0
    for i in np.arange(0.1, 0.51, 0.01):
        f1 = f1_score(y_true, preds > i)
        if f1 > best_f1:
            threshold = i
            best_f1 = f1
    return best_f1, threshold


# solver of model with validation
class NetSolver(object):

    def __init__(self, model, **kwargs):
        self.model = model

        # hyperparameters
        self.lr_init = kwargs.pop('lr_init', 0.001)
        self.lr_decay = kwargs.pop('lr_decay', 0.1)
        self.step_size = kwargs.pop('step_size', 10)
        self.print_every = kwargs.pop('print_every', 2000)
        self.checkpoint_name = kwargs.pop('checkpoint_name', 'qiqc')

        # setup optimizer
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                    lr=self.lr_init)
        #self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
        #                           lr=self.lr_init, momentum=0.9)
        #self.scheduler = StepLR(self.optimizer, step_size=self.step_size, gamma=self.lr_decay)

        self.model = self.model.to(device=device)
        self._reset()

    def _reset(self):
        """
        Set up some book-keeping variables for optimization. Don't call this
        manually.
        """
        self.best_val_loss = 0.
        self.best_val_f1 = 0.
        self.loss_history = []
        self.val_loss_history = []
        self.f1_history = []
        self.val_f1_history = []

    def _save_checkpoint(self, epoch, l_val, f_val):
        torch.save(self.model.state_dict(),
            output_path+self.checkpoint_name+'_%.3f_%.3f_epoch_%d.pth.tar' %(l_val, f_val, epoch))
        checkpoint = {
            'optimizer': str(type(self.optimizer)),
            'scheduler': str(type(self.scheduler)),
            'lr_init': self.lr_init,
            'lr_decay': self.lr_decay,
            'step_size': self.step_size,
            'epoch': epoch,
        }
        with open(output_path+'hyper_param_optim.json', 'w') as f:
            json.dump(checkpoint, f)


    def forward_pass(self, x, y):
        x = x.to(device=device, dtype=torch.long)
        y = y.to(device=device, dtype=dtype)
        scores = self.model(x)
        loss = F.binary_cross_entropy_with_logits(scores, y)
        return loss, torch.sigmoid(scores)


    def train(self, loaders, epochs):
        train_loader, val_loader = loaders

        # Start training for epochs
        for e in range(epochs):
            print('\nEpoch %d / %d:' % (e + 1, epochs))
            self.model.train()
            #self.scheduler.step()
            running_loss = 0.

            for x, y in train_loader:
                loss, _ = self.forward_pass(x, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * x.size(0)

            N = len(train_loader.dataset)
            train_loss = running_loss / N
            print('For train set,')
            train_f1, train_thres, _ = self.check_f1(train_loader, num_batches=50)
            print('For val set,')
            val_f1, val_thres, val_loss = self.check_f1(val_loader, save_scores=True)

            # Checkpoint and record/print metrics at epoch end
            self.log_and_checkpoint(e, train_loss, val_loss, train_f1, val_f1)


    def train_one_cycle(self, loaders, epochs, max_lr, moms=(.9, .8), div_factor=25, sep_ratio=0.3):
        train_loader, val_loader = loaders

        # one-cycle setup
        tot_it = epochs * len(train_loader)
        up_it = int(tot_it * sep_ratio)
        down_it = tot_it - up_it
        min_lr = max_lr / div_factor
        n, curr_lr, curr_mom = 0, min_lr, moms[0]
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = curr_lr
            param_group['betas'] = (curr_mom, 0.999)

        # Start training for epochs
        for e in range(epochs):
            self.model.train()
            print('\nEpoch %d / %d:' % (e + 1, epochs))
            running_loss = 0.

            for t, (x, y) in enumerate(train_loader):
                loss, _ = self.forward_pass(x, y)

                if (t + 1) % self.print_every == 0:
                    print('t = %d, loss = %.4f' % (t+1, loss.item()))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * x.size(0)

                # update lr, mom per iter
                n += 1
                if n <= up_it:
                    curr_lr = max_lr + (min_lr - max_lr)/2 * (np.cos(np.pi*n/up_it)+1)
                    curr_mom = moms[1] + (moms[0] - moms[1])/2 * (np.cos(np.pi*n/up_it)+1)
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = curr_lr
                        param_group['betas'] = (curr_mom, 0.999)
                else:
                    curr_lr = min_lr + (max_lr - min_lr)/2 * (np.cos(np.pi*(n-up_it)/down_it)+1)
                    curr_mom = moms[0] + (moms[1] - moms[0])/2 * (np.cos(np.pi*(n-up_it)/down_it)+1)
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = curr_lr
                        param_group['betas'] = (curr_mom, 0.999)

            N = len(train_loader.dataset)
            train_loss = running_loss / N
            print('For train set,')
            train_f1, train_thres, _ = self.check_f1(train_loader, num_batches=50)
            print('For val set,')
            val_f1, val_thres, val_loss = self.check_f1(val_loader, save_scores=True)

            self.log_and_checkpoint(e, train_loss, val_loss, train_f1, val_f1)


    def log_and_checkpoint(self, e, train_loss, val_loss, train_f1, val_f1):
        # Checkpoint and record/print metrics at epoch end
        self.loss_history.append(train_loss)
        self.val_loss_history.append(val_loss)
        self.f1_history.append(train_f1)
        self.val_f1_history.append(val_f1)

        # for floydhub metric graphs
        print('{"metric": "F1", "value": %.4f, "epoch": %d}' % (train_f1, e+1))
        print('{"metric": "Val. F1", "value": %.4f, "epoch": %d}' % (val_f1, e+1))
        print('{"metric": "Loss", "value": %.4f, "epoch": %d}' % (train_loss, e+1))
        print('{"metric": "Val. Loss", "value": %.4f, "epoch": %d}' % (val_loss, e+1))

        if e == 0:
            self.best_val_f1 = val_f1
            self.best_val_loss = val_loss
        if val_f1 > self.best_val_f1:
            print('updating best val f1...')
            self.best_val_f1 = val_f1
        if val_loss < self.best_val_loss:
            print('updating best val loss...')
            self.best_val_loss = val_loss
        print()


    def check_f1(self, loader, num_batches=None, save_scores=False):
        self.model.eval()
        targets, scores, losses = [], [], []

        with torch.no_grad():
            for t, (x, y) in enumerate(loader):
                l, score = self.forward_pass(x, y)
                targets.append(y.cpu().numpy())
                scores.append(score.cpu().numpy())
                losses.append(l.item())
                if num_batches is not None and (t+1) == num_batches:
                    break

        targets = np.concatenate(targets)
        scores = np.concatenate(scores)
        if save_scores:
            self.val_scores = scores  # to access from outside

        best_f1, threshold = f1_threshold(targets, scores)
        loss = np.mean(losses)
        print('Best threshold is {:.4f}, with F1 score: {:.4f}'.format(threshold, best_f1))

        return best_f1, threshold, loss


# model
class QIQCNet(nn.Module):

    def __init__(self, embed_dim, hidden_dim, vocab_size, embed):
        super(QIQCNet, self).__init__()
        # Record the arguments
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        # Init layers
        self.emb = nn.Embedding(vocab_size, embed_dim)
        self.dropout_seq = nn.Dropout2d(0.25)

        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.gru = nn.GRU(hidden_dim*2, hidden_dim, bidirectional=True, batch_first=True)

        self.dropout = nn.Dropout(0.1)
        self.out = nn.Linear(hidden_dim*4, 1)

        # Weight initialization
        self.emb.weight = nn.Parameter(torch.tensor(embed, dtype=torch.float32))
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            if 'weight_hh' in name:
                nn.init.orthogonal_(param)

        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            if 'weight_hh' in name:
                nn.init.orthogonal_(param)

    def forward(self, seq):
        emb = self.emb(seq)
        emb = self.dropout_seq(emb.transpose(1,2).unsqueeze(-1)).squeeze().transpose(1,2)
        o_lstm, _ = self.lstm(emb)
        o_gru, _ = self.gru(o_lstm)

        # pooling
        avg_pool = torch.mean(o_gru, 1)
        max_pool, _ = torch.max(o_gru, 1)
        x = torch.cat((avg_pool, max_pool), 1)
        out = self.out(self.dropout(x))

        return out


def load_and_preproc():
    train_df = pd.read_csv(path+'train.csv')
    test_df = pd.read_csv(path+'test.csv')

    print('cleaning text...')
    t0 = time.time()
    train_df['question_text'] = train_df['question_text'].apply(clean_text)
    test_df['question_text'] = test_df['question_text'].apply(clean_text)
    print('cleaning complete in {:.0f} seconds.'.format(time.time()-t0))

    return train_df, test_df


def tokenize_questions(train_df, test_df):
    y_train = train_df['target'].values
    full_text = train_df['question_text'].tolist() + test_df['question_text'].tolist()

    print('tokenizing...')
    t0 = time.time()
    idx_to_word, word_to_idx = word_idx_map(full_text, args.vocab_size)
    x_train = tokenize(train_df['question_text'], word_to_idx, args.maxlen)
    x_test = tokenize(test_df['question_text'], word_to_idx, args.maxlen)
    print('tokenizing complete in {:.0f} seconds.'.format(time.time()-t0))

    return x_train, y_train, x_test, word_to_idx


def train_val_split(train_x, train_y):
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=SEED)
    cv_indices = [(tr_idx, val_idx) for tr_idx, val_idx in skf.split(train_x, train_y)]
    return cv_indices


def main(args):
    # load data
    train_df, test_df = load_and_preproc()
    train_seq, train_tar, x_test, word_to_idx = tokenize_questions(train_df, test_df)

    # load pretrained embedding
    print('loading embeddings...')
    t0 = time.time()
    embed_mat_1 = get_embedding(EMBEDDING_FILE_GV, 300, word_to_idx, args.vocab_size)
    embed_mat_2 = get_embedding(EMBEDDING_FILE_PR, 300, word_to_idx, args.vocab_size)
    #embed_mat_3 = get_embedding(EMBEDDING_FILE_FT, 300, word_to_idx, args.vocab_size)
    embed_mat = np.mean([embed_mat_1, embed_mat_2], 0)
    print('loading complete in {:.0f} seconds.'.format(time.time()-t0))

    # training preparation
    train_preds = np.zeros((len(train_seq))) # matrix for the out-of-fold predictions
    test_preds = np.zeros((len(test_df))) # matrix for the predictions on the testset
    test_loader = prepare_loader(x_test, train=False)
    cv_indices = train_val_split(train_seq, train_tar)

    print()
    for i, (trn_idx, val_idx) in enumerate(cv_indices):
        print(f'Fold {i + 1}')

        # train/val split
        x_train, x_val = train_seq[trn_idx], train_seq[val_idx]
        y_train, y_val = train_tar[trn_idx], train_tar[val_idx]

        # model setup
        model = QIQCNet(300, 256, args.vocab_size, embed_mat)
        for name, param in model.named_parameters():
           if 'emb' in name:
               param.requires_grad = False
        solver = NetSolver(model, lr_init=args.lr)

        # train
        train_loader = prepare_loader(x_train, y_train, args.batch_size)
        val_loader = prepare_loader(x_val, y_val, train=False)
        t0 = time.time()
        solver.train_one_cycle(loaders=(train_loader, val_loader), epochs=args.epochs, max_lr=args.lr)
        time_elapsed = time.time() - t0
        print('time = {:.0f}m {:.0f}s.'.format(time_elapsed // 60, time_elapsed % 60))

        # inference
        solver.model.eval()
        test_scores = []
        with torch.no_grad():
            for x in test_loader:
                x = x.to(device=device, dtype=torch.long)
                score = torch.sigmoid(solver.model(x))
                test_scores.append(score.cpu().numpy())
        test_scores = np.concatenate(test_scores)

        train_preds[val_idx] = solver.val_scores.squeeze()
        test_preds += test_scores.squeeze() / args.n_splits

        print()

    # submit
    best_f1, best_threshold = f1_threshold(train_tar, train_preds)
    print('For whole train set, best threshold is {:.4f}, with F1 score: {:.4f}'.format(best_threshold, best_f1))

    submission = test_df[['qid']].copy()
    submission['prediction'] = (test_preds > best_threshold).astype(int)
    submission.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    main(args)