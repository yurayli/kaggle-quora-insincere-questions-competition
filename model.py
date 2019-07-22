from data_utils import *

USE_GPU = True
dtype = torch.float32
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)


class Attention(nn.Module):
    def __init__(self, feature_dim, **kwargs):
        super(Attention, self).__init__(**kwargs)

        self.feature_dim = feature_dim
        self.proj_to_alpha = nn.Linear(feature_dim, 1)

        nn.init.xavier_uniform_(self.proj_to_alpha.weight)
        self.proj_to_alpha.bias.data.fill_(0.)

    def forward(self, seq):   # seq: (N, T, D)
        mid = F.relu(seq)   # (N, T, D)
        eij = self.proj_to_alpha(mid).squeeze()   # (N, T)
        alpha = F.softmax(eij, dim=-1)   # (N, T)
        context = torch.sum(alpha.unsqueeze(-1) * seq, 1)   # (N, D)

        return context, alpha


class QIQCNet(nn.Module):

    def __init__(self, embed_dim, hidden_dim, vocab_size, embed_matrix):
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
        self.lstm_attention = Attention(hidden_dim*2)
        self.gru_attention = Attention(hidden_dim*2)

        self.dropout = nn.Dropout(0.1)
        self.out = nn.Linear(hidden_dim*8, 1)

        # Weight initialization
        self.emb.weight = nn.Parameter(torch.tensor(embed_matrix, dtype=torch.float32))
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
        emb = self.dropout_seq(emb.transpose(1,2).unsqueeze(-1)).squeeze(-1).transpose(1,2)
        o_lstm, _ = self.lstm(emb)
        o_gru, _ = self.gru(o_lstm)
        o_lstm_atten, _ = self.lstm_attention(o_lstm)
        o_gru_atten, _ = self.gru_attention(o_gru)

        # pooling
        avg_pool = torch.mean(o_gru, 1)
        max_pool, _ = torch.max(o_gru, 1)
        x = torch.cat((avg_pool, max_pool, o_lstm_atten, o_gru_atten), 1)
        out = self.out(self.dropout(x))

        return out


def model_test():
    x = torch.zeros((64, 40), dtype=torch.long)
    x = x.to(device=device)

    model = QIQCNet(300, 64, vocab_size, embed_mat)
    model = model.to(device=device)
    scores = model(x)
    print(scores.size())



