from data_utils import *
from solver import *
from model import *

path = '/input/'
model_path = '/model/'
output_path = '/output/'

SEED = 2019
EMBEDDING_FILE_GV = '/embeddings_1/glove.840B.300d.txt'
EMBEDDING_FILE_PR = '/embeddings_2/paragram_300_sl999/paragram_300_sl999.txt'
EMBEDDING_FILE_FT = '/embeddings_2/wiki-news-300d-1M/wiki-news-300d-1M.vec'

parser = argparse.ArgumentParser()
parser.add_argument('--maxlen', type=int, default=40,
                    help='maximum length of a question sentence')
parser.add_argument('--vocab-size', type=int, default=120000)
parser.add_argument('--n-splits', type=int, default=5,
                    help='splits of n-fold cross validation')
parser.add_argument('--log-plot', type=bool, default=1,
                    help='whether to plot the metrics log')
parser.add_argument('--preload', type=bool, default=0,
                    help='whether to pre-load model')
parser.add_argument('--model-name', type=str)
parser.add_argument('--batch-size', type=int, default=256,
                    help='batch size during training')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=5,
                    help='number of training epochs')
args = parser.parse_args()


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

    print('tokenizing...')
    t0 = time.time()
    idx_to_word, word_to_idx = word_idx_map(train_df['question_text'], args.vocab_size)
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

    # train/val split
    np.random.seed(SEED)
    mask = np.random.rand(len(train_seq)) > 0.1
    x_val = train_seq[~mask]
    x_train = train_seq[mask]
    y_val = train_tar[~mask]
    y_train = train_tar[mask]

    # load pretrained embedding
    print('loading embeddings...')
    t0 = time.time()
    embed_mat_1 = get_embedding(EMBEDDING_FILE_GV, 300, word_to_idx, vocab_size)
    embed_mat_2 = get_embedding(EMBEDDING_FILE_PR, 300, word_to_idx, vocab_size)
    #embed_mat_3 = get_embedding(EMBEDDING_FILE_FT, 300, word_to_idx, vocab_size)
    embed_mat = np.mean([embed_mat_1, embed_mat_2], 0)
    print('loading complete in {:.0f} seconds.'.format(time.time()-t0))

    # model training
    model = QIQCNet(300, 256, args.vocab_size, embed_mat)
    if args.preload:
        model.load_state_dict(torch.load(model_path+args.model_name))
    for name, param in model.named_parameters():
       if 'emb' in name:
           param.requires_grad = False
    solver = NetSolver(model, lr_init=args.lr)

    train_loader = prepare_loader(x_train, y_train, args.batch_size)
    val_loader = prepare_loader(x_val, y_val, train=False)

    t0 = time.time()
    solver.train(loaders=(train_loader, val_loader), epochs=args.epochs)
    time_elapsed = time.time() - t0
    print('Training complete in {:.0f}m {:.0f}s.'.format(time_elapsed // 60, time_elapsed % 60))

    # log plot
    if args.log_plot:
        history = [solver.f1_history, solver.val_f1_history, solver.loss_history, solver.val_loss_history]
        with open(output_path+'history.pkl', 'wb') as f:
            pickle.dump(history, f)
        plot_history(history, output_path+'qiqc_curve.png')


if __name__ == '__main__':
    main(args)

