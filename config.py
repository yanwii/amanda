import argparse

parser = argparse.ArgumentParser(description='Training on NewsQA dataset')
# data inputs
parser.add_argument('-indextr', '--indexed_train_json', type=str,
            default='data/train_indexed.json',
            help='Indexed train JSON file')

parser.add_argument('-devjs', '--dev_json', type=str, default='data/dev-v1.1.json',
            help='formatted dev JSON file')
parser.add_argument('-tokdev', '--tok_dev_json', type=str,
            default='data/tokenized-dev-v1.1.json',
            help='tokenized dev JSON file')
parser.add_argument('-indexdev', '--indexed_dev_json', type=str,
            default='data/dev_indexed.json',
            help='Indexed dev JSON file')

parser.add_argument('-id2c', '--id2char', type=str, default='prep-data/id2char.json',
            help='id2char JSON file')

# model configs
parser.add_argument('-pretrained', '--pretrained_weightpath', type=str,
            default=None,
            help='path of any pretrained model weight')
parser.add_argument('-initepoch', '--initial_epoch', type=int, default=0,
            help='Initial epoch count')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.001,
            help='learning rate')
parser.add_argument('-clip', '--clipnorm', type=float, default=5.0,
            help='Clipnorm threshold')
parser.add_argument('-optim', '--optimizer', type=str, default='adam',
            choices=['adamax', 'adam', 'rmsprop'],
            help='backpropagation algorithm')

parser.add_argument('-word_emb', '--embed_mat_path', type=str,
            default='prep-data/embed_mat.npy',
            help='path of the word embed mat (.npy format)')
parser.add_argument('-embtr', '--embed_trainable', type=bool, default=False,
            help='whether to refine word embedding weights during training')
parser.add_argument('-char_emb', '--char_embedding', type=bool, default=True,
            help='whether to consider char embedding')
parser.add_argument('-ch_embdim', '--char_embed_dim', type=int, default=50,
            help='character embedding dimension')
parser.add_argument('-maxwlen', '--maxwordlen', type=int, default=10,
            help='maximum number of chars in a word')
parser.add_argument('-chfw', '--char_cnn_filter_width', type=int, default=5,
            help='Character level CNN filter width')
parser.add_argument('-bm', '--border_mode', type=str, default=None,
            help='border mode for char CNN')

parser.add_argument('-qt', '--qtype', type=str, default='wh2',
            help='type of the question type representation')
parser.add_argument('-hdim', '--hidden_dim', type=int, default=150)
parser.add_argument('-rnnt', '--rnn_type', type=str, default='lstm',
            help='Type of the building block RNNs')
parser.add_argument('-dop', '--dropout_rate', type=float, default=0.3,
            help='Dropout rate')
parser.add_argument('-istr', '--is_training', type=bool, default=True,
            help='Is it a training script?')
parser.add_argument('-ne', '--num_epoch', type=int, default=10,
            help='number of training epochs')
parser.add_argument('-trbs', '--training_batch_size', type=int, default=60,
            help='Training batch size')
parser.add_argument('-cutctx', '--cut_context', type=bool, default=True,
            help='cut the context for faster training')
parser.add_argument('-adptlr', '--adapt_lr', type=bool, default=True,
                        help='whether to reduce learning rate')
parser.add_argument('-pbs', '--predict_batch_size', type=int, default=4,
            help='Prediction batch size')

# experiment directory
parser.add_argument('-exp', '--baseexp', type=str, default='exp-newsqa',
            help='name of the experiment directory')

args = parser.parse_args()