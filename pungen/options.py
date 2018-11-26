def add_scorer_args(parser):
    group = parser.add_argument_group('Scoring functions')
    group.add_argument('--lm-path', help='pretrained LM for scoring')
    group.add_argument('--word-counts-path', help='vocab word counts for the unigram model')
    group.add_argument('--oov-prob', type=float, default=0.03, help='oov probability for smoothing')
    group.add_argument('--scorer', choices=['random', 'surprisal', 'goodman', 'learned'], default='random', help='type of scorer')
    group.add_argument('--learned-scorer-weights', help='path to scikit-learn LR model')
    group.add_argument('--learned-scorer-features', help='path to LR model feature names')
    group.add_argument('--local-window-size', type=int, default=2, help='window size for computing local surprisal')

def add_editor_args(parser):
    group = parser.add_argument_group('Editing')
    group.add_argument('--type-dict-path', default='models/types.json', help='JSON file of word types')
    group.add_argument('--num-topic-words', type=int, default=100, help='number of neighbors to predict for pun word')
    parser.add_argument('--skipgram-model', nargs=2, help='pretrained skipgram model [vocab, model]')
    parser.add_argument('--skipgram-embed-size', type=int, default=300, help='word embedding size in skipgram model')
    parser.add_argument('--distance-to-pun-word', type=int, default=5, help='minimum distance of editted word to pun word')

def add_retriever_args(parser):
    group = parser.add_argument_group('Retrieval')
    group.add_argument('--doc-file', nargs='+', help='training corpus')
    group.add_argument('--retriever-model', default='models/retriever.pkl', help='retriever model path')
    group.add_argument('--overwrite-retriever-model', action='store_true', help='overwrite existing retriever model; rebuild from doc_file')
    group.add_argument('--num-candidates', type=int, default=500, help='number of sentences to retrieve')
    group.add_argument('--num-templates', type=int, default=10, help='number of maximum pun templates to return')
    group.add_argument('--pos-threshold', type=float, default=0., help='pun word must occur after the thresholding position [0, 1] in a sentence')

def add_generic_args(parser):
    parser.add_argument('--outdir', default='./results')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--pun-freq-threshold', type=int, default=100, help='only process pun/alternative words whose frequences are larger than the threshold')
