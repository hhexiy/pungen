def add_scorer_args(parser):
    group = parser.add_argument_group('Scoring functions')
    group.add_argument('--lm-path', help='pretrained LM for scoring')
    group.add_argument('--word-counts-path', help='vocab word counts for the unigram model')
    group.add_argument('--oov-prob', type=float, default=0.03, help='oov probability for smoothing')

def add_editor_args(parser):
    group = parser.add_argument_group('Editing')
    group.add_argument('--type-dict-path', default='models/types.json', help='JSON file of word types')
    group.add_argument('--num-topic-words', type=int, default=100, help='number of neighbors to predict for pun word')
    parser.add_argument('--skipgram-model', nargs=2, help='pretrained skipgram model [vocab, model]')
    parser.add_argument('--skipgram-embed-size', type=int, default=300, help='word embedding size in skipgram model')

def add_retriever_args(parser):
    group = parser.add_argument_group('Retrieval')
    group.add_argument('--doc-file', nargs='+', help='training corpus')
    group.add_argument('--retriever-model', default='models/retriever.pkl', help='retriever model path')
    group.add_argument('--overwrite-retriever-model', action='store_true', help='overwrite existing retriever model; rebuild from doc_file')
    group.add_argument('--num-candidates', type=int, default=500, help='number of sentences to retrieve')
    group.add_argument('--num-templates', type=int, default=10, help='number of maximum pun templates to return')
