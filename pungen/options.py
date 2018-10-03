def add_scorer_args(parser):
    group = parser.add_argument_group('Scoring functions')
    group.add_argument('--lm-path', help='pretrained LM for scoring')
    group.add_argument('--word-counts-path', help='vocab word counts for the unigram model')
    group.add_argument('--oov-prob', type=float, default=0.03, help='oov probability for smoothing')

def add_type_recognizer_args(parser):
    group = parser.add_argument_group('Word typing')
    group.add_argument('--type-dict-path', default='model/types.json', help='JSON file of word types')
