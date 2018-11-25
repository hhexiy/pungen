import sys, os
import random

def load_keywords(keyset, infile):
    with open(infile) as inf:
        for line in inf:
            keyset.add(line.strip())
    return keyset

def load_vocab(infile, threshold):
    vocab_dict=dict()
    with open(infile) as inf:
        for line in inf:
            elems = line.strip()
            vocab_dict[elems] = threshold
    return vocab_dict

def get_vocab(infile, vocab_dict=dict()):
    with open(infile) as inf:
        for line in inf:
            elems = line.strip().split()
            for el in elems:
                if el not in vocab_dict:
                    vocab_dict[el] = 0
                vocab_dict[el] += 1
    return vocab_dict

def process_file_fb(infile, keyset, vocab, max_sample=3, threshold=10):
    dirname = os.path.dirname(infile)
    basename = os.path.basename(infile)
    print('base name:', basename)
    if not os.path.exists(os.path.join(dirname, '1backward')):
        os.mkdir(os.path.join(dirname, '1backward'))
    if not os.path.exists(os.path.join(dirname, '2forward')):
        os.mkdir(os.path.join(dirname, '2forward'))
    with open(infile) as inf, open(os.path.join(dirname, '1backward', basename+'.in'), 'w') as bsout, \
            open(os.path.join(dirname, '1backward', basename+'.out'), 'w') as btout, \
            open(os.path.join(dirname, '2forward', basename+'.in'), 'w') as fsout, \
            open(os.path.join(dirname, '2forward', basename+'.out'), 'w') as ftout :
        for line in inf:
            elems = line.strip().split()
            kwset = set(elems) & keyset
            idxs = [elems.index(kw) for kw in kwset]
            if len(idxs) < max_sample:
                try:
                    sid = random.sample(set(range(len(elems)/2, len(elems)-1))-set(idxs), max_sample - len(idxs))
                except:
                    continue
                idxs.extend(sid)
            write_files(elems, idxs, vocab, bsout, btout, fsout, ftout, threshold)
    
def write_files(sentence, idxs, vocab, bsout, btout, fsout, ftout, threshold=10):
    conv_sent = [word if (word in vocab and vocab[word] >= threshold) else '<unk>' for word in sentence]
    for idx in idxs:
        bsout.write(conv_sent[idx] + '\n')
        btout.write(' '.join(conv_sent[idx-1::-1]) + '\n')
        fsout.write(' '.join(conv_sent[:idx+1]) + '\n')
        ftout.write(' '.join(conv_sent[idx+1:]) + '\n')


def print_vocab(outfile, vocab, threshold=10):
    counter = 0
    with open(outfile, 'w') as outf:
        outf.write('<unk>\n')
        outf.write('<s>\n')
        outf.write('</s>\n')
        for k, v in vocab.items():
            if v >= threshold:
                outf.write(k + '\n')
                counter += 1
        print('effective vocab size:', counter)


if __name__ == '__main__':
    infile = sys.argv[1]
    keyword_set = set()
    load_keywords(keyword_set, sys.argv[2])
    print('keyword size:', len(keyword_set))
    load_keywords(keyword_set, sys.argv[3])
    print('keyword size:', len(keyword_set))
    threshold=int(sys.argv[4])
    vocab_file = sys.argv[5]
    if os.path.exists(vocab_file):
        print('loading vocab from:', vocab_file)
        vocab_dict = load_vocab(vocab_file, threshold)
    else:
        vocab_dict = get_vocab(infile) 
        print_vocab(vocab_file, vocab_dict, threshold)
    print('vocab size:', len(vocab_dict))
    process_file_fb(infile, keyword_set, vocab_dict, max_sample=1, threshold=threshold)
