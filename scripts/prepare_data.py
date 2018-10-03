#!/usr/bin/python

import sys
import xml.etree.ElementTree as ET
from nltk.corpus import wordnet as wn
from wordfreq import word_frequency

def parse_xml(infile):
    tree = ET.parse(infile)
    root = tree.getroot()
    #print(root.tag, root.attrib)
    sentences = []
    for child in root:
        sent = []
        for gradc in child:
            sent.append(gradc.text)
            #print(gradc.tag, gradc.text, gradc.attrib)
        sentences.append(sent)
    return sentences

def parse_anno(infile, replace=False):
    annotations = []
    high_freq, low_freq = 0, 0
    with open(infile) as inf:
        for line in inf:
            elems = line.strip().split('\t')
            elems[0] = int(elems[0].split('_')[2])
            if replace:
                elems[1] = senset_2_replacement(elems[1])
                elems[2] = senset_2_replacement(elems[2])
            else:
                elems[1] = elems[1].split('%')[0] 
                elems[2] = elems[2].split('%')[0] 
            w1f = word_frequency(elems[1], 'en')
            w2f = word_frequency(elems[2], 'en')
            high_freq +=  w1f > w2f 
            low_freq +=  w1f < w2f 
            annotations.append(elems)
            print(w1f, w2f)
    print('the relative word frequencies:', high_freq, low_freq)
    return annotations

def senset_2_replacement(key):
    ss = None
    if ';' in key:
        key = key.split(';')[0]
    try: 
        ss = wn.lemma_from_key(key).synset()
    except:
        print('Cannot find %s in wordnet!' % key)    
        pass
    if ss:
        i = 0
        for lemma in ss.lemma_names():
            if i == 1:
                return ' '.join(lemma.split('_'))
            i += 1
        #print('%s only have 1 word in the synset! %s' % (key, ss.lemma_names()))
    return key.split('%')[0] 

if __name__ == '__main__':
    if 'homographic' in sys.argv[1]:
        homo = True
    else:
        homo = False
    
    sentences = parse_xml(sys.argv[1])
    annotations = parse_anno(sys.argv[2], homo)
    out_file = sys.argv[3]
    context = int(sys.argv[4])
    assert len(sentences) == len(annotations)
    with open(out_file+'.true', 'w') as tf, open(out_file+'.hypo', 'w') as hf:
      if homo:
        of = open(out_file+'.orig', 'w')
      for sent, anno in zip(sentences, annotations):
        sent = [w.lower() for w in sent]
        lower = max(0, anno[0]-1-context)
        upper = min(len(sent)+1, anno[0]+context)
        if not homo:
            try:
                assert sent[anno[0]-1].startswith(anno[1])
            except:
                print (sent, sent[anno[0]-1], anno)
                continue
            sent[anno[0]-1] = '<'+sent[anno[0]-1]+'>'
            newsent = ['<'+anno[2]+w[len(anno[1])+1:] if (i == anno[0]-1) else w for i, w in enumerate(sent)]
        else:
            of.write(' '.join(sent[lower:upper]) + '\n')
            sent[anno[0]-1] = '<'+anno[1]+'>'
            newsent = ['<'+anno[2]+'>' if (i == anno[0]-1) else w for i, w in enumerate(sent)]
        #print(lower, upper, anno[0])
        tf.write(' '.join(sent[lower:upper]) + '\n')
        hf.write(' '.join(newsent[lower:upper]) + '\n')

