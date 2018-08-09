#!/usr/bin/python

import sys
import xml.etree.ElementTree as ET

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

def parse_anno(infile):
    annotations = []
    with open(infile) as inf:
        for line in inf:
            elems = line.strip().split('\t')
            elems[0] = int(elems[0].split('_')[2])
            elems[1] = elems[1].split('%')[0] 
            elems[2] = elems[2].split('%')[0] 
            annotations.append(elems)
    return annotations


if __name__ == '__main__':
    sentences = parse_xml(sys.argv[1])
    annotations = parse_anno(sys.argv[2])
    out_file = sys.argv[3]
    assert len(sentences) == len(annotations)
    with open(out_file+'.true', 'w') as tf, open(out_file+'.hypo', 'w') as hf:
      for sent, anno in zip(sentences, annotations):
        sent = [w.lower() for w in sent]
        try:
            assert sent[anno[0]-1].startswith(anno[1])
        except:
            print (sent, sent[anno[0]-1], anno)
            continue
        newsent = [anno[2]+w[len(anno[1]):] if (i == anno[0]-1) else w for i, w in enumerate(sent)]
        tf.write(' '.join(sent) + '\n')
        hf.write(' '.join(newsent) + '\n')

