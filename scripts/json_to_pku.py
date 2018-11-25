import json
import sys

def data_to_keywords(infile, outfile):
    with open(infile) as inf, open(outfile, 'w') as outf:
        data = json.load(inf)
        for line in data:
            #print(line)
            outf.write(line['pun_word']+'\n') 
            outf.write(line['alter_word']+'\n') 

def json_to_pmi(infile, outfile):
    with open(infile) as inf, open(outfile, 'w') as outf:
        data = json.load(inf)
        for line in data:
            #print(line)
            outf.write(line['pun_word']+'\t'+' '.join(line['pun_topic_words']) + '\t' + ' '.join(['0.5'] * len(line['pun_topic_words']))+'\n')
            outf.write(line['alter_word']+'\t'+' '.join(line['alter_topic_words']) + '\t' + ' '.join(['0.5'] * len(line['alter_topic_words']))+'\n')

if __name__ == "__main__":
    #data_to_keywords(sys.argv[1], sys.argv[2])
    json_to_pmi(sys.argv[1], sys.argv[2])
