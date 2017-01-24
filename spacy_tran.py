#
# Process transcripts under corrected dir
#
# - loop through all .txt files
# - grab file name w/o .txt to be video id
# - obtain sentences
# - repeat 10 times sample 1/3 sentences without replacement
# - output a new DF

from __future__ import print_function
import os
import spacy
import random

nlp = spacy.load('en')

def sample_sents(sents):
    SENT_NUM = int(len(sents)/3)
    cols = []
    for _ in range(SEG_NUM):
        indices = random.sample(range(len(sents)), SENT_NUM) # http://bit.ly/2j85fvj
        cols.append(' '.join([sents[i] for i in sorted(indices)]))
    return(cols)

SEG_NUM = 10
cols = ['col'+str(x) for x in range(SEG_NUM)]
txt_dir = './corrected/'
_out = open('csv/sents.tsv', 'w')
_out.write('\t'.join(['vid'] + cols) + '\n')
for fname in os.listdir(txt_dir):
    if fname.endswith('.txt'):
        vid = fname[:-4] # remove .txt
        with open(txt_dir + fname, 'r') as _txt:
            lns = _txt.readlines()
            text = unicode('\n'.join(lns), 'utf-8')
            doc = nlp(text, parse=True)
            sents_lst = []
            for sent in doc.sents:
                sents_lst.append(str(sent))
            segs = sample_sents(sents_lst)
        _out.write('\t'.join([vid] + segs[0:SEG_NUM]) + '\n')
_out.close()
