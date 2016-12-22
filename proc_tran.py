#
# Process transcripts under corrected dir
#
# - loop through all .txt files
# - grab file name w/o .txt to be video id
# - split text content into 10 segments
# - output a new DF
from __future__ import print_function
import os
import textwrap
import pandas as pd

SEG_NUM = 10
cols = ['col'+str(x) for x in range(SEG_NUM)]
txt_dir = './corrected/'
_out = open('csv/all.tsv', 'w')
_out.write('\t'.join(['vid'] + cols) + '\n')
for fname in os.listdir(txt_dir):
    if fname.endswith('.txt'):
        vid = fname[:-4] # remove .txt
        with open(txt_dir + fname, 'r') as _txt:
            lns = _txt.readlines()
            text = ' '.join(lns)
            _size = int(len(text)/SEG_NUM)
            segs = textwrap.wrap(text, _size)
        _out.write('\t'.join([vid] + segs[0:SEG_NUM]) + '\n')
_out.close()
