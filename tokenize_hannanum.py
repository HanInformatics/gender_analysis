#coding:utf-8

from konlpy.tag import Hannanum
han = Hannanum()

import sys
infile = open(sys.argv[1], 'r', encoding='utf-8')
data = infile.readlines()
infile.close()

ALL=0
stop_tags = ('J', 'E', 'S', 'X', )
for l in data:
    if ALL :
        al = [w +'/'+t for w, t in han.pos(l)]
    else:
        al = [w.replace('"', '').replace('’','') for w,t in han.pos(l) if t not in ('J', 'E', 'S') ]
    print(' '.join(al))

