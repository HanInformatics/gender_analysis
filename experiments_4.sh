#!/bin/sh

echo "skipgram word2vec"

python3 gen_skipg_word2vec.py 여성남성2008.token.txt
python3 gen_skipg_word2vec.py 여성남성2008.txt
python3 gen_skipg_word2vec.py 여성남성2018.token.txt
python3 gen_skipg_word2vec.py 여성남성2018.txt

