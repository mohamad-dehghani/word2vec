from gensim.models import Word2Vec
import numpy as np
import codecs
import pandas as pd
import re

df = pd.read_excel('book4.xlsx', header=None)
df.columns = ['comment']

df['embedding'] = ''

# preprocess
from parsivar import Normalizer
my_normalizer = Normalizer()

def normalize(txt):
    return my_normalizer.normalize(txt)

df['normalized_txt'] = df['comment'].apply(normalize)

from parsivar import Tokenizer
my_tokenizer = Tokenizer()

def divide_to_tokens(tmp_text):
    words = my_tokenizer.tokenize_words(my_normalizer.normalize(tmp_text))
    return words

def divide_to_sents(tmp_text):
    sents = my_tokenizer.tokenize_sentences(my_normalizer.normalize(tmp_text))
    return sents

corpus = []
for index, row in df.iterrows():
    tmp = re.sub(r'[?|$|.|!|-|_|:|,|،|)|(]', r'', row['comment'])
    word_list = divide_to_tokens(tmp)
    corpus.append(word_list)
    
model = Word2Vec(sentences = corpus,
                 size = 30,
                 window = 3,
                 min_count = 1,
                 workers = 1,
                 iter = 100,
                 sg = 1)    

words = list(model.wv.vocab)

for index, row in df[:].iterrows():    
    tmp = re.sub(r'[?|$|.|!|-|_|:|,|،|)|(]', r'', row['comment'])
    words = divide_to_tokens(tmp)
    tt = []
    for w in words:
        try:
            tt.append(list(model.wv[w]))
        except:
            print(w)
            
    df.at[index, 'embedding'] = tt

df.to_excel('result.xlsx', index=False)
df.to_pickle("result.pkl")
