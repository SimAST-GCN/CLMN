#for dataset, java only
import pandas as pd
import javalang
import warnings
import json
from gensim.models.word2vec import Word2Vec
import multiprocessing
import os,sys,warnings
warnings.filterwarnings('ignore')
sys.setrecursionlimit(50000)

import swifter
import numpy as np
import random

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
# 设置随机数种子



if __name__=="__main__":
    proj = sys.argv[1]

    set_seed(42)

    #proj = ['accumulo','ambari','beam','cloudstack','commons-lang','flink','hadoop','incubator-pinot','kafka','lucene-solr','shardingsphere']
    proj = [proj]

    for pj in proj:
        print("dealing :",pj)

        prog_path = './dataset/'+pj+'/'+pj+'.pkl'

        prog = pd.read_pickle(prog_path)

        prog.columns = ['code1','code2','label','cmt','dsp']

        def parse_program(func):
            tokens = javalang.tokenizer.tokenize(func)
            parser = javalang.parser.Parser(tokens)
            tree = parser.parse_member_declaration()
            return tree

        from tree import get_sequence as func
        def trans_to_sequences(lst):
            cd = ''.join(lst)
            ast = parse_program(cd)
            sequence = []
            father = []
            func(ast, sequence, father, 0, cd)
            if len(sequence)>1000:
                sequence = sequence[:1000]
                father = father[:1000]
            return sequence, np.array(father,dtype = 'int32')

        print('change code to code and graph.')
        prog[['code1','graph1']] = prog['code1'].swifter.allow_dask_on_strings(enable=True).apply(trans_to_sequences).swifter.apply(pd.Series)
        prog[['code2','graph2']] = prog['code2'].swifter.allow_dask_on_strings(enable=True).apply(trans_to_sequences).swifter.apply(pd.Series)


        #word2vec
        print('train word to vec.')
        corpus = prog['code1'] + prog['code2']
        w2v = Word2Vec(corpus, size=512, sg=1, window=5 ,min_count = 3, workers=multiprocessing.cpu_count()) # max_final_vocab=3000 tmp ignore
        w2v.save('./dataset/'+pj+'/node_w2v_512')

        word2vec = w2v.wv
        vocab = word2vec.vocab
        max_token = word2vec.syn0.shape[0]

        def index(tmp):
            result = []
            for k in tmp:
                result.append(vocab[k].index if k in vocab else max_token)
            while len(result)<1000:result.append(max_token)
            return np.array(result, dtype = 'int32')


        print('code to idx.')
        prog['code1'] = prog['code1'].swifter.apply(index)
        prog['code2'] = prog['code2'].swifter.apply(index)

        #for pre train data
        #pre_train = list(zip(prog['code1'].tolist(),prog['graph1'].tolist())) + list(zip(prog['code2'].tolist(),prog['graph2'].tolist()))
        #pt = pd.DataFrame(pre_train)
        #pt.columns = ['code','graph']
        #print('save programs')
        #pt.to_pickle('dataset/'+pj+'/programs.pkl')

        ratio = '3:1:1'
        data_num = len(prog)
        ratios = [int(r) for r in ratio.split(':')]
        train_split = int(ratios[0]/sum(ratios)*data_num)
        val_split = train_split + int(ratios[1]/sum(ratios)*data_num)

        prog = prog.sample(frac=1, random_state=666)
        train = prog.iloc[:train_split]
        train.reset_index(drop=True, inplace=True)
        dev = prog.iloc[train_split:val_split]
        dev.reset_index(drop=True, inplace=True)
        test = prog.iloc[val_split:]
        test.reset_index(drop=True, inplace=True)
        
        trainp = 'dataset/'+pj+'/train.pkl'
        testp = 'dataset/'+pj+'/test.pkl'
        devp = 'dataset/'+pj+'/valid.pkl'
        #pp = 'data/'+p+'/token.pkl'
        #s.to_pickle(pp)
        print('save train')
        train.to_pickle(trainp)
        print('save test')
        test.to_pickle(testp)
        print('save dev')
        dev.to_pickle(devp)

        #for pre train data
        prog = train
        pre_train = list(zip(prog['code1'].tolist(),prog['graph1'].tolist())) + list(zip(prog['code2'].tolist(),prog['graph2'].tolist()))
        pt = pd.DataFrame(pre_train)
        pt.columns = ['code','graph']
        print('save programs')
        pt.to_pickle('dataset/'+pj+'/programs.pkl')
