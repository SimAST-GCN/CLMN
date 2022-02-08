import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader,WeightedRandomSampler
from gensim.models.word2vec import Word2Vec
import json
import random
import gensim
import numpy as np
import pandas as pd
import os,time
import warnings
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from sklearn import metrics
from sklearn.utils.class_weight import compute_class_weight 
from sklearn.metrics import matthews_corrcoef
import logging
import sys
from openpyxl import Workbook,load_workbook

from data_iter import DPDataset
from model import Code_Encoder
from eval import CLloss

warnings.filterwarnings('ignore')


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
# 设置随机数种子
#set_seed(42)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')


def run(p,gcnn):
    word2vec = Word2Vec.load('./dataset/'+p+'/node_w2v_512').wv
    MAX_TOKENS = word2vec.syn0.shape[0]

    train_path = './dataset/'+p+'/programs.pkl'
    train_dataset = DPDataset(train_path, MAX_TOKENS)

    batch_size = 64  #actual is 128

    train_loader = DataLoader(dataset = train_dataset,batch_size = batch_size,shuffle = True)

    epochs = 40  #40 when i use, 10 maybe ok?
    #learning_rate = 0.0002

    EMBEDDING_DIM = word2vec.syn0.shape[1]
    embeddings = np.zeros((MAX_TOKENS + 1, EMBEDDING_DIM), dtype="float32")
    embeddings[:word2vec.syn0.shape[0]] = word2vec.syn0
    embeddings = torch.tensor(embeddings).to(device)

    model = Code_Encoder(embeddings,MAX_TOKENS+1,gcnn).to(device)
    
    criterion = CLloss(device = device) #
    
    #criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adamax(model.parameters()) #, lr=learning_rate  ,weight_decay=0.00001
    #optimizer = torch.optim.Adam(model.parameters(),lr=0.001,weight_decay=0.00001)#weight_decay=0.00001  #adamax 0.0002
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.00001)

    total_step = len(train_loader)
    for epoch in range(epochs):
        logging.info("train epoch: "+str(epoch))
        model.train()
        start_time = time.time()
        for _,data in enumerate(train_loader):
            code,graph = data
            graph = graph.to(device).float()
            code = code.to(device).int()
            # Forward pass
            output1 = model(code,graph)
            output2 = model(code,graph)
            outputs = torch.cat((output1,output2),1).reshape(batch_size*2,-1)
            loss = criterion(outputs)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (_+1) % 10 == 0:
                end_time = time.time()
                logging.info('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Time: {}' 
                    .format(epoch+1, epochs, _+1, total_step, loss.item(), end_time - start_time))
                start_time = time.time()
    
    torch.save(model, path + 'pre_train_'+str(gcnn)+'.pth')

        
if __name__ == '__main__':
    project = sys.argv[1]
    gcnn = sys.argv[2]

    if not os.path.exists('./result'):os.mkdir('./result')
    path = './result/'+project+'/'
    if not os.path.exists(path):
        os.mkdir(path)

    logging.basicConfig(level=logging.INFO,
                        filename='./result/'+project+'/pre_train_'+gcnn+'.log',
                        filemode='a',
                        format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    run(project,int(gcnn))