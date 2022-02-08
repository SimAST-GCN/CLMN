import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
import random

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.kaiming_uniform_(self.weight.data)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameter()
        self.layer_norm = nn.LayerNorm(1024)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        #self.batch_norm = nn.BatchNorm1d(1000) #fixed, corresponding to the max len of the token

    def reset_parameter(self):
        torch.nn.init.kaiming_uniform_(self.weight, a = math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1/math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, text, adj):
        hidden = torch.matmul(text, self.weight)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        output = torch.matmul(adj, hidden)
        output = output / denom
        if self.bias is not None:
            output = self.layer_norm(output + self.bias)
        else:
            output = self.layer_norm(output)
        output = self.relu(output)
        #return output
        return self.dropout(output)


class Code_Encoder(nn.Module):
    def __init__(self, weights, vocab_size, gcnn):
        super(Code_Encoder, self).__init__()
        self.embedding_size = 512
        self.hidden_size = 512
        self.embedding=nn.Embedding(vocab_size,self.embedding_size)
        self.embedding.weight.data.copy_(weights)
        self.bigru1 = nn.GRU(self.embedding_size,self.hidden_size,num_layers=1,bidirectional=True,batch_first=True)  #,batch_first=True  ,batch_first=True
        self.gc1 = nn.ModuleList([GraphConvolution(2*self.hidden_size,2*self.hidden_size) for i in range(gcnn)])
        self.fc = nn.Linear(in_features = 2*self.hidden_size,out_features = 256)
        #self.dropout = nn.Dropout(0.1)
    
    def forward(self, code, graph):
        eo = self.embedding(code)
        oo,_ = self.bigru1(eo)

        o = oo
        for gcn in self.gc1:
            o = gcn(o,graph)

        alpha_mat = torch.matmul(o, oo.transpose(1, 2))
        alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)

        o = torch.matmul(alpha, oo).squeeze(1)

        ot = self.fc(o)
        return ot


class CLMN(nn.Module):
    def __init__(self, code_encoder, text_encoder):
        super(CLMN, self).__init__()
        self.code_encoder = code_encoder
        self.text_encoder = text_encoder
        self.svd = nn.Linear(in_features = 1024,out_features = 256)
        self.fc = nn.Linear(in_features = 512,out_features = 2)

    
    def forward(self, code_origin, graph_origin, code_revised, graph_revised, text_ids, text_mask):
        origin_emb = self.code_encoder(code_origin, graph_origin)
        revised_emb = self.code_encoder(code_revised, graph_revised)
        text_emb = self.text_encoder(text_ids,text_mask).pooler_output
        text_emb = self.svd(text_emb)
        code_dist = torch.abs(torch.add(origin_emb, -revised_emb)) #shape, batch*256
        combine_rep = torch.cat((code_dist, text_emb),1)
        output = self.fc(combine_rep)
        return output

