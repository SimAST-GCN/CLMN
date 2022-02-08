from torch.utils.data import Dataset, DataLoader, Sampler
import random,math
import pandas as pd
import numpy as np
import copy

random.seed(42)
np.random.seed(42)

#Dataset
class MyClassBalanceDataset(Dataset):
    def __init__(self, root):
        super(MyClassBalanceDataset, self).__init__()
        labels = []
        old = []
        new = []
        go = []
        gn = []
        source = pd.read_pickle(root)
        
        def graph(connection):
            lenx = len(connection)
            tmp = np.zeros((1000,1000),dtype='bool_')
            for i in range(lenx):
                tmp[i][i]=True
                tmp[i][connection[i]] = tmp[connection[i]][i] = True
            return tmp
        source['go'] = source['go'].apply(graph)
        source['gn'] = source['gn'].apply(graph)

        self.len = len(source)
        self.label = source['label'].tolist()
        self.old = source['old'].tolist()
        self.new = source['new'].tolist()
        self.go = source['go'].tolist()
        self.gn = source['gn'].tolist()
        self.one = []
        self.zero = []
        for i in range(len(self.label)):
            if self.label[i]==1:self.one.append(i)
            else :self.zero.append(i)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.old[index],self.go[index],self.new[index],self.gn[index],self.label[index]

class MyBatchSampler(Sampler):
    def __init__(self, data_source, batch_size, class_weight):
        super(MyBatchSampler, self).__init__(data_source)
        random.seed(20)
        self.data_source = data_source
        assert isinstance(class_weight, list)
        assert 1 - sum(class_weight) < 1e-5
        self.batch_size = batch_size

        _num = len(class_weight)
        number_in_batch = {i: 0 for i in range(_num)}
        for c in range(_num):
            number_in_batch[c] = math.floor(batch_size * class_weight[c])
        _remain_num = batch_size - sum(number_in_batch.values())
        number_in_batch[random.choice(range(_num))] += _remain_num
        self.number_in_batch = number_in_batch
        self.offset_per_class = {i: 0 for i in range(_num)}
        #print(f'setting number_in_batch: {number_in_batch}')
        #print('my sampler is inited.')

    def __iter__(self):
        #print('======= start __iter__ =======')
        batch = []
        i = 0
        while i < len(self):
            for c, num in self.number_in_batch.items():
                start = 0
                end = 0
                if c==0:
                    end = len(self.data_source.zero)
                    for _ in range(num):
                        idx = start + self.offset_per_class[c]
                        if idx >= end:
                            self.offset_per_class[c] = 0
                        idx = start + self.offset_per_class[c]
                        batch.append(self.data_source.zero[idx])
                        #batch.append(0)
                        self.offset_per_class[c] += 1
                else: 
                    end = len(self.data_source.one)
                    for _ in range(num):
                        idx = start + self.offset_per_class[c]
                        if idx >= end:
                            self.offset_per_class[c] = 0
                        idx = start + self.offset_per_class[c]
                        batch.append(self.data_source.one[idx])
                        #batch.append(0)
                        self.offset_per_class[c] += 1

            assert len(batch) == self.batch_size
            # random.shuffle(batch)
            yield batch
            batch = []
            i += 1

    def __len__(self):
        return len(self.data_source) // self.batch_size



class MyDataset(Dataset):
    def __init__(self,file_path):
        labels = []
        old = []
        new = []
        go = []
        gn = []
        source = pd.read_pickle(file_path)

        def graph(connection):
            lenx = len(connection)
            tmp = np.zeros((1000,1000),dtype='bool_')
            for i in range(lenx):
                tmp[i][i]=True
                tmp[i][connection[i]] = tmp[connection[i]][i] = True
            return tmp
        source['go'] = source['go'].apply(graph)
        source['gn'] = source['gn'].apply(graph)

        self.len = len(source)
        self.label = source['label'].tolist()
        self.old = source['old'].tolist()
        self.new = source['new'].tolist()
        self.go = source['go'].tolist()
        self.gn = source['gn'].tolist()

    def __len__(self):
        return self.len

    def __getitem__(self,index):
        return self.old[index],self.go[index],self.new[index],self.gn[index],self.label[index]



class CMDataset(Dataset):   #generate data for cl
    def __init__(self,file_path,max_token):
        source = pd.read_pickle(file_path)

        code = source['code'].tolist()
        graph = source['graph'].tolist()

        #random delete one word
        def del_one_word(code,graph,max_token):
            pos = np.random.randint(0,len(graph))
            code[pos] = max_token
            graph[pos] = -1
            return code,graph

        fcode = []
        fgraph = []
        codee = copy.deepcopy(code)  #deep copy for diff
        graphh = copy.deepcopy(graph)
        for i in range(len(code)):
            fcode.append(code[i])
            fgraph.append(graph[i])
            cd,gh = del_one_word(codee[i],graphh[i],max_token)
            fcode.append(cd)
            fgraph.append(gh)

        prog = pd.DataFrame(list(zip(fcode,fgraph)))
        prog.columns = ['code','graph']

        def graph(connection):
            lenx = len(connection)
            tmp = np.zeros((1000,1000),dtype='bool_')
            for i in range(lenx):
                if connection[i] < 0 : continue
                tmp[i][i]=True
                tmp[i][connection[i]] = tmp[connection[i]][i] = True
            return tmp
        prog['graph'] = prog['graph'].apply(graph)

        self.len = len(prog)
        self.code = prog['code'].tolist()
        self.graph = prog['graph'].tolist()

    def __len__(self):
        return self.len

    def __getitem__(self,index):
        return self.code[index],self.graph[index]


class DPDataset(Dataset):   #generate data for cl, using dropout
    def __init__(self,file_path,max_token):
        source = pd.read_pickle(file_path)

        def graph(connection):
            lenx = len(connection)
            tmp = np.zeros((1000,1000),dtype='bool_')
            for i in range(lenx):
                if connection[i] < 0 : continue
                tmp[i][i]=True
                tmp[i][connection[i]] = tmp[connection[i]][i] = True
            return tmp
        source['graph'] = source['graph'].apply(graph)

        self.len = len(source)
        self.code = source['code'].tolist()
        self.graph = source['graph'].tolist()

    def __len__(self):
        return self.len

    def __getitem__(self,index):
        return self.code[index],self.graph[index]



class MyDatasetCL(Dataset):
    def __init__(self,file_path,max_token):
        source = pd.read_pickle(file_path)

        def graph(connection):
            lenx = len(connection)
            tmp = np.zeros((1000,1000),dtype='bool_')
            for i in range(lenx):
                tmp[i][i]=True
                tmp[i][connection[i]] = tmp[connection[i]][i] = True
            return tmp
        source['graph'] = source['graph'].apply(graph)

        def pad(cd):
            while len(cd) < 1000:
                cd = np.append(cd,max_token)
            return cd

        source['code'] = source['code'].apply(pad)
        code = source['code'].tolist()
        graph = source['graph'].tolist()

        fc = []
        fg = []
        for i in range(len(code)):
            fc.append(code[i])
            fg.append(graph[i])
            fc.append(code[i])
            fg.append(graph[i])

        source = pd.DataFrame(list(zip(fc,fg)))
        source.columns = ['code','graph']

        self.len = len(source)
        self.code = source['code'].tolist()
        self.graph = source['graph'].tolist()

    def __len__(self):
        return self.len

    def __getitem__(self,index):
        return self.code[index],self.graph[index]



class MADataset(Dataset):  #multi_acr dataset , CLMN
    def __init__(self,file_path,tokenizer):
        source = pd.read_pickle(file_path)

        def graph(connection):
            lenx = len(connection)
            tmp = np.zeros((1000,1000),dtype='bool_')
            for i in range(lenx):
                tmp[i][i]=True
                tmp[i][connection[i]] = tmp[connection[i]][i] = True
            return tmp
        source['graph1'] = source['graph1'].apply(graph)
        source['graph2'] = source['graph2'].apply(graph)

        def to_int(lb):
            return int(lb)
        source['label'] = source['label'].apply(to_int)

        self.len = len(source)
        self.label = source['label'].tolist()
        self.code1 = source['code1'].tolist()
        self.code2 = source['code2'].tolist()
        self.graph1 = source['graph1'].tolist()
        self.graph2 = source['graph2'].tolist()
        self.text = source['cmt'].tolist()
        self.text = tokenizer(self.text, padding=True, truncation=True, return_tensors="pt")
        self.input_ids = self.text['input_ids'] 
        self.attention_mask = self.text['attention_mask']
        

    def __len__(self):
        return self.len

    def __getitem__(self,index):
        return self.code1[index],self.graph1[index],self.code2[index],self.graph2[index],self.input_ids[index],self.attention_mask[index],self.label[index]



class SGDataset(Dataset):  #SimAST-GCN
    def __init__(self,file_path):
        source = pd.read_pickle(file_path)

        def graph(connection):
            lenx = len(connection)
            tmp = np.zeros((1000,1000),dtype='bool_')
            for i in range(lenx):
                tmp[i][i]=True
                tmp[i][connection[i]] = tmp[connection[i]][i] = True
            return tmp
        source['graph1'] = source['graph1'].apply(graph)
        source['graph2'] = source['graph2'].apply(graph)

        def to_int(lb):
            return int(lb)
        source['label'] = source['label'].apply(to_int)

        self.len = len(source)
        self.label = source['label'].tolist()
        self.code1 = source['code1'].tolist()
        self.code2 = source['code2'].tolist()
        self.graph1 = source['graph1'].tolist()
        self.graph2 = source['graph2'].tolist()        

    def __len__(self):
        return self.len

    def __getitem__(self,index):
        return self.code1[index],self.graph1[index],self.code2[index],self.graph2[index],self.label[index]



class TBDataset(Dataset):  #multi_acr dataset , TBRNN, DACE 
    def __init__(self,file_path,tokenizer):
        source = pd.read_pickle(file_path)

        def to_int(lb):
            return int(lb)
        source['label'] = source['label'].apply(to_int)

        self.len = len(source)
        self.label = source['label'].tolist()
        self.code1 = source['code1'].tolist()
        self.code2 = source['code2'].tolist()
        self.text = source['cmt'].tolist()
        self.text = tokenizer(self.text, padding=True, truncation=True, return_tensors="pt")
        self.input_ids = self.text['input_ids'] 
        self.attention_mask = self.text['attention_mask']
        

    def __len__(self):
        return self.len

    def __getitem__(self,index):
        return self.code1[index],self.code2[index],self.input_ids[index],self.attention_mask[index],self.label[index]

