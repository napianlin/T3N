#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import preprocessor as p
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.optim as optim
import torch.nn.functional as F
import pickle as pkl
from collections import defaultdict
import pandas as pd
import os
import numpy as np
import json
import matplotlib.pyplot as plt
import random
from tqdm import tqdm, tqdm_notebook
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score
from collections import Counter
import spacy
from tqdm import tqdm, tqdm_notebook, tnrange
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix


# In[ ]:


class Node:
    def __init__(self,uid,tid,time_stamp,label):
        self.children = {}
        self.childrenList = []
        self.num_children = 0
        self.tid = tid
        self.uid = uid
        self.label = label
        self.time_stamp = time_stamp
    
    def add_child(self,node):
        if node.uid not in self.children:
            self.children[node.uid] = node
            self.num_children += 1
        else:
            self.children[node.uid] = node
        self.childrenList = list(self.children.values())


# In[ ]:


class Tree:
    def __init__(self,root):
        self.root = root
        self.tweet_id = root.tid
        self.uid = root.uid
        self.height = 0
        self.nodes = 0
    
    def show(self):
        queue = [self.root,0]
        
        while len(queue) != 0:
            toprint = queue.pop(0)
            if toprint == 0:
                print('\n')
            else:
                print(toprint.uid,end=' ')
                queue += toprint.children.values()
                queue.append(0)
                
    def insertnode(self,curnode,parent,child):
        if curnode.uid == parent.uid:
            curnode.add_child(child)
            return 1

        elif parent.uid in curnode.children:
            s = self.insertnode(curnode.children[parent.uid],parent,child)
            return 2
        else:
            for node in curnode.children:
                s = self.insertnode(curnode.children[node],parent,child)
                if s == 2:
                    break


# In[ ]:


def loadPklFileNum(datapath,incSize,fileNum):
    
    with open(datapath+str(incSize)+'inc_'+str(fileNum)+'.pickle', 'rb') as handle:
        twitTrees = pkl.load(handle)
    return twitTrees


# In[ ]:


def loadTreeFilesOfIncrement(datapath,incSize):
    twittertrees = {}
    
    files = [x for x in os.listdir(t15Datapath) if str(incSize)+'inc' in x]
    
    for file in tqdm(files):
        with open(datapath+file,'rb') as handle:
            partialTrees = pkl.load(handle)
        twittertrees.update(partialTrees)
        
    return twittertrees


# In[ ]:


dataset = 'twitter16'

if dataset == 'twitter15':
    get_ipython().run_line_magic('run', '../twitter15/twitter15_text_processing.ipynb')
    t15Datapath = '../twitter15/pickledTrees/'
    twitter15_trees = loadTreeFilesOfIncrement(t15Datapath,20)
    get_ipython().run_line_magic('run', '../twitter15/userdata_parser.ipynb')
    
if dataset == 'twitter16':
    get_ipython().run_line_magic('run', '../twitter16/twitter16_text_processing.ipynb')
    t15Datapath = '../twitter16/pickledTrees/'
    twitter15_trees = loadTreeFilesOfIncrement(t15Datapath,20)
    get_ipython().run_line_magic('run', '../twitter16/userdata_parser.ipynb')


# In[ ]:


for key in tqdm(userVects):
    userVects[key] = userVects[key].float()

userVects = defaultdict(lambda:torch.tensor([1.1100e+02, 1.5000e+01, 0.0000e+00, 7.9700e+02, 4.7300e+02, 0.0000e+00,
        8.3326e+04, 1.0000e+00]),userVects)


# In[ ]:


get_ipython().run_line_magic('run', './textEncoders.ipynb')
get_ipython().run_line_magic('run', './temporal_tree_model.ipynb')


# In[ ]:


if torch.cuda.is_available():
    device = 'cuda:2'
    device = 'cpu'
else:
    device = 'cpu'


# In[ ]:


labelMap = {'true':0,'false':1,'unverified':2,'non-rumor':3}


# In[ ]:


epochs = 10
X = []
y = []
X_text = []

for tid in twitter15_trees:
        if tid in twitter15_trees and tid in twitter15_labels:
            X.append(tuple((twitter15_trees[tid],twitter15_text[tid])))
            y.append(labelMap[twitter15_labels[tid]])
            X_text.append(twitter15_text[tid])
            
x_train,x_test,y_train,y_test = train_test_split(X,y,random_state=42)


# In[ ]:


f = lambda m, n: [(i*n//m + n//(2*m)) for i in range(m)]


# In[ ]:


class jointModel(nn.Module):
    def __init__(self, treeEncoderType, textEncoderType, pretrainFiles, textparams, treeparams, X, y, device):
        super(jointModel, self).__init__()
        if textEncoderType == 'rnn':
            self.textEncoderModel = TextEncoder(textEncoderType,textparams,X,y,device)
            textparams['hidden_dim'] = textparams['hidden_dim']*self.textEncoderModel.textEncoder.numDirs
            
        if textEncoderType == 'bert':
            self.textEncoderModel = BertTextEncoder(textEncoderType,{},X,y,device)
            textparams['hidden_dim'] = 768
        
        if textEncoderType == 'attn':
            self.textEncoderModel = AttentionTextEncoder(textEncoderType,textparams,X,y,device)
            textparams['hidden_dim'] = textparams['hidden_dim']*self.textEncoderModel.textEncoder.numDirs*self.textEncoderModel.seq_dim
            
        if treeEncoderType == 'standard':
            self.treeEncoderModel = treeEncoder(treeparams['cuda'],treeparams['in_dim'],treeparams['mem_dim'],treeparams['userVects'],treeparams['labels'],treeparams['labelMap'],treeparams['criterion'],device)
        if treeEncoderType == 'decay':
            self.treeEncoderModel = decayTreeEncoder(treeparams['cuda'],treeparams['in_dim'],treeparams['mem_dim'],treeparams['userVects'],treeparams['labels'],treeparams['labelMap'],treeparams['criterion'],device)
        
        if treeEncoderType == 'temporal':
            self.treeEncoderModel = lstmTreeEncoder(treeparams['cuda'],treeparams['in_dim'],treeparams['mem_dim'],treeparams['userVects'],treeparams['labels'],treeparams['labelMap'],treeparams['criterion'],device)
        
        if treeEncoderType == 'temporaldecay':
            self.treeEncoderModel = temporalDecayTreeEncoder(treeparams['cuda'],treeparams['in_dim'],treeparams['mem_dim'],treeparams['userVects'],treeparams['labels'],treeparams['labelMap'],treeparams['criterion'],device)
        
        if pretrainFiles:
            textcheckpoint = torch.load(pretrainFiles['text'])
            self.textEncoderModel.textEncoder.load_state_dict(textcheckpoint['state_dict'])
            
            treecheckpoint = torch.load(pretrainFiles['tree'])
            self.treeEncoderModel.load_state_dict(treecheckpoint['state_dict'])
        
        mem_dim = treeparams['mem_dim'] + textparams['hidden_dim']
        
        self.fc = nn.Linear(mem_dim,4)    
            
    def forward(self,tree,text):
        treeVec = self.treeEncoderModel(tree)
        treeVec = treeVec[0][1].reshape(-1)
        
        self.textEncoderModel.textEncoder = self.textEncoderModel.textEncoder.to('cpu')
        textVec = self.textEncoderModel(text)
        textVec = textVec.reshape(-1)
#         print(treeVec.shape)
#         print(textVec.shape)
        combVec =  torch.cat((treeVec,textVec))
#         combVec = textVec
        out = self.fc(combVec)
        return out


# In[ ]:


def trainModel(model,modelname):
    optimizer = torch.optim.Adagrad(model.treeEncoderModel.parameters(),0.01)
    
    bertoptimizer = AdamW(model.textEncoderModel.parameters(),
                   lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                   eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                 )
    
#    optimizer = torch.optim.Adagrad(model.parameters(),0.01)
    criterion = nn.CrossEntropyLoss()

    maxAcc = 0
    count = 0
    
    for i in range(5):
        for treeSet, text in tqdm(x_train):
            tree = treeSet[-1]
            count += 1
            optimizer.zero_grad()
            bertoptimizer.zero_grad()
            
            pred = model(tree.root,text)
            
            label = Variable(torch.tensor(labelMap[treeSet[0].root.label]).reshape(-1).to(device))
            loss = criterion(pred.reshape(1,4),label)
#             print(loss)
    
#                 print('opt')
            loss.backward()
            optimizer.step()
            bertoptimizer.step
            
        print('train loss: ',loss.item())   
        preds = []
        labels = []

        allLabels = []
        allPreds = []
        
        with torch.no_grad():
            for valSet, text in tqdm(x_test):
                finalTree = valSet[-1]

                predicted = model(finalTree.root,text)
                preds.append(predicted)
        #         print(predicted)
                predicted =  torch.softmax(predicted,0)
                predicted = torch.max(predicted, 0)[1].cpu().numpy().tolist()

                labels.append(labelMap[finalTree.root.label])

                allLabels.append(labelMap[finalTree.root.label])
                allPreds.append(predicted)

            predTensor = torch.stack(preds)
            labelTensor = torch.tensor(labels).to(device)

            print(allLabels,allPreds)

            loss = criterion(predTensor.reshape(-1,4), labelTensor.reshape(-1))

        cr = classification_report(allLabels,allPreds,output_dict=True)
        cr['loss'] = loss.item()
        cr['Acc'] = accuracy_score(allLabels,allPreds,)

        if cr['Acc'] > maxAcc:
            maxAcc = cr['Acc']
            torch.save({'state_dict': model.state_dict()}, './twitter16_results/'+modelname+'.pth')


        print('val loss: ',cr['loss'])
        print(cr['Acc'])

        with open('./twitter16_results/'+modelname+'.json', 'a') as fp:
            json.dump(cr, fp)
            fp.write('\n')

# -----------------------------------------------------------------------------------------------------------------------

def trainTemporalModel(model,modelname):
    optimizer = torch.optim.Adagrad(model.treeEncoderModel.parameters(),0.01)
    
    bertoptimizer = AdamW(model.textEncoderModel.parameters(),
                 lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
               )
    
#    optimizer = torch.optim.Adagrad(model.parameters(),0.01)
    criterion = nn.CrossEntropyLoss()

    maxAcc = 0
    count = 0
    
    for i in range(5):
        for treeSet, text in tqdm(x_train):
            tree = treeSet[-1]
            count += 1
            idxs = f(5,len(treeSet))
            trees = [ treeSet[idx] for idx in idxs ]
            
            optimizer.zero_grad()
            bertoptimizer.zero_grad()
            
            pred = model(trees,text)
            
            label = Variable(torch.tensor(labelMap[treeSet[0].root.label]).reshape(-1).to(device))
            loss = criterion(pred.reshape(1,4),label)
#             print(loss)
    
#                 print('opt')
            loss.backward()
            optimizer.step()
            bertoptimizer.step()
        
        print('train loss: ',loss.item())   
        preds = []
        labels = []

        allLabels = []
        allPreds = []

        with torch.no_grad():
            for valSet, text in tqdm(x_test):
                idxs = f(5,len(valSet))
                trees = [ valSet[idx] for idx in idxs ]

                predicted = model(trees,text)
                preds.append(predicted)
        #         print(predicted)
                predicted =  torch.softmax(predicted,0)
                predicted = torch.max(predicted, 0)[1].cpu().numpy().tolist()

                labels.append(labelMap[trees[0].root.label])

                allLabels.append(labelMap[trees[0].root.label])
                allPreds.append(predicted)

            predTensor = torch.stack(preds)
            labelTensor = torch.tensor(labels).to(device)

            print(allLabels,allPreds)

            loss = criterion(predTensor.reshape(-1,4), labelTensor.reshape(-1))

        cr = classification_report(allLabels,allPreds,output_dict=True)
        cr['loss'] = loss.item()
        cr['Acc'] = accuracy_score(allLabels,allPreds,)

        if cr['Acc'] > maxAcc:
            maxAcc = cr['Acc']
            torch.save({'state_dict': model.state_dict()}, './twitter16_results/'+modelname+'.pth')


        print('val loss: ',cr['loss'])
        print(cr['Acc'])

        with open('./twitter16_results/'+modelname+'.json', 'a') as fp:
            json.dump(cr, fp)
            fp.write('\n')


# treeparams = {
#     'cuda': torch.cuda.is_available(),
#     'in_dim':8,
#     'mem_dim':100,
#     'userVects':userVects,
#     'labels':twitter15_labels,
#     'labelMap':labelMap,
#     'criterion':nn.CrossEntropyLoss()
# }
# 
# textparams = {
#     'embedding_dim':256,
#     'hidden_dim': 50,
#     'output_dim':4,
#     'num_layers':1,
#     'bidir':True,
#     'rnnType':'gru'
# }
# 
# treeTypes = ['standard','decay']
# textTypes = ['bert']
# pretrainTypes = [True,False]
# 
# treeTypes = ['standard','decay']
# textTypes = ['bert']
# pretrainTypes = [True]
# 
# for textType in textTypes:
#     for treeType in treeTypes:
#         for pretrainType in pretrainTypes:
#             model = jointModel(treeType,textType,pretrainType,textparams,treeparams,X_text,y,device)
#             model = model.to(device)
#             modelname = textType+'_'+treeType+'-tree_pretrain-'+str(pretrainType)
#             print(modelname)
#             trainModel(model,modelname)

# In[5]:


treemodels = {
    'temporal':'./pretrainedModels-Twit16/std_tempTreeEnc_pretrained_inc5.pth',
    'standard':'./pretrainedModels-Twit16/stdTreeEnc.pth',
    'decay':'./pretrainedModels-Twit16/decayTreeEnc.pth',
    'temporaldecay':'./pretrainedModels-Twit16/decay_tempTreeEnc_pretrained_inc5.pth',
}

textmodels = {          'bert':'./pretrainedModels-Twit16/bertTextEnc.pth',
#                'bigru':'./pretrainedModels-Twit16/Bigru.pth',
#               'bilstm':'./pretrainedModels-Twit16/Bilstm.pth',
  #              'gru':'./pretrainedModels-Twit16/gru.pth',
 #               'lstm':'./pretrainedModels-Twit16/lstm.pth',
             }


# In[6]:


modeltypes = []

for treetype in treemodels:
    for texttype in textmodels:
        modeltypes.append(list((treetype,texttype)))


# In[7]:


modeltypes


# In[ ]:


treeparams = {
    'cuda': torch.cuda.is_available(),
    'in_dim':8,
    'mem_dim':100,
    'userVects':userVects,
    'labels':twitter15_labels,
    'labelMap':labelMap,
    'criterion':nn.CrossEntropyLoss()
}

textparams = {
    'embedding_dim':256,
    'hidden_dim': 50,
    'output_dim':4,
    'num_layers':1,
}


# treeparams = {
#     'cuda': torch.cuda.is_available(),
#     'in_dim':8,
#     'mem_dim':100,
#     'userVects':userVects,
#     'labels':twitter15_labels,
#     'labelMap':labelMap,
#     'criterion':nn.CrossEntropyLoss()
# }
# 
# textparams = {
#     'embedding_dim':256,
#     'hidden_dim': 50,
#     'output_dim':4,
#     'num_layers':1,
#     'bidir':True,
#     'rnnType':'gru'
# }
# 
# treeTypes = ['standard']
# textTypes = ['rnn']
# pretrainTypes = [False]
# bidirTypes = [True]
# rnnTypes = ['gru']
# # attnTypes = ['dot']
# 
# pretrainedFiles = {
#     'bigru':'./pretrainedModels-Twit15/bidirgru.pth',
#     'bilstm':'./pretrainedModels-Twit15/bidirlstm.pth',
#     'gru':'./pretrainedModels-Twit15/gru.pth',
#     'lstm':'./pretrainedModels-Twit15/lstm.pth',
#     'stdTreeEnc':'./pretrainedModels-Twit15/std_treeEnc_pretrained_withoutTreeLoss.pth',
#     'decayTreeEnc':'./pretrainedModels-Twit15/decay_treeEnc_pretrained_withoutTreeLoss.pth',
# }
# 
# for textType in textTypes:
#     for treeType in treeTypes:
#         for pretrainType in pretrainTypes:
#             for rnnType in rnnTypes:
#                 for bidirType in bidirTypes:
#                         textparams['rnnType'] = rnnType
#                         textparams['bidirType'] = bidirType
# 
#                         model = jointModel(treeType,textType,pretrainType,textparams,treeparams,X_text,y,device)
#                         model = model.to(device)
#                         modelname = textType+'_'+treeType+'-tree_pretrain-'+str(pretrainType)+'_'+rnnType+'bidir-'+str(bidirType)
#                         print(modelname)
#                         trainModel(model,modelname
i = 2
settings = modeltypes[i]

if settings[1][0] == 'b':
    textparams['bidir'] = True
    textparams['rnnType'] = settings[1][2:]
else:
    textparams['bidir'] = False
    textparams['rnnType'] = settings[1]

model = jointModel(settings[0],'bert',{'tree':treemodels[settings[0]],'text':textmodels[settings[1]]},textparams,treeparams,X_text,y,device)
model = model.to(device)
modelname = settings[0]+'_'+'bert'
print(modelname)

if settings[0][0] == 't':
    trainTemporalModel(model,modelname)
else:
    trainModel(model,modelname)
