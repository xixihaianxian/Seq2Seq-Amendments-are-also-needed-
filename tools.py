import torch
from torch import nn
from torch.utils import data
import json
import os
import jieba
import string
import config
from collections import defaultdict
from torch import optim

class Chatbotds:
    def __init__(self,dataPath,max_length,set_max_length=True):
        self.dataPath=dataPath
        self.set_max_length=set_max_length
        if self.set_max_length:
            self.max_length=max_length
        else:
            self.max_length=None
        self.logotype={"PAD":0,"UNK":1,"END":2}
        self.chinese_punctuations=config.chinese_punctuations

    def load_data(self)->list:
        """
        文件为json数据
        :return 返回的任然是列表包含着列表，列表里面是字符串
        """
        with open(self.dataPath,"r",encoding="utf-8") as file:
            content=file.read()

        contents=json.loads(content)["conversations"]

        return contents

    def cutWord(self,sentence):
        wordlist=jieba.lcut(sentence)
        return wordlist

    def get_max_length(self,contents):
        max_length=0
        if self.set_max_length:
            tokenize_list=self.tokenize(contents)
            max_length=max(len(tokenize) for tokenize in tokenize_list)
        self.max_length=max_length

    def cleanSentence(self,sentence:list):
        return [word for word in sentence if word not in string.digits and word not in string.punctuation and word not in self.chinese_punctuations]

    def revise_sentence_list(self,sentence_list:list,keep=True)->list:
        """
        :param sentence_list: 已经划分好的数据，列表里面是划分好的句子列表
        :return: list
        """
        result=[]
        for word_list in sentence_list:
            word_list=self.cleanSentence(word_list)
            if len(word_list)>self.max_length:
                word_list=word_list[:self.max_length]
            if len(word_list)<self.max_length:
                word_list.extend(["PAD"]*(self.max_length-len(word_list)))
            # if len(word_list)==self.max_length:
            word_list.append("END")
            if keep==True:
                result.append(word_list)
            else:
                result.extend(word_list)
        return result

    def split_chat(self,contents,people=True):
        self.people_sentences=list()
        self.robot_sentences=list()
        if isinstance(contents,list):
            for content in contents:
                if len(content)%2==0:
                    content=content
                else:
                    content=content[:-1]
                for index,sentence in enumerate(content):
                    if index%2==0:
                        self.people_sentences.append(self.cutWord(sentence))
                    else:
                        self.robot_sentences.append(self.cutWord(sentence))
            if people:
                return self.people_sentences
            else:
                return self.robot_sentences
        else:
            raise ValueError("contents need is list type")

    def tokenize(self,contents):
        result=[]
        for content in contents:
            for sentence in content:
                result.append(self.cutWord(sentence))
        return result

    def word_to_id(self,words):
        words=set(words)
        word2id=defaultdict(int)
        word2id.update(self.logotype)
        for word in words:
            if word not in word2id.keys():
                word2id[word]=len(word2id)
        return word2id

    def id_to_word(self,word2id:dict):
        return dict(zip(word2id.values(),word2id.keys()))

    def word_to_tensor(self,sentences,word2id):
        return torch.tensor([[word2id.get(word) if word in word2id.keys() else word2id.get("UNK") for word in sentence] for sentence in sentences])

    def tensor_to_word(self,tensor,id2word):
        return [[id2word.get(id.item()) for id in sentence] for sentence in tensor]


class ChatDataset(data.Dataset):
    def __init__(self,people,robot):
        super().__init__()
        self.people=people
        self.robot=robot

    def __len__(self):
        if len(self.people)!=len(self.robot):
            raise ValueError("sample and target need just as long")
        return len(self.people)

    def __getitem__(self,item):
        sample_people=self.people[item]
        sample_robot=self.robot[item]
        return sample_people,sample_robot

def getDataLoader(dataset,shuffle=True,batch_size=10):
    if shuffle:
        return data.DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True)
    else:
        return data.DataLoader(dataset=dataset,batch_size=batch_size,shuffle=False)


def get_optimizer(model:nn.Module,lr=0.001):
    return optim.Adam(params=model.parameters(),lr=lr)

# optim.Optimizer(params=[
#     {"params":'...',"lr":'...'},
#     {"params":'...',"lr":',,,'},
# ])

def loss_function():
    return nn.CrossEntropyLoss()

def cuda_or_cpu():
    return "cuda" if torch.cuda.is_available() else "cpu"

if __name__=="__main__":
    chatbots=Chatbotds(dataPath="./data/conversations.corpus.json",max_length=6,set_max_length=True)
    contents=chatbots.load_data()
    sentence_list=chatbots.tokenize(contents)
    people_sentences=chatbots.revise_sentence_list(chatbots.split_chat(contents,people=True),keep=True)
    robot_sentences=chatbots.revise_sentence_list(chatbots.split_chat(contents,people=False),keep=True)
    words=chatbots.revise_sentence_list(sentence_list,keep=False)
    w2i=chatbots.word_to_id(words)
    i2w=chatbots.id_to_word(w2i)
    print(i2w)
    people=chatbots.word_to_tensor(people_sentences,w2i)
    robot=chatbots.word_to_tensor(robot_sentences,w2i)
    chatdataset=ChatDataset(people,robot)
    chatdataloader=getDataLoader(dataset=chatdataset,shuffle=True,batch_size=10)
    for item in chatdataloader:
        print(item[0].shape)
        print(chatbots.tensor_to_word(item[0],i2w))
        print(chatbots.tensor_to_word(item[1],i2w))
        break