import torch
from torch import nn
from torch.utils import data
import json
import os
import jieba

class Chatbotds:
    def __init__(self,dataPath,max_length,set_max_length=True):
        self.dataPath=dataPath
        self.set_max_length=set_max_length
        if self.set_max_length:
            self.max_length=max_length
        else:
            self.max_length=None

    def load_data(self):
        """
        文件为json数据
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
        return max_length

    def revise_sentence_list(self,sentence_list):
        if len(sentence_list)>self.max_length:
            pass

    def split_chat(self,contents,people=True):
        self.people_sentences=list()
        self.robot_sentences=list()
        if isinstance(contents,list):
            for content in contents:
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
            raise ValueError("contents need is dict type")

    def tokenize(self,contents):
        result=[]
        for content in contents:
            for sentence in content:
                result.append(self.cutWord(sentence))
        return result