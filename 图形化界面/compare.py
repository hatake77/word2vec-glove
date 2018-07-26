from tkinter import *
import easygui 
import math
import File_Interface as FI
from operator import itemgetter as _itemgetter
import numpy as np
#import jieba
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn import preprocessing
from collections import Counter
import random

class Word2Vec():
    def __init__(self, vec_len=15000, learn_rate=0.025, win_len=8, model='skip'):
        self.cutted_text_list = None
        self.vec_len = vec_len
        self.learn_rate = learn_rate
        self.win_len = win_len
        self.model = model
        self.word_dict = None  # 每一个元素都是一个字典, 包括: word,possibility,vector,huffmancode
        self.huffman = None    # 用于存储Huffman树
        self.result_word=None
        self.int_to_vocab=None
        self.word_vec=None

    def Import_Model(self,model_path):    #直接读入word2vec模型
        model = FI.load_pickle(model_path)  # 以字典形式储存, {'word_dict','huffman','vec_len'}
        self.word_dict = model['word_dict']
        self.huffman = model['huffman']
        self.vec_len = model['vec_len']
        self.learn_rate = model['learn_rate']
        self.win_len = model['win_len']
        self.model = model['model']

    def validate(self,word):
        f2 = open('testtest.txt','a')
        word_cos={}
        for w in self.word_dict.keys():
            word_cos[w]=self.cos(self.word_dict[w]['vector'][0],self.word_dict[word]['vector'][0])
        
        tmp_word_cos= sorted(word_cos.items(), key=lambda d:d[1], reverse = True)
        cnt=0
        required_cnt=8
        f2.writelines("与 %s 最相近的单词是:\n "% word)
        #print("与 %s 最相近的单词是:\n "% word)
        for i in range(required_cnt):
            if tmp_word_cos[i][0] not in self.result_word:
                self.result_word.insert(0,tmp_word_cos[i][0])
                self.word_vec.insert(0,self.word_dict[tmp_word_cos[i][0]]['vector'][0])
            f2.writelines(str(tmp_word_cos[i]))
            f2.writelines('\n')
            #print(tmp_word_cos[i])
            
    def cos(self,vector1,vector2):  
        dot_product = 0.0;  
        normA = 0.0;  
        normB = 0.0; 
        for i in range(len(vector1)-1):
            dot_product += float(vector1[i])*float(vector2[i])  
            normA += float(vector1[i])**2  
            normB += float(vector2[i])**2 
        if normA == 0.0 or normB==0.0:  
            return -100000  
        else:  
            return dot_product / ((normA*normB)**0.5)
			
def calc():
    if v1.get()=='':
        easygui.msgbox('请先输入内容 !')
        return
    result=wv.cos(wv.word_dict[v1.get()]['vector'],wv.word_dict[v2.get()]['vector'])
    v3.set(result)
	
def test(content):
    return content.isdigit()  # 检查是不是符合要求 .
	
master = Tk()

frame = Frame(master)  # 确定一个框架用于美观

frame.pack(padx = 20,pady = 20)

v1 = StringVar() # 分别用于储存需要计算的数据和 结果
v2 = StringVar()
v3 = StringVar()

wv=Word2Vec(vec_len=500)
wv.Import_Model("model_g.pkl")
testCMD = frame.register(test)  # 将函数 进行包装 . 

e1 = Entry(frame,width=10,textvariable=v1,validate='key',\
           validatecommand=(test,'%p')).grid(row=0,column=0,pady=10) # %p 是输入框的最新内容 . 当输入框允许改变的时候该值有效 ,
Label(frame,text='与',padx=10).grid(row=0,column=1)

e2 = Entry(frame,width=10,textvariable=v2,validate='key',\
           validatecommand=(test,'%p')).grid(row=0,column=2)
Label(frame,text='相似度为：',padx=10).grid(row=0,column=3)

e3 = Entry(frame,width=25,textvariable=v3,state='readonly').grid(row=0,column=4)

Button(frame,text='计算相似度',command=calc).grid(row=2,column=3,pady=5)

mainloop()