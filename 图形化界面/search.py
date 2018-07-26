import tkinter
import pygeoip
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

class Word2Vec(object):
    def __init__(self, vec_len=15000, learn_rate=0.025, win_len=5, model='cbow'):
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
        #self.gi = pygeoip.GeoIP("./GeoLiteCity.dat")
        # 创建主窗口,用于容纳其它组件
        self.root = tkinter.Tk()
        # 给主窗口设置标题内容
        self.root.title("查询最相近词")
        # 创建一个输入框,并设置尺寸
        self.ip_input = tkinter.Entry(self.root,width=30)

        # 创建一个回显列表
        self.display_info = tkinter.Listbox(self.root, width=50)

        # 创建一个查询结果的按钮
        self.result_button = tkinter.Button(self.root, command = self.validate, text = "查询")
     
    # 完成布局
    def gui_arrang(self):
        self.ip_input.pack()
        self.display_info.pack()
        self.result_button.pack()

    def Import_Model(self,model_path):    #直接读入word2vec模型
        model = FI.load_pickle(model_path)  # 字典形式, {'word_dict','huffman','vec_len'}
        self.word_dict = model['word_dict']
        self.huffman = model['huffman']
        self.vec_len = model['vec_len']
        self.learn_rate = model['learn_rate']
        self.win_len = model['win_len']
        self.model = model['model']
        
    def validate(self):
        #f2 = open('testtest.txt','a')
        word = self.ip_input.get()
        word_cos={}
        for w in self.word_dict.keys():
            word_cos[w]=self.cos(self.word_dict[w]['vector'],self.word_dict[word]['vector'])
        
        tmp_word_cos= sorted(word_cos.items(), key=lambda d:d[1], reverse = True)
        cnt=0
        required_cnt=8
        result=[]
        #f2.writelines("与 %s 最相近的单词是:\n "% word)
        #print("与 %s 最相近的单词是:\n "% word)
        for i in range(required_cnt):
            result.insert(0,str(tmp_word_cos[i]))
            #print(tmp_word_cos[i])
        for item in range(required_cnt):
            self.display_info.insert(0,"")

        # 为回显列表赋值
        for item in result:
            self.display_info.insert(0,item)
        # 这里的返回值,没啥用,就是为了好看
        return result    
        
    def cos(self,vector1,vector2):  
        dot_product = 0.0;  
        normA = 0.0;  
        normB = 0.0;  
        for i in range(len(vector1)-1):
            #print(type(vector1[i]))
            dot_product += float(vector1[i])*float(vector2[i])  
            normA += float(vector1[i])**2  
            normB += float(vector2[i])**2 
        if normA == 0.0 or normB==0.0:  
            return -100000  
        else:  
            return dot_product / ((normA*normB)**0.5)



def main():
    # 初始化对象
    wv=Word2Vec(vec_len=500)
    wv.Import_Model("common.pkl")
    # 进行布局
    wv.gui_arrang()
    # 主程序执行
    tkinter.mainloop()
    pass


if __name__ == "__main__":
    main()
