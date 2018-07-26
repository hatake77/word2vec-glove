
import math
import File_Interface as FI
from operator import itemgetter as _itemgetter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn import preprocessing
from collections import Counter
import random

#itemgetter用于获取对象的哪些维的数据，是一个函数


class Word2Vec():
    def __init__(self, vec_len=15000, learn_rate=0.025, win_len=8, model='cbow'):
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

    def Load_Word_Freq(self,word_freq_path):
        # 加载词频数据
        # 加载词频后输出词典
        if self.word_dict is not None:
            raise RuntimeError('the word dict is not empty')
        word_freq = FI.load_pickle(word_freq_path)
        self.__Gnerate_Word_Dict(word_freq)

    def __Gnerate_Word_Dict(self,word_freq):
        # 用于生成词典
        # 词典包括 word, freq, possibility, 经过随机初始化的 vector 和 Huffman 值
        if not isinstance(word_freq,dict) and not isinstance(word_freq,list):
            raise ValueError('the word freq info should be a dict or list')

        word_dict = {}
        if isinstance(word_freq,dict):   #word_freq是词频
            # 如果词频是用字典形式存储
            sum_count = sum(word_freq.values())  #values()方法用于返回给定字典中可用的所有值的列表。
            
            for word in word_freq:
                #print(word)
                temp_dict = dict(     #为每个词建立一个dict
                    word = word,
                    freq = word_freq[word],
                    possibility = word_freq[word]/sum_count,
                    vector=np.random.random([1,self.vec_len]),
                    Huffman = None
                )
                word_dict[word] = temp_dict
        else:
            # 如果词频是用列表形式存储
            freq_list = [x[1] for x in word_freq]
            sum_count = sum(freq_list)
            for item in word_freq:
                temp_dict = dict(
                    word = item[0],
                    freq = item[1],
                    possibility = item[1]/sum_count,
                    vector=np.random.random([1,self.vec_len]),
                    Huffman = None
                )
                word_dict[item[0]] = temp_dict
        self.word_dict = word_dict
        print("__Gnerate_Word_Dict finish\n")

    def Import_Model(self,model_path):    #直接读入word2vec模型
        model = FI.load_pickle(model_path)  # 以字典形式储存, {'word_dict','huffman','vec_len'}
        self.word_dict = model['word_dict']
        self.huffman = model['huffman']
        self.vec_len = model['vec_len']
        self.learn_rate = model['learn_rate']
        self.win_len = model['win_len']
        self.model = model['model']

    def Export_Model(self,model_path):   #存储word2vec模型
        data=dict(
            word_dict = self.word_dict,
            huffman = self.huffman,
            vec_len = self.vec_len,
            learn_rate = self.learn_rate,
            win_len = self.win_len,
            model = self.model
        )
        FI.save_pickle(data,model_path)

    def Train_Model(self,text_list):
        print("Train_Model\n")
        # 生成字典和Huffman树
        if self.huffman==None:
            # 如果字典之前没有被加载过，那么就要生成新字典
            if self.word_dict==None :
                wc = WordCounter(text_list)  #使用wordcounter统计了词频，过滤掉了停用词和去掉词频过大或过小的词
                self.__Gnerate_Word_Dict(wc.count_res)
                #self.__Gnerate_Word_Dict(wc.count_res.less_than(300))
                self.cutted_text_list = wc.text_list

            # 根据各个词的possiblity生成Huffman树
            self.huffman = HuffmanTree(self.word_dict,vec_len=self.vec_len)
        print('word_dict and huffman tree already generated, ready to train vector')
        self.result_word=[]
        self.word_vec=[]
        # 开始训练词向量
        before = (self.win_len-1) >> 1
        after = self.win_len-1-before

        if self.model=='cbow':
            method = self.__Deal_Gram_CBOW
        else:
            method = self.__Deal_Gram_SkipGram
            
        f1 = open('test2.txt', 'w')
        f2 = open('result2.txt','w')
        if self.cutted_text_list:
            # 如果文本之前被处理过
            total = self.cutted_text_list.__len__()
            count = 0
            number=0
            line_len = self.cutted_text_list.__len__()
            line=self.cutted_text_list
            for i in range(line_len):                
                number+=1
                method(line[i],line[max(0,i-before):i]+line[i+1:min(line_len,i+after+1)])   #取窗口大小个临近词
            count += 1
            print('{c} of {d}'.format(c=count,d=total))

        else:
            # 如果文本之前没有被处理过
            number=0
            for line in text_list:
                #line = list(jieba.cut(line,cut_all=False))
                line_len = line.__len__()
                for i in range(line_len):
                    number+=1
                    method(line[i],line[max(0,i-before):i]+line[i+1:min(line_len,i+after+1)])
        print('word vector has been generated')

        cnt=0
        for word in self.word_dict.keys():
            cnt+=1
            if cnt%800==0:
                self.validate(self.word_dict[word]['word'])
                f1.writelines(self.word_dict[word]['word'])
                f1.writelines('\n')
                f1.writelines(str(self.word_dict[word]['vector']))
                f1.writelines('\n')
                f1.writelines('\n')
        self.int_to_vocab = {c: w for c, w in enumerate(self.result_word)}


    def __Deal_Gram_CBOW(self,word,gram_word_list):

        if not self.word_dict.__contains__(word):
            return

        word_huffman = self.word_dict[word]['Huffman']
        gram_vector_sum = np.zeros([1,self.vec_len])
        for i in range(gram_word_list.__len__())[::-1]:    #求窗口内各词向量的和作为算法输入
            item = gram_word_list[i]
            if self.word_dict.__contains__(item):
                gram_vector_sum += self.word_dict[item]['vector']
            else:
                gram_word_list.pop(i)   #如果该词没有词向量则剔除

        if gram_word_list.__len__()==0:
            return

        e = self.__GoAlong_Huffman(word_huffman,gram_vector_sum,self.huffman.root)

        for item in gram_word_list:  #根据误差调整窗口内单词的词向量
            self.word_dict[item]['vector'] += e
            self.word_dict[item]['vector'] = preprocessing.normalize(self.word_dict[item]['vector'])

    def __Deal_Gram_SkipGram(self,word,gram_word_list):

        if not self.word_dict.__contains__(word):
            return

        word_vector = self.word_dict[word]['vector']
        for i in range(gram_word_list.__len__())[::-1]:
            if not self.word_dict.__contains__(gram_word_list[i]):
                gram_word_list.pop(i)

        if gram_word_list.__len__()==0:
            return

        for u in gram_word_list:
            u_huffman = self.word_dict[u]['Huffman']
            e = self.__GoAlong_Huffman(u_huffman,word_vector,self.huffman.root)
            self.word_dict[word]['vector'] += e
            self.word_dict[word]['vector'] = preprocessing.normalize(self.word_dict[word]['vector'])

    def __GoAlong_Huffman(self,word_huffman,input_vector,root):

        node = root
        e = np.zeros([1,self.vec_len])
        for level in range(word_huffman.__len__()):  #根据huffman各个编码0、1判断
            huffman_charat = word_huffman[level]
            q = self.__Sigmoid(input_vector.dot(node.value.T))
            grad = self.learn_rate * (1-int(huffman_charat)-q)  #计算误差
            e += grad * node.value     #累计误差
            node.value += grad * input_vector  #修正中间节点的向量  
            node.value = preprocessing.normalize(node.value)
            if huffman_charat=='0':
                node = node.right
            else:
                node = node.left
        return e

    def __Sigmoid(self,value):
        return 1/(1+math.exp(-value))
        
    def to_txt(self,path):
        f = open(path,'w')
        for word in self.word_dict.keys():
            f.writelines(self.word_dict[word]['word'])
            f.writelines(' ')
            
            for item in self.word_dict[word]['vector']:
                f.writelines(str(item))
                f.writelines(' ')
            f.writelines('\n')
            
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

class HuffmanTreeNode():
    def __init__(self,value,possibility):
        # 生成内部节点和叶节点
        self.possibility = possibility
        self.left = None
        self.right = None
        # 叶节点的值是单词
        # 内部节点的值是中间向量
        self.value = value
        self.Huffman = "" # 存储Huffman编码

    def __str__(self):
        return 'HuffmanTreeNode object, value: {v}, possibility: {p}, Huffman: {h}' \
            .format(v=self.value,p=self.possibility,h=self.Huffman)

class HuffmanTree():
    def __init__(self, word_dict, vec_len=15000):
        self.vec_len = vec_len      # 词向量长度
        self.root = None

        word_dict_list = list(word_dict.values())
        
        node_list = [HuffmanTreeNode(x['word'],x['possibility']) for x in word_dict_list]  #单词向量先转为叶子结点
        self.build_tree(node_list)
        # self.build_CBT(node_list)
        self.generate_huffman_code(self.root, word_dict)

    def build_tree(self,node_list):

        while node_list.__len__()>1:
            i1 = 0  # i1表示概率最小的节点
            i2 = 1  # i2 概率第二小的节点
            if node_list[i2].possibility < node_list[i1].possibility :
                [i1,i2] = [i2,i1]
            for i in range(2,node_list.__len__()): # 找到最小的两个节点
                if node_list[i].possibility<node_list[i2].possibility :
                    i2 = i
                    if node_list[i2].possibility < node_list[i1].possibility :
                        [i1,i2] = [i2,i1]
            top_node = self.merge(node_list[i1],node_list[i2])
            if i1<i2:
                node_list.pop(i2)
                node_list.pop(i1)
            elif i1>i2:
                node_list.pop(i1)
                node_list.pop(i2)
            else:
                raise RuntimeError('i1 should not be equal to i2')
            node_list.insert(0,top_node)
        self.root = node_list[0]

    def build_CBT(self,node_list): # 生成完全二叉树
        node_list.sort(key=lambda  x:x.possibility,reverse=True)
        node_num = node_list.__len__()
        before_start = 0
        while node_num>1 :
            for i in range(node_num>>1):
                top_node = self.merge(node_list[before_start+i*2],node_list[before_start+i*2+1])
                node_list.append(top_node)
            if node_num%2==1:
                top_node = self.merge(node_list[before_start+i*2+2],node_list[-1])
                node_list[-1] = top_node
            before_start = before_start + node_num
            node_num = node_num>>1
        self.root = node_list[-1]

    def generate_huffman_code(self, node, word_dict):

        stack = [self.root]
        while (stack.__len__()>0):
            node = stack.pop()
            # go along left tree
            while node.left or node.right :
                code = node.Huffman
                node.left.Huffman = code + "1"
                node.right.Huffman = code + "0"
                stack.append(node.right)
                node = node.left
            word = node.value
            code = node.Huffman
            # print(word,'\t',code.__len__(),'\t',node.possibility)
            word_dict[word]['Huffman'] = code

    def merge(self,node1,node2):
        top_pos = node1.possibility + node2.possibility
        top_node = HuffmanTreeNode(np.zeros([1,self.vec_len]), top_pos)
        if node1.possibility >= node2.possibility :
            top_node.left = node1
            top_node.right = node2
        else:
            top_node.left = node2
            top_node.right = node1
        return top_node

class WordCounter():
    # 可以计算一段文本里的词频

    # for example
    # >>> data = ['Merge multiple sorted inputs into a single sorted output',
    #           'The API below differs from textbook heap algorithms in two aspects']
    # >>> wc = WordCounter(data)
    # >>> print(wc.count_res)

    # >>> MulCounter({' ': 18, 'sorted': 2, 'single': 1, 'below': 1, 'inputs': 1, 'The': 1, 'into': 1, 'textbook': 1,
    #                'API': 1, 'algorithms': 1, 'in': 1, 'output': 1, 'heap': 1, 'differs': 1, 'two': 1, 'from': 1,
    #                'aspects': 1, 'multiple': 1, 'a': 1, 'Merge': 1})

    def __init__(self, text_list):
        self.text_list = text_list
        self.count_res = None

        self.Word_Count(self.text_list)

    def Get_Stop_Words(self):
        ret = []
        ret = FI.load_pickle('./static/stop_words.pkl')
        return ret

    def Word_Count(self,text_list,cut_all=False):

        filtered_word_list = []
        count = 0
        for line in text_list:
            res=line
            filtered_word_list.append(res) 

        self.count_res = MulCounter(filtered_word_list)


class MulCounter(Counter):
    # 继承自 collections.Counter
    # 添加了一些方法, larger_than 和 less_than
    def __init__(self,element_list):
        super().__init__(element_list)

    def larger_than(self,minvalue,ret='list'):   #将词频大于最小值的找出，其余丢弃
        temp = sorted(self.items(),key=_itemgetter(1),reverse=True)
        low = 0
        high = temp.__len__()
        while(high - low > 1):
            mid = (low+high) >> 1
            if temp[mid][1] >= minvalue:
                low = mid
            else:
                high = mid
        if temp[low][1]<minvalue:
            if ret=='dict':
                return {}
            else:
                return []
        if ret=='dict':
            ret_data = {}
            for ele,count in temp[:high]:
                ret_data[ele]=count
            return ret_data
        else:
            return temp[:high]

    def less_than(self,maxvalue,ret='list'):  #将词频小于最大值的找出，其余丢弃
        temp = sorted(self.items(),key=_itemgetter(1))
        low = 0
        high = temp.__len__()
        while ((high-low) > 1):
            mid = (low+high) >> 1
            if temp[mid][1] <= maxvalue:
                low = mid
            else:
                high = mid
        if temp[low][1]>maxvalue:
            if ret=='dict':
                return {}
            else:
                return []
        if ret=='dict':
            ret_data = {}
            for ele,count in temp[:high]:
                ret_data[ele]=count
            return ret_data
        else:
            return temp[:high]

def preprocess(text, freq=5):
    '''
    对文本进行预处理
    
    参数
    ---
    text: 文本数据
    freq: 词频阈值
    '''
    # 对文本中的符号进行替换
    text = text.lower()
    text = text.replace('.', '')
    text = text.replace(',', '')
    text = text.replace('"', '')
    text = text.replace(';', '')
    text = text.replace('!', '')
    text = text.replace('?', '')
    text = text.replace('(', '')
    text = text.replace(')', '')
    text = text.replace('--', '')
    text = text.replace('?', '')
    # text = text.replace('\n', ' <NEW_LINE> ')
    text = text.replace(':', '')
    words = text.split(' ')
    
    # 删除低频词，减少噪音影响
    word_counts = Counter(words)
    trimmed_words = [word for word in words if word_counts[word] > freq]
    #print(str(trimmed_words))
    return trimmed_words
    

if __name__ == '__main__':
    
    wv=Word2Vec(vec_len=200)
    wv.Import_Model("word2vec150.pkl")
    wv.to_txt("word2vec150.txt")
    #wv.Train_Model(data)
    #wv.Export_Model("models.pkl")
    
    while True:
        inputs=input("please input two words to compare:")
        words = inputs.split(" ")
        if words[0] not in wv.word_dict.keys() or words[1] not in wv.word_dict.keys():
            print("sorry,words not in dict\n")
            continue
        result=wv.cos(wv.word_dict[words[0]]['vector'],wv.word_dict[words[1]]['vector'])
        print("result:")
        print(result)
    
    '''
    viz_words = 200
    tsne = TSNE()
    embed_mat=wv.word_vec
    embed_tsne = tsne.fit_transform(embed_mat[:viz_words])
    fig, ax = plt.subplots(figsize=(14, 14))
    for idx in range(viz_words):
        plt.scatter(*embed_tsne[idx, :], color='steelblue')
        plt.annotate(wv.int_to_vocab[idx], (embed_tsne[idx, 0], embed_tsne[idx, 1]), alpha=0.7)
    plt.show()
    '''
