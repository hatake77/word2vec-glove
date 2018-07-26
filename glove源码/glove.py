import numpy as np

from collections import Counter
import codecs
import logging
from math import log
import os.path
from random import shuffle
import threading


logger = logging.getLogger("glove")


def build_vocab(corpus, min_count=10):
    """
    从语料中统计词频，建成词频和索引字典
    return：字典vocab{word：(index, frequency)}

    函数参数：corpus（语料list）min_count（能保留的最低频次）
    返回：词频字典
    """

    logger.info("Building vocab from corpus")
    #统计词频
    vocab = Counter()
    for line in corpus:
        tokens = line.split()
        vocab.update(tokens)

    #去掉低频词
    keys=list(vocab.keys())
    for key in keys:
        if vocab[key]<=min_count:
            del[vocab[key]]


    logger.info("Done building vocab from corpus.")

    return {word: (i, freq) for i, (word, freq) in enumerate(vocab.items())}


def build_cooccur(vocab, corpus, window_size=8):
    """
    构建共现矩阵
    以字典形式存储词对和相应的共现次数：cooccurence{(main_id, context_id):cooccurence}
    main_id为中心词的索引，context_id为其语境词的索引，cooccurence为共现次数

    函数参数：vocab（词频字典）corpus（语料list）window_size（窗口大小）
    返回：共现矩阵的字典
    """

    vocab_size = len(vocab)

    # 稀疏矩阵
    #cooccurrences = sparse.lil_matrix((vocab_size, vocab_size),
    #                                  dtype=np.float64)

    cooccurrences={}
    for i, line in enumerate(corpus):
        if i % 1000 == 0:
            logger.info("Building cooccurrence matrix: on line %i", i)
        # 记录词语的索引
        tokens = line.strip().split()
        token_ids = [vocab[word][0] for word in tokens if word in vocab.keys()]

        for center_i, center_id in enumerate(token_ids):
            # 语境词：中心词左边窗口大小距离的词
            context_ids = token_ids[max(0, center_i - window_size) : center_i]
            contexts_len = len(context_ids)

            for left_i, left_id in enumerate(context_ids):
                # 语境词和中心词的距离
                distance = contexts_len - left_i

                # 共现次数与距离有关
                increment = 1.0 / float(distance)

                #构建字典
                if (center_id,left_id) in cooccurrences.keys():
                    cooccurrences[(center_id,left_id)] += increment
                    cooccurrences[(left_id,center_id)] += increment
                else:
                    cooccurrences[(center_id,left_id)] = increment
                    cooccurrences[(left_id,center_id)] = increment

    return cooccurrences


def run_iter(vocab, data, learning_rate=0.05, x_max=100, alpha=0.75):
    """
    用共现矩阵和初始化的梯度和偏置做一次训练的迭代，更新梯度、偏置、词向量

    函数参数：
    vocab（词频）
    data：传入的初始化过的词向量、梯度和共现矩阵
        (v_main, v_context,
         b_main, b_context,
         gradsq_W_main, gradsq_W_context,
         gradsq_b_main, gradsq_b_context,
         cooccurrence)
    每个元素是一个ndarray

    learning_rate（学习率）
    x_max（权重函数中的最大x）
    alpha（权重函数中的指数）

    返回 更新的代价
    """

    global_cost = 0

    # We want to iterate over data randomly so as not to unintentionally
    # bias the word vector contents
    #打乱词语的顺序
    shuffle(data)

    for (v_main, v_context, b_main, b_context, gradsq_W_main, gradsq_W_context,
         gradsq_b_main, gradsq_b_context, cooccurrence) in data:
        #权重 $$f(x)=(x/x_max)^α or 1$$
        weight = (cooccurrence / x_max) ** alpha if cooccurrence < x_max else 1
        # 初步的代价函数：$$J' = w_i^Tw_j + b_i + b_j - log(X_{ij})$$
        cost_inner = (v_main.dot(v_context)
                      + b_main[0] + b_context[0]
                      - log(cooccurrence))
        # 代价函数：$$J = f(X_{ij}) (J')^2$$
        cost = weight * (cost_inner ** 2)

        # 更新代价
        global_cost += 0.5 * cost

        #计算中心词和语境的梯度
        grad_main = weight * cost_inner * v_context
        grad_context = weight * cost_inner * v_main

        #计算偏置的梯度
        grad_bias_main = weight * cost_inner
        grad_bias_context = weight * cost_inner

        #AdaGrad更新中心词和语境的词向量
        v_main -= (learning_rate * grad_main / np.sqrt(gradsq_W_main))
        v_context -= (learning_rate * grad_context / np.sqrt(gradsq_W_context))

        #更新偏置
        b_main -= (learning_rate * grad_bias_main / np.sqrt(gradsq_b_main))
        b_context -= (learning_rate * grad_bias_context / np.sqrt(
                gradsq_b_context))

        #更新梯度
        gradsq_W_main += np.square(grad_main)
        gradsq_W_context += np.square(grad_context)
        gradsq_b_main += grad_bias_main ** 2
        gradsq_b_context += grad_bias_context ** 2

    return global_cost


def train_glove(vocab, cooccurrences, vector_size,
                iterations=1, thread_num=3):
    """
    用共现矩阵训练词向量，进行初始化工作并调用函数run_iter进行训练

    函数参数：vocab（词频字典）cooccurences（共现矩阵）vector_size（词向量维度）
             iterations（迭代次数）thread_num（线程个数）

    返回：W（训练好的词向量）
    """

    vocab_size = len(vocab)


    #词向量：维度为(2*vocab_size)*vector_size的矩阵，初始化为(-0.5,0.5]的浮点数
    #对每个词，为其作为中心词或语境词分别构建两个词向量
    W = (np.random.rand(vocab_size * 2, vector_size) - 0.5) / float(vector_size + 1)


    #偏置向量：维度为2*vocab_size的矩阵，初始化为(-0.5,0.5]的浮点数
    biases = (np.random.rand(vocab_size * 2) - 0.5) / float(vector_size + 1)

    # 梯度下降方式：自适应的梯度下降adaptive gradient descent (AdaGrad)
    # 利用之前计算的所有梯度之和更新学习率
    # 初始化为1
    gradient_squared = np.ones((vocab_size * 2, vector_size),
                               dtype=np.float64)

    #偏置梯度
    gradient_squared_biases = np.ones(vocab_size * 2, dtype=np.float64)

    # 将初始化的词向量、梯度矩阵和共现矩阵传入run_iter作为参数
    data = [(W[i_main], W[i_context + vocab_size],
             biases[i_main: i_main + 1],
             biases[i_context + vocab_size: i_context + vocab_size + 1],
             gradient_squared[i_main], gradient_squared[i_context + vocab_size],
             gradient_squared_biases[i_main: i_main + 1],
             gradient_squared_biases[i_context + vocab_size
                                     : i_context + vocab_size + 1],
             cooccurrence)
            for (i_main, i_context), cooccurrence in cooccurrences.items()]

    #多线程
    threads = []
    data_per_thread = int(len(data) / thread_num)
    m = 0
    for i in range(thread_num):
        n = int(len(data) - (thread_num - i - 1) * data_per_thread)
        threads.append(threading.Thread(target=run_iter, args=(vocab, data[m:n])))
        m += n

    for i in range(iterations):
        logger.info("\tBeginning iteration %i..", i)
        for t in threads:
            t.setDaemon(True)
            t.start()
    # num = 0
    # list = []
    # temp = []
    # data = []
    # for key in cooccurrences.keys():
    #     temp.append(key)
    #     num += 1
    #     if num == int(len(cooccurrences)/3) or num==int(len(cooccurrences)/3)*2 or num==len(cooccurrences):
    #         list.append(temp)
    #         temp=[]
    # for t in list:
    #     print("start data:")
    #     data = [(W[i_main], W[i_context + vocab_size],
    #          biases[i_main : i_main + 1],
    #          biases[i_context + vocab_size : i_context + vocab_size + 1],
    #          gradient_squared[i_main], gradient_squared[i_context + vocab_size],
    #          gradient_squared_biases[i_main : i_main + 1],
    #          gradient_squared_biases[i_context + vocab_size
    #                                  : i_context + vocab_size + 1],
    #          cooccurrences[(i_main, i_context)])
    #         for (i_main, i_context) in t]
    #     print("data")
    #     run_iter(vocab, data)
    #     print("iteration")
    #     data=[]

    return W


def save_model(W, vocab, vector_size, path):
    '''
    保存词向量至文本文档
    格式 word 词向量（每一维以空格分隔）
    函数参数：W（词向量） vocab（词频字典）vector_size（词向量维度）path（保存路径）
    '''
    logger.info("Begin to save vectors to %s", path)
    with open(path, 'w') as vector_f:
        for word in vocab.keys():
            vector_f.write(word)
            vector_f.write(" ")
            data = (W[vocab[word][0]]+W[vocab[word][0]+len(vocab)])/2
            for j in range(vector_size):
                vector_f.write(str(data[j]))
                vector_f.write(" ")
            vector_f.write('\n')



def main(corpus_path,vector_size):
    '''
    主函数
    流程：读取语料-构建词频字典-构建共现矩阵-训练词向量-保存词向量

    函数参数：corpus_path（语料路径）vector_size（词向量维度）
    '''
    #读取语料
    with open(corpus_path,"r") as file:
        corpus = file.readlines()

    #构建词频字典
    logger.info("Getting vocab..")
    vocab = build_vocab(corpus)
    logger.info("Vocab has %i elements.\n", len(vocab))

    #构建共现矩阵
    logger.info("Getting cooccurrence list..")
    cooccurrences = build_cooccur(vocab,corpus)
    logger.info("Cooccurence list scale： %i.\n", len(cooccurrences))
    logger.info("Cooccurrence list complete.\n")

    #训练词向量
    logger.info("Beginning GloVe training..")
    W = train_glove(vocab, cooccurrences,vector_size)

    #保存词向量
    save_model(W, vocab, vector_size, "result.txt")
    logger.info("Save done..")

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s\t%(message)s")
    #语料路径
    path = "text.txt"
    main(path,100)
