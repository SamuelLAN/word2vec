#!/usr/bin/Python
# -*- coding: utf-8 -*-
import os
import sys

# 将运行路径切换到当前文件所在路径
cur_dir_path = os.path.split(__file__)[0]
if cur_dir_path:
    os.chdir(cur_dir_path)
    sys.path.append(cur_dir_path)

import base
import load
import random
import numpy as np
import tensorflow as tf
from matplotlib import pylab


'''
  Skip-gram 模型
'''
class SkipGram(base.NN):
    MODEL_NAME = 'skip_gram'    # 模型的名称

    EPOCH_TIMES = 200           # 迭代的 epoch 次数
    BATCH_SIZE = 128            # 随机梯度下降的 batch 大小
    BASE_LEARNING_RATE = 1.0    # 初始学习率

    SKIP_WINDOW = 1             # 目标词 左边 或 右边的 上下文词语数量
    NUM_NEG_SAMPLE = 64         # 负样本个数

    VALID_SIZE = 16             # 验证集大小
    VALID_WINDOW = 100          # 从 data 中取前 100 个样本

    VOCABULARY_SIZE = 50000     # 设置词典的数据量
    EMBEDDING_SIZE = 128        # 词向量的大小
    SHAPE = [VOCABULARY_SIZE, EMBEDDING_SIZE]   # 权重矩阵的 shape

    def init(self):
        # 加载数据
        self.load()

        # 输入 与 label
        self.__X = tf.placeholder(tf.int32, [self.BATCH_SIZE])          # 训练集数据
        self.__y = tf.placeholder(tf.int32, [self.BATCH_SIZE, 1])       # 训练集 label

        # 从 1-100 中随机采样 valid_size 个样本作为验证集的样本
        self.__valExample = np.array(random.sample(range(self.VALID_WINDOW), self.VALID_SIZE))
        self.__valX = tf.constant(self.__valExample, dtype=tf.int32)    # 验证集数据

        # 常量
        self.__iterPerEpoch = int(self.VOCABULARY_SIZE // self.BATCH_SIZE)
        self.__steps = self.EPOCH_TIMES * self.__iterPerEpoch

        # 随训练次数增多而衰减的学习率
        self.__learningRate = self.getLearningRate(
            self.BASE_LEARNING_RATE, self.globalStep, self.BATCH_SIZE, self.__steps, self.DECAY_RATE
        )

        self.__finalEmbedding = np.array([])                            # 初始化最终的词向量


    ''' 加载数据 '''
    def load(self):
        self.__data = load.Data()
        self.__reverseDict = self.__data.getReverseDict()


    ''' 模型 '''
    def model(self):
        with tf.name_scope('layer_1'):
            self.__embeddings = tf.Variable(
                tf.random_uniform(self.SHAPE, -1.0, 1.0), name='embeddings'
            )

            self.__embed = tf.nn.embedding_lookup(self.__embeddings, self.__X, name='embed')

        with tf.name_scope('layer_2'):
            self.__W = self.initWeight(self.SHAPE)
            self.__b = self.initBias([self.VOCABULARY_SIZE])


    ''' 计算 loss '''
    def getLoss(self):
        with tf.name_scope('loss'):
            self.__loss = tf.reduce_mean(
                tf.nn.sampled_softmax_loss(weights=self.__W, biases=self.__b, inputs=self.__embed, labels=self.__y,
                                           num_sampled=self.NUM_NEG_SAMPLE, num_classes=self.VOCABULARY_SIZE)
            )


    ''' 获取 train_op '''
    def getTrainOp(self, loss, learning_rate, global_step):
        tf.summary.scalar('loss', loss)  # 记录 loss 到 TensorBoard

        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdagradOptimizer(learning_rate)
            return optimizer.minimize(loss, global_step=global_step)


    ''' 计算相似度 '''
    def __calSimilarity(self):
        with tf.name_scope('similarity'):
            # 只在 y 轴上求 平方和 然后 开根，x 轴维数不变 ( x 轴对应的是 batch 的数量 )
            norm = tf.sqrt( tf.reduce_sum( tf.square(self.__embeddings), 1, keep_dims=True ) )

            # normalize 当前的 embedding 参数
            self.__normalizeEmbedding = self.__embeddings / norm

            # 在当前的 embedding 参数中计算 验证集的词向量
            valid_embeddings = tf.nn.embedding_lookup(self.__normalizeEmbedding, self.__valX)

            self.__similarity = tf.matmul(valid_embeddings, tf.transpose(self.__normalizeEmbedding), name='similarity')


    ''' 将 tsne 降维后的数据 画成 2d 图像 '''
    def __plot2dEmbeddingSim(self, num_words_to_plot):
        self.echo('plotting 2d embeddings image by tsne ...')

        self.echo('Doing tsne ...')
        words_to_plot = [self.__reverseDict[i] for i in range(1, num_words_to_plot + 1)]  # 第 0 个为 UNK
        embeddings = self.tsne(self.__finalEmbedding, num_words_to_plot)
        self.echo('finish tsne')

        assert embeddings.shape[0] >= len(words_to_plot)   # embeddings 的维度需要比 label 的词数量多
        pylab.figure(figsize=(15, 15))
        for i, word in enumerate(words_to_plot):
            x, y = embeddings[i, :]
            pylab.scatter(x, y)
            pylab.annotate(word, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')

        pylab.show()

        self.echo('Finish')


    ''' 获取最终的 embeddings '''
    def getEmbeddings(self):
        return self.__finalEmbedding


    def run(self):
        # 生成模型
        self.model()

        # 计算 loss
        self.getLoss()

        # 生成训练的 op
        train_op = self.getTrainOp(self.__loss, self.__learningRate, self.globalStep)

        # 计算相似度
        self.__calSimilarity()

        # 初始化所有变量
        self.initVariables()

        # TensorBoard merge summary
        self.mergeSummary()

        best_loss = 999999                  # 记录训练集最好的 loss
        increase_loss_times = 0             # loss 连续上升的 epoch 次数

        for step in range(self.__steps):

            if step % 50 == 0:                                  # 输出进度
                self.echo('step: %d (%d|%.2f%%) / %d|%.2f%%     \r' % (step, self.__iterPerEpoch, 1.0 * step % self.__iterPerEpoch / self.__iterPerEpoch * 100.0, self.__steps, 1.0 * step / self.__steps * 100.0), False)

            batch_x, batch_y = self.__data.skipGramNextBatch(self.BATCH_SIZE)
            feed_dict = {self.__X: batch_x, self.__y: batch_y}

            self.sess.run(train_op, feed_dict)                  # 运行 训练

            if step % self.__iterPerEpoch == 0 and step != 0:   # 完成一个 epoch 时
                epoch = step // self.__iterPerEpoch             # 获取这次 epoch 的 index
                loss = self.sess.run(self.__loss, feed_dict)

                self.echo('\n****************** %d *********************' % epoch)

                self.addSummary(feed_dict, epoch)               # 输出数据到 TensorBoard

                # 用于输出与验证集对比的数据，查看训练效果
                sim = self.sess.run(self.__similarity)
                for i in range(self.VALID_SIZE):
                    valid_word = self.__reverseDict[self.__valExample[i]]

                    top_k = 8                                           # 设置最近的邻居个数
                    nearest = (-sim[i, :]).argsort()[1 : top_k + 1]     # 获取最近的 top_k 个邻居

                    # 输出展示最近邻居情况
                    log = 'Nearest to %s: ' % valid_word
                    for k in range(top_k):
                        log += self.__reverseDict[nearest[k]] + ' '
                    self.echo(log)

                if loss < best_loss:                    # 当前 loss 比 最好的 loss 低，则更新 best_loss
                    best_loss = loss
                    increase_loss_times = 0

                    self.saveModel()                    # 保存模型

                else:                                   # 否则
                    increase_loss_times += 1
                    if increase_loss_times > 30:
                        break

        self.closeSummary()  # 关闭 TensorBoard

        self.restoreModel()  # 恢复模型

        self.__finalEmbedding = self.sess.run(self.__normalizeEmbedding)    # 获取最终需要的词向量

        self.__plot2dEmbeddingSim(400)  # 把 词向量 降维后画成图像


o_nn = SkipGram()
o_nn.run()
