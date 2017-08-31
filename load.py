#!/usr/bin/Python
# -*- coding: utf-8 -*-
import os
import sys

# 将运行路径切换到当前文件所在路径
cur_dir_path = os.path.split(__file__)[0]
if cur_dir_path:
    os.chdir(cur_dir_path)
    sys.path.append(cur_dir_path)

import random
import zipfile
import collections
import numpy as np
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle


class Download:
    URL = 'http://mattmahoney.net/dc/'  # 数据的下载目录
    FILE_NAME = 'text8.zip'             # 文件名
    FILE_SIZE = 31344016                # 字节
    DATA_ROOT = 'data'                  # 数据存放的目录
    VOCABULARY_SIZE = 50000             # 设置词典的数据量
    SAVE_PATH = os.path.join(DATA_ROOT, 'text8_50000.pickle')  # 保存数据的文件路径；这里只取出现最多的 VOCABULARY_SIZE 个的词作为数据保存


    def __init__(self):
        pass


    ''' 主函数 '''
    @staticmethod
    def run():
        if os.path.isfile(Download.SAVE_PATH):
            print 'Already exist data in %s\nDone' % Download.SAVE_PATH
            return

        file_path = Download.__maybeDownload()

        words = Download.__readData(file_path)

        data, reverse_dictionary, count = Download.__buildDataSet(words)

        Download.__saveData(data, reverse_dictionary, count)


    ''' 下载数据 '''
    @staticmethod
    def __maybeDownload():
        """Download a file if not present, and make sure it's the right size."""
        if not os.path.isdir(Download.DATA_ROOT):           # 若 data 目录不存在，创建 data 目录
            os.mkdir(Download.DATA_ROOT)
        file_path = os.path.join(Download.DATA_ROOT, Download.FILE_NAME)

        if os.path.exists(file_path):                       # 若已存在该文件
            statinfo = os.stat(file_path)
            if statinfo.st_size == Download.FILE_SIZE:      # 若该文件正确，直接返回 file_path
                print('Found and verified %s' % file_path)
                return file_path
            else:                                           # 否则，删除文件重新下载
                os.remove(file_path)

        download_url = Download.URL + Download.FILE_NAME
        print('Downloading %s ...' % download_url)
        filename, _ = urlretrieve(download_url, file_path)  # 下载数据
        print('Finish downloading')

        statinfo = os.stat(filename)
        if statinfo.st_size == Download.FILE_SIZE:          # 校验数据是否正确下载
            print('Found and verified %s' % filename)
        else:
            print(statinfo.st_size)
            raise Exception('Failed to verify ' + filename + '. Can you get to it with a browser ?')
        return filename


    ''' 读取压缩文件 '''
    @staticmethod
    def __readData(file_path):
        """Extract the first file enclosed in a zip file as a list of words"""
        with zipfile.ZipFile(file_path) as f:
            tmp_path = f.namelist()[0]                      # 压缩文件里只有一个 'text8' 文件
            print '\nReading %s/%s' % (file_path, tmp_path)
            content = f.read(tmp_path)                      # 读取文件内容
            print 'Finish Reading %s/%s' % (file_path, tmp_path)

        return content.split()                              # 将内容以任意空格分开，返回的数据类型为 list


    ''' 构建 data、reverse_dictionary、count 三个 data_set ; 具体看下面注释介绍 '''
    @staticmethod
    def __buildDataSet(words):
        # Counter 是一个无序的容器类型，用来跟踪记录值出现的次数；以字典的键值对形式存储，如 [ ('key', '出现了多少次'), ('the', 143212) ]
        print '\nCalculating counter for words ...'
        counter = collections.Counter(words)
        print 'Finish calculating'

        # 从 counter 中取出 出现最多 的 vocabulary_size - 1 个词
        common_words = counter.most_common(Download.VOCABULARY_SIZE - 1)

        count = [['UNK', -1]] + common_words    # UNK 之后用于记录 words 中没有出现在 count 里的词的数量; UNK 表示 unknown

        # 存储 { 'key': 'count 中对应的位置', 'UNK': 0, 'the': 1, 'of': 2, ..., 'fawn': 45848, ... }
        dictionary = {}
        for word, _ in count:
            dictionary[word] = len(dictionary)  # 记录 count 中每个词对应的位置 index 到 dictionary 里

        print '\nCalculating unk count ...'
        data = []
        unk_count = 0
        for word in words:
            if word in dictionary:              # 若出现在 dictionary 中的词，index 为 count 中的位置
                index = dictionary[word]
            else:
                index = 0                       # 没有出现在 dictionary 中的词，index 记为 0
                unk_count += 1                  # 计算没有出现在 dictionary 中的词的个数
            data.append(index)
        print 'Finish calculating'

        count[0][1] = unk_count                 # 设置 ['UNK', unk_count]

        # 生成类似 count 类型的 reverse_count ; 但存储方式变为 [ ('count 中的 index', 'key'), (1, 'the'), (2, 'of'), ... ]
        reverse_count = zip(dictionary.values(), dictionary.keys())

        # 强制转换为 dict 类型，存储方式为 { 'count 中的 index': 'key', 0: 'UNK', 1: 'the', 2: 'of', 3: 'and', ...}
        reverse_dictionary = dict(reverse_count)

        return data, reverse_dictionary, count


    ''' 保存数据 '''
    @staticmethod
    def __saveData(data, reverse_dictionary, count):
        tmp_data = (data, reverse_dictionary, count)

        print '\nSaving %s ... ' % Download.SAVE_PATH
        with open(Download.SAVE_PATH, 'wb') as f:
            pickle.dump(tmp_data, f, pickle.HIGHEST_PROTOCOL)
        print 'Finish saving'


class Data:

    def __init__(self):
        self.__load()   # 加载数据

        # 初始化变量
        self.__windowSize = 1                               # 目标词 左边 或 右边的 上下文词语数量
        self.__numOfContext = self.__windowSize * 2         # 上下文的词语数量为 skip_window 的两倍
        self.__dataIndex = -1                               # 目前处于 data 的所在位置 index


    ''' 加载数据 '''
    def __load(self):
        with open(Download.SAVE_PATH, 'rb') as f:
            self.__data, self.__reverseDict, self.__count = pickle.load(f)
        self.__dataLen = len(self.__data)                   # data 的数据量


    '''
      设置 window_size ；没有设置时，默认为 1
        window_size 指：目标词 左边 或 右边的 上下文词语数量
          如：一句话 ['Today', 'is', 'a', 'nice', 'day', 'and', 'I', 'want', 'to', 'go', 'sighting']
          假如 目标词为 'nice'，
            若 window_size 为 1, 则 'nice' 的上下文为 ['a', 'day'] ;
            若 window_size 为 2, 则 'nice' 的上下文为 ['is', 'a', 'day', 'and'] ;
    '''
    def setWindowSize(self, window_size):
        self.__windowSize = min(int(window_size), 1)
        self.__numOfContext = 2 * self.__windowSize         # 上下文的词语数量为 skip_window 的两倍


    ''' Skip-gram 模型时，获取下一 batch 的数据 '''
    def skipGramNextBatch(self, batch_size, loop = True):
        assert batch_size % self.__numOfContext == 0        # batch_size 的大小必须为 上下文的整数倍
        if not loop and self.__dataIndex >= self.__dataLen - self.__windowSize:
            self.__dataIndex = 0
            return None, None

        batch = []
        label = []
        span = 2 * self.__windowSize + 1                    # [ skip_window target skip_window ]

        if self.__dataIndex != -1:                          # 由于初始化 _buffer 时会额外增加 span 的位置，需要修正 data_index
            if self.__dataIndex >= span:
                self.__dataIndex -= span
            else:
                self.__dataIndex = self.__dataLen + self.__dataIndex - span
        else:
            self.__dataIndex = 0                            # 第一次使用 data_index，初始化为 0

        _buffer = collections.deque(maxlen=span)             # 初始化 _buffer
        for _ in range(span):                               # 用最开始位置的 span 个元素填满 _buffer
            _buffer.append(self.__data[self.__dataIndex])
            self.__dataIndex = (self.__dataIndex + 1) % self.__dataLen

        for i in range(batch_size // self.__numOfContext):
            target = self.__windowSize                      # 目标词在 _buffer 中的所在位置
            targets_to_avoid = [ self.__windowSize ]        # 取上下文时，必须避开的位置；初始值为目标词汇的位置

            # 将一个 window 里的目标词汇与上下文，分别加入 batch 和 label
            for j in range(self.__numOfContext):

                # 选取上下文词汇，此时的 target 为上下文词汇在 _buffer 中的位置；不能选中目标词，且不能选中已经选过的同一个 window 里的词
                while target in targets_to_avoid:
                    target = random.randint(0, span - 1)
                targets_to_avoid.append(target)             # 将当前将要取的 上下文词汇 加入 targets_to_avoid，避免重复取

                batch.append(_buffer[self.__windowSize])     # 目标词汇始终位于 window 里的中间位置；window_size * 2 + 1 == window 的大小
                label.append(_buffer[target])                # 将上下文词汇加入到 label

            # 取完一个 window 里的数据，window 往右移动一位
            _buffer.append(self.__data[self.__dataIndex])
            self.__dataIndex = (self.__dataIndex + 1) % self.__dataLen

            if not loop and self.__dataIndex >= self.__dataLen - self.__windowSize:
                break

        batch = np.array(batch)
        label = np.array(label).reshape((batch_size, 1))
        return batch, label


    ''' CBOW 模型时，获取下一 batch 的数据 '''
    def cBOWNextBatch(self, batch_size, loop = True):
        assert batch_size <= self.__dataLen - self.__windowSize * 2 # batch_size 最大不能超过 数据的总量 减 上写文词汇的个数
        if not loop and self.__dataIndex >= self.__dataLen - self.__windowSize:
            self.__dataIndex = 0
            return None, None

        batch = []
        label = []
        span = 2 * self.__windowSize + 1  # [ skip_window target skip_window ]

        if self.__dataIndex != -1:  # 由于初始化 _buffer 时会额外增加 span 的位置，需要修正 data_index
            if self.__dataIndex >= span:
                self.__dataIndex -= span
            else:
                self.__dataIndex = self.__dataLen + self.__dataIndex - span
        else:
            self.__dataIndex = 0  # 第一次使用 data_index，初始化为 0

        _buffer = collections.deque(maxlen=span)  # 初始化 _buffer
        for _ in range(span):  # 用最开始位置的 span 个元素填满 _buffer
            _buffer.append(self.__data[self.__dataIndex])
            self.__dataIndex = (self.__dataIndex + 1) % self.__dataLen

        for i in range(batch_size):
            batch.append([j for index, j in enumerate(_buffer) if index != self.__windowSize])
            label.append(_buffer[self.__windowSize])

            # 取完一个 window 里的数据，window 往右移动一位
            _buffer.append(self.__data[self.__dataIndex])
            self.__dataIndex = (self.__dataIndex + 1) % self.__dataLen

            if not loop and self.__dataIndex >= self.__dataLen - self.__windowSize:
                break

        batch = np.array(batch).reshape((batch_size, span - 1))
        label = np.array(label).reshape((batch_size, 1))
        return batch, label


    '''
     获取 reverseDict
     reverseDict 的存储方式为 { 'count 中的 index': 'key', 0: 'UNK', 1: 'the', 2: 'of', 3: 'and', ...}
    '''
    def getReverseDict(self):
        return self.__reverseDict


# Download.run()

# o_data = Data()
# batch, label = o_data.skipGramNextBatch(10)
