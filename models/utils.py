import jieba
import logging
import pickle
from .config import * # 调用当前目录的config文件

jieba.setLogLevel(logging.INFO)

# 加载停用词
def load_stopwords():
    stopwords = open(STOPWORDS_PATH, encoding='utf-8-sig').read().split('\n')
    stopwords.extend(['\n'])
    return stopwords

# 分词和去停用词
def text_cut(text, cut_all=False):
    words = []
    stopwords = load_stopwords()
    for word in jieba.cut(text, cut_all=cut_all): # 全分词模式
        if word not in stopwords:
            words.append(word)
    return words


if __name__ == '__main__':
    load_stopwords()
    print(text_cut("我爱斋藤飞鸟。"))
