import pandas as pd
from models.config import *
from models.model import TFIDFModel
from models.utils import *

def compute_tfidf_save():
    df = pd.read_csv(CORPUS_PATH, sep='\t', encoding='utf-8', lineterminator='\n', usecols=[0], names=['text'])
    corpus = df['text'].to_list()
    model = TFIDFModel()
    model.load_corpus(corpus)
    model.compute_tfidf()
    # 获取属性
    idf_dict = model.idf_dict
    tfidf_list = model.tfidf_list
    # 缓存文件
    dump_cache((corpus, idf_dict, tfidf_list), CACHE_PATH)

if __name__ == '__main__':
    compute_tfidf_save()

"""
经过这样一个缓存的过程, 我们就把18w条新闻标题, 转化成了一一对应的TFIDF值, 相当于把文本转化成了数值信息。
做检索时, 就只需要把TFIDF的缓存内容加载出来, 挨个计算, 取最大的K个值就可以了。
"""