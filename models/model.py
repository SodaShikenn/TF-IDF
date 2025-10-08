import math
from .utils import text_cut
from tqdm import tqdm # 因为语料比较多，所以计算过程很慢，可以使用tqdm库加载一个进度条，了解处理进度。

class TFIDFModel():
    def __init__(self):
        self.all_words = []
        self.vocab = set() # 去重
        self.corpus_words = []
    
    def load_corpus(self, corpus):
        for sentence in tqdm(corpus, desc='load corpus'):
            # words = sentence.strip().split(' ') # 英文分词
            words = text_cut(sentence) # 基于jieba的中文分词
            self.corpus_words.append(words)
            self.all_words += words
            self.vocab.update(words)

    def compute_tf(self):
        tf_list = []
        for words in tqdm(self.corpus_words, desc='compute tf'):
            tf = {}
            for word in words:
                tf[word] = words.count(word) / len(words)
            tf_list.append(tf)
        self.tf_list = tf_list

    def compute_idf(self):
        idf_dict = {}
        for word in tqdm(self.vocab, desc='compute idf'):
            num = sum([1 if word in words else 0 for words in self.corpus_words])
            idf_dict[word] = math.log(len(self.corpus_words) / (num+1))
        self.idf_dict = idf_dict

    def compute_tfidf(self):
        self.compute_tf()
        self.compute_idf()
        tfidf_list = []
        for tf in tqdm(self.tf_list, desc='compute tfidf'):
            tfidf = {}
            for word, tf_val in tf.items():
                tfidf[word] = tf_val * self.idf_dict[word]
            tfidf_list.append(tfidf)
        self.tfidf_list = tfidf_list
        return tfidf_list