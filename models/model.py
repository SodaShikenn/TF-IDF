import math
from .utils import text_cut
from tqdm import tqdm # 因为语料比较多，所以计算过程很慢，可以使用tqdm库加载一个进度条，了解处理进度。

class TFIDFModel():
    def __init__(self):
        self.all_words = []
        self.vocab = set() # 去重
        self.corpus_words = []
    
    def load_corpus(self, corpus):
        for sentence in corpus:
            words = text_cut(text)
            self.all_words += words
            self.vocab = self.vocab.update(words)
            self.corpus_words.append(words)

    def compute_df(self,):
        self.tf_list = []
        for words in self.corpus_words:
            tf = {}
            for word in words:
                tf[word] = words.count(word) / len(words)
            self.tf_list.append(tf)

    def compute_idf(self):
        self.idf_dict = {}
        for word in self.vocab:
            num = sum([1 if word in words else 0 for words in self.corpus_words])
            self.idf_dict[word] = math.log(len(self.corpus_words) / (num+1))



for i in tqdm(lst, desc='comment'):
    pass