import math

class TFIDFModel():
    def __init__(self):
        self.all_words = []
        self.vocab = set() # 去重
        self.corpus_words = []

for sentence in corpus:
    words = sentence.strip().split(' ')
    corpus_words.append(words)
    all_words += words
    vocab.update(words)

# TF = 某个词在文本中出现的次数 / 句子的总词数
# 例如：TF(dog) = [1/6, 0/6]
tf_list = []
for words in corpus_words:
    tf = {}
    for word in words:
        tf[word] = words.count(word) / len(words)
    tf_list.append(tf)

# IDF = log(语料库句子总数 / (包含该词的文档数 + 1))
# 例如：IDF(dog) = log(2/(1+1))
idf_dict = {}
N = len(corpus_words)
for word in vocab:
    num = sum([1 if word in words else 0 for words in corpus_words])
    idf_dict[word] = math.log(N/(num+1))


tfidf_list = []
for tf in tf_list:
    tfidf = {}
    for word, tf_val in tf.items():
        tfidf[word] = tf_val * idf_dict[word]
    tfidf_list.append(tfidf)
# print(tfidf_list)


for tfidf in tfidf_list:
    tfidf_topk = sorted([(v,k) for k,v in tfidf.items()], reverse=True)[:3]
    # print([k for v,k in tfidf_topk])