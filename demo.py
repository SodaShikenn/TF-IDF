import math

corpus = [
    'my dog sat on my bed',
    'my cat sat on my knees',
]

all_words = []
vocab = set() # 去重
corpus_words = []

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




# dct = {'a':1, 'b':3, 'c':2}
# # 字典按值排序方法1
# lst = list(dct.items())
# sort_lst = sorted(lst, key= lambda x:x[1], reverse = True)

# # 字典按值排序方法2
# lst = [(v,k) for k,v in list(dct.items())]
# print(sorted(lst))
# exit()
"""
BUG分析
在计算IDF时,为了防止出现除0错误,在分母上加了1。log((N/(num+1))可能会出现三种取值：

(1)语料库足够大,N/(num+1)>1,IDF为正数,num越大,IDF值越小。

(2)语料库较小,仅有一篇文档不包含某个词,N/(num+1)=1,IDF为0,乘以任何数依然为0,导致TF值失效。

(3)语料库较小,每个文档中都含有某个词,N/(num+1)<1,IDF为负数,TF越大,TF-IDF值反而越小。

经过以上分析可以看出,要保证TF-IDF的本来含义,我们要保证IDF值为正数,也就是log()内的分数大于1。
"""

"""
TF-IDF的设计,本身就具有很多缺陷,其中一个就是在计算IDF时,如果语料库是同一主题,主题词会在不同文档中被重复提及,就会导致IDF值很小,从而降低重要性。

针对这个问题,研究者提出了改进的加权算法TF-IWF(Term Frequency - Inverse Word Frequency)。

IWF = log(语料库中所有词的个数/某个词在整个语料库中出现的次数)

这种加权方法,即使语料库中每个文档都包含主题词,IWF也不会接近0,更加精确的表达了重复出现的主题词,在整个语料库中的重要程度。

"""

iwf_dict = {}
N = len(all_words)
for word in vocab:
    num = all_words.count(word)
    iwf_dict[word] = math.log(N/num)
# print(iwf_dict)

tfiwf_list = []
for tf in tf_list:
    tfiwf = {}
    for word, tf_val in tf.items():
        tfiwf[word] = tf_val * iwf_dict[word]
    tfiwf_list.append(tfiwf)
# print(tfiwf_list)

for tfiwf in tfiwf_list:
    tfiwf_topk = sorted([(v,k) for k,v in tfiwf.items()], reverse=True)[:3]
    print([k for v,k in tfiwf_topk])