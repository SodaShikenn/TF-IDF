from models.config import *
from models.utils import *
import numpy as np

# lst = [4,7,9,3,2]
# res = np.argsort(lst)[::-1][:3]
# print(res)
# exit()

def search(text, topK=5):
    # 搜索内容分词
    words = text_cut(text, cut_all=True)
    corpus, idf_dict, tfidf_list = load_cache(CACHE_PATH)
    # 搜索词过滤，语料库中未出现的词就不用遍历了
    words = [word for word in words if word in idf_dict]
    # 计算语料库中的句子，与搜索词的相关度
    score_list = []
    for tfidf in tfidf_list:
        score_list.append(sum([tfidf.get(word, 0) for word in words])) # 列表内18w个返回值
    # 按相关度从大到小排序，返回id值
    ids = [id for id in np.argsort(score_list)[::-1] if score_list[id]!=0] # 索引值的逆序排列
    return ids[:topK], [corpus[id] for id in ids[:topK]]

if __name__ == '__main__':
    print(search('万科房地产公司怎么样？', 5))