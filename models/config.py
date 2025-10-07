import os

BASE_PATH = os.path.dirname(__file__)
STOPWORDS_PATH = os.path.join(BASE_PATH, 'data/stopwords.txt')
CORPUS_PATH = os.path.join(BASE_PATH, 'data/news.txt')
CACHE_PATH = os.path.join(BASE_PATH, 'data/tfidf_cache.pkl')