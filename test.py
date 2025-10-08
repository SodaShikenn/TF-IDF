from models.utils import text_cut
from tqdm import tqdm
import time
from models.model import TFIDFModel

if __name__ == '__main__':
    model = TFIDFModel()
    corpus = [
        '我爱斋藤飞鸟',
        '斋藤飞鸟是我老婆'
    ]
    model.load_corpus(corpus)
    model.compute_tfidf()
    print(model.tfidf_list)

# if __name__ == '__main__':
#     LST = [1,2,3]
#     for i in tqdm(LST, desc='comment'):
#         print(i)
#         time.sleep(1)
        