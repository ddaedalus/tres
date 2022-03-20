from configuration.taxonomy import taxonomy_keywords
from keywords.taxonomy_stats import mean_value, tp
from utils.hyperparameters import *
from configuration.config import *
from utils.utils import normalize
from crawling.textPreprocessor import *
from keywords.keyword_appearance import keyword_appearance

from sklearn.metrics.pairwise import cosine_similarity 
import pickle
import numpy as np

def cos_sim(w1, w2):
    w1 = tp.to_TAG(w1)
    w2 = tp.to_TAG(w2)
    try:
        cos = cosine_similarity(np.reshape(normalize(tp.w2v[w1]), (1,-1)), 
                                np.reshape(normalize(tp.w2v[w2]), (1,-1)))[0][0]
    except:
        cos = -10
    return cos

def is_new_keyword(w, verbose=False):
    try:
        w_tag = tp.to_TAG(w)
    except:
        return False
    m = []
    for k in list(taxonomy_keywords.keys()):
        cos = cos_sim(w_tag, tp.to_TAG(k))
        if cos == -10: continue
        m.append(cos_sim(w_tag, tp.to_TAG(k)))
    if verbose: print(np.mean(m))
    if m == []: m = [0.0]
    if np.mean(m) > mean_value:
        return True
    return False

if __name__ == "__main__":

    with open(f"./files/{domain}.pickle", "rb") as fp:
        dd = pickle.load(fp)

    print(f"mean_value: {mean_value}")

    new_keywords = {}
    for body in list(dd.values()):
        body = body.lower()
        body = body.split()
        if body != []:
            for w in body:
                if not is_new_keyword(w) or (w in new_keywords) or (w in taxonomy_keywords): continue
                new_keywords[w] = 0
        print(new_keywords.keys())

    with open("./files/" + 'new_keywords_' + domain + '.pickle', 'wb') as handle:
        new_keywords = pickle.dump(new_keywords, handle)

    keyword_appearance()
