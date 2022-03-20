from crawling.textPreprocessor import *
from .keyword_filter import *
from configuration.taxonomy import taxonomy, taxonomy_keywords, taxonomy_phrases, new_keywords
from utils.hyperparameters import *
from configuration.config import *

from sklearn.metrics.pairwise import cosine_similarity

tp = TextPreprocessor()

m = []
for w in list(taxonomy_keywords.keys()):
    for ww in list(taxonomy_keywords.keys()):
        if w == ww: continue
        try:
            cos = cosine_similarity(np.reshape(normalize(tp.w2v[tp.to_TAG(w)]), (1,-1)), 
                                    np.reshape(normalize(tp.w2v[tp.to_TAG(ww)]), (1,-1)))[0][0]
        except: continue
        m.append(cos)

print(f"{domain}")
print("mean:", np.mean(m))
print("max:", np.max(m))
print("min:", np.min(m))
print("std:", np.std(m))
print()

mean_value = np.mean(m)     # will be used for new_keywords extraction
