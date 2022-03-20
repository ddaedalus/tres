from .keyword_filter import *
from configuration.taxonomy import taxonomy_keywords, taxonomy_phrases
from utils.hyperparameters import *
from configuration.config import *

import pickle, math
import numpy as np

def keyword_appearance():

    domains = ['Arts', 'Business', 'Computers', 'Health',  'News', 
               'Recreation', 'Reference', 'Science', 'Society', 'Sports']

    domain_num = -1
    if domain in domains:
        for i,name in enumerate(domains):
            if domain == name: domain_num = i
    if domain_num == -1:
        # Not found in domains
        domains.append(domain)
        domain_num = len(domains) - 1

    with open(path + 'new_keywords_' + domain + '.pickle', 'rb') as handle:
        new_keywords = pickle.load(handle)

    keyword_filter = KeywordFilter(taxonomy_keywords=taxonomy_keywords, new_keywords=new_keywords,
                                   taxonomy_phrases=taxonomy_phrases)

    for i,domain_name in enumerate(domains):

        with open(f'{path}{domain_name}.pickle', 'rb') as fp:
            d = pickle.load(fp) 

        sums = []
        history = {}    # dict[key] = total times found
        for doc in list(d.values()):
            dd = keyword_filter.find_keywords(doc)
            sums.append( np.sum(list(dd.values())) )
            for k in list(dd.keys()):
                if k in history:
                    history[k] += dd[k]
                else:
                    history[k] = dd[k]

        print(domain_name)
        print({k: v for k, v in sorted(history.items(), key=lambda item: item[1])[::-1]})
        print("Median keywords per body:", np.median(sums))
        print("Mean keywords per body:", np.mean(sums))
        print("std keywords per body:", np.std(sums))
        print("Max keywords per body:", np.max(sums))
        print("Min keywords per body:", np.min(sums))

        sum_array = np.array(sums)
        print("All docs:", len(sum_array))
        print("No keywords in:", len(sum_array[sum_array == 0]))
        print()
        print()

        if i == domain_num:
            SHORTCUT1 = math.ceil(np.mean(sums))

    print(f"SHORTCUT1: {SHORTCUT1}")
    with open(f'{path}{domain}_SHORTCUT1.pickle', 'wb') as fp:
        pickle.dump(SHORTCUT1, fp) 

    return
