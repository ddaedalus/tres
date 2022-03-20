from tensorflow.keras.optimizers import Adam, SGD, RMSprop
import pickle
import pathlib
from pathlib import Path

from configuration.config import *

path = "./files/"

# TEXTS
MAXSEQLEN = 300
WORD_DIM = 300
SHORTCUT2 = 3
try:
    with open(f'{path}{domain}_SHORTCUT1.pickle', 'rb') as fp:
        SHORTCUT1 = pickle.load(fp) 
except:
    SHORTCUT1 = None
print(f"SHORTCUT1 = {SHORTCUT1}")

# CLASSIFICATION
CLASSIFICATION_METHOD = "KwBiLSTM" 
MAX_LIMIT = 1800        # the maximum number of relevant samples in classification
CLASSIFICATION_MODEL_SAVE = True

# CRAWLING
TREE_REFRESH_PERIOD = 10 if POLICY != "random" else 5000

# REPLAY BUFFER
if POLICY == "random":
    REPLAY_START_SIZE = TOTAL_TIME_STEPS
else:
    REPLAY_START_SIZE = 500 
BATCH_SIZE = 60
TAKE_BATCHES = 1        # number of training epochs, but each epoch uses 1 batch

# STATE-ACTION
STATE_DIM = 3
OBS_SHAPE = (WORD_DIM,)
if HUB_FEATURES or POLICY == "random": ACTION_DIM = 6
else: ACTION_DIM = 4

# Q-NETWORK
OPTIMIZER = Adam(LEARNING_RATE, clipvalue=1.0)
INPUT_DIM = STATE_DIM + ACTION_DIM 

# TREE-FRONTIER
if POLICY == "random": 
    MAX_DEPTH = 0 if not USE_TREE else 18
    MIN_SAMPLES_PER_NODE_THRESHOLD = 0.2
    MIN_SAMPLES_PER_SPLIT = 20
else:
    if MAX_DOMAIN_PAGES == 20:
        MIN_SAMPLES_PER_NODE_THRESHOLD = 0.2
        MIN_SAMPLES_PER_SPLIT = 10
        MAX_DEPTH = 18
    elif MAX_DOMAIN_PAGES == 10:
        MIN_SAMPLES_PER_NODE_THRESHOLD = 0.0
        MIN_SAMPLES_PER_SPLIT = 5
        MAX_DEPTH = 18
        REPLAY_START_SIZE = 300
    elif MAX_DOMAIN_PAGES == 100:
        MIN_SAMPLES_PER_NODE_THRESHOLD = 0.0
        MIN_SAMPLES_PER_SPLIT = 5
        MAX_DEPTH = 18
    elif MAX_DOMAIN_PAGES == 5:
        MIN_SAMPLES_PER_NODE_THRESHOLD = 0.0
        MIN_SAMPLES_PER_SPLIT = 5
        MAX_DEPTH = 18
        REPLAY_START_SIZE = 300
        BATCH_SIZE = 128
        TREE_REFRESH_PERIOD = 5
        REPLAY_PERIOD = 1
    else:
        MIN_SAMPLES_PER_NODE_THRESHOLD = 0.0
        MIN_SAMPLES_PER_SPLIT = 5
        MAX_DEPTH = 18
        REPLAY_START_SIZE = 300
URL_DIM = INPUT_DIM        

# SAVE_DATA_INFO
HUBS_STR = "HUBS" if HUB_FEATURES else "NO_HUBS"
if POLICY == "random": 
    if USE_TREE:
        CRAWLER_STR = "TreeRandomCrawl"
    else:
        CRAWLER_STR = "RandomCrawl"
else:
    CRAWLER_STR = "TreeRLFC"
machine = f"{CRAWLER_STR}_{TOTAL_TIME_STEPS}_SEEDS_1__MAX_{MAX_DOMAIN_PAGES}_{HUBS_STR}_{SUFFIX_STR}"
folder = f"./{domain}_{CRAWLER_STR}_{TOTAL_TIME_STEPS}_SEEDS_1_MAX_{MAX_DOMAIN_PAGES}_{HUBS_STR}/"
Path(f"{folder}").mkdir(parents=True, exist_ok=True)            # create folder if it does not exist

# TREE-FRONTIER FEATURE NAMES
if HUB_FEATURES:
    FEATURES_NAMES = {
        1:  "Father_Rel",
        2:  "Path_Rel_Perc",
        3:  "dist_from_last_relevant",
        4:  "keywords_urls_found", 
        5:  "keywords_anchor_found",
        6:  "keyphrases_anchor_found",
        7:  "domain_relevance_ratio",
        8:  "unknown_domain_relevance",
        9:  "rel_prob"
    }
else:
    FEATURES_NAMES = {
        1:  "Father_Rel",
        2:  "Path_Rel_Perc",
        3:  "dist_from_last_relevant",
        4:  "keywords_urls_found", 
        5:  "keywords_anchor_found",
        6:  "keyphrases_anchor_found",
        7:  "rel_prob"
    }
