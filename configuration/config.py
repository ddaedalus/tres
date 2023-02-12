# This is the configuration file of our implementation
# ----------------------------------------------------

# Domain
# the name of the domain to be fetched. Please don't forget to use a capital first letter for the domain word.
# the domain word is very important for the crawling process. Please use the real domain word (eg. "Sports", "Food" etc) 
domain = "Food"     # give the name of the domain of your interest  

# GPU
GPU_AVAILABLE = True

# CLASSIFICATION
OVERSAMPLING = False
OVERSAMPLING_RATIO = 0.3        # if oversampling
CLASSIFICATION_BATCH_SIZE = 256
CLASSIFICATION_EPOCHS = 25      # 25 proposed
FOLDS = 2                       # k-fold-cross-validation (k>1)

# CRAWLING
TOTAL_TIME_STEPS = 20000        # we used 20000
POLICY = "no random"            # use "random" or "no random"
USE_TREE = True                 # for POLICY = "random": USE_TREE = True -> TreeRandom Crawler, USE_TREE = False -> Random Crawler
MAX_DOMAIN_PAGES = 5           # maximum number of domain (web site) web pages to be fetched by the crawler
HUB_FEATURES = False            # no effect when POLICY = "random" 

# MAX_DOMAIN_PAGES fixed scheduling adaptation
ADAPTATION = False
ADAPTATION_STEP = int(TOTAL_TIME_STEPS/3)             
ADAPTATION_STEP_DIV = 2         # div   PLEASE USE != 0
MIN_MAX_DOMAIN_PAGES = 2        # the minimum of MAX_DOMAIN_PAGES after adaptation (online adjusting MAX_DOMAIN_PAGES)

# REPLAY BUFFER
BUFFER_CAPACITY = 20000         # Experience Replay capacity of DQN 

# Q-NETWORK
LEARNING_RATE = 0.001  
LR_DECAY = True
EPOCHS = 1
QNETWORK_VERBOSE_TRAIN = 0
REPLAY_PERIOD = 3

# AGENT
GAMMA = 0.99
TARGET_UPDATE_PERIOD = 500

# VERBOSE (main)
VERBOSE = True
VERBOSE_PERIOD = 1   

# SAVE_DATA_INFO
SUFFIX_STR = "food_wiki"	# Give a name for your suffix save file (without .filetype)
