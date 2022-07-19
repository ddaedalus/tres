from keywords.keyword_filter import *
from models.abcmodel import KwBiLSTM
from .textReprGenerator import *
from .htmlParser import *
from .url_extractor import *
from configuration.taxonomy import *

import tensorflow as tf
import requests
import gc


class CrawlerSys(URLextractor, HTMLParser, TextReprGenerator):
    # The Crawler class
    def __init__(self, keyword_filter, clf=None):
        '''
            clf:                ABCModel (X, X_shortcut1, X_shortcut2)
            keyword_filter:     KeywordFilter   
        '''
        self.clf = clf
        URLextractor.__init__(self)
        HTMLParser.__init__(self)
        TextReprGenerator.__init__(self, keyword_filter=keyword_filter)
        self.false_messages = [
            "cannot be found",
            "404 not found",
            "page not found",
            "access forbidden",
            "could not connect",
            "access denied",
            "permission denied"
        ]
        self.domain_relevance = {}      # dict[domain]=[domain_relevance_ratio, domain_num_relevants, domain_num_all]
        self.times = 1
        self.times_verbose = 0
        self.session_domains = {}       # dict[domain]=session
        self.session_domains_count = {}

    def visit(self, url, content="body"):
        '''
            A method for fetching the html body of the given url

            Params:
                - url:      String
                - content:  "body" | "title" | "both"
            - Returns:
                (The preprocessed html) body: list of words
        '''
        self.times += 1

        domain = self.getDomain(url)
        try:
            self.session = self.session_domains[domain]
            self.session_domains_count[domain] += 1
            if self.session_domains_count[domain] > 100:
                self.session = requests.Session()
                self.session_domains[domain] = self.session
                self.session_domains_count[domain] = 0
        except:
            self.session = requests.Session()
            self.session_domains[domain] = self.session
            self.session_domains_count[domain] = 0

        body = self.getHTML(url)
        # Only body
        if content == "body":
            body = self.getBody(url)
            if body == "" or body is None: 
                body = "Access denied, permission denied, not a given URL."
            return body.lower()
        # Only title
        elif content == "title":
            title = self.getTitle(url)
            if title == "" or title is None: 
                title = "Access denied, permission denied, not a given URL."
            return title.lower()
        elif content == "both":
            body = self.getBody(url)
            if body == "" or body is None: 
                body = "Access denied, permission denied, not a given URL."
            title = self.getTitle(url)
            if title == "" or title is None: 
                title = "Access denied, permission denied, not a given URL."
            return body.lower(), title.lower()  
        else:
            print("No param given in content")
            return None

    def expand(self, webpage, pooling="", padding=True, maxseqlen=MAXSEQLEN):
        '''
            A method for: 
            ! expandind the given webpage (with fixed url) e.g. the one with the highest Q-value from frontier,
            ! extracting the state features,
            ! extracting the outlinks/actions (features) and 
            ! creating the state-action pairs that will be passed afterwards to the action scorer.
            
            - Parameters:
                - webpage:  Webpage,    e.g. the webpage with the highest Q-value from frontier
                - pooling:  String
                - padding:  Boolean

                            REPRESENTATIONS:
                            ----------------
            State
            -----
            num of relevant parents of the path (n1), num of irrelevant parents of the path (n2)
            [ father's page relevance, dist_from_last_relevant, path_relevance_ratio ]

            Action  (With no HUB_FEATURES)
            ------
            [ keywords_urls_found, keywords_anchor_found, keyphrases_anchor_found, prob_is_relevant ]

            Action  (With HUB_FEATURES)
            ------
            [ keywords_urls_found, keywords_anchor_found, keyphrases_anchor_found, domain_relevance_ratio, unknown_domain_relevance, prob_is_relevant ]

            Returns:
                A list of Webpage, that is the extracted webpages (state-action pairs). Each Webpage has a url and an id 
                an id (int) that corresponds to an outlink url and, thus, an action.
        '''
        domain = self.getDomain(webpage.url)

        # State features
        relevant_father = webpage.relevance
        relevant_parents = (webpage.relevant_parents + 1) if relevant_father == 1.0 else webpage.relevant_parents
        irrelevant_parents = webpage.irrelevant_parents if relevant_father == 1.0 else (webpage.irrelevant_parents + 1)
        path_relevance_ratio = np.float32( relevant_parents / (relevant_parents + irrelevant_parents) )
        if relevant_father == 1.0:
            relevance_dist = 1.0
        else:
            relevance_dist = webpage.relevance_dist + 1     # all pages have a relevant parent in their path
        state = [relevant_father, quantize(1/relevance_dist), quantize(path_relevance_ratio)]   
        
        # Extract links/Actions from state and creat state-action pairs
        extracted_urls, extracted_titles = self.extractURLS(webpage.url, html=self.html)      
        len_extracted = len(extracted_urls)     

        # Check if no outlink is available
        if len_extracted == 0: return []

        # GPU limit 
        max_limit = 10000
        if len(extracted_titles) > 10000: 
            extracted_titles = extracted_titles[0:max_limit]
            extracted_urls = extracted_urls[0:max_limit]
            len_extracted = len(extracted_urls) 

        # Expand actions
        actions = []
        for i,anchor in enumerate(extracted_titles):
            action_url = extracted_urls[i]
            action_domain = self.getDomain(action_url) 
            if isinstance(anchor, str):
                ## Random Crawling
                if POLICY == "random" and USE_TREE == False:
                    actions.append( [1.0,1.0,1.0,1.0,1.0,1.0] )

                ## Focused Crawling    
                else:
                 
                    # Keywords and Keyphrases identified
                    keywords_urls_found = [ self.keyword_filter.find_keywords_bin(extracted_urls[i], url=True) ]
                    keywords_found = [ self.keyword_filter.find_keywords_bin(anchor, url=False) ]
                    phrases_found = [ self.keyword_filter.find_keyphrases(anchor) ]
                    
                    # HUB_FEATURES = False
                    if not HUB_FEATURES:
                        actions.append( keywords_urls_found + keywords_found + phrases_found )

                    # HUB_FEATURES = True
                    else:
                        try:
                            domain_relevance_ratio = [ quantize( self.domain_relevance[action_domain][0] ) ]
                            unknown_domain_relevance = [0.0]
                        except:
                            self.domain_relevance[action_domain] = [0.5, 0, 0]
                            domain_relevance_ratio = [0.5]
                            unknown_domain_relevance = [1.0]
                        actions.append( keywords_urls_found + keywords_found + phrases_found + domain_relevance_ratio + unknown_domain_relevance )   

            # Catch an extreme case
            if anchor is None:
                print("Fetch: No title")
                actions.append( [0.0, 0.0, 0.0] + [0.0] )       
            continue

        list_repr = [ self.create_instance_repr(extracted_titles[i], extracted_urls[i]) for i in range(len_extracted) ]

        if POLICY != "random" or USE_TREE == True:
            # Predict actions
            embs = np.array(list(map(lambda x: x[0], list_repr)))
            embs = np.reshape(embs, (len_extracted, -1, WORD_DIM))
            key1s = np.array(list(map(lambda x: np.reshape(x[1], (-1,)), list_repr)))
            key2s = np.array(list(map(lambda x: np.reshape(x[2], (-1,)), list_repr)))
            pred_actions = self.clf(embs, key1s, key2s, batch_size=1024)
            actions = [action + [pred_actions[i][0]] for i,action in enumerate(actions)]

        actions = np.array(actions)
        actions = np.asarray(actions).astype('float32')
        extracted_webpages = {} # dictionary of the form of dict["url"] = Webpage

        for i,url in enumerate(extracted_urls):
            # state-action representation x
            x = np.concatenate((state, actions[i,:])) #  x = [state, action]    
            
            # Catch an extreme exception (solved)
            for j in range(len(x)):
                if str(x[j]) == "nan":
                    x[j] = 0.5          
                    continue

            # extracted webpage (action)
            new_webpage = Webpage(url=url, x=x, 
                                  relevant_father=relevant_father, relevant_parents=relevant_parents, 
                                  irrelevant_parents=irrelevant_parents, relevance_dist=relevance_dist,
                                  anchor=extracted_titles[i])
            # add it to extracted webpages dict
            extracted_webpages[url] = new_webpage
            continue

        # Return a list of the extracted webpages
        extracted = list(extracted_webpages.values())
        if VERBOSE and self.times_verbose % VERBOSE_PERIOD == 0: 
            print("Fetched Citations:", len(extracted))

        return extracted       

    def create_initial_state_actions(self, seed_urls, maxseqlen=MAXSEQLEN, pooling="", padding=True):
        '''
            A method that generates the dataset seeds representation array 
            for the initialization of the experience replay buffer.

            - Parameters:
                - seed_urls:            list of String
                - maxseqlen:            int
                - pooling:              String
                - padding:              Boolean

            - Returns:
                - seed_experiences:     list of tuple: ( Webpage.x, Webpage.id, reward:1.0 )
                - seed_webpages:        list of Webpage
        '''
        assert isinstance(seed_urls, list)
        print("Creating the experiences from seeds...")
        
        # Initial State
        state = [0.0, 0.0, 0.0]  

        # Reward of seeds always 1.0 (relevant)
        reward = 1.0

        # Create seed experiences
        seed_experiences = []
        seed_webpages = []
        for i,url in enumerate(seed_urls):
            title = self.visit(url, content="title")
            print(title)

            ## Random Crawling
            if POLICY == "random" and USE_TREE == False:
                action = [1.0,1.0,1.0,1.0,1.0,1.0]
            
            ## Focused Crawling
            else:
                domain = self.getDomain(url)
                keywords_urls_found = [ self.keyword_filter.find_keywords_bin(url, url=True) ]
                keywords_found = [ self.keyword_filter.find_keywords_bin(title, url=False) ]
                phrases_found = [ self.keyword_filter.find_keyphrases(title) ]
                prob = self.classify(title, url)  
                self.domain_relevance[domain] = [1, 0, 0]

                # HUB_FEATURES = False
                if not HUB_FEATURES:
                    action = keywords_urls_found + keywords_found + phrases_found + [prob]

                # HUB_FEATURES = True
                else:
                    action = keywords_urls_found + keywords_found + phrases_found + [1.0] + [0.0] + [prob]
            x = np.concatenate((state, action))
            id = -1 - i
            seed_experiences.append( (x, id, reward) )
            seed_webpages.append(Webpage(url=url, x=x, id=id))
            continue
        return seed_experiences, seed_webpages

    def classify(self, doc, url=""):
        '''
            Classify a given doc using self.clf

            Params:
            - doc:      String
            - url:      String

            Returns:
            - prob: float; the probability belonging to the relevant class (class_id=0) 
        '''
        emb, key1, key2 = self.create_instance_repr(doc, url)
        if emb[0,0,0] == -10: return 0.0
        prob = self.clf(emb, key1, key2)[0][0]
        return prob


