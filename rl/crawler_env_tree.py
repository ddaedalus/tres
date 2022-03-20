from utils.hyperparameters import *
from configuration.config import *
from crawling.treeFrontier import *
from crawling.closure import *
from crawling.crawler_sys import *
from utils.timeout import timeout
from utils.utils import *

import gym 
from gym import spaces

import logging

import numpy as np
from numpy.linalg import norm
import math
import pickle, time

class TreeCrawlerEnv(gym.Env):
    '''
        We define a Focused Crawler environment for selecting topic-relevant publications.

        The environment will not have any episodes, since the crawling task is to always discover and fetch new pages.
        When a page is fetched by the crawler, then the outlinks, that have never been seen before, are extracted. In
        theory, we should consider the fetched page as the state and its outlinks as the actions. However, because of 
        the inconsistent state and action space, that is each link is different from all the others, we consider that 
        for each outlink a pair (fetched page, outlink) is generated that would be the state-action pair of the crawler 
        environment. 

        In this environment, in order to demonstrate this shared representation of states and actions, we will take 
        the following considerations:
        
        - State:
        A vector of state_dim length, that is the shared representation of (fetched page, outlink)

        - Action:
        An integer, that is the id of the chosen url which the crawler will fetch in the next timestep. We consider that
        the action_space is (0, +Inf). We note that a url can have multiple actions, since it can probably be fetched by
        multiple pages (states). 
        
    '''
    def __init__(self, seed_urls, crawler_sys,
                 TOTAL_TIME_STEPS=TOTAL_TIME_STEPS, obs_shape=OBS_SHAPE):
        '''
            Params:
            - TOTAL_TIME_STEPS:         int, the maximun number of timesteps that the environment will go through
            - seed_urls:                list of String, the initial seed urls
            - crawler_sys:              CrawlerSys
            - obs_shape:                tuple, the shape of agent's state_space
            
            Details:
            - crawler_sys has the relevance_calculator - as the clf.
            - reward_function:          A trained classifier of webpages with two classes; 
                                        (1) relevant, (2) irrelevant
        '''
        super(TreeCrawlerEnv, self).__init__()

        self.__version__ = "0.0.1"
        logging.info(f"CrawlerEnv - Version {self.__version__}")

        # General variables defining the environment
        self.TOTAL_TIME_STEPS = TOTAL_TIME_STEPS
        self.current_step = 0

        # Define the agent's action space 
        self.action_space = spaces.Box(low=0, high=np.Inf, shape=())

        # Define the agent's state space
        self.obs_shape = obs_shape
        self.observation_space = spaces.Box(low=-np.Inf, high=np.Inf, shape=obs_shape, dtype=np.float32)

        # Focused Crawler variables
        self.seed_urls = seed_urls                      # initial seed docs 
        self.crawler_sys = crawler_sys
        self.has_extracted_counter = 0                  # pages that the crawler has extracted
        self.crawling_history_ids = {}                  # dict[id:int]=Webpage
        self.tree_frontier = TreeFrontier()             # The Crawl Tree Frontier
        self.closure = Closure()                        # The Crawl Closure
        self.domain_pages = {}                          # dict[domain] = num_of_pages_fetched_from_this_domain
        self.different_domains = {}                     # dict[domain] = 1 if at least one relevant page is found
        self.fetch_history = {}                         # dict["url"]=list of Webpage )  

        # Metric variables
        self.relevant = 0                               # number of relevant urls fetched
        return

    @timeout(10)
    def step(self, action: int):
        """
        The agent takes a step in the environment.
        Parameters:
            - action:   int
        
        Returns:
            - state, reward, done=False, info : tuple
                - state:    List[int]
                    an environment-specific object representing your observation of
                    the environment.
                - reward:   float
                    amount of reward achieved by the previous action. The scale
                    varies between environments, but the goal is always to increase
                    your total reward.
                - done:     Boolean
                    whether it's time to reset the environment again. Most (but not
                    all) tasks are divided up into well-defined episodes, and done
                    being True indicates the episode has terminated. (For example,
                    perhaps the pole tipped too far, or you lost your last life.)
                - info:     Dict
                    diagnostic information useful for debugging. It can sometimes
                    be useful for learning (for example, it might contain the raw
                    probabilities behind the environment's last state change).
                    However, official evaluations of your agent are not allowed to
                    use this for learning.
        """
        if self.current_step == self.TOTAL_TIME_STEPS:
            # Finish crawling
            return None, None, True, {}         # done = True

        # assert self.action_space.contains(np.array(action))
        self.current_step += 1

        try:
            # Take the corresponding action and, thus, the new state and the reward
            self.state_title = None                    # text (String)
            self.state_body = None                     # text (String)
            self.state_webpage = None                  # Webpage of the fetched url that was inside the frontier
            self.state = self._take_action(action)     # state representation of the fetched page
            reward = self._get_reward()
            self.state_webpage.setRelevance(relevance=reward)
            t3 = time.time()
            for message in self.crawler_sys.false_messages:
                try:
                    if re.search(message, self.state_body[:100]):
                        self.current_step -= 1
                        return False, None, False, {}
                except:
                    pass
            t4 = time.time()

        except:
            # Catch timeout exception
            self.state_webpage = False
            reward = None

        return self.state_webpage, reward, False, {}     # done = False

    def _take_action(self, action:int):
        '''
            Details:
                - Take the chosen action
                - Fetch the corresponding URL
                - Return the corresponding state (new fetched page -- only html body)
        
            Parameters:
                - action:     int

            Returns:
                Array | None
        '''
        # Get the corresponding url from the chosen action
        self.state_webpage = self.crawling_history_ids[action] 
        assert self.state_webpage.id == action
        url = self.state_webpage.url

        # Insert url to closure
        self.closure.push(url)
        
        # Fetch url and get its abstract and title
        t1 = time.time()
        self.state_body, self.state_title = self.crawler_sys.visit(url, content="both")     # text
        t2 = time.time()
        print(f"take_action visit: {t2-t1}")

        t1 = time.time()        
        obs = self.crawler_sys.create_instance_repr(self.state_body, url=url)               # (emb, key1, key2)
        t2 = time.time()

        # Passing the condition test for observation
        # assert self.observation_space.contains(np.mean(obs[0][0], axis=0))      # mean pooling dim: word_dim
        return obs      # self.state

    def _get_reward(self) -> float:
        '''
            The Reward function of environment.
            - 1.0 for Relevant
            - 0.0 for Irrelevant
        '''
        state_url = self.state_webpage.url
        y_body = self.crawler_sys.classify(self.state_body, state_url) 
        domain = self.crawler_sys.getDomain(state_url)
        if y_body >= 0.5:
            self.relevant += 1
            self.last_reward = 1.0
            if HUB_FEATURES and POLICY != "random":
                self.crawler_sys.domain_relevance[domain][1] += 1
            # Store relevant domain
            self.different_domains[domain] = 1
        else:
            self.last_reward = 0.0

        # HUB_FEATURES = True
        if HUB_FEATURES and POLICY != "random":
            # Change the domain_relevance_ratio
            self.crawler_sys.domain_relevance[domain][2] += 1
            self.crawler_sys.domain_relevance[domain][0] = self.crawler_sys.domain_relevance[domain][1] / self.crawler_sys.domain_relevance[domain][2] 

        # Increment num domain pages
        self.domain_pages[domain] += 1
        if VERBOSE and self.crawler_sys.times_verbose % VERBOSE_PERIOD == 0:
            print(f"Domain: {domain}; {self.domain_pages[domain]}")

        # Return reward
        return self.last_reward

    def extractStateActions(self):
        '''
            - Extract outlinks (actions) from current page (state) and make the shared representation,
            - Check whether these urls extracted have not been seen in closure before,  
            - Update the crawling_history_ids 
        
            Returns:
                not_seen_extracted:     list of Webpage
        '''
        # Extract actions
        extracted = self.crawler_sys.expand(self.state_webpage) 
        self.fetch_history[self.state_webpage.url] = extracted  # for replay buffer

        # Keep those that have not been seen before
        not_seen_extracted = []
        for page in extracted:
            domain = self.getDomain(page.url)
            try: self.domain_pages[domain]
            except: self.domain_pages[domain] = 0
            if self.closure.seen(page.url) or self.domain_pages[domain] >= MAX_DOMAIN_PAGES:  
                # Closure has seen before this URL
                continue
            self.has_extracted_counter += 1
            page.setID(self.has_extracted_counter)
            not_seen_extracted.append(page)
            self.crawling_history_ids[page.id] = page    # dict[id]=Webpage 
        del extracted
        return not_seen_extracted

    def reset(self):
        '''
            The reset method of a gym-like environment. The agent resets the environment.

            Returns:
                flatten_seed_webpages:  list of Webpage
        '''
        # Reset environment variables
        self.current_step = 0

        # Reset focused crawler variables and structures
        self.tree_frontier = TreeFrontier()
        self.closure = Closure()
        self.has_extracted_counter = 0 
        self.crawling_history_ids = {}                      # dict[id]=Webpage
        self.relevant = 0

        # Initialize closure: dict["url"]
        for url in self.seed_urls:
            self.closure.push(url)

        # Extract webpages from seed urls 
        seed_webpages = [] # list of list of Webpage
        for url in self.seed_urls:
            print(url)
            seed_webpage = Webpage(url=url, relevance=1.0, irrelevant_parents=0.0, 
                                   relevant_parents=0.0, relevant_father=0.0, relevance_dist=0.0)
            self.crawler_sys.getHTML(url)
            extracted = self.crawler_sys.expand(seed_webpage)
            self.fetch_history[url] = extracted
            seed_webpages.append( extracted )
    
        flatten_seed_webpages = [] # extracted webpages from seed urls, list of Webpage
        for group in seed_webpages:
            for page in group:
                if self.closure.seen(page.url):
                    # If closure has seen before this URL
                    continue
                self.has_extracted_counter += 1
                page.setID(self.has_extracted_counter)
                self.crawling_history_ids[page.id] = page   # dict[id]=Webpage
                flatten_seed_webpages.append(page)
        return flatten_seed_webpages

    def create_initial_state_actions(self, seed_urls, maxseqlen=MAXSEQLEN):
        return self.crawler_sys.create_initial_state_actions(seed_urls=seed_urls, maxseqlen=maxseqlen)

    def harvestRate(self):
        '''
            Harvest Rate (HR) metric
            -------------------
            We define:
                Harvest Rate = (NUMBER_OF_RELEVANT_PAGES) / (TOTAL_NUMBER_OF_FETCHED_PAGES)
        '''
        return self.relevant / self.current_step

    def render(self, mode='human', close=False) -> None: 
        '''
            Prints frontier and HR
        '''
        self.tree_frontier.print_tree()
        print("Harvest Rate =", self.harvestRate(), '\n')

    def getDomain(self, url):
        '''
            url:        String

            Returns     String; the domain of the given url
        '''
        return self.crawler_sys.getDomain(url)

    def _frontier(self):
        '''
            Returns:
                self.frontier.frontier:     List
        '''
        return self.tree_frontier.leafs

