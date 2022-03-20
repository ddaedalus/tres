from utils.hyperparameters import *
from configuration.config import *
from .replay_buffer import *
from crawling.webpage import *
from utils.utils import *
from utils.timeout import timeout

from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf
from tensorflow.keras.activations import tanh
import time, gc


class Agent(ABC):
    # The Agent -- abstract class -- of the (tree) crawling environment

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def train(self):
        pass
        
    @abstractmethod
    def policy(self):
        pass


class TreeDDQNAgent(Agent):
    '''
        The Double Deep Q-Network Agent.

        Q-Network:          parameter θ
        Target Q-Network:   parameter θ-
    '''
    def __init__(self, env, q_network, target_q_network, batch_size=BATCH_SIZE, 
                target_update_period=TARGET_UPDATE_PERIOD, gamma=GAMMA):
        Agent.__init__(self)
        self.env = env
        self.target_update_period = target_update_period
        self.input_dim = self.env.obs_shape[0]
        self.batch_size = batch_size
        self.gamma = gamma
        self.q_network = q_network
        self.target_q_network = target_q_network
        self.buffer = ReplayBuffer(batch_size=batch_size)

    def initialize(self):
        '''
            Initialize agent networks.
        '''
        self.q_network.build()
        self.target_q_network.build()
        self.updateTarget()

    def decreaseLR(self):
        '''
            Decrease Learning Rate of Q-Network's model
        '''
        if self.q_network.lr <= 0.001:
            self.q_network.lr *= 0.9
            return     
        self.q_network.lr /= 10
        return

    def getLR(self):
        '''
            Returns Learning Rate
        '''
        return self.q_network.lr

    def check_for_target_update(self):
        '''
            Checks if target_update_period timesteps have passed, in order the target network to be updated
        
            Returns:
                True,   if agent should update target network
                False,  otherwise
        '''
        if self.env.current_step % self.target_update_period == 0:
            return True
        return False

    def evaluate_and_updateFrontier(self, _state_action):
        '''
            Calculates given state_actions' Q-values and inserts them to frontier.
        
            Parameters:
                _state_action:      StateAction | List of StateAction
        '''
        if not isinstance(_state_action, list):
            _state_action = [_state_action]
        t1 = time.time()
        for state_action in _state_action:
            # qvalue = float(self.q_network.predict(state_action.reshape()).numpy())
            # state_action.setQvalue(qvalue)
            assert isinstance(state_action, Webpage)
            assert state_action.id == self.env.crawling_history_ids[state_action.id].id   
            if self.env.closure.seen(state_action.url):
                # Check if this Webpage.url is already in env.closure
                continue
            self.env.tree_frontier.addSample(state_action, flag="frontier")
        t2 = time.time()
        return

    def updateFrontier(self, _state_action):
        '''
            Inserts given state_actions to frontier.
        
            Parameters:
                _state_action:      StateAction | List of StateAction
        '''
        if not isinstance(_state_action, list):
            _state_action = [_state_action]
        for state_action in _state_action:
            assert isinstance(state_action, Webpage)
            self.env.tree_frontier.addSample(state_action, flag="frontier")
        return

    def updateTarget(self):
        ''' 
            Copies the weights of Q-Network to the target network.
        '''
        weights = self.q_network.model.get_weights()
        self.target_q_network.model.set_weights(weights)
        return
    
    def policy(self):
        return self.tree_policy(policy=POLICY)

    # @timeout(20)
    def tree_policy(self, policy=POLICY, times=0):
        # try:
        f = self.env.tree_frontier.get_frontier_samples_from_leafs()        # d[leaf_id] = Webpage (random selected in each leaf)
        best_q_value = - 10000000

        ## Random Crawling or REPLAY_START_SIZE not reached 
        if policy == "random" or REPLAY_START_SIZE > self.env.current_step or times >= 850:
            ids = list(f.keys())
            best_leaf_id = np.random.choice(ids)
            best_q_value = 0    # arbitrarily fixed
            best_idx = f[best_leaf_id][0]
            best_state_action = f[best_leaf_id][1]

            # HUB_FEATURES = True
            if HUB_FEATURES and policy != "random":
                # Update domain_relevance_ratio
                url = best_state_action.identifier()
                domain = self.env.getDomain(url)
                best_state_action.x[-3] = quantize( self.env.crawler_sys.domain_relevance[domain][0] )

                # Update unknown_domain_relevance
                if self.env.crawler_sys.domain_relevance[domain][2] == 0.0:
                    best_state_action.x[-2] = 1.0
                else:
                    best_state_action.x[-2] = 0.0
        
        ## Focused Crawling and REPLAY_START_SIZE has been reached 
        else:
            for leaf_id in f:
                state_action = f[leaf_id][1]
                assert isinstance(state_action, Webpage)
                idx = f[leaf_id][0]

                # HUB_FEATURES = True
                if HUB_FEATURES:
                    # Update domain_relevance_ratio
                    url = state_action.identifier()
                    domain = self.env.getDomain(url)
                    state_action.x[-3] = quantize( self.env.crawler_sys.domain_relevance[domain][0] )

                    # Update unknown_domain_relevance
                    if self.env.crawler_sys.domain_relevance[domain][2] == 0.0:
                        state_action.x[-2] = 1.0
                    else:
                        state_action.x[-2] = 0.0
                
                # Calculate q-value
                qvalue = float(self.q_network.predict(state_action.reshape()).numpy())
                if qvalue > best_q_value:
                    best_leaf_id = leaf_id
                    best_q_value = qvalue
                    best_idx = idx
                    best_state_action = state_action
                continue

        assert best_state_action.id == self.env.crawling_history_ids[best_state_action.id].id      
        # Delete the best state_action from tree frontier (only once delete; not for every sample with the same URL)
        self.env.tree_frontier.delete_sample_from_leaf(leaf_id=best_leaf_id, idx=best_idx)

        # Update Webpage.qvalue
        best_state_action.setQvalue(best_q_value)

        # Update domain_pages  
        domain = self.env.getDomain(best_state_action.url)
        try: self.env.domain_pages[domain]
        except: self.env.domain_pages[domain] = 0

        # Check if this Webpage.url is already inside env.closure
        if self.env.closure.seen(best_state_action.url) or self.env.domain_pages[domain] >= MAX_DOMAIN_PAGES:
            # Run policy again ...
            self.refreshFrontierLeafs()
            best_state_action = self.tree_policy(policy=POLICY, times=times+1)
        else:
            if VERBOSE and self.env.crawler_sys.times_verbose % VERBOSE_PERIOD == 0:
                print(f"{WARNING}Leaf selected: {best_leaf_id}{ENDC}")

        return best_state_action
        
    def train(self):
        '''
            Perform training over a batch of data from the buffer and update the agent's Q-Network.
        '''
        # Train only after a defined number of timesteps have passed
        if REPLAY_START_SIZE > self.env.current_step:
            return         

        states_actions, actions, rewards = self.buffer.get_next()
        y_targets = np.empty(len(rewards))
        idx_to_fit = []
        for i, action in enumerate(actions):
            r = rewards[i]
            url = self.env.crawling_history_ids[action].url
            try:
                extracted = self.env.fetch_history[url]
            except:
                print("Agent not found fetched history")
                extracted = self.env.crawler_sys.expand(Webpage(url=url)) 
                self.env.fetch_history[url] = extracted
            x_new_state_actions = np.array([page.x for page in extracted]) # shape=( len(extracted), input_dim )
            try:
                # Q(s',argmax_a(;θ),θ')
                preds = self.q_network.predict(x_new_state_actions)
                try:
                    argmax_idx = int(tf.argmax(preds, axis=1))
                except:
                    argmax_idx = int(tf.argmax(preds))
                new_qvalue = float( self.target_q_network.predict(x_new_state_actions[argmax_idx:argmax_idx+1]) )
                idx_to_fit.append(i)
            except:
                continue
            # y_target = r + γQ(s',a'(θ),θ')
            y = r + self.gamma * new_qvalue
            y_targets[i] = y
        
        # Keep valid states_actions and y_targets
        states_actions = states_actions[idx_to_fit] 
        y_targets = y_targets[idx_to_fit]

        # Dataset for training
        try:
            train_ds = tf.data.Dataset.from_tensor_slices((states_actions, y_targets)).batch(self.batch_size)   
        except:
            print("Exception in agent.train.train_ds()")

        # Train Q-Network with respect to θ
        self.q_network.fit(train_ds)
        return

    def refreshFrontierLeafs(self):
        '''
            Refresh the frontier.leafs maintaining only the non-fetched URLs
        '''
        count_deleted = 0
        for i in self.env.tree_frontier.leafs:
            del_indexes = []
            for idx, w in enumerate(self.env.tree_frontier.leafs[i].frontier_samples):
                assert isinstance(w, Webpage)
                domain = self.env.crawler_sys.getDomain(w.url)
                try:
                    to_delete = self.env.domain_pages[domain] >= MAX_DOMAIN_PAGES
                except:
                    to_delete = False
                if to_delete or self.env.closure.seen(w.url):
                    del_indexes.append(idx)

            self.env.tree_frontier.leafs[i].frontier_size -= len(del_indexes)
            count_deleted += len(del_indexes)
            self.env.tree_frontier.leafs[i].frontier_samples = [k for j, k in enumerate(self.env.tree_frontier.leafs[i].frontier_samples) if j not in del_indexes]

        print(f"Deleted from frontier: {count_deleted}")
        return
