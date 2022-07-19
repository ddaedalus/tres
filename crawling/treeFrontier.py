from .webpage import *
from utils.hyperparameters import *
from configuration.config import *

import numpy as np
import random
import time
import pickle

class TreeNode:
    # The Node class of our Tree Frontier
    def __init__(self, exp_samples=None, frontier_samples=None, isleaf=False, parent=None):
        '''
            Params:
            - exp_samples:          tuple | list
            - frontier_samples:     list
            - isleaf:               bool
            - parent:               TreeNode | None
        '''
        self.parent = parent
        self.isleaf = isleaf

        # Check if it is root node
        if parent is None: 
            self.depth = 0
            self.leaf_id = 0
        else: self.depth = parent.depth + 1
            
        # Samples from Experience
        if type(exp_samples) == tuple:
            self.exp_samples_reprs = exp_samples[0]
            self.exp_samples_ids = exp_samples[1]
            self.exp_samples_rewards = exp_samples[2]
            self.exp_samples = exp_samples
        else:
            self.exp_samples = exp_samples                                                          # list of (array, int, float)
            self.exp_samples_reprs = np.array(list( map(lambda x: x[0], exp_samples) ) )            # array: shape=(len(rewards), URL_DIM)
            self.exp_samples_ids = np.array(list( map(lambda x: x[1], exp_samples) ) )              # array: shape=(len(rewards), )
            self.exp_samples_rewards = np.array(list( map(lambda x: x[2], exp_samples) ) )          # array: shape=(len(rewards), )
        self.size = len(self.exp_samples_rewards)

        # Samples from Frontier
        self.frontier_samples = frontier_samples
        try: self.frontier_size = len(self.frontier_samples)
        except: self.frontier_size = 0

        # Children (binary tree) - to be decided
        self.left = None                        # would be a TreeNode
        self.right = None                       # would be a TreeNode
        self.decision_criterion = None          # would be a (feature, value)
        return

    def split(self, left, right, decision_criterion, leaf_counter):
        '''
            Split node (self) given the related feature and value. Use the mean value of value and the previous value from self.exp_samples

            Params:
            - left:                 TreeNode; left child
            - right:                TreeNode; right child
            - decision_criterion:   tuple
            - leaf_counter:         int

            Returns left and right children
        '''
        # left child
        self.left = left
        self.left.leaf_id = leaf_counter + 1
        self.left.size = len(self.left.exp_samples_rewards)
        assert len(self.left.exp_samples_rewards) == len(self.left.exp_samples_ids)

        # right child
        self.right = right
        self.right.leaf_id = leaf_counter + 2
        self.right.size = len(self.right.exp_samples_rewards)
        assert len(self.right.exp_samples_rewards) == len(self.right.exp_samples_ids)

        # parent
        self.decision_criterion = decision_criterion
        self.isleaf = False
        del self.exp_samples
        del self.exp_samples_reprs
        del self.exp_samples_ids
        del self.exp_samples_rewards
        self.exp_samples = None
        self.exp_samples_reprs = None
        self.exp_samples_ids = None
        self.exp_samples_rewards = None

        # Transfer parent.frontier_samples to children using the decision_criterion
        feature = decision_criterion[0]
        value = decision_criterion[1]
        self.left.frontier_samples = [w for w in self.frontier_samples if w.x[feature] <= value]
        self.right.frontier_samples = [w for w in self.frontier_samples if w.x[feature] > value]
        self.left.frontier_size = len(self.left.frontier_samples)
        self.right.frontier_size = len(self.right.frontier_samples)
        del self.frontier_samples
        self.frontier_samples = None
        print(f"Splited {self}")
        return     

    def __str__(self):        
        if self.decision_criterion is None:
            return f"NODE {self.leaf_id}. Depth: {self.depth}, Exp_samples: {self.size}, Frontier_samples: {self.frontier_size}, Feature: {None}, Value: {None}"
        return f"NODE {self.leaf_id}. Depth: {self.depth}, Exp_samples: {self.size}, Frontier_samples: {self.frontier_size}, Feature: {FEATURES_NAMES[self.decision_criterion[0]+1]}, Value: {self.decision_criterion[1]}"

    __repr__ =  __str__


class TreeFrontier:
    # The Tree Frontier class of our Focused Crawler
    def __init__(self,
                 max_depth=MAX_DEPTH, 
                 min_samples_per_node_threshold=MIN_SAMPLES_PER_NODE_THRESHOLD,
                 min_samples_per_split=MIN_SAMPLES_PER_SPLIT,
                 url_dim=URL_DIM):
        '''
            A binary tree 
        '''
        self.root = None
        self.depth = None
        self.max_depth = max_depth
        self.min_samples_per_node_threshold = min_samples_per_node_threshold
        self.min_samples_per_split = min_samples_per_split
        self.url_dim = url_dim
        self.leaf_counter = 0
        self.leafs = {}         # dict[self.leaf_counter] = leaf Node
        return

    def initialize(self, initial_exp_samples, initial_frontier_samples):
        '''
            Params:
            - initial_exp_samples:          list of Tuple: <URL Repr, Webpage.id, Reward>; i.e list of (array, int, float)
            - initial_frontier_samples:     list of Webpage
        '''
        self.depth = 0 

        # Initialize root with initial_exp_samples and initial_frontier_samples
        self.root = TreeNode(exp_samples=initial_exp_samples, frontier_samples=initial_frontier_samples, parent=None, isleaf=True)          
        self.leafs[self.root.leaf_id] = self.root
        
        # Expand tree (root) using root.exp_samples
        self.expand_tree(self.root, run_children=True)
        del initial_exp_samples
        del initial_frontier_samples
        return

    def expand_tree(self, node, run_children=False):
        '''
            - node:             TreeNode; the current root of the (sub)tree to be expanded
            - run_children:     bool; if True run recursively for node.left and node.right

            It expands the tree starting from root.exp_samples and continues expanding the tree until no other split can be done.
        '''

        # Check if node can be splited [MAX_DEPTH, MIN_SAMPLES_PER_SPLIT]
        if not self.check_split_capability(parent=node): return

        best_var_red = 0
        best_decision_criterion = None
        best_left = None
        best_right = None
        for feature in range(self.url_dim):         # type(feature) = int
            # Sort exp_samples on feature
            sorted_idxs = node.exp_samples_reprs[:, feature].argsort()
            node.exp_samples_reprs = node.exp_samples_reprs[sorted_idxs]        # reprs
            node.exp_samples_ids = node.exp_samples_ids[sorted_idxs]            # ids
            node.exp_samples_rewards = node.exp_samples_rewards[sorted_idxs]    # rewards

            # Try splits on this feature, using node as the parent node for the split
            feature_array = node.exp_samples_reprs[:,feature]
            unique_values = np.unique(feature_array)
            for value in unique_values:
                left_idxs = (feature_array <= value)
                right_idxs = ~left_idxs
                left_size = np.sum(left_idxs)
                right_size = np.sum(right_idxs)
                if not self.check_valid_split(parent=node, left_size=left_size, right_size=right_size): 
                    del left_idxs
                    del right_idxs
                    continue
                assert left_size + right_size == node.size          # parent size was increased + 1
                left = TreeNode(exp_samples=(node.exp_samples_reprs[left_idxs], node.exp_samples_ids[left_idxs], node.exp_samples_rewards[left_idxs]), 
                                parent=node, isleaf=True)
                right = TreeNode(exp_samples=(node.exp_samples_reprs[right_idxs], node.exp_samples_ids[right_idxs], node.exp_samples_rewards[right_idxs]), 
                                parent=node, isleaf=True)
                var_red = self.variance_reduction_test(parent=node, left=left, right=right)
                if best_var_red < var_red:
                    best_var_red = var_red
                    best_decision_criterion = (feature, value)
                    best_left = left
                    best_right = right
                else:
                    del left
                    del right
                continue
            continue

        # Split found
        if best_var_red > 0:
            # Split parent node
            node.split(left=best_left, right=best_right, decision_criterion=best_decision_criterion, leaf_counter=self.leaf_counter)

            # Update tree variables and parent
            self.leaf_counter += 2
            del self.leafs[node.leaf_id]
            self.leafs[node.left.leaf_id] = node.left
            self.leafs[node.right.leaf_id] = node.right
            if self.depth < node.left.depth: self.depth = node.left.depth 
            if self.depth < node.right.depth: self.depth = node.right.depth

            # Recursively run children until no more splits can be done 
            if run_children:
                self.expand_tree(node.left)
                self.expand_tree(node.right)
        return

    def check_split_capability(self, parent):
        '''
            - parent:   TreeNode

            Returns True if a split can be done; else False
        '''
        if parent.size < self.min_samples_per_split or parent.depth >= self.max_depth: return False
        return True

    def check_valid_split(self, parent, left_size, right_size):
        '''
            - parent:   TreeNode
            - left:     int
            - right:    int

            Returns True if the split is valid (checking min_samples_per_node_threshold); else False
        '''
        if (left_size / parent.size) < self.min_samples_per_node_threshold: return False
        if (right_size / parent.size) < self.min_samples_per_node_threshold: return False
        return True

    def getLeaf(self, node, sample, flag="frontier", increment_size=True):
        '''
            - node:     TreeNode
            - sample:   Webpage | tuple
            - flag:     bool; "frontier" | "exp"
        '''
        def getLeaf_Frontier(node, increment_size=increment_size):
            '''
                Get the leaf corresponding to given frontier sample

                Params:
                - node:     TreeNode
                - flag:     bool; "frontier" | "exp"

                If node is leaf then returns node. Else returns node.right | node.left
            '''
            # sample must be Webpage
            assert isinstance(sample, Webpage)

            if increment_size:
                # Increment node frontier size 
                node.frontier_size += 1

            # Find the corresponding leaf node
            if node.isleaf: return node
            feature = node.decision_criterion[0]
            value = node.decision_criterion[1]
            if x[feature] <= value: return getLeaf_Frontier(node=node.left)
            return getLeaf_Frontier(node=node.right)

        def getLeaf_Exp(node, increment_size=increment_size):
            '''
                Get the leaf corresponding to given exp_sample

                Params:
                - node:     TreeNode

                If node is leaf then returns node. Else returns node.right | node.left
            '''

            # sample must be tuple
            assert isinstance(sample, tuple)
            
            if increment_size:
                # Increment node exp size 
                node.size += 1

            # Find the corresponding leaf node
            if node.isleaf: return node
            feature = node.decision_criterion[0]
            value = node.decision_criterion[1]
            if x[feature] <= value: return getLeaf_Exp(node=node.left)
            return getLeaf_Exp(node=node.right)

        if flag == "frontier": 
            x = np.reshape(sample.x, (-1,))
            return getLeaf_Frontier(node=node, increment_size=increment_size)
        elif flag == "exp": 
            x = np.reshape(sample[0], (-1,))
            return getLeaf_Exp(node=node, increment_size=increment_size)
        return       

    def addSample(self, sample, flag="frontier"):
        '''
            Add sample to tree using either addExpSample or addFrontierSample with the corresponding flag 
            
            Params:
            - sample:   Webpage
            - flag:     bool; "frontier" | "exp"
        '''
        def addFrontierSample(sample):
            '''
                Params:
                - sample:   Webpage
            ''' 
            # Sample must be a Webpage instance
            assert type(sample) == Webpage

            t1 = time.time()
            # Get the corresponding leaf node
            leaf_node = self.getLeaf(node=self.root, sample=sample, flag="frontier", increment_size=True)
            t2 = time.time()         
            if t2 - t1 > 0.5:   
                print(f"getLeaf_frontier: {t2-t1} secs")

            t1 = time.time()
            # Add the given Webpage frontier sample to that leaf node
            leaf_node.frontier_samples += [sample]
            t2 = time.time()       
            if t2 - t1 > 0.5:      
                print(f"+= : {t2-t1} secs")
            return

        def addExpSample(sample):
            '''
                Params:
                - sample:   tuple; (URL Repr, webpage.id, reward) -> (array, int, floay)
            '''
            # sample must be tuple
            assert type(sample) == tuple

            # Get the corresponding leaf node
            leaf_node = self.getLeaf(node=self.root, sample=sample, flag="exp", increment_size=True)

            # Add the given exp_sample to that leaf node
            repr_array = sample[0]
            webpage_id = sample[1]
            reward = sample[2]
            repr_array = np.reshape(repr_array, (1,-1))
            leaf_node.exp_samples_reprs = np.concatenate((leaf_node.exp_samples_reprs, repr_array))
            leaf_node.exp_samples_ids = np.concatenate((leaf_node.exp_samples_ids, [webpage_id]))
            leaf_node.exp_samples_rewards = np.concatenate((leaf_node.exp_samples_rewards, [reward]))
            
            # Split leaf node if is needed
            self.expand_tree(node=leaf_node, run_children=False)
            return

        if flag == "frontier": return addFrontierSample(sample=sample)
        elif flag == "exp": return addExpSample(sample=sample)
        return

    def delete_sample_from_leaf(self, leaf_id, idx):
        '''
            Delete the given (frontier) sample from leafs.frontier_samples
        
            Params:
            - leaf_id:      int; the leaf_id of the corresponding sample
            - idx:          int; the index of the corresponding sample in self.leafs[leaf_id].frontier_samples
        '''
        del self.leafs[leaf_id].frontier_samples[idx]
        self.leafs[leaf_id].frontier_size -= 1
        return

    def get_random_from_leaf(self, leaf_id):
        '''
            Get a random frontier_sample from leaf with the given leaf_id and delete it afterwards

            Params:
            - leaf_id:      int
        '''
        frontier_samples = self.leafs[leaf_id].frontier_samples
        random_idx = np.random.randint(0, len(frontier_samples))
        random_frontier_sample = frontier_samples[random_idx]
        del self.leafs[leaf_id].frontier_samples[random_idx]
        return random_frontier_sample 

    def getLeafs(self):
        '''
            Returns a list of TreeNode
        '''
        return list(self.leafs.values())

    def get_frontier_samples_from_leafs(self):
        '''
            From each leaf, get a random candidate frontier sample for predicting its Q-value.
        
            Returns a dict[leaf_id]= ( random_idx, frontier_sample(Webpage) )
        '''
        f = {}   
        random_idxs = []
        for leaf_id in self.leafs:
            frontier_samples = self.leafs[leaf_id].frontier_samples
            if len(frontier_samples) == 0: continue
            random_idx = np.random.randint(0, len(frontier_samples))
            f[leaf_id] = ( random_idx, frontier_samples[random_idx] )
        return f

    @staticmethod
    def variance_reduction_test(parent, left, right):
        '''
            - parent:   TreeNode
            - left:     TreeNode
            - right:    TreeNode

            Returns the variance reduction score (using the rewards of exp_samples)
        '''
        var_parent = np.var( parent.exp_samples_rewards )    
        var_left = np.var(list( left.exp_samples_rewards ))
        var_right = np.var(list( right.exp_samples_rewards ))
        var_reduction = var_parent - var_left * (left.size / parent.size) - var_right * (right.size / parent.size)
        return float(var_reduction)

    def print_leafs(self):
        leafs = self.getLeafs()     # list of TreeNode
        [print(leaf) for leaf in leafs]
        print(f"Number of leafs: {len(leafs)}")
        return

    def print_tree(self, node=None):
        '''
            Print tree utilizing BFS
        '''
        print("* * * * * * * * * * *Tree Frontier* * * * * * * * * * * *")
        print("----------------------------------------------------------")
        if node is None: node = self.root
        nodes = [node]
        while nodes != []:
            new_nodes = []
            for j,node in enumerate(nodes):
                print(node)
                l_left = [node.left] if node.left is not None else []
                l_right = [node.right] if node.right is not None else []
                new_nodes += l_left + l_right
            del nodes
            nodes = new_nodes
            del new_nodes
        print("----------------------------------------------------------")
        return


