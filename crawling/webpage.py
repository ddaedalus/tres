from abc import ABC, abstractmethod
import numpy as np

class Action(ABC):
    # Abstract Class of the Action in Reinforcement Learning environment

    @abstractmethod
    def __init__(self, id, qvalue, x):
        '''
            id:         int, the id of the state-action
            qvalue:     float, the Q-value of the state-action
        '''
        self.id = id
        self.qvalue = qvalue
        

class StateAction(Action):
    # Abstract Class of the State-Action representation

    @abstractmethod
    def __init__(self, id, qvalue, x):
        '''
            id:         int, the id of the state-action
            qvalue:     float, the Q-value of the state-action
            x:          Array, the state-action representation
        '''
        self.id = id
        self.qvalue = qvalue
        self.x = x

    def setQvalue(self, qvalue):
        ''' 
            Method for setting the Q-value of the state-action

            Parameters:
                qvalue:     float
        '''
        self.qvalue = qvalue
        return  self

    def reshape(self):
        return np.reshape(self.x, ((1,-1)))

    @abstractmethod
    def identifier(self):
        pass

    @abstractmethod
    def __gt__(self, other):
        pass

    @abstractmethod
    def __lt__(self, other):
        pass     


class Webpage(StateAction):
    # The webpage representation

    def __init__(self, url="www.google.com", qvalue=0, x=None, id=0, 
                 relevant_parents=0.0, irrelevant_parents=0.0, relevant_father=0.0, relevance=None, 
                 prediction=None, relevance_dist=0.0, anchor=None):
        '''
            url:                String, the url of the webpage
            qvalue:             float, the Q-value of the webpage
            x:                  numpy array, the state-action representation of the webpage
            id:                 int
            relevant_parents:   float, the number of relevant parents of the url
            irrelevant_parents: float, the number of irrelevant parents of the url
            relevant_father:    float, 1.0 -> relevant father, 0.0 -> irrelevant father
            relevance:          float
            relevance_dist:     float, the distance from the last relevant webpage inside the path
            anchor:             String, the anchor text
        '''
        self.url = url
        self.qvalue = qvalue
        self.x = x
        self.id = id
        self.relevant_parents = relevant_parents
        self.irrelevant_parents = irrelevant_parents
        self.relevant_father = relevant_father
        self.relevance = relevance
        self.relevance_dist = relevance_dist
        self.anchor = anchor
        return 

    def setURL(self, url):
        '''
            Setter method for page URL
            
            Parameters:
                url:        String
        '''
        self.url = url
        return

    def setRelevance(self, relevance):
        '''
            - Parameters:
                - relevance:    float
        '''
        self.relevance = relevance

    def setID(self, _id):
        ''' 
            Setter method for page id
        
            Parameters:
                _id:        String
        '''
        self.id = _id 
        return

    def identifier(self):
        '''
            Returns the identifier of Webpage, that is url.
        '''
        return self.url

    def __gt__(self, other):
        # We use negative logic, since we use min-heap priority queue
        '''
            other:      Pair
        '''
        return self.qvalue < other.qvalue

    def __lt__(self, other):
        # We use negative logic, since we use min-heap priority queue
        '''
            other:      Pair
        '''
        return self.qvalue > other.qvalue

    def __str__(self):
        return self.url + "-" + str(self.qvalue)

    __repr__ =  __str__
    
