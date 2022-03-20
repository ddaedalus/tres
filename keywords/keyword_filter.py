import re
import time 

class KeywordFilter:
    def __init__(self, taxonomy_keywords={}, new_keywords={}, taxonomy_phrases={}):
        '''
            taxonomy_keywords:      dict[keyword] = int
            new_keywords:           dict[keyword] = int
            taxonomy_phrases:       dict[phrase] = int
        '''
        self.taxonomy_keywords_url = taxonomy_keywords
        self.new_keywords_url = new_keywords
        self.taxonomy_phrases = taxonomy_phrases

        # Create keywords dicts for anchor texts and bodies
        self.taxonomy_keywords = {}
        self.new_keywords = {}
        l = list(self.taxonomy_keywords_url.keys())
        for keyword in l:
            if keyword[0:1] == " ": continue
            self.taxonomy_keywords[f" {keyword} "] = 1
        
        l = list(self.new_keywords_url.keys())
        for keyword in l:
            if keyword[0:1] == " ": continue
            self.new_keywords[f" {keyword} "] = 1
        return

    def search(self, key):
        if re.search(key, self.doc): return 1
        return 0

    def find_keywords_bin(self, doc, url=False):
        '''
            Param:
                - doc:      String
                - url:      bool

            Returns:
                - res:      1 | 0
        '''
        self.doc = doc.lower()
        if url:
            # Search in taxonomy keywords
            for key in list(self.taxonomy_keywords_url.keys()):
                if re.search(key, self.doc): return 1

            # Search in new keywords
            for key in list(self.taxonomy_keywords_url.keys()):
                if re.search(key, self.doc): return 1           

        else:
            # Search in taxonomy keywords
            for key in list(self.taxonomy_keywords.keys()):
                if re.search(key, self.doc): return 1

            # Search in new keywords
            for key in list(self.new_keywords.keys()):
                if re.search(key, self.doc): return 1
        return 0

    def find_keywords(self, doc):
        '''
            Param:
                - doc:      String

            Returns:
                - res:      dict[key] = times found in doc > 0
        '''
        doc = doc.lower()
        res = {}
        # Search in taxonomy keywords
        for key in list(self.taxonomy_keywords.keys()):
            found = re.findall(key, doc)
            if found == []: 
                continue
            res[key] = len(found)
        # Search in new keywords
        for key in list(self.new_keywords.keys()):
            found = re.findall(key, doc)
            if found == []: continue
            res[key] = len(found)
        return res

    def find_keyphrases(self, doc):
        '''
            Param:
                - doc:      String

            Returns:
                - int:      1 if keyphrase found, 0 otherwise
        '''
        self.doc = doc
        doc = doc.lower()
        for key in list(self.taxonomy_phrases.keys()):
            if re.search(key, doc): return 1
        return 0
