import re

class Closure:
    # The Focused Crawler's Closure of visited pages

    def __init__(self):
        '''
            Closure is a dictionary of the form dict["URL"]: (Website | None)
        '''
        self.closure = {}
        self.size = 0 

    def seen(self, url):
        ''' 
            Method for checking if a given webpage has been visited before by the crawler
        
            Parameters:
                url:    String
        '''
        try:
            self.closure[url]
            return True
        except:
            try:
                url = re.sub("www.", "", url)
                self.closure[url]
                return True
            except:
                return False
        return

    def push(self, url):
        ''' 
            Method for inserting a url in the closure
        
            Parameters:
                url:    String
        '''
        self.closure[url] = None
        self.size += 1

    def printer(self):
        ''' 
            Method for printing frontier's elements
        '''
        return print(list(self.closure.keys()))

