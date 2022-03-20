from crawling.htmlParser import *
from configuration.config import *

import pickle, time

if __name__ == "__main__":

    path = "./files/"

    parser = HTMLParser()

    # Read data.txt file that contains the URLs of initial data for classification 
    f = open(f"{path}data.txt", "r")
    urls = "".join(f.readlines()).split('\n')
    f.close()

    d = {}
    for i,url in enumerate(urls):
        if i % 50 == 0: print(i)
        parser.getHTML(url)
        d[url] = parser.getBody(url)

    with open(f"{path}{domain}.pickle", "wb") as fp:
        pickle.dump(d, fp)
