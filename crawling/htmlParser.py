from bs4 import BeautifulSoup
import traceback
import re
import requests
import lxml.html
import time
import urllib
import numpy as np

from utils.timeout import timeout

class HTMLParser:
    # The HTMLParser class
    def __init__(self):
      self.html = ''
      self.session = requests.Session()

    @timeout(10)
    def getHTML(self, url):
        headers = [ {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.121 Safari/537.36"}, 
                    {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64; rv:47.0) Gecko/20100101 Firefox/47.3'}        
        ]  
        h = np.random.randint(0,2)
        try:
            r = self.session.get(url, headers=headers[h]) 
            html = r.text
        except:
            html = ""    
        self.html = html
        return html

    def getLang(self, url):
        request = requests.head(url)
        return request.headers

    def getTitle(self, url):
        html = self.html #self.getHTML(url)
        if html == "":
            return ""
        title_pattern = re.compile("<title>(.*?)</title>", re.IGNORECASE)
        # regex is an order of magnitude faster than beautifulsoup when extracting title
        try:
            res = title_pattern.search(html)
            if res:
                return res.group(1).strip()
            else:
                return ""
        except:
            traceback.print_exc()
            return ""

    def getBody(self, url):
        html = self.html #self.getHTML(url)
        if html == "":
            return ""
        try:
            soup = BeautifulSoup(html, "lxml")
            for script in soup(["script", "style"]):
                script.extract()
            text = soup.get_text()
            text = " ".join(text.split())
            if text == "" or text == " ":
                return self.getTitle(html)
            return text
        except:
            return self.getTitle(html)

    def getMeta(self, url):
        html = self.html #self.getHTML(url)
        """
            Extract title, description, keywords
            Returns:
            --------
            a lower case string that is the concatination of title, desc and keywords.
        """
        try:
            metadata = []
            soup = BeautifulSoup(html, 'lxml')
            title = soup.find('title')
            title_text = title.text if title else ""
            metadata.append(title_text.strip())
                
            metatags = soup.find_all('meta')
            for tag in metatags:
                if 'name' in tag.attrs.keys() and tag.attrs['name'].strip().lower() in ['description', 'keywords']:
                    try:
                        metadata.append(tag.attrs['content'].strip())
                    except:
                        print(tag.attrs.keys())

            res = ' '.join(metadata) 
            if not res:
                print("Empty Metadata")
            return res
        except:
            print("Metadata extraction fails")
            return ""


if __name__== "__main__":
    ## Just an example

    import time
    t1 = time.time()

    parser = HTMLParser()

    url = "https://computer.howstuffworks.com/virus.htm"

    parser.getHTML(url)

    # print(parser.html)

    title = parser.getTitle(url)
    # meta = parser.getMeta(url)
    body = parser.getBody(url)

    print("Title:")
    print(title)
    # print("meta:")
    # print(meta)
    print("body:")
    print(body)
