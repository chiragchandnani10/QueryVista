#Basic scraper to scrape the paragraph tag out of the html file of the website
import requests
from bs4 import BeautifulSoup

def scrape_article(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        content = " ".join([p.get_text() for p in paragraphs])
        return content
    else:
        return None
