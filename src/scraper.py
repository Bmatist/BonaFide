from newspaper import Article

import requests
from bs4 import BeautifulSoup

def scrape_article(url):
    """
    Fetches the article using a hybrid approach:
    1. Try Newspaper3k (best for cleanup).
    2. Fallback to smart BeautifulSoup extraction if Newspaper3k fails or yields < 200 chars.
    """
    text = ""
    
    # Method 1: Newspaper3k
    try:
        article = Article(url)
        article.download()
        article.parse()
        text = article.text
    except Exception:
        pass # Fallback to BS4

    # Use Newspaper3k result if it looks valid and substantial
    if text and len(text) > 200:
        return text

    # Method 2: BeautifulSoup Fallback (Smart Extraction)
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # 1. Remove obvious noise
        for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form", "iframe", "noscript"]):
            tag.decompose()
            
        # 2. Remove elements by common class/id names for noise
        noise_keywords = ['comment', 'reply', 'sidebar', 'widget', 'related', 'ads', 'recommended', 'share', 'menu']
        for tag in soup.find_all(attrs={"class": True}):
            classes = " ".join(tag.get("class", [])).lower()
            if any(n in classes for n in noise_keywords):
                tag.decompose()
        for tag in soup.find_all(attrs={"id": True}):
            ids = tag.get("id", "").lower()
            if any(n in ids for n in noise_keywords):
                tag.decompose()

        # 3. Target main content container
        content_node = soup.find('article')
        if not content_node:
            content_node = soup.find('main')
        if not content_node:
            # Fallback to body but with the noise usage removed above
            content_node = soup.body

        if content_node:
            text = content_node.get_text(separator=' ', strip=True)
            
            # Clean up whitespace
            lines = [line.strip() for line in text.splitlines()]
            chunks = [phrase.strip() for line in lines for phrase in line.split("  ")]
            text = ' '.join(chunk for chunk in chunks if chunk)
            return text
            
        return "" # Failed to find content
        
    except Exception as e:
        raise Exception(f"Failed to scrape article: {str(e)}")
