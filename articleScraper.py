import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from nlp.extractive import extractiveSummarize
from nlp.abstractive import abstractiveSummarize


def scrapeArticles(abstractive=False):
    url = "https://www.hindustantimes.com/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Referer": "https://www.hindustantimes.com/",
    }

    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, "html.parser")
    links = []

    topnews_div = soup.find("div", id="topnews")
    if topnews_div:
        world_section = topnews_div.find(
            "section",
            class_="worldSection sections ht-ad-holder noSponsorAd cohort_box",
        )
        if world_section:
            impression_divs = world_section.find_all(
                "div", class_="htImpressionTracking"
            )
            if impression_divs:
                for impression_div in impression_divs:
                    for article_div in impression_div.find_all(
                        "div",
                        class_=[
                            "cartHolder bigCart track timeAgo",
                            "cartHolder listView track timeAgo",
                        ],
                    ):
                        headline = article_div.find(["h2", "h3"], class_="hdg3")
                        if headline:
                            link = headline.find("a", href=True)
                            if link:
                                full_link = urljoin(url, link["href"])
                                links.append(full_link)

    articles = []

    for link in links:
        response = requests.get(link, headers=headers)
        soup = BeautifulSoup(response.content, "html.parser")
        article_data = {}
        story_div = soup.find("div", id="storyMainDiv")
        if story_div:
            title_tag = story_div.find("h1", class_="hdg1")
            if title_tag:
                article_data["title"] = title_tag.get_text(strip=True)

        description = []
        detail_div = soup.find("div", class_="detail")
        if detail_div:
            p_tags = detail_div.find_all("p")
            for p in p_tags:
                description.append(p.get_text(strip=True))

        article_data["description"] = (
            " ".join(description) if description else "No description found."
        )
        inputTitle = article_data["title"]
        inputText = article_data["description"]
        summary = extractiveSummarize(inputTitle, inputText)
        if abstractive:
            summary = abstractiveSummarize(summary)
        article_data["summary"] = summary
        article_data["link"] = link
        articles.append(article_data)

    return articles
