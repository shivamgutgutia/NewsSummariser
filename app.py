from flask import Flask, render_template
from articleScraper import scrapeArticles
import os

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/retrieve-articles-extractive")
def retrieveArticlesExtractive():
    articles = scrapeArticles()
    return render_template("articles.html", articlesList=articles)


@app.route("/retrieve-articles-abstractive")
def retrieveArticlesAbstractive():
    articles = scrapeArticles(abstractive=True)
    return render_template("articles.html", articlesList=articles)


if __name__ == "__main__":
    app.run(port=int(os.getenv("PORT", 5000)), host="0.0.0.0")
