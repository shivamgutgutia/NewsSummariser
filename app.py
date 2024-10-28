from flask import Flask, render_template
from articleScraper import scrapeArticles

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/retrieve-articles")
def retrieveArticles():
    articles = scrapeArticles()
    return render_template("articles.html", articlesList=articles)


if __name__ == "__main__":
    app.run(host="0.0.0.0")
