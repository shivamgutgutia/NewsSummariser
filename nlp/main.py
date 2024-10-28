import re, nltk, math
from collections import Counter

stopwords = {
    "",
    ",",
    "-",
    ".",
    "0",
    "1",
    "10",
    "2",
    "2012",
    "2013",
    "2014",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "a",
    "about",
    "above",
    "across",
    "after",
    "afterwards",
    "again",
    "against",
    "ako",
    "all",
    "almost",
    "alone",
    "along",
    "already",
    "also",
    "although",
    "always",
    "am",
    "among",
    "amongst",
    "amoungst",
    "amount",
    "an",
    "and",
    "ang",
    "another",
    "any",
    "anyhow",
    "anyone",
    "anything",
    "anyway",
    "anywhere",
    "april",
    "are",
    "around",
    "as",
    "at",
    "august",
    "back",
    "be",
    "became",
    "because",
    "become",
    "becomes",
    "becoming",
    "been",
    "before",
    "beforehand",
    "behind",
    "being",
    "below",
    "beside",
    "besides",
    "between",
    "beyond",
    "both",
    "bottom",
    "but",
    "by",
    "call",
    "can",
    "can't",
    "cannot",
    "co",
    "con",
    "could",
    "couldn't",
    "de",
    "december",
    "describe",
    "detail",
    "did",
    "do",
    "done",
    "down",
    "due",
    "during",
    "e",
    "each",
    "eg",
    "eight",
    "either",
    "eleven",
    "else",
    "elsewhere",
    "empty",
    "enough",
    "etc",
    "even",
    "ever",
    "every",
    "everyone",
    "everything",
    "everywhere",
    "except",
    "february",
    "few",
    "fifteen",
    "fifty",
    "fill",
    "find",
    "fire",
    "first",
    "five",
    "for",
    "former",
    "formerly",
    "forty",
    "found",
    "four",
    "fri",
    "friday",
    "from",
    "front",
    "full",
    "further",
    "get",
    "give",
    "go",
    "got",
    "government",
    "had",
    "has",
    "hasnt",
    "have",
    "he",
    "hence",
    "her",
    "here",
    "hereafter",
    "hereby",
    "herein",
    "hereupon",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "home",
    "homepage",
    "how",
    "however",
    "hundred",
    "i",
    "ie",
    "if",
    "in",
    "inc",
    "indeed",
    "inquirer",
    "into",
    "is",
    "it",
    "it's",
    "its",
    "itself",
    "january",
    "july",
    "june",
    "just",
    "keep",
    "ko",
    "last",
    "latter",
    "latterly",
    "least",
    "less",
    "like",
    "ltd",
    "made",
    "make",
    "manila",
    "many",
    "march",
    "may",
    "me",
    "meanwhile",
    "might",
    "mill",
    "mine",
    "mon",
    "monday",
    "more",
    "moreover",
    "most",
    "mostly",
    "move",
    "much",
    "must",
    "my",
    "myself",
    "na",
    "name",
    "namely",
    "neither",
    "never",
    "nevertheless",
    "new",
    "news",
    "newsinfo",
    "next",
    "ng",
    "nine",
    "no",
    "nobody",
    "none",
    "noone",
    "nor",
    "not",
    "nothing",
    "november",
    "now",
    "nowhere",
    "o",
    "october",
    "of",
    "off",
    "often",
    "on",
    "once",
    "one",
    "only",
    "onto",
    "or",
    "other",
    "others",
    "otherwise",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "part",
    "people",
    "per",
    "percent",
    "perhaps",
    "philippine",
    "photo",
    "please",
    "pm",
    "police",
    "put",
    "rappler",
    "rapplercom",
    "rather",
    "re",
    "reuters",
    "sa",
    "said",
    "same",
    "sat",
    "saturday",
    "says",
    "section",
    "see",
    "seem",
    "seemed",
    "seeming",
    "seems",
    "september",
    "several",
    "she",
    "should",
    "show",
    "side",
    "since",
    "sincere",
    "six",
    "sixty",
    "so",
    "some",
    "somehow",
    "someone",
    "something",
    "sometime",
    "sometimes",
    "somewhere",
    "sports",
    "still",
    "stories",
    "story",
    "such",
    "sun",
    "sunday",
    "t",
    "take",
    "ten",
    "than",
    "that",
    "the",
    "their",
    "them",
    "themselves",
    "then",
    "thence",
    "there",
    "thereafter",
    "thereby",
    "therefore",
    "therein",
    "thereupon",
    "these",
    "they",
    "thickv",
    "thin",
    "third",
    "this",
    "those",
    "though",
    "three",
    "through",
    "throughout",
    "thru",
    "thu",
    "thursday",
    "thus",
    "time",
    "to",
    "together",
    "too",
    "top",
    "toward",
    "towards",
    "tue",
    "tuesday",
    "tweet",
    "twelve",
    "twenty",
    "two",
    "u",
    "un",
    "under",
    "until",
    "up",
    "upon",
    "us",
    "use",
    "very",
    "via",
    "want",
    "was",
    "we",
    "wed",
    "wednesday",
    "well",
    "were",
    "what",
    "whatever",
    "when",
    "whence",
    "whenever",
    "where",
    "whereafter",
    "whereas",
    "whereby",
    "wherein",
    "whereupon",
    "wherever",
    "whether",
    "which",
    "while",
    "whither",
    "who",
    "whoever",
    "whole",
    "whom",
    "whose",
    "why",
    "will",
    "with",
    "within",
    "without",
    "would",
    "yahoo",
    "year",
    "years",
    "yet",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "yun",
    "yung",
}


def splitWords(text):
    text = re.sub(r"[^\w ]", "", text)
    return [x.strip(".").lower() for x in text.split()]


def splitSentences(text):
    import nltk.data

    tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")

    sentences = tokenizer.tokenize(text)
    sentences = [x.replace("\n", "") for x in sentences if len(x) > 10]
    return sentences


def keywords(text):
    NUM_KEYWORDS = 10
    text = splitWords(text)
    if text:
        num_words = len(text)
        text = [x for x in text if x not in stopwords]
        freq = {}
        for word in text:
            if word in freq:
                freq[word] += 1
            else:
                freq[word] = 1

        min_size = min(NUM_KEYWORDS, len(freq))
        keywords = sorted(freq.items(), key=lambda x: (x[1], x[0]), reverse=True)
        keywords = keywords[:min_size]
        keywords = dict((x, y) for x, y in keywords)

        for k in keywords:
            articleScore = keywords[k] * 1.0 / max(num_words, 1)
            keywords[k] = articleScore * 1.5 + 1
        return dict(keywords)
    else:
        return dict()


def titleScore(title, sentence):
    if title:
        title = [x for x in title if x not in stopwords]
        count = 0.0
        for word in sentence:
            if word not in stopwords and word in title:
                count += 1.0
        return count / max(len(title), 1)
    else:
        return 0


def sentencePosition(i, size):
    normalized = i * 1.0 / size
    if normalized > 1.0:
        return 0
    elif normalized > 0.9:
        return 0.15
    elif normalized > 0.8:
        return 0.04
    elif normalized > 0.7:
        return 0.04
    elif normalized > 0.6:
        return 0.06
    elif normalized > 0.5:
        return 0.04
    elif normalized > 0.4:
        return 0.05
    elif normalized > 0.3:
        return 0.08
    elif normalized > 0.2:
        return 0.14
    elif normalized > 0.1:
        return 0.23
    elif normalized > 0:
        return 0.17
    else:
        return 0


def title_score(title, sentence):
    if title:
        title = [x for x in title if x not in stopwords]
        count = 0.0
        for word in sentence:
            if word not in stopwords and word in title:
                count += 1.0
        return count / max(len(title), 1)
    else:
        return 0


def title_score(title, sentence):
    if title:
        title = [x for x in title if x not in stopwords]
        count = 0.0
        for word in sentence:
            if word not in stopwords and word in title:
                count += 1.0
        return count / max(len(title), 1)
    else:
        return 0


def lengthScore(sentence_len):
    return 1 - math.fabs(20 - sentence_len) / 20


def sbs(words, keywords):
    score = 0.0
    if len(words) == 0:
        return 0
    for word in words:
        if word in keywords:
            score += keywords[word]
    return (1.0 / math.fabs(len(words)) * score) / 10.0


def dbs(words, keywords):
    if len(words) == 0:
        return 0
    summ = 0
    first = []
    second = []

    for i, word in enumerate(words):
        if word in keywords:
            score = keywords[word]
            if first == []:
                first = [i, score]
            else:
                second = first
                first = [i, score]
                dif = first[0] - second[0]
                summ += (first[1] * second[1]) / (dif**2)
    k = len(set(keywords.keys()).intersection(set(words))) + 1
    return 1 / (k * (k + 1.0)) * summ


def score(sentences, titleWords, keywords):
    senSize = len(sentences)
    ranks = Counter()
    for i, s in enumerate(sentences):
        sentence = splitWords(s)
        titleFeature = titleScore(titleWords, sentence)
        sentenceLength = lengthScore(len(sentence))
        sentencePositionValue = sentencePosition(i + 1, senSize)
        sbsFeature = sbs(sentence, keywords)
        dbsFeature = dbs(sentence, keywords)
        frequency = (sbsFeature + dbsFeature) / 2.0 * 10.0
        totalScore = (
            titleFeature * 1.5
            + frequency * 2.0
            + sentenceLength * 1.0
            + sentencePositionValue * 1.0
        ) / 4.0
        ranks[(i, s)] = totalScore
    return ranks


def summarizeUtil(title="", text="", maxSents=5):
    if not text or not title or maxSents <= 0:
        return []

    summaries = []
    sentences = splitSentences(text)
    keys = keywords(text)
    titleWords = splitWords(title)

    ranks = score(sentences, titleWords, keys).most_common(maxSents)
    for rank in ranks:
        summaries.append(rank[0])
    summaries.sort(key=lambda summary: summary[0])
    return [summary[1] for summary in summaries]


def summarize(title, text):
    textKeywords = list(keywords(text).keys())
    titleKeywords = list(keywords(title).keys())
    maxSentences=5
    return " ".join(summarizeUtil(title=title, text=text, maxSents=maxSentences))
