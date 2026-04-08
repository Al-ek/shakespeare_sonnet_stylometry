import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

roman_numerals = re.compile(r'^\s*[IVXLCDM]+\s*$', re.IGNORECASE | re.MULTILINE)


def clean_text(text: str) -> str:
    text = roman_numerals.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text.lower()


def ngram(authors, by_author, target_text, ngram_sizes=(3, 4, 5, 6)):
    ngram_results = {}

    cleaned_by_author = {}
    for author in authors:
        cleaned_by_author[author] = clean_text(by_author[author])

    cleaned_target = clean_text(target_text)

    for author in authors:
        ngram_results[author] = {}

    for n in ngram_sizes:
        docs = [cleaned_by_author[author] for author in authors] + [cleaned_target]

        vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(n, n),
            min_df=1,
            max_df=1.0,
            sublinear_tf=True,
            lowercase=False,
            norm="l2",
        )

        X = vectorizer.fit_transform(docs)
        X_target = X[-1]

        for i, author in enumerate(authors):
            X_author = X[i]
            score = cosine_similarity(X_target, X_author)[0, 0]
            ngram_results[author][f"{n}gram"] = float(score)

    return ngram_results