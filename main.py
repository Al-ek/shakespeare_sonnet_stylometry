from pathlib import Path
import matplotlib.pyplot as plt
import re
import nltk
import delta
import chi
import ngram
import csv
import argparse
import joblib
import pandas
import os
import train

BY_AUTHOR_SONNET_CORPUS = Path("by_author_sonnet_corpus")
MODEL_TRAINING_SONNETS = Path("model_training_sonnets")
MODEL_EVALUATION_SONNETS = Path("model_evaluation_sonnets")
MODEL_TRAINING_DATASET_FILE = "model_dataset.csv"

roman_numerals = re.compile(r'^\s*[IVXLCDM]+\s*$', re.IGNORECASE | re.MULTILINE)

# The amount of characters to be included in the authors stylometry corpus
# enter -1 for all
character_count = -1

authors = ("Gem", "Gemini", "Shakespeare",
           "ChatGPT", "Claude", "Copilot", "Preplexity")


def tokenize_target(target_file):
    with open(target_file, 'r', encoding="utf-8") as file:
        target_text = file.read()
        target_text = roman_numerals.sub("", target_text)
        target_text = target_text.strip()
        tokens = nltk.word_tokenize(target_text)
        return ([token for token in tokens if any(c.isalpha() for c in token)])


def extract_author_data():
    by_author = {}
    for author in BY_AUTHOR_SONNET_CORPUS.iterdir():
        file = author / "sonnets.txt"
        if file.exists():
            text = file.read_text(encoding="utf-8")
            # remove roman numerals
            text = roman_numerals.sub("", text)
            text = text.strip()
            by_author[author.name] = text[:character_count]
    return by_author


def extract_author_data_tokens(by_author):
    by_author_tokens = {}
    for author in by_author:
        tokens = nltk.word_tokenize(by_author[author])
        by_author_tokens[author] = ([token for token in tokens if any(c.isalpha() for c in token)])
    return by_author_tokens


def add_to_csv(chi_results, delta_results, ngram_results, true_author):

    row = {}

    # enforce fixed author order
    for author in authors:
        row[f"chi2_{author}"] = chi_results[author]

    for author in authors:
        row[f"delta_{author}"] = delta_results[author]

    for author in authors:
        row[f"ngram_3_{author}"] = ngram_results[author]["3gram"]
        row[f"ngram_4_{author}"] = ngram_results[author]["4gram"]
        row[f"ngram_5_{author}"] = ngram_results[author]["5gram"]
        row[f"ngram_6_{author}"] = ngram_results[author]["6gram"]

    row["true_author"] = true_author

    file_exists = Path(MODEL_TRAINING_DATASET_FILE).exists()

    with open(MODEL_TRAINING_DATASET_FILE, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=row.keys())
        # write header only once
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, default='predict-author',
                        choices=['build-dataset', 'predict-author', 'train', 'evaluate-model', 'model-parameters'])
    args = parser.parse_args()

    by_author = extract_author_data()
    by_author_tokens = extract_author_data_tokens(by_author)

    if args.mode == 'build-dataset':

        if os.path.exists(MODEL_TRAINING_DATASET_FILE):
            os.remove(MODEL_TRAINING_DATASET_FILE)

        for author in MODEL_TRAINING_SONNETS.iterdir():
            if author.is_dir():
                true_author = author.name
                print("Calculating scores for", true_author)
                for file in author.iterdir():
                    target_text_tokens = tokenize_target(file)
                    target_text = Path(file).read_text(encoding="utf-8")

                    chi_results = chi.chi(authors, by_author_tokens, target_text_tokens)
                    delta_results = delta.delta(authors, by_author_tokens, target_text_tokens)
                    ngram_results = ngram.ngram(authors, by_author, target_text)

                    add_to_csv(chi_results, delta_results, ngram_results, true_author)

    if args.mode == 'predict-author':
        model = joblib.load("sonnet_model.pkl")
        target_text_tokens = tokenize_target("Target.txt")
        target_text = Path("Target.txt").read_text(encoding="utf-8")

        chi_results = chi.chi(authors, by_author_tokens, target_text_tokens)
        delta_results = delta.delta(authors, by_author_tokens, target_text_tokens)
        ngram_results = ngram.ngram(authors, by_author, target_text)

        print("Chi results:", chi_results)
        authors_list = list(chi_results.keys())
        chi_values = list(chi_results.values())
        plt.figure(figsize=(10, 6))
        plt.barh(authors_list, chi_values)
        plt.xticks(rotation=45)
        plt.xlabel("Chi-squared Value")
        plt.title("Chi-squared Comparison by Author")
        plt.tight_layout()

        print("Delta results:", delta_results)
        authors_list = list(delta_results.keys())
        delta_values = list(delta_results.values())
        plt.figure(figsize=(10, 6))
        plt.barh(authors_list, delta_values)
        plt.xticks(rotation=45)
        plt.xlabel("Delta Score Value")
        plt.title("Delta Score Comparison by Author")
        plt.tight_layout()

        print("N-gram results:", ngram_results)

        row = {}
        for author in authors:
            row[f"chi2_{author}"] = chi_results[author]

        for author in authors:
            row[f"delta_{author}"] = delta_results[author]

        for author in authors:
            row[f"ngram_3_{author}"] = ngram_results[author]["3gram"]
            row[f"ngram_4_{author}"] = ngram_results[author]["4gram"]
            row[f"ngram_5_{author}"] = ngram_results[author]["5gram"]
            row[f"ngram_6_{author}"] = ngram_results[author]["6gram"]

        X_new = pandas.DataFrame([row])
        prediction = model.predict(X_new)[0]

        print("Predicted author:", prediction)
        plt.show()

    if args.mode == 'train':
        train.train()

    if args.mode == 'evaluate-model':
        model = joblib.load("sonnet_model.pkl")
        correct = 0
        count = 0
        for author in MODEL_EVALUATION_SONNETS.iterdir():
            if author.is_dir():
                true_author = author.name
                print("Prediciting evaluation sonnets for", true_author)
                for file in author.iterdir():
                    target_text_tokens = tokenize_target(file)
                    target_text = Path(file).read_text(encoding="utf-8")

                    chi_results = chi.chi(authors, by_author_tokens, target_text_tokens)
                    delta_results = delta.delta(authors, by_author_tokens, target_text_tokens)
                    ngram_results = ngram.ngram(authors, by_author, target_text)

                    row = {}
                    for author in authors:
                        row[f"chi2_{author}"] = chi_results[author]

                    for author in authors:
                        row[f"delta_{author}"] = delta_results[author]

                    for author in authors:
                        row[f"ngram_3_{author}"] = ngram_results[author]["3gram"]
                        row[f"ngram_4_{author}"] = ngram_results[author]["4gram"]
                        row[f"ngram_5_{author}"] = ngram_results[author]["5gram"]
                        row[f"ngram_6_{author}"] = ngram_results[author]["6gram"]

                    X_new = pandas.DataFrame([row])
                    prediction = model.predict(X_new)[0]

                    count += 1
                    if prediction == true_author:
                        correct += 1
        acc = correct / count
        print("Accuracy:", acc)

    if args.mode == 'model-parameters':
        # Load trained model
        model = joblib.load("sonnet_model.pkl")

        # Load dataset to get feature names
        df = pandas.read_csv("model_dataset.csv")
        X = df.drop(columns=["true_author"])

        # Get feature importances
        importances = model.feature_importances_
        feature_names = X.columns

        # Create a sorted DataFrame
        importance_df = pandas.DataFrame({
            "feature": feature_names,
            "importance": importances
        }).sort_values(by="importance", ascending=False)

        print(importance_df)