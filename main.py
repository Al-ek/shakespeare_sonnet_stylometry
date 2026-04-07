from pathlib import Path
import re
import nltk
import delta
import chi
import csv
import argparse
import joblib
import pandas
import os
import train

BY_AUTHOR_SONNET_CORPUS = Path("by_author_sonnet_corpus")
MODEL_TRAINING_SONNETS = Path("model_training_sonnets")
MODEL_TRAINING_DATASET_FILE = "model_dataset.csv"

roman_numerals = re.compile(r'^\s*[IVXLCDM]+\s*$', re.IGNORECASE | re.MULTILINE)

# The amount of characters to be included in the authors stylometry corpus
# enter -1 for all
character_count = -1

authors = ("Gem", "Gemini", "Shakespeare", 
           "ChatGPT", "Claude", "Copilot", "Preplexity")

def tokenize_target(target_file):
    with open(target_file, 'r') as file:
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
            text = file.read_text()
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

def add_to_csv(chi_results, delta_results, true_author):

    row = {}

    # enforce fixed author order
    for author in authors:
        row[f"chi2_{author}"] = chi_results[author]
        
    for author in authors:
        row[f"delta_{author}"] = delta_results[author]

    row["true_author"] = true_author

    file_exists = Path(MODEL_TRAINING_DATASET_FILE).exists()

    with open(MODEL_TRAINING_DATASET_FILE, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=row.keys())
        # write header only once
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print("Sample appended to dataset.")
   

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, default='predict-author', choices=['build-model-dataset','predict-author', 'train'])
    args = parser.parse_args()

    by_author = extract_author_data()
    by_author_tokens = extract_author_data_tokens(by_author)

    if args.mode == 'build-model-dataset':

        if os.path.exists(MODEL_TRAINING_DATASET_FILE):
            os.remove(MODEL_TRAINING_DATASET_FILE)
        for author in MODEL_TRAINING_SONNETS.iterdir():
                if author.is_dir():
                    true_author = author.name
                    print("Calculating scores for", true_author)
                    for file in author.iterdir():
                        target_text_tokens = tokenize_target(file)

                        chi_results =  chi.chi(authors, by_author_tokens, target_text_tokens)
                        delta_results = delta.delta(authors, by_author_tokens, target_text_tokens)

                        add_to_csv(chi_results, delta_results, true_author)

    if args.mode == 'predict-author':
        model = joblib.load("sonnet_model.pkl")
        target_text_tokens = tokenize_target("Target.txt")
        chi_results =  chi.chi(authors, by_author_tokens, target_text_tokens)
        delta_results = delta.delta(authors, by_author_tokens, target_text_tokens)
        print("Chi results:", chi_results)
        print("Delta results:", delta_results)

        row = {}
        for author in authors:
            row[f"chi2_{author}"] = chi_results[author]

        for author in authors:
            row[f"delta_{author}"] = delta_results[author]

        X_new = pandas.DataFrame([row])
        prediction = model.predict(X_new)[0]

        print("Predicted author:", prediction)

    if args.mode == 'train':
        train.train()


    
