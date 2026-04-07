import nltk


def delta(authors, by_author_tokens, target_text_tokens):
    delta_results = {}

    # Convert papers to lowercase to count all tokens of the same word together
    # regardless of case
    for author in authors:
        by_author_tokens[author] = (
            [tok.lower() for tok in by_author_tokens[author]])

    # Combine every paper except our test case into a single corpus
    whole_corpus = []
    for author in authors:
        whole_corpus += by_author_tokens[author]

    # Get a frequency distribution
    whole_corpus_freq_dist = list(nltk.FreqDist(whole_corpus).most_common(30))
    whole_corpus_freq_dist[ :10 ]

    # The main data structure
    features = [word for word,freq in whole_corpus_freq_dist]
    feature_freqs = {}

    for author in authors:
        # A dictionary for each candidate's features
        feature_freqs[author] = {}

        # A helper value containing the number of tokens in the author's subcorpus
        overall = len(by_author_tokens[author])

        # Calculate each feature's presence in the subcorpus
        for feature in features:
            presence = by_author_tokens[author].count(feature)
            feature_freqs[author][feature] = presence / overall


    import math

    # The data structure into which we will be storing the "corpus standard" statistics
    corpus_features = {}

    # For each feature...
    for feature in features:
        # Create a sub-dictionary that will contain the feature's mean
        # and standard deviation
        corpus_features[feature] = {}

        # Calculate the mean of the frequencies expressed in the subcorpora
        feature_average = 0
        for author in authors:
            feature_average += feature_freqs[author][feature]
        feature_average /= len(authors)
        corpus_features[feature]["Mean"] = feature_average

        # Calculate the standard deviation using the basic formula for a sample
        feature_stdev = 0
        for author in authors:
            diff = feature_freqs[author][feature] - corpus_features[feature]["Mean"]
            feature_stdev += diff*diff
        feature_stdev /= (len(authors) - 1)
        feature_stdev = math.sqrt(feature_stdev)
        corpus_features[feature]["StdDev"] = feature_stdev


    feature_zscores = {}
    for author in authors:
        feature_zscores[author] = {}
        for feature in features:

            # Z-score definition = (value - mean) / stddev
            # We use intermediate variables to make the code easier to read
            feature_val = feature_freqs[author][feature]
            feature_mean = corpus_features[feature]["Mean"]
            feature_stdev = corpus_features[feature]["StdDev"]
            feature_zscores[author][feature] = ((feature_val-feature_mean) /
                                                feature_stdev)

    testcase_tokens = target_text_tokens

    # Calculate the test case's features
    overall = len(testcase_tokens)
    testcase_freqs = {}
    for feature in features:
        presence = testcase_tokens.count(feature)
        testcase_freqs[feature] = presence / overall

    # Calculate the test case's feature z-scores
    testcase_zscores = {}
    for feature in features:
        feature_val = testcase_freqs[feature]
        feature_mean = corpus_features[feature]["Mean"]
        feature_stdev = corpus_features[feature]["StdDev"]
        testcase_zscores[feature] = (feature_val - feature_mean) / feature_stdev

    for author in authors:
        delta = 0
        for feature in features:
            delta += math.fabs((testcase_zscores[feature] -
                                feature_zscores[author][feature]))
        delta /= len(features)
        delta_results[author] = delta
    return delta_results