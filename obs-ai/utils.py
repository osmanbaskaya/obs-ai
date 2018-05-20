from constants import NLP
import matplotlib.pyplot as plt
from collections import Counter
from functools import partial
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from itertools import cycle


# Frequent (most likely not useful) words.
def get_all_words(train_df):
    all_words = []
    list(map(all_words.extend, train_df['text'].apply(lambda x: x.split())))
    return Counter(all_words)


def get_label_word_counts(train_df):
    label_word_counts = {}
    for key, dfg in train_df.groupby("label"):
        words = []
        list(map(words.extend, dfg.apply(lambda r: r['text'].split(), axis=1)))
        word_counts = Counter(words)
        label_word_counts[key] = word_counts
    return label_word_counts


def most_common_for_each_labels(train_df, top_n=50):
    label_word_counts = get_label_word_counts(train_df)
    for label, wc in label_word_counts.items():
        print(f'{label}\n{wc.most_common(top_n)}\n')


def get_stopwords():
    return stopwords.words('english')


def lemmatize(doc, nlp=NLP):
    return " ".join(token.lemma_.lower() for token in nlp(doc) if token.lemma_ != "-PRON-")


def lemmatize_data(texts, nlp=NLP):
    return list(map(partial(lemmatize, nlp=nlp), texts))


def calc_inverse_weight_for_labels(df):
    labels = df['label'].value_counts()
    inverse_weight = (1 / (labels / labels.sum()))
    return (inverse_weight / inverse_weight.sum()).to_dict()


def calc_uniform_weight_for_labels(df):
    labels = df['label'].unique().tolist()
    return dict(zip(labels, cycle([1])))


def create_sample_weight():
    return {'applicant': 200, 'customer': 10, 'misc': 300, 'order': 1, 'shopper': 100}


def dim_reduction_with_pca(X, vectorizer=None):
    stop_words = get_stopwords()
    if vectorizer is None:
        vectorizer = TfidfVectorizer(stop_words=stop_words, max_df=.9)

    pca = PCA()
    pca.fit(vectorizer.fit_transform(X).todense())
    return pca


def draw_explained_variance(pca):
    plt.figure(1, figsize=(4, 3))
    plt.clf()
    plt.axes([.2, .2, .7, .7])
    plt.plot(pca.explained_variance_, linewidth=2)
    plt.axis('tight')
    plt.xlabel('n_components')
    plt.ylabel('Variance explained')


def write_preds(filename, preds):
    with open(filename, mode='wt+') as f:
        list(map(lambda p: f.write("%s\n" % p), preds))
