from copy import deepcopy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.pipeline import Pipeline
from pprint import pprint

from time import time
from sklearn.decomposition import PCA
import warnings
from utils import get_stopwords, write_preds
from data import read_data, get_train_dev_data, downsample_train_data, draw_label_dist
from constants import NLP


warnings.filterwarnings(action='always')
warnings.simplefilter('ignore')


def grid_search(classifiers, tf_idf_params, X, y, stop_words=(), scoring='f1_weighted'):

    best_clfs = []

    for clf, clf_param in classifiers:

        parameters = {}
        parameters.update(tf_idf_params)
        parameters.update(clf_param)

        steps = [('tfidf', TfidfVectorizer(stop_words=stop_words)), ('clf', clf)]

        pipeline = Pipeline(steps=steps)
        grid_search = GridSearchCV(pipeline, parameters, scoring=scoring, n_jobs=-1, verbose=1)

        print("Performing grid search...")
        print("pipeline:", [name for name, _ in pipeline.steps])
        print("parameters:")
        pprint(parameters)
        t0 = time()
        grid_search.fit(X, y)
        print("done in %0.3fs" % (time() - t0))
        print()
        print("Best score: %0.3f" % grid_search.best_score_)
        print("Best parameters set:")
        best_parameters = grid_search.best_estimator_.get_params()
        for param_name in sorted(parameters.keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))

        best_clfs.append(pipeline.set_params(**grid_search.best_estimator_.get_params()))
    return best_clfs


def test_on_dev_one(clf, X_train, y_train, X_dev, y_dev):
    clf.fit(X_train, y_train)
    preds = clf.predict(X_dev)
    score = f1_score(y_dev, preds, average='weighted')
    print(f"Score on Dev data {clf.get_params()['clf'].__class__.__name__}: {score}")


def test_on_dev(clfs, X_train, y_train, X_dev, y_dev):
    for clf in clfs:
        test_on_dev_one(clf, X_train, y_train, X_dev, y_dev)


def get_confusion_matrix(clf, X_train, y_train, X_dev, y_dev):
    labels = list(set(y_train))
    clf.fit(X_train, y_train)
    preds_train = clf.predict(X_train)
    conf_train = confusion_matrix(y_train, preds_train, labels)
    preds_dev = clf.predict(X_dev)
    conf_dev = confusion_matrix(y_dev, preds_dev, labels)
    pprint(conf_train)
    print()
    pprint(conf_dev)
    print()
    print(labels)
    return conf_train, conf_dev, labels


def vanilla_experiment(dataset, classifiers, tf_idf_params, nlp=NLP):
    train_df = read_data(dataset['train'])
    dev_df = read_data(dataset['dev'])

    stop_words = list(get_stopwords())

    X_train, y_train, X_dev, y_dev = get_train_dev_data(train_df, dev_df, nlp)
    print(len(X_train), len(X_dev), flush=True)
    best_clfs = grid_search(classifiers, tf_idf_params, X_train, y_train, stop_words=stop_words)
    test_on_dev(best_clfs, X_train, y_train, X_dev, y_dev)
    conf_train, conf_dev, labels = get_confusion_matrix(best_clfs[0], X_train, y_train, X_dev, y_dev)
    pprint(conf_train)
    pprint(conf_dev)
    return best_clfs


def run_downsample_experiment(dataset, classifiers, tf_idf_params, frac=0.6, stop_words=(), nlp=NLP):
    # Downsample the major class.

    train_df = read_data(dataset['train'])
    dev_df = read_data(dataset['dev'])

    train_df = downsample_train_data(train_df, frac=frac)
    draw_label_dist(train_df)
    X_train, y_train, X_dev, y_dev = get_train_dev_data(train_df, dev_df, nlp)
    print(len(X_train), len(X_dev), flush=True)
    best_clfs = grid_search(classifiers, tf_idf_params, X_train, y_train, stop_words=stop_words)
    test_on_dev(best_clfs, X_train, y_train, X_dev, y_dev)
    for clf in best_clfs:
        get_confusion_matrix(clf, X_train, y_train, X_dev, y_dev)

    return best_clfs


def calc_score_on_dev_after_dim_reduction1(pipe, X_train, y_train, X_dev, y_dev, n=200, stop_words=None):
    # Vectorizer
    clf = pipe.get_params()['clf']
    vectorizer = pipe.get_params()['tfidf']
    vectorizer.fit(X_train)

    # Dim reduction
    pca = PCA()
    X_train_projected = pca.fit_transform(vectorizer.transform(X_train).todense())[:, :n]

    # Classifier training
    clf.fit(X_train_projected, y_train)

    print("Training weighted f1-score: ", f1_score(y_train, clf.predict(X_train_projected), average='weighted'))

    X_dev_projected = pca.transform(vectorizer.transform(X_dev).todense())[:, :n]

    # Check F1 average accuracy.
    preds = clf.predict(X_dev_projected)
    dev_score = f1_score(y_dev, preds, average='weighted')
    print(dev_score)
    return dev_score, clf, pca, vectorizer


def grid_search_with_dim_red_one(clf, vectorizer, X_train, y_train, X_dev, y_dev, n=10):
    # Vectorizer
    vectorizer.fit(X_train)

    # Dim reduction
    pca = PCA()
    X_train_projected = pca.fit_transform(vectorizer.transform(X_train).todense())[:, :n]

    score = cross_val_score(clf, X_train_projected, y_train, cv=3, scoring='f1_weighted')

    clf.fit(X_train_projected, y_train)

    X_dev_projected = pca.transform(vectorizer.transform(X_dev).todense())[:, :n]

    # Check F1 average accuracy.
    preds = clf.predict(X_dev_projected)
    return score, f1_score(y_dev, preds, average='weighted'), n


def grid_search_dim_red(*args):
    # Dummy search.
    stop_words = get_stopwords()
    n_estimators = (30, 40, 50)
    max_depths = [5, 7, 10, 20]
    criterions = ('gini', 'entropy')
    pca_params = [7, 14, 21, 30]
    i = 1
    for max_depth in max_depths:
        for criterion in criterions:
            for n_estimator in n_estimators:
                for pca_param in pca_params:
                    print(f'{i}. {max_depth}, {criterion}, {n_estimator}, {pca_param}.')
                    vectorizer = TfidfVectorizer(stop_words=stop_words, max_df=.7)
                    clf = RandomForestClassifier(n_estimators=n_estimator, criterion=criterion, max_depth=max_depth)
                    print(grid_search_with_dim_red_one(clf, vectorizer, *args, n=pca_param))
                    i += 1


def pred_test_with_best_model(clf, X_train, y_train, X_dev, y_dev, X_test, write_to_file=True):
    X = deepcopy(X_train)
    y = deepcopy(y_train)
    X.extend(X_dev)
    y.extend(y_dev)
    print(len(X), len(y))

    clf.fit(X, y)
    preds = clf.predict(X_test)

    if write_to_file:
        write_preds('../data/predictions_for_test.txt', preds)

    return preds