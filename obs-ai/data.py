import pandas as pd
import warnings
from utils import lemmatize_data
from constants import DATASET, NLP

warnings.filterwarnings(action='always')
warnings.simplefilter('ignore')


def get_data_path():
    return DATASET


def read_data(filename):
    return pd.read_csv(filename, header=0, sep=',')


# Train/dev label dist.
def draw_label_dist(df):
    df['label'].value_counts().plot('bar')


def prepare_data(df, no_label=False, nlp=NLP):
    X = lemmatize_data(df['text'].tolist(), nlp)
    if no_label:
        return X

    y = df['label'].tolist()
    return X, y


def downsample_train_data(train_df, frac=0.6, label='order'):
    major_class_df = train_df[train_df['label'] == label]
    major_class_df = major_class_df.sample(frac=frac)
    index_to_drop = train_df[train_df['label'] == label].index
    train_df.drop(index_to_drop, inplace=True)
    frames = [major_class_df, train_df]
    return pd.concat(frames).reset_index(drop=True)


def get_train_dev_data(train_df, dev_df, nlp=NLP):
    X_train, y_train = prepare_data(train_df, nlp)
    X_dev, y_dev = prepare_data(dev_df, nlp)
    return X_train, y_train, X_dev, y_dev


def get_test_data(df, nlp=NLP):
    return prepare_data(df, nlp=nlp, no_label=True)

