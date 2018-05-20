import spacy

# train dataset is cleaned before. There are some duplication entries.
DATASET = {'train': '../data/data_train.clean.csv',
           'test': '../data/data_eval.csv',
           'dev': '../data/data_dev.csv'}

NLP = spacy.load('en', disable=['tagger', 'parser', 'ner'])

