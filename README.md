# obs-ai

### Installation

```
pip install -r requirements.txt
python -m spacy download en
```

# Data Cleaning

I used spacy to lemmatize all the data. 
Data:
- lowercase
- alphanumeric
- no functional words.
- tf-idf is used. I used different parameters for TF-IDF vectorizer.

I noticed some duplication in training data. I removed those duplications.

# Experiments

Vanilla experiment (exp 1) uses two different classifiers (SVM and SGD [linear SVM]). I also tried RandomForest but I filtered out that. I tried to optimize hyperparameters but I didn't go well with my current laptop. It takes for a while to do gridsearch. The best F1-weighted accuracy on Dev set was 0.83 with SGD. Because of the skewness of the data I used class_weight params for each classifier but I couldn't search well the parameter space I think; it didn't give good regularization. According to confusion matrix, either I am regularize too much (classifier labels everything with **misc** or too less; classifier labels most of the stuff with majority of classes (**customer**, or **order**). Better hyperparameter optimization is necessary.

Another experiment (exp2) uses the same classifiers but this time I tried downsampling, too. I used different fractions of the instances labeled as **order**.

Third experiment was using PCA to reduce the dimensions and run the similar experiments. This experiment showed, especially for Random Forest classifier, few number of dimensions (7 for instance) is enough to get .81 weighted F1 score. This is good for scalability.

Unfortunately, I couldn't make much experiments on different ngram sizes which may be very helpful. I would try to use bigger ngrams but my machine does not allow me to search the parameter space.

# Other remedies to try:

1. Train a language model and generate data.
2. Regularize better
    - more experiment with hyperparams.
        - any possibility to give weights to scoring metric? It's possible to give weights to samples, so we can do it. The problem is I'm using grid_search function of sklearn. GridSearch makes the K-Folds inside. It does not allow me to give sample_weight immediately. I think changing sample weights might increase weighted f1, but for this project, I am going to skip this.
        3. Ensemble: According confusion matrix of each model; the models seem complementary. I might combine those models by VotingClassifier.  
        4. There are other methods to search hyperspace more efficiently. For instance, random search (http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a) or Distributed Asynchronous Hyper-parameter Optimization (hyperopt)

# Output

Lastly, I embed all the data (train + dev) and predict test data for two classifiers with their best settings.
