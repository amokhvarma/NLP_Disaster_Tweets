# NLP_Disaster_Tweets
Finding whether a given tweet corresponds to an actual disaster or not.
This is a kaggle contest and can be found at https://www.kaggle.com/c/nlp-getting-started \
The jupyter notebook contains a basic Naive Bayes approach of solving the problem.
To know more about Naive Bayes, please visit https://en.wikipedia.org/wiki/Naive_Bayes_classifier .

### Principle
The code works by choosing the most frequent words from the training data (excluding stop words ) and training a naive bayes classifier on them.
This is just a basic baseline model and can be used as a minimum threshold for further models.
