import pandas as pd

"""
Pre-Processing
"""

# Importing the Data
sms_data = pd.read_table('smsspamcollection/SMSSpamCollection', sep='\t', header=None, names=['label', 'sms_message'])

# Mapping labels
sms_data['label'] = sms_data.label.map({'ham': 0, 'spam': 1})

# Bag Of Words Count Vectorizer
from sklearn.feature_extraction.text import CountVectorizer
count_vector = CountVectorizer(lowercase=True, token_pattern='(?u)\\b\\w\\w+\\b', stop_words='english')

# split into training and testing sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(sms_data['sms_message'],
                                                    sms_data['label'],
                                                    random_state=1)


# Fit the training data and then return the matrix
training_data = count_vector.fit_transform(X_train)

# Transform testing data and return the matrix. Note we are not fitting the testing data into the CountVectorizer()
testing_data = count_vector.transform(X_test)

"""
Model fitting
"""

from sklearn.naive_bayes import MultinomialNB
naive_bayes = MultinomialNB()
naive_bayes.fit(training_data, y_train)

y_pred = naive_bayes.predict(testing_data)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print('Accuracy score: ', format(accuracy_score(y_pred, y_test)))
print('Precision score: ', format(precision_score(y_pred, y_test)))
print('Recall score: ', format(recall_score(y_pred, y_test)))
print('F1 score: ', format(f1_score(y_pred, y_test)))

""""
Results:
    Accuracy score:  0.9877961234745154
    Precision score:  0.9459459459459459
    Recall score:  0.9615384615384616
    F1 score:  0.9536784741144414
"""