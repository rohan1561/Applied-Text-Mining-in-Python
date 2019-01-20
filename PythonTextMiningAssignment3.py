import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC


def add_feature(X, feature_to_add):
    return hstack([X, csr_matrix(feature_to_add).T], 'csr')


def add_multiple_features(X, *features_to_add):
    for feature in features_to_add:
        X = add_feature(X, feature)
    return X


# Read data into a DataFrame
spam_data = pd.read_csv(os.getcwd() + '/course4_downloads/spam.csv')
spam_data['target'] = np.where(spam_data['target']=='spam', 1, 0)
print('Sample of the spam data\n')
print(spam_data.head(10))
print ('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')

# Split into training and test sets. Outputs pd.Series type
X_train, X_test, y_train, y_test = train_test_split(spam_data['text'], spam_data['target'], random_state=0)


def answer_one():
    return spam_data['target'].mean()*100


print('Answer one: {}'.format(answer_one()))


def answer_two():
    # Vectorize the text
    vect = CountVectorizer().fit(X_train)
    longest_token = sorted(vect.get_feature_names(), key=len)[-1]
    return longest_token


print('Answer two: {}'.format(answer_two()))

def answer_three():
    vect = CountVectorizer().fit(X_train)
    X_train_vectorized = vect.transform(X_train)

    # Model fitting
    model = MultinomialNB(alpha=0.1)
    model.fit(X_train_vectorized, y_train)
    predictions = model.predict(vect.transform(X_test))
    AUC_score = roc_auc_score(y_test, predictions)

    return AUC_score


print('Answer three: {}'.format(answer_three()))


def answer_four():
    vect = TfidfVectorizer().fit(X_train)
    X_train_vectorized = vect.transform(X_train)
    features = np.array(vect.get_feature_names())
    sorted_indices = X_train_vectorized.max(0).toarray()[0].argsort()

    # Features with high and low TFIDF
    high_tfidf_features = features[sorted_indices[-20:]]
    low_tfidf_features = features[sorted_indices[:20]]

    # Values of TFIDF, high and low
    high_tfidf_values = X_train_vectorized.max(0).toarray()[0][sorted_indices[-20:]]
    low_tfidf_values = X_train_vectorized.max(0).toarray()[0][sorted_indices[:20]]

    high_value_pairs = list(zip(high_tfidf_features, high_tfidf_values))
    low_value_pairs = list(zip(low_tfidf_features, low_tfidf_values))
    low_value_series = []
    for value in sorted(set(low_tfidf_values)):
        value_pairs = sorted(list(filter(lambda x: x[1] == value, low_value_pairs)))
        low_value_series += value_pairs

    high_value_series = []
    for value in sorted(set(high_tfidf_values), reverse=True):
        value_pairs = sorted(list(filter(lambda x: x[1] == value, high_value_pairs)))
        high_value_series += value_pairs

    low_value_series = pd.Series(list(map(lambda x: x[1], low_value_series)), index=list(map(lambda x: x[0], low_value_series)))
    high_value_series = pd.Series(list(map(lambda x: x[1], high_value_series)), index=list(map(lambda x: x[0], high_value_series)))

    return (low_value_series, high_value_series)


print('Answer four: {}'.format(answer_four()))


def answer_five():
    vect = TfidfVectorizer(min_df=3).fit(X_train)
    X_train_vectorized = vect.transform(X_train)
    model = MultinomialNB(alpha=0.1)
    model.fit(X_train_vectorized, y_train)
    predictions = model.predict(vect.transform(X_test))
    AUC_score = roc_auc_score(y_test, predictions)

    return AUC_score


print('Answer five: {}'.format(answer_five()))


def answer_six():
    len_spam = spam_data[spam_data['target']==1]['text'].str.len().mean()
    len_non_spam = spam_data[spam_data['target']==0]['text'].str.len().mean()

    return (len_non_spam, len_spam)


print('Answer six: {}'.format(answer_six()))


def answer_seven():
    vect = TfidfVectorizer(min_df=5).fit(X_train)
    X_train_vectorized = vect.transform(X_train)
    X_train_vectorized = add_feature(X_train_vectorized, X_train.str.len())
    X_test_vectorized = add_feature(vect.transform(X_test), X_test.str.len())

    # Fit a SVM
    model = SVC(C=10000)
    model.fit(X_train_vectorized, y_train)
    predictions = model.predict(X_test_vectorized)
    AUC_score = roc_auc_score(y_test, predictions)

    return AUC_score


print('Answer_seven: {}'.format(answer_seven()))


def answer_eight():
    spam_digits = list(map(len, (spam_data[spam_data['target'] == 1]['text'].str.findall(r'\d'))))
    non_spam_digits = list(map(len, spam_data[spam_data['target'] == 0]['text'].str.findall(r'\d')))

    return(float(sum(non_spam_digits))/len(non_spam_digits), float(sum(spam_digits))/len(spam_digits))


print('Answer eight: {}'.format(answer_eight()))


def answer_nine():
    vect = TfidfVectorizer(min_df=5, ngram_range=(1, 3)).fit(X_train)
    X_train_vectorized = vect.transform(X_train)
    X_test_vectorized = vect.transform(X_test)

    # Add features to X_train_vectorized
    X_train_length = X_train.str.len()
    X_train_digits = list(map(len, X_train.str.findall(r'\d')))
    X_train_vectorized = add_multiple_features(X_train_vectorized, X_train_length, X_train_digits)

    # Add features to X_test_vectorized
    X_test_length = X_test.str.len()
    X_test_digits = list(map(len, X_test.str.findall(r'\d')))
    X_test_vectorized = add_multiple_features(X_test_vectorized, X_test_length, X_test_digits)

    # Fit model and get AUC score
    model = LogisticRegression(C=100)
    model.fit(X_train_vectorized, y_train)
    predictions = model.predict(X_test_vectorized)
    AUC_score = roc_auc_score(y_test, predictions)

    return AUC_score


print('Answer nine: {}'.format(answer_nine()))


def answer_ten():
    nonW_spam = list(map(len, spam_data[spam_data['target'] == 1]['text'].str.findall(r'\W')))
    nonW_nonSpam = list(map(len, spam_data[spam_data['target'] == 0]['text'].str.findall(r'\W')))

    return(float(sum(nonW_nonSpam))/len(nonW_nonSpam), float(sum(nonW_spam))/len(nonW_spam))


print('Answer ten: {}'.format(answer_ten()))


def answer_eleven():
    vect = CountVectorizer(min_df=5, ngram_range=(2,5), analyzer='char_wb').fit(X_train)
    X_train_vectorized = vect.transform(X_train)

    # Add features to X_train
    X_train_digits = list(map(len, X_train.str.findall(r'(\d)')))
    X_train_nonW = list(map(len, X_train.str.findall(r'(\W)')))
    X_train_vectorized = add_multiple_features(X_train_vectorized, X_train.str.len(), X_train_digits, X_train_nonW)

    # Add features to X_test
    X_test_vectorized = vect.transform(X_test)
    X_test_digits = list(map(len, X_test.str.findall(r'(\d)')))
    X_test_nonW = list(map(len, X_test.str.findall(r'(\W)')))
    X_test_vectorized = add_multiple_features(X_test_vectorized, X_test.str.len(), X_test_digits, X_test_nonW)

    # Fit the Logistic Regression model and get Area Under Curve score
    model = LogisticRegression(C=100).fit(X_train_vectorized, y_train)
    predictions = model.predict(X_test_vectorized)
    AUC_score = roc_auc_score(y_test, predictions)

    features = np.array(vect.get_feature_names())
    features = np.concatenate((features, np.array(['length_of_doc', 'digit_count', 'non_word_char_count'])))

    # Smallest and largest coefficients
    sorted_coef_indices = model.coef_[0].argsort()
    smallest_coef = list(features[sorted_coef_indices[:10]])
    largest_coef = list(features[sorted_coef_indices[:-11:-1]])

    return AUC_score, smallest_coef, largest_coef


print('Answer eleven: {}'.format(answer_eleven()))
