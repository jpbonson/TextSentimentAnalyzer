'''
Created on 12.2015
@author: Jessica Pauli de C. Bonson (jpbonson)
'''

import numpy as np
import os
import glob
import random
import re
import sys
import time
import pprint
from collections import defaultdict, Counter
from nltk import stem
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection.univariate_selection import SelectKBest, chi2
from sklearn.feature_selection import f_classif
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from nltk import corpus

CONFIG = {
    'dataset': 'reviews_polarity', # 'reviews_polarity', 'sentences_polarity'
    'preprocessing': {
        'preserve_some_special_symbols': True,
        'use_stopwords': 'custom_stopwords', # None, custom_stopwords, all_stopwords
        'use_intensifiers_and_negators': True,
        'use_bigrams': True,
        'use_stemmer': 'snowball', # None, 'lancaster', 'porter', 'snowball'
        'stemmer': {
            'lancaster': stem.lancaster.LancasterStemmer(), # too aggressive
            'porter': stem.porter.PorterStemmer(),
            'snowball': stem.snowball.EnglishStemmer(), # lighter
        },
        'use_tfidf': True, # if False, use words presence
        'tfidf': {
            'max_df': 0.99,
            'min_df': 0.01,
        },
        'use_feature_selection': 'ANOVA', # None, 'chi2', 'ANOVA', 'tfidf', 'random_forest'
        'top_features_to_select': 1000,
    },
    'model': 'linear_svm', # 'linear_svm', 'naive_bayes', 'knn', 'neural_networks', 'random_forest', 
                            # 'decision_tree', 'ada_boost', 'logistic_regression', 'nonlinear_svm'
    'n_folds': 5,
    'runs': 10,
}

def get_documents():
    dataset = CONFIG['dataset']

    main_path = "datasets/"

    if dataset == "sentences_polarity":
        document_neg = []
        with open(main_path+"movie_datasets/sentences_polarity/rt-polarity.neg") as f:
            for line in f:
                document_neg.append(line)
        document_pos = []
        with open(main_path+"movie_datasets/sentences_polarity/rt-polarity.pos") as f:
            for line in f:
                document_pos.append(line)
        return document_neg, document_pos

    if dataset == "reviews_polarity":
        path = main_path+"movie_datasets/review_polarity/"
        document_neg = []
        files = glob.glob(path+"neg/*")
        for p in files:
            with open(p) as f:
                temp = ""
                for line in f:
                    temp += line+"\n"
            document_neg.append(temp)
        document_pos = []
        files = glob.glob(path+"pos/*")
        for p in files:
            with open(p) as f:
                temp = ""
                for line in f:
                    temp += line+"\n"
            document_pos.append(temp)
        return document_neg, document_pos

    raise ValueError("Undefined method for the dataset '"+str(dataset)+"'")

all_stopwords = set(corpus.stopwords.words('english'))
# the lists below were created by hand
custom_stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'yo', 
    'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
    'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them',
    'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this',
    'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
    'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 
    'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
    'into', 'through', 'during', 'before', 'after',
    'to', 'from', 'in', 'out', 'on', 'off', 'over', 
    'further', 'then', 'once', 'here', 'there', 'when', 'where',
    'both', 'each', 'some', 'own', 'also',
    'than', 's', 't', 'can', 'will', 'now']
intensifiers = ['more', 'most', 'too', 'very', 'much', 'few', 'certainly', 'obviously', 'really', 'simply', 'literally',
    'completely', 'totally', 'undoubtedly', 'absolutely', 'so', 'well', 'extremely', 'slightly', 'strongly', 'fairly', 
    'quite', 'pretty', 'rather', 'little', 'all', 'ever', 'best', 'terrible', 'bad', 'poor', 'worst', 'worse', 'great',
    'perfect', 'horrible', 'dumb', 'sweet', 'beautiful', 'messy', 'ridiculous', 'boring', 'bore', 'stupid', 'hilarious',
    'strong', 'better', 'excellent', 'waste', 'least', 'brilliant', 'solid', 'many']
negators = ['no', 'not', 'don', 'nor', 'won', 'didn', 'wasn', 'weren', 'wouldn']

def apply_basic_data_cleaning(data):
    if CONFIG['preprocessing']['use_stopwords'] is None:
        stopwords = []
    elif CONFIG['preprocessing']['use_stopwords'] == 'custom_stopwords':
        stopwords = custom_stopwords
    elif CONFIG['preprocessing']['use_stopwords'] == 'all_stopwords':
        stopwords = all_stopwords
    else:
        raise ValueError("No stopwords list for "+str(CONFIG['preprocessing']['use_stopwords']))

    cleaned_data = []
    for document in data:
        cleaned_doc = document
        # increasing the length of ? and ! so they will not be removed due to being too short or by being a special symbol
        # i kept the number, /, ! and ? because I think they are able to express sentiments
        # eg.: "it was 10/10!", "amazing!!!", "what were they thinking?!", "bullshit!"
        if CONFIG['preprocessing']['preserve_some_special_symbols']:
            cleaned_doc = re.sub(r'\?', ' question_mark ', cleaned_doc)
            cleaned_doc = re.sub(r'!', ' exclamation_mark ', cleaned_doc)
            cleaned_doc = re.sub(r'[/]+', '_out_of_', cleaned_doc)
            cleaned_doc = re.sub(r':\)', ' happy_face ', cleaned_doc)
            cleaned_doc = re.sub(r':D', ' happy_face ', cleaned_doc)
            cleaned_doc = re.sub(r'=D', ' happy_face ', cleaned_doc)
            cleaned_doc = re.sub(r'=\)', ' happy_face ', cleaned_doc)
            cleaned_doc = re.sub(r':\(', ' unhappy_face ', cleaned_doc)
            cleaned_doc = re.sub(r'=\(', ' unhappy_face ', cleaned_doc)
            cleaned_doc = re.sub(r':/', ' unhappy_face ', cleaned_doc)
            cleaned_doc = re.sub(r'=/', ' unhappy_face ', cleaned_doc)
            cleaned_doc = re.sub(r'-_-', ' unhappy_face ', cleaned_doc)
        # remove tags, such as html and formatting tags
        cleaned_doc = re.sub(r'<[^<|^>]*>', '', cleaned_doc)
        # remove puntuaction and weird symbols
        cleaned_doc = re.sub(r'[^A-Za-z0-9_]+', ' ', cleaned_doc)
        # lower case, remove stopword, remove 1-length words
        cleaned_doc = ' '.join([word.lower() for word in cleaned_doc.split() if word not in stopwords and len(word) > 1])
        # add extra words when a word was amplified or negated
        if CONFIG['preprocessing']['use_intensifiers_and_negators']:
            temp = cleaned_doc.split()
            extras = []
            for index, word in enumerate(temp):
                if index < len(temp)-1:
                    if word in intensifiers:
                        extras.append("more_"+temp[index+1])
                    if word in negators:
                        extras.append("not_"+temp[index+1])
            cleaned_doc = ' '.join(temp+extras)
        cleaned_data.append(cleaned_doc)
    return cleaned_data

def apply_stemmer(data):
    """
    Example:
        a = "eggs are greater than bacon but I like and love this world and its ducks and wolves"
        stemmer = stem.lancaster.LancasterStemmer()
        ' '.join([stemmer.stem(word) for word in a.split()])
        'eg ar gre than bacon but i lik and lov thi world and it duck and wolv'
        stemmer = stem.porter.PorterStemmer()
        ' '.join([stemmer.stem(word) for word in a.split()])
        u'egg are greater than bacon but I like and love thi world and it duck and wolv'
        stemmer = stem.snowball.EnglishStemmer()
        ' '.join([stemmer.stem(word) for word in a.split()])
        u'egg are greater than bacon but i like and love this world and it duck and wolv'
    """
    stemmer = CONFIG['preprocessing']['stemmer'][CONFIG['preprocessing']['use_stemmer']]
    stemmed_data = []
    for document in data:
        stemmed_data.append(' '.join([stemmer.stem(word) for word in document.split()]))
    return stemmed_data

def create_X_and_y(document_neg, document_pos):
    X = np.array(document_neg + document_pos)
    y = np.array(['neg'] * len(document_neg) + ['pos'] * len(document_pos))
    combined = zip(X, y)
    random.shuffle(combined)
    X[:], y[:] = zip(*combined)
    return X, y

def apply_tfidf(X_train, X_test):
    max_df = CONFIG['preprocessing']['tfidf']['max_df']
    min_df = CONFIG['preprocessing']['tfidf']['min_df']
    if CONFIG['preprocessing']['use_bigrams']:
        ngram_range = (1,2)
    else:
        ngram_range = (1,1)
    if CONFIG['preprocessing']['use_feature_selection'] == 'tfidf':
        max_features = CONFIG['preprocessing']['top_features_to_select']
    else:
        max_features = 10000
    vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_df=max_df, min_df=min_df, max_features=max_features)
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
    features = vectorizer.get_feature_names()
    return X_train, X_test, features

def frequency_to_presence(matrix):
    matrix_new = []
    for row_index, row in enumerate(matrix.toarray()):
        temp = []
        for col_index, value in enumerate(row): 
            if value > 0:
                temp.append(True)
            else:
                temp.append(False)
        matrix_new.append(temp)
    return np.array(matrix_new)

def apply_presence(X_train, X_test):
    X_train, X_test, features = apply_tfidf(X_train, X_test)
    X_train_new = frequency_to_presence(X_train)
    X_test_new = frequency_to_presence(X_test)
    return X_train_new, X_test_new, features

def apply_feature_selection(X_train, y_train, X_test, features):
    if CONFIG['preprocessing']['use_feature_selection'] == 'random_forest':
        clf = RandomForestClassifier()
        clf = clf.fit(X_train.toarray(), y_train)
        features_scores = [(feature, score) for (score,feature) in sorted(zip(clf.feature_importances_, features), reverse=True)]
        selected_features = features_scores[:CONFIG['preprocessing']['top_features_to_select']]
        selected_indeces = np.searchsorted(features, [f[0] for f in selected_features])
        X_train = X_train[:,selected_indeces]
        X_test = X_test[:,selected_indeces]
        return X_train, y_train, X_test, selected_features
    if CONFIG['preprocessing']['use_feature_selection'] == 'chi2':
        algorithm = chi2
    elif CONFIG['preprocessing']['use_feature_selection'] == 'ANOVA':
        algorithm = f_classif
    else:
        raise ValueError("No implementation for "+str(CONFIG['preprocessing']['use_feature_selection']))
    feature_selector = SelectKBest(algorithm, k=CONFIG['preprocessing']['top_features_to_select'])
    feature_selector.fit(X_train, y_train)
    X_train = feature_selector.fit_transform(X_train, y_train)
    X_test = feature_selector.transform(X_test)
    features = [(feature, score) for (score,feature) in sorted(zip(feature_selector.scores_, features), reverse=True)]
    selected_features = features[:CONFIG['preprocessing']['top_features_to_select']]
    return X_train, y_train, X_test, selected_features

def train_and_test_classifier(X_train, y_train, X_test):
    if CONFIG['model'] == 'linear_svm':
        classifier = LinearSVC(tol=0.00001, C=1.0)
    elif CONFIG['model'] == 'naive_bayes':
        classifier = GaussianNB()
    elif CONFIG['model'] == 'knn':
        classifier = KNeighborsClassifier(n_neighbors=20, weights='uniform', leaf_size=60, p=2)
    elif CONFIG['model'] == 'neural_networks':
        classifier = Perceptron(n_iter=40)
    elif CONFIG['model'] == 'random_forest':
        classifier = RandomForestClassifier(n_estimators=160, criterion='entropy', min_samples_split=2, 
            min_samples_leaf=1, max_features='auto')
    elif CONFIG['model'] == 'decision_tree':
        classifier = DecisionTreeClassifier(criterion='entropy', min_samples_split=2, min_samples_leaf=1, max_features=None)
    elif CONFIG['model'] == 'ada_boost':
        classifier = AdaBoostClassifier(n_estimators=500, learning_rate=1.0)
    elif CONFIG['model'] == 'logistic_regression':
        classifier = LogisticRegression(C=2.0)
    elif CONFIG['model'] == 'nonlinear_svm':
        classifier = SVC(C=1.0, kernel='linear', probability=False, tol=0.00001)
    else:
        raise ValueError("No implementation for "+str(CONFIG['model']))
    classifier.fit(X_train.toarray(), y_train) # train
    predicted = classifier.predict(X_test.toarray()) # predict
    return predicted

def calculate_metrics(results_per_fold, y_test, predicted):
    results_per_fold['accuracy'].append(metrics.accuracy_score(y_test, predicted))
    results_per_fold['matthews_corrcoef'].append(metrics.matthews_corrcoef(y_test, predicted))
    mapping = {'pos': 0, 'neg': 1}
    results_per_fold['auc'].append(metrics.average_precision_score([mapping[x] for x in y_test], [mapping[x] for x in predicted]))
    results_per_fold['confusion_matrix'].append(metrics.confusion_matrix(y_test, predicted))
    results_per_fold['overall']['recall'].append(metrics.recall_score(y_test, predicted, pos_label = None))
    results_per_fold['overall']['precision'].append(metrics.precision_score(y_test, predicted, pos_label = None))
    results_per_fold['overall']['f1_score'].append(metrics.f1_score(y_test, predicted, pos_label = None))
    results_per_fold['neg']['recall'].append(metrics.recall_score(y_test, predicted, pos_label = 'neg'))
    results_per_fold['neg']['precision'].append(metrics.precision_score(y_test, predicted, pos_label = 'neg'))
    results_per_fold['neg']['f1_score'].append(metrics.f1_score(y_test, predicted, pos_label = 'neg'))
    results_per_fold['pos']['recall'].append(metrics.recall_score(y_test, predicted, pos_label = 'pos'))
    results_per_fold['pos']['precision'].append(metrics.precision_score(y_test, predicted, pos_label = 'pos'))
    results_per_fold['pos']['f1_score'].append(metrics.f1_score(y_test, predicted, pos_label = 'pos'))
    return results_per_fold

def process_results(results_per_fold):
    all_accuracies = results_per_fold['accuracy']
    for key, value in results_per_fold.iteritems():
        if key == 'confusion_matrix':
            pos00 = [v[0][0] for v in value]
            pos01 = [v[0][1] for v in value]
            pos10 = [v[1][0] for v in value]
            pos11 = [v[1][1] for v in value]
            results_per_fold[key] = {
                'mean': [[round_value(np.mean(pos00)), round_value(np.mean(pos01))],
                    [round_value(np.mean(pos10)), round_value(np.mean(pos11))]], 
                'std': [[round_value(np.std(pos00)), round_value(np.std(pos01))],
                    [round_value(np.std(pos10)), round_value(np.std(pos11))]]}
        elif isinstance(value, list):
            results_per_fold[key] = {'mean': round_value(np.mean(value)), 'std': round_value(np.std(value))}
        else:
            for key2, value2 in value.iteritems():
                results_per_fold[key][key2] = {'mean': round_value(np.mean(value2)), 'std': round_value(np.std(value2))}
            results_per_fold[key] = dict(results_per_fold[key])
    results_per_fold['all_accuracies'] = all_accuracies
    return results_per_fold

def words_frequency_per_class(document_neg, document_pos, features_per_fold):
    merged_features = {}
    for features in features_per_fold:
        for feature, score in features:
            if feature not in merged_features:
                merged_features[feature] = score
            else:
                merged_features[feature] += score
    sorted_features = sorted(merged_features.items(), key=lambda x: x[1], reverse=True)
    sorted_features = [(feature, score/float(len(features_per_fold))) for feature, score in sorted_features]
    words_neg = [[word for word in document.split()] for document in document_neg]
    words_pos = [[word for word in document.split()] for document in document_pos]
    words_neg = sum(words_neg, [])
    words_pos = sum(words_pos, [])
    words_neg_cont = Counter(words_neg)
    words_pos_cont = Counter(words_pos)
    info = {}
    for index, (feature, score) in enumerate(sorted_features):
        info[index] = {}
        info[index][feature] = {}
        info[index][feature]['avg_score'] = round_value(score, 3)
        info[index][feature]['neg'] = round_value(words_neg_cont[feature]/float(len(words_neg)), 6)
        info[index][feature]['pos'] = round_value(words_pos_cont[feature]/float(len(words_pos)), 6)
    return info

def round_value(value, round_decimals_to = 3):
    number = float(10**round_decimals_to)
    return int(value * number) / number

if __name__ == "__main__":
    start_time = time.time()

    final_output = ""
    m = "############# Config\n"+pprint.pformat(CONFIG)+"\n"
    print m
    final_output += m

    document_neg, document_pos = get_documents()

    # preprocessing
    document_neg = apply_basic_data_cleaning(document_neg)
    document_pos = apply_basic_data_cleaning(document_pos)
    if CONFIG['preprocessing']['use_stemmer'] is not None:
        document_neg = apply_stemmer(document_neg)
        document_pos = apply_stemmer(document_pos)

    results_per_fold = defaultdict(list)
    results_per_fold['overall'] = defaultdict(list)
    results_per_fold['neg'] = defaultdict(list)
    results_per_fold['pos'] = defaultdict(list)
    selected_features_per_fold = []
    for run in range(CONFIG['runs']):
        print "######### Executing run "+str(run+1)+"...\n"
        # prepare datasets for k-fold crossvalidation
        X, y = create_X_and_y(document_neg, document_pos)
        skf = StratifiedKFold(y, n_folds = CONFIG['n_folds'])

        # apply tfidf, feature extraction, and machine learning model for each fold
        for index, (train_index, test_index) in enumerate(skf):
            print "### Executing fold "+str(index+1)+" (run "+str(run+1)+")...\n"
            X_train, y_train = X[train_index], y[train_index]
            X_test, y_test = X[test_index], y[test_index]

            # create features
            if CONFIG['preprocessing']['use_tfidf']:
                X_train, X_test, features = apply_tfidf(X_train, X_test)
                print "training set size (tfidf): "+str(X_train.shape)+", "+str(y_train.shape)
                print "test set size (tfidf): "+str(X_test.shape)+", "+str(y_test.shape)
                print
            else:
                X_train, X_test, features = apply_presence(X_train, X_test)
                print "training set size (presence): "+str(X_train.shape)+", "+str(y_train.shape)
                print "test set size (presence): "+str(X_test.shape)+", "+str(y_test.shape)
                print

            # feature selection
            if CONFIG['preprocessing']['use_feature_selection'] is not None and CONFIG['preprocessing']['use_feature_selection'] != 'tfidf':
                X_train, y_train, X_test, selected_features = apply_feature_selection(X_train, y_train, X_test, features)
                print "training set size (feature selection): "+str(X_train.shape)+", "+str(y_train.shape)
                print "test set size (feature selection): "+str(X_test.shape)+", "+str(y_test.shape)
                print
            else:
                selected_features = [(feature, 1.0) for feature in features]

            # classifier
            predicted = train_and_test_classifier(X_train, y_train, X_test)

            # evaluation
            results_per_fold = calculate_metrics(results_per_fold, y_test, predicted)
            selected_features_per_fold.append(selected_features)
            print "Metrics:"
            print "- accuracy: "+str(results_per_fold['accuracy'][-1])
            print "- matthews correlation coefficient: "+str(results_per_fold['matthews_corrcoef'][-1])
            print "- precision-recall curve AUC: "+str(results_per_fold['auc'][-1])
            print "- confusion matrix:\n"+str(results_per_fold['confusion_matrix'][-1])
            print "- report:\n"+metrics.classification_report(y_test, predicted)
            print

    final_output += "\ntotal features (without feature selection): "+str(len(features))+"\n"
    if selected_features:
        final_output += "total features (with feature selection): "+str(len(selected_features))+"\n"

    # process and save results
    results_per_fold = process_results(results_per_fold)
    m = "\n############# Results\n"+pprint.pformat(dict(results_per_fold))+"\n"
    print m
    final_output += m

    features_info = words_frequency_per_class(document_neg, document_pos, selected_features_per_fold)
    m = "\nselected features:\n"+str(pprint.pformat(dict(features_info)))
    final_output += m

    if len(sys.argv) > 1:
        filename = "results_"+sys.argv[1]+".log"
    else:
        filename = "results.log"

    if not os.path.exists("output"):
                os.makedirs("output")
    with open("output/"+filename, 'w') as f:
        f.write(final_output)

    elapsed_time = round_value((time.time() - start_time)/60.0)
    print "\nFinished, elapsed time: "+str(elapsed_time)+" mins"