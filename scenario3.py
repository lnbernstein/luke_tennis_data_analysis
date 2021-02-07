import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_curve, roc_auc_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, chi2
import numpy as np

"""Data is reformated to add a win/loss class and all columns are renamed to provide one column for each performance
    characteristic. There is now a separate row for winners and losers. """


def get_all_data():
    """
    :return: concated df of all csv's since 1985
    :rtype: dataFrame
    """
    path = 'tennis_atp_1985>'
    all_data = pd.DataFrame()
    for file in os.listdir('tennis_atp_1985>'):
        file_path = os.path.join(path, file)
        all_data = all_data.append(pd.read_csv(file_path))

    return all_data


def data_clean(data):
    """
    Filters all unnecessary features from data set containg matches since 1985
    :param data: data set compiled in get_all_data
    :type data: dataFrame
    :return clean:
    :rtype clean: dataFrame
    """
    # select all features of winning participants
    winners = data.filter(['winner_name', 'winner_hand', 'winner_ht', 'winner_age', 'w_ace', 'w_df',
                           'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon', 'w_SvGms', 'w_bpSaved', 'w_bpFaced'])
    winners['won'] = 1
    # select all features of losing participants
    losers = data.filter(['loser_name', 'loser_hand', 'loser_ht', 'loser_age', 'l_ace', 'l_df',
                          'l_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_SvGms', 'l_bpSaved', 'l_bpFaced'])
    losers['won'] = 0

    winners.rename(columns={'winner_name': 'name', 'winner_hand': 'hand', 'winner_ht': 'ht', 'winner_age': 'age',
                            'w_ace': 'ace', 'w_df': 'df', 'w_svpt': 'svpt', 'w_1stIn': '1stIn', 'w_1stWon':
                            '1stWon', 'w_2ndWon': '2ndWon', 'w_SvGms': 'svGms', 'w_bpSaved': 'bpSaved', 'w_bpFaced':
                                'bpFaced'}, inplace=True)

    losers.rename(columns={'loser_name': 'name', 'loser_hand': 'hand', 'loser_ht': 'ht', 'loser_age': 'age', 'l_ace':
                            'ace', 'l_df': 'df', 'l_svpt': 'svpt', 'l_1stIn': '1stIn', 'l_1stWon': '1stWon',
                           'l_2ndWon': '2ndWon', 'l_SvGms': 'svGms', 'l_bpSaved': 'bpSaved', 'l_bpFaced': 'bpFaced'},
                  inplace=True)
    clean = pd.concat([winners, losers], axis=0)
    clean['serving_bp_won'] = clean['bpSaved'] / clean['bpFaced']
    clean['serving_bp_lost'] = 1 - clean['serving_bp_won']
    clean['returning_bp_won'] = clean['bpSaved'] / clean['bpFaced']
    clean['returning_bp_lost'] = 1 - clean['returning_bp_won']
    # Null values are safely dropped and this indicates matches where there was a 0 for any of these categores
    clean.dropna(inplace=True)
    print(clean.isnull().values.any())
    # one-hot encoded dummy variable for hand of the participant
    clean = pd.get_dummies(clean, prefix='hand', columns=['hand'])
    return clean


"""Uses Select K best and Extra Trees to find best features for ml models"""


def select_features(clean):
    """
    Uses SelectKBest and ChiSquared to determine most useful features
    :param clean: filtered df from data_clean, only 2019 used as these features are most applicable for prediction
    :type clean: dataFrame
    :return features: five most useful features for predicting the outcome
    :rtype: np array
    """
    X = clean.loc[:, clean.columns != 'won']
    X = X.select_dtypes(exclude=['object'])
    y = np.array(clean['won'])
    best_features = SelectKBest(score_func=chi2, k=10)
    fit = best_features.fit(X, y)
    df_scores = pd.DataFrame(fit.scores_)
    df_columns = pd.DataFrame(X.columns)

    feature_scores = pd.concat([df_columns, df_scores], axis=1)
    feature_scores.columns = ['Specs', 'Score']

    features = (feature_scores.nlargest(10, 'Score'))
    features.drop(['Score'], axis=1, inplace=True)

    features = features[:5]
    features = np.array(features['Specs'])
    print(features)
    features = np.append(features, ['1stIn', 'svGms', 'hand_R'])

    return features


def log_regression(clean, met_features):
    """
    Performs Logistic Regression using SciKit Learn
    Produces results using Classification Report Class from SciKit Learn
    :param clean: df from data_clean
    :type clean: dataFrame
    :param met_features: array returned from select_features
    :type met_features: np array
    """
    X = clean[met_features]
    y = np.array(clean['won'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.8, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_predicted = model.predict(X_test)
    y_pred_probs = model.predict_proba(X_test)
    y_pred_probs = y_pred_probs[:, 1]
    auc = roc_auc_score(y_test, y_pred_probs)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)

    plot_roc(fpr, tpr, auc)
    plot_cm(y_test, y_predicted)
    print(classification_report(y_test, y_predicted))


def decision_tree(clean, met_features):
    """
        Performs Decision Tree Classification using SciKit Learn
        Produces results using Classification Report Class from SciKit Learn
        :param clean: df from data_clean
        :type clean: dataFrame
        :param met_features: array returned from select_features
        :type met_features: np array
        """
    X = clean[met_features]
    y = np.array(clean['won'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.8, random_state=42)
    model = DecisionTreeClassifier(max_depth=4)
    model.fit(X_train, y_train)

    y_predicted = model.predict(X_test)
    y_pred_probs = model.predict_proba(X_test)
    y_pred_probs = y_pred_probs[:, 1]
    auc = roc_auc_score(y_test, y_pred_probs)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)

    plot_roc(fpr, tpr, auc)
    plot_cm(y_test, y_predicted)
    print(classification_report(y_test, y_predicted))


def random_forest(clean, met_features):
    """
        Performs Random Forest Classification using SciKit Learn
        Produces results using Classification Report Class from SciKit Learn
        :param clean: df from data_clean
        :type clean: dataFrame
        :param met_features: array returned from select_features
        :type met_features: np array
        """

    X = clean[met_features]
    y = np.array(clean['won'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.8, random_state=42)
    model = RandomForestClassifier(n_estimators=100, criterion='entropy')
    model.fit(X_train, y_train)

    y_predicted = model.predict(X_test)
    y_pred_probs = model.predict_proba(X_test)
    y_pred_probs = y_pred_probs[:, 1]
    auc = roc_auc_score(y_test, y_pred_probs)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)

    plot_roc(fpr, tpr, auc)
    plot_cm(y_test, y_predicted)
    print(classification_report(y_test, y_predicted))


def plot_roc(fpr, tpr, auc):
    """
    Plots area under the Receiver Operating Characteristic Curve
    True Positive Rate / False Positive Rate
    calculated using roc_curve function from SciKit
    :param fpr: False Positivity Rate
    :type fpr: float
    :param tpr: True Positivity Rate
    :type tpr: float
    :param auc: Area Under the Curve
    :type auc: float
    """
    plt.plot(fpr, tpr, color='orange', label='ROC (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()


def plot_cm(y_true, y_pred):
    """
    Plot a confusion matrix for each ML model
    Calculated using SciKit and heatmap generated by Seaborn
    True Positive, False Positive
    False Negative, True Negative
    :param y_true:
    :type y_true:
    :param y_pred:
    :type y_pred:
    """
    cf = confusion_matrix(y_true, y_pred)
    print(cf)
    sns.heatmap(cf / np.sum(cf), annot=True, fmt='.2%', cmap='rocket')
    plt.show()


if __name__ == '__main__':
    all_data = data_clean(get_all_data())
    data_2019 = data_clean(pd.read_csv('tennis_atp_1985>/atp_matches_2019.csv'))
    main_features = select_features(data_2019)
    log_regression(all_data, main_features)
    decision_tree(all_data, main_features)
    random_forest(all_data, main_features)
