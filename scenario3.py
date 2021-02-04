import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import classification_report, roc_curve, roc_auc_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, chi2
import numpy as np

"""Data is reformated to add a win/loss class and all columns are renamed to provide one column for each performance
    characteristic. There is now a separate row for winners and losers. """


def data_clean(path):
    data = pd.read_csv(path)
    winners = data.filter(['tourney_date', 'winner_name', 'winner_hand', 'winner_ht', 'winner_age', 'w_ace', 'w_df',
                           'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon', 'w_SvGms', 'w_bpSaved', 'w_bpFaced'])
    winners['won'] = 1
    losers = data.filter(['tourney_date', 'loser_name', 'loser_hand', 'loser_ht', 'loser_age', 'l_ace', 'l_df',
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
    # print(data.columns)
    # print(winners.head)
    # print(losers.head)
    clean = pd.concat([winners, losers], axis=0)
    clean['tourney_date'] = pd.to_datetime(winners['tourney_date'], format='%Y%m%d')
    clean.set_index('tourney_date', inplace=True)
    clean.sort_index(inplace=True)
    # print(clean.columns)
    clean['serving_bp_won'] = clean['bpSaved'] / clean['bpFaced']
    clean['serving_bp_lost'] = 1 - clean['serving_bp_won']
    clean['returning_bp_won'] = clean['bpSaved'] / clean['bpFaced']
    clean['returning_bp_lost'] = 1 - clean['returning_bp_won']
    clean.dropna(inplace=True)
    clean = pd.get_dummies(clean, prefix='hand', columns=['hand'])
    # print(clean.head)
    # print(pd.isnull(clean).any())

    return clean


def select_features(clean):
    X = clean.loc[:, clean.columns != 'won']
    X = X.select_dtypes(exclude=['object'])
    y = np.array(clean['won'])
    best_features = SelectKBest(score_func=chi2, k=10)
    fit = best_features.fit(X,y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']
    print(featureScores.nlargest(10, 'Score'))

    model = ExtraTreesClassifier()
    model.fit(X, y)
    print(model.feature_importances_)  # use inbuilt class feature_importances of tree based classifiers
    # plot graph of feature importances for better visualization
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    feat_importances.nlargest(10).plot(kind='barh')
    plt.show()

    corrmat = clean.corr()
    top_corr_features = corrmat.index
    plt.figure(figsize=(20, 20))
    # plot heat map
    g = sns.heatmap(clean[top_corr_features].corr(), annot=True, cmap="RdYlGn")
    plt.show()


def log_regression(clean):
    X = clean[['bpFaced', '1stWon', 'ace', 'bpSaved', '2ndWon', '1stIn', 'svGms', 'hand_R']].to_numpy()
    # print(X)
    y = np.array(clean['won'])
    # print(X.shape)

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


def decision_tree(clean):
    X = clean[['bpFaced', '1stWon', 'ace', 'bpSaved', '2ndWon', '1stIn', 'svGms', 'hand_R']].to_numpy()
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


def random_forest(clean):
    X = clean[['bpFaced', '1stWon', 'ace', 'bpSaved', '2ndWon', '1stIn', 'svGms', 'hand_R']].to_numpy()
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


"""Plots area under the Receiver Operating Characteristic Curve 
    True Positive Rate / False Positive Rate"""


def plot_roc(fpr, tpr, auc):
    plt.plot(fpr, tpr, color='orange', label='ROC (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()


"""Plot a confusion matrix for each ML model
    True Positive, False Positive
    False Negative, True Negative"""


def plot_cm(y_true, y_pred):
    cf = confusion_matrix(y_true, y_pred)
    print(cf)
    sns.heatmap(cf / np.sum(cf), annot=True, fmt='.2%', cmap='rocket')
    plt.show()


if __name__ == '__main__':
    data_2019 = data_clean('tennis_atp_1985>/atp_matches_2019.csv')
    # select_features(data_2019)
    log_regression(data_2019)
    decision_tree(data_2019)
    random_forest(data_2019)
