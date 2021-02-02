import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve, roc_auc_score
import numpy as np


def find_most_wins(path, n):
    data = pd.read_csv(path)
    print(data.shape)
    data['tourney_date'] = pd.to_datetime(data['tourney_date'], format='%Y%m%d')
    data = data.set_index('tourney_date')
    ten_best = data['winner_name'].value_counts()[:n].index.tolist()
    print(ten_best)
    data = data[data['winner_name'].isin(ten_best)]
    print(data.shape)
    data = data.sort_index()
    print(data['winner_name'])


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
                            '1stWon', 'w_2ndWon': '2ndWon', 'w_SvGms': 'svGms', 'w_bpSaved': 'bpSaved'
                            , 'w_bpFaced': 'bpFaced'}, inplace=True)

    losers.rename(columns={'loser_name': 'name', 'loser_hand': 'hand', 'loser_ht': 'ht', 'loser_age': 'age', 'l_ace':
                            'ace', 'l_df': 'df', 'l_svpt': 'svpt', 'l_1stIn': '1stIn', 'l_1stWon':
                            '1stWon', 'l_2ndWon': '2ndWon', 'l_SvGms': 'svGms', 'l_bpSaved': 'bpSaved'
                            , 'l_bpFaced': 'bpFaced'}, inplace=True)
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
    # print(clean.head)
    # print(pd.isnull(clean).any())

    return clean


def log_regression(clean):
    print(clean.head)
    plt.scatter(clean['serving_bp_won'], clean['won'])
    # plt.show()
    X = clean[['serving_bp_won']].to_numpy()
    # X = clean[['serving_bp_won']].to_numpy()
    print(X)
    y = np.array(clean['won'])
    # y = y.reshape((1, 2806))
    # print(X.shape)
    print(y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.8)
    print(X_test)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    # print(X_test)
    y_predicted = model.predict(X_test)
    print(model.predict_proba(X_test))
    print(model.score(X_test, y_test))
    print(classification_report(y_test, y_predicted))
    logit_roc_auc = roc_auc_score(y_test, model.predict(X_test))
    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    plt.figure()
    plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('Log_ROC')
    plt.show()


if __name__ == '__main__':
    data_2019 = data_clean('tennis_atp_1985>/atp_matches_2019.csv')
    log_regression(data_2019)
