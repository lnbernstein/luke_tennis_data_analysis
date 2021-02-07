import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Alright, we want to compare the number of break points won and break points saved for the players
# with the most wins each year against the players with the most losses each year

plt.style.use('seaborn-whitegrid')


def filtering():
    data = pd.read_csv('tennis_atp_1985>/atp_matches_2019.csv')
    winners = data.filter(
        ['w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon', 'w_svGms', 'w_bpSaved',
         'w_bpFaced'])
    winners['won'] = 1
    print(winners.columns)
    print(winners.dtypes)
    winners = winners.dropna()
    print(winners.isnull().any())
    Xtrain, Xtest, ytrain, ytest = train_test_split(winners.iloc[:, : 7], winners.iloc[:, -1], random_state=42)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(Xtrain, ytrain)
    res_pred = clf.predict(Xtest)
    score = accuracy_score(ytest, res_pred)
    print(score)


def find_winners(path):
    data = pd.read_csv(path)
    print(data.columns)

    winners = data.filter(
        ['tourney_date', 'winner_name', 'w_bpSaved', 'w_bpFaced', 'l_bpSaved',
         'l_bpFaced'])

    winners['tourney_date'] = pd.to_datetime(winners['tourney_date'], format='%Y%m%d')
    winners = winners.set_index('tourney_date')

    ten_best = np.array(winners['winner_name'].value_counts()[:10].index)

    # Make a histogram of the ten most winners
    ten_best_nums = np.array(winners['winner_name'].value_counts()[:10])
    index = np.arange(len(ten_best))
    plt.bar(index, ten_best_nums)
    plt.xlabel('Name', fontsize=5)
    plt.ylabel('Um', fontsize=5)
    plt.xticks(index, ten_best, fontsize=5, rotation=30)
    plt.show()

    # print(ten_best)

    winners = winners[winners['winner_name'].isin(ten_best)]
    winners = winners.sort_index()
    # print(winners.head)
    # print(winners.dtypes)

    winners['serving_bp_won'] = winners['w_bpSaved'] / winners['w_bpFaced']
    winners['serving_bp_lost'] = 1 - winners['serving_bp_won']
    winners['returning_bp_won'] = winners['l_bpSaved'] / winners['l_bpFaced']
    winners['returning_bp_lost'] = 1 - winners['returning_bp_won']
    winners = winners.drop(columns=['winner_name', 'l_bpSaved', 'l_bpFaced', 'w_bpSaved', 'w_bpFaced'])
    winners = winners.dropna()
    print(winners.describe())

    return winners


def find_losers(path):
    data = pd.read_csv(path)
    # print(data.columns)
    # print(data.head())

    losers = data.filter(
        ['tourney_date', 'loser_name', 'w_bpSaved', 'w_bpFaced', 'l_bpSaved',
         'l_bpFaced'])

    losers['tourney_date'] = pd.to_datetime(losers['tourney_date'], format='%Y%m%d')
    losers = losers.set_index('tourney_date')

    ten_worst = np.array(losers['loser_name'].value_counts()[:10].index)

    losers = losers[losers['loser_name'].isin(ten_worst)]
    losers = losers.sort_index()

    losers['serving_bp_won'] = losers['l_bpSaved'] / losers['l_bpFaced']
    losers['serving_bp_lost'] = 1 - losers['serving_bp_won']
    losers['returning_bp_lost'] = losers['w_bpSaved'] / losers['w_bpFaced']
    losers['returning_bp_won'] = 1 - losers['returning_bp_lost']
    losers = losers.drop(columns=['loser_name', 'l_bpSaved', 'l_bpFaced', 'w_bpSaved', 'w_bpFaced'])

    losers = losers.dropna()
    print(losers.describe())

    return losers


if __name__ == "__main__":
    filtering()
    print('Winners')
    find_winners('tennis_atp_1985>/atp_matches_2019.csv')
    print('Losers')
    find_losers('tennis_atp_1985>/atp_matches_2019.csv')



