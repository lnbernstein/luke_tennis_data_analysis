import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Alright, we want to compare the number of break points won and break points saved for the players
# with the most wins each year against the players with the most losses each year

plt.style.use('seaborn-whitegrid')


def find_winners(path):
    data = pd.read_csv(path)
    print(data.columns)
    # print(data.head())

    winners = data.filter(
        ['tourney_date', 'winner_name', 'w_bpSaved', 'w_bpFaced', 'l_bpSaved',
         'l_bpFaced'])

    winners['tourney_date'] = pd.to_datetime(winners['tourney_date'], format='%Y%m%d')
    winners = winners.set_index('tourney_date')

    # print(winners['winner_name'].value_counts()[:10])

    ten_best = np.array(winners['winner_name'].value_counts()[:10].index)

    # Make a histogram of the ten most winners
    # ten_best_nums = np.array(winners['winner_name'].value_counts()[:10])
    # index = np.arange(len(ten_best))
    # plt.bar(index, ten_best_nums)
    # plt.xlabel('Name', fontsize=5)
    # plt.ylabel('Um', fontsize=5)
    # plt.xticks(index, ten_best, fontsize=5, rotation=30)
    # plt.show()

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
    # print(winners.head)
    # print(winners.columns)
    # print(pd.isnull(winners).any())
    winners = winners.dropna()
    # print(pd.isnull(winners).any())
    print(winners.describe())


def find_losers(path):
    data = pd.read_csv(path)
    # print(data.columns)
    # print(data.head())

    losers = data.filter(
        ['tourney_date', 'loser_name', 'w_bpSaved', 'w_bpFaced', 'l_bpSaved',
         'l_bpFaced'])

    losers['tourney_date'] = pd.to_datetime(losers['tourney_date'], format='%Y%m%d')
    losers = losers.set_index('tourney_date')

    # print(losers['loser_name'].value_counts()[:10])

    ten_worst = np.array(losers['loser_name'].value_counts()[:10].index)
    # print(ten_worst)

    losers = losers[losers['loser_name'].isin(ten_worst)]
    losers = losers.sort_index()
    # print(losers.head)
    # print(losers.dtypes)

    losers['serving_bp_won'] = losers['l_bpSaved'] / losers['l_bpFaced']
    losers['serving_bp_lost'] = 1 - losers['serving_bp_won']
    losers['returning_bp_lost'] = losers['w_bpSaved'] / losers['w_bpFaced']
    losers['returning_bp_won'] = 1 - losers['returning_bp_lost']
    losers = losers.drop(columns=['loser_name', 'l_bpSaved', 'l_bpFaced', 'w_bpSaved', 'w_bpFaced'])
    # print(losers.head)
    # print(losers.columns)
    # print(pd.isnull(losers).any())
    losers = losers.dropna()
    # print(pd.isnull(losers).any())
    print(losers.describe())


def main():
    print('Winners')
    find_winners('tennis_atp_1985>/atp_matches_2019.csv')
    print('Losers')
    find_losers('tennis_atp_1985>/atp_matches_2019.csv')
    # plt.scatter(clean['w_bpSaved'], clean['w_bpFaced'], c='r')
    # plt.show()


if __name__ == "__main__":
    main()



