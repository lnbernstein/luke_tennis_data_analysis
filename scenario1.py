import pandas as pd
import os


"""cleans an individual csv file into a data frame containing only tournament level of A or G, tourney_date as index
winner and loser name and rank and round of match for only matches between the top 20 players in  world at that time"""


def clean_csv(path):
    data_2020 = pd.read_csv(path)
    # print(data_2020.columns)
    # print(data_2020.head)

    scenario1 = data_2020.filter(
        ['tourney_level', 'tourney_date', 'winner_name', 'loser_name', 'round', 'winner_rank', 'loser_rank'])

    scenario1['tourney_date'] = pd.to_datetime(scenario1['tourney_date'], format='%Y%m%d')
    scenario1 = scenario1.set_index('tourney_date')

    scenario1 = scenario1[scenario1.winner_rank < 20.0]
    scenario1 = scenario1[scenario1.loser_rank < 20.0]

    mask1 = scenario1.tourney_level == 'G'
    mask2 = scenario1.tourney_level == 'A'
    mask = mask1 | mask2

    scenario1 = scenario1[mask]

    return scenario1


"""takes all csv data from 1985 to 2020 and appends to one Data Frame"""


def get_all_data():
    path = 'tennis_atp_1985>'
    all_data = pd.DataFrame()
    for file in os.listdir('tennis_atp_1985>'):
        file_path = os.path.join(path, file)
        all_data = all_data.append(clean_csv(file_path))

    all_data = all_data.sort_index()
    return all_data


def most_common_winners(all_data):
    n = 10
    print(all_data['winner_name'].value_counts()[:n])
    print(all_data['winner_name'].value_counts()[:n].index.tolist())


def most_common_losers(all_data):
    n = 10
    print(all_data['loser_name'].value_counts(ascending=True)[:n])
    print(all_data['loser_name'].value_counts()[:n].index.tolist())


if __name__ == "__main__":
    all_data = get_all_data()
    most_common_winners(all_data)
    most_common_losers(all_data)
