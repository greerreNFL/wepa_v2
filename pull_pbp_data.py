## packages ##
import pandas as pd
import numpy
import datetime


## folder structure ##
data_folder = 'YOUR FOLDER' ## no final backslash here ##


## set starting season ##
starting_season = 1999
current_season = starting_season


## set max season ##
if datetime.date.today().month > 8:
    ending_season = datetime.date.today().year
else:
    ending_season = datetime.date.today().year - 1


## pull data ##
all_dfs = []
while current_season <= ending_season:
    print('Downloading {0} pbp data...'.format(current_season))
    reg_pbp_df = pd.read_csv(
        'https://raw.githubusercontent.com/guga31bb/nflfastR-data/master/data/play_by_play_{0}.csv.gz'.format(current_season),
        low_memory=False,
        compression='gzip'
    )
    all_dfs.append(reg_pbp_df)
    reg_pbp_df.to_csv('{0}/pbp_{1}.csv'.format(pbp_folder_path, current_season))
    current_season += 1


## save all seasons to a single file ##
all_seasons = pd.concat(all_dfs)
all_seasons.to_csv('{0}/pbp.csv'.format(data_folder))
