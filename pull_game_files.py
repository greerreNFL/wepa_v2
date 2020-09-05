## packages ##
import pandas as pd
import numpy

## folder structure ##
data_folder = 'YOUR FOLDER' ## no final backslash here ##

## pull from Lee Sharpe ##
game_df = pd.read_csv('https://raw.githubusercontent.com/leesharpe/nfldata/master/data/games.csv')


## standardize franchise names ##
pbp_team_standard_dict = {

    'ARI' : 'ARI',
    'ATL' : 'ATL',
    'BAL' : 'BAL',
    'BUF' : 'BUF',
    'CAR' : 'CAR',
    'CHI' : 'CHI',
    'CIN' : 'CIN',
    'CLE' : 'CLE',
    'DAL' : 'DAL',
    'DEN' : 'DEN',
    'DET' : 'DET',
    'GB'  : 'GB',
    'HOU' : 'HOU',
    'IND' : 'IND',
    'JAC' : 'JAX',
    'JAX' : 'JAX',
    'KC'  : 'KC',
    'LA'  : 'LAR',
    'LAC' : 'LAC',
    'LV'  : 'OAK',
    'MIA' : 'MIA',
    'MIN' : 'MIN',
    'NE'  : 'NE',
    'NO'  : 'NO',
    'NYG' : 'NYG',
    'NYJ' : 'NYJ',
    'OAK' : 'OAK',
    'PHI' : 'PHI',
    'PIT' : 'PIT',
    'SD'  : 'LAC',
    'SEA' : 'SEA',
    'SF'  : 'SF',
    'STL' : 'LAR',
    'TB'  : 'TB',
    'TEN' : 'TEN',
    'WAS' : 'WAS',

}

game_df['home_team'] = game_df['home_team'].replace(pbp_team_standard_dict)
game_df['away_team'] = game_df['away_team'].replace(pbp_team_standard_dict)

## replace game_id using standardized franchise names ##
game_df['game_id'] = (
    game_df['season'].astype('str') +
    '_' +
    game_df['week'].astype('str').str.zfill(2) +
    '_' +
    game_df['away_team'] +
    '_' +
    game_df['home_team']
)

## export ##
game_df.to_csv('{0}/games.csv'.format(data_folder))
