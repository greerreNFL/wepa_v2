## packages used ##
import pandas as pd
import numpy
import statsmodels.api as sm
from scipy.optimize import minimize
import time


## file paths ##
pbp_filepath = 'YOUR PBP FILEPATH'
game_filepath = 'YOUR GAME FILEPATH' ## needed to calc accurate margins ##
output_folder = 'YOUR OUTPUT FOLDER/' ## include final back slash! ##

## create objective function ##
## takes an array of best guesses ##
best_guesses_all_time = [

    1.15,    ## qb_rush ##
    1.30,    ## neutral_second_down_rush ##
    -0.69,    ## incompletion_depth_s ##
    -0.60,    ## non_sack_fumble ##
    -0.15,    ## int ##
    -1.00,    ## goalline ##
    -0.55,    ## scaled_win_prob ##
    1.17,    ## d_qb_rush ##
    1.49,    ## d_neutral_second_down_rush ##
    -1.00,    ## d_incompletion_depth_s ##
    -0.90,    ## d_sack_fumble ##
    -0.75,    ## d_int ##
    -0.85,    ## d_fg ##
    -0.45,    ## d_third_down_pos ##
    -0.20     ## defense_adj ##

]

## define weight names for reference ##
weight_names_list = [

    'qb_rush',
    'neutral_second_down_rush',
    'incompletion_depth_s',
    'non_sack_fumble',
    'int',
    'goalline',
    'scaled_win_prob',
    'd_qb_rush',
    'd_neutral_second_down_rush',
    'd_incompletion_depth_s',
    'd_sack_fumble',
    'd_int',
    'd_fg',
    'd_third_down_pos',
    'defense_adj'

]


## define objective function for optimization##
def wepa_objective(x, train_df):
    ## define weights ##
    ## play style ##
    train_df['qb_rush_weight'] = numpy.where((train_df['qb_scramble'] == 1) & (train_df['fumble_lost'] != 1), 1 + x[0], 1)
    train_df['neutral_second_down_rush_weight'] = numpy.where(
        (train_df['down'] == 2) &
        (train_df['play_call'] == 'Run') &
        (train_df['yardline_100'] > 20) &
        (train_df['yardline_100'] < 85) &
        ((train_df['wp'] < .90) | (train_df['wp'] > .10)) &
        (train_df['qb_scramble'] != 1) &
        (train_df['fumble_lost'] != 1) &
        (train_df['epa'] < 0),
        1 + x[1],
        1
    )
    train_df['incompletion_depth_s_weight'] = 1 + numpy.where(
        (train_df['incomplete_pass'] == 1) & (train_df['interception'] != 1),
        numpy.where(numpy.isnan(x[2] * (2 * (1/(1 + numpy.exp(-0.1 * train_df['air_yards'] + .75)) - 0.5))),0,(x[2] * (2 * (1/(1 + numpy.exp(-0.1 * train_df['air_yards'] + .75)) - 0.5)))),
        0
    )
    ## events ##
    train_df['non_sack_fumble_weight'] = numpy.where((train_df['sack'] != 1) & (train_df['fumble_lost'] == 1), 1 + x[3], 1)
    train_df['int_weight'] = numpy.where(train_df['interception'] == 1, 1 + x[4], 1)
    ## contextual ##
    train_df['goalline_weight'] = numpy.where((train_df['yardline_100'] < 3) & (train_df['down'] < 4), 1 + x[5], 1)
    train_df['scaled_win_prob_weight'] = 1 + (-x[6] * numpy.where(train_df['wp'] <= .5, 1/(1+numpy.exp(-10*(2*train_df['wp']-0.5)))-0.5,1/(1+numpy.exp(-10*(2*(1-train_df['wp'])-0.5)))-0.5))
    ## define defensive weights ##
    ## play style ##
    train_df['d_qb_rush_weight'] = numpy.where((train_df['qb_scramble'] == 1) & (train_df['fumble_lost'] != 1), 1 + x[7], 1)
    train_df['d_neutral_second_down_rush_weight'] = numpy.where(
        (train_df['down'] == 2) &
        (train_df['play_call'] == 'Run') &
        (train_df['yardline_100'] > 20) &
        (train_df['yardline_100'] < 85) &
        ((train_df['wp'] < .90) | (train_df['wp'] > .10)) &
        (train_df['qb_scramble'] != 1) &
        (train_df['fumble_lost'] != 1) &
        (train_df['epa'] < 0),
        1 + x[8],
        1
    )
    train_df['d_incompletion_depth_s_weight'] = 1 + numpy.where(
        (train_df['incomplete_pass'] == 1) & (train_df['interception'] != 1),
        numpy.where(numpy.isnan(x[9] * (2 * (1/(1 + numpy.exp(-0.1 * train_df['air_yards'] + .75)) - 0.5))),0,(x[9] * (2 * (1/(1 + numpy.exp(-0.1 * train_df['air_yards'] + .75)) - 0.5)))),
        0
    )
    ## events ##
    train_df['d_sack_fumble_weight'] = numpy.where((train_df['sack'] == 1) & (train_df['fumble_lost'] == 1), 1 + x[10], 1)
    train_df['d_int_weight'] = numpy.where(train_df['interception'] == 1, 1 + x[11], 1)
    train_df['d_fg_weight'] = numpy.where(train_df['play_type'] == 'field_goal', 1 + x[12], 1)
    ## contextual ##
    train_df['d_third_down_pos_weight'] = numpy.where(
        (train_df['down'] == 3) &
        (train_df['epa'] > 0),
        1 + x[13],
        1
    )
    ## add weights to list to build out headers and loops ##
    weight_names = [
        'qb_rush',
        'neutral_second_down_rush',
        'incompletion_depth_s',
        'non_sack_fumble',
        'int',
        'goalline',
        'scaled_win_prob'
    ]
    d_weight_names = [
        'd_qb_rush',
        'd_neutral_second_down_rush',
        'd_incompletion_depth_s',
        'd_sack_fumble',
        'd_int',
        'd_fg',
        'd_third_down_pos'
    ]
    ## create a second list for referencing the specifc weights ##
    weight_values = []
    for weight in weight_names:
        weight_values.append('{0}_weight'.format(weight))
    ## defense ##
    d_weight_values = []
    for weight in d_weight_names:
        d_weight_values.append('{0}_weight'.format(weight))
    ## create structures for aggregation ##
    aggregation_dict = {
        'margin' : 'max', ## game level margin added to each play, so take max to get 1 ##
        'wepa' : 'sum',
        'd_wepa' : 'sum',
        'epa' : 'sum',
    }
    headers = [
        'game_id',
        'posteam',
        'defteam',
        'season',
        'game_number',
        'margin',
        'wepa',
        'd_wepa',
        'epa'
    ]
    ## dictionary to rename second half of the season metrics ##
    rename_to_last_dict = {
        'margin' : 'margin_L8',
        'wepa_net' : 'wepa_net_L8',
    }
    ## disctionary to join oppoenets epa to net out ##
    rename_opponent_dict = {
        'margin' : 'margin_against',
        'wepa' : 'wepa_against',
        'd_wepa' : 'd_wepa_against',
        'epa' : 'epa_against',
    }
    ## create wepa ##
    train_df['wepa'] = train_df['epa']
    for weight in weight_values:
        train_df['wepa'] = train_df['wepa'] * train_df[weight]
    train_df['d_wepa'] = train_df['epa'] * (1 + x[14])
    for weight in d_weight_values:
        train_df['d_wepa'] = train_df['d_wepa'] * train_df[weight]
    ## bound wepa to prevent extreme values from introducing volatility ##
    train_df['wepa'] = numpy.where(train_df['wepa'] > 10, 10, train_df['wepa'])
    train_df['wepa'] = numpy.where(train_df['wepa'] < -10, -10, train_df['wepa'])
    ## defense ##
    train_df['d_wepa'] = numpy.where(train_df['d_wepa'] > 10, 10, train_df['d_wepa'])
    train_df['d_wepa'] = numpy.where(train_df['d_wepa'] < -10, -10, train_df['d_wepa'])
    ## aggregate from pbp to game level ##
    game_level_df = train_df.groupby(['posteam','defteam','season','game_id','game_number']).agg(aggregation_dict).reset_index()
    game_level_df = game_level_df.sort_values(by=['posteam','game_id'])
    game_level_df = game_level_df[headers]
    ## add net epa ##
    ## create an opponent data frame ##
    game_level_opponent_df = game_level_df.copy()
    game_level_opponent_df['posteam'] = game_level_opponent_df['defteam']
    game_level_opponent_df = game_level_opponent_df.drop(columns=['defteam','season','game_number'])
    game_level_opponent_df = game_level_opponent_df.rename(columns=rename_opponent_dict)
    ## merge to main game file ##
    game_level_df = pd.merge(
        game_level_df,
        game_level_opponent_df,
        on=['posteam', 'game_id'],
        how='left'
    )
    ## calculate net wepa and apply defensive adjustment ##
    game_level_df['wepa_net'] = game_level_df['wepa'] - game_level_df['d_wepa_against']
    ## create comparison and regressions ##
    ## split into first and second halves of the season ##
    first_half_df = game_level_df.copy()
    first_half_df = first_half_df[first_half_df['game_number'] <= 8]
    second_half_df = game_level_df.copy()
    second_half_df = second_half_df[(second_half_df['game_number'] > 8) & (second_half_df['game_number'] < 17)]
    first_half_df = first_half_df.drop(columns=['game_id', 'game_number', 'wepa', 'wepa_against', 'd_wepa_against', 'epa_against'])
    second_half_df = second_half_df.drop(columns=['game_id', 'game_number', 'wepa', 'wepa_against', 'd_wepa_against', 'epa_against'])
    ## change margins from max to sum (since its only aggregated at the game level) ##
    new_agg_dict = {'margin' : 'sum', 'wepa_net' : 'sum',}
    first_half_df = first_half_df.groupby(['posteam', 'season']).agg(new_agg_dict).reset_index()
    second_half_df = second_half_df.groupby(['posteam', 'season']).agg(new_agg_dict).reset_index()
    ## rename the second half dict ##
    second_half_df = second_half_df.rename(columns=rename_to_last_dict)
    ## join into a single df ##
    final_df = pd.merge(
        first_half_df,
        second_half_df[['posteam', 'season', 'margin_L8']],
        on=['posteam', 'season'],
        how='left'
    )
    ## join test & train info ##
    final_df = pd.merge(
        final_df,
        team_season_df[['posteam','season','data_set']],
        on=['posteam', 'season'],
        how='left'
    )
    ## filter ##
    filtered_df = final_df[final_df['data_set'] == 'Train']
    ## optimize ##
    ## set objective to be rsq ##
    model = sm.OLS(filtered_df['margin_L8'],filtered_df['wepa_net'])
    results = model.fit()
    return 1 - results.rsquared



## define function for grading against the forward window ##
def wepa_test(x, test_df):
    ## define weights ##
    ## play style ##
    test_df['qb_rush_weight'] = numpy.where((test_df['qb_scramble'] == 1) & (test_df['fumble_lost'] != 1), 1 + x[0], 1)
    test_df['neutral_second_down_rush_weight'] = numpy.where(
        (test_df['down'] == 2) &
        (test_df['play_call'] == 'Run') &
        (test_df['yardline_100'] > 20) &
        (test_df['yardline_100'] < 85) &
        ((test_df['wp'] < .90) | (test_df['wp'] > .10)) &
        (test_df['qb_scramble'] != 1) &
        (test_df['fumble_lost'] != 1) &
        (test_df['epa'] < 0),
        1 + x[1],
        1
    )
    test_df['incompletion_depth_s_weight'] = 1 + numpy.where(
        (test_df['incomplete_pass'] == 1) & (test_df['interception'] != 1),
        numpy.where(numpy.isnan(x[2] * (2 * (1/(1 + numpy.exp(-0.1 * test_df['air_yards'] + .75)) - 0.5))),0,(x[2] * (2 * (1/(1 + numpy.exp(-0.1 * test_df['air_yards'] + .75)) - 0.5)))),
        0
    )
    ## events ##
    test_df['non_sack_fumble_weight'] = numpy.where((test_df['sack'] != 1) & (test_df['fumble_lost'] == 1), 1 + x[3], 1)
    test_df['int_weight'] = numpy.where(test_df['interception'] == 1, 1 + x[4], 1)
    ## contextual ##
    test_df['goalline_weight'] = numpy.where((test_df['yardline_100'] < 3) & (test_df['down'] < 4), 1 + x[5], 1)
    test_df['scaled_win_prob_weight'] = 1 + (-x[6] * numpy.where(test_df['wp'] <= .5, 1/(1+numpy.exp(-10*(2*test_df['wp']-0.5)))-0.5,1/(1+numpy.exp(-10*(2*(1-test_df['wp'])-0.5)))-0.5))
    ## define defensive weights ##
    ## play style ##
    test_df['d_qb_rush_weight'] = numpy.where((test_df['qb_scramble'] == 1) & (test_df['fumble_lost'] != 1), 1 + x[7], 1)
    test_df['d_neutral_second_down_rush_weight'] = numpy.where(
        (test_df['down'] == 2) &
        (test_df['play_call'] == 'Run') &
        (test_df['yardline_100'] > 20) &
        (test_df['yardline_100'] < 85) &
        ((test_df['wp'] < .90) | (test_df['wp'] > .10)) &
        (test_df['qb_scramble'] != 1) &
        (test_df['fumble_lost'] != 1) &
        (test_df['epa'] < 0),
        1 + x[8],
        1
    )
    test_df['d_incompletion_depth_s_weight'] = 1 + numpy.where(
        (test_df['incomplete_pass'] == 1) & (test_df['interception'] != 1),
        numpy.where(numpy.isnan(x[9] * (2 * (1/(1 + numpy.exp(-0.1 * test_df['air_yards'] + .75)) - 0.5))),0,(x[9] * (2 * (1/(1 + numpy.exp(-0.1 * test_df['air_yards'] + .75)) - 0.5)))),
        0
    )
    ## events ##
    test_df['d_sack_fumble_weight'] = numpy.where((test_df['sack'] == 1) & (test_df['fumble_lost'] == 1), 1 + x[10], 1)
    test_df['d_int_weight'] = numpy.where(test_df['interception'] == 1, 1 + x[11], 1)
    test_df['d_fg_weight'] = numpy.where(test_df['play_type'] == 'field_goal', 1 + x[12], 1)
    ## contextual ##
    test_df['d_third_down_pos_weight'] = numpy.where(
        (test_df['down'] == 3) &
        (test_df['epa'] > 0),
        1 + x[13],
        1
    )
    ## add weights to list to build out headers and loops ##
    weight_names = [
        'qb_rush',
        'neutral_second_down_rush',
        'incompletion_depth_s',
        'non_sack_fumble',
        'int',
        'goalline',
        'scaled_win_prob'
    ]
    d_weight_names = [
        'd_qb_rush',
        'd_neutral_second_down_rush',
        'd_incompletion_depth_s',
        'd_sack_fumble',
        'd_int',
        'd_fg',
        'd_third_down_pos'
    ]
    ## create a second list for referencing the specifc weights ##
    weight_values = []
    for weight in weight_names:
        weight_values.append('{0}_weight'.format(weight))
    ## defense ##
    d_weight_values = []
    for weight in d_weight_names:
        d_weight_values.append('{0}_weight'.format(weight))
    ## create structures for aggregation ##
    aggregation_dict = {
        'margin' : 'max', ## game level margin added to each play, so take max to get 1 ##
        'wepa' : 'sum',
        'd_wepa' : 'sum',
        'epa' : 'sum',
    }
    headers = [
        'game_id',
        'posteam',
        'defteam',
        'season',
        'game_number',
        'margin',
        'wepa',
        'd_wepa',
        'epa'
    ]
    ## dictionary to rename second half of the season metrics ##
    rename_to_last_dict = {
        'margin' : 'margin_L8',
        'wepa_net' : 'wepa_net_L8',
        'epa_net' : 'epa_net_L8',
    }
    ## disctionary to join oppoenets epa to net out ##
    rename_opponent_dict = {
        'margin' : 'margin_against',
        'wepa' : 'wepa_against',
        'd_wepa' : 'd_wepa_against',
        'epa' : 'epa_against',
    }
    ## create wepa ##
    test_df['wepa'] = test_df['epa']
    for weight in weight_values:
        test_df['wepa'] = test_df['wepa'] * test_df[weight]
    test_df['d_wepa'] = test_df['epa'] * (1 + x[14])
    for weight in d_weight_values:
        test_df['d_wepa'] = test_df['d_wepa'] * test_df[weight]
    ## bound wepa to prevent extreme values from introducing volatility ##
    test_df['wepa'] = numpy.where(test_df['wepa'] > 10, 10, test_df['wepa'])
    test_df['wepa'] = numpy.where(test_df['wepa'] < -10, -10, test_df['wepa'])
    ## defense ##
    test_df['d_wepa'] = numpy.where(test_df['d_wepa'] > 10, 10, test_df['d_wepa'])
    test_df['d_wepa'] = numpy.where(test_df['d_wepa'] < -10, -10, test_df['d_wepa'])
    ## aggregate from pbp to game level ##
    game_level_df = test_df.groupby(['posteam','defteam','season','game_id','game_number']).agg(aggregation_dict).reset_index()
    game_level_df = game_level_df.sort_values(by=['posteam','game_id'])
    game_level_df = game_level_df[headers]
    ## add net epa ##
    ## create an opponent data frame ##
    game_level_opponent_df = game_level_df.copy()
    game_level_opponent_df['posteam'] = game_level_opponent_df['defteam']
    game_level_opponent_df = game_level_opponent_df.drop(columns=['defteam','season','game_number'])
    game_level_opponent_df = game_level_opponent_df.rename(columns=rename_opponent_dict)
    ## merge to main game file ##
    game_level_df = pd.merge(
        game_level_df,
        game_level_opponent_df,
        on=['posteam', 'game_id'],
        how='left'
    )
    ## calculate net wepa and apply defensive adjustment ##
    game_level_df['wepa_net'] = game_level_df['wepa'] - game_level_df['d_wepa_against']
    game_level_df['epa_net'] = game_level_df['epa'] - game_level_df['epa_against']
    ## create comparison and regressions ##
    ## split into first and second halves of the season ##
    first_half_df = game_level_df.copy()
    first_half_df = first_half_df[first_half_df['game_number'] <= 8]
    second_half_df = game_level_df.copy()
    second_half_df = second_half_df[(second_half_df['game_number'] > 8) & (second_half_df['game_number'] < 17)]
    first_half_df = first_half_df.drop(columns=['game_id', 'game_number', 'wepa', 'wepa_against', 'd_wepa_against', 'epa_against'])
    second_half_df = second_half_df.drop(columns=['game_id', 'game_number', 'wepa', 'wepa_against', 'd_wepa_against', 'epa_against'])
    ## change margins from max to sum (since its only aggregated at the game level) ##
    new_agg_dict = {'margin' : 'sum', 'wepa_net' : 'sum', 'epa_net' : 'sum',}
    first_half_df = first_half_df.groupby(['posteam', 'season']).agg(new_agg_dict).reset_index()
    second_half_df = second_half_df.groupby(['posteam', 'season']).agg(new_agg_dict).reset_index()
    ## rename the second half dict ##
    second_half_df = second_half_df.rename(columns=rename_to_last_dict)
    ## join into a single df ##
    final_df = pd.merge(
        first_half_df,
        second_half_df[['posteam', 'season', 'margin_L8']],
        on=['posteam', 'season'],
        how='left'
    )
    ## join test & train info ##
    final_df = pd.merge(
        final_df,
        team_season_df[['posteam','season','data_set']],
        on=['posteam', 'season'],
        how='left'
    )
    ## filter ##
    filtered_df = final_df[final_df['data_set'] == 'Test']
    ## grade ##
    ## return rsq of the past rates in predicting future games ##
    model = sm.OLS(filtered_df['margin_L8'],filtered_df['wepa_net'])
    results = model.fit()
    ## create checks that should persist for any optimization to ensure everything is working ##
    model_baseline = sm.OLS(filtered_df['margin_L8'],filtered_df['epa_net'])
    results_baseline = model_baseline.fit()
    second_half_margin_observations = filtered_df['margin_L8'].count()
    first_half_wepa_observations = filtered_df['wepa_net'].count()
    first_half_epa_observations = filtered_df['epa_net'].count()
    ## return all pertinant scoring and checking information ##
    return [
        ## scoring ##
        results.rsquared,
        ## checking ##
        results_baseline.rsquared,
        first_half_epa_observations,
        first_half_wepa_observations,
        second_half_margin_observations
    ]





## prep file ##
## load pbp file and prep for analysis ##
pbp_df = pd.read_csv(pbp_filepath, low_memory=False, index_col=0)

## standardize names. Gamefile is already standardized ##
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


pbp_df['posteam'] = pbp_df['posteam'].replace(pbp_team_standard_dict)
pbp_df['defteam'] = pbp_df['defteam'].replace(pbp_team_standard_dict)
pbp_df['penalty_team'] = pbp_df['penalty_team'].replace(pbp_team_standard_dict)
pbp_df['home_team'] = pbp_df['home_team'].replace(pbp_team_standard_dict)
pbp_df['away_team'] = pbp_df['away_team'].replace(pbp_team_standard_dict)

## replace game_id using standardized franchise names ##
pbp_df['game_id'] = (
    pbp_df['season'].astype('str') +
    '_' +
    pbp_df['week'].astype('str').str.zfill(2) +
    '_' +
    pbp_df['away_team'] +
    '_' +
    pbp_df['home_team']
)

## fix some data formatting issues ##
pbp_df['yards_after_catch'] = pd.to_numeric(pbp_df['yards_after_catch'], errors='coerce')

## denote pass or run ##
## seperate offensive and defensive penalties ##
pbp_df['off_penalty'] = numpy.where(pbp_df['penalty_team'] == pbp_df['posteam'], 1, 0)
pbp_df['def_penalty'] = numpy.where(pbp_df['penalty_team'] == pbp_df['defteam'], 1, 0)

## pandas wont group nans so must fill with a value ##
pbp_df['penalty_type'] = pbp_df['penalty_type'].fillna('No Penalty')

## accepted pentalites on no plays need additional detail to determine if they were a pass or run ##
## infer pass plays from the play description ##
pbp_df['desc_based_dropback'] = numpy.where(
    (
        (pbp_df['desc'].str.contains(' pass ', regex=False)) |
        (pbp_df['desc'].str.contains(' sacked', regex=False)) |
        (pbp_df['desc'].str.contains(' scramble', regex=False))
    ),
    1,
    0
)

## infer run plays from the play description ##
pbp_df['desc_based_run'] = numpy.where(
    (
        (~pbp_df['desc'].str.contains(' pass ', regex=False, na=False)) &
        (~pbp_df['desc'].str.contains(' sacked', regex=False, na=False)) &
        (~pbp_df['desc'].str.contains(' scramble', regex=False, na=False)) &
        (~pbp_df['desc'].str.contains(' kicks ', regex=False, na=False)) &
        (~pbp_df['desc'].str.contains(' punts ', regex=False, na=False)) &
        (~pbp_df['desc'].str.contains(' field goal ', regex=False, na=False)) &
        (pbp_df['desc'].str.contains(' to ', regex=False)) &
        (pbp_df['desc'].str.contains(' for ', regex=False))
    ),
    1,
    0
)


## coalesce coded and infered drop backs ##
pbp_df['qb_dropback'] = pbp_df[['qb_dropback', 'desc_based_dropback']].max(axis=1)

## coalesce coaded and infered rush attemps ##
pbp_df['rush_attempt'] = pbp_df[['rush_attempt', 'desc_based_run']].max(axis=1)


## create a specific field for play call ##
pbp_df['play_call'] = numpy.where(
                            pbp_df['qb_dropback'] == 1,
                            'Pass',
                            numpy.where(
                                pbp_df['rush_attempt'] == 1,
                                'Run',
                                numpy.nan
                            )
)

## Structure game file to attach to PBP data ##
## calc margin ##
game_file_df = pd.read_csv(game_filepath, index_col=0)
game_file_df['home_margin'] = game_file_df['home_score'] - game_file_df['away_score']
game_file_df['away_margin'] = game_file_df['away_score'] - game_file_df['home_score']

## flatten file to attach to single team
game_home_df = game_file_df.copy()[['game_id', 'week', 'season', 'home_team', 'home_margin']].rename(columns={
    'home_team' : 'posteam',
    'home_margin' : 'margin',
})
game_away_df = game_file_df.copy()[['game_id', 'week', 'season', 'away_team', 'away_margin']].rename(columns={
    'away_team' : 'posteam',
    'away_margin' : 'margin',
})
flat_game_df = pd.concat([game_home_df,game_away_df], ignore_index=True).sort_values(by=['game_id'])

## calculate game number to split in regressions ##
flat_game_df['game_number'] = flat_game_df.groupby(['posteam', 'season']).cumcount() + 1

## merge to pbp now, so you don't have to merge on every loop ##
pbp_df = pd.merge(
    pbp_df,
    flat_game_df[['posteam','game_id','margin', 'game_number']],
    on=['posteam','game_id'],
    how='left'
)

pbp_df = pbp_df[pbp_df['game_number'] < 17]





## Optimize ##
ow = []
## run 50 iterations of the test / train optimization to eliminate any variance
## that may occur from the test / train split ##
for iteration in range(0,50):
    opti_time_start = int(time.time())
    print('On optimization group {0}'.format(iteration))
    print('     On all weight optimization...')
    ## split data into training and test ##
    ## can't split at the pbp level b/c data becomes aggregated at the season & team level ##
    ## create a df with just teams and season ##
    team_season_df = pbp_df.copy()[['posteam', 'season']].drop_duplicates()
    ## assign to test or train using random ##
    team_season_df['rand_val'] = numpy.random.rand(len(team_season_df),1)
    team_season_df['data_set'] = numpy.where(team_season_df['rand_val'] < .50, 'Test', 'Train')
    ## this filter will be applied within the test and train functions once the aggregation has occured
    ## note, a team and their plays may be in the training data when they're on offense, and then
    ## in the test data when they're on defense ##
    ## run an optimization for each bound on, off, and alone to get a read on their individual
    ## and collective importance w/ this secondary validation method ##
    ## set bounds ##
    bound = (-1,2)
    bounds_l = []
    for i in best_guesses_all_time:
        bounds_l.append(bound)
    ## turn list of bounds into the tuple the optimizer needs ##
    bounds = tuple(bounds_l)
    ## optimize everything turned on ##
    ## train model ##
    solution = minimize(wepa_objective, best_guesses_all_time, args=(pbp_df.copy()), bounds=bounds, method='SLSQP')
    ## grade ##
    graded = wepa_test(solution.x, pbp_df.copy())
    opti_time_end = int(time.time())
    ow.append({
        'optimization_group' : iteration,
        'model_version' : 'all_variables',
        'rsq_train' : 1 - solution.fun,
        'rsq_wepa_test' : graded[0],
        'rsq_epa_test' : graded[1],
        'qb_rush' : solution.x[0],
        'neutral_second_down_rush' : solution.x[1],
        'incompletion_depth_s' : solution.x[2],
        'non_sack_fumble' : solution.x[3],
        'int' : solution.x[4],
        'goalline' : solution.x[5],
        'scaled_win_prob' : solution.x[6],
        'd_qb_rush' : solution.x[7],
        'd_neutral_second_down_rush' : solution.x[8],
        'd_incompletion_depth_s' : solution.x[9],
        'd_sack_fumble' : solution.x[10],
        'd_int' : solution.x[11],
        'd_fg' : solution.x[12],
        'd_third_down_pos' : solution.x[13],
        'defense_adj' : solution.x[14],
        'epa_observations_check' : graded[2],
        'wepa_observations_check' : graded[3],
        'margin_observations_check' : graded[4],
        'optimization_time' : (opti_time_end - opti_time_start) / 60,
    })
    best_guesses_current = solution.x
    ## run an optimization for each bound on, off, and alone to get a read on their individual
    ## and collective importance w/ this secondary validation method ##
    for w in range(0,len(best_guesses_all_time)):
        print('     On weight {0}...'.format(w))
        weight_name = weight_names_list[int(w)]
        ## set bounds ##
        bound = (-1,2)
        bounds_l = []
        for i in best_guesses_all_time:
            bounds_l.append(bound)
        ## zero out the ith bound to remove it from the optimization ##
        bounds_l[int(w)] = (0,0)
        ## turn list of bounds into the tuple the optimizer needs ##
        bounds = tuple(bounds_l)
        opti_time_start = int(time.time())
        ## optimize ##
        solution = minimize(wepa_objective, best_guesses_current, args=(pbp_df.copy()), bounds=bounds, method='SLSQP')
        ## grade ##
        graded = wepa_test(solution.x, pbp_df.copy())
        opti_time_end = int(time.time())
        ## append ##
        ow.append({
            'optimization_group' : iteration,
            'model_version' : '{0}_excluded'.format(weight_name),
            'rsq_train' : 1 - solution.fun,
            'rsq_wepa_test' : graded[0],
            'rsq_epa_test' : graded[1],
            'qb_rush' : solution.x[0],
            'neutral_second_down_rush' : solution.x[1],
            'incompletion_depth_s' : solution.x[2],
            'non_sack_fumble' : solution.x[3],
            'int' : solution.x[4],
            'goalline' : solution.x[5],
            'scaled_win_prob' : solution.x[6],
            'd_qb_rush' : solution.x[7],
            'd_neutral_second_down_rush' : solution.x[8],
            'd_incompletion_depth_s' : solution.x[9],
            'd_sack_fumble' : solution.x[10],
            'd_int' : solution.x[11],
            'd_fg' : solution.x[12],
            'd_third_down_pos' : solution.x[13],
            'defense_adj' : solution.x[14],
            'epa_observations_check' : graded[2],
            'wepa_observations_check' : graded[3],
            'margin_observations_check' : graded[4],
            'optimization_time' : (opti_time_end - opti_time_start) / 60,
        })
        ## do the same, but with everything else off ##
        ## set bounds ##
        bound = (0,0)
        bounds_l = []
        for i in best_guesses_all_time:
            bounds_l.append(bound)
        ## zero out the ith bound to remove it from the optimization ##
        bounds_l[int(w)] = (-1,2)
        ## turn list of bounds into the tuple the optimizer needs ##
        bounds = tuple(bounds_l)
        opti_time_start = int(time.time())
        ## optimize ##
        solution = minimize(wepa_objective, best_guesses_all_time, args=(pbp_df.copy()), bounds=bounds, method='SLSQP')
        ## grade ##
        graded = wepa_test(solution.x, pbp_df.copy())
        opti_time_end = int(time.time())
        ## append ##
        ow.append({
            'optimization_group' : iteration,
            'model_version' : '{0}_only'.format(weight_name),
            'rsq_train' : 1 - solution.fun,
            'rsq_wepa_test' : graded[0],
            'rsq_epa_test' : graded[1],
            'qb_rush' : solution.x[0],
            'neutral_second_down_rush' : solution.x[1],
            'incompletion_depth_s' : solution.x[2],
            'non_sack_fumble' : solution.x[3],
            'int' : solution.x[4],
            'goalline' : solution.x[5],
            'scaled_win_prob' : solution.x[6],
            'd_qb_rush' : solution.x[7],
            'd_neutral_second_down_rush' : solution.x[8],
            'd_incompletion_depth_s' : solution.x[9],
            'd_sack_fumble' : solution.x[10],
            'd_int' : solution.x[11],
            'd_fg' : solution.x[12],
            'd_third_down_pos' : solution.x[13],
            'defense_adj' : solution.x[14],
            'epa_observations_check' : graded[2],
            'wepa_observations_check' : graded[3],
            'margin_observations_check' : graded[4],
            'optimization_time' : (opti_time_end - opti_time_start) / 60,
        })
        output_df = pd.DataFrame(ow)
        output_df.to_csv('{0}validation_two_final.csv'.format(output_folder))
