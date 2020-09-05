## packages used ##
import pandas as pd
import numpy
import statsmodels.api as sm
import math



## file paths ##
pbp_filepath = 'YOUR PBP FILEPATH'
game_filepath = 'YOUR GAME FILEPATH' ## needed to calc accurate margins ##
output_folder = 'YOUR OUTPUT FOLDER/' ## include final back slash! ##


## function for applying weights ##
def feature_test(pbp_df, weight):
    ## define weights ##
    ## play style ##
    pbp_df['pass_weight'] = numpy.where(pbp_df['pass_attempt'] == 1, 1 + weight, 1)
    pbp_df['pass_ex_ints_weight'] = numpy.where((pbp_df['pass_attempt'] == 1) & (pbp_df['interception'] != 1) , 1 + weight, 1)
    pbp_df['rush_weight'] = numpy.where((pbp_df['rush_attempt'] == 1) & (pbp_df['qb_scramble'] != 1), 1 + weight, 1)
    pbp_df['pos_epa_rush_weight'] = numpy.where((pbp_df['rush_attempt'] == 1) & (pbp_df['qb_scramble'] != 1) & (pbp_df['epa'] > 0), 1 + weight, 1)
    pbp_df['neg_epa_rush_weight'] = numpy.where((pbp_df['rush_attempt'] == 1) & (pbp_df['qb_scramble'] != 1) & (pbp_df['epa'] < 0), 1 + weight, 1)
    pbp_df['short_yardage_rush_weight'] = numpy.where(
        (pbp_df['rush_attempt'] == 1) &
        (pbp_df['qb_scramble'] != 1) &
        (pbp_df['ydstogo'] < 5),
        1 + weight,
        1
    )
    pbp_df['long_yardage_rush_weight'] = numpy.where(
        (pbp_df['rush_attempt'] == 1) &
        (pbp_df['qb_scramble'] != 1) &
        (pbp_df['ydstogo'] >= 5),
        1 + weight,
        1
    )
    pbp_df['short_yardage_pos_rush_weight'] = numpy.where(
        (pbp_df['rush_attempt'] == 1) &
        (pbp_df['qb_scramble'] != 1) &
        (pbp_df['ydstogo'] < 5) &
        (pbp_df['epa'] > 0),
        1 + weight,
        1
    )
    pbp_df['short_yardage_neg_rush_weight'] = numpy.where(
        (pbp_df['rush_attempt'] == 1) &
        (pbp_df['qb_scramble'] != 1) &
        (pbp_df['ydstogo'] < 5) &
        (pbp_df['epa'] < 0),
        1 + weight,
        1
    )
    pbp_df['long_yardage_pos_rush_weight'] = numpy.where(
        (pbp_df['rush_attempt'] == 1) &
        (pbp_df['qb_scramble'] != 1) &
        (pbp_df['ydstogo'] >= 5) &
        (pbp_df['epa'] > 0),
        1 + weight,
        1
    )
    pbp_df['long_yardage_neg_rush_weight'] = numpy.where(
        (pbp_df['rush_attempt'] == 1) &
        (pbp_df['qb_scramble'] != 1) &
        (pbp_df['ydstogo'] >= 5) &
        (pbp_df['epa'] < 0),
        1 + weight,
        1
    )
    pbp_df['qb_rush_weight'] = numpy.where(pbp_df['qb_scramble'] == 1, 1 + weight, 1)
    pbp_df['completion_weight'] = numpy.where(pbp_df['complete_pass'] == 1, 1 + weight, 1)
    pbp_df['completion_over_20_weight'] = numpy.where((pbp_df['complete_pass'] == 1) & (pbp_df['air_yards'] >= 20), 1 + weight, 1)
    pbp_df['completion_between_10_20_weight'] = numpy.where((pbp_df['complete_pass'] == 1) & (pbp_df['air_yards'] >= 10) & (pbp_df['air_yards'] < 20), 1 + weight, 1)
    pbp_df['completion_under_10_weight'] = numpy.where((pbp_df['complete_pass'] == 1) & (pbp_df['air_yards'] < 10), 1 + weight, 1)
    pbp_df['completion_depth_linear_weight'] = 1 + numpy.where(
        pbp_df['complete_pass'] == 1,
        numpy.where(numpy.isnan((pbp_df['air_yards'] - 10) * (weight / 10)), 0, (pbp_df['air_yards'] - 10) * (weight / 10)),
        0
    )
    pbp_df['completion_depth_s_weight'] = 1 + numpy.where(
        pbp_df['complete_pass'] == 1,
        numpy.where(numpy.isnan(weight * (2 * (1/(1 + numpy.exp(-0.1 * pbp_df['air_yards'] + .75)) - 0.5))),0,(weight * (2 * (1/(1 + numpy.exp(-0.1 * pbp_df['air_yards'] + .75)) - 0.5)))),
        0
    )
    pbp_df['completion_gained_s_weight'] = 1 + numpy.where(
        pbp_df['complete_pass'] == 1,
        numpy.where(numpy.isnan(weight * (2 * (1/(1 + numpy.exp(-0.1 * pbp_df['yards_gained'] + .75)) - 0.5))),0,(weight * (2 * (1/(1 + numpy.exp(-0.1 * pbp_df['yards_gained'] + .75)) - 0.5)))),
        0
    )
    pbp_df['yac_under_05_weight'] = numpy.where((pbp_df['complete_pass'] == 1) & (pbp_df['yards_after_catch'] < 5), 1 + weight, 1)
    pbp_df['yac_between_05_10_weight'] = numpy.where((pbp_df['complete_pass'] == 1) & (pbp_df['yards_after_catch'] >= 5) & (pbp_df['yards_after_catch'] < 10), 1 + weight, 1)
    pbp_df['yac_between_10_15_weight'] = numpy.where((pbp_df['complete_pass'] == 1) & (pbp_df['yards_after_catch'] >= 10) & (pbp_df['yards_after_catch'] < 15), 1 + weight, 1)
    pbp_df['yac_between_15_20_weight'] = numpy.where((pbp_df['complete_pass'] == 1) & (pbp_df['yards_after_catch'] >= 15) & (pbp_df['yards_after_catch'] < 20), 1 + weight, 1)
    pbp_df['yac_over_20_weight'] = numpy.where((pbp_df['complete_pass'] == 1) & (pbp_df['yards_after_catch'] >= 20), 1 + weight, 1)
    pbp_df['incompletion_weight'] = numpy.where((pbp_df['incomplete_pass'] == 1) & (pbp_df['interception'] != 1), 1 + weight, 1)
    pbp_df['incompletion_depth_s_weight'] = 1 + numpy.where(
        (pbp_df['incomplete_pass'] == 1) & (pbp_df['interception'] != 1),
        numpy.where(numpy.isnan(weight * (2 * (1/(1 + numpy.exp(-0.1 * pbp_df['air_yards'] + .75)) - 0.5))),0,(weight * (2 * (1/(1 + numpy.exp(-0.1 * pbp_df['air_yards'] + .75)) - 0.5)))),
        0
    )
    ## penalties ##
    pbp_df['off_holding_weight'] = numpy.where(pbp_df['penalty_type'] == 'Offensive Holding', 1 + weight, 1)
    pbp_df['dpi_weight'] = numpy.where(pbp_df['penalty_type'] == 'Defensive Pass Interference', 1 + weight, 1)
    pbp_df['false_start_weight'] = numpy.where(pbp_df['penalty_type'] == 'False Start', 1 + weight, 1)
    pbp_df['roughing_passer_weight'] = numpy.where(pbp_df['penalty_type'] == 'Roughing the Passer', 1 + weight, 1)
    pbp_df['def_holding_weight'] = numpy.where(pbp_df['penalty_type'] == 'Defensive Holding', 1 + weight, 1)
    pbp_df['illegal_block_weight'] = numpy.where(pbp_df['penalty_type'] == 'Illegal Block Above the Waist', 1 + weight, 1)
    pbp_df['def_offside_weight'] = numpy.where(pbp_df['penalty_type'] == 'Defensive Offside', 1 + weight, 1)
    pbp_df['off_unsportsmanlike_weight'] = numpy.where((pbp_df['penalty_type'] == 'Unsportsmanlike Conduct') & (pbp_df['posteam'] == pbp_df['penalty_team']), 1 + weight, 1)
    pbp_df['off_uncessary_roughness_weight'] = numpy.where((pbp_df['penalty_type'] == 'Unnecessary Roughness') & (pbp_df['posteam'] == pbp_df['penalty_team']), 1 + weight, 1)
    pbp_df['def_unsportsmanlike_weight'] = numpy.where((pbp_df['penalty_type'] == 'Unsportsmanlike Conduct') & (pbp_df['posteam'] != pbp_df['penalty_team']), 1 + weight, 1)
    pbp_df['def_uncessary_roughness_weight'] = numpy.where((pbp_df['penalty_type'] == 'Unnecessary Roughness') & (pbp_df['posteam'] != pbp_df['penalty_team']), 1 + weight, 1)
    pbp_df['opi_weight'] = numpy.where(pbp_df['penalty_type'] == 'Offensive Pass Interference', 1 + weight, 1)
    pbp_df['ol_penalty_weight'] = numpy.where(numpy.isin(pbp_df['penalty_type'],['False Start', 'Offensive Holding']), 1 + weight, 1)
    ## events ##
    pbp_df['qb_hit_pos_epa_weight'] = numpy.where((pbp_df['qb_hit'] == 1) & (pbp_df['interception'] != 1) & (pbp_df['epa'] > 0), 1 + weight, 1)
    pbp_df['qb_hit_neg_epa_weight'] = numpy.where((pbp_df['qb_hit'] == 1) & (pbp_df['interception'] != 1) & (pbp_df['epa'] < 0), 1 + weight, 1)
    pbp_df['sack_weight'] = numpy.where(pbp_df['sack'] == 1, 1 + weight, 1)
    pbp_df['fumble_weight'] = numpy.where(pbp_df['fumble_lost'] == 1, 1 + weight, 1)
    pbp_df['all_fumble_weight'] = numpy.where(pbp_df['fumble'] == 1, 1 + weight, 1)
    pbp_df['sack_fumble_weight'] = numpy.where((pbp_df['sack'] == 1) & (pbp_df['fumble_lost'] == 1), 1 + weight, 1)
    pbp_df['non_fumble_sack_weight'] = numpy.where((pbp_df['sack'] == 1) & (pbp_df['fumble_lost'] != 1), 1 + weight, 1)
    pbp_df['non_sack_fumble_weight'] = numpy.where((pbp_df['sack'] != 1) & (pbp_df['fumble_lost'] == 1), 1 + weight, 1)
    pbp_df['int_weight'] = numpy.where(pbp_df['interception'] == 1, 1 + weight, 1)
    pbp_df['return_td_weight'] = numpy.where(pbp_df['return_touchdown'] == 1,
        numpy.where(pbp_df['play_type'] == 'kickoff', -1 * (1 + weight), 1 + weight),
    1)
    pbp_df['punt_block_weight'] = numpy.where(pbp_df['punt_blocked'] == 1, 1 + weight, 1)
    pbp_df['fg_weight'] = numpy.where(pbp_df['play_type'] == 'field_goal', 1 + weight, 1)
    pbp_df['kickoff_weight'] = numpy.where(pbp_df['play_type'] == 'kickoff', 1 + weight, 1)
    pbp_df['punt_weight'] = numpy.where(pbp_df['play_type'] == 'punt', 1 + weight, 1)
    ## contextual ##
    pbp_df['first_down_weight'] = numpy.where((pbp_df['down'] == 1), 1 + weight, 1)
    pbp_df['second_down_weight'] = numpy.where((pbp_df['down'] == 2), 1 + weight, 1)
    pbp_df['third_down_weight'] = numpy.where((pbp_df['down'] == 3), 1 + weight, 1)
    pbp_df['fourth_down_weight'] = numpy.where(
        (pbp_df['down'] == 4) &
        (
            (pbp_df['pass_attempt'] == 1) |
            (pbp_df['rush_attempt'] == 1)
        ),
        1 + weight,
        1
    )
    pbp_df['third_down_ex_weight'] = numpy.where(
        (pbp_df['down'] == 3) &
        (pbp_df['interception'] != 1) &
        (pbp_df['sack'] != 1),
        1 + weight,
        1
    )
    pbp_df['third_down_pos_ex_weight'] = numpy.where(
        (pbp_df['down'] == 3) &
        (pbp_df['interception'] != 1) &
        (pbp_df['epa'] > 0) &
        (pbp_df['sack'] != 1),
        1 + weight,
        1
    )
    pbp_df['third_down_long_ex_weight'] = numpy.where(
        (pbp_df['down'] == 3) &
        (pbp_df['interception'] != 1) &
        (pbp_df['ydstogo'] > 5) &
        (pbp_df['sack'] != 1),
        1 + weight,
        1
    )
    pbp_df['first_down_rush_weight'] = numpy.where((pbp_df['down'] == 1) & (pbp_df['play_call'] == 'Run'), 1 + weight, 1)
    pbp_df['second_down_rush_weight'] = numpy.where((pbp_df['down'] == 2) & (pbp_df['play_call'] == 'Run'), 1 + weight, 1)
    pbp_df['third_down_rush_weight'] = numpy.where((pbp_df['down'] == 3) & (pbp_df['play_call'] == 'Run'), 1 + weight, 1)
    pbp_df['fourth_down_rush_weight'] = numpy.where((pbp_df['down'] == 4) & (pbp_df['play_call'] == 'Run'), 1 + weight, 1)
    pbp_df['first_down_pass_weight'] = numpy.where((pbp_df['down'] == 1) & (pbp_df['play_call'] == 'Pass'), 1 + weight, 1)
    pbp_df['second_down_pass_weight'] = numpy.where((pbp_df['down'] == 2) & (pbp_df['play_call'] == 'Pass'), 1 + weight, 1)
    pbp_df['third_down_pass_weight'] = numpy.where((pbp_df['down'] == 3) & (pbp_df['play_call'] == 'Pass'), 1 + weight, 1)
    pbp_df['fourth_down_pass_weight'] = numpy.where((pbp_df['down'] == 4) & (pbp_df['play_call'] == 'Pass'), 1 + weight, 1)
    pbp_df['neutral_second_down_rush_weight'] = numpy.where(
        (pbp_df['down'] == 2) &
        (pbp_df['play_call'] == 'Run') &
        (pbp_df['yardline_100'] > 20) &
        (pbp_df['yardline_100'] < 85) &
        ((pbp_df['wp'] < .90) | (pbp_df['wp'] > .10)),
        1 + weight,
        1
    )
    pbp_df['early_down_weight'] = numpy.where((pbp_df['down'] == 1) | (pbp_df['down'] == 2), 1 + weight, 1)
    pbp_df['early_down_non_fumble_sack_weight'] = numpy.where(
        ((pbp_df['down'] == 1) | (pbp_df['down'] == 2)) &
        (pbp_df['sack'] == 1) &
        (pbp_df['fumble_lost'] != 1),
        1 + weight,
        1
    )
    pbp_df['red_zone_weight'] = numpy.where((pbp_df['yardline_100'] <= 20), 1 + weight, 1)
    pbp_df['goal_to_go_weight'] = numpy.where((pbp_df['yardline_100'] == pbp_df['ydstogo']) & (pbp_df['down'] < 4), 1 + weight, 1)
    pbp_df['goalline_weight'] = numpy.where((pbp_df['yardline_100'] < 3) & (pbp_df['down'] < 4), 1 + weight, 1)
    pbp_df['plus_territory_weight'] = numpy.where((pbp_df['yardline_100'] <= 50), 1 + weight, 1)
    pbp_df['low_win_prob_weight'] = numpy.where(((pbp_df['wp'] >= .95) | (pbp_df['wp'] <= .5)), 1 + weight,1)
    pbp_df['garbage_time_weight'] = numpy.where(((pbp_df['wp'] >= .95) | (pbp_df['wp'] <= .5)) & (pbp_df['game_seconds_remaining'] <= 900), 1 + weight,1)
    pbp_df['asym_low_win_prob_weight'] = numpy.where(((pbp_df['wp'] >= .95) | (pbp_df['wp'] <= .20)), 1 + weight,1)
    pbp_df['asym_garbage_time_weight'] = numpy.where(((pbp_df['wp'] >= .95) | (pbp_df['wp'] <= .20)) & (pbp_df['game_seconds_remaining'] <= 900), 1 + weight,1)
    pbp_df['scaled_win_prob_weight'] = 1 + (-weight * numpy.where(pbp_df['wp'] <= .5, 1/(1+numpy.exp(-10*(2*pbp_df['wp']-0.5)))-0.5,1/(1+numpy.exp(-10*(2*(1-pbp_df['wp'])-0.5)))-0.5))
    ## meta ##
    pbp_df['posteam_home_weight'] = numpy.where(pbp_df['posteam'] == pbp_df['home_team'], 1 + weight, 1)
    pbp_df['posteam_away_weight'] = numpy.where(pbp_df['posteam'] == pbp_df['away_team'], 1 + weight, 1)
    ## add weights to list to build out headers and loops ##
    weight_names = [
        'pass',
        'pass_ex_ints',
        'rush',
        'pos_epa_rush',
        'neg_epa_rush',
        'short_yardage_rush',
        'long_yardage_rush',
        'short_yardage_pos_rush',
        'short_yardage_neg_rush',
        'long_yardage_pos_rush',
        'long_yardage_neg_rush',
        'qb_rush',
        'completion',
        'completion_over_20',
        'completion_between_10_20',
        'completion_under_10',
        'completion_depth_linear',
        'completion_depth_s',
        'completion_gained_s',
        'yac_under_05',
        'yac_between_05_10',
        'yac_between_10_15',
        'yac_between_15_20',
        'yac_over_20',
        'incompletion',
        'incompletion_depth_s',
        ## penalties ##
        'off_holding',
        'dpi',
        'false_start',
        'roughing_passer',
        'def_holding',
        'illegal_block',
        'def_offside',
        'off_uncessary_roughness',
        'off_unsportsmanlike',
        'def_uncessary_roughness',
        'def_unsportsmanlike',
        'opi',
        'ol_penalty',
        ## events ##
        'qb_hit_pos_epa',
        'qb_hit_neg_epa',
        'sack',
        'fumble',
        'all_fumble',
        'sack_fumble',
        'non_fumble_sack',
        'non_sack_fumble',
        'int',
        'return_td',
        'punt_block',
        'fg',
        'kickoff',
        'punt',
        ## contextual ##
        'first_down',
        'second_down',
        'third_down',
        'fourth_down',
        'third_down_ex',
        'third_down_pos_ex',
        'third_down_long_ex',
        'first_down_rush',
        'second_down_rush',
        'third_down_rush',
        'fourth_down_rush',
        'first_down_pass',
        'second_down_pass',
        'third_down_pass',
        'fourth_down_pass',
        'neutral_second_down_rush',
        'early_down',
        'early_down_non_fumble_sack',
        'red_zone',
        'goal_to_go',
        'goalline',
        'plus_territory',
        'low_win_prob',
        'garbage_time',
        'asym_low_win_prob',
        'asym_garbage_time',
        'scaled_win_prob',
        ## meta ##
        'posteam_home',
        'posteam_away'
    ]
    ## add a baseline weight for comparison ##
    pbp_df['baseline_weight'] = 1
    weight_names.append('baseline')
    ## create structures for aggregation ##
    aggregation_dict = {
        'margin' : 'max', ## game level margin added to each play, so take max to get 1 ##
    }
    headers = [
        'game_id',
        'posteam',
        'defteam',
        'season',
        'game_number',
        'margin'
    ]
    ## dictionary to rename second half of the season metrics ##
    rename_to_last_dict = {
        'margin' : 'margin_L8',
    }
    ## disctionary to join oppoenets epa to net out ##
    rename_opponent_dict = {
        'margin' : 'margin_against',
    }
    ## add each weight to aggregation structures ##
    for weight_name in weight_names:
        pbp_df['{0}_epa'.format(weight_name)] = pbp_df['epa'] * pbp_df['{0}_weight'.format(weight_name)]
        aggregation_dict['{0}_epa'.format(weight_name)] = 'sum' ## calc total epa for the game ##
        headers.append('{0}_epa'.format(weight_name))
        rename_to_last_dict['{0}_epa'.format(weight_name)] = '{0}_epa_L8'.format(weight_name)
        rename_opponent_dict['{0}_epa'.format(weight_name)] = '{0}_epa_against'.format(weight_name)
    ## aggregate from pbp to game level ##
    game_level_df = pbp_df.groupby(['posteam','defteam','season','game_id','game_number']).agg(aggregation_dict).reset_index()
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
    ## calculate net epa ##
    for weight_name in weight_names:
        game_level_df['{0}_epa_net'.format(weight_name)] = game_level_df['{0}_epa'.format(weight_name)] - game_level_df['{0}_epa_against'.format(weight_name)]
    ## create seperate weight for defensive weighting of the baseline epa ##
    game_level_df['defense_adj_epa_net'] = game_level_df['baseline_epa'] - ((1 + weight) * game_level_df['baseline_epa_against'])
    ## add defensive weighting to the weight list for iteration in regression ##
    weight_names.append('defense_adj')
    ## create comparison and regressions ##
    ## split into first and second halves of the season ##
    first_half_df = game_level_df.copy()
    first_half_df = first_half_df[first_half_df['game_number'] <= 8]
    second_half_df = game_level_df.copy()
    second_half_df = second_half_df[(second_half_df['game_number'] > 8) & (second_half_df['game_number'] < 17)]
    first_half_df = first_half_df.drop(columns=['game_id', 'game_number'])
    second_half_df = second_half_df.drop(columns=['game_id', 'game_number'])
    ## change margins from max to sum (since its only aggregated at the game level) ##
    aggregation_dict['margin'] = 'sum'
    ## add net epa to agg dict ##
    for weight_name in weight_names:
        aggregation_dict['{0}_epa_net'.format(weight_name)] = 'sum'
    first_half_df = first_half_df.groupby(['posteam', 'season']).agg(aggregation_dict).reset_index()
    second_half_df = second_half_df.groupby(['posteam', 'season']).agg(aggregation_dict).reset_index()
    ## rename the second half dict ##
    second_half_df = second_half_df.rename(columns=rename_to_last_dict)
    ## join into a single df ##
    final_df = pd.merge(
        first_half_df,
        second_half_df[['posteam', 'season', 'margin_L8']],
        on=['posteam', 'season'],
        how='left'
    )
    ## output this file for QA ##
    final_df.to_csv('{0}first_second_half_game_file.csv'.format(output_folder))
    ## calculate rsq's for an individual season ##
    for season in range(1999,2020):
        print('     Calculating Seasons...')
        result_row_season = {
            'weight' : weight,
            'season' : season,
        }
        final_df_season = final_df.copy()
        final_df_season = final_df_season[final_df_season['season'] == season]
        for weight_name in weight_names:
            model = sm.OLS(final_df_season['margin_L8'], final_df_season['{0}_epa_net'.format(weight_name)])
            results = model.fit()
            result_row_season['{0}'.format(weight_name)] = results.rsquared
        season_data.append(result_row_season)
    ## calculate windowed view  ##
    ## this constrains the seasons used to measure how weights have changed ##
    for window in range(1,21):
        print('     Calculating Windows...')
        result_row_window = {
            'weight' : weight,
            'season' : 2019,
            'window' : window,
        }
        final_df_window = final_df.copy()
        final_df_window = final_df_window[final_df_window['season'] > 2019 - window]
        for weight_name in weight_names:
            model = sm.OLS(final_df_window['margin_L8'], final_df_window['{0}_epa_net'.format(weight_name)])
            results = model.fit()
            result_row_window['{0}'.format(weight_name)] = results.rsquared
        window_data.append(result_row_window)
        result_row_window = {
            'weight' : weight,
            'season' : 1999,
            'window' : window,
        }
        final_df_window = final_df.copy()
        final_df_window = final_df_window[final_df_window['season'] < 1999 + window]
        for weight_name in weight_names:
            model = sm.OLS(final_df_window['margin_L8'], final_df_window['{0}_epa_net'.format(weight_name)])
            results = model.fit()
            result_row_window['{0}'.format(weight_name)] = results.rsquared
        window_data.append(result_row_window)




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


## run regressions ##
## create weight list ##
list_of_weights = []
for i in range(-200,201):
    list_of_weights.append(i/100)


season_data = []
window_data = []

for i in list_of_weights:
    print('On {0}...'.format(i))
    feature_test(pbp_df, i)
    pd.DataFrame(season_data).to_csv('{0}season_rsqs.csv'.format(output_folder))
    pd.DataFrame(window_data).to_csv('{0}window_rsqs.csv'.format(output_folder))
