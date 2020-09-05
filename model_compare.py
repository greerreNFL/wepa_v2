## packages used ##
import pandas as pd
import numpy
import statsmodels.api as sm


## file paths ##
wepa_filepath = 'YOUR WEPA FLAT FILEPATH'
dvoa_filepath = 'YOUT WEEKLY DVOA FILEPATH'
output_folder = 'YOUR OUTPUT FOLDER/' ## include final back slash! ##

## load data ##
wepa_df = pd.read_csv(wepa_filepath, index_col=0)
dvoa_df = pd.read_csv(dvoa_filepath, index_col=0)

## prep data ##
## prep wepa ##
## only looking at seasons w/ at least 10 years of training data ##
wepa_df = wepa_df[wepa_df['season'] >= 2009]


## prep dvoa data ##
## add game number ##
dvoa_df = dvoa_df.sort_values(by=['team', 'season', 'week'])
## teams will have a row for every week even if they didnt play ##
## determined games played by record ##
dvoa_records = dvoa_df['record'].str.split('-',expand=True)
dvoa_records = dvoa_records.rename(columns={
    0 : 'wins',
    1 : 'losses',
    2 : 'ties',
})
## convert record strings to ints ##
dvoa_records['wins'] = dvoa_records['wins'].astype(int)
dvoa_records['losses'] = dvoa_records['losses'].astype(int)
dvoa_records['ties'] = dvoa_records['ties'].fillna(value='0').astype(int)
## calc games played ##
dvoa_records['game_number'] = dvoa_records['wins'] + dvoa_records['losses'] + dvoa_records['ties']
## merge to dvoa_df ##
dvoa_df = dvoa_df.join(
    dvoa_records
)
## dvoa will update over the teams bye week as SoS changes ##
## only keep game before the bye so as not to include future info ##
dvoa_df = dvoa_df.drop_duplicates(subset=['team','season','game_number'], keep='first')



## define function for comparing by season & game ##
def model_compare_season(season_of, through_game):
    ## split data by before and after game number ##
    wepa_df_pre = wepa_df.copy()
    wepa_df_pre = wepa_df_pre[
        (wepa_df_pre['season'] == season_of) &
        (wepa_df_pre['game_number'] <= through_game)
    ]
    ## aggregate data ##
    agg_wepa_df_pre = wepa_df_pre.groupby(['season', 'team']).agg(
        wepa_pre = ('wepa_net', 'sum'),
        margin_pre = ('margin', 'sum'),
    ).reset_index()
    wepa_df_post = wepa_df.copy()
    wepa_df_post = wepa_df_post[
        (wepa_df_post['season'] == season_of) &
        (wepa_df_post['game_number'] > through_game)
    ]
    ## aggregate data ##
    agg_wepa_df_post = wepa_df_post.groupby(['season', 'team']).agg(
        wepa_post = ('wepa_net', 'sum'),
        margin_post = ('margin', 'sum'),
    ).reset_index()
    ## merge ##
    merged_wepa_df = pd.merge(
        agg_wepa_df_pre,
        agg_wepa_df_post,
        on=['season', 'team'],
        how='left'
    )
    ## filter dvoa data ##
    dvoa_df_filtered = dvoa_df.copy()
    dvoa_df_filtered = dvoa_df_filtered[
        (dvoa_df_filtered['season'] == season_of) &
        (dvoa_df_filtered['game_number'] == through_game)
    ]
    dvoa_df_filtered = dvoa_df_filtered[[
        'season',
        'team',
        'total_dvoa',
        'weighted_dvoa'
    ]]
    ## merge ##
    merged_wepa_df = pd.merge(
        merged_wepa_df,
        dvoa_df_filtered,
        on=['season', 'team'],
        how='left'
    )
    ## run regressions ##
    merged_wepa_df['intercept_constant'] = 1
    model_margin = sm.OLS(merged_wepa_df['margin_post'], merged_wepa_df[['margin_pre','intercept_constant']], hasconst=True)
    results_margin = model_margin.fit()
    model_wepa = sm.OLS(merged_wepa_df['margin_post'], merged_wepa_df[['wepa_pre','intercept_constant']], hasconst=True)
    results_wepa = model_wepa.fit()
    model_total_dvoa = sm.OLS(merged_wepa_df['margin_post'], merged_wepa_df[['total_dvoa','intercept_constant']], hasconst=True)
    results_total_dvoa = model_total_dvoa.fit()
    weighted_dvoa = sm.OLS(merged_wepa_df['margin_post'], merged_wepa_df[['weighted_dvoa','intercept_constant']], hasconst=True)
    results_weighted_dvoa = weighted_dvoa.fit()
    ## return results ##
    return [
        results_margin.rsquared,
        results_wepa.rsquared,
        results_total_dvoa.rsquared,
        results_weighted_dvoa.rsquared,
        ## add checks to make sure every team was matched correctly ##
        len(merged_wepa_df) - merged_wepa_df['margin_post'].count(),
        len(merged_wepa_df) - merged_wepa_df['wepa_post'].count(),
        len(merged_wepa_df) - merged_wepa_df['total_dvoa'].count(),
        len(merged_wepa_df) - merged_wepa_df['weighted_dvoa'].count()
    ]



## define function for comparing by season & game ##
def model_compare_all(through_game):
    ## split data by before and after game number ##
    wepa_df_pre = wepa_df.copy()
    wepa_df_pre = wepa_df_pre[
        (wepa_df_pre['game_number'] <= through_game)
    ]
    ## aggregate data ##
    agg_wepa_df_pre = wepa_df_pre.groupby(['season', 'team']).agg(
        wepa_pre = ('wepa_net', 'sum'),
        margin_pre = ('margin', 'sum'),
    ).reset_index()
    wepa_df_post = wepa_df.copy()
    wepa_df_post = wepa_df_post[
        (wepa_df_post['game_number'] > through_game)
    ]
    ## aggregate data ##
    agg_wepa_df_post = wepa_df_post.groupby(['season', 'team']).agg(
        wepa_post = ('wepa_net', 'sum'),
        margin_post = ('margin', 'sum'),
    ).reset_index()
    ## merge ##
    merged_wepa_df = pd.merge(
        agg_wepa_df_pre,
        agg_wepa_df_post,
        on=['season', 'team'],
        how='left'
    )
    ## filter dvoa data ##
    dvoa_df_filtered = dvoa_df.copy()
    dvoa_df_filtered = dvoa_df_filtered[
        (dvoa_df_filtered['game_number'] == through_game)
    ]
    dvoa_df_filtered = dvoa_df_filtered[[
        'season',
        'team',
        'total_dvoa',
        'weighted_dvoa'
    ]]
    ## merge ##
    merged_wepa_df = pd.merge(
        merged_wepa_df,
        dvoa_df_filtered,
        on=['season', 'team'],
        how='left'
    )
    ## run regressions ##
    merged_wepa_df['intercept_constant'] = 1
    model_margin = sm.OLS(merged_wepa_df['margin_post'], merged_wepa_df[['margin_pre','intercept_constant']], hasconst=True)
    results_margin = model_margin.fit()
    model_wepa = sm.OLS(merged_wepa_df['margin_post'], merged_wepa_df[['wepa_pre','intercept_constant']], hasconst=True)
    results_wepa = model_wepa.fit()
    model_total_dvoa = sm.OLS(merged_wepa_df['margin_post'], merged_wepa_df[['total_dvoa','intercept_constant']], hasconst=True)
    results_total_dvoa = model_total_dvoa.fit()
    weighted_dvoa = sm.OLS(merged_wepa_df['margin_post'], merged_wepa_df[['weighted_dvoa','intercept_constant']], hasconst=True)
    results_weighted_dvoa = weighted_dvoa.fit()
    ## return results ##
    return [
        results_margin.rsquared,
        results_wepa.rsquared,
        results_total_dvoa.rsquared,
        results_weighted_dvoa.rsquared,
        ## add checks to make sure every team was matched correctly ##
        len(merged_wepa_df) - merged_wepa_df['margin_post'].count(),
        len(merged_wepa_df) - merged_wepa_df['wepa_post'].count(),
        len(merged_wepa_df) - merged_wepa_df['total_dvoa'].count(),
        len(merged_wepa_df) - merged_wepa_df['weighted_dvoa'].count()
    ]



## run and export ##
## containers for holding row data ##
ow_season = []
ow_all = []

## iterate seasons and game_numbers ##
for season in range(2009,2020):
    for game_num in range(1,16):
        func_output = model_compare_season(season, game_num)
        ow_season.append({
            'season' : season,
            'through_x_games' : game_num,
            'margin_rsq' : func_output[0],
            'wepa_rsq' : func_output[1],
            'total_dvoa_rsq' : func_output[2],
            'weighted_dvoa_rsq' : func_output[3],
            'margin_join_errors' : func_output[4],
            'wepa_join_errors' : func_output[5],
            'total_dvoa_join_errors' : func_output[6],
            'weighted_dvoa_join_errors' : func_output[7],
        })

## iterate just game_numbers ##
for game_number in range(1,16):
    func_output = model_compare_all(game_number)
    ow_all.append({
        'through_x_games' : game_number,
        'margin_rsq' : func_output[0],
        'wepa_rsq' : func_output[1],
        'total_dvoa_rsq' : func_output[2],
        'weighted_dvoa_rsq' : func_output[3],
        'margin_join_errors' : func_output[4],
        'wepa_join_errors' : func_output[5],
        'total_dvoa_join_errors' : func_output[6],
        'weighted_dvoa_join_errors' : func_output[7],
    })

pd.DataFrame(ow_season).to_csv('{0}model_compare_by_season.csv'.format(output_folder))
pd.DataFrame(ow_all).to_csv('{0}model_compare_all.csv'.format(output_folder))


## compare yoy stability ##
wepa_yoy_df = wepa_df.copy()

## agg stats at the season level ##
wepa_yoy_df = wepa_yoy_df.groupby(['team','season']).agg(
    wepa = ('wepa_net', 'sum'),
    margin = ('margin', 'sum'),
).reset_index()

## add dvoa info ##
dvoa_yoy_df = dvoa_df.copy()
## need as of the teams last game for the complete season dvoa ##
dvoa_yoy_df = dvoa_yoy_df[dvoa_yoy_df['game_number'] == 16]
## merge to wepa df ##
wepa_yoy_df = pd.merge(
    wepa_yoy_df,
    dvoa_yoy_df,
    on=['season', 'team'],
    how='left'
)

## add next years values ##
wepa_yoy_df['margin_t_1'] = wepa_yoy_df.groupby(['team'])['margin'].shift(-1)
wepa_yoy_df['wins_t_1'] = wepa_yoy_df.groupby(['team'])['wins'].shift(-1)
wepa_yoy_df['wepa_t_1'] = wepa_yoy_df.groupby(['team'])['wepa'].shift(-1)
wepa_yoy_df['total_dvoa_t_1'] = wepa_yoy_df.groupby(['team'])['total_dvoa'].shift(-1)
wepa_yoy_df['weighted_dvoa_t_1'] = wepa_yoy_df.groupby(['team'])['weighted_dvoa'].shift(-1)

## add constant for regressions ##
wepa_yoy_df[['intercept_constant'] = 1

## create matrix of rsqs ##
owm = []
metric_list = ['margin', 'wins', 'wepa', 'total_dvoa', 'weighted_dvoa']
for x in metric_list:
    ## current period ##
    for y in metric_list:
        model = sm.OLS(
            wepa_yoy_df[y],
            wepa_yoy_df[[x,'intercept_constant']],
            hasconst=True
        ).fit()
        owm.append({
            'comparison' : '{0}_x_{1}'.format(x,y),
            'rsq' : model.rsquared,
        })
    ## forward period ##
    for t in metric_list:
        model = sm.OLS(
            wepa_yoy_df['{0}_t_1'.format(t)],
            wepa_yoy_df[[x,'intercept_constant']],
            hasconst=True
        ).fit()
        owm.append({
            'comparison' : '{0}_x_{1}_t_1'.format(x,t),
            'rsq' : model.rsquared,
        })

pd.DataFrame(owm).to_csv('{0}yoy_rsq_matrix.csv'.format(output_folder))
