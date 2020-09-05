## package for determining what data is needed ##
import os
import glob
## packages for pulling data ##
import requests
from bs4 import BeautifulSoup
import csv
import time
from datetime import datetime
import random
## packages for manipulating data ##
import pandas as pd
import numpy


## high level script params ##
user_name = 'YOUR FBO USER NAME'
password = 'YOUR FBO PASSWORD'
data_folder = 'YOUR FOLDER' ## no final backslash here ##
season_start = 2009


if int(datetime.today().month) < 6:
    season_end = int(datetime.today().year) - 1
else:
    season_end = int(datetime.today().year)

week_start = 1
week_end = 17


## get login form id ##
url = 'https://www.footballoutsiders.com/'
raw = requests.get(url)
parsed = BeautifulSoup(raw.content, "html.parser")

form = parsed.find_all('form', {'class' : 'user-login-form'})[0]
form_id = form.find_all('input', {'name' : 'form_build_id'})[0]['value']

## construct login payload ##
payload = {
    'name' : user_name,
    'pass' : password,
    'form_build_id' : form_id,
    'form_id' : 'user_login_form',
    'op' : 'Login',
}

## start a requests session and login ##
s = requests.session()
s.post(url, data=payload)


## get weekly data ##
## find any existing data ##
print('Searching for most recent weekly DVOA data...')
try:
    most_recent_df = pd.read_csv('{0}/weekly_dvoa_data.csv'.format(data_folder))
    most_recent_season = int(most_recent_df['season'].max())
    most_recent_week = int(most_recent_df[most_recent_df['season'] == most_recent_season]['week'].max())
    print('     Found data as of Week {0}, {1}...'.format(most_recent_week, most_recent_season))
except:
    most_recent_df = None
    most_recent_season = 0
    most_recent_week = 0
    print('     Did not find existing database. Will rebuild from 2009...')


## set start week for data pull ##
current_season = None
current_week = None

if most_recent_week == 0:
    current_season = season_start
    current_week = 1
elif most_recent_week == 17:
    current_season = most_recent_season + 1
    current_week = 1
else:
    current_season = most_recent_season
    current_week = most_recent_week + 1


## pull data ##
data_rows = []
while current_week <= week_end and current_season <= season_end:
    print('          Pulling Week {0}, {1}...'.format(current_week,current_season))
    time.sleep((1.5 + random.random() * 2))
    weekly_url = 'https://www.footballoutsiders.com/premium/dvoa-specific-week?year={0}&week={1}&offense_defense=offense'.format(current_season,current_week)
    weekly_raw = s.get(weekly_url)
    weekly_parsed = BeautifulSoup(weekly_raw.content, "html.parser")
    table_rows = weekly_parsed.find_all('tbody')[0].find_all('tr')
    if len(table_rows) == 1:
        print('               Most recent week reached...')
        print('               Ending pull...')
        current_week = week_end + 1
        current_season = season_end + 1
        continue
    for row in table_rows:
        data = []
        for table_data in row.find_all('td'):
            stringed = str(table_data.text) ##convert from unicode##
            stringed = stringed.replace('\n','') ## replace line breaks with blanks##
            stringed = stringed.strip() ## remove spaces ##
            data.append(stringed)
        data_row = {
            'season' : current_season,
            'week' : current_week,
            'team' : data[0],
            'record' : data[1],
            'total_dvoa' : data[2],
            'total_dvoa_rank' : data[3],
            'weighted_dvoa' : data[4],
            'weighted_dvoa_rank' : data[5],
            'offensive_dvoa' : data[6],
            'offensive_dvoa_rank' : data[7],
            'weighted_offensive_dvoa' : data[8],
            'weighted_offensive_dvoa_rank' : data[9],
            'defensive_dvoa' : data[10],
            'defensive_dvoa_rank' : data[11],
            'weighted_defensive_dvoa' : data[12],
            'weighted_defensive_dvoa_rank' : data[13],
            'st_dvoa' : data[14],
            'st_dvoa_rank' : data[15],
            'weighted_st_dvoa' : data[16],
            'weighted_st_dvoa_rank' : data[17],
        }
        data_rows.append(data_row)
    if current_week == week_end:
        current_season += 1
        current_week = 1
    else:
        current_week += 1


update_df = pd.DataFrame(data_rows)

## format ##
float_fields = [
    'total_dvoa',
    'weighted_dvoa',
    'offensive_dvoa',
    'weighted_offensive_dvoa',
    'defensive_dvoa',
    'weighted_defensive_dvoa',
    'st_dvoa',
    'weighted_st_dvoa'
]

int_fields = [
    'total_dvoa_rank',
    'weighted_dvoa_rank',
    'offensive_dvoa_rank',
    'weighted_offensive_dvoa_rank',
    'defensive_dvoa_rank',
    'weighted_defensive_dvoa_rank',
    'st_dvoa_rank',
    'weighted_st_dvoa_rank'
]

for field in float_fields:
    update_df[field] = update_df[field].str.replace('%', '0').astype('float') / 100.0


for field in int_fields:
    try:
        update_df[field] = update_df[field].astype('int')
    except:
        pass


dvoa_to_standard_dict = {
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
    'JAX' : 'JAX',
    'JAC' : 'JAX',
    'KC'  : 'KC',
    'LAC' : 'LAC',
    'LAR' : 'LAR',
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

update_df['team'] = update_df['team'].replace(dvoa_to_standard_dict)

headers = [
    'season',
    'week',
    'team',
    'record',
    'total_dvoa',
    'total_dvoa_rank',
    'weighted_dvoa',
    'weighted_dvoa_rank',
    'offensive_dvoa',
    'offensive_dvoa_rank',
    'weighted_offensive_dvoa',
    'weighted_offensive_dvoa_rank',
    'defensive_dvoa',
    'defensive_dvoa_rank',
    'weighted_defensive_dvoa',
    'weighted_defensive_dvoa_rank',
    'st_dvoa',
    'st_dvoa_rank',
    'weighted_st_dvoa',
    'weighted_st_dvoa_rank'
]

update_df = update_df[headers]

## merge and export ##
if most_recent_df is None:
    output_df = update_df
else:
    most_recent_df = most_recent_df[headers]
    output_df = pd.concat([most_recent_df,update_df])

output_df = output_df.sort_values(by=['season','week']).reset_index(drop=True)
output_df = output_df[headers]


through_season = output_df['season'].max()
through_week = output_df[output_df['season'] == through_season]['week'].max()

output_df.to_csv('{0}/weekly_dvoa_data.csv'.format(data_folder))
