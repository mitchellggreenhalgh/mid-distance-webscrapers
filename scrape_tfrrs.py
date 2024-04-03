'''A module to create a TFRRSScraper class to scrape 800m, 400m, and 1500m/Mile data from TFRRS.org'''

import pandas as pd
from glob import glob
from datetime import datetime

class TFRRSScraper():
    def __init__(self, other_event: int, sex: str | None = None):
        '''Initialize a TFRRSScraper object
        
        Parameters:
          -  other_event (int): Dictates which other event to download. Options: 400 or 1500
          -  sex (str | None): 'f', 'm', or None. If 'f' or 'm', download only that sex's data
        '''
        self.website = 'https://tfrrs.org/'
        self.other_event = other_event
        self.sex = sex


    def __repr__(self):
        return f'Scraper object for [{self.website}]'


    def extract_event_data(self, url_root_800: str, url_root_other: str, sex: str) -> pd.DataFrame:
        '''Extract data from tfrrs.org.
        
        Parameters:
          -  url_root_800 (str): URL to a TFRRS 800m table
          -  url_root_other (str): URL to a TFRRS 400m or 1500m/Mile table
          -  sex (str): 'f' for women and 'm' for men

        Returns:
          -  df (pd.DataFrame): DataFrame containing athletes who ran the 400 and 800 during the 2024 indoor season and had both times in the top 500 of Division I performances
        '''

        df_800 = pd.read_html(f'{url_root_800[:-1]}{sex}',
                            attrs={'id': 'myTable'})[0][['Athlete', 'Team', 'Time']]

        df_other = pd.read_html(f'{url_root_other[:-1]}{sex}',
                            attrs={'id': 'myTable'})[0][['Athlete', 'Team', 'Time']]

        # Merge DataFrames
        df = df_800.merge(df_other, how = 'left', left_on=['Athlete', 'Team'], right_on=['Athlete', 'Team']) \
            .dropna()

        # Translate 800m time to seconds, remove '@' and '#' tags
        df['Time_x'] = 60 * df['Time_x'].str[0].astype('float') + df['Time_x'].str[2:7].astype('float')

        # Translate 400m time to seconds if necessary, remove '@' and '#' tags in 400m
        if self.other_event == 400:
          if df['Time_y'].dtype == 'O':
              df['Time_y'] = df['Time_y'].apply(self.clean_400m_times)
        # Translate 1500m/mile time to seconds if necessary, remove '@' and '#' tags in 400m
        elif self.other_event == 1500:
          df['Time_y'] = 60 * df['Time_y'].str[0].astype('float') + df['Time_y'].str[2:7].astype('float')

        # Make columns more intuitive
        df.columns = ['athlete', 'team', 'time_800', f'time_{self.other_event}']

        df['athlete_team'] = df['athlete'] + ' ' + df['team']
        df = df.drop(columns = ['athlete', 'team'])

        return df[['athlete_team', 'time_800', f'time_{self.other_event}']]


    def clean_400m_times(self, row: str | float) -> float:
        '''Take each row of a pd.Series and modify it according to its format
        
        Parameters:
          -  row (str | float): the value of the row in the pd.Series
        '''

        if str(row)[0] in ['4', '5']:
            return float(str(row)[:5])
        else:
            return 60 * int(str(row)[0]) + float(str(row)[2:7])


    def merge_data(self, url_root_800: str, url_root_other: str) -> pd.DataFrame:
        '''Merges TFRRS data tables from the different sex categories together
        
        Parameters:
          -  url_root_800 (str): URL to a TFRRS 800m table
          -  url_root_other (str): URL to a TFRRS 400m or 1500m/Mile table
            
        Returns:
          -  df_all (pd.DataFrame): a pd.DataFrame that merges the 400m or 1500m/800 data for both sexes into one table.
        '''
        
        df_women = self.extract_event_data(url_root_other=url_root_other, url_root_800=url_root_800, sex='f').assign(sex='f')
        df_men = self.extract_event_data(url_root_other=url_root_other, url_root_800=url_root_800, sex='m').assign(sex='m')

        df_all = pd.concat([df_women, df_men])

        return df_all


    def download_urls(self, url_root: str, season: str) -> pd.DataFrame:
        '''Merges TFRRS 400m/1500m/mile and 800m data tables from the different sex categories together. If the event is the mile, convert the time to a 1500 time.
        
        Parameters:
          -  url_root_800 (str): URL to a TFRRS 800m table
          -  season (str): 'indoor' or 'outdoor'. Depending on the season, the values for the events change. In indoor: 53 = 400m, 54 = 800m, 57 = mile. In outdoor: 11 = 400m, 12 = 800m, 13 = 1500m.
            
            
        Returns:
          -  df (pd.DataFrame): a pd.DataFrame that merges the (400m or 1500m) and 800m data for both sexes into one table
        '''
        if self.other_event == 400:
          if season == 'indoor':
              df = self.merge_data(url_root_800 = url_root,
                                  url_root_other = url_root.replace('event_type=54', 'event_type=53'))
          elif season == 'outdoor':
              df = self.merge_data(url_root_800 = url_root,
                                  url_root_other = url_root.replace('event_type=12', 'event_type=11'))
        elif self.other_event == 1500:
          if season == 'indoor':
              df = self.merge_data(url_root_800 = url_root,
                                  url_root_other = url_root.replace('event_type=54', 'event_type=57'))
              # Convert mile to 1500
              df['time_1500'] = (df['time_1500'] * 0.93205678835).round(decimals=2)

          elif season == 'outdoor':
              df = self.merge_data(url_root_800 = url_root,
                                  url_root_other = url_root.replace('event_type=12', 'event_type=13'))

        return df


    def download_seasons(self, seasons: dict, division: str) -> pd.DataFrame:
        '''Takes a dictionary of seasons and urls for a division and combines them all into a single pd.DataFrame
        
        Parameters:
          -  seasons (dict): Keys follow the pattern [season]_[year], and the values are the URL to the Top 500 800m runners table for that season-year-division combination on TFFRS.
          -  division (str): with NCAA Division is being downloaded
            
        Returns:
          -  dfs (pd.DataFrame): a pd.DataFrame of multiple seasons of women and men's 400m and 800m times.
        '''
        
        dfs = None

        for season in seasons:
            df = self.download_urls(url_root=seasons[season],
                            season=season.split('_')[0])
            
            df['season'] = season
            
            if dfs is None:
                dfs = df
                continue

            dfs = pd.concat([dfs, df])

        # Export
        dfs.to_csv(f'data/tfrrs_{division}_{list(seasons.keys())[0]}-{list(seasons.keys())[-1]}_{self.other_event}_{datetime.now():%Y-%m-%d}.csv', index=False)

        return dfs


    def merge_tfrrs_data(self) -> pd.DataFrame:
        '''Grabs all the TFRRS .csv files in the data directory and concatenates them together
        
        Returns:
          -  dfs (pd.DataFrame): a single pd.DataFrame of all the TFRRS data in the data directory
        '''
        data_list = glob(f'data/tfrrs*{self.other_event}*.csv')

        dfs = None

        for csv in data_list:
            df = pd.read_csv(csv)

            if dfs is None:
                dfs = df
                continue

            dfs = pd.concat([dfs, df])

        return dfs.drop_duplicates()
