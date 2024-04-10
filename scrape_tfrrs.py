'''A module to create a TFRRSScraper class to scrape track event data from TFRRS.org'''

import re
from datetime import datetime
from glob import glob
from typing import Dict, List
from os import makedirs

import pandas as pd

__valid_events__ = [
  '100', 
  '110H', 
  '200', 
  '400',
  '400H', 
  '800', 
  '1500', 
  '3000', 
  '3000SC', 
  '5000', 
  '10000'
]


class TFRRSScraper():
    def __init__(self, 
                 seasons_dict: Dict[str, str],
                 *,
                 division: str,
                 outcome_event: str, 
                 predictor_events: str | List[str] | None = None, 
                 sex: str | None = None,
                 merge: bool = False):
        '''Initializes a TFRRSScraper object.
        
        Parameters:
          -  seasons_dict (`Dict[str, str]`): A dictionary of links to a TFRRS table of a given season. The dictionary should be in chronological order.
          -  division (`str`): the level of collegiate competition is the dictionary for. Options: 'DI', 'DII', 'DIII', 'NAIA', 'NJCAA'
          -  predictor_event (`int`): The main event of interest. Options listed below.
          -  outcome_events (`int` | `List[str]` | `None`): Dictates which other event to download. Options: any single event or combination of the events listed below exclusive of predictor_event.
          -  sex (`str` | `None`): 'f', 'm', or None. If 'f' or 'm', download only that sex's data

        Event options: '100', '110H', '200', '400', '400H', '800', '1500', '3000', '3000SC', '5000', '10000'

        Regarding indoor events, '100' retrieves 60m data, '110H' retrieves 60mH data, and '1500' retrieves mile data, though the column names still express '100', '100H', and '1500'. '3000' is only offered indoor with no outdoor equivalent.
    
        Regarding outdoor events, '110H' retrieves 100mH data for girls. '400H', '3000SC', and '10000' are only performed during outdoors and have no indoor equivalent.
        '''
        self.website = 'https://tfrrs.org/'

        self.seasons_dict = seasons_dict
        self.division = division.lower()
        self.outcome_event = outcome_event.upper()
        self.sex = sex
        self.merge = merge

        if predictor_events is None:
            self.predictor_events = None
        else:
            self.predictor_events = [predictor_events.upper()] if isinstance(predictor_events, str) else list(map(str.upper, predictor_events))

        self.event_dict = {
            'indoor_60': 'event_type=46',
            'indoor_60H': 'event_type=50',
            'indoor_200': 'event_type=51',
            'indoor_400': 'event_type=53',
            'indoor_800': 'event_type=54',
            'indoor_mile': 'event_type=57',
            'indoor_3000': 'event_type=60',
            'indoor_5000': 'event_type=62',
            'outdoor_100': 'event_type=6',    
            'outdoor_100H': 'event_type=4',   
            'outdoor_110H': 'event_type=5',   
            'outdoor_200': 'event_type=7',
            'outdoor_400': 'event_type=11',
            'outdoor_400H': 'event_type=9',    
            'outdoor_800': 'event_type=12',
            'outdoor_1500': 'event_type=13',   
            'outdoor_3000SC': 'event_type=19', 
            'outdoor_5000': 'event_type=21',
            'outdoor_10000': 'event_type=22'   
        }

        #TODO: #15 change data2 to data for full production
        makedirs('data2', exist_ok=True)

        if isinstance(self.outcome_event, list):
            raise ValueError('The outcome event must be a single event of type string')
        
        if self.outcome_event not in __valid_events__:
            raise ValueError(f'{self.outcome_event} is not a valid event. Please enter a valid track event: {__valid_events__}')
        
        if self.predictor_events is not None:
          for event in self.predictor_events:
              if event not in __valid_events__:
                  raise ValueError(f'{event} is not a valid event. Please enter a valid track event: {__valid_events__}')
        
        if self.division not in ['di', 'dii', 'diii', 'naia', 'njcaa']:
            raise ValueError('Please enter an appropriate collegiate division')
        
        if self.sex is not None and self.sex not in ['f', 'm']:
            raise ValueError('''Please enter an appropriate sex: 'f' or 'm' ''')


    def __repr__(self):
        return f'Scraper object for [{self.website}]'
    

    def __call__(self):
        self.download_and_export_data()
        return NotImplementedError


    def produce_outcome_url_by_season_key(self,
                                      season_idx: str, 
                                      sex: str,
                                      outcome_event: str | None = None) -> str:
        '''Produce a list of URLs for TFRRS tables of the outcome event'''
        
        base_url = self.seasons_dict[season_idx]
        season = season_idx.split('_')[0]

        if outcome_event is None:
            outcome_event = self.outcome_event
        
        # Logic for special event cases
        match season:
            case 'indoor':
                match outcome_event:
                    case '100':
                        outcome_event = '60'
                    case '110H':
                        outcome_event = '60H'
                    case '1500':
                        outcome_event = 'mile'

            case 'outdoor':
                if outcome_event == '110H' and sex == 'f':
                    outcome_event = '100H'
                    
        key = f'{season}_{outcome_event}'  

        if outcome_event in ['400H', '3000SC', '10000'] and season == 'indoor':
            return
        
        if outcome_event in ['3000'] and season == 'outdoor':
            return

        url = f'''{re.sub(pattern=r'event_type=[0-9]{2}', 
                          repl=self.event_dict[key], 
                          string=base_url)[:-1]}{sex}'''              

        return url
    

    def produce_outcome_urls_by_season_dict(self, 
                                            sex: str | None = None,
                                            seasons_dict: Dict[str, str] | None = None,
                                            outcome_event: List[str] | None = None) -> List[str]:
        '''Takes the season_dict and produces all the URLs for all the predictor events'''

        if seasons_dict is None:
            seasons_dict = self.seasons_dict

        if outcome_event is None:
            outcome_event = self.outcome_event

        if sex is None:
            sex = self.sex

        if sex is None:
          urls_for_seasons_dict = {}
          for key in seasons_dict:
              urls_for_seasons_dict = urls_for_seasons_dict | {f'{key}_m_{self.outcome_event}': self.produce_outcome_url_by_season_key(season_idx=key, 
                                                                            sex='m', 
                                                                            outcome_event=outcome_event)}
              
              urls_for_seasons_dict = urls_for_seasons_dict | {f'{key}_f_{self.outcome_event}': self.produce_outcome_url_by_season_key(season_idx=key, 
                                                                            sex='f', 
                                                                            outcome_event=outcome_event)}
          return urls_for_seasons_dict
        
        else:
            urls_for_seasons_dict = {}
            for key in seasons_dict:
                urls_for_seasons_dict = urls_for_seasons_dict | {f'{key}_{sex}_{self.outcome_event}': self.produce_outcome_url_by_season_key(season_idx=key, 
                                                                              sex=sex, 
                                                                              outcome_event=outcome_event)}
            return urls_for_seasons_dict

    
    def produce_predictor_urls_by_season_key(self,
                                         season_idx: str,
                                         sex: str,
                                         event_list: List[str] | None = None) -> List[str]:
        '''Produce a list of URLs for TFRRS tables of the predictor event(s) from a single input URL from the seasons_dict'''

        base_url = self.seasons_dict[season_idx]
        season = season_idx.split('_')[0]

        if event_list is None:
            event_list = self.predictor_events

        url_dict = {}
        for event in event_list:
            # Logic for special event cases
            match season:
                case 'indoor':
                    match event:
                        case '100':
                            event = '60'
                        case '110H':
                            event = '60H'
                        case '1500':
                            event = 'mile'

                case 'outdoor':
                    if event == '110H' and sex == 'f':
                        event = '100H'
                        
            key = f'{season}_{event}'  

            if event in ['400H', '3000SC', '10000'] and season == 'indoor':
                continue

            url_dict = url_dict | {f'{season_idx}_{sex}_{event}': 
                                                f'''{re.sub(pattern=r'event_type=[0-9]{2}', 
                                                    repl=self.event_dict[key], 
                                                    string=base_url)[:-1]}{sex}'''}                

        return url_dict
    

    def produce_predictor_urls_by_season_dict(self, 
                                              sex: str | None = None,
                                              seasons_dict: Dict[str, str] | None = None,
                                              event_list: List[str] | None = None) -> List[str]:
        '''Takes the season_dict and produces all the URLs for all the predictor events'''

        if seasons_dict is None:
            seasons_dict = self.seasons_dict

        if event_list is None:
            event_list = self.predictor_events

        if sex is None:
            sex = self.sex

        if sex is None:
            urls_for_seasons_dict = {}
            for key in seasons_dict:
                urls_for_seasons_dict = urls_for_seasons_dict | self.produce_predictor_urls_by_season_key(season_idx=key, 
                                                                              sex='m', 
                                                                              event_list=event_list)
                
                urls_for_seasons_dict = urls_for_seasons_dict | self.produce_predictor_urls_by_season_key(season_idx=key, 
                                                                              sex='f', 
                                                                              event_list=event_list)
                
            return urls_for_seasons_dict
        
        else:
            urls_for_seasons_dict = {}
            for key in seasons_dict:
                urls_for_seasons_dict = urls_for_seasons_dict | self.produce_predictor_urls_by_season_key(season_idx=key, 
                                                                              sex=sex,
                                                                              event_list=event_list)
                
            return urls_for_seasons_dict


    def convert_event(self, df: pd.DataFrame, key: str) -> pd.DataFrame:
        '''Take a pd.DataFrame of a TFRRS table and convert the time column into seconds
        
        Parameters:
          -  df (`pd.DataFrame`): the output of pd.read_html() directly from a TFRRS table URL
          -  key (`str`): the key of the url dictionary that produced the URL to the TFRRS table
          
        Returns:
          -  df (`pd.DataFrame`): the same dataframe, but with the time column converted to seconds from `%m:%s.%f`
        '''
        
        event = key.split('_')[3]

        match event:
            case '100' | '110H' | '200':
                df[f'time_{event}'] = df[f'time_{event}'].astype('float')

            case '400' | '400H':
                df[f'time_{event}'] = df[f'time_{event}'].apply(self.clean_400m_times)

            case _:
                df[f'time_{event}'] = df[f'time_{event}'].apply(self.clean_distance_times) 

        return df


    def download_single_outcome_table(self, url: str) -> pd.DataFrame:
        '''Extract data from a given tfrrs.org URL.
        
        Parameters:
          -  url (`str`): the URL to a TFRRS table
        '''

        df = pd.read_html(url, attrs={'id': 'myTable'})[0][['Athlete', 'Team', 'Time']]

        df[[f'time_{self.outcome_event}', f'{self.outcome_event}_tag']] = df['Time'].str.split(' ', n=1, expand=True)
        df['athlete_team'] = df['Athlete'] + ' | ' + df['Team']

        df_format = ['athlete_team', f'time_{self.outcome_event}', f'{self.outcome_event}_tag']
        
        return df.drop(columns=['Athlete', 'Team', 'Time'])[df_format]
    

    def download_single_predictor_table(self, url: str, key: str) -> pd.DataFrame:
        '''Docstring'''

        df = pd.read_html(url, attrs={'id': 'myTable'})[0][['Athlete', 'Team', 'Time']]

        

        return df.to_dict()

    

    def download_outcome_event(self, export: bool = False) -> pd.DataFrame | None:
        '''Combine all TFRRS tables of outcome variable.
        
        Parameters:
          -  export (`bool`): if True, the outcome event's table is exported to the data folder and no pd.DataFrame is returned. If False, the data table is returned as a pd.DataFrame and not exported.'''

        url_dict = self.produce_outcome_urls_by_season_dict()
        dfs = None

        for i in url_dict:
            if url_dict[i] is None:
                continue

            key_vars = i.split('_')

            df = self.download_single_outcome_table(url_dict[i]) \
                .assign(sex=key_vars[2],
                        season=f'{key_vars[0]}_{key_vars[1]}') \
                .pipe(self.convert_event, key=i)
                        
            if dfs is None:
                dfs = df
                continue

            dfs = pd.concat([dfs, df])

        if export:
            season_range = dfs['season'].unique()[[0, -1]]
            export_path = f'data2/tfrrs_{self.division}_{self.outcome_event}_{season_range[0]}-{season_range[1]}_{datetime.now():%Y-%m-%d}.csv'
            dfs.to_csv(export_path, index=False)
            return
        
        return dfs.reset_index(drop=True)
    

    def download_and_export_data(self) -> None:

        if not self.merge:
            if self.predictor_events is None:
                self.download_outcome_event(export=True)
                
            self.download_outcome_event()
            # self.download_predictor_events()


    def clean_400m_times(self, row: str | float) -> float:
        '''Take each row of a pd.Series and modify it according to its format
        
        Parameters:
          -  row (str | float): the value of the row in the pd.Series
        '''

        if str(row)[0] in ['4', '5']:
            return float(str(row))
        else:
            time_split = str(row).split(':')
            return 60 * float(time_split[0]) + float(time_split[1])
        
    
    def clean_distance_times(self, row: str | float) -> float:
        '''Take each row of a pd.Series and modify it according to its format
        
        Parameters:
          -  row (str | float): the value of the row in the pd.Series
        '''

        time_split = str(row).split(':')
        return 60 * float(time_split[0]) + float(time_split[1])


    # def download_and_merge_sex_data(self, url: str) -> pd.DataFrame:
    #     '''Merges TFRRS data tables from the different sex categories together
        
    #     Parameters:
    #       -  url (str): URL to a TFRRS table
            
    #     Returns:
    #       -  df_all (pd.DataFrame): a pd.DataFrame that merges the 400m or 1500m/800 data for both sexes into one table.
    #     '''
        
    #     # df_women = self.download_and_clean_single_event(url=, sex='f').assign(sex='f')
    #     # df_men = self.download_and_clean_single_event(url=, sex='m').assign(sex='m')

    #     df_all = pd.concat([df_women, df_men])

    #     return df_all


    # def download_urls(self, url_root: str, season: str, event: str) -> pd.DataFrame:
    #     '''Merges TFRRS track event tables from the different sex categories together. If the event is the mile, convert the time to a 1500 time.
        
    #     Parameters:
    #       -  url_root_800 (str): URL to a TFRRS 800m table
    #       -  season (str): 'indoor' or 'outdoor'. Depending on the season, the values for the events change. In indoor: 53 = 400m, 54 = 800m, 57 = mile. In outdoor: 11 = 400m, 12 = 800m, 13 = 1500m.
    #       -  event (str): the event of interest
            
            
    #     Returns:
    #       -  df (pd.DataFrame): a pd.DataFrame that merges the (400m or 1500m) and 800m data for both sexes into one table
    #     '''
        
    #     event_key = f'{season}_{event}'

        


    #     match season:
    #         case 'indoor':
    #             df = self.download_and_merge_sex_data()


    #     if self.predictor_events == '400':
    #       if season == 'indoor':
    #           df = self.download_and_merge_sex_data(url_root_800 = url_root,
    #                               url_root_other = url_root.replace('event_type=54', 'event_type=53'))
    #       elif season == 'outdoor':
    #           df = self.download_and_merge_sex_data(url_root_800 = url_root,
    #                               url_root_other = url_root.replace('event_type=12', 'event_type=11'))
    #     elif self.predictor_events == '1500':
    #       if season == 'indoor':
    #           df = self.download_and_merge_sex_data(url_root_800 = url_root,
    #                               url_root_other = url_root.replace('event_type=54', 'event_type=57'))
    #           # Convert mile to 1500
    #           df['time_1500'] = (df['time_1500'] * 0.9321).round(decimals=2)

    #       elif season == 'outdoor':
    #           df = self.download_and_merge_sex_data(url_root_800 = url_root,
    #                               url_root_other = url_root.replace('event_type=12', 'event_type=13'))

    #     return df


    # def download_seasons(self, seasons: dict, division: str) -> pd.DataFrame:
    #     '''Takes a dictionary of seasons and urls for a division and combines them all into a single pd.DataFrame
        
    #     Parameters:
    #       -  seasons (dict): Keys follow the pattern [season]_[year], and the values are the URL to the Top 500 800m runners table for that season-year-division combination on TFFRS.
    #       -  division (str): with NCAA Division is being downloaded
            
    #     Returns:
    #       -  dfs (pd.DataFrame): a pd.DataFrame of multiple seasons of women and men's 400m and 800m times.
    #     '''
        
    #     dfs = None

    #     for season in seasons:
    #         df = self.download_urls(url_root=seasons[season],
    #                                 season=season.split('_')[0])
            
    #         df['season'] = season
            
    #         if dfs is None:
    #             dfs = df
    #             continue

    #         dfs = pd.concat([dfs, df])

    #     # Export
    #     dfs.to_csv(f'data/tfrrs_{division}_{list(seasons.keys())[0]}-{list(seasons.keys())[-1]}_{self.predictor_events}_{datetime.now():%Y-%m-%d}.csv', index=False)

    #     return dfs


    # def merge_tfrrs_data(self, event: str) -> pd.DataFrame:
    #     '''Grabs all the TFRRS .csv files in the data directory and concatenates them together

    #     Parameters:
    #       -  event (`str`): the track event of interest
        
    #     Returns:
    #       -  dfs (pd.DataFrame): a single pd.DataFrame of all the TFRRS data in the data directory
    #     '''
    #     data_list = glob(f'data/tfrrs*{event}*.csv')

    #     dfs = None

    #     for csv in data_list:
    #         df = pd.read_csv(csv)

    #         if dfs is None:
    #             dfs = df
    #             continue

    #         dfs = pd.concat([dfs, df])

    #     return dfs.drop_duplicates().reset_index(drop=True)
