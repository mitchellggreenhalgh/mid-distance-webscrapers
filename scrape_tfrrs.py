'''A module to create a TFRRSScraper class to scrape track event data from TFRRS.org'''

import re
from datetime import datetime
from glob import glob
from typing import Dict, List, Tuple, Any, Type
from os import makedirs
from deprecated import deprecated

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

PredictorResultsDict = Type[Dict[str, Dict[str, Dict[int, Any]]]]


class TFRRSScraper():
    def __init__(self, 
                 seasons_dict: Dict[str, str],
                 *,
                 division: str,
                 outcome_event: str, 
                 predictor_events: str | List[str] | None = None, 
                 sex: str | None = None,
                 merge: bool = False) -> None:
        '''Initializes a TFRRSScraper object.
        
        Parameters:
          -  seasons_dict (`Dict[str, str]`): A dictionary of links to TFRRS tables of given seasons.
          -  division (`str`): the level of collegiate competition is the dictionary for. Options: 'DI', 'DII', 'DIII', 'NAIA', 'NJCAA'.
          -  predictor_event (`int`): The main event of interest. Options listed below.
          -  outcome_events (`int` | `List[str]` | `None`): Dictates which other event to download. Options: any single event or combination of the events listed below exclusive of predictor_event.
          -  sex (`str` | `None`): 'f', 'm', or None. If 'f' or 'm', download only that sex's data.
          -  merge (`bool`): If `True`, the outcome and predictor event(s) are merged when exported.

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

        makedirs('data', exist_ok=True)

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

        if self.predictor_events is None and self.merge:
            raise ValueError('Merge can only be `True` if predictor event(s) are specified')


    def __str__(self) -> str:
        return f"TFRRSScraper object for {self.division.upper()} events {', '.join([self.outcome_event] + self.predictor_events)} on [{self.website}]"
    

    def __repr__(self) -> str:
        return f"TFRRSScraper(division={self.division}, outcome_event={self.outcome_event}, predictor_events=({', '.join(self.predictor_events)}), sex={self.sex if self.sex is not None else '(m, f)'})"


    def __call__(self) -> None:
        self.download_and_export_data()


    def produce_outcome_url_by_season_key(self,
                                      season_idx: str, 
                                      sex: str,
                                      outcome_event: str | None = None) -> str:
        '''Uses a key to the season dictionary to produce a URL for a TFRRS table of the outcome event for a certain sex
        
        Parameters:
          -  season_idx (`str`): A key to the season dictionary
          -  sex (`str`): The sex to make the URL for
          -  outcome_event (`str` | `None`): The event to make the URL for. The default is the outcome event specified in the initializer.
          
        Returns:
          -  url (`str`): the URL to a TFRRS table for a certain event, sex, and season
        '''
        
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
                                            outcome_event: str | None = None) -> Dict[str, str]:
        '''Uses the season dictionary and modifies the provided links to create URLs for outcome event across all specified seasons
        
        Parameters:
          -  sex (`str` | `None`): The sex to create URLs for. If `None`, URLs are created for both sexes.
          -  seasons_dict (`Dict` | `None`): A dictionary whose keys are the 'season_year' specifier for the values, which are valid links to a TFRRS table for that season. The default is the dictionary provided in the initializer is used.
          -  outcome_event (`str`): The outcome event of interest. The default is the outcome event specified in the initializer.
          
        Returns:
          -  urls_for_season_dict (`Dict`): A dictionary whose keys follow the pattern 'season_year_sex_event', and whose values are the URLs to the TFRRS table matching the variables in the key.
        '''

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
                                         event_list: List[str] | None = None) -> Dict[str, str]:
        '''Produce a dictionary of URLs matching TFRRS tables for the predictor event(s) for a single season and sex, as specified by a key to the seasons dictionary.
        
        Parameters:
          -  season_idx (`str`): A key to the season dictionary
          -  sex (`str`): The sex to make the URL for
          -  event_list (`List` | `None`): The event(s) to make the URL for. The default is the predictor event(s) specified in the initializer.
          
        Returns:
          -  url_dict (`Dict`):  A dictionary whose keys follow the pattern 'season_year_sex_event' for a given season, sex, and all predictor event(s), and whose values are the URLs to the TFRRS table matching the variables in the key. 
        '''

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

            if event in ['3000'] and season == 'outdoor':
                continue

            url_dict = url_dict | {f'{season_idx}_{sex}_{event}': 
                                                f'''{re.sub(pattern=r'event_type=[0-9]{2}', 
                                                    repl=self.event_dict[key], 
                                                    string=base_url)[:-1]}{sex}'''}                

        return url_dict
    

    def produce_predictor_urls_by_season_dict(self, 
                                              sex: str | None = None,
                                              seasons_dict: Dict[str, str] | None = None,
                                              event_list: List[str] | None = None) -> Dict[str, str]:
        '''Produces a dictionary of URLs matching TFRRS tables for the predictor event(s) for all seasons, sexes, and predictor event(s), as specified by a key to the URL dictionary produced by `produce_predictor_urls_by_season_key()`.
        
        Parameters:
          -  sex (`str`): The sex to make the URL for. The default is both sexes.
          -  seasons_dict (`Dict` | `None`): The seasons dictionary of 'season_year' keys and base URLs to make the new URLs from. The default is the seasons dictionary specified in the initializer.
          -  event_list (`List` | `None`): The event(s) to make the URL for. The default is the predictor event(s) specified in the initializer.
          
        Returns:
          -  urls_for_seasons_dict (`Dict`):  A dictionary whose keys follow the pattern 'season_year_sex_event' for a given season, sex, and each predictor event, and whose values are the URLs to the TFRRS table matching the variables in the key. Differs from `produce_predictor_urls_by_season_key()` in that it produces key value pairs for all seasons specified in the seasons dictionary, where as the other only provides the URLs for a single season. 
        '''

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


    def download_single_outcome_table(self, url: str) -> pd.DataFrame:
        '''Download a DataFrame of table data from a given URL.
        
        Parameters:
          -  url (`str`): the URL to a TFRRS table

        Returns:
          -  df (`pd.DataFrame`): A DataFrame that contains athlete/team information, the time of the given event in seconds, and any tags the data has regarding altitude or conversions.
        '''

        df = pd.read_html(url, attrs={'id': 'myTable'})[0][['Athlete', 'Team', 'Time']]
        df['athlete_team'] = df['Athlete'] + ' | ' + df['Team']

        # Try splitting tags, if there aren't any then manually assign a tag column with None values
        try:
            df[[f'time_{self.outcome_event}', f'{self.outcome_event}_tag']] = df['Time'].str.split(' ', n=1, expand=True)
        except ValueError:
            df[f'time_{self.outcome_event}'] = df['Time'].str.split(' ', n=1, expand=True)
            df = df.assign(tag=None)
            df[f'{self.outcome_event}_tag'] = df['tag']

        df_format = ['athlete_team', f'time_{self.outcome_event}', f'{self.outcome_event}_tag']
        
        return df[df_format]
    

    def download_single_predictor_table(self, 
                                        url: str, 
                                        key: str) -> PredictorResultsDict:
        '''Download a single TFRRS table using a URL from the predictors URL dictionary.
        
        Parameters:
          -  url (`str`): A URL from the predictor URL dictionary
          -  key (`str`): A key from the predictor URL dictionary that produced the given URL
          
        Returns:
          -  data_dict (`Dict`): A dictionary of pd.DataFrames converted to dictionaries for a single predictor event, season, and sex. The first key is the 'season_year_sex_event' identifier for the data. The value of this key is another dictionary whose keys are the column names of the downloaded TFRRS table. The values of the column name keys are a final set of dictionaries, whose keys are the row indices of the downloaded table and whose values are the corresponding data values from the TFRRS table.
          
        Running the function `pd.DataFrame.from_dict()` on `data_dict[key]` will return the downloaded `pd.DataFrame`
        '''

        df = pd.read_html(url, attrs={'id': 'myTable'})[0][['Athlete', 'Team', 'Time']]

        df['athlete_team'] = df['Athlete'] + ' | ' + df['Team']

        # Need to match url specific events to predictor event(s)
        key_vars = key.split('_')
        event = key_vars[3]
        match event:
            case '60':
                time_col = 'time_100'
            case '60H':
                time_col = 'time_110H'
            case '100H':
                time_col = 'time_110H'
            case 'mile':
                time_col = 'time_1500'
            case _:
                time_col = f'time_{event}'

        # Try splitting tags, if there aren't any then manually assign a tag column with None values
        try:
            df[[time_col, f"{time_col.split('_')[1]}_tag"]] = df['Time'].str.split(' ', n=1, expand=True)
        except ValueError:
            df[time_col] = df['Time'].str.split(' ', n=1, expand=True)
            df = df.assign(tag=None)
            df[f"{time_col.split('_')[1]}_tag"] = df['tag']

        df = df.pipe(self.convert_event, predictor_col=time_col) \
               .assign(sex=key_vars[2],
                       season=f'{key_vars[0]}_{key_vars[1]}')
        
        df_format = ['athlete_team', time_col, f"{time_col.split('_')[1]}_tag", 'sex', 'season']

        return {f"{'_'.join(key_vars[:3])}_{time_col.split('_')[1]}": df[df_format].to_dict()}
            

    def download_outcome_event(self, export: bool = False) -> pd.DataFrame | None:
        '''Download or export a DataFrame of all TFRRS tables of the outcome variable across all the seasons specified in the seasons dictionary.
        
        Parameters:
          -  export (`bool`): if True, the outcome event's table is exported to the data folder and no pd.DataFrame is returned. If False, the data table is returned as a pd.DataFrame and not exported.
          
        Returns:
          - dfs (`pd.DataFrame` | `None`): if `export=True`, `None` is returned. Otherwise, a `pd.DataFrame` of all the outcome event data from the seasons dictionary is returned containing athlete/team data, season data, sex data, time for the outcome event in seconds, and tags for the times regarding altitude and conversions. By default, dfs is returned and no table is exported.
        '''

        url_dict = self.produce_outcome_urls_by_season_dict()
        dfs = None

        for i in url_dict:
            if url_dict[i] is None:
                continue

            key_vars = i.split('_')

            df = self.download_single_outcome_table(url_dict[i]) \
                .assign(sex=key_vars[2],
                        season=f'{key_vars[0]}_{key_vars[1]}') \
                .pipe(self.convert_event, outcome_key=i)
                        
            if dfs is None:
                dfs = df
                continue

            dfs = pd.concat([dfs, df])

        if export:
            season_range = dfs['season'].unique()[[0, -1]]
            export_path = f'data/tfrrs_{self.division}_{self.outcome_event}_{season_range[0]}-{season_range[1]}_{datetime.now():%Y-%m-%d}.csv'
            dfs.to_csv(export_path, index=False)
            return
        
        return dfs.reset_index(drop=True)
    

    def combine_predictor_event_dicts(self) -> PredictorResultsDict:
        '''Combines dictionaries of TFRRS tables for all predictor event(s).
        
        Returns:
          -  data_dict (`PredictorResultsDict`): A dictionary of dictionaries for all predictor events. The first key is the 'season_year_sex_event' identifier for the data. The value of this key is another dictionary whose keys are the column names of the downloaded TFRRS table. The values of the column name keys are a final set of dictionaries, whose keys are the row indices of the downloaded table and whose values are the corresponding data values from the TFRRS table.
          
        Running the function `pd.DataFrame.from_dict()` on `data_dict[key]` will return the original `pd.DataFrame`
        '''

        url_dict = self.produce_predictor_urls_by_season_dict()
        data_dict = {}

        for i in url_dict:
            if url_dict[i] is None:
                continue

            pred_dict = self.download_single_predictor_table(url=url_dict[i], key=i)

            data_dict = data_dict | pred_dict

        return data_dict
    

    def concatenate_predictor_event(self, data_dict: PredictorResultsDict, event: str) -> pd.DataFrame:
        '''Produces a DataFrame of all the dictionaries in the data dictionary produced by `combine_predictor_event_dicts` that match a specified predictor event.
        
        Parameters:
          -  data_dict (`PredictorResultsDict`): A dictionary of dictionaries for all predictor events. The first key is the 'season_year_sex_event' identifier for the data. The value of this key is another dictionary whose keys are the column names of the downloaded TFRRS table. The values of the column name keys are a final set of dictionaries, whose keys are the row indices of the downloaded table and whose values are the corresponding data values from the TFRRS table.
          -  event (`str`): A specified predictor event to filter the data dictionary for and concatenate all contained original DataFrames.
          
        Returns:
          -  dfs (`pd.DataFrame`): A DataFrame of all of the data for a predictor event that was contained in the data dictionary.
        '''

        data_dict = data_dict
        key_list = list(data_dict.keys())

        event_keys = [key for key in key_list if key.endswith(event)]
        dfs = None
        for key in event_keys:

            df = pd.DataFrame.from_dict(data_dict[key])
            if dfs is None:
                dfs = df
                continue

            dfs = pd.concat([dfs, df])

        return dfs


    def download_predictor_events(self, export: bool = False) -> pd.DataFrame | None:
        '''Download or export a DataFrame of all the predictor events for all the seasons specified in the seasons dictionary.
        
        Parameters:
          -  export (`bool`): If `True`, the DataFrame of all predictor events is exported to the data directory. Otherwise, the DataFrame is returned.
          
        Returns:
          -  dfs (`pd.DataFrame` | `None`): If `export=False`, dfs is returned containing athlete/team data, season data, sex data, time data in seconds for each of the specified predictor events, and altitude or conversion tags for each of the specified predictor events. By default, dfs is returned and no data is exported.
        '''

        data_dict = self.combine_predictor_event_dicts()
        dfs = None
        merge_cols = ['athlete_team', 'season', 'sex']

        for predictor in self.predictor_events:
            df = self.concatenate_predictor_event(data_dict=data_dict, event=predictor)
            
            if len(self.predictor_events) == 1:
                return df

            if dfs is None:
                dfs = df
                continue

            dfs = dfs.merge(right=df,
                            how='outer',
                            left_on=merge_cols,
                            right_on=merge_cols)
            
        if export:
            season_range = self.sort_seasons(df=dfs)
            pred_events = '_'.join(self.predictor_events)
            export_path = f'data/tfrrs_{self.division}_{pred_events}_{season_range}_{datetime.now():%Y-%m-%d}.csv'
            dfs.to_csv(export_path, index=False)
            return
        
        return dfs.reset_index(drop=True)

        
    def download_and_export_data(self, download_only: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame] | None:
        '''Downloads or exports the outcome event DataFrame and the predictor event(s) DataFrame. If `merge` is `True` in the initializer, the outcome and predictor DataFrames will be merged before being exported or returned.
        
        Parameters:
          -  download_only (`bool`): If `True`, no data will be exported to the data directory, and the DataFrames will instead be returned as a Tuple. By default, the data is exported to the data directory as CSVs.
          
        Returns:
          - outcome_data, predictors_data (`Tuple`): The outcome event DataFrame and the predictor event(s) DataFrame. By default, they are exported to the data directory as CSVs. If `merge=True` in the specifier, they are exported or returned as a single DataFrame.'''

        if download_only:
            if self.merge:
                return pd.merge(
                    left=self.download_outcome_event(), 
                    right=self.download_predictor_events(),
                    how='outer',
                    left_on=merge_cols,
                    right_on=merge_cols
                ).reset_index(drop=True)

            return self.download_outcome_event(), self.download_predictor_events()

        if self.merge:
            merge_cols = ['athlete_team', 'season', 'sex']

            outcomes = self.download_outcome_event()
            predictors = self.download_predictor_events()

            df = pd.merge(left=outcomes, right=predictors,
                          how='outer',
                          left_on=merge_cols,
                          right_on=merge_cols) \
                    .reset_index(drop=True)
            
            season_range = self.sort_seasons(df=df)
            events = '_'.join([self.outcome_event] + self.predictor_events)
            export_path =  f'data/tfrrs_{self.division}_{events}_{season_range}_{datetime.now():%Y-%m-%d}.csv'
            df.to_csv(export_path, index=False)
            return

        if self.predictor_events is None:
            self.download_outcome_event(export=True)
            return
            
        self.download_outcome_event(export=True)
        self.download_predictor_events(export=True)
        return


    def sort_seasons(self, df: pd.DataFrame) -> str:
        '''Identifies the unique season identifiers in a DataFrame, sorts them according to year and season, then exports a string of the earliest season and latest season.
        
        Parameters:
          -  df (`pd.DataFrame`): An outcome event DataFrame, a predictor event(s) DataFrame, or a merged outcome and predictor events DataFrame.
        
        Returns:
          - season_range (`str`): A string containing the earliest season and latest season in the DataFrames, and by proxy the seasons dictionary. Returned in the format 'earliest_season-latest_season'. 
        '''
        season_list = list(df['season'].unique())
        season_list.sort(key=lambda x: (x.split('_')[1], x.split('_')[0]))

        return f'{season_list[0]}-{season_list[-1]}'


    def convert_event(self, 
                      df: pd.DataFrame, 
                      outcome_key: str | None = None, 
                      predictor_col: str | None = None) -> pd.DataFrame:
        '''Take a pd.DataFrame of a TFRRS table and convert the time column to seconds. Takes either `outcome_key` or `predictor_col`, but not both.
        
        Parameters:
          -  df (`pd.DataFrame`): The output of `pd.read_html()` on a TFRRS table URL.
          -  outcome_key (`str`): The key of the URL dictionary that produced the URL of the TFRRS table of an outcome event.
          -  predictor_col (`str`): The time column of the DataFrame produced from the URL of the TFRRS table of a predictor event.
          
        Returns:
          -  df (`pd.DataFrame`): The original DataFrame, but with the time column converted to seconds from `%-M:%S.%f`
        '''
        
        if predictor_col is None and outcome_key is None:
            raise ValueError('Either `outcome_key` or `predictor_col` must be specified')
        
        if predictor_col is not None and outcome_key is not None:
            raise ValueError('Only one of `outcome_key` or `predictor_col` can be specified')

        if predictor_col is None:
            event = outcome_key.split('_')[3]
        elif outcome_key is None:
            event = predictor_col.split('_')[1]  

        match event:
            case '100' | '110H' | '200':
                df[f'time_{event}'] = df[f'time_{event}'].astype('float')

            case '400' | '400H':
                df[f'time_{event}'] = df[f'time_{event}'].apply(self.clean_400m_times)

            case _:
                df[f'time_{event}'] = df[f'time_{event}'].apply(self.clean_distance_times) 

        return df


    def clean_400m_times(self, row: str | float) -> float:
        '''Take each row of a pd.Series of 400m/400mH data and convert it to seconds according to its format
        
        Parameters:
          -  row (`str` | `float`): the value of the row in the pd.Series

        Returns:
          - time (`float`): the time value of an event in seconds
        '''

        # If the row is already in '%S.%f' format, convert it to a float. Otherwise, convert `%-M:%S.%f` to a float
        if str(row)[0] in ['4', '5']:
            return float(str(row))
        else:
            time_split = str(row).split(':')
            return 60 * float(time_split[0]) + float(time_split[1])
        
    
    def clean_distance_times(self, row: str) -> float:
        '''Take each row of a pd.Series of a distance event and convert it to seconds according to its format. Target events are the 800m and longer.
        
        Parameters:
          -  row (`str`): the value of the row in the pd.Series

        Returns:
          - time (`float`): the time value of an event in seconds
        '''

        time_split = str(row).split(':')
        return 60 * float(time_split[0]) + float(time_split[1])


@deprecated('''
    Not super functional right now, will figure something out eventually. Since I added the merge
    argument in the initializer, you don't really need this as much. Still useful for combining
    different divisions, but use with caution.
            
    Issue #1: All data files with multiple events need to be specified in the same order as they 
    were saved in the name of the CSV.
            
    E.g. If you ran the TFRRSScraper with outcome = '800', and predictors = ['400', '1500'], the
    CSV would be saved with the name: 'tfrrs_*_800_400_1500_*.csv'. To read that file, the list 
    provided must follow the same order: ['800', '400', '1500']. If ['800', '1500', '400'] is 
    entered, the files won't be read.
''')
def read_tfrrs_data(events: str | List[str], drop_na: bool = False) -> pd.DataFrame:
    '''Concatenates all the TFRRS CSV files in the data directory for given outcome and predictor events. 

    Parameters:
        -  event (`str` | `List`): The event(s) of interest. If a list of events is provided, the events must be specified in the same order as they were specified when they were saved -- [outcome, predictor_1, predictor_2, ...]. See deprecated notes.
        -  drop_na (`bool`): If `True`, all tag columns are removed from the DataFrames, then all rows with missing data are removed. Default is `False`.
    
    Returns:
        -  dfs (`pd.DataFrame`): A single pd.DataFrame of all the TFRRS data in the data directory, with duplicates removed. Contains athlete/team data, season data, sex data, the time of each outcome and predictor event in seconds, and tags regarding altitude and other conversions.
    '''
        
    try:
        event_string = events if isinstance(events, str) else '_'.join(events)
        events = [events] if isinstance(events, str) else events
        data_list = glob(f'data/tfrrs*{event_string}*.csv')
        col_format = ['athlete_team', 'sex', 'season'] + [f'time_{event}' for event in events]

        dfs = None
        for csv in data_list:
            df = pd.read_csv(csv)[col_format]
            
            if dfs is None:
                dfs = df
                continue

            dfs = pd.concat([dfs, df])

        if drop_na:
            
            return dfs \
                        .drop_duplicates() \
                        .dropna()

        return dfs.drop_duplicates()
    
    except AttributeError:
        print(f'''
    [INFO] No TFRRS data matching the specified event(s) found in data directory. 
    1) Make sure the specified events are in the same order as in the CSV names.
    2) Make sure all events match one of the following:
    {__valid_events__}
            ''')
