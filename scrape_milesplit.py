'''A module for a MileSplitScraper class that uses selenium to scrape running data from MileSplit'''

import pandas as pd
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium import webdriver
import time
from dotenv import dotenv_values
from io import StringIO
from datetime import datetime
from typing import List


FIRST_URL = 'https://www.milesplit.com/rankings/events/high-school-girls/indoor-track-and-field/800m?year=2024&accuracy=fat&grade=all&ageGroup=&league=0&meet=0&team=0&venue=0&conversion=n&page=1'

config = dotenv_values('.env')
    

class MileSplitScraper():
    '''A class to download running data from the national MileSplit database.
    

    Event options: '100m', '200m', '400m', '110H', '300H', '400H', '800m', '1500m', '1600m', 'Mile', '3000m', '3200m', '2Mile', '2000mSC'

    Regarding indoor events, '100m' retrieves 60m data, and '110H' retrieves 60mH data, though the column names still express '100' and '100H'.
    
    Regarding outdoor events, '110H' retrieves 100mH data for girls.
    '''

    def __init__(self, 
                 *,
                 outcome_event: str, 
                 predictor_events: str | List[str] | None=None, 
                 url: str=FIRST_URL, 
                 sex: str | None=None) -> None:
        '''Initializes the MileSplitScraper class. If only an outcome event is specified, the entire event is downloaded with no other modifications. If one or more predictor events are specified, the outcome event and the predictor event(s) will be joined together and filtered such that the results will only consist of athletes who have run all of the specified events in a single season and whose top performances for the specified events all appear in the top 1000 in the national rankings.
        
        Parameters:
          -  outcome_event (`str`): The main event of interest to download. See below for full list of options.
          -  predictor_events (`str` | `List[str]`): Dictates which other event to download. Any events from the list below exclusive of the `outcome_event`
          -  url (`str`): a URL to the milesplit rankings portion of the website to log in on and start scraping
          -  sex (`str` | `None`): 'm', 'f', or None. If 'm' or 'f' is indicated, only that sex will be downloaded

        Event options: '100m', '200m', '400m', '110H', '300H', '400H', '800m', '1500m', '1600m', 'Mile', '2000mSC', '3000m', '3200m', '2Mile' 
        '''

        self.outcome_event = outcome_event
        self.predictor_events = predictor_events
        self.url = url
        self.website = 'https://www.milesplit.com/'
        self.USERNAME = config['USERNAME']
        self.PASSWORD = config['PASSWORD']
        self.EXE_PATH = config['EXE_PATH']
        self.sex = sex

        self.OPTIONS = Options()
        self.OPTIONS.add_argument('--ignore-certificate-errors')
        self.OPTIONS.add_experimental_option('excludeSwitches', ['enable-logging'])
        self.OPTIONS.add_argument("--disable-blink-features=AutomationControlled")

        self.conversion_factors = {
            '1600_to_1500': 0.9375,
            '3200_to_3000': 0.9375,
            'mile_to_1500': 0.9321,
            '2mile_to_3000': 0.9321
        }

    
    def __repr__(self) -> str:
        return f'Scraper object for [{self.website}]'
    

    def __call__(self, start: int, end: int) -> pd.DataFrame:
        '''Run the whole scraping program for all levels, all sexes, all events, over the specified range of years

        Parameters:
          -  start (`int`): [yyyy], the first year to be downloaded
          -  end (`int`): [yyyy], the last year to be downloaded

        Returns:
          -  df (pd.DataFrame): A pd.DataFrame of all the MS and HS results from indoor and outdoor of a specified range of years
        '''

        df = self.download_and_export(start=start, end=end)

        return df


    def log_in(self, url: str | None = None, options: Options | None = None) -> webdriver.Chrome:
        '''Uses a Milesplit URL to log in

        This could be optimized a lot I think. There are lots of 'time.sleep()' calls to prevent JS detectors from kicking the the program out, and some of them could be removed or minimized. The login isn't always successful, sometimes it just doesn't enter the keys, and I'm not sure how to automate checking if the login was successful and the program is running. Also, my computer is just very old and slow so it takes a while to load ads/go fromp page to page.
        
        Parameters:
          -  url (`str`): the URL to access the login
          -  options (`Options`): Selenium Chrome Driver options

        Returns:
          -  driver (`webdriver.Chrome`): the driver being used to navigate MileSplit
        '''

        if url is None:
            url = self.url
        
        if options is None:
            options = self.OPTIONS

        # Start Browser and Driver
        webdriver_service = Service(executable_path=self.EXE_PATH)
        driver = webdriver.Chrome(service=webdriver_service, options=options)
        driver.implicitly_wait(45)

        # Log in sequence
        driver.get(url)

        # Let ads load
        time.sleep(5)

        # Click account button
        account_button = driver.find_element(By.XPATH, '//*[@id="account"]/div[1]')
        account_button.click()
        time.sleep(2)

        # Click login link
        login_link = driver.find_element(By.XPATH, '//*[@id="account"]/div[2]/section/ul/li[1]/a')
        login_link.click()
        time.sleep(5)

        # Enter Username and Password
        email = driver.find_element(By.ID, 'email')
        email.send_keys(self.USERNAME)
        time.sleep(1)

        password = driver.find_element(By.ID, 'password')
        password.send_keys(self.PASSWORD)
        time.sleep(1)

        # Submit form
        submit = driver.find_element(By.ID, 'frmSubmit')
        submit.click()

        # Let login process
        time.sleep(3)

        return driver


    def create_url(self,
                   level: str, 
                   sex: str, 
                   season: str, 
                   event: str, 
                   year: str, 
                   page: int) -> str:
        '''Use the different variables in the milesplit url to make the url for the next download.
        
        Parameters:
          -  level (`str`): 'middle' for middle school or 'high' for high school
          -  sex (`str`): 'girls' or 'boys'
          -  season (`str`): 'indoor' or 'outdoor'
          -  event (`str`): the event of interest. Options available in class documentation
          -  year (`str`): 'yyyy', the year of the season (2000-2024)
          -  page (`int`): options are 1 - 20

        Returns:
          -  url (`str`): the full URL for the Milesplit page to download
        '''
        
        # TODO: #3 Add state functionality, does the rest of the URL stay the same when you add the state prefix?
        url = f'https://www.milesplit.com/rankings/events/{level}-school-{sex}/{season}-track-and-field/{event}?year={year}&accuracy=fat&grade=all&ageGroup=&league=0&meet=0&team=0&venue=0&conversion=n&page={page}'

        return url


    def clean_400m_times(self, row: str | float) -> float:
        '''Take each row of a pd.Series of 400m data and modify it according to its format
        
        Parameters:
          -  row (`str` | `float`): Pages with everything below 1 minute comes in as a float, a page witheverything above comes as a string. If a page of results has both under 1 minute and above 1 minute on the same page, they all come as a string.
        '''

        if str(row)[0] in ['4', '5']:
            return row
        else:
            return 60 * int(str(row)[0]) + float(str(row)[2:7])
    
    
    def clean_hurdle_times(self, row: str | float) -> float:
        '''Take each row of a pd.Series of 300H and 400H data and modify it according to its format
        
        Parameters:
          -  row (`str` | `float`): Pages with everything below 1 minute comes in as a float, and a page witheverything above comes as a string. If a page of results has both under 1 minute and above 1 minute on the same page, they all come as a string.
        '''

        if str(row)[0] in ['3', '4', '5']:
            return row
        else:
            return 60 * int(str(row)[0]) + float(str(row)[2:7])
        

    def clean_1500_3000_times(self, row: str) -> float:
        '''Take each row of a pd.DataFrame column of 1500m or 3000m times and convert it to seconds
        
        Parameters:
          -  row (`str`): the rows of the `pd.DataFrame` 1500m or 3000m column

        Returns:
          - val (`float`): the value of the 1500m/3000m time in seconds
        '''

        if str(row)[0] == '1':  # 10 mins or greater
            return round(((60 * int(str(row)[:2]) + float(str(row)[3:8]))), ndigits=2)
        else:
            return round(((60 * int(str(row)[0]) + float(str(row)[2:7]))), ndigits=2)
    
    
    def clean_steeple_times(self, row: str) -> float:
        '''Take each row of a pd.DataFrame column of 1500m times and convert it to seconds
        
        Parameters:
          -  row (`str`): the rows of the `pd.DataFrame` 1500m column

        Returns:
          - val (`float`): the value of the 1500m time in seconds
        '''

        if str(row)[0] == '1':  # 10 mins or greater
            return round(((60 * int(str(row)[:2]) + float(str(row)[3:8]))), ndigits=2)
        else:
            return round(((60 * int(str(row)[0]) + float(str(row)[2:7]))), ndigits=2)


    def convert_mile_times(self, row: str, event: str) -> float:
        '''Take each row of a pd.Series of mile or 1600m data and convert it to 1500m times in seconds
        
        Parameters:
          -  row (`str`): the rows of the `pd.DataFrame` column
          -  event (`str`): the event of interest identified for conversion

        Returns:
          - val (`float`): the value of the converted 1500m time in seconds
        '''
        
        CONV_FACTOR = self.conversion_factors['mile_to_1500'] if event == 'Mile' else self.conversion_factors['1600_to_1500']

        if str(row)[0] == '1':  # 10 mins or greater
            return round(((60 * int(str(row)[:2]) + float(str(row)[3:8])) * CONV_FACTOR), ndigits=2)
        else:
            return round(((60 * int(str(row)[0]) + float(str(row)[2:7])) * CONV_FACTOR), ndigits=2)
    
    
    def convert_2mile_times(self, row: str, event: str) -> float:
        '''Take each row of a pd.Series of 2 mile or 3200m data and convert it to 3000m times in seconds
        
        Parameters:
          -  row (`str`): the rows of the `pd.DataFrame` column
          -  event (`str`): the event of interest identified for conversion

        Returns:
          - val (`float`): the value of the converted 3000m time in seconds
        '''
        
        CONV_FACTOR = self.conversion_factors['2mile_to_3000'] if event == '2Mile' else self.conversion_factors['3200_to_3000']

        if str(row)[0] == '1':  # 10 mins or greater
            return round(((60 * int(str(row)[:2]) + float(str(row)[3:8])) * CONV_FACTOR), ndigits=2)
        else:
            return round(((60 * int(str(row)[0]) + float(str(row)[2:7])) * CONV_FACTOR), ndigits=2)
                  
    
    def determine_col_name(self, event: str) -> str:
        '''Determine and export the name of the column holding times for a given track event
        
        Parameters:
          -  event (`str`): the specified track event. Options available in class and __init__ documentation'''
        
        match event:
            case '1500m' | '1600m' | 'Mile':
                return 'time_1500'
            case '3000m' | '3200m' | '2Mile':
                return 'time_3000'
            case '110H' | '300H' | '400H' | '2000mSC':
                return f'time_{event}'
            case '100H':
                return 'time_110H'
            case '60m':
                return 'time_100'
            case '60H':
                return 'time_110H'
            case _:
                return f'time_{event[:-1]}' 


    def download_and_clean(self, data: pd.DataFrame, event: str, season: str, year: str) -> pd.DataFrame:
        '''Takes the output from the page content and cleans it up
        
        Parameters:
          -  data (`pd.DataFrame`): The pd.DataFrame returned from the page source
          -  event (`str`): the event of interest. Options available in class documentation
          -  season (`str`): 'indoor' or 'outdoor'
          -  year (`str`): 'yyyy', the year of the season (2000-2024)
            
        Returns:
          -  df (`pd.DataFrame`): a pd.DataFrame that's been cleaned and ready for concatenation with other DataFrames
        '''

        df = data
        df.columns = df.columns.str.lower()

        # Combine id columns
        df['athlete_team'] = df['athlete/team'] + ' ' + df['grade'].astype('str')
        df['athlete_team'] = df['athlete_team'].str.replace('.0', '')

        # Process time columns and convert 1600m and Mile to 1500m times and 3200m and 2Mile to 3000m times
        try:
            match event:
                case '60m' | '60H' | '100m' | '200m' | '100H' | '110H':
                    df[self.determine_col_name(event)] = df['time'].astype('float')

                case '300H' | '400H':
                    if df['time'].dtype == 'float64':
                        df[self.determine_col_name(event)] = df['time']
                    else:
                        df[self.determine_col_name(event)] = df['time'].apply(self.clean_hurdle_times)

                case '400m':
                    df[self.determine_col_name(event)] = df['time'].apply(self.clean_400m_times)

                case '800m':
                    df[self.determine_col_name(event)] = 60 * df['time'].str[0].astype('int') + df['time'].str[2:7].astype('float')

                case '1500m':
                    if all(df['time'].str.len() == 7):
                        df[self.determine_col_name(event)] = (60 * df['time'].str[0].astype('int') + df['time'].str[2:7].astype('float')).round(2)
                    else:
                        df[self.determine_col_name(event)] = df['time'].apply(self.clean_1500_3000_times)

                case '1600m':
                    if all(df['time'].str.len() == 7):
                        df[self.determine_col_name(event)] = ((60 * df['time'].str[0].astype('int') + df['time'].str[2:7].astype('float')) * self.conversion_factors['1600_to_1500']).round(2)
                    else:
                        df[self.determine_col_name(event)] = df['time'].apply(self.convert_mile_times, event=event)

                case 'Mile':
                    if all(df['time'].str.len() == 7):
                        df[self.determine_col_name(event)] = ((60 * df['time'].str[0].astype('int') + df['time'].str[2:7].astype('float')) * self.conversion_factors['mile_to_1500']).round(2)
                    else:
                        df[self.determine_col_name(event)] = df['time'].apply(self.convert_mile_times, event=event)

                case '2000mSC':
                    if all(df['time'].str.len() == 7):
                        df[self.determine_col_name(event)] = (60 * df['time'].str[0].astype('int') + df['time'].str[2:7].astype('float')).round(2)
                    else:
                        df[self.determine_col_name(event)] = df['time'].apply(self.clean_steeple_times)

                case '3000m':
                    if all(df['time'].str.len() == 7):
                        df[self.determine_col_name(event)] = (60 * df['time'].str[0].astype('int') + df['time'].str[2:7].astype('float')).round(2)
                    else:
                        df[self.determine_col_name(event)] = df['time'].apply(self.clean_1500_3000_times)

                case '3200m':
                    if all(df['time'].str.len() == 7):
                        df[self.determine_col_name(event)] = ((60 * df['time'].str[0].astype('int') + df['time'].str[2:7].astype('float')) * self.conversion_factors['3200_to_3000']).round(2)
                    else:
                        df[self.determine_col_name(event)] = df['time'].apply(self.convert_2mile_times, event=event)

                case '2Mile':
                    if all(df['time'].str.len() == 7):
                        df[self.determine_col_name(event)] = ((60 * df['time'].str[0].astype('int') + df['time'].str[2:7].astype('float')) * self.conversion_factors['2mile_to_3000']).round(2)
                    else:
                        df[self.determine_col_name(event)] = df['time'].apply(self.convert_2mile_times, event=event)

        except ValueError:
            print(f'''ValueError in {year}'s {season} {event}''')

        # Drop extra columns
        df = df.drop(columns = ['rank', 'athlete/team', 'grade', 'meet  date  place', 'time'])

        # Add season info
        df['season'] = f'{season}_{year}'

        time_col = self.determine_col_name(event) 

        return df[['athlete_team', time_col, 'season']]
        

    def download_single_event(self,
                              driver: webdriver.Chrome, 
                              level: str, 
                              sex: str, 
                              season: str, 
                              event: str, 
                              year: str) -> pd.DataFrame:
        '''Download all 20 pages of an event of milesplit data for a single sex
        
        Parameters:
          -  driver (`webdriver.Chrome`): the webdriver used to log in
          -  level (`str`): 'middle' for middle school or 'high' for high school
          -  sex (`str`): 'girls' or 'boys'
          -  season (`str`): 'indoor' or 'outdoor'
          -  event (`str`): any valid track event
          -  year (`str`): 'yyyy', the year of the season (2000-2024)

        Returns:
          -  dfs (`pd.DataFrame`): a pd.DataFrame of all 20 pages of event data that belongs to a single sex
        '''

        # Indoor sprint specifications and outdoor girls/boys short hurdles specifications
        match season:
            case 'indoor':
                match event:
                    case '100m':
                        event = '60m'
                    case '110H' | '100H':
                        event = '60H'

            case 'outdoor':
                if event == '110H':
                    match sex:
                        case 'boys':
                            event = '110H'
                        case 'girls':
                            event = '100H'

        # If 2k Steeple or long hurdles, season must be 'outdoor'
        if event in ['2000mSC', '300H', '400H']:
            season = 'outdoor'
      
        # Iterate through pages
        dfs = None

        # TODO: For full production, turn the range from 1 -> 21
        for page in range(1, 21, 1):
            # Create URL
            url = self.create_url(level, sex, season, event, year, page)
            driver.get(url)

            # Let the page and ads load
            time.sleep(5)

            # Grab and download page content
            content = driver.page_source

            try:
                df = pd.read_html(StringIO(content))[0].pipe(self.download_and_clean, event=event, season=season, year=year)
            except ValueError:
                print(f'No Tables Found in [{url}]')
                continue

            if dfs is None:
                dfs = df
                continue

            dfs = pd.concat([dfs, df])

        return dfs


    def download_and_join_events(self,
                             driver: webdriver.Chrome, 
                             level: str, 
                             sex: str, 
                             season: str, 
                             year: str) -> pd.DataFrame:
        '''Download running event data from Milesplit and join the tables together
        
        Parameters:
          -  driver (`webdriver.Chrome`): the webdriver used to log in
          -  level (`str`): 'middle' for middle school or 'high' for high school
          -  sex (`str`): 'girls' or 'boys'
          -  season (`str`): 'indoor' or 'outdoor'
          -  year (`str`): 'yyyy', the year of the season (2000-2024)

        Returns:
          -  df (`pd.DataFrame`): a pd.DataFrame of a single track event or a pd.DataFrame of multiple track events where a single individual has run all of the requested events, that belongs to a single sex for a single season. The latter is created such that you can analyze the performances between different events that individuals can produce.
        '''

        df_outcome = self.download_single_event(driver=driver,
                                                level=level,
                                                sex=sex,
                                                season=season,
                                                event=self.outcome_event,
                                                year=year)
        
        outcome_col = self.determine_col_name(self.outcome_event)

        # If you're only downloading the predictor event, skip merging step
        if self.predictor_events is None:
            return df_outcome[['athlete_team', outcome_col, 'season']]
        
        dfs = None
        if dfs is None:
            dfs = df_outcome

        if isinstance(self.predictor_events, str):  # Check for singular event or list of events
            predictor_cols = self.determine_col_name(self.predictor_events)
            df_predictor = self.download_single_event(driver=driver,
                                                level=level,
                                                sex=sex,
                                                season=season,
                                                event=self.predictor_events,
                                                year=year)
            
            dfs = dfs.merge(df_predictor, how='left', left_on=['athlete_team', 'season'], right_on=['athlete_team', 'season'])

            return dfs[['athlete_team', outcome_col, predictor_cols, 'season']].dropna()
            
        else:
            predictor_cols = [self.determine_col_name(event) for event in self.predictor_events]
            for i in range(len(self.predictor_events)):
                df_predictor = self.download_single_event(driver=driver,
                                                    level=level,
                                                    sex=sex,
                                                    season=season,
                                                    event=self.predictor_events[i],
                                                    year=year)
                
                # Join data into same DataFrame
                dfs = dfs.merge(df_predictor, how='left', left_on=['athlete_team', 'season'], right_on=['athlete_team', 'season'])

            return dfs[['athlete_team', outcome_col] + predictor_cols + ['season']].dropna()


    def combine_sexes(self,
                      driver: webdriver.Chrome, 
                      level: str, 
                      season: str, 
                      year: str) -> pd.DataFrame:
        '''Download track event data from Milesplit for both sexes and combines them together
        
        Parameters:
          -  driver (`webdriver.Chrome`): the webdriver used to log in
          -  level (`str`): 'middle' for middle school or 'high' for high school
          -  season (`str`): 'indoor' or 'outdoor'
          -  year (`str`): 'yyyy', the year of the season (2000-2024)

        Returns:
          -  df (`pd.DataFrame`): a pd.DataFrame of track event data for both sexes for a single season
        '''

        match self.sex:
            case 'f':
                df_f = self.download_and_join_events(driver, level, 'girls', season, year).assign(sex='f')
                return df_f

            case 'm':
                df_m = self.download_and_join_events(driver, level, 'boys', season, year).assign(sex='m')
                return df_m

            case _:
                df_f = self.download_and_join_events(driver, level, 'girls', season, year).assign(sex='f')
                df_m = self.download_and_join_events(driver, level, 'boys', season, year).assign(sex='m')

                df = pd.concat([df_f, df_m])
                return df


    def download_seasons(self,
                         driver: webdriver.Chrome, 
                         level: str, 
                         year: str) -> pd.DataFrame:
        '''Download track event data from Milesplit for both sexes and combines them together
        
        Parameters:
          -  driver (`webdriver.Chrome`): the webdriver used to log in
          -  level (`str`): 'middle' for middle school or 'high' for high school
          -  year (`str`): 'yyyy', the year of the season (2000-2024)

        Returns:
          -  df (`pd.DataFrame`): a pd.DataFrame of track event for both sexes for a indoor and outdoor of a certain year for a certain level of competition
        '''

        df_indoor = self.combine_sexes(driver=driver, level=level, season='indoor', year=year)
        df_outdoor = self.combine_sexes(driver=driver, level=level, season='outdoor', year=year)

        df = pd.concat([df_indoor, df_outdoor])

        return df
                    

    def download_levels(self,
                        driver: webdriver.Chrome, 
                        year: str) -> pd.DataFrame:
        '''Download track event data from Milesplit for both sexes in middle school and high school and combines them together
        
        Parameters:
          -  driver (`webdriver.Chrome`): the webdriver used to log in
          -  year (`str`): 'yyyy', the year of the season (2000-2024)

        Returns:
          -  df (`pd.DataFrame`): a pd.DataFrame of track event data for both sexes for a indoor and outdoor of a certain year for both middle school and high school
        '''

        df_hs = self.download_seasons(driver=driver, year=year, level='high')
        df_ms = self.download_seasons(driver=driver, year=year, level='middle')
        
        df = pd.concat([df_hs, df_ms])

        return df


    def download_years(self, driver: webdriver.Chrome, start: int, end: int) -> pd.DataFrame:
        '''Download track event data from Milesplit for both sexes in middle school and high school and combines them together for multiple years of data
        
        Parameters:
          -  driver (`webdriver.Chrome`): the webdriver used to log in
          -  start (`int`): [yyyy], the first year to be downloaded
          -  end (`int`): [yyyy], the last year to be downloaded

        Returns:
          -  dfs (`pd.DataFrame`): a pd.DataFrame of both track event data for both sexes for a indoor and outdoor for all specified years for both middle school and high school
        '''

        dfs = None

        for year in range(start, end + 1, 1):
            df = self.download_levels(driver=driver, year=str(year))

            if dfs is None:
                dfs = df
                continue

            dfs = pd.concat([dfs, df])
        
        dfs = dfs.drop_duplicates()
        
        return dfs


    def download_and_export(self, start: int, end: int) -> pd.DataFrame:
        '''Run the whole scraping program for all levels, all sexes, all events, over the specified range of years

        Parameters:
          -  start (`int`): [yyyy], the first year to be downloaded
          -  end (`int`): [yyyy], the last year to be downloaded

        Returns:
          -  df (pd.DataFrame): A pd.DataFrame of all the MS and HS results from indoor and outdoor of a specified range of years
        '''

        driver = self.log_in()
        df = self.download_years(driver, start, end)
        driver.close()

        if self.sex is None:
          sex = ''
        else:
          sex = f'_{self.sex}'        

        df.drop_duplicates().to_csv(f'data/milesplit_indoor_{start}-outdoor_{end}_{self.predictor_events}{sex}_{datetime.now():%Y-%m-%d}.csv', index=False)

        return df.drop_duplicates()

