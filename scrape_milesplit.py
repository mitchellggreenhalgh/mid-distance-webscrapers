'''A module for a MileSplitScraper class that uses selenium to scrape 800m and 400m, 1600m, or mile data from MileSplit'''

import pandas as pd
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium import webdriver
import time
from dotenv import dotenv_values
from io import StringIO


FIRST_URL = 'https://www.milesplit.com/rankings/events/high-school-girls/indoor-track-and-field/800m?year=2024&accuracy=fat&grade=all&ageGroup=&league=0&meet=0&team=0&venue=0&conversion=n&page=1'

config = dotenv_values('.env')


class MileSplitScraper():
    '''A class to download 800m and 400m or 1600m data from the National MileSplit database.'''

    def __init__(self, other_event: str, url: str=FIRST_URL):
        '''Initialize the MileSplitScraper class
        
        Parameters:
          -  other_event (str): Dictates which other event to download. Options: '400m', '1600m', and 'Mile'
          -  url (str): a URL to the milesplit rankings portion of the website to log in on and start scraping
        '''

        self.other_event = other_event
        self.url = url
        self.website = 'https://www.milesplit.com/'
        self.USERNAME = config['USERNAME']
        self.PASSWORD = config['PASSWORD']

        self.OPTIONS = Options()
        self.OPTIONS.add_argument('--ignore-certificate-errors')
        self.OPTIONS.add_experimental_option('excludeSwitches', ['enable-logging'])
        self.OPTIONS.add_argument("--disable-blink-features=AutomationControlled")

    
    def __repr__(self):
        return f'Scraper object for [{self.website}]'


    def log_in(self, url: str | None=None, options: Options | None=None) -> webdriver.Chrome:
        '''Uses a Milesplit URL to log in

        This could be optimized a lot I think. There are lots of 'time.sleep()' calls to prevent JS detectors from kicking the the program out, and some of them could be removed or minimized. The login isn't always successful, sometimes it just doesn't enter the keys, and I'm not sure how to automate checking if the login was successful and the program is running.
        
        Parameters:
          -  url (str): the URL to access the login
          -  options (Options): Selenium Chrome Driver options

        Returns:
          -  driver (webdriver.Chrome): the driver being used to navigate MileSplit
        '''

        if url is None:
            url = self.url
        
        if options is None:
            options = self.OPTIONS

        # Start Browser and Driver
        webdriver_service = Service(executable_path=r"C:\Users\mitch\Documents\Programs\chromedriver\chrome-win32-121\chromedriver-win32\chromedriver.exe")
        driver = webdriver.Chrome(service=webdriver_service, options=options)
        driver.implicitly_wait(30)

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
          -  level (str): 'middle' for middle school or 'high' for high school
          -  sex (str): 'girls' or 'boys'
          -  season (str): 'indoor' or 'outdoor'
          -  event (str): the event of interest. Options: '800m', '400m', '1600m', 'Mile'
          -  year (str): 'yyyy', the year of the season (2000-2024)
          -  page (int): options are 1 - 20

        Returns:
          -  url (str): the full URL for the Milesplit page to download
        '''
        
        # TODO: #3 Add state functionality, does the rest of the URL stay the same when you add the state prefix?
        url = f'https://www.milesplit.com/rankings/events/{level}-school-{sex}/{season}-track-and-field/{event}?year={year}&accuracy=fat&grade=all&ageGroup=&league=0&meet=0&team=0&venue=0&conversion=n&page={page}'

        return url


    def clean_400m_times(self, row: str | float) -> float:
        '''Take each row of a pd.Series of 400m data and modify it according to its format
        
        Parameters:
          -  row (str | float): Pages with everything below 1 minute comes in as a float, a page witheverything above comes as a string. If a page of results has both under 1 minute and above 1 minute on the same page, they all come as a string.
        '''

        if str(row)[0] in ['4', '5']:
            return row
        else:
            return 60 * int(str(row)[0]) + float(str(row)[2:7])


    def clean_mile_times(self, row: str) -> float:
        '''Take each row of a pd.Series of mile data and modify it according to its format
        
        Parameters:
          -  row (str): only need to modify rows that have 10 minutes or more
        '''
        SCALAR = 0.93205678835 if self.other_event == 'Mile' else 0.9375

        if str(row)[0] == '1':
            return round(((60 * int(str(row)[:2]) + float(str(row)[3:8])) * SCALAR), ndigits=2)
        else:
            return round(((60 * int(str(row)[0]) + float(str(row)[2:7])) * SCALAR), ndigits=2)
                  

    def download_and_clean(self, data: pd.DataFrame, event: str, season: str, year: str) -> pd.DataFrame:
        '''Takes the output from the page content and cleans it up
        
        Parameters:
          -  data (pd.DataFrame): The pd.DataFrame returned from the page source
          -  event (str): the event of interest. Options: '800m', '400m', '1600m', 'Mile'
          -  season (str): 'indoor' or 'outdoor'
          -  year (str): 'yyyy', the year of the season (2000-2024)
            
        Returns
          -  df (pd.DataFrame): a pd.DataFrame that's been cleaned and ready for concatenation with other DataFrames
        '''

        df = data
        df.columns = df.columns.str.lower()

        # Combine id columns
        df['athlete_team'] = df['athlete/team'] + ' ' + df['grade'].astype('str')
        df['athlete_team'] = df['athlete_team'].str.replace('.0', '')

        # Process time column and convert 1600m and Mile to 1500m times
        try:
            if event == '800m':
                df[f'time_{event[:-1]}'] = 60 * df['time'].str[0].astype('int') + df['time'].str[2:7].astype('float')
            elif event == '1600m':
                if all(df['time'].str.len() == 7):
                    df['time_1500'] = ((60 * df['time'].str[0].astype('int') + df['time'].str[2:7].astype('float')) * 0.9375).round(2)
                else:
                    df['time_1500'] = df['time'].apply(self.clean_mile_times)
            elif event == 'Mile':
                if all(df['time'].str.len() == 7):
                    df['time_1500'] = ((60 * df['time'].str[0].astype('int') + df['time'].str[2:7].astype('float')) * 0.93205678835).round(2)
                else:
                    df['time_1500'] = df['time'].apply(self.clean_mile_times)
            elif event == '400m':
                df[f'time_{event[:-1]}'] = df['time'].apply(self.clean_400m_times)
        except ValueError:
            print(f'''ValueError in {year}'s {season} {event}''')

        # Drop extra columns
        df = df.drop(columns = ['rank', 'athlete/team', 'grade', 'meet  date  place', 'time'])

        # Add season info
        df['season'] = f'{season}_{year}'

        time_col = f'time_{event[:-1]}' if event in ['400m', '800m'] else 'time_1500'

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
          -  driver (webdriver.Chrome): the webdriver used to log in
          -  level (str): 'middle' for middle school or 'high' for high school
          -  sex (str): 'girls' or 'boys'
          -  season (str): 'indoor' or 'outdoor'
          -  event (str): typically '800m' and '400m', but also takes any other valid track event
          -  year (str): 'yyyy', the year of the season (2000-2024)

        Returns:
          -  dfs (pd.DataFrame): a pd.DataFrame of all 20 pages of event data that belongs to a single sex
        '''

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
            df = pd.read_html(StringIO(content))[0].pipe(self.download_and_clean, event=event, season=season, year=year)

            if dfs is None:
                dfs = df
                continue

            dfs = pd.concat([dfs, df])

        return dfs


    def download_both_events(self,
                             driver: webdriver.Chrome, 
                             level: str, 
                             sex: str, 
                             season: str, 
                             year: str) -> pd.DataFrame:
        '''Download 800m and 400m, 1600m, or mile data from Milesplit and join the tables together
        
        Parameters:
          -  driver (webdriver.Chrome): the webdriver used to log in
          -  level (str): 'middle' for middle school or 'high' for high school
          -  sex (str): 'girls' or 'boys'
          -  season (str): 'indoor' or 'outdoor'
          -  year (str): 'yyyy', the year of the season (2000-2024)

        Returns:
          -  df (pd.DataFrame): a pd.DataFrame of both 800m and 400m, 1600m, or mile data that belongs to a single sex for a single season
        '''

        df_800 = self.download_single_event(driver=driver,
                                            level=level,
                                            sex=sex,
                                            season=season,
                                            event='800m',
                                            year=year)
        
        df_other = self.download_single_event(driver=driver,
                                            level=level,
                                            sex=sex,
                                            season=season,
                                            event=self.other_event,
                                            year=year)
        
        df = df_800.merge(df_other, how='left', left_on=['athlete_team', 'season'], right_on=['athlete_team', 'season']) \
            .dropna()
        
        col_3 = 'time_400' if self.other_event == '400m' else 'time_1500'

        return df[['athlete_team', 'time_800', col_3, 'season']]


    def combine_sexes(self,
                      driver: webdriver.Chrome, 
                      level: str, 
                      season: str, 
                      year: str) -> pd.DataFrame:
        '''Download 800m and 400m, 1600m, or mile data from Milesplit for both sexes and combines them together
        
        Parameters:
          -  driver (webdriver.Chrome): the webdriver used to log in
          -  level (str): 'middle' for middle school or 'high' for high school
          -  season (str): 'indoor' or 'outdoor'
          -  year (str): 'yyyy', the year of the season (2000-2024)

        Returns:
          -  df (pd.DataFrame): a pd.DataFrame of both 800m and 400m, 1600m, or mile data that for both sexes for a single season
        '''

        df_f = self.download_both_events(driver, level, 'girls', season, year)
        df_m = self.download_both_events(driver, level, 'boys', season, year)

        df = pd.concat([df_f, df_m])

        return df


    def download_seasons(self,
                         driver: webdriver.Chrome, 
                         level: str, 
                         year: str) -> pd.DataFrame:
        '''Download 800m and 400m data from Milesplit for both sexes and combines them together
        
        Parameters:
          -  driver (webdriver.Chrome): the webdriver used to log in
          -  level (str): 'middle' for middle school or 'high' for high school
          -  year (str): 'yyyy', the year of the season (2000-2024)

        Returns:
          -  df (pd.DataFrame): a pd.DataFrame of both 800m and 400m data that for both sexes for a indoor and outdoor of a certain year for a certain level of competition
        '''

        df_indoor = self.combine_sexes(driver=driver, level=level, season='indoor', year=year)
        df_outdoor = self.combine_sexes(driver=driver, level=level, season='outdoor', year=year)

        df = pd.concat([df_indoor, df_outdoor])

        return df
                    

    def download_levels(self,
                        driver: webdriver.Chrome, 
                        year: str) -> pd.DataFrame:
        '''Download 800m and 400m data from Milesplit for both sexes in middle school and high school and combines them together
        
        Parameters:
          -  driver (webdriver.Chrome): the webdriver used to log in
          -  year (str): 'yyyy', the year of the season (2000-2024)

        Returns:
          -  df (pd.DataFrame): a pd.DataFrame of both 800m and 400m data that for both sexes for a indoor and outdoor of a certain year for both middle school and high school
        '''

        df_hs = self.download_seasons(driver=driver, year=year, level='high')
        df_ms = self.download_seasons(driver=driver, year=year, level='middle')
        
        df = pd.concat([df_hs, df_ms])

        return df


    def download_years(self, driver: webdriver.Chrome, start: int, end: int) -> pd.DataFrame:
        '''Download 800m and 400m, 1600m, or mile data from Milesplit for both sexes in middle school and high school and combines them together for multiple years of data
        
        Parameters:
          -  driver (webdriver.Chrome): the webdriver used to log in
          -  start (int): [yyyy], the first year to be downloaded
          -  end (int): [yyyy], the last year to be downloaded

        Returns:
          -  dfs (pd.DataFrame): a pd.DataFrame of both 800m and 400m data that for both sexes for a indoor and outdoor for all specified years for both middle school and high school
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
          -  start (int): [yyyy], the first year to be downloaded
          -  end (int): [yyyy], the last year to be downloaded

        Returns:
          -  df (pd.DataFrame): A pd.DataFrame of all the MS and HS results from indoor and outdoor of a specified range of years
        '''

        driver = self.log_in()
        df = self.download_years(driver, start, end)
        driver.close()
        df.to_csv(f'data/milesplit_indoor_{start}-outdoor_{end}_{self.other_event}.csv', index=False)

        return df


if __name__ == '__main__':
    from datetime import datetime
    start = datetime.now()
    ms = MileSplitScraper(other_event='Mile')
    ms.download_and_export(2020, 2024)
    end = datetime.now()
    print(start - end)
