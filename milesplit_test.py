import pandas as pd
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium import webdriver
import time
from dotenv import dotenv_values
from io import StringIO


# set url
first_url = 'https://www.milesplit.com/rankings/events/high-school-girls/indoor-track-and-field/800m?year=2024&accuracy=fat&grade=all&ageGroup=&league=0&meet=0&team=0&venue=0&conversion=n&page=1'

config = dotenv_values('.env')

USERNAME = config['USERNAME']
PASSWORD = config['PASSWORD']

# Set Options (don't know if this does anything rn)
options = Options()
options.add_argument('--ignore-certificate-errors')
options.add_experimental_option('excludeSwitches', ['enable-logging'])
options.add_argument("--disable-blink-features=AutomationControlled")

# Start Browser and Driver
webdriver_service = Service(executable_path=r"C:\Users\mitch\Documents\Programs\chromedriver\chrome-win32-121\chromedriver-win32\chromedriver.exe")
driver = webdriver.Chrome(service=webdriver_service, options=options)

# Log in sequence
driver.get(first_url)

# Let ads load
time.sleep(5)

# Click account button
account_button = driver.find_element(By.XPATH, '//*[@id="account"]/div[1]')
account_button.click()
time.sleep(2)

# Click login link
login_link = driver.find_element(By.XPATH, '//*[@id="account"]/div[2]/section/ul/li[1]/a')
login_link.click()
time.sleep(2)

# Enter Username and Password
email = driver.find_element(By.ID, 'email')
email.send_keys(USERNAME)
time.sleep(1)

password = driver.find_element(By.ID, 'password')
password.send_keys(PASSWORD)
time.sleep(1)

# Submit form
submit = driver.find_element(By.ID, 'frmSubmit')
submit.click()

# Let login process
time.sleep(5)

# Start scraping tables
def create_url(level: str, 
               sex: str, 
               season: str, 
               event: str, 
               year: int, 
               page: int) -> str:
    '''Use the different variables in the milesplit url to make the url for the next download.
    
    Parameters:
        level (str): 'middle' for middle school or 'high' for high school
        sex (str): 'girls' or 'boys'
        season (str): 'indoor' or 'outdoor'
        event (str): typically '800m', but also takes '400m'
        year (int): [yyyy], the year of the season
        page (int): options are 1 - 20

    Returns:
        url_dict (): the full URL for the Milesplit page to download
    '''
    
    url = f'https://www.milesplit.com/rankings/events/{level}-school-{sex}/{season}-track-and-field/{event}?year={year}&accuracy=fat&grade=all&ageGroup=&league=0&meet=0&team=0&venue=0&conversion=n&page={page}'

    return url



driver.get(create_url('high', 'girls', 'indoor', '800m', 2024, 1))
content = driver.page_source
print(pd.read_html(content)[0])


