# code for scraping the SST data from the NOAA website
# use selenium to download the data
from tqdm import tqdm
import wget as wget

URL = 'https://www.ncei.noaa.gov/thredds-ocean/catalog/ghrsst/L4/GLOB/NCEI/AVHRR_OI/catalog.html'
# URL = 'https://www.ncei.noaa.gov/thredds-ocean/catalog/ghrsst/L4/GLOB/NCEI/AVHRR_OI/2023/021/catalog.html?dataset=ghrsst/L4/GLOB/NCEI/AVHRR_OI/2023/021/20230121120000-NCEI-L4_GHRSST-SSTblend-AVHRR_OI-GLOB-v02.0-fv02.1.nc'
from selenium.webdriver.chrome.service import Service
from selenium import webdriver
import bs4 as bs
import os

from utils.config_loader import Configs

DATA_DIR = Configs.get('DATA_DIR')

SST_DIR = os.path.join(DATA_DIR, 'SST')

# create the directory if it doesn't exist
if not os.path.exists(SST_DIR):
    os.mkdir(SST_DIR)


class SSTScraper:
    def __init__(self, url=URL, start_year=2017, end_year=2023, use_template=True,
                 base_url='https://www.ncei.noaa.gov/thredds-ocean/fileServer/ghrsst/L4/GLOB/NCEI/AVHRR_OI/YYYY/DAY_NUM/',
                 template='YYYYMMDD120000-NCEI-L4_GHRSST-SSTblend-AVHRR_OI-GLOB-v02.0-fv02.1.nc'):

        self.start_year = start_year
        self.end_year = end_year

        if use_template:
            self.base_url = base_url
            self.template = template
            self.download_using_template()
        else:

            # load selenium webdriver
            self.use_template = use_template
            self.soup = None
            self.source = None
            self.year_list = [str(year) for year in range(start_year, end_year + 1)]
            self.urls_yearly = []
            driver_path = Configs.get('CHROME_DRIVER_PATH')
            service = Service(executable_path=driver_path)
            self.driver = webdriver.Chrome(service=service)
            self.url = url

    def download_using_template(self):
        num_days_per_month_non_leap = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        sum_of_days_per_month_non_leap = [sum(num_days_per_month_non_leap[:i]) for i in range(1, 13)]
        sum_of_days_per_month_non_leap.insert(0, 0)
        num_days_per_month_leap = num_days_per_month_non_leap.copy()
        num_days_per_month_leap[1] = 29
        sum_of_days_per_month_leap = [sum(num_days_per_month_leap[:i]) for i in range(1, 13)]
        sum_of_days_per_month_leap.insert(0, 0)

        # go through each year and download the data
        for year in tqdm(range(self.start_year, self.end_year + 1), 'Years'):
            if year % 4 == 0:
                days_total = 366
            else:
                days_total = 365
            # go through each day and download the data
            for day_num in tqdm(range(1, days_total+1), 'Days'):
                # convert day_num to month and day for example: 32 -> 02/01
                if year % 4 == 0:
                    sum_of_days_per_month = sum_of_days_per_month_leap
                else:
                    sum_of_days_per_month = sum_of_days_per_month_non_leap
                # find the month by checking which sum_of_days_per_month_leap is greater than day_num
                month = [i for i, x in enumerate(sum_of_days_per_month) if x >= day_num][0]
                # find the day by subtracting the sum_of_days_per_month_leap[month - 1] from day_num
                day = f'{day_num - sum_of_days_per_month[month - 1]:02d}'
                month = f'{month:02d}'
                # get the url
                url = self.base_url.replace('YYYY', str(year)).replace('DAY_NUM',
                                                                       f'{day_num:03d}') + self.template.replace(
                    'YYYYMMDD', f'{year}{month}{day}')
                try:
                    # download the file
                    download_dir = os.path.join(SST_DIR, str(year), f'{month}/{day}')
                    if os.path.exists(download_dir):
                        if not len(os.listdir(download_dir)) > 0:
                            wget.download(url, out=download_dir)
                        else:
                            print(f'File already exists for {year}/{month}/{day}')
                    else:
                        os.makedirs(download_dir)
                        wget.download(url, out=download_dir)

                except Exception as e:
                    print('\n', e, "Error downloading file for", year, month, day, f'({day_num})\n')
                    print('url:', url)

    def get_year_urls(self):
        # get the html source
        self.driver.get(self.url)
        elems = self.driver.find_elements("xpath", "//a[@href]")
        for elem in elems:
            self.urls_yearly.append(elem.get_attribute("href"))

        # filter the urls to only include the years
        self.urls_yearly = [url for url in self.urls_yearly if url.split('/')[-2] in self.year_list]
        print(self.urls_yearly)

    def get_daily_url(self):
        days = [f'{x:03d}' for x in range(0, 366)]
        self.urls_daily = {}
        # go through each year and get the download url
        for url in self.urls_yearly:
            year_ = url.split('/')[-2]
            # extract each clickable link url
            self.driver.get(url)
            elems = self.driver.find_elements("xpath", "//a[@href]")
            # filter the links to only include the days
            elems = [elem for elem in elems if elem.get_attribute("href").split('/')[-2] in days]
            urls_daily = []
            for elem in elems:
                # get the href attribute
                href = elem.get_attribute("href")
                # add the url to the list
                urls_daily.append(href)
            self.urls_daily[year_] = urls_daily

    def download_file(self):
        # go through each year and get the download url
        for year_, urls_daily in tqdm(self.urls_daily.items(), 'Years'):
            for url in tqdm(urls_daily, 'Days'):
                day_ = url.split('/')[-2]
                download_dir = os.path.join(SST_DIR, year_, day_)
                if not os.path.exists(download_dir):
                    os.makedirs(download_dir)
                elif len(os.listdir(download_dir)) > 0:
                    print('File already exists for', year_, day_)
                    continue
                # go to the url
                self.driver.get(url)
                # get all the clickable links
                elems = self.driver.find_elements("xpath", "//a[@href]")
                hrefs = [elem.get_attribute("href") for elem in elems if elem.get_attribute("href").endswith('.nc')]
                # open the url
                self.driver.get(hrefs[0])
                # get all the clickable links
                elems = self.driver.find_elements("xpath", "//a[@href]")
                hrefs = [elem.get_attribute("href") for elem in elems if 'fileServer' in elem.get_attribute("href")]
                # go to the download url
                # download file from the link in the given address using wget
                wget.download(hrefs[0], out=download_dir)

        download_dir = os.path.join(SST_DIR, year_, day_)

        # go to the url
        self.driver.get(url)
        # get all the clickable links
        elems = self.driver.find_elements("xpath", "//a[@href]")
        # get the download url with fileserver in it
        elems = [elem for elem in elems if 'fileServer' in elem.get_attribute("href")]
        # go to the download url
        # download file from the link in the given address using wget
        wget.download(elems[0].get_attribute("href"), out=download_dir)


if __name__ == '__main__':
    scrp = SSTScraper(use_template=True)
    # scrp.get_year_urls()
    # scrp.get_daily_url()
    # scrp.download_file()
