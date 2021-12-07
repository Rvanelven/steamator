import sys
import re
import csv
import requests
from os import path
from bs4 import BeautifulSoup


def scrape_from_internet(ID):
    ''' Use `requests` to get the HTML page of search results for given steam ID '''
    url = f'https://steamspy.com/app/{ID}'
    response = requests.get(url).text
    return response
