"""This script crawls data from:
    www.encyclopedia-titanica.org
"""
import logging
import os
import urllib.request
from multiprocessing import Pool
from urllib.parse import urljoin

import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

import data.integration.parse as parse

logging.basicConfig(format='[{levelname}][{name}] {message}',
                    style='{',
                    level=logging.INFO)
_logger = logging.getLogger(__name__)


_BASE_URL = 'https://www.encyclopedia-titanica.org'
_VICTIMS_SUBURL = 'titanic-victims'
_SURVIVORS_SUBURL = 'titanic-survivors'
_OUTPUT_PATH = os.path.join(os.environ['HOME'],
                            'kaggle',
                            'titanic',
                            'data',
                            'extra_data.csv')


def _parse_passenger_page(passenger_dict):
    url = urljoin(_BASE_URL, passenger_dict['UrlId'])
    response = urllib.request.urlopen(url)
    web_content = response.read().decode('utf-8')
    soup = BeautifulSoup(str(web_content), 'html.parser')
    summary_box = \
        soup.find('div', attrs={'itemtype': 'http://schema.org/Person'})
    extra_info_dict = parse.parse_passenger_summary_box(summary_box)
    passenger_dict.update(extra_info_dict)
    return passenger_dict


def _parse_page(url, title=None):
    response = urllib.request.urlopen(url)
    web_content = response.read().decode('utf-8')
    soup = BeautifulSoup(str(web_content), 'html.parser')

    passengers = []

    # Get all the "person" items.
    for passenger_row in tqdm(soup.find_all('tr'),
                              desc=title or 'Collecting passengers'):
        if passenger_row.get('itemtype') != 'http://schema.org/Person':
            # Not a person.
            continue

        passenger_dict = parse.parse_passenger_row(passenger_row)
        passengers.append(passenger_dict)

    return passengers


def _main():
    victims_url = urljoin(_BASE_URL, _VICTIMS_SUBURL)
    victims = _parse_page(victims_url, title='Collect victims')
    for passenger_dict in victims:
        passenger_dict['survived'] = 0.

    survivors_url = urljoin(_BASE_URL, _SURVIVORS_SUBURL)
    survivors = _parse_page(survivors_url, title='Collect survivors')
    for passenger_dict in survivors:
        passenger_dict['survived'] = 1.

    all_passengers = victims + survivors
    _logger.info('Collected {} passengers'.format(len(all_passengers)))

    pool = Pool()
    passengers = list(tqdm(pool.imap_unordered(_parse_passenger_page,
                                               all_passengers[:100]),
                           desc='Parse passengers pages',
                           total=len(all_passengers)))

    df = pd.DataFrame(passengers)
    df.to_csv(_OUTPUT_PATH)


if __name__ == '__main__':
    _main()
