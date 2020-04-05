"""In this module we integrate and correct the data provided by Kaggle by
crawling the data from:
    www.encyclopedia-titanica.org
"""
import logging
import os
import urllib.request

import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

logging.basicConfig(format='[{levelname}][{name}] {message}',
                    style='{',
                    level=logging.DEBUG)
_logger = logging.getLogger(__name__)


_BASE_URL = 'https://www.encyclopedia-titanica.org'
_VICTIMS_SUBURL = 'titanic-victims'
_SURVIVORS_SUBURL = 'titanic-survivors'
_OUTPUT_PATH = os.path.join(os.environ['HOME'],
                            'kaggle',
                            'titanic',
                            'data',
                            'extra_data.csv')


class Passenger:

    def __init__(self,
                 first_name,
                 title,
                 last_name,
                 age,
                 pclass):
        self.first_name = first_name
        self.title = title
        self.last_name = last_name
        self.age = age
        self.pclass = pclass

        self.survived = 0.

    def __repr__(self):
        return '{}, {} {}; age {}; {}'.format(self.last_name,
                                              self.title,
                                              self.first_name,
                                              self.age,
                                              self.pclass)


def _build_url(base_url, suburl):
    return '/'.join((base_url, suburl))


def _age_from_string(age_str):
    if not age_str:
        return None

    if age_str[-1] == 'm':
        # Age is given in months.
        months = int(age_str[:-1])
        age = months / 12
        return age

    # Age given in years.
    return int(age_str)


def _parse_passenger_row(passenger_row):
    last_name_allcaps = \
        passenger_row.find('span', attrs={'itemprop': 'familyName'}).getText()
    last_name = last_name_allcaps.lower().capitalize()

    title = passenger_row.find('span', attrs={'itemprop': 'honorificPrefix'}) \
        .getText()

    first_name = \
        passenger_row.find('span', attrs={'itemprop': 'givenName'}).getText()

    # Each passenger row contains 3 td items: name, age, class, picture.
    _, age_td, class_td, _ = passenger_row.find_all('td')

    try:
        age_str = age_td.find('a').getText()
    except AttributeError:
        age_str = age_td.getText().strip()

    age = _age_from_string(age_str)

    pclass = class_td.find('span').getText()

    passenger = Passenger(first_name=first_name,
                          title=title,
                          last_name=last_name,
                          age=age,
                          pclass=pclass)
    return passenger


def _parse_page(url, title=None):
    # HTTP request of the page.
    response = urllib.request.urlopen(url)
    _logger.debug('Response code {} for URL: {}'.format(response.code, url))
    web_content = response.read().decode('utf-8')

    # Parse the page content.
    soup = BeautifulSoup(str(web_content), 'html.parser')

    passengers = []

    # Get all the "person" items.
    for passenger_row in tqdm(soup.find_all('tr'),
                              desc=title or 'Parsing passengers'):
        if passenger_row.get('itemtype') != 'http://schema.org/Person':
            # Not a person.
            continue

        passenger = _parse_passenger_row(passenger_row)
        passengers.append(passenger)

    return passengers


def _export_to_dataframe(passengers):
    data = {
        'Survived': [p.survived for p in passengers],
        'LastName': [p.last_name for p in passengers],
        'FirstName': [p.first_name for p in passengers],
        'Title': [p.title for p in passengers],
        'Age': [p.age for p in passengers],
        'Pclass': [p.pclass for p in passengers],
    }
    df = pd.DataFrame(data)
    return df


def _main():
    victims_url, survivors_url = map(lambda suburl: _build_url(_BASE_URL,
                                                               suburl),
                                     (_VICTIMS_SUBURL, _SURVIVORS_SUBURL))
    victims = _parse_page(victims_url, title='Parsing victims')
    survivors = _parse_page(survivors_url, title='Parsing survivors')

    for passenger in survivors:
        passenger.survived = 1.

    all_passengers = victims + survivors
    df = _export_to_dataframe(all_passengers)
    df.to_csv(_OUTPUT_PATH)


if __name__ == '__main__':
    _main()
