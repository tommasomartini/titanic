"""In this module we integrate and correct the data provided by Kaggle by
crawling the data from:
    www.encyclopedia-titanica.org
"""
import logging
import os
import re
import urllib.request
from copy import deepcopy
from datetime import datetime

import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

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

_ticket_pattern = \
    re.compile(r'Ticket No\. (?P<ticket_no>.*?), (?P<ticket_price>.*)')


class Passenger:

    def __init__(self,
                 first_name,
                 title,
                 last_name,
                 age,
                 pclass,
                 birth_date,
                 gender,
                 marital_status,
                 birth_place,
                 ticket_no,
                 residence,
                 job,
                 embarked,
                 nationality,
                 ticket_price,
                 url_id,
                 relationships):
        self.relationships = relationships
        self.url_id = url_id
        self.ticket_price = ticket_price
        self.nationality = nationality
        self.birth_place = birth_place
        self.ticket_no = ticket_no
        self.residence = residence
        self.job = job
        self.embarked = embarked
        self.gender = gender
        self.marital_status = marital_status
        self.first_name = first_name
        self.title = title
        self.last_name = last_name
        self.age = age
        self.pclass = pclass
        self.birth_date = birth_date

        self.survived = 0.

    def __repr__(self):
        repr_str = '{}, {} {}'.format(self.last_name,
                                      self.title,
                                      self.first_name)

        all_attributes = deepcopy(self.__dict__)
        del all_attributes['last_name']
        del all_attributes['first_name']
        del all_attributes['title']

        for attr_name, attr_value in sorted(all_attributes.items()):
            if type(attr_value) == datetime:
                attr_value = datetime.strftime(attr_value, '%Y-%m-%d')
            repr_str += '\n {}: {}'.format(attr_name, attr_value)

        return repr_str


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


def _parse_extra_info(passenger_info, info_box):
    extra_info = {}

    # Birth date.
    try:
        birth_date_span = info_box.find('span', attrs={'itemprop': 'birthDate'}).get('content')
        birth_date = datetime.strptime(birth_date_span, '%Y-%m-%d')
    except:
        birth_date = None
    extra_info['birth_date'] = birth_date

    # Birth place.
    try:
        birth_place = info_box.find('span', attrs={'itemprop': 'birthPlace'}).get('content')
    except:
        birth_place = None
    extra_info['birth_place'] = birth_place

    # Gender.
    try:
        gender = info_box.find('span', attrs={'itemprop': 'gender'}).getText()
    except:
        gender = None
    extra_info['gender'] = gender

    # Marital status.
    try:
        marital_status = info_box.find('a', attrs={'title': 'List of unmarried Titanic passengers and crew'}).getText()
    except:
        marital_status = None
    extra_info['marital_status'] = marital_status

    # Residence.
    try:
        residence = info_box.find('span', attrs={'itemprop': 'homeLocation'}).getText()
    except:
        residence = None
    extra_info['residence'] = residence

    # Occupation.
    try:
        job = info_box.find('span', attrs={'itemprop': 'jobTitle'}).getText()
    except:
        job = None
    extra_info['job'] = job

    # Embarked.
    embarking_cities = ['Southampton', 'Cherbourg', 'Queenstown']
    embarked = None
    for city_name in embarking_cities:
        emb_title = \
            'Titanic passengers and crew that embarked at {}'.format(city_name)
        embarked_item = info_box.find('a', attrs={'title': emb_title})
        if embarked_item:
            embarked = city_name
    extra_info['embarked'] = embarked

    # Ticket number.
    extra_info['ticket_no'] = None
    extra_info['ticket_price'] = None
    for div in info_box.find_all('div', recursive=False):
        for strong in div.find_all('strong'):
            try:
                if strong.getText() == 'Ticket No':
                    ticket_match = _ticket_pattern.match(div.getText().strip())
                    if ticket_match:
                        extra_info['ticket_no'] = \
                            ticket_match.group('ticket_no')
                        extra_info['ticket_price'] = \
                            ticket_match.group('ticket_price')

                if strong.getText() == 'Nationality':
                    extra_info['nationality'] = div.getText().strip().split()[-1]
            except:
                continue

    # Relationships.
    relationships = []
    for relationship_div in info_box.find_all('div', attrs={'itemprop': 'knows'}):
        related_person_id = \
            relationship_div.find('a', attrs={'itemprop': 'url'}).get('href')
        relationships.append(related_person_id)
    extra_info['relationships'] = relationships

    passenger_info.update(**extra_info)

    return passenger_info


def _parse_passenger_page(passenger_info, url):
    response = urllib.request.urlopen(url)
    _logger.debug('Response code {} for URL: {}'.format(response.code, url))
    web_content = response.read().decode('utf-8')
    soup = BeautifulSoup(str(web_content), 'html.parser')

    info_box = soup.find('div', attrs={'itemtype': 'http://schema.org/Person'})
    passenger_info = _parse_extra_info(passenger_info, info_box)

    return passenger_info


def _parse_passenger_row(passenger_row):
    last_name_allcaps = \
        passenger_row.find('span', attrs={'itemprop': 'familyName'}).getText()
    last_name = last_name_allcaps.lower().capitalize()

    title = passenger_row.find('span', attrs={'itemprop': 'honorificPrefix'}) \
        .getText()

    first_name = \
        passenger_row.find('span', attrs={'itemprop': 'givenName'}).getText()

    # Each passenger row contains 3 td items: name, age, class, picture.
    name_td, age_td, class_td, _ = passenger_row.find_all('td')

    try:
        age_str = age_td.find('a').getText()
    except AttributeError:
        age_str = age_td.getText().strip()

    age = _age_from_string(age_str)

    pclass = class_td.find('span').getText()

    passenger_page_url_postfix = \
        name_td.find('a', attrs={'itemprop': 'url'}).get('href')

    passenger_info = dict(first_name=first_name,
                          title=title,
                          last_name=last_name,
                          age=age,
                          pclass=pclass,
                          url_id=passenger_page_url_postfix)

    passenger_page_url = _build_url(_BASE_URL, passenger_page_url_postfix)
    passenger_info = _parse_passenger_page(passenger_info=passenger_info,
                                           url=passenger_page_url)

    passenger = Passenger(**passenger_info)

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
    survivors = []
    # survivors = _parse_page(survivors_url, title='Parsing survivors')

    for passenger in survivors:
        passenger.survived = 1.

    all_passengers = victims + survivors
    df = _export_to_dataframe(all_passengers)
    df.to_csv(_OUTPUT_PATH)


if __name__ == '__main__':
    _main()
