import json
import logging
import os
import urllib.request
from multiprocessing import Pool
from urllib.error import URLError
from urllib.parse import urljoin

import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

import dataset as ds

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
                            'relationships_data.json')


def _parse_relationships(passenger_page, url):
    relationships = []

    linked_bios_text = \
        passenger_page.find('strong', text='Linked Biographies') \
        or passenger_page.find('strong', text='Linked Biography')
    if linked_bios_text is None:
        return relationships

    linked_bios_div = linked_bios_text.parent.parent
    all_bios = linked_bios_div.find_all(
        'div', attrs=dict(itemtype='http://schema.org/Person'))

    for bio in all_bios:
        relationship_type = bio.get('itemprop')
        if relationship_type is None:
            # Sometimes we seem to end up in the Summary box div. No idea why.
            continue

        relationship_type = relationship_type.lower() \
            .replace("'", "").replace('"', '')

        person_id = bio.find('a', attrs=dict(itemprop='url'))
        person_id = \
            person_id.get('href').lower().replace("'", "").replace('"', '')

        description = bio.find('small')
        if description is not None:
            description = description.get_text() \
                .lower().replace("'", "").replace('"', '')

        relationships.append((person_id, relationship_type, description))

    return relationships


def _parse_passenger_page(passenger_dict):
    url = urljoin(_BASE_URL, passenger_dict['UrlId'])
    try:
        response = urllib.request.urlopen(url)
        web_content = response.read().decode('utf-8')
        passenger_page = BeautifulSoup(str(web_content), 'html.parser')
        relationship_list = _parse_relationships(passenger_page, url)
        passenger_dict['Relationships'] = relationship_list
        return passenger_dict

    except TimeoutError:
        return _parse_passenger_page(passenger_dict)

    except URLError:
        return _parse_passenger_page(passenger_dict)


def _main():
    training_df, _ = ds.load_training_set()
    test_df = ds.load_test_set()
    df = pd.merge(training_df.reset_index(), test_df.reset_index(),
                  how='outer').set_index(ds.ID_COLUMN_NAME)
    url_ids = df.UrlId.tolist()

    all_passengers = [dict(UrlId=url_id) for url_id in url_ids]

    pool = Pool(processes=None)
    passengers = list(tqdm(pool.imap_unordered(_parse_passenger_page,
                                               all_passengers),
                           desc='Parse passengers pages',
                           total=len(all_passengers)))

    with open(_OUTPUT_PATH, 'w') as f:
        json.dump(passengers, f, indent=2)


if __name__ == '__main__':
    _main()
