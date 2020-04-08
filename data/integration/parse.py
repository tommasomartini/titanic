import logging
import re
from datetime import datetime

_logger = logging.getLogger(__name__)


_ticket_pattern = \
    re.compile(r'Ticket No\. (?P<ticket_no>.*?), (?P<ticket_price>.*)')
_embarking_cities = ['Southampton', 'Cherbourg', 'Queenstown']


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


def _parse_birth_date(summary_box):
    try:
        birth_date_span = \
            summary_box.find('span', attrs={'itemprop': 'birthDate'})
        birth_date_str = \
            birth_date_span.get('content', birth_date_span.getText())
        try:
            birth_date = datetime.strptime(birth_date_str, '%Y-%m-%d')
        except ValueError:
            try:
                birth_date = datetime.strptime(birth_date_str, '%Y')
            except ValueError:
                birth_date = datetime.strptime(birth_date_str, '%B %Y')
    except AttributeError:
        birth_date = None
    return birth_date


def _parse_birth_place(summary_box):
    try:
        birth_place = summary_box \
            .find('span', attrs={'itemprop': 'birthPlace'}) \
            .get('content')
    except AttributeError:
        birth_place = None
    return birth_place


def _parse_gender(summary_box):
    try:
        gender = summary_box \
            .find('span', attrs={'itemprop': 'gender'}) \
            .getText()
    except AttributeError:
        gender = None
    return gender


def _parse_marital_status(summary_box):
    try:
        marital_status = summary_box \
            .find('a', attrs={'title': 'List of unmarried Titanic '
                                       'passengers and crew'}) \
            .getText()
    except AttributeError:
        marital_status = None
    return marital_status


def _parse_residence(summary_box):
    try:
        residence = summary_box \
            .find('span', attrs={'itemprop': 'homeLocation'}) \
            .getText()
    except AttributeError:
        residence = None
    return residence


def _parse_occupation(summary_box):
    try:
        job = summary_box.find('span', attrs={'itemprop': 'jobTitle'}).getText()
    except AttributeError:
        job = None
    return job


def _parse_embark_port(summary_box):
    try:
        for city_name in _embarking_cities:
            emb_title = 'Titanic passengers and ' \
                        'crew that embarked at {}'.format(city_name)
            embarked_item = summary_box.find('a', attrs={'title': emb_title})
            if embarked_item:
                return city_name
    except AttributeError:
        return None

    return None


def _parse_ticket_info_and_nationality(summary_box):
    ticket_no = None
    ticket_price = None
    nationality = None
    for div in summary_box.find_all('div', recursive=False):
        for strong in div.find_all('strong'):
            if strong.getText() == 'Ticket No':
                ticket_match = _ticket_pattern.match(div.getText().strip())
                if ticket_match:
                    ticket_no = ticket_match.group('ticket_no')
                    ticket_price = ticket_match.group('ticket_price')

            elif strong.getText() == 'Nationality':
                nationality = div.getText().strip().split()[-1]

    return ticket_no, ticket_price, nationality


def _parse_relationships(summary_box):
    relationships = []
    for relationship_div in \
            summary_box.find_all('div', attrs={'itemprop': 'knows'}):
        related_person_id = relationship_div \
            .find('a', attrs={'itemprop': 'url'}) \
            .get('href')
        relationships.append(related_person_id)

    return relationships


def parse_passenger_summary_box(summary_box):
    passenger_info = dict()

    passenger_info['birth_date'] = _parse_birth_date(summary_box)
    passenger_info['birth_place'] = _parse_birth_place(summary_box)
    passenger_info['gender'] = _parse_gender(summary_box)
    passenger_info['marital_status'] = _parse_marital_status(summary_box)
    passenger_info['residence'] = _parse_residence(summary_box)
    passenger_info['job'] = _parse_occupation(summary_box)
    passenger_info['embarked'] = _parse_embark_port(summary_box)

    ticket_no, ticket_price, nationality = \
        _parse_ticket_info_and_nationality(summary_box)
    passenger_info['ticket_no'] = ticket_no
    passenger_info['ticket_price'] = ticket_price
    passenger_info['nationality'] = nationality

    passenger_info['relationships'] = _parse_relationships(summary_box)

    return passenger_info


def parse_passenger_row(passenger_row):
    last_name_allcaps = passenger_row \
        .find('span', attrs={'itemprop': 'familyName'}) \
        .getText()
    last_name = last_name_allcaps.lower().capitalize()

    title = passenger_row \
        .find('span', attrs={'itemprop': 'honorificPrefix'}) \
        .getText()

    first_name = passenger_row \
        .find('span', attrs={'itemprop': 'givenName'}) \
        .getText()

    # Each passenger row contains 3 td items: name, age, class, picture.
    name_td, age_td, class_td, _ = passenger_row.find_all('td')

    try:
        age_str = age_td.find('a').getText()
    except AttributeError:
        age_str = age_td.getText()

    age = _age_from_string(age_str.strip())

    pclass = class_td.find('span').getText()

    passenger_page_url_postfix = name_td \
        .find('a', attrs={'itemprop': 'url'}) \
        .get('href')

    passenger_info = dict(first_name=first_name,
                          title=title,
                          last_name=last_name,
                          age=age,
                          pclass=pclass,
                          url_id=passenger_page_url_postfix)

    return passenger_info
