"""Collection of functions to parse the website www.encyclopedia-titanica.org
to collect extra data.
"""

import logging

_logger = logging.getLogger(__name__)

_embarking_cities = ['Southampton', 'Cherbourg', 'Queenstown']


def _parse_birth_date(summary_box):
    try:
        birth_date_span = \
            summary_box.find('span', attrs={'itemprop': 'birthDate'})
        return birth_date_span.get('content', birth_date_span.getText())
    except AttributeError:
        return None


def _parse_birth_place(summary_box):
    try:
        return summary_box \
            .find('span', attrs={'itemprop': 'birthPlace'}) \
            .get('content')
    except AttributeError:
        return None


def _parse_gender(summary_box):
    try:
        return summary_box \
            .find('span', attrs={'itemprop': 'gender'}) \
            .getText()
    except AttributeError:
        return None


def _parse_marital_status(summary_box):
    # Try to get both the states, then OR them to keep the positive one.
    marital_status_married = summary_box \
        .find('a', attrs={'title': 'List of married '
                                   'Titanic passengers and crew'})
    marital_status_single = summary_box \
        .find('a', attrs={'title': 'List of unmarried '
                                   'Titanic passengers and crew'})

    marital_status = marital_status_married or marital_status_single

    if not marital_status:
        return None

    return marital_status.getText()


def _parse_residence(summary_box):
    try:
        return summary_box \
            .find('span', attrs={'itemprop': 'homeLocation'}) \
            .getText()
    except AttributeError:
        return None


def _parse_occupation(summary_box):
    try:
        return summary_box \
            .find('span', attrs={'itemprop': 'jobTitle'}) \
            .getText()
    except AttributeError:
        return None


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


def _parse_nationality(summary_box):
    try:
        return summary_box \
            .find('span', attrs={'itemprop': 'nationality'}) \
            .getText()
    except AttributeError:
        return None


def _parse_ticket(summary_box):
    try:
        return summary_box.find('strong', text='Ticket No').parent.getText()
    except AttributeError:
        return None


def _parse_relationships(summary_box):
    relationships = []
    for relationship_div in \
            summary_box.find_all('div', attrs={'itemprop': 'knows'}):
        related_person_id = relationship_div \
            .find('a', attrs={'itemprop': 'url'}) \
            .get('href')
        relationships.append(related_person_id)

    return relationships


def _parse_cabin(summary_box):
    try:
        return summary_box.find('strong', text='Cabin No.').parent.getText()
    except AttributeError:
        return None


def _parse_destination(summary_box):
    try:
        return summary_box.find('strong', text='Destination').parent.getText()
    except AttributeError:
        return None


def parse_passenger_summary_box(summary_box):
    passenger_info = dict()

    passenger_info['BirthDate'] = _parse_birth_date(summary_box)
    passenger_info['BirthPlace'] = _parse_birth_place(summary_box)
    passenger_info['Sex'] = _parse_gender(summary_box)
    passenger_info['MaritalStatus'] = _parse_marital_status(summary_box)
    passenger_info['Residence'] = _parse_residence(summary_box)
    passenger_info['Job'] = _parse_occupation(summary_box)
    passenger_info['Embarked'] = _parse_embark_port(summary_box)
    passenger_info['Ticket'] = _parse_ticket(summary_box)
    passenger_info['Nationality'] = _parse_nationality(summary_box)
    passenger_info['Relationships'] = _parse_relationships(summary_box)
    passenger_info['Cabin'] = _parse_cabin(summary_box)
    passenger_info['Destination'] = _parse_destination(summary_box)

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

    pclass = class_td.find('span').getText()

    passenger_page_url_postfix = name_td \
        .find('a', attrs={'itemprop': 'url'}) \
        .get('href')

    passenger_info = dict(FirstName=first_name,
                          Title=title,
                          LastName=last_name,
                          Age=age_str,
                          Pclass=pclass,
                          UrlId=passenger_page_url_postfix)

    return passenger_info
