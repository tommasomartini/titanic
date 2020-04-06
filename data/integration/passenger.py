from copy import deepcopy
from datetime import datetime


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
                 relationships,
                 survived):
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
        self.survived = survived

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
