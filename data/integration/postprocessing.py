"""Collection of functions to format the extra data downloaded from
    www.encyclopedia-titanica.org
"""

import datetime as dt
import json
import os

import numpy as np
import pandas as pd

# The date of the sinking.
_sinking_date = dt.date(year=1912, month=4, day=15)

# List of UrlId of passengers embarked in Belfast.
_belfast_passengers_path = os.path.join(os.environ['HOME'],
                                        'kaggle',
                                        'titanic',
                                        'data',
                                        'belfast_passengers.json')


def manually_fix_titles(df):
    # Fix missing titles.
    df.loc[df['LastName'] == 'Banfi', ['Title']] = 'Mr'
    df.loc[df['FirstName'] == 'Lucy Christiana, Lady',
           ['FirstName', 'Title']] = ['Lucy Christiana', 'Lady']
    df.loc[df['FirstName'] == 'Lucy Noël Martha, Countess of',
           ['FirstName', 'Title']] = ['Lucy Noël Martha', 'Countess']
    df.loc[df['LastName'] == 'Walsh', ['Title']] = 'Msr'

    # Mr Eugene Joseph Abbott is 13 yo: he's probably a Master instead of a Mr.
    df.loc[df['UrlId'] == '/titanic-victim/eugene-joseph-abbott.html',
           ['Title']] = 'Master'

    # Sra. Asuncion Durán i Moné and Sra. Florentina Durán i Moné were both
    # single: let's make it clear by changing "Sra" with "Miss".
    df.loc[df['UrlId'].isin(['/titanic-survivor/asuncion-duran-y-more.html',
                             '/titanic-survivor/florentina-duran-y-more.html']),
           ['Title']] = 'Miss'

    # Mr Colonel (Oberst) Alfons Simonius-Blumer is a Colonel.
    df.loc[df['UrlId'] == '/titanic-survivor/alfons-simonius-blumer.html',
           ['Title']] = 'Colonel'

    return df


def extract_ticket_number_and_price(df):
    # Remove the initial "Ticket No." string and split number and price.
    df[['Ticket', 'TicketPrice']] = df['Ticket'] \
        .str.split('No. ', expand=True)[1] \
        .str.strip() \
        .str.split(', ', expand=True)

    # Extract the ticket number.
    ticket_number_pattern = r'^\D*(?P<TicketNumber>\d+)$'
    df = df.join(df['Ticket'] \
                 .str.strip()
                 .str.extract(ticket_number_pattern, expand=True)
                 .astype(float))

    # Extract the ticket price: first extract pounds, shillings and pence and
    # then sum them together:
    #  1 pound (£) = 20 shillings (s)
    #  1 shilling (s) = 12 pence (d)
    # This extraction requires the creation of 3 intermediate columns.
    ticket_price_pattern = r'(?:£(?P<Pounds>\d+))\s*' \
                           r'(?:(?P<Shillings>\d+)s)?\s*' \
                           r'(?:(?P<Pence>\d+)d)?'
    df = df.join(df['TicketPrice'] \
                 .str.extract(ticket_price_pattern, expand=True)
                 .astype(float))

    # Fill the nan with zeros. This should hold because the people without
    # ticket seem to be employees of the cruise, hence they actually paid £0.
    df[['Pounds', 'Shillings', 'Pence']] = \
        df[['Pounds', 'Shillings', 'Pence']].fillna(0)

    df['TicketPrice'] = df['Pounds'] \
                        + (1 / 20) * df['Shillings'] \
                        + (1 / 240) * df['Pence']

    # Drop the intermediate columns.
    df = df.drop(columns=['Pounds', 'Shillings', 'Pence'])

    return df


def _extract_city_region_country_from_location(df, field_name):
    suffixes = ['CityRegion', 'City', 'Region', 'Country']
    city_region, city, region, country = \
        ['{}{}'.format(field_name, suffix) for suffix in suffixes]

    df[[city_region, country]] = \
        df[field_name].str.strip().str.rsplit(', ', n=1, expand=True)
    df[[city, region]] = df[city_region].str.split(', ', n=1, expand=True)
    df = df.drop(columns=city_region)
    return df


def extract_birth_city_region_country(df):
    return _extract_city_region_country_from_location(df, 'BirthPlace')


def extract_residence_city_region_country(df):
    return _extract_city_region_country_from_location(df, 'Residence')


def extract_destination_city_region_country(df):
    df['Destination'] = df['Destination'] \
        .str.split(':', expand=True)[1] \
        .str.strip()
    return _extract_city_region_country_from_location(df, 'Destination')


def embarked_as_single_character(df):
    df[['Embarked']] = df['Embarked'].str.slice(stop=1)
    return df


def add_belfast_as_embarking_city(df):
    with open(_belfast_passengers_path, 'r') as f:
        belfast_passengers = json.load(f)
    df.loc[df['UrlId'].isin(belfast_passengers), ['Embarked']] = 'Belfast'
    return df


def manually_fill_missing_nationalities(df):
    # Only three passenger have a missing nationality and all of them sound
    # English by the name.
    df.loc[df['Nationality'].isna(), ['Nationality']] = 'English'
    return df


def manually_fill_missing_birth_dates(df):
    df.loc[(df['FirstName'] == 'Juho') & (df['LastName'] == 'Niskanen'),
           ['BirthDate']] = '1870'
    return df


def extract_birth_year(df):
    birth_year_pattern = r'.*(?P<BirthYear>\d{4}).*'
    df = df.join(df['BirthDate'] \
                 .str.extract(birth_year_pattern, expand=True))
    return df


def extract_cabin_deck(df):
    # This function also formats the Cabin column.
    df['Cabin'] = df['Cabin'].str.split(':', expand=True)[1].str.strip()
    df['CabinDeck'] = df['Cabin'].str.extract(r'^\W*(\w)', expand=True)

    # Passenger Benoît Picard does not have the deck indicated,
    # but it was probably deck F.
    df.loc[df['UrlId'] == '/titanic-survivor/berk-pickard.html',
           ['CabinDeck']] = 'F'

    # Some passengers are assigned deck "R", which is actually a location
    # in deck F.
    df.loc[df['CabinDeck'] == 'R', ['CabinDeck']] = 'F'
    return df


def gender_to_lower_case(df):
    df['Sex'] = df['Sex'].str.lower()
    return df


def compute_age_in_days(df):
    # Explicitly set the missing ages to NaN.
    df.loc[df['BirthDate'].isna(), ['Age']] = np.nan

    # Convert the dates in the format "January 1912" to "1912-04".
    # We don't use directly the to_datetime function because we don't want
    # all the missing values to be set to 1, as it is by default.
    # We prefer to randomize the missing months and days.
    mask = ~(df['BirthDate'].str.extract(r'([A-Za-z])').isna())[0]
    numeral_dates = df[mask]['BirthDate'].apply(
        lambda d: dt.datetime.strptime(d, '%B %Y').strftime('%Y-%m'))
    df.loc[mask, 'BirthDate'] = numeral_dates

    # Split the birth date in year, month and day.
    df[['year', 'month', 'day']] = df['BirthDate'] \
        .str.split('-', expand=True) \
        .astype(float)

    # Randomize the missing months.
    nan_months = df.loc[df.month.isna(), ['month']]
    nan_months['month'] = np.random.randint(1, 13, size=len(nan_months))
    df.loc[df.month.isna(), ['month']] = nan_months.astype(int)

    # Randomize the missing days.
    nan_days = df.loc[df.day.isna(), ['day']]
    nan_days['day'] = np.random.randint(1, 29, size=len(nan_days))
    df.loc[df.day.isna(), ['day']] = nan_days.astype(int)

    df['AgeInDays'] = pd.to_datetime(df[['year', 'month', 'day']]).apply(
        lambda d: (pd.Timestamp(_sinking_date) - d).days)

    # Get rid of the utility columns.
    df = df.drop(columns=['year', 'month', 'day'])

    return df


def assign_id(df):
    df['PassengerId'] = df.index
    return df
