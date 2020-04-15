def manually_fix_missing_titles(df):
    df.loc[df['LastName'] == 'Banfi', ['Title']] = 'Mr'
    df.loc[df['FirstName'] == 'Lucy Christiana, Lady',
           ['FirstName', 'Title']] = ['Lucy Christiana', 'Lady']
    df.loc[df['FirstName'] == 'Lucy Noël Martha, Countess of',
           ['FirstName', 'Title']] = ['Lucy Noël Martha', 'Countess']
    df.loc[df['LastName'] == 'Walsh', ['Title']] = 'Msr'

    return df


def extract_ticket_number_and_price(df):
    # Remove the initial "Ticket No." string and split number and price.
    df[['Ticket', 'TicketPrice']] = df['Ticket'] \
        .str.split('No. ', expand=True)[1]\
        .str.strip()\
        .str.split(', ', expand=True)

    # Extract the ticket number.
    ticket_number_pattern = r'^\D*(?P<TicketNumber>\d+)$'
    df = df.join(df['Ticket']\
                 .str.strip()
                 .str.extract(ticket_number_pattern, expand=True)
                 .astype(float))

    # Extract the ticket price: first extract pounds, shillings and pennies and
    # then sum them together:
    #  1 pound (£) = 12 shillings (s)
    #  1 shilling (s) = 20 pennies (d)
    # This extraction requires the creation of 3 intermediate columns.
    ticket_price_pattern = r'(?:£(?P<Pounds>\d+))\s*' \
                           r'(?:(?P<Shillings>\d+)s)?\s*' \
                           r'(?:(?P<Pennies>\d+)d)?'
    df = df.join(df['TicketPrice']\
                 .str.extract(ticket_price_pattern, expand=True)
                 .astype(float))

    # Fill the nan with zeros. This should hold because the people without
    # ticket seem to be employees of the cruise, hence they actually paid £0.
    df[['Pounds', 'Shillings', 'Pennies']] = \
        df[['Pounds', 'Shillings', 'Pennies']].fillna(0)

    df['TicketPrice'] = df['Pounds'] \
                        + (1 / 12) * df['Shillings'] \
                        + (1 / 240) * df['Pennies']

    # Drop the intermediate columns.
    df = df.drop(columns=['Pounds', 'Shillings', 'Pennies'])

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
    return _extract_city_region_country_from_location(df, 'Destination')


def embarked_as_single_character(df):
    df[['Embarked']] = df['Embarked'].str.slice(stop=1)
    return df


def deduce_missing_embarking_harbour(df):
    # We will assume that the people living in Southampton embarked
    # in that city.
    df.loc[(df['Embarked'].isna()) &
           (df['Residence'].str.contains('Southampton')),
           ['Embarked']] = 'Southampton'
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
    df = df.join(df['BirthDate']\
                 .str.extract(birth_year_pattern, expand=True))
    return df


def extract_cabin_deck(df):
    # This function also formats the Cabin column.
    df['Cabin'] = df['Cabin'].str.split(':', expand=True)[1].str.strip()
    df['CabinDeck'] = df['Cabin'].str.extract(r'^\W*(\w)', expand=True)
    return df


def gender_to_lower_case(df):
    df['Sex'] = df['Sex'].str.lower()
    return df
