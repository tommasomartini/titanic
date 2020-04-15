import data.integration.postprocessing as postpro
import pandas as pd
import os


_extra_data_filepath = os.path.join(os.environ['HOME'],
                                    'kaggle',
                                    'titanic',
                                    'data',
                                    'extra_data_8.csv')


def import_extra_data():
    df = pd.read_csv(_extra_data_filepath)
    return df


def apply_post_processing(df):
    post_processing_functions = [
        postpro.manually_fix_missing_titles,
        postpro.extract_ticket_number_and_price,
        postpro.extract_birth_city_region_country,
        postpro.extract_residence_city_region_country,
        postpro.extract_destination_city_region_country,
        postpro.deduce_missing_embarking_harbour,
        postpro.embarked_as_single_character,
        postpro.manually_fill_missing_nationalities,
        postpro.manually_fill_missing_birth_dates,
        postpro.extract_birth_year,
        postpro.extract_cabin_deck,
        postpro.gender_to_lower_case,
    ]

    for func in post_processing_functions:
        df = func(df)

    return df


def main():
    df = pd.read_csv(_extra_data_filepath)


if __name__ == '__main__':
    main()
