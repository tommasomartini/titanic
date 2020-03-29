import os
import time


_project_path = os.path.join(os.environ['HOME'], 'kaggle', 'titanic')
_output_dir = os.path.join(_project_path, 'submissions')


def _get_filename(notes):
    curr_time = time.strftime('%Y%m%d_%H%M')
    if notes:
        return '_'.join((curr_time, notes))

    return curr_time


def output_submission_file(X_testset_df, notes=None):
    filename = '{}.csv'.format(_get_filename(notes))
    output_path = os.path.join(_output_dir, filename)
    X_testset_df.reset_index().to_csv(output_path,
                                      columns=['PassengerId', 'Survived'],
                                      index=False)
