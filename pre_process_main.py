import pandas as pd
import numpy as np

# custom modules
from lib.db import connect
# from lib.enums import ACQUISITION_RAW_COLUMN_NAMES, PERFORMANCE_RAW_COLUMN_NAMES
from lib.helpers.pre_process import PreProcessors
from lib.enums import PRE_PROCESSING_ENCODERS_PICKLE_PATH, EXCLUDED_CATEGORY_COLUMNS
from lib.helpers.load_data import main as load_data


def remove_hidden(cls):
    return [m for m in dir(cls) if not m.startswith('_') and not m.endswith('_')]


def check_pre_processor():
    test = PreProcessors()
    test.load(PRE_PROCESSING_ENCODERS_PICKLE_PATH)
    print('\n *** CHECKING PRE PROCESSORS *** ')
    print('PreProcessor:', remove_hidden(test))
    for name, encoder in test.encoders.items():
        print('Encoder: {}'.format(name))
        print('\t columns:', encoder.columns)
        print('\t columns length:', len(encoder.columns))
        print('\t encoder methods:', remove_hidden(encoder.encoder))
    print(' ***  END CHECK *** \n')


def main(conn=None, dry_run=False):
    pp = PreProcessors(
        conn=conn,
        excluded_columns=EXCLUDED_CATEGORY_COLUMNS,
        target_column='current_loan_delinquency_status'
    )
    # pp.load(PRE_PROCESSING_ENCODERS_PICKLE_PATH)
    load_data(path='raw_data', conn=conn, pre_processor=pp, dry_run=dry_run)

    pp.encode_categorical_columns('acquisition')
    pp.encode_categorical_columns('performance')
    pp.encode_target_column('performance')
    pp.save(PRE_PROCESSING_ENCODERS_PICKLE_PATH)


if __name__ == '__main__':
    # connection = connect(local=True)
    connection = connect(local=False)
    np.seterr(divide='ignore', invalid='ignore')
    pd.set_option('display.max_columns', None)  # or 1000
    main(conn=connection)
    check_pre_processor()
