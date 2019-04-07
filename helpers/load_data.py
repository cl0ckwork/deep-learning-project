import pandas as pd
import os

pd.set_option('display.max_columns', None)  # or 1000

# custom modules
from lib.db import connect
from lib.enums import ACQUISITION_RAW_COLUMN_NAMES, PERFORMANCE_RAW_COLUMN_NAMES

conn = connect(local=True)
# curr = conn.cursor()

def read_in_data(file_path=None):
    names = ACQUISITION_RAW_COLUMN_NAMES if 'Acquisition' in file_path else PERFORMANCE_RAW_COLUMN_NAMES
    return pd.read_table('../raw_data/{}'.format(file_path), sep='|', names=names, chunksize=5000)


def collect_raw_data_from_dirs():
    for r, d, f in os.walk("../raw_data"):
        if not d:
            iterators = []
            for file in f:
                file_path = '{}/{}'.format(r, file)
                print('READING: {}' .format(file_path))
                iterators.append(read_in_data(file_path))
            yield iterators

def iterate_and_load(iterator, table=None):
    print('LOADING: {}'.format(table))
    count = 0
    # sample = next(iterator)
    # print(sample.sample(3))
    for df in iterator:
        count += df.shape[0]
        print(count)
        df.to_sql(name=table, con=conn, index=False, if_exists='append')
    print('\nLOADED: #{}'.format(count))

def main():
    for acq, perf in collect_raw_data_from_dirs():
        # iterate_and_load(acq, 'acquisition')
        iterate_and_load(perf, 'performance')


if __name__ == '__main__':
    main()
