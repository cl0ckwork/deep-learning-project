import pandas as pd
import numpy as np
import os
from multiprocessing import Pool
from itertools import repeat


# custom modules
from lib.db import connect
from lib.enums import ACQUISITION_RAW_COLUMN_NAMES, PERFORMANCE_RAW_COLUMN_NAMES


def group(iterator, count):
    while True:
        yield tuple([next(iterator) for i in range(count)])


def read_in_data(file_path=None):
    names = ACQUISITION_RAW_COLUMN_NAMES if 'Acquisition' in file_path else PERFORMANCE_RAW_COLUMN_NAMES
    return pd.read_table('../raw_data/{}'.format(file_path), sep='|', names=names, chunksize=5000)


def collect_raw_data_from_dirs():
    for r, d, f in os.walk("../raw_data"):
        if not d:
            iterators = []
            for file in f:
                file_path = '{}/{}'.format(r, file)
                print('READING: {}'.format(file_path))
                iterators.append(read_in_data(file_path))
            yield iterators


def cyclical_month_encode(col):
    return np.sin((col - 1) * (2. * np.pi / 12)), np.cos((col - 1) * (2. * np.pi / 12))


def cyclical_hour_encode(col):
    return np.sin(col * (2. * np.pi / 24)), np.cos(col * (2. * np.pi / 24))


def set_cyclical_my(field_name, df, outname):
    month, year = df[field_name].fillna('/').str.split('/').str
    df['{}_month_sin'.format(outname)], \
    df['{}_month_cos'.format(outname)] = cyclical_month_encode(month.replace('', 0).fillna(0).astype(int))
    df['{}_year'.format(outname)] = year.replace('', 0).fillna(0).astype(int)
    return df


def set_cyclical_mdy(field_name, df, outname):
    month, day, year = df[field_name].fillna('//').str.split('/').str
    df['{}_month_sin'.format(outname)], \
    df['{}_month_cos'.format(outname)] = cyclical_month_encode(month.replace('', 0).fillna(0).astype(int))
    df['{}_year'.format(outname)] = year.replace('', 0).fillna(0).astype(int)
    return df


def transform_columns(df, table):
    if 'acquisition' in table:
        set_cyclical_my('origination_date_string', df, 'origination')
        set_cyclical_my('first_payment_date_string', df, 'first_payment')
        df.relocation_mortgage_indicator = df.relocation_mortgage_indicator.eq('Y').mul(1)

    elif 'performance' in table:
        set_cyclical_mdy('monthly_reporting_period', df, 'monthly_reporting')
        set_cyclical_mdy('last_paid_installment_date_string', df, 'last_paid_installment')
        set_cyclical_mdy('foreclosure_date_string', df, 'foreclosure')
        set_cyclical_mdy('disposition_date_string', df, 'disposition')
        set_cyclical_my('maturity_date_string', df, 'maturity')
        set_cyclical_my('zero_balance_effective_date_string', df, 'zero_balance_effective')

        df.modification_flag = df.modification_flag.eq('Y').mul(1)
        df.repurchase_make_whole_proceeds_flag = df.repurchase_make_whole_proceeds_flag.eq('Y').mul(1)
        df.servicing_activity_indicator = df.servicing_activity_indicator.eq('Y').mul(1)
    return df


def to_sql(df, table):
    # transform_columns(df, table)
    df.to_sql(name=table, con=conn, index=False, if_exists='append')
    return df.shape[0]


def iterate_and_load(iterator, table=None):
    print('LOADING: {}'.format(table))
    count = 0
    # sample = next(iterator)
    # print(sample.sample(3))
    # for df in iterator:
    #     count += to_sql(df, table)

    for dfs in group(iterator, 4):
        # for r in pool.imap(transform_columns, zip(dfs, repeat(table))):
        for df in pool.starmap(transform_columns,  zip(dfs, repeat(table))):
            count += to_sql(df, table)

            # count += r
        print('LOADED: #{}'.format(count))
    print('LOADED TOTAL: {}'.format(count))


def main():
    for acq, perf in collect_raw_data_from_dirs():
        # iterate_and_load(acq, 'acquisition')
        iterate_and_load(perf, 'performance')


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)  # or 1000
    pool = Pool(processes=4)
    conn = connect(local=True, pool_size=5)
    main()
    # conn.close()
