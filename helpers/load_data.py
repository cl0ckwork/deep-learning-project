import pandas as pd
import numpy as np
import os
from multiprocessing import Pool
from sqlalchemy.exc import IntegrityError
from sqlalchemy import event
# from itertools import repeat

# custom modules
from lib.db import connect
from lib.enums import ACQUISITION_RAW_COLUMN_NAMES, PERFORMANCE_RAW_COLUMN_NAMES
from helpers.pre_process import PreProcessors, PRE_PROCESSING_ENCODERS_PICKLE_PATH


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

        int_colunns = ['loan_id', 'original_upb', 'original_loan_term',
                       'original_loan_to_value', 'number_of_units', 'zip_code_short',
                       'relocation_mortgage_indicator', 'origination_year',
                       'first_payment_year']

        pp.encode_numerical_columns(table, df[int_colunns].drop(['loan_id'], axis=1).astype('float64'))
        # categories = df.select_dtypes(include='object')
        # pp.encode_categorical_columns(table, [(name, categories[name]) for name in categories])

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

        ints = df.select_dtypes(include=['int'])
        print(ints.columns)
        pp.encode_numerical_columns(table, ints.drop(['loan_id'], axis=1).astype('float64'))

    return df


def to_sql(df, table):
    try:
        df.to_sql(name=table, con=conn, index=False, if_exists='append')
    except IntegrityError:
        print('rows already exist')
        return 0
    return df.shape[0]


def iterate_and_load(iterator, table=None):
    print('LOADING: {}'.format(table))
    count = 0
    for df in iterator:
        transform_columns(df, table)
        count += df.shape[0]
        break
        # count += to_sql(df, table)

        # for dfs in group(iterator, 4):
        #     for df in pool.starmap(transform_columns, zip(dfs, repeat(table))):
        # count += to_sql(df, table)
        # count += df.shape[0]
        # print('LOADED: #{}'.format(count))
    print('LOADED TOTAL: {}'.format(count))


def main():
    for acq, perf in collect_raw_data_from_dirs():
        iterate_and_load(acq, 'acquisition')
        iterate_and_load(perf, 'performance')



# @event.listens_for(conn, 'before_cursor_execute')
# def receive_before_cursor_execute(conn, cursor, statement, params, context, executemany):
#     if executemany:
#         cursor.fast_executemany = True


if __name__ == '__main__':
    conn = connect(local=True)
    pd.set_option('display.max_columns', None)  # or 1000
    # pool = Pool(processes=4)

    pp = PreProcessors()
    main()
    pp.save(PRE_PROCESSING_ENCODERS_PICKLE_PATH)

    # test encoders
    # a = load_pickle('../pickles/pre_processors.pkl')
    # p = load_encoder('../pickles/performance_encoder.pkl')

    pp = PreProcessors()
    pp.load(PRE_PROCESSING_ENCODERS_PICKLE_PATH)
    print(dir(pp))
    print(pp.encoders)
    # print(a.original_loan_to_value.mean_)
    #
    # print(dir(p))
    # print(p.loan_age.mean_)

    # conn.close()
