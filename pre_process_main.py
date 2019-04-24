import pandas as pd
import numpy as np
import argparse
from os import environ, listdir

# custom modules
from lib.db import connect
from lib.helpers.pre_process import PreProcessors
from lib.enums import PRE_PROCESSING_ENCODERS_PICKLE_PATH, LIVE_PRE_PROCESSING_ENCODERS_PICKLE_PATH, EXCLUDED_CATEGORY_COLUMNS
from lib.helpers.load_data import main as load_data


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def remove_hidden(cls):
    return [m for m in dir(cls) if not m.startswith('_') and not m.endswith('_')]


def check_pre_processor(pp=None, path=None):
    if not pp:
        test = PreProcessors()
        test.load(path)
    else:
        test = pp
    print('\n *** CHECKING PRE PROCESSORS *** \n')
    print('PreProcessor:', remove_hidden(test))
    for name, encoder in test.encoders.items():
        print(' --- ')
        print('Encoder: {}'.format(name))
        print('\t columns:', encoder.columns)
        print('\t columns length:', len(encoder.columns))
        print('\t encoder methods:', remove_hidden(encoder.encoder))
    print('\n ***  END CHECK *** \n')


def main(conn=None, dry_run=False, load=False, save=True, pp_data=True, pp_cats=True, live=True):
    pp = PreProcessors(
        conn=conn,
        excluded_columns=EXCLUDED_CATEGORY_COLUMNS,
        target_column='sdq'
    )

    if load:
        pp.load(LIVE_PRE_PROCESSING_ENCODERS_PICKLE_PATH if live else PRE_PROCESSING_ENCODERS_PICKLE_PATH)

    if pp_data:
        load_data(path='raw_data', conn=conn, pre_processor=pp, dry_run=dry_run)

    if pp_cats:
        print('ENCODING: categorical features')
        pp.encode_categorical_columns('acquisition')
        # pp.encode_categorical_columns('performance')
        # pp.encode_target_column('performance')
    if save:
        pp.save(LIVE_PRE_PROCESSING_ENCODERS_PICKLE_PATH if live else PRE_PROCESSING_ENCODERS_PICKLE_PATH)
    return pp


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pre-process and load data.')
    parser.add_argument("--run",
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=True,
                        help="run the loading process?"
                        )
    parser.add_argument("--dry_run",
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=False,
                        help="insert data into database?"
                        )
    parser.add_argument("--local",
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=False,
                        help="use local database?"
                        )
    parser.add_argument("--check",
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=False,
                        help="check pre-processors for encoders?"
                        )
    parser.add_argument("--save",
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=False,
                        help="save pre-processors as pickle?"
                        )
    parser.add_argument("--load",
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=False,
                        help="load pre-processors from pickle before processing?"
                        )
    parser.add_argument("--pp_data",
                        type=str2bool,
                        nargs='?',
                        const=True,
                        help="load data and pre-processor numerical information?"
                        )
    parser.add_argument("--pp_cats",
                        type=str2bool,
                        nargs='?',
                        const=True,
                        help="pre-process categorical data?"
                        )
    parser.add_argument("--host",
                        nargs='?',
                        default=environ.get('DB_HOST'),
                        help="Database host to connect to, default: {} [environ.DB_HOST]".format(environ.get('DB_HOST'))
                        )
    parser.add_argument("--pickle",
                        nargs='?',
                        default=PRE_PROCESSING_ENCODERS_PICKLE_PATH,
                        help="check a pickle by path"
                        )
    parser.add_argument("--live",
                        type=str2bool,
                        nargs='?',
                        const=True,
                        help="save encoders as live?"
                        )

    args = parser.parse_args()
    print(args)
    np.seterr(divide='ignore', invalid='ignore')
    pd.set_option('display.max_columns', None)  # or 1000

    connection = connect(local=args.local, host=args.host or ('localhost' if args.local else args.host))
    # connection = connect(local=True, host='localhost')

    manual = dict(
        conn=connection,
        dry_run=False,
        save=False,
        load=False,
        pp_data=True,
        pp_cats=True,
        live=False
    )

    if args.run:
        main(
            # **manual,
            conn=connection,
            dry_run=args.dry_run,
            save=args.save,
            load=args.load,
            pp_data=args.pp_data,
            pp_cats=args.pp_cats,
            live=args.live
        )

    print('pickles:', listdir('pickles'))
    if args.check:
        try:
            check_pre_processor(path=args.pickle or LIVE_PRE_PROCESSING_ENCODERS_PICKLE_PATH if args.live else PRE_PROCESSING_ENCODERS_PICKLE_PATH)
        except Exception as ex:
            print('ERROR During check:', repr(ex))
