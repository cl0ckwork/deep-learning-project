from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
import numpy as np
import pickle
from os import listdir, environ
from lib.db import connect
from lib.enums import PRE_PROCESSING_ENCODERS_PICKLE_PATH

LOGGING = environ.get('LOGGING', False) == 'true'


def load_pickle(path):
    with open(path, "rb") as infile:
        loaded = pickle.load(infile)
    return loaded


class ActiveEncoder(object):
    __slots__ = ('encoder', 'columns')

    def __init__(self, encoder=None, columns=None):
        self.encoder = encoder
        self.columns = columns


class Encoding:
    def __init__(self, table=None):
        self.table = table

    def categorical_(self, name, categories):
        if not hasattr(self, name):
            ohe = OneHotEncoder(sparse=False, categories='auto')
            setattr(self, name, ohe)
        else:
            ohe = getattr(self, name)
        try:
            ohe.fit(categories)
            if LOGGING:
                print('OneHotEncoder:', name, ohe.categories_)
            return ohe
        except Exception as ex:
            print(ex)

    def numerical_(self, name, items):
        if not hasattr(self, name):
            ss = StandardScaler()
            setattr(self, name, ss)
        else:
            ss = getattr(self, name)

        ss.partial_fit(items)
        if LOGGING:
            print('StandardScaler:', name, ss.mean_, ss.var_)
        return ss

    def target_(self, name='target', items=None):
        if not hasattr(self, name):
            le = LabelEncoder()
            setattr(self, name, le)
        else:
            le = getattr(self, name)

        le.fit(items)
        if LOGGING:
            print('LabelEncoder:', name, le.classes_)
        return le

    def apply_encoding(self, name, categories):
        if not hasattr(self, name):
            raise Exception(
                'You did not create an encoder for {} categories, use the categorical() method first'.format(name))
        return self[name].transform(categories)

    def save(self, path=None):
        name = "{}/{}_encoder.pkl".format(path or '.', self.table)
        with open(name, "wb") as outfile:
            pickled = pickle.dump(self, outfile)
        # print("pickled: {} encoders @ {}".format(self.table, name))
        return pickled

    def load(self, path=None):
        name = "{}/{}_encoder.pkl".format(path or '.', self.table)
        return load_pickle(name)


class PreProcessors:
    __slots__ = ('_conn', 'encoders', '_excluded_columns', 'target_column')

    def __init__(self, conn=None, excluded_columns=None, target_column=None):
        self._conn = conn
        self.encoders = dict()
        self._excluded_columns = excluded_columns or []
        self.target_column = target_column

    def get_column_by_type(self, table, type, names_only=False):
        cur = self._conn.execute("""
           SELECT column_name
           FROM information_schema.columns
           WHERE table_name = '{}'
           AND data_type = '{}'
           """.format(table, type)
                                 )
        if names_only:
            return cur

        for name, *others in cur:
            if name not in self._excluded_columns and name != self.target_column:
                c = self._conn.execute(
                    "SELECT DISTINCT({0}) FROM {1} ORDER BY {0} ASC".format(name, table))
                categories = list(map(lambda x: x if x[0] else ['X'], c))
                yield name, np.asarray(categories)

    def encode_categorical_columns(self, table, iterator=None):
        items = iterator or self.get_column_by_type(table, 'character varying')
        encode = self.encoders.get('{}_categorical'.format(table), ActiveEncoder()).encoder or Encoding(table)
        names = []
        for name, col in items:
            encode.categorical_(name, col)
            names.append(name)
        self.encoders['{}_categorical'.format(table)] = ActiveEncoder(encoder=encode, columns=names)
        return encode

    def encode_numerical_columns(self, table, df):
        encode = self.encoders.get('{}_numerical'.format(table), ActiveEncoder()).encoder or Encoding(table)
        encode.numerical_('standard', df)
        self.encoders['{}_numerical'.format(table)] = ActiveEncoder(encoder=encode, columns=df.columns.tolist())
        return encode

    def encode_target_column(self, table, targets=None):
        q = "SELECT DISTINCT({0}) FROM {1} ORDER BY {0} ASC".format(self.target_column, table)
        targets = targets or list(map(lambda x: x if x[0] else ['X'], self._conn.execute(q)))

        encode = self.encoders.get('target', ActiveEncoder()).encoder or Encoding('target')
        encode.target_('target', np.asarray(targets).ravel())
        self.encoders['target'] = ActiveEncoder(encoder=encode, columns=[self.target_column])
        return encode

    def save(self, path=None):
        del self._conn
        p = path or 'pre_processing_encoders'
        name = "{}.pkl".format(p) if not p.endswith('.pkl') else p
        with open(name, "wb") as outfile:
            pickled = pickle.dump(self, outfile)
        print("pickled: {} encoders @ {}".format(len(self.encoders), name))
        return pickled

    def load(self, path):
        print('Attempting load of:', path)
        pp = load_pickle(path)
        if pp and hasattr(pp, 'encoders'):
            print('Loading pre-processors from {}, adding {} encoders'.format(path, len(pp.encoders)))
            self.encoders = pp.encoders
        else:
            print('Pre-processor has no encoders', dir(pp))

    def save_individual(self, path=None):
        for name, encoder_dict in self.encoders.items():
            encoder_dict.get('encoder').save(path)

    def load_individual(self, path):
        pickles = listdir(path)
        for file in pickles:
            encoder = load_pickle('{}/{}'.format(path, file))
            name = file.replace("_encoder.pkl", '')
            self.encoders[name] = encoder


if __name__ == '__main__':
    conn = connect(local=True).connect()

    EXCLUDED = [
        'origination_date_string', 'first_payment_date_string',
        'monthly_reporting_period', 'maturity_date_string',
        'zero_balance_effective_date_string', 'last_paid_installment_date_string',
        'disposition_date_string', 'foreclosure_date_string', 'product_type'
    ]

    pp = PreProcessors(conn=conn, excluded_columns=EXCLUDED, target_column='current_loan_delinquency_status')
    pp.load(PRE_PROCESSING_ENCODERS_PICKLE_PATH)
    pp.encode_categorical_columns('acquisition')
    pp.encode_categorical_columns('performance')

    pp.save(PRE_PROCESSING_ENCODERS_PICKLE_PATH)

    test = PreProcessors()
    test.load(PRE_PROCESSING_ENCODERS_PICKLE_PATH)
    print(dir(test))
    print(test.encoders)
