from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np
import pickle
from lib.db import connect


def load_encoder(table):
    with open("{}_encoder.pkl".format(table), "rb") as infile:
        loaded = pickle.load(infile)
    return loaded


def create_encoding(table, column_names):
    class Encoding:
        __slots__ = ('table', *column_names)

        def __init__(self, table=None):
            self.table = table

        def categorical(self, name, categories):
            ohe = OneHotEncoder(sparse=False)
            ohe.fit(categories)
            setattr(self, name, ohe)
            print('OneHotEncoder', ohe.categories_)
            return ohe

        def integer(self, name, categories):
            ss = StandardScaler()
            ss.fit(categories)
            setattr(self, name, ss)
            print('StandardScaler', ss.mean_)
            return ss

        def apply_encoding(self, name, categories):
            if not hasattr(self, name):
                raise Exception(
                    'You did not create an encoder for {} categories, use the categorical() method first'.format(name))
            return self[name].transform(categories)

        def save(self):
            with open("{}_encoder.pkl".format(self.table), "wb") as outfile:
                pickled = pickle.dump(self, outfile)
            print("pickled: {} encoders".format(self.table))
            return pickled

    return Encoding(table)


def get_column_by_type(table, type, conn):
    cur = conn.execute("""
       SELECT column_name
       FROM information_schema.columns
       WHERE table_name = '{}'
       AND data_type = '{}'
       """.format(table, type)
                       )
    for name, *others in cur:
        c = conn.execute("SELECT DISTINCT({0}) FROM {1} ORDER BY {0} ASC".format(name, table))
        categories = list(c)
        # print(name, categories)
        yield name, np.asarray(categories)


def encode_categorical_columns(table, iterator):
    items = list(iterator)
    encode = create_encoding(table, list(map(lambda n: n[0], items)))
    for name, col in items:
        encode.categorical(name, col)
    return encode


def encode_integer_columns(table, iterator):
    items = list(iterator)
    encode = create_encoding(table, list(map(lambda n: n[0], items)))
    for name, col in items:
        encode.integer(name, np.array(col, dtype=int))
    return encode


def encode_table(table, conn):
    # char_columns = get_column_by_type(table, 'character varying', conn)
    # char_encoder = encode_categorical_columns(table, char_columns)
    int_columns = get_column_by_type(table, 'integer', conn)
    int_encoder = encode_integer_columns(table, int_columns)
    # encoders = dict(char_encoder=char_encoder, int_encoder=int_encoder)
    encoders = dict(char_encoder='', int_encoder=int_encoder)

    # for k, encoder in encoders.items():
    #     print(k, [e for e in dir(encoder) if not e.startswith('__')])
    return encoders


if __name__ == '__main__':
    conn = connect(local=True).connect()
    acq_encoders = encode_table('acquisition', conn)
    # perf_encoders = encode_table('performance', conn)

    # e = load_encoder('acquisition')
    # print(dir(e))
