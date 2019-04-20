import psycopg2
# import SQLAlchemy as sql
from sqlalchemy import create_engine, orm


def connect(local=True, **kwargs):
    host = dict(host='localhost') if local else dict(host="34.74.8.14")
    connection = dict(database="loan_performance", user="gwu-dl-user", password="Gw_ml2_@")
    engine = create_engine(
        "postgresql://{user}:{password}@{host}:5432/{database}".format(**{**connection, **host, **kwargs}))
    return engine


if __name__ == '__main__':
    e = connect()
    count = e.execute("SELECT COUNT(*) FROM acquisition LIMIT 1").scalar()
    print('acquisition count:', count)
