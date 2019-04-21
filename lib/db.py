from os import environ
from sqlalchemy import create_engine


def connect(local=True, **kwargs):
    host = dict(host='localhost') if local else dict(host=environ.get('DB_HOST') or "34.74.8.14")
    connection = dict(database="loan_performance", user="gwu-dl-user", password="Gw_ml2_@")
    print('Connecting:', host)
    engine = create_engine(
        "postgresql://{user}:{password}@{host}:5432/{database}".format(**{**connection, **host, **kwargs}))
    return engine


if __name__ == '__main__':
    e = connect()
    count = e.execute("SELECT COUNT(*) FROM acquisition LIMIT 1").scalar()
    print('acquisition count:', count)
