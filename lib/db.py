import psycopg2
# import SQLAlchemy as sql
from sqlalchemy import create_engine, orm


def connect(local=True, **kwargs):
    host = dict(host='localhost') if local else dict(host="35.231.162.16")
    connection = dict(database="loan_performance", user="gwu-dl-user", password="Gw_ml2_@")
    engine = create_engine(
        "postgresql://{user}:{password}@{host}:5432/{database}".format(**{**connection, **host, **kwargs}))
    return engine
    # session_factory = orm.sessionmaker(bind=engine)
    # return orm.scoped_session(session_factory)()
    # return psycopg2.connect(database="loan_performance", user="gwu-dl-user", password="Gw_ml2_@", **connection)

# sql_engine = sql.create_engine("postgresql://{user}:{password}@{hostname}:5432/{database}}".format())

# Session = sessionmaker(bind=dest_db_con)
# session = Session()
# session.bulk_insert_mappings(MentorInformation, df.to_dict(orient="records"))
# session.close()
