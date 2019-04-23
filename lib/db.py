from os import environ
from sqlalchemy import create_engine, Column, Integer, String, BigInteger, Float
from sqlalchemy.ext.declarative import declarative_base
from lib.enums import ACQUISITION_DB_COLUMN_NAMES

Base = declarative_base()


def connect(local=True, **kwargs):
    host = dict(host='localhost') if local else dict(
        host=kwargs.pop('host', None) or environ.get('DB_HOST', None) or "34.74.8.14"
    )
    connection = dict(database="loan_performance", user="gwu-dl-user", password="Gw_ml2_@")
    print('Connecting:', host, kwargs)
    try:
        engine = create_engine(
            "postgresql://{user}:{password}@{host}:5432/{database}".format(**{**connection, **host, **kwargs}))
        return engine
    except Exception as ex:
        print('CANNOT CONNECT TO DB:', repr(ex))
        print(ex)


#
class Acquisition(Base):
    __tablename__ = 'acquisition'

    loan_id = Column(BigInteger, primary_key=True)
    origin_channel = Column(String)
    seller_name = Column(String)
    original_interest_rate = Column(Integer)
    original_upb = Column(Integer)
    original_loan_term = Column(Integer)
    origination_date_string = Column(String)
    origination_month_sin = Column(Float)
    origination_month_cos = Column(Float)
    origination_year = Column(Integer)
    first_payment_date_string = Column(String)
    first_payment_month_sin = Column(Float)
    first_payment_month_cos = Column(Float)
    first_payment_year = Column(Integer)
    original_loan_to_value = Column(Integer)
    original_combined_loan_to_value = Column(Integer)
    number_of_borrowers = Column(Integer)
    original_debt_to_income_ratio = Column(Integer)
    borrower_credit_score_at_origination = Column(Integer)
    first_time_homebuyer_indicator = Column(String)
    loan_purpose = Column(String)
    property_type = Column(String)
    number_of_units = Column(Integer)
    occupancy_type = Column(String)
    property_state = Column(String)
    zip_code_short = Column(String)
    primary_mortgage_insurance_percent = Column(Integer)
    product_type = Column(String)
    co_borrower_credit_score_at_origination = Column(Integer)
    mortgage_insurance_type = Column(Integer)
    relocation_mortgage_indicator = Column(Integer)
    sdq = Column(Integer)

    def __repr__(self):
        return ', '.join(ACQUISITION_DB_COLUMN_NAMES)
        # return "<Acquisition(name='%s', fullname='%s', nickname='%s')>" % (
        #     self.name, self.fullname, self.nickname)


if __name__ == '__main__':
    e = connect()
    count = e.execute("SELECT COUNT(*) FROM acquisition LIMIT 1").scalar()
    print('acquisition count:', count)
