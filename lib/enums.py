ACQUISITION_RAW_COLUMN_NAMES = ['loan_id', 'origin_channel', 'seller_name', 'original_interest_rate', 'original_upb',
                                'original_loan_term', 'origination_date_string', 'first_payment_date_string',
                                'original_loan_to_value', 'original_combined_loan_to_value', 'number_of_borrowers',
                                'original_debt_to_income_ratio', 'borrower_credit_score_at_origination',
                                'first_time_homebuyer_indicator', 'loan_purpose', 'property_type', 'number_of_units',
                                'occupancy_type', 'property_state', 'zip_code_short',
                                'primary_mortgage_insurance_percent',
                                'product_type', 'co_borrower_credit_score_at_origination', 'mortgage_insurance_type',
                                'relocation_mortgage_indicator', 'sdq']
ACQUISITION_DB_COLUMN_NAMES = ACQUISITION_RAW_COLUMN_NAMES

FEATURE1 = ['loan_id,','original_combined_loan_to_value,','borrower_credit_score_at_origination']
FEATURE2 = ['loan_id,','original_loan_term','original_combined_loan_to_value','borrower_credit_score_at_origination','original_interest_rate','original_upb']
FEATURE3 = ['loan_id','original_combined_loan_to_value','borrower_credit_score_at_origination','original_interest_rate','original_debt_to_income_ratio']


PERFORMANCE_RAW_COLUMN_NAMES = ['loan_id', 'monthly_reporting_period', 'servicer_name', 'current_interest_rate',
                                'current_actual_upb', 'loan_age', 'remaining_months_to_legal_maturity',
                                'adjusted_months_to_maturity', 'maturity_date_string', 'metro_stats_area',
                                'current_loan_delinquency_status', 'modification_flag', 'zero_balance_code',
                                'zero_balance_effective_date_string', 'last_paid_installment_date_string',
                                'foreclosure_date_string', 'disposition_date_string', 'foreclosure_costs',
                                'property_preservation_and_repair_costs', 'asset_recovery_costs',
                                'misc_holding_expenses_credits', 'taxes_for_holding_property', 'net_sale_proceeds',
                                'credit_enhancement_proceeds', 'repurchase_make_whole_proceeds',
                                'other_foreclosure_proceeds', 'non_interest_bearing_upb',
                                'principle_forgiveness_amount',
                                'repurchase_make_whole_proceeds_flag', 'foreclosure_principle_write_off_amount',
                                'servicing_activity_indicator']
PERFORMANCE_DB_COLUMN_NAMES = ['id'] + PERFORMANCE_RAW_COLUMN_NAMES

PRE_PROCESSING_ENCODERS_PICKLE_PATH = 'pickles/pre_processing_encoders.pkl'
LIVE_PRE_PROCESSING_ENCODERS_PICKLE_PATH = 'pickles/LIVE.pre_processing_encoders.pkl'

EXCLUDED_CATEGORY_COLUMNS = [
    'origination_date_string', 'first_payment_date_string',
    'monthly_reporting_period', 'maturity_date_string',
    'zero_balance_effective_date_string', 'last_paid_installment_date_string',
    'disposition_date_string', 'foreclosure_date_string', 'product_type'
]
