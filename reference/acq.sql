--
-- PostgreSQL database dump
--

-- Dumped from database version 11.2 (Debian 11.2-1.pgdg90+1)
-- Dumped by pg_dump version 11.2

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_with_oids = false;

--
-- Name: acquisition; Type: TABLE; Schema: public; Owner: gwu-dl-user
--

CREATE TABLE public.acquisition (
    loan_id bigint NOT NULL,
    origin_channel character varying,
    seller_name character varying,
    original_interest_rate integer,
    original_upb integer,
    original_loan_term integer,
    origination_date_string character varying,
    origination_month_sin double precision,
    origination_month_cos double precision,
    origination_year integer,
    first_payment_date_string character varying,
    first_payment_month_sin double precision,
    first_payment_month_cos double precision,
    first_payment_year integer,
    original_loan_to_value integer,
    original_combined_loan_to_value integer,
    number_of_borrowers integer,
    original_debt_to_income_ratio integer,
    borrower_credit_score_at_origination integer,
    first_time_homebuyer_indicator character varying(1),
    loan_purpose character varying(1),
    property_type character varying,
    number_of_units integer,
    occupancy_type character varying,
    property_state character varying(2),
    zip_code_short character varying,
    primary_mortgage_insurance_percent integer,
    product_type character varying,
    co_borrower_credit_score_at_origination integer,
    mortgage_insurance_type integer,
    relocation_mortgage_indicator integer,
    sdq integer NOT NULL
);


ALTER TABLE public.acquisition OWNER TO "gwu-dl-user";

--
-- Data for Name: acquisition; Type: TABLE DATA; Schema: public; Owner: gwu-dl-user
--

COPY public.acquisition (loan_id, origin_channel, seller_name, original_interest_rate, original_upb, original_loan_term, origination_date_string, origination_month_sin, origination_month_cos, origination_year, first_payment_date_string, first_payment_month_sin, first_payment_month_cos, first_payment_year, original_loan_to_value, original_combined_loan_to_value, number_of_borrowers, original_debt_to_income_ratio, borrower_credit_score_at_origination, first_time_homebuyer_indicator, loan_purpose, property_type, number_of_units, occupancy_type, property_state, zip_code_short, primary_mortgage_insurance_percent, product_type, co_borrower_credit_score_at_origination, mortgage_insurance_type, relocation_mortgage_indicator, sdq) FROM stdin;
\.


--
-- Name: acquisition acquisition_pkey; Type: CONSTRAINT; Schema: public; Owner: gwu-dl-user
--

ALTER TABLE ONLY public.acquisition
    ADD CONSTRAINT acquisition_pkey PRIMARY KEY (loan_id);


--
-- Name: acquisition_loan_id_uindex; Type: INDEX; Schema: public; Owner: gwu-dl-user
--

CREATE UNIQUE INDEX acquisition_loan_id_uindex ON public.acquisition USING btree (loan_id);


--
-- PostgreSQL database dump complete
--

