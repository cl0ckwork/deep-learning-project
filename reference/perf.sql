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
-- Name: performance; Type: TABLE; Schema: public; Owner: gwu-dl-user
--

CREATE TABLE public.performance (
    id bigint NOT NULL,
    loan_id bigint,
    monthly_reporting_period character varying,
    monthly_reporting_month_sin double precision,
    monthly_reporting_month_cos double precision,
    monthly_reporting_year integer,
    servicer_name character varying,
    current_interest_rate integer,
    current_actual_upb integer,
    loan_age integer,
    remaining_months_to_legal_maturity integer,
    adjusted_months_to_maturity integer,
    maturity_date_string character varying,
    maturity_month_sin double precision,
    maturity_month_cos double precision,
    maturity_year integer,
    metro_stats_area double precision,
    current_loan_delinquency_status character varying,
    modification_flag integer,
    zero_balance_code double precision,
    zero_balance_effective_date_string character varying,
    zero_balance_effective_month_sin double precision,
    zero_balance_effective_month_cos double precision,
    zero_balance_effective_year integer,
    last_paid_installment_date_string character varying,
    last_paid_installment_month_sin double precision,
    last_paid_installment_month_cos double precision,
    last_paid_installment_year integer,
    foreclosure_date_string character varying,
    foreclosure_month_sin double precision,
    foreclosure_month_cos double precision,
    foreclosure_year integer,
    disposition_date_string character varying,
    disposition_month_sin double precision,
    disposition_month_cos double precision,
    disposition_year integer,
    foreclosure_costs integer,
    property_preservation_and_repair_costs integer,
    asset_recovery_costs integer,
    misc_holding_expenses_credits integer,
    taxes_for_holding_property integer,
    net_sale_proceeds integer,
    credit_enhancement_proceeds integer,
    repurchase_make_whole_proceeds integer,
    other_foreclosure_proceeds integer,
    non_interest_bearing_upb integer,
    principle_forgiveness_amount integer,
    repurchase_make_whole_proceeds_flag integer,
    foreclosure_principle_write_off_amount integer,
    servicing_activity_indicator integer
);


ALTER TABLE public.performance OWNER TO "gwu-dl-user";

--
-- Name: performance_id_seq; Type: SEQUENCE; Schema: public; Owner: gwu-dl-user
--

CREATE SEQUENCE public.performance_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.performance_id_seq OWNER TO "gwu-dl-user";

--
-- Name: performance_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: gwu-dl-user
--

ALTER SEQUENCE public.performance_id_seq OWNED BY public.performance.id;


--
-- Name: performance id; Type: DEFAULT; Schema: public; Owner: gwu-dl-user
--

ALTER TABLE ONLY public.performance ALTER COLUMN id SET DEFAULT nextval('public.performance_id_seq'::regclass);


--
-- Data for Name: performance; Type: TABLE DATA; Schema: public; Owner: gwu-dl-user
--

COPY public.performance (id, loan_id, monthly_reporting_period, monthly_reporting_month_sin, monthly_reporting_month_cos, monthly_reporting_year, servicer_name, current_interest_rate, current_actual_upb, loan_age, remaining_months_to_legal_maturity, adjusted_months_to_maturity, maturity_date_string, maturity_month_sin, maturity_month_cos, maturity_year, metro_stats_area, current_loan_delinquency_status, modification_flag, zero_balance_code, zero_balance_effective_date_string, zero_balance_effective_month_sin, zero_balance_effective_month_cos, zero_balance_effective_year, last_paid_installment_date_string, last_paid_installment_month_sin, last_paid_installment_month_cos, last_paid_installment_year, foreclosure_date_string, foreclosure_month_sin, foreclosure_month_cos, foreclosure_year, disposition_date_string, disposition_month_sin, disposition_month_cos, disposition_year, foreclosure_costs, property_preservation_and_repair_costs, asset_recovery_costs, misc_holding_expenses_credits, taxes_for_holding_property, net_sale_proceeds, credit_enhancement_proceeds, repurchase_make_whole_proceeds, other_foreclosure_proceeds, non_interest_bearing_upb, principle_forgiveness_amount, repurchase_make_whole_proceeds_flag, foreclosure_principle_write_off_amount, servicing_activity_indicator) FROM stdin;
\.


--
-- Name: performance_id_seq; Type: SEQUENCE SET; Schema: public; Owner: gwu-dl-user
--

SELECT pg_catalog.setval('public.performance_id_seq', 2416025, true);


--
-- Name: performance performance_pkey; Type: CONSTRAINT; Schema: public; Owner: gwu-dl-user
--

ALTER TABLE ONLY public.performance
    ADD CONSTRAINT performance_pkey PRIMARY KEY (id);


--
-- Name: performance_id_uindex; Type: INDEX; Schema: public; Owner: gwu-dl-user
--

CREATE UNIQUE INDEX performance_id_uindex ON public.performance USING btree (id);


--
-- Name: performance performance_loan_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: gwu-dl-user
--

ALTER TABLE ONLY public.performance
    ADD CONSTRAINT performance_loan_id_fkey FOREIGN KEY (loan_id) REFERENCES public.acquisition(loan_id) ON DELETE CASCADE;


--
-- PostgreSQL database dump complete
--

