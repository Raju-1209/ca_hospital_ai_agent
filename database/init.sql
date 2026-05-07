-- ── CA Hospital AI Agent — PostgreSQL Init Script ──────────────────────────
-- Run this to create all tables in your PostgreSQL database
-- Tables will be auto-populated when you run: python database/postgres_connector.py

-- Claims and Billing
CREATE TABLE IF NOT EXISTS claims_and_billing (
    claim_id           VARCHAR(50) PRIMARY KEY,
    patient_id         VARCHAR(50),
    encounter_id       VARCHAR(50),
    insurance_provider VARCHAR(100),
    payment_method     VARCHAR(50),
    billed_amount      DECIMAL(12,2),
    paid_amount        DECIMAL(12,2),
    claim_status       VARCHAR(50),
    denial_reason      TEXT,
    claim_date         DATE
);

-- Denials
CREATE TABLE IF NOT EXISTS denials (
    denial_id                  VARCHAR(50) PRIMARY KEY,
    claim_id                   VARCHAR(50),
    denial_reason_code         VARCHAR(20),
    denial_reason_description  TEXT,
    denied_amount              DECIMAL(12,2),
    denial_date                DATE,
    appeal_filed               VARCHAR(10),
    appeal_status              VARCHAR(50),
    final_outcome              VARCHAR(50)
);

-- Patients
CREATE TABLE IF NOT EXISTS patients (
    patient_id     VARCHAR(50) PRIMARY KEY,
    first_name     VARCHAR(100),
    last_name      VARCHAR(100),
    age            INTEGER,
    gender         VARCHAR(20),
    ethnicity      VARCHAR(100),
    insurance_type VARCHAR(100),
    marital_status VARCHAR(50),
    city           VARCHAR(100),
    state          VARCHAR(50)
);

-- Encounters
CREATE TABLE IF NOT EXISTS encounters (
    encounter_id     VARCHAR(50) PRIMARY KEY,
    patient_id       VARCHAR(50),
    provider_id      VARCHAR(50),
    visit_date       DATE,
    discharge_date   DATE,
    visit_type       VARCHAR(50),
    department       VARCHAR(100),
    reason_for_visit TEXT,
    diagnosis_code   VARCHAR(20),
    admission_type   VARCHAR(50),
    length_of_stay   INTEGER,
    readmitted_flag  VARCHAR(10)
);

-- Diagnoses
CREATE TABLE IF NOT EXISTS diagnoses (
    diagnosis_id          VARCHAR(50) PRIMARY KEY,
    encounter_id          VARCHAR(50),
    diagnosis_code        VARCHAR(20),
    diagnosis_description TEXT,
    primary_flag          VARCHAR(10),
    chronic_flag          VARCHAR(10)
);

-- Lab Tests
CREATE TABLE IF NOT EXISTS lab_tests (
    lab_test_id  VARCHAR(50) PRIMARY KEY,
    encounter_id VARCHAR(50),
    test_name    VARCHAR(100),
    test_result  VARCHAR(50),
    units        VARCHAR(50),
    normal_range VARCHAR(100),
    test_date    DATE,
    status       VARCHAR(50)
);

-- Medications
CREATE TABLE IF NOT EXISTS medications (
    medication_id   VARCHAR(50) PRIMARY KEY,
    encounter_id    VARCHAR(50),
    drug_name       VARCHAR(200),
    dosage          VARCHAR(100),
    route           VARCHAR(50),
    frequency       VARCHAR(100),
    duration        VARCHAR(100),
    prescribed_date DATE,
    cost            DECIMAL(10,2)
);

-- Procedures
CREATE TABLE IF NOT EXISTS procedures (
    procedure_id          VARCHAR(50) PRIMARY KEY,
    encounter_id          VARCHAR(50),
    procedure_code        VARCHAR(20),
    procedure_description TEXT,
    procedure_date        DATE,
    procedure_cost        DECIMAL(12,2)
);

-- Providers
CREATE TABLE IF NOT EXISTS providers (
    provider_id      VARCHAR(50) PRIMARY KEY,
    name             VARCHAR(200),
    department       VARCHAR(100),
    specialty        VARCHAR(100),
    years_experience INTEGER,
    location         VARCHAR(200),
    inhouse          VARCHAR(10)
);

-- Indexes for faster queries
CREATE INDEX IF NOT EXISTS idx_claims_patient    ON claims_and_billing(patient_id);
CREATE INDEX IF NOT EXISTS idx_claims_status     ON claims_and_billing(claim_status);
CREATE INDEX IF NOT EXISTS idx_encounters_patient ON encounters(patient_id);
CREATE INDEX IF NOT EXISTS idx_denials_claim     ON denials(claim_id);
CREATE INDEX IF NOT EXISTS idx_lab_encounter     ON lab_tests(encounter_id);
CREATE INDEX IF NOT EXISTS idx_med_encounter     ON medications(encounter_id);

-- Done
SELECT 'CA Hospital DB initialized successfully!' AS status;
