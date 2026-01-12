-- BIGQUERY VALIDATION TEMPLATE (NO hard-coded values)

-- Required placeholders (must be replaced before execution):
-- {{PROJECT_ID}}
-- {{DATASET_ID}}
-- {{TABLE_ID}}
-- {{REQUIRED_COLUMNS}}
-- {{MAX_BLOOD_TYPE}}
-- {{MAX_NULL_ADMISSION}}
-- {{MAX_NULL_MEDCOND}}

DECLARE validated_table STRING DEFAULT '{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}';
DECLARE results_table   STRING DEFAULT '{PROJECT_ID}.{DATASET_ID}.validation_results';

-- Required columns (array literal injected by pipeline)
DECLARE required_columns ARRAY<STRING> DEFAULT [{REQUIRED_COLUMNS}];

-- Null thresholds
DECLARE max_blood_type FLOAT64 DEFAULT {MAX_BLOOD_TYPE};
DECLARE max_null_admission FLOAT64 DEFAULT {MAX_NULL_ADMISSION};
DECLARE max_null_medcond FLOAT64 DEFAULT {MAX_NULL_MEDCOND};

DECLARE failed BOOL DEFAULT FALSE;
-- ============================
-- Schema validation
-- ============================
CREATE TEMP TABLE existing_columns AS
SELECT column_name
FROM `{PROJECT_ID}.{DATASET_ID}.INFORMATION_SCHEMA.COLUMNS`
WHERE table_name = '{TABLE_ID}';

CREATE TEMP TABLE missing_columns AS
SELECT ARRAY_AGG(col) AS missing
FROM UNNEST(required_columns) col
WHERE col NOT IN (SELECT column_name FROM existing_columns);

-- ============================
-- Row counts and null checks
-- ============================
CREATE TEMP TABLE stats AS
SELECT
  COUNT(*) AS total_rows,
  COUNTIF(`Blood Type` IS NULL) AS null_blood_type,
  COUNTIF(`Date of Admission` IS NULL) AS null_admission,
  COUNTIF(`Medical Condition` IS NULL) AS null_medcond
FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}`;

-- ============================
-- Date consistency
-- ============================
CREATE TEMP TABLE bad_dates AS
SELECT COUNT(*) AS bad_date_count
FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}`
WHERE SAFE_CAST(`Discharge Date` AS DATE)
    < SAFE_CAST(`Date of Admission` AS DATE);

-- ============================
-- Final decision
-- ============================

IF (SELECT ARRAY_LENGTH(missing) FROM missing_columns) > 0 THEN
  SET failed = TRUE;
END IF;

IF (SELECT total_rows FROM stats) = 0 THEN
  SET failed = TRUE;
END IF;

IF SAFE_DIVIDE((SELECT null_blood_type FROM stats), (SELECT total_rows FROM stats)) > max_blood_type THEN
  SET failed = TRUE;
END IF;

IF SAFE_DIVIDE((SELECT null_admission FROM stats), (SELECT total_rows FROM stats)) > max_null_admission THEN
  SET failed = TRUE;
END IF;

IF SAFE_DIVIDE((SELECT null_medcond FROM stats), (SELECT total_rows FROM stats)) > max_null_medcond THEN
  SET failed = TRUE;
END IF;

IF (SELECT bad_date_count FROM bad_dates) > 0 THEN
  SET failed = TRUE;
END IF;

-- ============================
-- Persist validation result
-- ============================
CREATE TABLE IF NOT EXISTS `{PROJECT_ID}.{DATASET_ID}.validation_results` (
  run_ts TIMESTAMP,
  validated_table STRING,
  total_rows INT64,
  missing_columns ARRAY<STRING>,
  null_summary STRING,
  bad_date_count INT64,
  passed BOOL
);

INSERT INTO `{PROJECT_ID}.{DATASET_ID}.validation_results`
SELECT
  CURRENT_TIMESTAMP() AS run_ts,
  validated_table AS validated_table,
  (SELECT total_rows FROM stats) AS total_rows,
  (SELECT missing FROM missing_columns) AS missing_columns,
  TO_JSON_STRING(STRUCT(
    null_blood_type,
    null_admission,
    null_medcond,
    total_rows
  )) AS null_summary,
  (SELECT bad_date_count FROM bad_dates) AS bad_date_count,
  NOT failed AS passed
FROM stats;
