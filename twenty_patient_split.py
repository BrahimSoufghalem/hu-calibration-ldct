"""
20-Patient Kaggle Experiment Split
===================================
A balanced subset of the original 100-patient split designed to fit within
Kaggle free-tier GPU quota (T4, ~30 h/week).

  Train  : 10 patients (5 Chest + 5 Abdomen) from EXPECTED_TRAIN
  Val    :  5 patients (3 Chest + 2 Abdomen) from EXPECTED_VAL
  Test   :  5 patients (3 Chest + 2 Abdomen) from EXPECTED_TEST
  Total  : 20 unique patients

Scientific note
---------------
Comparisons between models trained on these exact 20 patients are internally
valid. Results are NOT directly comparable to the published benchmark table
(which uses all 100 patients).
"""

# ---------------------------------------------------------------------------
# TRAIN  -  10 patients from EXPECTED_TRAIN
# ---------------------------------------------------------------------------
TRAIN_20P = {
    # Chest (5)
    'C095', 'C261', 'C218', 'C224', 'C099',
    # Abdomen (5)
    'L081', 'L248', 'L203', 'L219', 'L210',
}

# ---------------------------------------------------------------------------
# VAL  -  5 patients from EXPECTED_VAL
# ---------------------------------------------------------------------------
VAL_20P = {
    # Chest (3)
    'C202', 'C219', 'C107',
    # Abdomen (2)
    'L033', 'L187',
}

# ---------------------------------------------------------------------------
# TEST  -  5 patients from EXPECTED_TEST
# ---------------------------------------------------------------------------
TEST_20P = {
    # Chest (3)
    'C121', 'C249', 'C170',
    # Abdomen (2)
    'L241', 'L107',
}

# All 20 unique patients that need to be in the Kaggle dataset
ALL_20P = TRAIN_20P | VAL_20P | TEST_20P

assert len(ALL_20P) == 20, f"Expected 20 unique patients, got {len(ALL_20P)}"
assert len(TRAIN_20P) == 10
assert len(VAL_20P) == 5
assert len(TEST_20P) == 5
