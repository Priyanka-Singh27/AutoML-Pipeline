"""
constants.py

Provides single-source-of-truth tuning thresholds and categorical fallback constants 
to be used across the Preprocessor and Target Detection layers.
"""

ONEHOT_THRESHOLD   = 15   # below this — OneHotEncoder
ORDINAL_THRESHOLD  = 50   # below this in clustering — OrdinalEncoder
                          # above this in clustering — HashingEncoder

# Sub-categorizations
DEF_VERY_SMALL = 500
DEF_SMALL = 2000
DEF_LARGE = 50000
DEF_VERY_LARGE = 500000

LEAKAGE_THRESHOLD = 0.95  # Pearson, Spearman, or Mutual Information correlation boundary
