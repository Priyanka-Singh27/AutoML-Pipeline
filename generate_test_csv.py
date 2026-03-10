from mocks.mock_high_correlation import mock_df
import os

mock_df.to_csv("test_tmp.csv", index=False)
print("Saved test_tmp.csv")
