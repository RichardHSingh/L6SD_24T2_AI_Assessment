# ===================================================================================================================================================
# ========================================================== IMPORTING LIBRARY SECTION ==============================================================
# ===================================================================================================================================================


# ===================== IMPORTNG LIBRARIES =====================
import pytest
import pandas as pd
from preprocessing_data import nwd_preprocessing_data



# ===================================================================================================================================================
# ============================================================== TEST CASES 1 IRRELEVANT COLUMNS ====================================================
# ===================================================================================================================================================

# ===================== INPUT DATA =====================
# Input data - Verifying the input data doesn't contain unnecessary columns

def test_nwd_reprocess_input_columns():
    # Read the CSV file into a DataFrame
    df_nwd = pd.read_csv('Net_Worth_Data.csv')

    # Call the preprocessing function
    X_scaled, _, _, _ = nwd_preprocessing_data(df_nwd)

    # Define the input columns that should remain after preprocessing
    expected_columns = ['Gender', 'Age', 'Income', 'Credit Card Debt', 'Healthcare Cost', "Inherited Amount", "Stocks", "Bonds", "Mutual Funds", "ETFs", "REITs"]
    
    # Convert the scaled features back to a DataFrame for testing
    X_df_nwd = pd.DataFrame(X_scaled, columns = expected_columns)
    
    # Check if X_df is a DataFrame
    assert isinstance(X_df_nwd, pd.DataFrame)
    
    for col in expected_columns:
        assert col in X_df_nwd.columns, f"Column {col} should exist however it doesn't."

if __name__ == "__main__":
    pytest.main()





# ===================================================================================================================================================
# ============================================================== TEST CASES 2 OUTPUT COLUMNS ========================================================
# ===================================================================================================================================================


# # ===================== OUTPUT DATA =====================
# # Output data - Verifying the output data doesn't contain unnecessary columns

def test_nwd_reprocess_output_columns():
    df_nwd = pd.read_csv('Net_Worth_Data.csv')

    # Call the preprocessing function
    _, Y_scaled, _, _ = nwd_preprocessing_data(df_nwd)

    # defining the appropriate column for output
    output_column = 'Net Worth'

    # sclaing to convert to dataframe for testing
    Y_df_cpd = pd.DataFrame(Y_scaled, columns = [output_column])
      

    # Check if Y_df_cpd is a DataFrame
    assert isinstance(Y_df_cpd, pd.DataFrame)
    
    assert output_column in Y_df_cpd.columns


if __name__ == "__main__":
    pytest.main()





# ===================================================================================================================================================
# ============================================================== TEST CASES 3 SHAPING ===============================================================
# ===================================================================================================================================================
def test_dataset_shape():
    df_nwd = pd.read_csv('Net_Worth_Data.csv')

    # Call the preprocessing function
    X_scaled, Y_scaled, _, _ = nwd_preprocessing_data(df_nwd)

    # exeptected columns --> 1 = column and 0 = rows
    assert X_scaled.shape[1] == 11, "Expecting 11 columns in the dataset after the preprocessing"
    assert Y_scaled.shape[1] == 1, "Expecting 1 column as the output in the dataset after the preprocessing"

    expected_rows = df_nwd.shape[0]
    assert X_scaled.shape[0] == expected_rows, "Expecting 500 rows in the dataset after the preprocessing"
    assert Y_scaled.shape[0] == expected_rows, "Expecting 500 rows as the output in the dataset after the preprocessing"


if __name__ == "__main__":
    pytest.main()





# ===================================================================================================================================================
# ============================================================== TEST SCRIPT IN TERMINAL ============================================================
# ===================================================================================================================================================
# terminal syntax for running code
# pytest -s


