# ===================================================================================================================================================
# ========================================================== IMPORTING LIBRARY SECTION ==============================================================
# ===================================================================================================================================================


# ===================== IMPORTNG LIBRARIES =====================
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler



# ===================== CAR PURCHASE DATASET ====================
# loading given dataset --> using csv as xlsx file not loading
df_nwd = pd.read_csv('Net_Worth_Data.csv')


def nwd_preprocessing_data(df_nwd):
    #definging the appropriate columns for intended use
    irrelevant_columns = ['Client Name','Client e-mail','Profession', 'Education', 'Country','Net Worth']
   # input_used_columns = ['Gender', 'Age', 'Annual Salary', 'Credit Card Debt', 'Healthcare', "Cost", "Inherited Amount", "Stocks", "Bonds", "Mutual Funds", "ETFs", "REITs"]
    output_column = ['Net Worth']   

    # dropping irrelevant column X and extracting the output column Y
    X = df_nwd.drop(irrelevant_columns, axis = 1)
    Y = df_nwd[output_column]

    # scaling the dropped columns
    x_scaler = MinMaxScaler()
    X_scaled = x_scaler.fit_transform(X)

    # scaling hte output column
    y_scaler = MinMaxScaler()
    y_reshape = Y.values.reshape(-1,1)
    Y_scaled = y_scaler.fit_transform(y_reshape)  

    return X_scaled, Y_scaled, x_scaler, y_scaler


    
    



# ===================== ORIGINAL CAR PURCHASE FUNCTION ====================
# def cpd_preprocessing_data(df_cpd):
#     irrelevant_columns = ['Customer Name','Customer e-mail','Country', 'Car Purchase Amount']
#     input_used_columns = ['Gender', 'Age', 'Annual Salary', 'Credit Card Debt', 'Net Worth']
    
#     dropped_columns = [
#         col for col in df_cpd.columns if col in irrelevant_columns
#         ]
    
#     remainder_columns = [
#         col for col in df_cpd.columns if col not in dropped_columns
#         ]
    
#     return df_cpd[remainder_columns], dropped_columns, remainder_columns

# # scale and shape the data --> x scale and y scale and shape it .fit_transform