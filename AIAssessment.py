import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn



# =========== Model Libraries ===============
# Package needed --> pip install scikit-learn
from sklearn.preprocessing import MinMaxScaler  
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from mpl_toolkits.mplot3d import Axes3D
from joblib import dump, load
# from joblib import load



# =========== SCIKIT AI MODELS ===============
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR 
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.cross_decomposition import PLSRegression



# =========== USER INTERFACE ===============
import tkinter as tk
from tkinter import messagebox



# ============= Import the dataset ============
# Reading given dataset - converted to csv as xlsx couldn't be read
#df_nwd = pd.read_excel('Net_Worth_Data.xlsx')
df_nwd = pd.read_csv("Net_Worth_Data.csv")





# # ============= Display first few rows of the dataset ===========
# print(df_nwd.head())
# #or
# print(df_nwd[:5])

# # # creating space from content to content
# # print("\n\n")





# # # ============= Display last few rows of the dataset ============
# print(df_nwd.tail(5))


# # creating space from content to content
# print("\n\n")





# ============= Determine shape of the dataset (shape - total numbers of rows and columns) ============
shape = df_nwd.shape
print(f"\nShape of given dataframe {shape}.To be precise, this dataset consists of {shape[0]} rows and {shape[1]} columns\n")

# creating space from content to content
print("\n\n")






# ============= Display concise summary of the dataset (info) ============
print(f"\nConcise summary of given data set is {df_nwd.info()}\n")

# creating space from content to content
print("\n\n")





# # ============= Check the null values in dataset (isnull) ============
# # isnull will show nulls
# # sum will show count of the null values
# print(f"\nTotal ammount of null values in data set is: \n{df.isnull().sum()}\n")

# # creating space from content to content
# print("\n\n")





# # ============ Identify library to plot graph to understand relations among various columns =============
# # installing seaborn library pip install seaborn
# # imported matplotlib
# # shows relationship between data
# # no need to print as show does it

# # Pair plot
# sbn.pairplot(df)
# plt.show()

# # creating space from content to content
# print("\n\n")

# # individual columns
# sbn.scatterplot(x ='Age', y = 'Customer Name', data = df)
# plt.show()

# # creating space from content to content
# print("\n\n")





# ============ Create input dataset from original dataset by dropping irrelevant features ============
# dropped columns will be stored in variable X
# axis=1: This parameter specifies the axis along which to drop the columns. 
#In this case, axis=1 means that you want to drop columns (as opposed to rows, which would be axis=0).

# columns stated are irrelevant to this dataset hence being dropped
X = df_nwd.drop(['Client Name','Client e-mail','Profession', 'Education', 'Country','Net Worth'], axis=1)
print(f"\nDropped columns from dataset is {X}\n")

# creating space from content to content
print("\n\n")





# ============ Create output dataset from original dataset =============
# output column will be stored in variable Y
Y = df_nwd['Net Worth']
print(f"\nOutput column from dataset is {Y}\n")

# creating space from content to content
print("\n\n")





# ============ Transform input dataset into percentage based weighted between 0 and 1 =============
# input columns are the output colouts to which we need to use Gender, Age, Annual Salary, Credit Card Debt, Net Worth, Car Purchase Amount

sc = MinMaxScaler()
x_scaler = sc.fit_transform(X)
print(x_scaler)

# creating space from content to content
print("\n\n")




# ============ Transform output dataset into percentage based weighted between 0 and 1 =============

sc1 = MinMaxScaler()
y_reshape = Y.values.reshape(-1,1)
y_scaler = sc1.fit_transform(y_reshape)
print(y_scaler)

# creating space from content to content
print("\n\n")



# ============ Print first few rows of scaled input dataset =============
# :5 number is based on what is written in array so if it says :7, 7 rows of data will show
print(f"\nthe follow is the data from dropped columns{x_scaler[:5]}\n")

# creating space from content to content
print("\n\n")





# ============ Print first few rows of scaled output dataset =============
# y_scaler.head() nto working - used below mixmaxScaler function
# :5 number is based on what is written in array so if it says :7, 7 rows of data will show
print(f"\nthe follow is the data from output columns{y_scaler[:7]}\n")

# creating space from content to content
print("\n\n")




# ============ Split data into training and testing sets =============
# split the dataset
# if the size has been dictated, shuffle wouldnt take affect
# ratio is 80% train and 20% test
X_train, X_test, Y_train, Y_test = train_test_split( x_scaler, y_scaler, test_size = 0.2, train_size = 0.8, random_state = 42)

print("X_train results:\n", X_train)
print("X_test results:", X_test)
print("Y_train results:", Y_train)
print("Y_test results:\n", Y_test)

# creating space from content to content
print("\n\n")




# ============ Print shape of test and training data =============
print("Shape of training given dateset:\n", X_train.shape)
print("Shape of test given dateset", X_test.shape)

# creating space from content to content
print("\n")

print("Shape of training given dateset:", Y_train.shape)
print("Shape of test given dateset\n", Y_test.shape)


# creating space from content to content
print("\n\n")





# ============ Print first few rows of test and training data =============
print("First few rows of training given dateset:\n", X_train[:5])
print("First few rows of test given dateset", X_test[:6])

# creating space from content to content
print("\n")

print("First few rows of training given dateset:", Y_train[:7])
print("First few rows of test given dateset\n", Y_test[:8])

# creating space from content to content
print("\n\n")




# ============ Import and initialize AI models (need 10) =============

# assign variables to make it easier to use later
lnr = LinearRegression()
lso = Lasso()
rdg = Ridge()
eln = ElasticNet(alpha = 1.0, l1_ratio = 0.5)
dtr = DecisionTreeRegressor()
svm = SVR()
xgr = XGBRegressor()
rdf = RandomForestRegressor()
bsr = BayesianRidge()
pls = PLSRegression()




# ============ Train models using training data =============
# use .fit -->  method is how a machine learning model learns from the training data. It adjusts the model's parameters so that it can make accurate predictions.
#x and y train used from data split conducted earlier
lnr.fit(X_train, Y_train)
lso.fit(X_train, Y_train)
rdg.fit(X_train, Y_train)
eln.fit(X_train, Y_train)
dtr.fit(X_train, Y_train)
svm.fit(X_train, Y_train)
xgr.fit(X_train, Y_train)
rdf.fit(X_train, Y_train)
bsr.fit(X_train, Y_train)
pls.fit(X_train, Y_train)





# ============ Prediction on test data =============
# predicting the x_test  to the models variables from earlier
# assigned variables --> easier to utilise later when needed
lnr_pred = lnr.predict(X_test)
lso_pred = lso.predict(X_test)
rdg_pred = rdg.predict(X_test)
eln_pred = eln.predict(X_test)
dtr_pred = dtr.predict(X_test)
svm_pred = svm.predict(X_test)
xgr_pred = xgr.predict(X_test)
rdf_pred = rdf.predict(X_test)
bsr_pred = bsr.predict(X_test)
pls_pred = pls.predict(X_test)



# ============ Evaluate model performance =============
# use RMSE --> Root Mean Squared Error
# mean_squared_error alias = mse

# variable assigned
lnr_rmse = mse(Y_test, lnr_pred, squared = False)
lso_rmse = mse(Y_test, lso_pred, squared = False)
rdg_rmse = mse(Y_test, rdg_pred, squared = False)
eln_rmse = mse(Y_test, eln_pred, squared = False)
dtr_rmse = mse(Y_test, dtr_pred, squared = False)
svm_rmse = mse(Y_test, svm_pred, squared = False)
xgr_rmse = mse(Y_test, xgr_pred, squared = False)
rdf_rmse = mse(Y_test, rdf_pred, squared = False)
bsr_rmse = mse(Y_test, bsr_pred, squared = False)
pls_rmse = mse(Y_test, pls_pred, squared = False)

# creating space from content to content
print("\n\n")




# ============ Display evaluation results =============
# displaying the RMSE and result from last exercise
print(f"Linear Regression RMSE results are: {lnr_rmse}")
print(f"Lasso Regression RMSE results are: {lso_rmse}")
print(f"Ridge Regression RMSE results are: {rdg_rmse}")
print(f"Elastic Net Regression RMSE results are: {eln_rmse}")
print(f"Desicion Tree Regression RMSE results are: {dtr_rmse}")
print(f"Support Vector Regression RMSE results are: {svm_rmse}")
print(f"Xtreme Gradient Boosting Regression RMSE results are: {xgr_rmse}")
print(f"Bayesian Ridge Regression RMSE results are: {bsr_rmse}")
print(f"Random Forest Regression RMSE results are: {rdf_rmse}")
print(f"Partial Least Squares Regression RMSE results are: {pls_rmse}")



# creating space from content to content
print("\n\n")




# # ============ Choose best model ============
# initialise the models
model_entity = [lnr, lso, rdg, eln, dtr, svm, xgr, rdf, bsr, pls]
rmse_results = [lnr_rmse, lso_rmse, rdg_rmse, eln_rmse, dtr_rmse]

optimum_model_index = rmse_results.index(min(rmse_results))
optimum_model_entity = model_entity[optimum_model_index]


#  pictorial representaion of data
modelling_object = ["Linear Regression", "Lasso Regression", "Ridge Regression", "Elastic Net Regression", "Decision Tree Regression", "Support Vector Regression", "Xtreme Gradient Boosting Regression", "Bayesian Ridge Regression", "Random Forest Regression", "Partial Least Squares Regression"]
rmse_results = [lnr_rmse, lso_rmse, rdg_rmse, eln_rmse, dtr_rmse, svm_rmse, xgr_rmse, bsr_rmse, rdf_rmse, pls_rmse]

# Creating figure
plt.figure(figsize = (10, 6))

# Plotting the bars and colours
colors = ["#4169E1", "#50C878", "#DC143C", "#FFD700", "#FF6F61", "#725D34", "#128594", "#81255A", "#935613", "#9B079F"]  
bars = plt.bar(modelling_object, rmse_results, color = colors)

# Adding labels to the bars and its value
# displaying value in center of vertical and horizontal alignment
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.000001, round(yval, 6), va = "bottom", ha = "center")


# Labels and title
plt.xlabel("Regression Models")
plt.ylabel("RMSE")
plt.title("RMSE Regression Model Comparison")
plt.xticks(rotation = 45)

# Showing the plot
plt.show()


# # ===============================================================================================================================================================================
# # ========================================================================= RETRAINING ==========================================================================================
# # ===============================================================================================================================================================================

# Retraining the model for given dataset
linReg_retrain = LinearRegression()
linReg_retrain.fit(x_scaler, y_scaler)


# Saving the models
# dump & load = serialising (saving) and deserialising (loading) || # referes to the optimum_model_entity created earlier || # give saved model a name with .joblin extension
dump(optimum_model_entity, "Net_Worth.joblib")

# loading the saved file from earlier
load_model = load( "Net_Worth.joblib")




# # ============ User Inputs ============
# # taking in user inputs
# gender = int(input("Please enter your gender 'Enter 0 for FEMALE and 1 for MALE': "))
# age = int(input("Please enter your age: "))
# annual_income = float(input("Please enter your yearly income: "))
# credit_card_debt = float(input("Please enter your credit card debt: "))
# healthcare_cost = float(input("Please enter your health care cost: "))
# inherited_amount = float(input("Please enter your inheritance info: "))
# stocks = float(input("Please enter your stock investments: "))
# bonds = float(input("Please enter your bonds investments: "))
# mutual_funds = float(input("Please enter your mutual fund info: "))
# etf = float(input("Please enter your info on exchange-trade funds: "))
# reit = float(input("Please enter your info on real estate investment trusts: "))


# print("\n\n")





# # ============ Predicting using given inputs ============
# # sc = alias for mixmaxascaler
# # saving given detials in a list
# input_details = sc.transform([[gender, age, annual_income, credit_card_debt, healthcare_cost,inherited_amount, stocks, bonds, mutual_funds, etf, reit ]])
# input_pred = load_model.predict(input_details)


# print(input_pred)
# # print("\n")
# print(f"value of net worth prediction based on your given information is: {sc1.inverse_transform(input_pred)}")


# print("\n\n")




# ===============================================================================================================================================================================
# ========================================================================= USER INTERFACE ======================================================================================
# ===============================================================================================================================================================================
def nwd_prediction():
    try:
        # ============ User input field ============
        # getting users input from UI entry fields
        gender = int(gender_input.get())
        age = int(age_input.get())
        annual_income = float(annual_income_input.get())
        credit_card_debt = float(credit_card_debt_input.get())
        healthcare_cost = float(healthcare_cost_input.get())
        inherited_amount = float(inherited_amount_input.get())
        stocks = float(stocks_input.get())
        bonds = float(bonds_input.get())
        mutual_funds = float(mutual_funds_input.get())
        etf = float(etf_input.get())
        reit = float(reit_input.get())



        # ============ Predicting using given inputs ============
        # sc = alias for mixmaxascaler
        input_details = sc.transform([[gender, age, annual_income, credit_card_debt, healthcare_cost, inherited_amount, stocks, bonds, mutual_funds, etf, reit]])
        input_pred = load_model.predict(input_details)
        predicted_amount = sc1.inverse_transform(input_pred)


        # ============ Predicting Output ============
        result_label.config(text = f"Prediction based on your given information is: {predicted_amount[0][0]:.5f}")
    
    except ValueError:
        messagebox.showerror("OOPS! Invalid input... Please enter numerical values into valid fields!")



# ============================================
# ============ CREATING UI WINDOW ============
root = tk.Tk()
# title for the UI
root.title("Net Worth Prediction")


# =============================================
# ============ ADDING INPUT FIELDS ============
#padx and pady = paddying x and y

# Gender Input Field
tk.Label(root, text = "Please enter your gender 'Enter 0 for FEMALE and 1 for MALE': ").grid(row = 0, column = 0, padx = 10, pady = 5)
gender_input = tk.Entry(root)
gender_input.grid(row = 0, column = 1, padx = 10, pady = 5)


# Age Input Field
tk.Label(root, text = "Please enter your age: ").grid(row = 1, column = 0, padx = 10, pady = 5)
age_input = tk.Entry(root)
age_input.grid(row = 1, column = 1, padx = 10, pady = 5)

# Annual Income Input Field
tk.Label(root, text = "Please enter your yearly income: ").grid(row = 2, column = 0, padx = 10, pady = 5)
annual_income_input = tk.Entry(root)
annual_income_input.grid(row = 2, column = 1, padx = 10, pady = 5)


# Credit Card Debt Input Field
tk.Label(root, text = "Please enter your credit card debt: ").grid(row = 3, column = 0, padx = 10, pady = 5)
credit_card_debt_input = tk.Entry(root)
credit_card_debt_input.grid(row = 3, column = 1, padx = 10, pady = 5)


# HealthCare Cost Input Field
tk.Label(root, text = "Please enter your health care cost: ").grid(row = 4, column = 0, padx = 10, pady = 5)
healthcare_cost_input = tk.Entry(root)
healthcare_cost_input.grid(row = 4, column = 1, padx = 10, pady = 5)


# Interited Amount Input Field
tk.Label(root, text = "Please enter your inheritance info: ").grid(row = 5, column = 0, padx = 10, pady = 5)
inherited_amount_input = tk.Entry(root)
inherited_amount_input.grid(row = 5, column = 1, padx = 10, pady = 5)


# Stock Investment Input Field
tk.Label(root, text = "Please enter your stock investments: ").grid(row = 6, column = 0, padx = 10, pady = 5)
stocks_input = tk.Entry(root)
stocks_input.grid(row = 6, column = 1, padx = 10, pady = 5)


# Bonds Investment Input Field
tk.Label(root, text = "Please enter your bonds investments: ").grid(row = 7, column = 0, padx = 10, pady = 5)
bonds_input = tk.Entry(root)
bonds_input.grid(row = 7, column = 1, padx = 10, pady = 5)


# Mutual Funds Input Field
tk.Label(root, text = "Please enter your mutual fund info: ").grid(row = 8, column = 0, padx = 10, pady = 5)
mutual_funds_input = tk.Entry(root)
mutual_funds_input.grid(row = 8, column = 1, padx = 10, pady = 5)


# Exchange Trade Funds Input Field
tk.Label(root, text = "Please enter your info on exchange-trade funds: ").grid(row = 9, column = 0, padx = 10, pady = 5)
etf_input = tk.Entry(root)
etf_input.grid(row = 9, column = 1, padx = 10, pady = 5)


# Real Estate Investment Trust Input Field
tk.Label(root, text = "Please enter your info on real estate investment trusts: ").grid(row = 10, column = 0, padx = 10, pady = 5)
reit_input = tk.Entry(root)
reit_input.grid(row = 10, column = 1, padx = 10, pady = 5)

# ===========================================================
# ============ ADDING PREDICT BUTTON & PREDCTION ============

# Button for click event --> command = function name
predict_button = tk.Button(root, text = "Predict Net Worth", command = nwd_prediction)
predict_button.grid(row = 13, column = 0, columnspan = 2, pady = 10)

result_label = tk.Label(root, text = "")
result_label.grid(row = 15, column = 0, columnspan = 2, pady = 10)





# ===============================================
# ============ RUNNING TKINTER EVENT ============
root.mainloop()

#script to run file
# python AIAssessment.py