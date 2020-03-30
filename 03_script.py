#!/urs/bin/env python3

# Library Loading
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn

##########################
# Dataset 1: Movie Sales #
##########################

# Read the Sales Dataset
sales = pd.read_csv('./data/sales_16_17.csv')
sales_pred = pd.read_csv('./data/sales_18.csv')
sales = sales.append(sales_pred, ignore_index=True)

# Date transformation
try:
    sales['Day'] = pd.to_datetime(sales['Day'], format='%d/%m/%Y')
    #df.open_date.apply(lambda d: datetime.strptime(d, "%Y-%m-%d"))
except Exception as e:
    print(e)

# Add weekday and weekend label

sales['month'] = sales['Day'].dt.month_name()
sales['day_name'] = sales['Day'].dt.day_name()
sales['day_type'] = np.where(sales['Day'].dt.dayofweek < 5, 'weekday', 'weekend')

# Sort by day

sales = sales.sort_values(by=['Day'])

# Incorporate holidays
# Same procedure and merge tables
holidays = pd.read_csv('./data/holidays.csv')
holidays.rename(columns={'Fecha': 'Day'}, inplace=True)
holidays['Day'] = holidays['Day'].astype('datetime64[ns]')
result = pd.merge(sales, holidays, how='outer', on='Day')

# Change names to be machine readable
# Drop unnecesary columns
day_dic = {'Descanso obligatorio': 'RDAY', 'Vacaciones escolares': 'HDAY', 'No obligatorio': 'LDAY', np.nan: 'NDAY'}
result = result.replace({'Tipo': day_dic})
result = result.rename(columns={'Tipo': 'day_tipo'})
result = result.drop(columns=['Día', 'Nombre'])

# Make dummy varialbes for month, weekday name, day type
df_mon_name = pd.get_dummies(result['month'])
df_day_name = pd.get_dummies(result['day_name'])
df_day_type = pd.get_dummies(result['day_type'])
df_day_tipo = pd.get_dummies(result['day_tipo'])
result = pd.concat([result, df_mon_name, df_day_name, df_day_type, df_day_tipo], axis=1)

# Save it to file.
result.to_csv('./data/sales_by_day_bis.csv', index=False)

##########################
# Dataset 2: Movie Data  #
##########################

# Read the data file
df = pd.read_csv('./data/movie_data(copy).csv')

# Fix the opening date and closing date
try:
    df['open_date'] = pd.to_datetime(df['open_date'], format='%Y-%m-%d')
    #df.open_date.apply(lambda d: datetime.strptime(d, "%Y-%m-%d"))
except Exception as e:
    print(e)
temp = df['days_in_theater'].apply(np.ceil).apply(lambda x: pd.Timedelta(x, unit='D'))
df['close_date'] = pd.to_datetime(df['open_date']) + temp

# Missing values are treated as zero

mask = (df['international_gross'] == 0)| (df['budget'] == 0)
df['mov_budget'] = df['international_gross'].div(df['budget'], fill_value=0).where(~mask,0)

# Bin values according to categories for budget, number of theaters
# days in theater.
# Change names to be machine readable for movie studios

bins_0 = [-1, 1, 2, 5, 250]
labels_0 = ['N','L','M','H']
df['mov_budget'] = pd.cut(df['mov_budget'], bins=bins_0, labels=labels_0)
df['mov_budget'] = df['mov_budget'].astype(str)

bins_1 = [-1, 1500, 3500, 5000]
labels_1 = ['L','M','H']
df['mov_exp'] = pd.cut(df['num_theaters'], bins=bins_1, labels=labels_1)
df['mov_exp'] = df['mov_exp'].astype(str)

bins_2 = [-1, 100, 200, 300]
labels_2 = ['L','M','H']
df['mov_ext'] = pd.cut(df['days_in_theater'], bins=bins_2, labels=labels_2)
df['mov_ext'] = df['mov_ext'].astype(str)

dict_studio = {'Buena Vista':'BV', '20th Century Fox': 'FX', 'Sony': 'SO', 'Universal':'UN', 'Paramount':'PT'}
df = df.replace({'studio': dict_studio})
df['studio'] = df['studio'].astype(str)

# Join binns to create new categories.
df['mov_cat'] = df[['mov_budget', 'mov_exp', 'mov_ext']].apply(lambda x: ''.join(x), axis=1)

##########################
# Dataset 3: Combination #
##########################

# Combine sets
sales = pd.read_csv('./data/sales_by_day_bis.csv')
sales['Day'] = pd.to_datetime(sales['Day'])
studios = ['BV', 'WB', 'FX', 'SO', 'UN', 'PT']
for studio in studios:
    sales[studio] = 0
    for ind in sales.index:
        sales[studio][ind] = (((sales['Day'][ind] >= df['open_date']) & (sales['Day'][ind] <= df['close_date'])) & (df['studio'] == studio)).astype(int).sum()
        movcategories = ['MHM', 'LMM', 'MHH', 'LML', 'HLL', 'MML', 'NML', 'HHL', 'MMM', \
          'NMM', 'LHM', 'HML', 'MHL', 'NHM', 'HHM', 'NHL', 'LHL', 'NLL', \
          'HMM', 'NLM', 'HMH']
for movcategory in movcategories:
    sales[movcategory] = 0
    for ind in sales.index:
        sales[movcategory][ind] = (((sales['Day'][ind] >= df['open_date']) & (sales['Day'][ind] <= df['close_date'])) & (df['mov_cat'] == movcategory)).astype(int).sum()

############################
# Dataset 4: Split Dataset #
############################

# Split in train and test
sales_train = sales.iloc[:731]
sales_test = sales.iloc[-7:]

# Save datasets
sales_train.to_csv('./data/sales_train.csv', index=False)
sales_test.to_csv('./data/sales_test.csv', index=False)

###########################
# Baseline: Linear Model  #
###########################

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import r2_score
from statsmodels.tools.tools import add_constant
from statsmodels.regression.linear_model import OLS
from sklearn import linear_model
import time

# Load train dataset
train = pd.read_csv('./data/sales_train.csv')

# Select independent variables and dependent variable
X = train.drop(['Day', 'BOLETOS', 'month', 'day_name', 'day_type', 'day_tipo'], axis=1)
y = train.BOLETOS
Xcols = np.append(X.columns.values, ['constant'])

# Standarized values
ss = StandardScaler()
Xscaled = ss.fit_transform(X)
Xscaled = add_constant(Xscaled)

# Split
X_train, X_test, y_train, y_test = train_test_split(Xscaled, y, test_size = 0.25, random_state =42)

# Baseline: Linear model
lin_model = OLS(y_train, X_train).fit()
lin_pred = lin_model.predict(X_test)
lin_model_score = r2_score(y_test, lin_pred)

################################
# Functions to compare models  #
################################

def evaluate(model):        
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    model_score = r2_score(y_test, pred)
    
    param = cross_val_score(model, X, y, cv=5, scoring='r2', n_jobs=-1)
    
    print('==============================================================================')
    print('R^2 score:', model_score)
    print('Cross Validation Score: {:0.3} ± {:0.3}'.format(param.mean().round(3), param.std().round(3)))
    print('==============================================================================')
    
    df = pd.DataFrame(columns=['tickets', 'predictions', 'residuals'])
    df.attendance = y
    df.predictions = model.predict(Xscaled)
    df.residuals = df.attendance - df.predictions
    #df.plot(x='predictions', y='residuals', kind='scatter')
    #plt.show()
    print('==============================================================================')
    
def feature_imp(model):
    feat_imp = pd.DataFrame({'importance': model.feature_importances_}, index=Xcols).sort_values('importance', ascending=False).reset_index()
    print('==============================================================================')
    print('Most important features in the model:')
    print(feat_imp[0:9])
    print('==============================================================================')
    #feat_imp[0:20].plot(x='index', y='importance', kind='bar')
    #plt.xlabel('   ')
    #plt.ylabel('Importance')
    #plt.legend().set_visible(False)
    #plt.show()
    return feat_imp

def grid_search(model):
    params = {
        'max_features' : [0.1, 0.25, 0.5, 0.75, 'auto', 'sqrt'],
        'max_depth' : [10, 20, 30, None],
    }

    gs_rf = GridSearchCV(model, params, n_jobs=-1, cv=5)
    gs_rf.fit(X_train, y_train)

    print(gs_rf.best_params_)
    params_list = gs_rf.best_params_
    return params_list


######################################
# Models aming to beat the Baseline  #
######################################

# Load Dataset and Standarize
test = pd.read_csv('./data/sales_test.csv')
X_pred = test.drop(['Day', 'BOLETOS', 'month', 'day_name', 'day_type', 'day_tipo'], axis=1)
Xsc_pred = ss.fit_transform(X_pred)
Xsc_pred = add_constant(Xsc_pred)

# Linear Prediction
lin_pred = lin_model.predict(Xsc_pred)
print(lin_pred)

# -------------
# Random Forest
# -------------

RndFR = RandomForestRegressor(n_estimators=250)
evaluate(RndFR)
RFR = feature_imp(RndFR)

# 
# Prediction and Tuning
# 

# Tuning Random Forest
RndFR_params = grid_search(RndFR)
RndFR = RandomForestRegressor(n_estimators=250, max_depth=RndFR_params['max_depth'], max_features=RndFR_params['max_features'])
evaluate(RndFR)
Rnd_feat = feature_imp(RndFR)

# Prediction
forecast_RndFR = RndFR.predict(Xsc_pred)

# --------------
# Gradient Boost
# --------------

GradBR = GradientBoostingRegressor(n_estimators=300)
evaluate(GradBR)
GBR = feature_imp(GradBR)

# 
# Prediction and Tuning
# 

# Tuning Gradient Boost

GradBR_params = grid_search(GradBR)
GradBR = GradientBoostingRegressor(n_estimators=300, max_depth=GradBR_params['max_depth'], max_features=GradBR_params['max_features'])
evaluate(GradBR)
Grad_feat = feature_imp(GradBR)

# Prediction
forecast_GradBR = GradBR.predict(Xsc_pred)

# --------------
# Time Series
# --------------

# Load Dataset
time_s = pd.read_csv('./data/sales_train.csv')
time_s = time_s[['Day', 'BOLETOS']]

# Use datetime
time_s['Day'] = pd.to_datetime(time_s['Day'])
time_s.set_index('Day', inplace=True)

# Stationary test
from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    #Determing rolling statistics
    rolmean = timeseries.rolling(center=False, window=7).mean()
    rolstd = timeseries.rolling(center=False, window=7).std()
    #Plot rolling statistics:
    #plt.plot(timeseries, color='blue',label='Original')
    #plt.plot(rolmean, color='red', label='Rolling Mean')
    #plt.plot(rolstd, color='black', label = 'Rolling Std')
    #plt.legend(loc='best')
    #plt.title('Rolling Mean & Standard Deviation')
    #plt.show()
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)


# Using and testing moving averages
mov_avg = time_s['BOLETOS'].rolling(center=False, window=7).mean()
ts_mov_avg_diff = time_s['BOLETOS'] - mov_avg
ts_mov_avg_diff = pd.DataFrame(ts_mov_avg_diff)
ts_mov_avg_diff.dropna(inplace=True)
test_stationarity(ts_mov_avg_diff['BOLETOS'])

# Time Series Decomposition
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(ts_mov_avg_diff)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

# Forecasting with ARIMA
from statsmodels.tsa.arima_model import ARIMA

model = ARIMA(ts_mov_avg_diff, order=(7, 0, 1))  
results_AR = model.fit(disp=-1)

X_forecast = test[['Day', 'BOLETOS']]
X_forecast['Day'] = pd.to_datetime(X_forecast['Day'])

start_day = X_forecast['Day'][0]
end_day = X_forecast['Day'][6]
X_forecast.set_index('Day', inplace=True)

model = ARIMA(time_s, order=(7, 0, 1))  
results_AR = model.fit(disp=0)  
forecast_ARIMA = results_AR.predict(start=start_day, end=end_day)

#########################################################
# Comparing Results Models and Time Series Forecasting  #
#########################################################

# Using forecasting results
forecast_results = pd.DataFrame(forecast_ARIMA)
forecast_results.rename(columns={0: 'ARIMA'}, inplace=True)
forecast_results['RndFR'] = forecast_RndFR
forecast_results['GradBR'] = forecast_GradBR
forecast_results['LR'] = lin_pred

# Calculating some parameters
forecast_results['Min'] = forecast_results[['ARIMA','RndFR', 'GradBR', 'LR']].min(axis=1)
forecast_results['Max'] = forecast_results[['ARIMA','RndFR', 'GradBR', 'LR']].max(axis=1)
forecast_results['Mean'] = forecast_results[['ARIMA','RndFR', 'GradBR', 'LR']].mean(axis=1)
print(forecast_results.head(7))

# Plotting Results
train_plot = train[['Day', 'BOLETOS']]
train_plot['Day'] = pd.to_datetime(train_plot['Day'])
plt.figure(figsize=(12, 4))
plt.plot(time_s['2017-09-01':], label="actual")
plt.plot(forecast_results['ARIMA'], c="r", linestyle='--', label="ARIMA", alpha=0.7)
plt.plot(forecast_results['RndFR'], c="g", linestyle=':', label="Random Forest", alpha=0.7)
plt.plot(forecast_results['GradBR'], c="y", linestyle='-', label="Gradient Boost", alpha=0.7)
plt.plot(forecast_results['LR'], c="purple", linestyle='-.', label="Linear Regression", alpha=0.7)
plt.legend(loc='best')
plt.fill_between(forecast_results.index, forecast_results['Min'], forecast_results['Max'], color='b', alpha=0.2)
plt.show()