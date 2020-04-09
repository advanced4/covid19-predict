import os
from statsmodels.tsa.api import Holt
import pandas as pd
from sklearn import svm
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from datetime import date, timedelta, datetime
import numpy as np
from settings import file_to_load

########## SETTINGS ##########
forecast_days = 20
show_fit_on_actual = False
show_that_one_giant_red_line = True
ignore_shitty_ones = True
##############################

if ignore_shitty_ones:
    kernels = ['linear', 'poly']
else:
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']  # for SVM stuf

def getplotdata(clf):
    x,y = [],[]

    if show_fit_on_actual:
        start = 0
    else:
        start = tomorrow

    for i in range(start, tomorrow+forecast_days):
        x.append(i)
        y.append(clf.predict([[i]])[0])
    return x,y


def predict_forecast_days(type, clf):
    #print(type)
    #for i in range(0,forecast_days+1):
    #    print("\t" + str(tomorrow+i) + str(clf.predict([[tomorrow+i]])))
    x1, y1 = getplotdata(clf)
    plt.plot(x1, y1, label=type)


def get_time_labels():
    edate = date.today() + timedelta(days=forecast_days)  # end date
    delta = edate - start_date  # as timedelta

    labels = []
    for i in range(0,delta.days + 1,1):
        day = start_date + timedelta(days=i)
        labels.append(str(day.month) + "/" + str(day.day))
    return labels

##############################################################
df = pd.read_csv('data' +os.path.sep + file_to_load+'.csv')
start_date = datetime.strptime(df['date'][0], '%m/%d/%y').date()

total_data = df['total_cum'].values.tolist()
time_data = list(range(0, len(total_data)))
tomorrow = len(time_data)

air = pd.Series(total_data, time_data)

fit1 = Holt(air).fit(smoothing_level=0.8, smoothing_slope=0.2, optimized=False)
fcast1 = fit1.forecast(forecast_days)

fit2 = Holt(air, exponential=True).fit(smoothing_level=0.8, smoothing_slope=0.2, optimized=False)
fcast2 = fit2.forecast(forecast_days)

fit3 = Holt(air, damped=True).fit(smoothing_level=0.8, smoothing_slope=0.2)
fcast3 = fit3.forecast(forecast_days)

ax = air.plot(color="black", marker="o", figsize=(12,8), legend=True, label="actual")

if show_fit_on_actual:
    fit1.fittedvalues.plot(ax=ax, color='blue', marker="o", label="Holt's Linear Trend")
fcast1.plot(ax=ax, color='blue', legend=True, label="Holt's Linear trend - Pred.")

if show_that_one_giant_red_line:
    if show_fit_on_actual:
        fit2.fittedvalues.plot(ax=ax, color='red', marker="o", label="Holt's Exponential Trend")
    fcast2.plot(ax=ax, color='red', legend=True, label="Holt's Exponential Trend - Pred.")

if show_fit_on_actual:
    fit3.fittedvalues.plot(ax=ax, color='green', marker="o", label="Holt's Additive damped trend")
fcast3.plot(ax=ax, color='green', legend=True, label="Holt's Additive damped trend - Pred.")

######################

y = np.asarray(total_data).reshape(-1,1)
x = np.asarray(time_data).reshape(-1,1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.01, random_state=422)
#####################
clf = LinearRegression(n_jobs=-1)
clf.fit(x_train, y_train)
predict_forecast_days("Linear Regression", clf)
#####################
if not ignore_shitty_ones:
    clf = SVR()
    clf.fit(x_train, y_train)
    predict_forecast_days("SVR", clf)
#####################
for k in kernels:
    clf = svm.SVR(kernel=k)
    clf.fit(x_train, y_train)
    predict_forecast_days("SVR "+k, clf)

# idk why i have to do this
plt.xticks(ticks=range(0, len(get_time_labels())+1), labels=get_time_labels())
plt.xticks(rotation=45)
plt.ylabel("Total confirmed cases")
plt.title(file_to_load)
plt.legend()
plt.grid()
plt.show()
