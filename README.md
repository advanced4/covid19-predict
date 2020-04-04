# covid19-predict
A little python example that generates a graph showing multiple common prediction models

# Common Models
## Overview

This an example usage of common time series forecasting models, applied to COVID-19 data.
The expected format is a *.csv file that looks like the following:

    total_cum,total_recovered,total_deaths,date
    1,0,0,3/16/20
    2,1,0,3/17/20
    4,1,1,3/18/20
    8,2,2,3/19/20
    ...
        
where the first column is the total cumulative confirmed cases of that date,
the second column is the total cumulative recovered cases as of that day, the third column is the total cumulative
 confirmed deaths as of that day, and finally the date

I used Python 3.8.2 for this. Main requirements are sklearn, pandas, statsmodels, and matplotlib

## Settings

Most Settings are directly in the *.py file, at the top

    # How many days in the future to forecast
    forecast_days = 7
    
    # Show how well the models fit on the actual/existing data.
    # If false, it only shows the prediction/future portions
    show_fit_on_actual = False
    
    # Included are several models, some of which probably suck.
    # The exponential one usually throws off the scale of everything else,
    # so you can turn it off here
    show_that_one_giant_red_line = True
    
    # Some models definitely will suck, such as linear regression, and SVR sigmoid
    # you can turn them off here
    ignore_shitty_ones = True
    
    # The name of the file under the local 'data' dir, without the extension
    file_to_load = "example"

There is also an example-settings.py file. Rename to settings.py to use
## Example output
![Covid19Example](https://github.com/advanced4/covid19-predict/raw/master/example_output.png)

# SIR Model
## Overview
Based mostly on work from https://github.com/Lewuathe/COVID19-SIR and https://www.kaggle.com/saga21/covid-global-forecast-sir-model-ml-regressions/notebook

see https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology#The_SIR_model

The SIR Model:
"The model consists of three compartments: S for the number of susceptible, I for the number of infectious, and R for the number recovered (or immune) individuals. This model is reasonably predictive for infectious diseases which are transmitted from human to human"

An example of what it looks like:
![SIRExample](https://github.com/advanced4/covid19-predict/raw/master/example_sir.png)

Fitting the SIR model to data
![SIRFitExample](https://github.com/advanced4/covid19-predict/raw/master/example_sir_fit.png)

Predicting w/ SIR model
Fitting the SIR model to data
![SIRPrediction](https://github.com/advanced4/covid19-predict/raw/master/example_sir_prediction.png)