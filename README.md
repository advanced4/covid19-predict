# covid19-predict
A little python example that generates a graph showing multiple common prediction models

## Overview

This an example usage of common time series forecasting models, applied to COVID-19 data.
The expected format is a *.csv file that looks like the following:

    entry,total,date
    0,1,3/16/20
    1,2,3/17/20
    2,4,3/18/20
    3,13,3/19/20
    ...
    
where the first column is simply an incrementing integer,
the second column is the total confirmed cases, and the third column is the date

## Settings

Settings are directly in the *.py file, at the top

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
    
## Example output
![Covid19Example](https://github.com/advanced4/covid19-predict/example_output.png)