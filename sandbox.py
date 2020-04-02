import pandas as pd
import os

########### SETTINGS ############
file_to_load = "example"
skip_first_few = True
################################

df = pd.read_csv('data' +os.path.sep + file_to_load+'.csv')

total_data = df['total_cum'].values.tolist()

previous = 0
counter = 0
roi_total = 0.0
for entry in total_data:
    # skip everything prior to 10 confirmed cases
    # this rules out some of the extreme values. i.e. 1 --> 3 is a 200% increase
    if skip_first_few and entry < 14:
        continue
    diff = entry - previous
    if previous == 0.0:
        previous = entry
        continue
    roi = diff/previous
    print("Rate of increase: " + str(round(roi,2)))
    roi_total += roi
    previous = entry
    counter += 1

print("Average rate of increase: " + str((roi_total/counter)*100))
