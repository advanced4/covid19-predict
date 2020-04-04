import pandas as pd
import os
import matplotlib.pyplot as plt
from datetime import date, timedelta, datetime
from settings import file_to_load

########### SETTINGS ############
skip_first_few = True
################################

def get_time_labels():
    edate = date.today() #+ timedelta(days=forecast_days)  # end date
    delta = edate - start_date  # as timedelta

    labels = []
    for i in range(0,delta.days + 1,1):
        day = start_date + timedelta(days=i)
        labels.append(str(day.month) + "/" + str(day.day))
    return labels

df = pd.read_csv('data' +os.path.sep + file_to_load+'.csv')
total_data = df['total_cum'].values.tolist()

previous = 0
counter = 0
roi_total = 0.0
roi_arr = []
aroi_arr = []
num_skipped = 0
for entry in total_data:
    # skip everything prior to 13 confirmed cases
    # this rules out some of the extreme values. i.e. 1 --> 3 is a 200% increase
    if skip_first_few and entry < 14:
        num_skipped += 1
        continue

    diff = entry - previous

    if previous == 0.0:
        previous = entry
        continue

    roi = diff/previous
    roi_total += roi
    previous = entry
    counter += 1

    print("Rate of increase on " + str(df['date'][num_skipped + counter]) + ": " + str(round(roi*100, 2)) + "%")

    roi_arr.append(round(roi,2))
    aroi_arr.append((roi_total/counter)*100)

start_date = datetime.strptime(df['date'][num_skipped+1], '%m/%d/%y').date()
print("Average rate of increase between: " + str(start_date) + " and " + str(datetime.strptime(df['date'][len(df['date'])-1], '%m/%d/%y').date()) + " -- " + str((roi_total/counter)*100))

print(roi_arr)
print(aroi_arr)

fig, ax = plt.subplots(figsize=(15, 10))
plt.plot([round(element*100, 2) for element in roi_arr])
plt.ylabel('% Rate of increase')
plt.xticks(ticks=range(0, len(get_time_labels())+1), labels=get_time_labels())
for label in ax.xaxis.get_ticklabels()[::2]:
    label.set_visible(False)
plt.show()


fig, ax = plt.subplots(figsize=(15, 10))
plt.plot(aroi_arr)
plt.ylabel('average % rate of increase')
plt.xticks(ticks=range(0, len(get_time_labels())+1), labels=get_time_labels())
for label in ax.xaxis.get_ticklabels()[::2]:
    label.set_visible(False)
plt.show()