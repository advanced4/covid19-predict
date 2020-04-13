import os
from datetime import date, timedelta, datetime

import matplotlib.pyplot as plt
import pandas as pd

from settings import file_to_load


def get_time_labels():
    edate = date.today()  # + timedelta(days=forecast_days)  # end date
    delta = edate - start_date  # as timedelta

    labels = []
    for i in range(0, delta.days + 1, 1):
        day = start_date + timedelta(days=i)
        labels.append(str(day.month) + "/" + str(day.day))
    return labels


df = pd.read_csv('data' + os.path.sep + file_to_load + '.csv')
total_data = df['total_cum'].values.tolist()

previous = 0
counter = 0
roi_total = 0.0
diff_arr = []
num_skipped = 0
for entry in total_data:
    diff = entry - previous
    previous = entry
    diff_arr.append(diff)

start_date = datetime.strptime(df['date'][num_skipped], '%m/%d/%y').date()

fig, ax = plt.subplots(figsize=(15, 10))

plt.bar(get_time_labels(), diff_arr)
plt.ylabel('Day by day increase')
# plt.xticks(ticks=range(0, len(get_time_labels())+1), labels=get_time_labels())
# for label in ax.xaxis.get_ticklabels()[::2]:
#    label.set_visible(False)
plt.show()
