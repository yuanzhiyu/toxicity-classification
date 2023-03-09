import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

train = pd.read_csv('./jigsaw-unintended-bias-in-toxicity-classification/train.csv')
test = pd.read_csv('./jigsaw-unintended-bias-in-toxicity-classification/test.csv')


plt.figure(figsize=(11,8))
plot = train.target.plot(kind='hist', bins=10, color='teal')
ax = plot.axes
for p in ax.patches:
    ax.annotate(f'{p.get_height() * 100 / train.shape[0]:.2f}%',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center',
                va='center',
                fontsize=16,
                color='black',
                xytext=(0,7),
                textcoords='offset points'
                )
ax.tick_params(axis='y',labelsize=17)
ax.tick_params(axis='x',labelsize=19)
ax.set_ylabel('Frequency', fontsize=20)
ax.set_title('Target Distribution (Raw)', fontsize=22)
plt.savefig("raw.pdf")
plt.show()
plt.close()


import copy
new_train = copy.deepcopy(train['target'])
new_data = ['Toxic' if x>=0.5 else 'Non-toxic' for x in new_train]
plt.figure(figsize=(11, 8))
plot = sns.countplot(new_data)
ax = plot.axes

for p in ax.patches:
    ax.annotate(f'{p.get_height() * 100 / new_train.shape[0]:.2f}%',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center',
                va='center',
                fontsize=17,
                color='black',
                xytext=(0, 7),
                textcoords='offset points')
ax.tick_params(axis='y',labelsize=17)
ax.tick_params(axis='x',labelsize=19)
ax.set_ylabel('',fontsize=0)
plt.title('Target Distribution (Binary)', fontsize=22)
plt.savefig("binary.pdf")
plt.show()
plt.close()