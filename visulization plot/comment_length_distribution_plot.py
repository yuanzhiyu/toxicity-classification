import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


data_dir = './jigsaw-unintended-bias-in-toxicity-classification/'
train = pd.read_csv(data_dir+'train.csv')
test = pd.read_csv(data_dir+'test.csv')

train_char_len = train['comment_text'].apply(len)
train_word_len = train['comment_text'].apply(lambda comment: len(comment.split()))
test_char_len = test['comment_text'].apply(len)
test_word_len = test['comment_text'].apply(lambda comment: len(comment.split()))


plt.figure()
sns.distplot(train_char_len, bins = 'auto', hist = True, norm_hist = True,
     label = 'Train set',)
sns.distplot(test_char_len, bins = 'auto', hist = True, norm_hist = True,
     label = 'Test set',)
plt.ylabel('Frequency', fontsize=12)
plt.xlabel('Text length', fontsize=12)
plt.title("Comment length (char level)", fontsize=14)
plt.legend(loc='upper right', fontsize=12)
plt.savefig('char-level.pdf')
plt.show()

plt.figure()
sns.distplot(train_word_len, bins = 'auto', hist = True, norm_hist = True,
     label = 'Train set',)
sns.distplot(train_word_len, bins = 'auto', hist = True, norm_hist = True,
     label = 'Test set',)
plt.ylabel('Frequency', fontsize=12)
plt.xlabel('Text length', fontsize=12)
plt.title("Comment length (word level)", fontsize=14)
plt.legend(loc='upper right', fontsize=12)
plt.savefig('word-level.pdf')
plt.show()