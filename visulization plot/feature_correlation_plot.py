import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


train = pd.read_csv('ft-train.csv')
test = pd.read_csv('ft-test.csv')

features = [ 'ft_capitals', 'ft_caps_vs_length',
                     'ft_num_exclamation_marks', 'ft_num_question_marks', 'ft_num_punctuation',
                     'ft_num_symbols', 'ft_num_words', 'ft_num_unique_words', 'ft_words_vs_unique',
                     'ft_num_smilies', 'ft_num_special_punc', 'ft_num_strange_font', 'ft_num_toxic_words',
                     'ft_toxic_words_score', 'ft_num_good_words', 'ft_good_words_score']
columns = ('target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat',  'likes', 'disagree', 'sexual_explicit')
rows = [{c:train[f].corr(train[c]) for c in columns} for f in features]
train_correlations = pd.DataFrame(rows, index=features)
print(train_correlations)

plt.figure(figsize=(12, 7))
ax = sns.heatmap(train_correlations,  vmin=-0.1, vmax=0.6, robust=True,  linewidths=0.1, annot=True, annot_kws={"fontsize":8})
plt.savefig("feature.pdf")
plt.show()