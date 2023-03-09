import pandas as pd
from math import log2
from matplotlib import pyplot as plt
from tqdm import tqdm

data_dir = './jigsaw-unintended-bias-in-toxicity-classification/'
train = pd.read_csv(data_dir+'train.csv')

# [key, value] : [word, [word frequency, sentence frequency]]
non_toxic_comment_words = dict()
toxic_comment_words = dict()

non_toxic_comment_counts = 0
toxic_comment_counts = 0
non_toxic_comment_word_counts = 0
toxic_comment_word_counts = 0


def get_clean_word(token):
    removed_symbols = {'.', ',', '!', '?', '\'', '\"', '…', '“', '”'}
    begin = 0
    end = len(token)
    while begin < end  and token[begin] in removed_symbols:
        begin += 1
    while begin < end  and token[end-1] in removed_symbols:
        end -= 1
    return token[begin:end]

for value, text in tqdm(zip(train['target'], train['comment_text'])):
    tokens = [get_clean_word(x) for x in text.lower().split()]
    token_set = set(tokens)
    if value>=0.5:
        toxic_comment_word_counts += len(tokens)
        toxic_comment_counts += 1
        for w in tokens:
            if w in toxic_comment_words:
                toxic_comment_words[w][0] += 1
            else:
                toxic_comment_words[w] = [1, 0]
        for w in token_set:
            toxic_comment_words[w][1] += 1
    else:
        non_toxic_comment_word_counts += len(tokens)
        non_toxic_comment_counts += 1
        for w in tokens:
            if w in non_toxic_comment_words:
                non_toxic_comment_words[w][0] += 1
            else:
                non_toxic_comment_words[w] = [1, 0]
        for w in token_set:
            non_toxic_comment_words[w][1] += 1


weight_adjust = (float(non_toxic_comment_counts)*non_toxic_comment_word_counts) / (toxic_comment_counts*toxic_comment_word_counts)


# toxic word
#########################################
toxic_candidates = dict()
word_only_in_toxic_comments = []
for w, count in tqdm(toxic_comment_words.items()):
    count_word, count_sentence = count
    if count_word<=5 or count_sentence<=5:
        continue

    # 1 is used as default count for non existed words in non-toxic comments
    if w not in non_toxic_comment_words:
        word_only_in_toxic_comments.append(w)
        toxic_candidates[w] = log2(weight_adjust * float(toxic_comment_words[w][0] * toxic_comment_words[w][1]) / (1 * 1))
    else:
        toxic_candidates[w] = log2(weight_adjust * float(toxic_comment_words[w][0] * toxic_comment_words[w][1]) / ((1+non_toxic_comment_words[w][0]) * (1+non_toxic_comment_words[w][1])))


sorted_toxic_candidates = sorted(toxic_candidates.items(), key= lambda x: x[1], reverse=True)
tokens = [x[0] for x in sorted_toxic_candidates]
weights = [x[1] for x in sorted_toxic_candidates]
toxic_candidates_frame = pd.DataFrame({"Token": tokens, "Weight":weights})
toxic_candidates_frame.to_csv("toxic_word_candidates.csv")


word_toxic_weights = dict()
swear_word_list = []
weight_threshold = 3
for word, weight in tqdm(sorted_toxic_candidates):
    if weight<=weight_threshold:
        break
    word_toxic_weights[word] = weight

    if toxic_comment_words[word][0]>15 and toxic_comment_words[word][1]>15 and weight>5:
        swear_word_list.append(word)

# good word
#########################################
good_candidates = dict()
word_only_in_good_comments = []
for w, count in tqdm(non_toxic_comment_words.items()):
    count_word, count_sentence = count
    if count_word<=10 or count_sentence<=10:
        continue

    # 1 is used as default count for non existed words in non-toxic comments
    if w not in toxic_comment_words:
        word_only_in_good_comments.append(w)
        good_candidates[w] = log2(float(non_toxic_comment_words[w][0] * non_toxic_comment_words[w][1]) / (1 * 1) / weight_adjust)
    else:
        good_candidates[w] = log2(float(non_toxic_comment_words[w][0] * non_toxic_comment_words[w][1]) / ((1+toxic_comment_words[w][0]) * (1+toxic_comment_words[w][1])) / weight_adjust)



sorted_good_candidates = sorted(good_candidates.items(), key= lambda x: x[1], reverse=True)
tokens = [x[0] for x in sorted_good_candidates]
weights = [x[1] for x in sorted_good_candidates]
good_candidates_frame = pd.DataFrame({"Token": tokens, "Weight":weights})
good_candidates_frame.to_csv("good_word_candidates.csv")

word_good_weights = dict()
good_word_list = []
weight_threshold = 4
for word, weight in tqdm(sorted_good_candidates):
    if weight<=weight_threshold:
        break
    word_good_weights[word] = weight

    if non_toxic_comment_words[word][0]>100 and non_toxic_comment_words[word][1]>100 and weight>5:
        good_word_list.append(word)