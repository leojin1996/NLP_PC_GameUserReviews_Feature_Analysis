#!/usr/bin/env python
#!/usr/bin/env python
# coding: utf-8
# Load datatest into Jupternoteboook
import numpy as np
import modin.pandas as pd
import time
user_comments = pd.read_csv('metacritic_game_user_comments.csv',index_col = 0)
game_inf = pd.read_csv('metacritic_game_info.csv',index_col = 0)
# extract PC games from game_inf dataset and create new dataset called "PC_games"
PC_games = game_inf[game_inf['Platform'] == 'PC']
#create a list that lists game genres
game_genres = ['Action','Adventure','Role-Playing','Strategy','Sports']
# define a fuction that filters genres
def genre_filter(genre):
    genre_list = genre.split(';')
    for g in game_genres:
        if g in genre_list:
            return g
        else:
            pass
# Apply the function above to filter genres and create a new column called 'genre_modified'
PC_games['Genre_modified'] = PC_games['Genre'].apply(lambda x: genre_filter(x))
PC_games['Genre_modified'].unique()
#delete row with None in column named 'Genre_modified'
PC_games = PC_games.dropna(subset=['Genre_modified'])
PC_games['Genre_modified'].unique()
#Create table for PC_Games with "Action" genre
PC_games_Action = PC_games[PC_games['Genre_modified'] == 'Action']
# extract PC games from user_comments dataset which show user comments from different platforms
PC_user_comments = user_comments[user_comments['Platform'] =='PC']
PC_title = PC_user_comments['Title'].unique()
PC_title_list = PC_title.tolist()
#title from comments database
title_words = []
for title in PC_title_list:
    words = title.split()
    for w in words:
        w = w.lower()
        title_words.append(w)
def check_float(comment):
    if type(comment) is float:
        return 'Float'
PC_user_comments['is_float'] = PC_user_comments['Comment'].apply(lambda x: check_float(x))
PC_user_comments = PC_user_comments[PC_user_comments['is_float'] !='Float']
# remove extra space and punctuation marks
import re
import string
exclude = set(string.punctuation)
PC_user_comments['Comment'] = PC_user_comments['Comment'].apply(lambda x: x.strip('\n'))
PC_user_comments['Comment'] = PC_user_comments['Comment'].apply(lambda x: x.strip('\n'))
PC_user_comments['Comment'] = PC_user_comments['Comment'].apply(lambda x: x.strip('\t'))
PC_user_comments['Comment'] = PC_user_comments['Comment'].apply(lambda x: x.replace('\n',' '))
PC_user_comments['Comment'] = PC_user_comments['Comment'].apply(lambda x: x.replace('\t',' '))
PC_user_comments['Comment'] = PC_user_comments['Comment'].apply(lambda x: re.sub(r"\s+"," ",x))
PC_user_comments['Comment'] = PC_user_comments['Comment'].apply(lambda x: re.sub(r"^\s+","",x))
PC_user_comments['Comment'] = PC_user_comments['Comment'].apply(lambda x: re.sub(r"\s+$","",x))
PC_user_comments['Comment'] = PC_user_comments['Comment'].apply(lambda x: ''.join(ch for ch in x if ch not in exclude))
# remove stop words from user comments
from nltk.corpus import stopwords
stop = stopwords.words('english')
PC_user_comments['Comment_without_stopword_title_lem'] = PC_user_comments['Comment'].apply(lambda x: ' '.join([word for word in x.lower().split() if word not in (stop)]))
PC_user_comments['Comment_without_stopword_title_lem'] = PC_user_comments['Comment_without_stopword_title_lem'].apply(lambda x: ' '.join([word for word in x.split() if word not in (title_words)]))

# define function that groups the different forms of words which are syntactically different but semantically equal
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
def Lemmatization(comment):
    comment_list = []
    comment = comment.lower().split()
    for word in comment:
        for tag in ['a', 'n', 'r', 'v']:
            w = lemmatizer.lemmatize(word,tag)
        comment_list.append(w)
    return ' '.join(comment_list)
PC_user_comments['Comment_without_stopword_title_lem'] = PC_user_comments['Comment_without_stopword_title_lem'].apply(lambda x: Lemmatization(x))

PC_user_comments.head()
from textblob import TextBlob, Word
def lemmatize_second(comments):
    sent = TextBlob(comments)
    return " ". join([w.lemmatize() for w in sent.words])
PC_user_comments['Comment_without_stopword_title_lem'] = PC_user_comments['Comment_without_stopword_title_lem'].apply(lambda x: lemmatize_second(x))
PC_user_comments.head()
# Build a function that split combined words
import wordsegment
from wordsegment import load, segment
load()
def segment_word(comments):
    words = []
    comments = comments.split()
    for w in comments:
        a = " ".join(segment(w))
        words.append(a)
    return " ".join(words)
PC_user_comments['Comment_without_stopword_title_lem'] = PC_user_comments['Comment_without_stopword_title_lem'].apply(lambda x: segment_word(x))
PC_user_comments.iloc[30000:33000,:]

#build a function that correct words with spelling errors
from spellchecker import SpellChecker
spell = SpellChecker(distance=1)
def correct_word_spelling(comments):
    correct_words = []
    comments = comments.split()
    for word in comments:
        correct_words.append(spell.correction(word))
    return ' '.join(correct_words)
PC_user_comments['Comment_without_stopword_title_lem'] = PC_user_comments['Comment_without_stopword_title_lem'].apply(lambda x: correct_word_spelling(x))

PC_user_comments['Comment_without_stopword_title_lem'] = PC_user_comments['Comment_without_stopword_title_lem'].apply(lambda x: ' '.join([word for word in x.split() if word not in (title_words)]))
PC_user_comments = PC_user_comments[['Title','Comment','Username','Comment_without_stopword_title_lem']]
# define a function that checks whether the game in PC_games dataset is in PC_user_comments dataset
def title_check(title):
        if str(title) in PC_title:
            return title
        else:
            return 'No'
# apply the function above that check whether the game in PC_games dataset is in PC_user_comments dataset
PC_games_Action['Title_check'] = PC_games_Action['Title'].apply(lambda x: title_check(x))
PC_games_Action = PC_games_Action[PC_games_Action['Title_check'] != 'No']
#PC_games_Action.head()
Action_game_title = PC_games_Action['Title'].unique()
#Create Action games comments dataset
Action_games_comments = pd.DataFrame()
for title in Action_game_title:
    action_game_comments = PC_user_comments[PC_user_comments['Title'] == title]
    Action_games_comments =Action_games_comments.append(action_game_comments)
Action_Volume = len(Action_games_comments)
Action_Volume
len(Action_games_comments['Username'].unique())
Action_games_comments = Action_games_comments.reset_index()
Action_games_comments = Action_games_comments[['Title','Comment','Username','Comment_without_stopword_title_lem','genre']]
len(Action_games_comments)
Action_games_comments.head()
Action_games_comments['one_word'] = Action_games_comments['Comment_without_stopword_title_lem'].apply(lambda x: x.split())
Action_games_comments['one_word'] = Action_games_comments['one_word'].apply(lambda x: [word for word in x if word not in stop])
one_word_list = []
for comment_list in Adventure_games_comments_train['one_word']:
    for word in comment_list:
        one_word_list.append(word)
one_word_dict = dict()
for n in one_word_list:
    if n not in one_word_dict.keys():
        one_word_dict[n] = 1
    else:
        one_word_dict[n] = one_word_dict[n] + 1
# create a new column that lists tokens with two words
import nltk
from nltk.collocations import *
bigram_measures = nltk.collocations.BigramAssocMeasures()
trigram_measures = nltk.collocations.TrigramAssocMeasures()
def bi_tokens(comments):
    tokens = nltk.wordpunct_tokenize(comments)
    finder = BigramCollocationFinder.from_words(tokens)
    scored = finder.score_ngrams(bigram_measures.raw_freq)
    return sorted(finder.ngram_fd.items(), key=lambda t: (-t[1], t[0]))
Action_games_comments['two_words_freq'] = Action_games_comments['Comment_without_stopword_title_lem'].apply(lambda x:bi_tokens(x))
# count frequency of tokens with two words
two_words_list = []
for two_word_list in Action_games_comments_train['two_words_freq']:
    for two_word in two_word_list:
        two_words_list.append(two_word)
two_words = dict()
for i in range(len(two_words_list)):
    if two_words_list[i][0] not in two_words.keys():
        two_words[two_words_list[i][0]] = two_words_list[i][1]
    else:
        two_words[two_words_list[i][0]] = two_words[two_words_list[i][0]] + two_words_list[i][1]
two_words_frequency = pd.DataFrame(list(two_words.items()), columns=['two_words_Features', 'Frequency'])
two_words_frequency = two_words_frequency.sort_values(by = 'Frequency', ascending = False)
two_words_frequency.to_csv('action_games_twowords_features.csv')
# create a new column that lists tokens with three words
import nltk
from nltk.collocations import *
bigram_measures = nltk.collocations.BigramAssocMeasures()
trigram_measures = nltk.collocations.TrigramAssocMeasures()
def tri_tokens(comments):
    tokens = nltk.wordpunct_tokenize(comments)
    finder = TrigramCollocationFinder.from_words(tokens)
    scored = finder.score_ngrams(trigram_measures.raw_freq)
    return sorted(finder.ngram_fd.items(), key=lambda t: (-t[1], t[0]))
Action_games_comments['three_words_freq'] = Action_games_comments['Comment_without_stopword_title_lem'].apply(lambda x:tri_tokens(x))
# count frequency of tokens with three words
three_words_list = []
for three_word_list in Action_games_comments_train['three_words_freq']:
    for three_word in three_word_list:
        three_words_list.append(three_word)
three_words = dict()
for i in range(len(three_words_list)):
    if three_words_list[i][0] not in three_words.keys():
        three_words[three_words_list[i][0]] = three_words_list[i][1]
    else:
        three_words[three_words_list[i][0]] = three_words[three_words_list[i][0]] + three_words_list[i][1]
three_words_frequency = pd.DataFrame(list(three_words.items()), columns=['three_words_Features', 'Frequency'])
three_words_frequency = three_words_frequency.sort_values(by = 'Frequency', ascending = False)
three_words_frequency.to_csv('action_games_threewords_features.csv')
# define a function that make sentiment analysis on each comment and score each comment
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk import sent_tokenize, word_tokenize, pos_tag
def penn_to_wn(tag):
    """
    Convert between the PennTreebank tags to simple Wordnet tags
    """
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return None
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
def swn_polarity(text):
    """
    """

    sentiment = 0.0
    tokens_count = 0


    raw_sentences = sent_tokenize(text)
    for raw_sentence in raw_sentences:
        tagged_sentence = pos_tag(word_tokenize(raw_sentence))

        for word, tag in tagged_sentence:
            wn_tag = penn_to_wn(tag)
            if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
                continue

            lemma = lemmatizer.lemmatize(word, pos=wn_tag)
            if not lemma:
                continue

            synsets = wn.synsets(lemma, pos=wn_tag)
            if not synsets:
                continue

            # Take the first sense, the most common
            synset = synsets[0]
            swn_synset = swn.senti_synset(synset.name())

            sentiment += swn_synset.pos_score() - swn_synset.neg_score()
            tokens_count += 1
    if not tokens_count:
        return 0
    if sentiment > 5:
        return 5
    elif sentiment <-5:
        return -5
    else:
        return sentiment
import nltk
ps = PorterStemmer()
words_data = ['good']
words_data = [ps.stem(x) for x in words_data]
pos_val = nltk.pos_tag(words_data)
senti_val = [get_sentiment(x,y) for (x,y) in pos_val]
# apply function above into Action games comments
Action_games_comments['Sentiment_Score'] = Action_games_comments['Comment'].apply(lambda x:swn_polarity(x))
Action_games_comments.to_csv('Action_games_comments.csv')
def count_sentiment_scores(scores_list):
    positive = []
    negative = []
    neutral = []
    for score in scores_list:
        if score >= 2:
            positive.append(score)
        elif score < 0:
            negative.append(score)
        else:
            neutral.append(score)
    percent_positive = len(positive)/len(scores_list)
    percent_negative = len(negative)/len(scores_list)
    percent_neutral = len(neutral)/len(scores_list)
    return 'positive: {}, negative: {}, neutral: {}'.format(percent_positive,percent_negative,percent_neutral)

import numpy as np
import pandas as pd
def sentiment_scores_avg_median_std(features):
    sentiment_scores = []
    for i,one_word in enumerate(Adventure_games_comments_train['one_word']):
        if features in one_word:
            sentiment_scores.append(Adventure_games_comments_train['Sentiment_Score'][i])
    return (count_sentiment_scores(sentiment_scores), len(sentiment_scores))
#Create table for PC_Games with "Adventure" genre
PC_games_Adventure = PC_games[PC_games['Genre_modified'] == 'Adventure']
#PC_games_Adventure.head()
PC_games_Adventure['Title_check'] = PC_games_Adventure['Title'].apply(lambda x: title_check(x))
PC_games_Adventure = PC_games_Adventure[PC_games_Adventure['Title_check'] != 'No']
#PC_games_Adventure.head(100)
PC_game_Adventure_title = PC_games_Adventure['Title'].unique()
#Create Adventure games comments dataset
Adventure_games_comments = pd.DataFrame()
for title in PC_game_Adventure_title:
    Adventure_game_comments = PC_user_comments[PC_user_comments['Title'] == title]
    Adventure_game_comments['genre'] = 'Adventure'
    Adventure_games_comments =Adventure_games_comments.append(Adventure_game_comments)
len(Adventure_games_comments)
len(Adventure_games_comments['Username'].unique())
Adventure_games_comments = Adventure_games_comments.reset_index()
Adventure_games_comments = Adventure_games_comments[['Title','Comment','Username','Comment_without_stopword_title_lem','genre']]
Adventure_games_comments.head()
Adventure_games_comments['one_word'] = Adventure_games_comments['Comment_without_stopword_title_lem'].apply(lambda x: x.split())
Adventure_games_comments['one_word'] = Adventure_games_comments['one_word'].apply(lambda x: [word for word in x if word not in stop])
one_word_list = []
for comment_list in Adventure_games_comments_train['one_word']:
    for word in comment_list:
        one_word_list.append(word)
one_word_dict = dict()
for n in one_word_list:
    if n not in one_word_dict.keys():
        one_word_dict[n] = 1
    else:
        one_word_dict[n] = one_word_dict[n] + 1
#extract nouns verbs and adjective from user comments
Adventure_games_comments_train['comment_noun'] = Adventure_games_comments_train['Comment_without_stopword_title_lem'].apply(lambda x: noun(x))
Adventure_games_comments_train['comment_adjective'] = Adventure_games_comments_train['Comment_without_stopword_title_lem'].apply(lambda x: adjective(x))
# create a new column that lists tokens with two words
import nltk
from nltk.collocations import *
bigram_measures = nltk.collocations.BigramAssocMeasures()
trigram_measures = nltk.collocations.TrigramAssocMeasures()
def bi_tokens(comments):
    tokens = nltk.wordpunct_tokenize(comments)
    finder = BigramCollocationFinder.from_words(tokens)
    scored = finder.score_ngrams(bigram_measures.raw_freq)
    return sorted(finder.ngram_fd.items(), key=lambda t: (-t[1], t[0]))
Adventure_games_comments['two_words_freq'] = Adventure_games_comments['Comment_without_stopword_title_lem'].apply(lambda x:bi_tokens(x))
# count frequency of tokens with two words
two_words_list = []
for two_word_list in Adventure_games_comments_train['two_words_freq']:
    for two_word in two_word_list:
        two_words_list.append(two_word)
two_words = dict()
for i in range(len(two_words_list)):
    if two_words_list[i][0] not in two_words.keys():
        two_words[two_words_list[i][0]] = two_words_list[i][1]
    else:
        two_words[two_words_list[i][0]] = two_words[two_words_list[i][0]] + two_words_list[i][1]
two_words_frequency = pd.DataFrame(list(two_words.items()), columns=['two_words_Features', 'Frequency'])
two_words_frequency = two_words_frequency.sort_values(by = 'Frequency', ascending = False)

two_words_frequency.to_csv('adventure_games_twowords_features.csv')
#two_words_frequency.to_csv('action_games_twowords_features.csv')
# create a new column that lists tokens with three words
import nltk
from nltk.collocations import *
bigram_measures = nltk.collocations.BigramAssocMeasures()
trigram_measures = nltk.collocations.TrigramAssocMeasures()
def tri_tokens(comments):
    tokens = nltk.wordpunct_tokenize(comments)
    finder = TrigramCollocationFinder.from_words(tokens)
    scored = finder.score_ngrams(trigram_measures.raw_freq)
    return sorted(finder.ngram_fd.items(), key=lambda t: (-t[1], t[0]))
Adventure_games_comments['three_words_freq'] = Adventure_games_comments['Comment_without_stopword_title_lem'].apply(lambda x:tri_tokens(x))
# count frequency of tokens with three words
three_words_list = []
for three_word_list in Adventure_games_comments_train['three_words_freq']:
    for three_word in three_word_list:
        three_words_list.append(three_word)
three_words = dict()
for i in range(len(three_words_list)):
    if three_words_list[i][0] not in three_words.keys():
        three_words[three_words_list[i][0]] = three_words_list[i][1]
    else:
        three_words[three_words_list[i][0]] = three_words[three_words_list[i][0]] + three_words_list[i][1]
three_words_frequency = pd.DataFrame(list(three_words.items()), columns=['three_words_Features', 'Frequency'])
three_words_frequency = three_words_frequency.sort_values(by = 'Frequency', ascending = False)
three_words_frequency.to_csv('adventure_games_threewords_features.csv')
# apply function above into Action games comments
Adventure_games_comments['Sentiment_Score'] = Adventure_games_comments['Comment'].apply(lambda x:swn_polarity(x))
Adventure_games_comments_train = Adventure_games_comments_train.reset_index()
Adventure_games_comments.to_csv('Adventure_games_features_sentiment.csv')
PC_games_Role_Playing = PC_games[PC_games['Genre_modified'] == 'Role-Playing']
PC_games_Role_Playing['Title_check'] = PC_games_Role_Playing['Title'].apply(lambda x: title_check(x))
PC_games_Role_Playing = PC_games_Role_Playing[PC_games_Role_Playing['Title_check'] != 'No']
PC_game_Role_Playing_title = PC_games_Role_Playing['Title'].unique()
#Create Role playing games comments dataset
Role_Playing_games_comments = pd.DataFrame()
for title in PC_game_Role_Playing_title:
    Role_Playing_game_comments = PC_user_comments[PC_user_comments['Title'] == title]
    Role_Playing_game_comments['genre'] = 'Role Playing'
    Role_Playing_games_comments =Role_Playing_games_comments.append(Role_Playing_game_comments)
len(Role_Playing_games_comments)
len(Role_Playing_games_comments['Username'].unique())
Role_Playing_games_comments = Role_Playing_games_comments.reset_index()
Role_Playing_games_comments = Role_Playing_games_comments[['Title','Comment','Username','Comment_without_stopword_title_lem','genre']]
Role_Playing_games_comments.head()
#extract nouns verbs and adjective from user comments
Role_Playing_games_comments_train['comment_noun'] = Role_Playing_games_comments_train['Comment_without_stopword_title_lem'].apply(lambda x: noun(x))
Role_Playing_games_comments_train['comment_adjective'] = Role_Playing_games_comments_train['Comment_without_stopword_title_lem'].apply(lambda x: adjective(x))
#remove stopwords again
Role_Playing_games_comments_train['comment_adjective'] = Role_Playing_games_comments_train['comment_adjective'].apply(lambda x: [word for word in x if word not in stop])
Role_Playing_games_comments_train['comment_noun'] = Role_Playing_games_comments_train['comment_noun'].apply(lambda x: [word for word in x if word not in stop])
Role_Playing_games_comments['one_word'] = Role_Playing_games_comments['Comment_without_stopword_title_lem'].apply(lambda x: x.split())
Role_Playing_games_comments['one_word'] = Role_Playing_games_comments['one_word'].apply(lambda x: [word for word in x if word not in stop])
# create a new column that lists tokens with two words
import nltk
from nltk.collocations import *
bigram_measures = nltk.collocations.BigramAssocMeasures()
trigram_measures = nltk.collocations.TrigramAssocMeasures()
def bi_tokens(comments):
    tokens = nltk.wordpunct_tokenize(comments)
    finder = BigramCollocationFinder.from_words(tokens)
    scored = finder.score_ngrams(bigram_measures.raw_freq)
    return sorted(finder.ngram_fd.items(), key=lambda t: (-t[1], t[0]))
Role_Playing_games_comments['two_words_freq'] = Role_Playing_games_comments['Comment_without_stopword_title_lem'].apply(lambda x:bi_tokens(x))
# count frequency of tokens with two words
two_words_list = []
for two_word_list in Role_Playing_games_comments_train['two_words_freq']:
    for two_word in two_word_list:
        two_words_list.append(two_word)
two_words = dict()
for i in range(len(two_words_list)):
    if two_words_list[i][0] not in two_words.keys():
        two_words[two_words_list[i][0]] = two_words_list[i][1]
    else:
        two_words[two_words_list[i][0]] = two_words[two_words_list[i][0]] + two_words_list[i][1]
two_words_frequency = pd.DataFrame(list(two_words.items()), columns=['two_words_Features', 'Frequency'])
two_words_frequency = two_words_frequency.sort_values(by = 'Frequency', ascending = False)
two_words_frequency.to_csv('Role_Playing_games_twowords_features.csv')
#two_words_frequency.to_csv('action_games_twowords_features.csv')
# create a new column that lists tokens with three words
import nltk
from nltk.collocations import *
bigram_measures = nltk.collocations.BigramAssocMeasures()
trigram_measures = nltk.collocations.TrigramAssocMeasures()
def tri_tokens(comments):
    tokens = nltk.wordpunct_tokenize(comments)
    finder = TrigramCollocationFinder.from_words(tokens)
    scored = finder.score_ngrams(trigram_measures.raw_freq)
    return sorted(finder.ngram_fd.items(), key=lambda t: (-t[1], t[0]))
Role_Playing_games_comments['three_words_freq'] = Role_Playing_games_comments['Comment_without_stopword_title_lem'].apply(lambda x:tri_tokens(x))
# count frequency of tokens with three words
three_words_list = []
for three_word_list in Role_Playing_games_comments_train['three_words_freq']:
    for three_word in three_word_list:
        three_words_list.append(three_word)
three_words = dict()
for i in range(len(three_words_list)):
    if three_words_list[i][0] not in three_words.keys():
        three_words[three_words_list[i][0]] = three_words_list[i][1]
    else:
        three_words[three_words_list[i][0]] = three_words[three_words_list[i][0]] + three_words_list[i][1]
three_words_frequency = pd.DataFrame(list(three_words.items()), columns=['three_words_Features', 'Frequency'])
three_words_frequency = three_words_frequency.sort_values(by = 'Frequency', ascending = False)
three_words_frequency.to_csv('Role_Playing_games_threewords_features.csv')
# apply function above into Role_Playing games comments
Role_Playing_games_comments['Sentiment_Score'] = Role_Playing_games_comments['Comment'].apply(lambda x:swn_polarity(x))
Role_Playing_games_comments.to_csv('Role_Playing_games_features_sentiment.csv')
PC_games_Sports = PC_games[PC_games['Genre_modified'] == 'Sports']
PC_games_Sports['Title_check'] = PC_games_Sports['Title'].apply(lambda x: title_check(x))
PC_games_Sports = PC_games_Sports[PC_games_Sports['Title_check'] != 'No']
len(PC_games_Sports)
PC_game_Sports_title = PC_games_Sports['Title'].unique()
#Create Sports games comments dataset
Sports_games_comments = pd.DataFrame()
for title in PC_game_Sports_title:
    Sports_game_comments = PC_user_comments[PC_user_comments['Title'] == title]
    Sports_game_comments['genre'] = 'Sports'
    Sports_games_comments =Sports_games_comments.append(Sports_game_comments)
Sports_games_comments = Sports_games_comments.reset_index()
Sports_games_comments = Sports_games_comments[['Title','Comment','Username','Comment_without_stopword_title_lem','genre']]
Sports_games_comments.head()
#split Sports_game_dataset into train dataset and test dataset
Sports_games_comments_train = Sports_games_comments.sample(frac=0.7)
Sports_games_comments_test = Sports_games_comments.loc[~Sports_games_comments.index.isin(Sports_games_comments_train.index)]
#extract nouns verbs and adjective from user comments
Sports_games_comments_train['comment_noun'] = Sports_games_comments_train['Comment_without_stopword_title_lem'].apply(lambda x: noun(x))
Sports_games_comments_train['comment_adjective'] = Sports_games_comments_train['Comment_without_stopword_title_lem'].apply(lambda x: adjective(x))
#remove stopwords again
Sports_games_comments_train['comment_adjective'] = Sports_games_comments_train['comment_adjective'].apply(lambda x: [word for word in x if word not in stop])
Sports_games_comments_train['comment_noun'] = Sports_games_comments_train['comment_noun'].apply(lambda x: [word for word in x if word not in stop])
# create a new column that lists tokens with two words
import nltk
from nltk.collocations import *
bigram_measures = nltk.collocations.BigramAssocMeasures()
trigram_measures = nltk.collocations.TrigramAssocMeasures()
def bi_tokens(comments):
    tokens = nltk.wordpunct_tokenize(comments)
    finder = BigramCollocationFinder.from_words(tokens)
    scored = finder.score_ngrams(bigram_measures.raw_freq)
    return sorted(finder.ngram_fd.items(), key=lambda t: (-t[1], t[0]))
Sports_games_comments_train['two_words_freq'] = Sports_games_comments_train['Comment_without_stopword_title_lem'].apply(lambda x:bi_tokens(x))
# count frequency of tokens with two words
two_words_list = []
for two_word_list in Sports_games_comments_train['two_words_freq']:
    for two_word in two_word_list:
        two_words_list.append(two_word)
two_words = dict()
for i in range(len(two_words_list)):
    if two_words_list[i][0] not in two_words.keys():
        two_words[two_words_list[i][0]] = two_words_list[i][1]
    else:
        two_words[two_words_list[i][0]] = two_words[two_words_list[i][0]] + two_words_list[i][1]
two_words_frequency = pd.DataFrame(list(two_words.items()), columns=['two_words_Features', 'Frequency'])
two_words_frequency = two_words_frequency.sort_values(by = 'Frequency', ascending = False)
two_words_frequency.to_csv('Sports_games_twowords_features.csv')
# create a new column that lists tokens with three words
import nltk
from nltk.collocations import *
bigram_measures = nltk.collocations.BigramAssocMeasures()
trigram_measures = nltk.collocations.TrigramAssocMeasures()
def tri_tokens(comments):
    tokens = nltk.wordpunct_tokenize(comments)
    finder = TrigramCollocationFinder.from_words(tokens)
    scored = finder.score_ngrams(trigram_measures.raw_freq)
    return sorted(finder.ngram_fd.items(), key=lambda t: (-t[1], t[0]))
Sports_games_comments_train['three_words_freq'] = Sports_games_comments_train['Comment_without_stopword_title_lem'].apply(lambda x:tri_tokens(x))
# count frequency of tokens with three words
three_words_list = []
for three_word_list in Sports_games_comments_train['three_words_freq']:
    for three_word in three_word_list:
        three_words_list.append(three_word)
three_words = dict()
for i in range(len(three_words_list)):
    if three_words_list[i][0] not in three_words.keys():
        three_words[three_words_list[i][0]] = three_words_list[i][1]
    else:
        three_words[three_words_list[i][0]] = three_words[three_words_list[i][0]] + three_words_list[i][1]
three_words_frequency = pd.DataFrame(list(three_words.items()), columns=['three_words_Features', 'Frequency'])
three_words_frequency = three_words_frequency.sort_values(by = 'Frequency', ascending = False)
three_words_frequency.to_csv('Sports_games_threewords_features.csv')
# apply function above into Role_Playing games comments
Sports_games_comments_train['Sentiment_Score'] = Sports_games_comments_train['Comment'].apply(lambda x:swn_polarity(x))
Sports_games_comments_train.to_csv('Sports_games_features_sentiment.csv')
PC_games_Strategy = PC_games[PC_games['Genre_modified'] == 'Strategy']
PC_games_Strategy['Title_check'] = PC_games_Strategy['Title'].apply(lambda x: title_check(x))
PC_games_Strategy = PC_games_Strategy[PC_games_Strategy['Title_check'] != 'No']
PC_game_Strategy_title = PC_games_Strategy['Title'].unique()
#Create Sports games comments dataset
Strategy_games_comments = pd.DataFrame()
for title in PC_game_Strategy_title:
    Strategy_game_comments = PC_user_comments[PC_user_comments['Title'] == title]
    Strategy_game_comments['genre'] = 'Strategy'
    Strategy_games_comments =Strategy_games_comments.append(Strategy_game_comments)
len(Strategy_games_comments)
len(Strategy_games_comments['Username'].unique())
Strategy_games_comments = Strategy_games_comments.reset_index()
Strategy_games_comments = Strategy_games_comments[['Title','Comment','Username','Comment_without_stopword_title_lem','genre']]
sctrategy_games_comments['one_word'] = Role_Playing_games_comments['Comment_without_stopword_title_lem'].apply(lambda x: x.split())
Strategy_games_comments['one_word'] = Role_Playing_games_comments['one_word'].apply(lambda x: [word for word in x if word not in stop])
#extract nouns verbs and adjective from user comments
Strategy_games_comments_train['comment_noun'] = Strategy_games_comments_train['Comment_without_stopword_title_lem'].apply(lambda x: noun(x))
Strategy_games_comments_train['comment_adjective'] = Strategy_games_comments_train['Comment_without_stopword_title_lem'].apply(lambda x: adjective(x))
#remove stopwords again
Strategy_games_comments_train['comment_adjective'] = Strategy_games_comments_train['comment_adjective'].apply(lambda x: [word for word in x if word not in stop])
Strategy_games_comments_train['comment_noun'] = Strategy_games_comments_train['comment_noun'].apply(lambda x: [word for word in x if word not in stop])
# create a new column that lists tokens with two words
import nltk
from nltk.collocations import *
bigram_measures = nltk.collocations.BigramAssocMeasures()
trigram_measures = nltk.collocations.TrigramAssocMeasures()
def bi_tokens(comments):
    tokens = nltk.wordpunct_tokenize(comments)
    finder = BigramCollocationFinder.from_words(tokens)
    scored = finder.score_ngrams(bigram_measures.raw_freq)
    return sorted(finder.ngram_fd.items(), key=lambda t: (-t[1], t[0]))
Strategy_games_comments['two_words_freq'] = Strategy_games_comments['Comment_without_stopword_title_lem'].apply(lambda x:bi_tokens(x))
# count frequency of tokens with two words
two_words_list = []
for two_word_list in Strategy_games_comments_train['two_words_freq']:
    for two_word in two_word_list:
        two_words_list.append(two_word)
two_words = dict()
for i in range(len(two_words_list)):
    if two_words_list[i][0] not in two_words.keys():
        two_words[two_words_list[i][0]] = two_words_list[i][1]
    else:
        two_words[two_words_list[i][0]] = two_words[two_words_list[i][0]] + two_words_list[i][1]
two_words_frequency = pd.DataFrame(list(two_words.items()), columns=['two_words_Features', 'Frequency'])
two_words_frequency = two_words_frequency.sort_values(by = 'Frequency', ascending = False)
two_words_frequency.to_csv('Strategy_games_twowords_features.csv')
# create a new column that lists tokens with three words
import nltk
from nltk.collocations import *
bigram_measures = nltk.collocations.BigramAssocMeasures()
trigram_measures = nltk.collocations.TrigramAssocMeasures()
def tri_tokens(comments):
    tokens = nltk.wordpunct_tokenize(comments)
    finder = TrigramCollocationFinder.from_words(tokens)
    scored = finder.score_ngrams(trigram_measures.raw_freq)
    return sorted(finder.ngram_fd.items(), key=lambda t: (-t[1], t[0]))
Strategy_games_comments['three_words_freq'] = Strategy_games_comments['Comment_without_stopword_title_lem'].apply(lambda x:tri_tokens(x))
# count frequency of tokens with three words
three_words_list = []
for three_word_list in Strategy_games_comments_train['three_words_freq']:
    for three_word in three_word_list:
        three_words_list.append(three_word)
three_words = dict()
for i in range(len(three_words_list)):
    if three_words_list[i][0] not in three_words.keys():
        three_words[three_words_list[i][0]] = three_words_list[i][1]
    else:
        three_words[three_words_list[i][0]] = three_words[three_words_list[i][0]] + three_words_list[i][1]
three_words_frequency = pd.DataFrame(list(three_words.items()), columns=['three_words_Features', 'Frequency'])
three_words_frequency = three_words_frequency.sort_values(by = 'Frequency', ascending = False)
three_words_frequency.to_csv('Strategy_games_threewords_features.csv')
# apply function above into Strategy games comments
Strategy_games_comments['Sentiment_Score'] = Strategy_games_comments['Comment'].apply(lambda x:swn_polarity(x))
Strategy_games_comments.head()
Strategy_games_comments.to_csv('Strategy_games_features_sentiment.csv')

# game features analysis
# Load datatest into Jupternoteboook
import numpy as np
import pandas as pd
#Action
Action_games_sentiments = pd.read_csv('action_games_features_sentiment.csv',index_col = 'Title')
Action_games_sentiments = Action_games_sentiments[['one_word','two_words_freq','three_words_freq','Sentiment_Score']]
# Adventure
Adventure_games_sentiments = pd.read_csv('Adventure_games_features_sentiment.csv')
Adventure_games_sentiments = Adventure_games_sentiments[['one_word','two_words_freq','three_words_freq','Sentiment_Score']]
# Role_playing
Role_playing_games_sentiments = pd.read_csv('Role_Playing_games_features_sentiment.csv',index_col = 'Title')
Role_playing_games_sentiments = Role_playing_games_sentiments[['one_word','two_words_freq','three_words_freq','Sentiment_Score']]
# Strategy
Strategy_games_sentiments = pd.read_csv('Strategy_games_features_sentiment.csv',index_col = 'Title')
Strategy_games_sentiments = Strategy_games_sentiments[['one_word','two_words_freq','three_words_freq','Sentiment_Score']]
# create function to collect frequency of features based on positive, negative and netural category
def count_sentiment_scores(scores_list):
    positive = []
    negative = []
    neutral = []
    for score in scores_list:
        if score >= 2:
            positive.append(score)
        elif score < 0:
            negative.append(score)
        else:
            neutral.append(score)
    return [len(scores_list),len(positive),len(negative),len(neutral)]
#create function to get the sentiment scores of each comment based on features
def get_sentiment_scores_frequency(features):
    sentiment_scores = []
    if len(features) == 2:
        for i, words in enumerate(Action_games_sentiments['two_words_freq']):
            words = eval(words)
            for two_word in words:
                if features == two_word[0]:
                    sentiment_scores.append(Action_games_sentiments['Sentiment_Score'][i])
        return count_sentiment_scores(sentiment_scores)
    elif len(features) == 3:
        for i, words in enumerate(Action_games_sentiments['three_words_freq']):
            words = eval(words)
            for three_word in words:
                if features == three_word[0]:
                    sentiment_scores.append(Action_games_sentiments['Sentiment_Score'][i])
        return count_sentiment_scores(sentiment_scores)
    else:
        for i,one_word in enumerate(Action_games_sentiments['one_word']):
            one_word = eval(one_word)
            if features[0] in one_word:
                sentiment_scores.append(Action_games_sentiments['Sentiment_Score'][i])
        return count_sentiment_scores(sentiment_scores)
# create function to build a dataframe showing feature name, frequency, positive frequency, negative frequecy, and neutral frequency
def sentiment_scores_analysis(features_list):
    sentiment_scores_frequency = pd.DataFrame(columns = ['Genre' , 'Feature', 'Frequency', 'Positive_Frequency' , 'Negative_Frequency','Neutral_Frequency'])
    for feature in features_list:
        feature_row = get_sentiment_scores_frequency(feature)
        sentiment_scores_frequency = sentiment_scores_frequency.append({'Genre' : 'Action' , 'Feature' : feature,'Frequency':feature_row[0],'Positive_Frequency':feature_row[1],'Negative_Frequency':feature_row[2],'Neutral_Frequency':feature_row[3]} , ignore_index=True)
    return sentiment_scores_frequency
# Create an Action game features list
Action_features = [('load','time'),('voice','act'),('main','character'),('character','development'),
                  ('interest','character'),('plot', 'twist'),['fps'],('first','person','shooters'),
                  ('firstperson','shooter'),('single','player'),['multiplayer'],['rpg'],('role','play'),
                  ['roleplay'],('boss','fight'),('replay','value'),['graphics'],('art','style'),('art','direction'),
                  ('sound','effect'),('sound','design'),('level','design'),('solve','puzzle'),('difficulty', 'level'),
                  ['pace'],('learn','curve'),('map','design'),['physics']]
# create a dataframe about action game that contains the frequency of sentiments of each features
Action_sentiment_scores_feature = sentiment_scores_analysis(Action_features)
# Save dataframe to csv file
sentiment_scores_feature.to_csv('Action_sentiment_scores_feature.csv')
#create function to get the sentiment scores of each comment based on features
def get_sentiment_scores_frequency(features):
    sentiment_scores = []
    if len(features) == 2:
        for i, words in enumerate(Adventure_games_sentiments['two_words_freq']):
            words = eval(words)
            for two_word in words:
                if features == two_word[0]:
                    sentiment_scores.append(Adventure_games_sentiments['Sentiment_Score'][i])
        return count_sentiment_scores(sentiment_scores)
    elif len(features) == 3:
        for i, words in enumerate(Adventure_games_sentiments['three_words_freq']):
            words = eval(words)
            for three_word in words:
                if features == three_word[0]:
                    sentiment_scores.append(Adventure_games_sentiments['Sentiment_Score'][i])
        return count_sentiment_scores(sentiment_scores)
    else:
        for i,one_word in enumerate(Adventure_games_sentiments['one_word']):
            one_word = eval(one_word)
            if features[0] in one_word:
                sentiment_scores.append(Adventure_games_sentiments['Sentiment_Score'][i])
        return count_sentiment_scores(sentiment_scores)
# create function to build a dataframe showing feature name, frequency, positive frequency, negative frequecy, and neutral frequency
def sentiment_scores_analysis(features_list):
    sentiment_scores_frequency = pd.DataFrame(columns = ['Genre' , 'Feature', 'Frequency', 'Positive_Frequency' , 'Negative_Frequency','Neutral_Frequency'])
    for feature in features_list:
        feature_row = get_sentiment_scores_frequency(feature)
        sentiment_scores_frequency = sentiment_scores_frequency.append({'Genre' : 'Adventure' , 'Feature' : feature,'Frequency':feature_row[0],'Positive_Frequency':feature_row[1],'Negative_Frequency':feature_row[2],'Neutral_Frequency':feature_row[3]} , ignore_index=True)
    return sentiment_scores_frequency
# Create a adventure game features list
Adventure_features = [('choices','make'),('decisions','make'),('build','anything'),('build','whatever'),
                     ('voice','act'),('voice','actors'),('main', 'character'),('character','development'),
                      ('character','interest'),('plot','twist'),('narrative','experience'),('single','player'),('survival','mode'),
                     ('creative','mode'),('visual','novel'),('interactive','movie'),('replay','value'),('sound','effect'),('sound','design'),
                      ['graphics'],('pixel','art'),('work','art'),('solve','puzzle'),('level','design'),('puzzle','easy')]
# create a dataframe about adventure game that contains the frequency of sentiments of each features
Adventure_sentiment_scores_features = sentiment_scores_analysis(Adventure_features)
# Save dataframe to csv file
Adventure_sentiment_scores_features.to_csv('Adventure_sentiment_scores_features.csv')
#create function to get the sentiment scores of each comment based on features
def get_sentiment_scores_frequency(features):
    sentiment_scores = []
    if len(features) == 2:
        for i, words in enumerate(strategy_games_sentiments['two_words_freq']):
            words = eval(words)
            for two_word in words:
                if features == two_word[0]:
                    sentiment_scores.append(strategy_games_sentiments['Sentiment_Score'][i])
        return count_sentiment_scores(sentiment_scores)
    elif len(features) == 3:
        for i, words in enumerate(strategy_games_sentiments['three_words_freq']):
            words = eval(words)
            for three_word in words:
                if features == three_word[0]:
                    sentiment_scores.append(strategy_games_sentiments['Sentiment_Score'][i])
        return count_sentiment_scores(sentiment_scores)
    else:
        for i,one_word in enumerate(strategy_games_sentiments['one_word']):
            one_word = eval(one_word)
            if features[0] in one_word:
                sentiment_scores.append(strategy_games_sentiments['Sentiment_Score'][i])
        return count_sentiment_scores(sentiment_scores)
# create function to build a dataframe showing feature name, frequency, positive frequency, negative frequecy, and neutral frequency
def sentiment_scores_analysis(features_list):
    sentiment_scores_frequency = pd.DataFrame(columns = ['Genre' , 'Feature', 'Frequency', 'Positive_Frequency' , 'Negative_Frequency','Neutral_Frequency'])
    for feature in features_list:
        feature_row = get_sentiment_scores_frequency(feature)
        sentiment_scores_frequency = sentiment_scores_frequency.append({'Genre' : 'strategy' , 'Feature' : feature,'Frequency':feature_row[0],'Positive_Frequency':feature_row[1],'Negative_Frequency':feature_row[2],'Neutral_Frequency':feature_row[3]} , ignore_index=True)
    return sentiment_scores_frequency
# Create a strategy game features list
strategy_features = [['communication'],['community'],('resource','management'),('voice','act'),('single','player'),
                                      ['multiplayer'],('multi','player'),('turn','base'),('turnbased','strategy'),['turnbased'],('skirmish','mode'),('ironman','mode'),
                                      ['diretide'],['givediretide'],('halloween','event'),('replay','ability'),['graphics'],['animations'],('learn','curve'),['paytowin'],
                                      ('tech','tree'),['matchmaking'],['mmr'],['challenge'],('difficulty','curve')]
# create a dataframe about Strategy game that contains the frequency of sentiments of each features
Strategy_sentiment_scores_features = sentiment_scores_analysis(strategy_features)
# Save dataframe to csv file
Strategy_sentiment_scores_features.to_csv('Strategy_sentiment_scores_features.csv')
def get_sentiment_scores_frequency(features):
    sentiment_scores = []
    if len(features) == 2:
        for i, words in enumerate(Role_playing_games_sentiments['two_words_freq']):
            words = eval(words)
            for two_word in words:
                if features == two_word[0]:
                    sentiment_scores.append(Role_playing_games_sentiments['Sentiment_Score'][i])
        return count_sentiment_scores(sentiment_scores)
    elif len(features) == 3:
        for i, words in enumerate(Role_playing_games_sentiments['three_words_freq']):
            words = eval(words)
            for three_word in words:
                if features == three_word[0]:
                    sentiment_scores.append(Role_playing_games_sentiments['Sentiment_Score'][i])
        return count_sentiment_scores(sentiment_scores)
    else:
        for i,one_word in enumerate(Role_playing_games_sentiments['one_word']):
            one_word = eval(one_word)
            if features[0] in one_word:
                sentiment_scores.append(Role_playing_games_sentiments['Sentiment_Score'][i])
        return count_sentiment_scores(sentiment_scores)
# create function to build a dataframe showing feature name, frequency, positive frequency, negative frequecy, and neutral frequency
def sentiment_scores_analysis(features_list):
    sentiment_scores_frequency = pd.DataFrame(columns = ['Genre' , 'Feature', 'Frequency', 'Positive_Frequency' , 'Negative_Frequency','Neutral_Frequency'])
    for feature in features_list:
        feature_row = get_sentiment_scores_frequency(feature)
        sentiment_scores_frequency = sentiment_scores_frequency.append({'Genre' : 'Role playing' , 'Feature' : feature,'Frequency':feature_row[0],'Positive_Frequency':feature_row[1],'Negative_Frequency':feature_row[2],'Neutral_Frequency':feature_row[3]} , ignore_index=True)
    return sentiment_scores_frequency
# Create a adventure features list
Role_playing_features = [('modding','community'),('customer','service'),('character','customization'),('build','character'),
                        ('create','character'),('dialogue','options'),('dialogue','wheel'),('internet','connection'),('server','issue'),
                        ('erorr','37'),('load','screen'),('load','time'),('keyboard','mouse'),('mouse','keyboard'),('xbox','360'),('xbox','controller'),
                        ('bug','glitches'),('user','interface'),('voice','act'),('main','character'),('character','creation'),('interest','character'),('great','character'),
                        ('character','model'),('character','class'),('character', 'progression'),('character', 'development'),('main','plot'),('main','line'),
                         ('main','storyline'),['gameplay'],('plot','hole'),('gameplay','mechanics'),('single','player'),['rpg'],('rpg','elements'),('role','play'),
                        ('turn','base'),('hack','and','slash'),('hack', 'n','slash'),['mmo'],('replay','value'),('get','repetitive'),('boss','fight'),('user','score'),
                        ('give','score'),('art','style'),['graphics'],('art','direction'),('voice','actor'),('sound','effect'),('frame','rate'),('cd','projekt'),
                        ('pc','version'),('console','port'),('pc','port'),['mode'],('level','design'),('level','character'),('difficulty','level'),('level','up'),
                        ('skill','tree'),('skill','point'),('talent','tree'),('side','quest'),('side','missions'),('fetch','quest'),('fast','pace'),
                        ('fast','travel'),('learn','curve')]
# create a dataframe about role playing game that contains the frequency of sentiments of each features
Role_playing_sentiment_scores_features = sentiment_scores_analysis(Role_playing_features)
# Save dataframe to csv file
Role_playing_sentiment_scores_features.to_csv('Role_playing_sentiment_scores_features.csv')
