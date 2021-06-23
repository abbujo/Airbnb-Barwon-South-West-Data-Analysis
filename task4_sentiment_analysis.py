import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from langdetect import detect
import missingno as msno
import nltk

nltk.downloader.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.downloader.download('stopwords')
from nltk.corpus import stopwords
from wordcloud import WordCloud

# Fetching the files to work on​
# For sentiment analysis purpose we are using both the listings data as well as review data.
# This is because both the tables combined will give a wholesome meaning to the analysis rather than on itself
listings = pd.read_csv('listings.csv')
reviews = pd.read_csv('reviews.csv')

# Merging the data from both the files by doing a left join
merged_data = pd.merge(listings,
                       reviews,
                       left_on='id',
                       right_on='listing_id',
                       how='left')
# First of all we delete the duplicate column
merged_data.drop(['listing_id'], axis=1, inplace=True)
# We now have to drop columns that we will not be using for our analysis
merged_data.drop([
    'host_name', 'neighbourhood_group', 'last_review', 'id_y', 'reviewer_id',
    'reviewer_name'
],
                 axis=1,
                 inplace=True)
# Because of the merge some column names because of having same name get renamed
# As well as renaming other column names that will use to make more sense
merged_data.rename(columns={
    'id_x': 'id',
    'name': 'accomodation_name'
},
                   inplace=True)

# Treating missing values

# Checking if there is any missing data
print(merged_data.isna().sum())

# it shows date, comments and reviews_per_month are missing some values
# since our aim is to analyse comment, if there is no comment let us delete those rows
merged_data.dropna(subset=["comments"], inplace=True)

# Checking if there is any missing data
print(merged_data.isna().sum())
# No missing data anymore, all missing data have been dealt with

# Detecting the language

# def language_detection(text):
#     try:
#         return detect(text)
#     except:
#         return None

# # Adding new column to the existing dataset that gives us information about the language used
# merged_data['language'] = merged_data['comments'].apply(language_detection)
# merged_data.to_csv('processed_merged_data.csv', index=False)
merged_data = pd.read_csv('processed_merged_data.csv')

print(merged_data['language'].value_counts())
# Top 11 languages are
# en       209937 - 'English',
# zh-cn      4812 - 'Chinese',
# fr         1122 - 'French',
# ko          970 - 'Korean',
# de          678 - 'German',
# ro          646 - 'Romanian',
# af          421 - 'Afrikaans',
# so          287 - 'Somali',
# ca          233 - 'Catalan',
# nl          203 - 'Dutch',
# es          201 - 'Spanish'
# visualizing the comments' languages b) neat and clean
ax = merged_data.language.value_counts().head(11).plot(kind='barh',
                                                       figsize=(9, 5),
                                                       color="lightcoral",
                                                       fontsize=12)

ax.set_title("\nMost used language in comments\n",
             fontsize=12,
             fontweight='bold')
ax.set_xlabel(" Total Comments", fontsize=10)
ax.set_yticklabels([
    'English', 'Chinese', 'French', 'Korean', 'German', 'Romanian',
    'Afrikaans', 'Somali', 'Catalan', 'Dutch', 'Spanish'
])

# create a list to collect the plt.patches data
totals = []
# find the ind. values and append to list
for i in ax.patches:
    totals.append(i.get_width())
# get total
total = sum(totals)

# set individual bar labels using above list
for i in ax.patches:
    ax.text(x=i.get_width(),
            y=i.get_y() + .35,
            s=str(round((i.get_width() / total) * 100, 2)) + '%',
            fontsize=10,
            color='black')

# invert for largest on top
ax.invert_yaxis()
plt.show()

# selecting data with only english comments
merged_data_eng = merged_data[(merged_data['language'] == 'en')]

# initialising sentiment intenesity analyzer
analyzer = SentimentIntensityAnalyzer()


# use the polarity_scores() method to get the sentiment metrics
def print_sentiment_scores(sentence):
    snt = analyzer.polarity_scores(sentence)
    print("{:-<40} {}".format(sentence, str(snt)))


# getting only the negative score
def negative_score(text):
    negative_value = analyzer.polarity_scores(text)['neg']
    return negative_value


# getting only the neutral score
def neutral_score(text):
    neutral_value = analyzer.polarity_scores(text)['neu']
    return neutral_value


# getting only the positive score
def positive_score(text):
    positive_value = analyzer.polarity_scores(text)['pos']
    return positive_value


# getting only the compound score
def compound_score(text):
    compound_value = analyzer.polarity_scores(text)['compound']
    return compound_value


# merged_data_eng['sentiment_neg'] = merged_data_eng['comments'].apply(negative_score)
# merged_data_eng['sentiment_neu'] = merged_data_eng['comments'].apply(neutral_score)
# merged_data_eng['sentiment_pos'] = merged_data_eng['comments'].apply(positive_score)
# merged_data_eng['sentiment_compound'] = merged_data_eng['comments'].apply(compound_score)
# # write the dataframe to a csv file in order to avoid the long runtime
# merged_data_eng.to_csv('sentiment_merged_data_eng.csv', index=False)

merged_data = pd.read_csv('sentiment_merged_data_eng.csv')

# Percentage distribution of different sentiments
percentiles = merged_data.sentiment_compound.describe(
    percentiles=[.05, .1, .2, .3, .4, .5, .6, .7, .8, .9])

# Assign the respective data
neg = percentiles['5%']
mid = percentiles['20%']
pos = percentiles['max']
names = ['Negative Comments', 'Neutral Comments', 'Positive Comments']
size = [neg, mid, pos]

# Creating Pie Chart to check percentage of negative, neutral and positive comments
plt.pie(size,
        labels=names,
        colors=['red', 'yellow', 'blue'],
        autopct='%.2f%%',
        pctdistance=0.8,
        wedgeprops={
            'linewidth': 7,
            'edgecolor': 'white'
        })

# create circle for the center of the plot to make the pie look like a donut
my_circle = plt.Circle((0, 0), 0.6, color='white')

# plot the donut chart
fig = plt.gcf()
fig.set_size_inches(7, 7)
fig.gca().add_artist(my_circle)
plt.show()

# full dataframe with POSITIVE comments
merged_data_pos = merged_data.loc[merged_data.sentiment_compound >= 0.95]

# full dataframe with NEGATIVE comments
merged_data_neg = merged_data.loc[merged_data.sentiment_compound < 0.0]

# Word Cloud is a data visualization technique used for representing text data in 
# which the size of each word indicates its frequency or importance.
def plot_wordcloud(wordcloud, language):
    plt.figure(figsize=(12, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(language, fontsize=18, fontweight='bold')
    plt.show()

wordcloud = WordCloud(max_font_size=200,
                      max_words=200,
                      background_color="palegreen",
                      width=3000,
                      height=2000,
                      stopwords=stopwords.words('english')).generate(
                          str(merged_data_pos.comments.values))

plot_wordcloud(wordcloud, '\nWhy people like the accomodations?')

wordcloud = WordCloud(max_font_size=200,
                      max_words=200,
                      background_color="powderblue",
                      width=3000,
                      height=2000,
                      stopwords=stopwords.words('english')).generate(
                          str(merged_data_neg.comments.values))

plot_wordcloud(wordcloud, '\nWhy people dislike the accomodations?')