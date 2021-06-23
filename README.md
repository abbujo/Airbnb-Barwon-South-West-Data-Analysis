# Airbnb-Bayron-West-Data-Analysis
Data warehouse and mining Assignment

Abhishek Joshi

S4635371 |  Masters of Applied Information Technology | Victoria University

DATA WAREHOUSE AND MINING

NIT6160 – PROJECT REPORT



# Contents              
[Introduction](#_Toc75306444)

[Dataset and Datafiles](#_Toc75306445)

[Listings CSV file and data dictionary](#_Toc75306446)

[Reviews CSV file and data dictionary](#_Toc75306447)

[Background Research](#_Toc75306448)

[Task 1: Data Pre-processing](#_Toc75306449)

[Task 2: Exploratory Data Analysis (EDA) with Data Visualization](#_Toc75306450)

[Task 3: Building the Accommodation Prediction Model](#_Toc75306451)

[Task 4: Sentiment analysis](#_Toc75306452)

[Discusssion and Validation](#_Toc75306458)

[Task 1: Data Pre-processing](#_Toc75306459)

[Task 2: Exploratory Data Analysis (EDA) with Data Visualization](#_Toc75306460)

[Task 3: Building the Accommodation Prediction Model](#_Toc75306461)

[Task 4: Sentiment analysis](#_Toc75306462)

[Conclusion](#_Toc75306463)

[References](#_Toc75306464)




# Introduction
##
Airbnb is a pioneer in the homestay/vacation rentals industry. Ever since its emergence in 2008 from San Fransisco California, the American company has expanded its services to a number of countries all around the globe. Inside Airbnb**(Inside Airbnb. Adding data to the debate., 2021)** is an effort at unifying data from all across the globe from Airbnbs. In this project we will be exploring data from the Barwon South West region provided in inside Airbnb. Our aim is to find something valuable to potential investors and hosts with the help of data mining and machine learning.
## Dataset and Datafiles
Inside Airbnb is an independent, non-commercial set of tools and data that allows us to explore how Airbnb is really being used in cities around the world.
### Listings CSV file and data dictionary
The listings file contains information about all the AirBnb listings that are present in the Barwon South West region. The file contains various information such as:

|**S.No**|**Column Name**|**Description**|**Type**|
| :- | :- | :- | :- |
|1|id|ID of the Listing|integer|
|2|name|NAME of the Listing|string|
|3|host\_id|HOST\_ID of the Listing|integer|
|4|host\_name|HOST\_NAME of the Listing|string|
|5|neighbourhood\_group|NEIGHBOURHOOD\_GROUP of the Listing|null|
|6|neighbourhood|NEIGHBOURHOOD of the Listing|string|
|7|latitude|LATITUDE of the Listing|float|
|8|longitude|LONGITUDE of the Listing|float|
|9|room\_type|ROOM\_TYPE of the Listing|enum(string)|
|10|price|PRICE of the Listing|integer|
|11|minimum\_nights|MINIMUM\_NIGHTS of the Listing|integer|
|12|number\_of\_reviews|NUMBER\_OF\_REVIEWS of the Listing|integer|
|13|last\_review|LAST\_REVIEW of the Listing|date|
|14|reviews\_per\_month|REVIEWS\_PER\_MONTH of the Listing|float|
|15|calculated\_host\_listings\_count|CALCULATED\_HOST\_LISTINGS\_COUNT of the Listing|integer|
|16|availability\_365|AVAILABILITY\_365 of the Listing|integer|

### Reviews CSV file and data dictionary
Reviews file contains all the information relating to reviews left by users for the listings in this region. The data dictionary for reviews file is:

|**S.No**|**Column Name**|**Description**|**Type**|
| :- | :- | :- | :- |
|1|listing\_id|listing\_id of the review|integer|
|2|id|id of the review|integer|
|3|date|date of the review|date|
|4|reviewer\_id|reviewer\_id of the review|integer|
|5|reviewer\_name|reviewer\_name of the review|string|
|6|comments|comments of the review|string|
# Background Research

## Task 1: Data Pre-processing
This step is a crucial step in which we preprocess the data coming from listings.csv file.

This step involves following steps:

1. Finding an index.
   1. index for the dataframe so that it will be easier to access records when needed
1. Removing redundant data.
   1. Removing duplicates
   1. Removing contants
   1. With the help of heatmap we find most correlated columns and remove the unwanted ones
1. Removing columns that will not have any effect on the pricing of the airbnb.
1. Dealing with missing values
   1. Delete the column if there are too many missing values
   1. Fill an appropriate value (either nearest value or 0)
1. Identifying noisy data and removing them by means of Outlier Analysis.
   1. By using boxplot, finding where most of the data is lying and if any points are suspiciously far away from the rest of the cluster, we can identify outliers and remove them

Types of diagrams and tools identified:

1. We will be using following types diagrams
   1. Boxplots
   1. Correlation heatmap
1. Methods being used:
   1. Correlation Analysis: Helps us to identify features that are similar so that we can remove such features from our dataframe for further processing.
   1. Outlier Anaysis : Used to eliminate noisy data in step 5. Outlier analysis helps us to find out stand out points in the scatter plot, which are unusually away from rest of the cluster and hence must be a noisy data.
1. Tools identified:
   1. Pandas library for data manipulation and analysis
   1. Matplotlib and seaborn libraries for plotting graphs
## Task 2: Exploratory Data Analysis (EDA) with Data Visualization
Here we will follow the following process:

1. Price Visualisation
   1. BoxPlot for price range in each neighbourhood
   1. BoxPlot for different type of accommodation in each neighbourhood
1. Map Visualisation
   1. Visualising the heatmap 
   1. Visualising marker maps for different accommodations Private, shared, hotel, entire home/apt
1. Number of accomodation in each region/market
   1. Count of listing by neighbourhood – plot bar graph and pie chart
   1. Count of listing by type– plot bar graph and pie chart
   1. Count of listing by neighbourhood by type -– plot bar graph
1. Mean Price in each region/market
   1. Mean price of listing by neighbourhood – plot bar graph
   1. Mean price of listing by type – plot bar graph
   1. Mean price of listing by neighbourhood by type – plot bar graph

Types of diagrams and tools identified:

1. We will be using following types diagrams
   1. Boxplots
   1. Horizontal bar charts
   1. Vertical Bar charts
   1. Pie chart
   1. Heatmap
   1. Marker maps
1. Tools identified:
   1. Pandas library for data manipulation and analysis
   1. Matplotlib library for plotting graphs
   1. Folium for map visualisation
   1. Web broweser will be needed to open the html files saved as output of folium code
## Task 3: Building the Accommodation Prediction Model

When it comes to predicting the value of a variable from a given set of data, we have algorithms such as forecast algorithms, regression algorithms, classification algorithms, clustering algorithms, etc. Choosing an algorithm to proceed with is a question in itself. In order to look for the answer, I did some study on algorithms that work the best in use cases similar to ours. I came across an article which endorsed the use of XGBoost because of the following reasons **(Pedro, 2021)**:

1. The results can be easily interpreted by visualizing the final trees.
1. Speed and accuracy — XG boost is quick and accurate compared to other algorithms.

The reason why XGBoost is better is because it is an iterative learning model which means it predicts once initially and then self-analyses its mistakes as a predictive toiler and give more weightage to the data points in which it made a wrong prediction in the next iteration. This process happens in cycle giving XGBoost better results as compared to other models.

## Task 4: Sentiment analysis

Sentiment analysis is one of the most common tasks of Natural Language Processing (NLP). It involves classifying a textual data into a set of pre-defined sentiment. Natural Language Toolkit (NLTK), is the most commonly used library for NLP tasks in Python.

In our task, I will be using VADER ( Valence Aware Dictionary for Sentiment Reasoning) algorithm to carry out sentiment analysis. This model is sensitive to both polarity (positive/negative) and intensity (strength) of emotion denoted by the text. The ability to define both polarity and intensity adds a lot of value to our use case as we will be able to identify what are the most common texts/features that people leave in a positive or a negative comment with its strength. We will use wordcloud visualization tool to visualise the output of the analysis.  **(Beri, 2021)**

The reasons for choosing VADER over other methods are: **(Pandey, 2021)**

1. It works any type of data be it social media text or any other general comments.
1. It doesn’t require any training data. Just as in our case we do not have any training data available to us.
1. It is fast, and
1. Not much prone to speed-performance tradeoff.
1. The algorithm can calculate sentiment score for negative, neutral, positive and compound classes of sentiments. **(Beri, 2021)**

# Discusssion and Validation

## Task 1: Data Pre-processing
In pre-processing tasks main aim was to remove all the dirty data.

It mainly involved these two steps in a number of iterations (repeatedly):

1. Identifying the type of dirty data.
1. Correcting/Treating the data

To identify I used various techniques such as looking at the count of nulls in columns, use advanced methods such as correlation heatmap, outlier analysis using, etc.

At the end a csv file was generated for further use to be used in other tasks.
## Task 2: Exploratory Data Analysis (EDA) with Data Visualization
In this section the aim was to visualise the data in terms of various graphs, charts or maps.

This section was helpful for getting various insights about the data relating to the price distribution of the accommodations as well as the physical distribution of the accommodations.

Boxplot was really helpful to visualise the range of distribution. Heatmap was helpful in visualising the density of distribution in the map. Bar plots were helpful in visualising various features and pie chart was helpful to demonstrate the percentage distribution.

We used folium for Map visualisation which uses open source maps from Leaflet.

## Task 3: Building the Accommodation Prediction Model
During its implementation we followed the following steps:

1. Obtained the dataframe,
1. Categorical features were one-hot encoded,
1. The numeric features were standardized and normalised by logarithmic transformation.
1. Split the dataset into predictive features and target.
1. Scale the predictive features. I had chosen Max Abs Scaler.
1. Split the data into test and train data.
1. Obtain the model, train the model against train dataset and made the predictions against the test dataset.
1. Find out the performance of the model using Mean squared error and R-squared error values.
1. Finding out the feature weights in the prediction model.

The results of prediction were as follows:

- Training Mean Square Error: 0.0521
- Validation Mean Square Error: 0.1911
- Training R-squared score %: 85.9245
- Validation R-squared score %: 51.1053

The accuracy of price prediction wasn’t as great as anticipated.

## Task 4: Sentiment analysis

Used VADER algorithm to carry out sentiment analysis. The results were positive, and I was able to successfully get results.

The algorithm was successfully able to categorise the comments into positive, negative and neutral comments.

I used wordcloud to visualise the most influential words that were present in positive and negative comments.

# Conclusion
Main insights gained from the project are as follows:

1. Most common listings are of type Entire home/apt.
1. Least common listings are of type Shared rooms.
1. Surf coast has the greatest number of listings followed by Greater Geelong.
1. Southern Grampians has the least number of listings followed by Glenelg.
1. Surf coast is the most expensive area while Southern Grampians is the least expensive.
1. Entire home/apt are most expensive room type while Private room type are the cheapest ones.
1. Entire home/apt room type, Private room type and Greater Geelong are the three most influential features for prediction algorithm respectively.
1. Top 3 languages in which comment is left are English, Chinese and French respectively.
1. 44.56% of the comments are positive, 20.46% of the comments are negative while rest 4.98% comments are neutral.
1. From the wordcloud it can be said that people tend to like an accommodation because of these keywords: “secure”, “friendly”, “comfortable”, “communication” etc.
1. From the wordcloud it can be said that people tend to dislike an accommodation because of these keywords: "cleanliness", "maintenance", "unresponsive", "lock", etc

Limitations and challenges:

1. The only limitation in my experience was the accuracy of price prediction wasn’t as great as anticipated. This was due to very limited information in the dataset. The number of features that determine the prices were not much insightful and insufficient to generate satisfactory results.

This could have been solved by including more features such as information about amnesties available, activities available, etc.

# References
Beri, A., 2021. SENTIMENTAL ANALYSIS USING VADER. [online] Medium. Available at: <https://towardsdatascience.com/sentimental-analysis-using-vader-a3415fef7664> [Accessed 22 June 2021].

Inside Airbnb. 2021. Inside Airbnb. Adding data to the debate.. [online] Available at: <http://insideairbnb.com/about.html> [Accessed 21 June 2021].

Pedro, M., 2021. How to use PCA, TSNE, XGBoost, and finally Bayesian Optimization to predict the price of houses!. [online] Medium. Available at: <https://towardsdatascience.com/how-to-use-pca-tsne-xgboost-and-finally-bayesian-optimization-to-predict-the-price-of-houses-626dbaf242ae> [Accessed 22 June 2021].

Pandey, P., 2021. Simplifying Sentiment Analysis using VADER in Python (on Social Media Text). [online] Medium. Available at: <https://medium.com/analytics-vidhya/simplifying-social-media-sentiment-analysis-using-vader-in-python-f9e6ec6fc52f> [Accessed 22 June 2021].
