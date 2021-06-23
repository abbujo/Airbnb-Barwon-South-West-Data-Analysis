import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# DATA PREPROCESSING
# 1. Finding an index
# 2. Removing redundant data
# 3. Removing columns that will not have any effect on the pricing of the airbnb
# 4. Dealing with missing values
# 5. Identifying noisy data and removing them by means of Outlier Analysis
# 6. Output the preprocessed data to a csv file so that it can be used in other tasks

listings = pd.read_csv('listings.csv')

# 1. Finding an index
# Now we need to find out index for the dataset so that it will be easier to access them when needed
# Index should be set to a column that has unique values for each entry
# To find it let us find the number of unique values in each column

for column in listings.columns:
    print('Number of unique values in {} is {}'.format(
        column, len(listings[column].unique())))

# The above information suggests us that column id has unique value for each column
# Hence it should be made the index.
listings.set_index('id', inplace=True)

# 2. Removing redundant data
listings.drop_duplicates(inplace=True)

# With the help of heatmap we find most correlated columns and remove the unwanted ones
plt.figure(figsize=(10,8))
corr_mat = listings.corr()
sns.heatmap(corr_mat)
plt.tight_layout()
plt.show()

# reviews per month and number of reviews seem to be correlated and hence one can be removed
listings.drop(['reviews_per_month'], axis=1, inplace=True)

# 3. Removing columns that will not have any effect on the pricing of the airbnb
# Now we will explore more about data and remove id like columns which do not carry much information
# To not carry much info means that the column might not have any direct impact on the price of the airbnb
# Fields such as name, host_name, host_id do not make any difference to the pricing of the airbnb technically
# Therefore we can drop all those fields

listings.drop(['name', 'host_name', 'host_id'], axis=1, inplace=True)
print(listings.shape)

# 4. Dealing with missing values
# Find out where are the missing values by calculating percentage of missing values for each column

print(listings.isnull().sum() * 100 / len(listings))

# This suggests us that neighbourhood_group is entirely missing, hence can be dropped from the data
# last_review is also missing and since it does not have much significance on the price it can be dropped too
listings.drop(['neighbourhood_group', 'last_review'], axis=1, inplace=True)

# checking again if everything has been treated
print(listings.isnull().sum() * 100 / len(listings))

# 5. Identifying noisy data and removing them by means of Outlier Analysis
# Since we are more interested in finding how other features impact the price
# We will only look at price column to find if there are any outliers
# At first let us check information about the price distribution

print("Information about PRICE Column:")
print(listings['price'].describe())

# Everything looks normal apart from a few things
# The maximum value is unusually high from the mean value
# Creating a boxplot with the scatter points will help us find outlier points
# Setting xlim as minimum price and maximum price

priceMin = listings['price'].min()
priceMax = listings['price'].max()

listings['price'].plot(kind='box',
                       xlim=(priceMin, priceMax),
                       vert=False,
                       figsize=(10, 6))
plt.title("Boxplot showing price distribution")
plt.show()
plt.close()

# Price distribution from above example shows two clear outliers that are around the $ 10000 mark
# There is another point that is flirting around the $ 4000 mark.
# Let us try to find out further by plotting them by category

listings.assign(index=listings.groupby('room_type').cumcount()).pivot(
    'index', 'room_type', 'price').plot(kind='box')
plt.title("Boxplot showing price distribution wrt Room type")
plt.show()
plt.close()

# Here it clearly shows that for type Entire room/Apt and private the 2 points around 10000 are clearly outliers
# as they are very far from the mean of the data. While the point that was slightly less than 4000, seems to be a justified for
# its accomodation type which is usually a bit expensive
# Therefore we can remove those two points from the data.
# To remove the points we are only including all the data points that are below 4000.

listings = listings[listings['price'] < 4000]

# Trying to print the shape of listings after all the cleaning process
# Also to confirm that only 2 rows have been deleted and not for after outlier analysis
print(listings.shape)

# 6. Output the preprocessed data to a csv file so that it can be used in other tasks
listings.to_csv("Preprocessed_Listings.csv", index=False)
