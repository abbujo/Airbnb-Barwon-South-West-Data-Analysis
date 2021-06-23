import pandas as pd
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap

listings = pd.read_csv('Preprocessed_Listings.csv')

# Subtask A: Price column visualisation using boxplot
# Here we will plot the following graphs
# A.1. BoxPlot for price range in each neighbourhood
# A.2. BoxPlot for different type of accomodation in each neighbourhood
listings.assign(index=listings.groupby('neighbourhood').cumcount()).pivot(
    'index', 'neighbourhood', 'price').plot(kind='box')

plt.title("Boxplot showing price distribution wrt Neighbourhood")
plt.show()
plt.close()

# Finding out number of unique neighbourhoods

n_unique_neighbourhood = len(listings['neighbourhood'].unique())

print("Number of unique neighbourhoods is ", n_unique_neighbourhood)

# Since the number is 9 here, i will create a 9 subplots using 3x3 grid
fig, axs = plt.subplots(3, 3)

i = 0
j = 0
for neighbourhood in listings['neighbourhood'].unique():
    neighbourhoodData = listings[listings['neighbourhood'] == neighbourhood]

    neighbourhoodData.boxplot(column='price', by='room_type', ax=axs[i, j])
    axs[i, j].set_title(neighbourhood)
    if (j == 2):
        j = 0
        i = i + 1
    else:
        j = j + 1

fig.tight_layout()
plt.show()
plt.close()

# B.1: Visualising the heatmap 
# B.2: Visualising marker maps for different accomodations Private, shared, hotel, entire home/apt
# C.1: Count of listing by neighbourhood
# C.2: Count of listing by type
# C.3: Count of listing by neighbourhood by type
# D.1: Mean price of listing by neighbourhood
# D.2: Mean price of listing by type
# D.3: Mean price of listing by neighbourhood by type

def getMap(listings):
    latitudeMean = listings['latitude'].mean()
    longitudeMean = listings['longitude'].mean()
    map = folium.Map(location=[latitudeMean, longitudeMean], zoom_start=10)
    return map


def bounds(map, listing_location):
    sw = listing_location.min().values.tolist()
    ne = listing_location.max().values.tolist()
    map.fit_bounds([sw, ne])
    return map


listings_heat_map = getMap(listings)
listing_location = listings[['latitude', 'longitude']]

heat_data = [[row['latitude'], row['longitude']]
             for index, row in listing_location.iterrows()]

# B.1: Visualising the heatmap 
HeatMap(heat_data).add_to(listings_heat_map)
listings_heat_map = bounds(listings_heat_map, listing_location)
listings_heat_map.save('heat_map.html')

private_marker_map = getMap(listings)
shared_marker_map = getMap(listings)
hotel_marker_map = getMap(listings)
apt_marker_map = getMap(listings)

# add marker one by one on the map
for i in range(0, len(listings)):
    if (listings.iloc[i]['room_type'] == "Hotel room"):
        folium.Marker(location=[
            listings.iloc[i]['latitude'], listings.iloc[i]['longitude']
        ],
                      icon=folium.Icon(color='red')).add_to(hotel_marker_map)
for i in range(0, len(listings)):
    if (listings.iloc[i]['room_type'] == "Private room"):
        folium.Marker(
            location=[
                listings.iloc[i]['latitude'], listings.iloc[i]['longitude']
            ],
            icon=folium.Icon(color='blue')).add_to(private_marker_map)
for i in range(0, len(listings)):
    if (listings.iloc[i]['room_type'] == "Shared room"):
        folium.Marker(
            location=[
                listings.iloc[i]['latitude'], listings.iloc[i]['longitude']
            ],
            icon=folium.Icon(color='yellow')).add_to(shared_marker_map)
for i in range(0, len(listings)):
    if (listings.iloc[i]['room_type'] == "Entire home/apt"):
        folium.Marker(location=[
            listings.iloc[i]['latitude'], listings.iloc[i]['longitude']
        ],
                      icon=folium.Icon(color='green')).add_to(apt_marker_map)

#  B.2: Visualising marker maps for different accomodations Private, shared, hotel, entire home/apt
private_marker_map = bounds(private_marker_map, listing_location)
shared_marker_map = bounds(shared_marker_map, listing_location)
hotel_marker_map = bounds(hotel_marker_map, listing_location)
apt_marker_map = bounds(apt_marker_map, listing_location)
private_marker_map.save('private.html')
shared_marker_map.save('shared.html')
hotel_marker_map.save('hotel.html')
apt_marker_map.save('apt.html')

# C.1: Count of listing by neighbourhood
listings_count_by_neighbourhood = listings.groupby([
    'neighbourhood'
]).size().sort_values(ascending=False).reset_index(name="count")
print(listings_count_by_neighbourhood)
listings_count_by_neighbourhood.plot(kind="barh").set_yticklabels(
    listings_count_by_neighbourhood['neighbourhood'])
plt.title("Accomodation count by neighbourhood")
plt.show()
plt.close()
listings_count_by_neighbourhood.plot(
    kind="pie",
    y="count",
    labels=listings_count_by_neighbourhood['neighbourhood'],
    figsize=(10, 10),
    radius=0.5,
    autopct='%.2f')
plt.title("Accomodation count % by neighbourhood")
plt.tight_layout()
plt.show()
plt.close()

# C.2: Count of listing by type
listings_count_by_type = listings.groupby([
    'room_type'
]).size().sort_values(ascending=False).reset_index(name="count")
listings_count_by_type.plot(kind="barh").set_yticklabels(
    listings_count_by_type['room_type'])
plt.xticks(rotation=30, horizontalalignment="right")
plt.title("Accomodation count by room type")
plt.show()
plt.close()
listings_count_by_type.plot(
    kind="pie",
    y="count",
    labels=listings_count_by_type['room_type'],
    figsize=(10, 10),
    radius=0.5,
    autopct='%.2f')
plt.title("Accomodation count % by room type")
plt.tight_layout()
plt.show()
plt.close()


# C.3: Count of listing by neighbourhood by type
listings_count_by_neighbourhood_by_type = listings.groupby([
    'neighbourhood', 'room_type'
]).size().sort_values(ascending=False).reset_index(name="count")
listings_count_by_neighbourhood_by_type.pivot('neighbourhood', 'room_type',
                                              'count').plot(kind="bar")
plt.xticks(rotation=30, horizontalalignment="right")
plt.title("Accomodation count by neighbourhood by room type")
plt.show()
plt.close()

# D.1: Mean price of listing by neighbourhood
mean_prices_by_neighbourhood = listings.groupby(
    ['neighbourhood'])['price'].mean().sort_values(ascending=False)
mean_prices_by_neighbourhood.plot(kind="bar")
plt.xticks(rotation=30, horizontalalignment="right")
plt.title("Accomodation mean prices by neighbourhood")
plt.show()
plt.close()

# D.2: Mean price of listing by type
mean_prices_by_type = listings.groupby(
    ['room_type'])['price'].mean().sort_values(ascending=False)
mean_prices_by_type.plot(kind="bar")
plt.xticks(rotation=30, horizontalalignment="right")
plt.title("Accomodation mean prices by room type")
plt.show()
plt.close()

# D.3: Mean price of listing by neighbourhood by type
mean_prices_by_neighbourhood_by_type = listings.groupby(
    ['neighbourhood',
     'room_type'])['price'].mean().reset_index(name="mean_price")

mean_prices_by_neighbourhood_by_type.pivot('neighbourhood', 'room_type',
                                           'mean_price').plot(kind="bar")
plt.xticks(rotation=30, horizontalalignment="right")
plt.title("Accomodation mean prices by neighbourhood by room type")
plt.show()
plt.close()