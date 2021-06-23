import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

# Read the file to be used for prediction
listings = pd.read_csv('Preprocessed_Listings.csv')

# Categorical features have to be one-hot encoded
transformed_listings = pd.get_dummies(listings)

# Select only the numeric features for standardization and normalisation
numeric_features = [
    'availability_365', 'calculated_host_listings_count', 'minimum_nights',
    'number_of_reviews', 'price', 'longitude', 'latitude'
]

# Plot histogram of only numeric features to find if there is any skewness
transformed_listings[numeric_features].hist(figsize=(10, 11))
plt.show()
plt.close()
# shows there is a positive skew in all features other than 'availability_365', 'calculated_host_listings_count','longitude'

# Removing items not to be transformed
numeric_features = [
    i for i in numeric_features if i not in
    ['availability_365', 'calculated_host_listings_count', 'longitude']
]

# Transforming the features with logarithmic transformation
for col in numeric_features:
    transformed_listings[col] = transformed_listings[col].astype(
        'float64').replace(0.0, 0.01)
    transformed_listings[col] = np.log(transformed_listings[col])

transformed_listings[numeric_features].hist(figsize=(10, 11))
plt.show()
plt.close()

# spliting predictive features and target
X = transformed_listings.drop('price', axis=1)
y = transformed_listings.price

# Scaling predictive features X with a max abs scaler
scaler = MaxAbsScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=list(X.columns))

# Splitting dataset into test and train dataset
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=123)

# XgBoost Model
xgboost = xgb.XGBRegressor()
# Training the model on train dataset
xgboost.fit(X_train, y_train)
# Creating predictions from the model that was trained
training_predictions_xgboost = xgboost.predict(X_train)
validation_predictions_xgboost = xgboost.predict(X_test)

# Finding out Mean square error and R-squared score
print("\nTraining Mean Square Error:",
      round(mean_squared_error(y_train, training_predictions_xgboost), 4))
print("Validation Mean Square Error:",
      round(mean_squared_error(y_test, validation_predictions_xgboost), 4))
print("\nTraining R-squared score %:",
      round(r2_score(y_train, training_predictions_xgboost) * 100, 4))
print("Validation R-squared score %:",
      round(r2_score(y_test, validation_predictions_xgboost) * 100, 4))

# Calculating feature weights in the prediction model 
feature_weights_xgboost = pd.DataFrame(xgboost.feature_importances_,
                                       columns=['weight'],
                                       index=X_train.columns)
feature_weights_xgboost.sort_values('weight', inplace=True)

plt.figure(figsize=(8, 20))
plt.barh(feature_weights_xgboost.index,
         feature_weights_xgboost.weight,
         align='center')
plt.title("Feature importances in the prediction model (XGBoost)", fontsize=14)
plt.xlabel("Feature importance")
plt.tight_layout()
plt.margins(y=0.01)
plt.show()
plt.close()