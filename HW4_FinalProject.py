import kagglehub
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('GTK3Agg')  # Or 'Qt5Agg', 'GTK3Agg', etc.
import matplotlib.pyplot as plt
import seaborn as sns


# Authenticate
# kagglehub.login() # This will prompt you for your credentials.
# We also offer other ways to authenticate (credential file & env variables): https://github.com/Kaggle/kagglehub?tab=readme-ov-file#authenticate

# Download latest version
path = kagglehub.dataset_download("prishasawhney/imdb-dataset-top-2000-movies")
print(path + '/imdb_top_2000_movies.csv')
imdb = pd.read_csv(path + '/imdb_top_2000_movies.csv')
print(imdb.head())

# EDA
print(imdb.shape)
print(imdb.columns)  # Print columns to verify the name
imdb.info()
imdb.describe()
imdb['Genre'].value_counts()

# Clean the data
imdb.dropna(inplace=True)
imdb.drop_duplicates(inplace=True)
imdb['Genre'] = imdb['Genre'].str.replace(' ','')
imdb['Genre'] = imdb['Genre'].str.lower()
imdb['Gross'] = imdb['Gross'].str.replace('$','')
imdb['Gross'] = imdb['Gross'].str.replace(',','')
imdb['Gross'] = imdb['Gross'].str.replace('M','')
imdb['Gross'] = pd.to_numeric(imdb['Gross'])
imdb['Votes'] = imdb['Votes'].str.replace(',','')
imdb['Votes'] = pd.to_numeric(imdb['Votes'])
imdb['IMDB Rating'] = pd.to_numeric(imdb['IMDB Rating'])
imdb['Release Year'] = imdb['Release Year'].str.replace('I', '', regex=False)  # Remove "I " specifically
imdb['Release Year'] = pd.to_numeric(imdb['Release Year'])
imdb.info()

# Visualizations
plt.ion()
plt.figure(figsize=(10, 6))
sns.histplot(imdb['IMDB Rating'], kde=True)
plt.title('Distribution of Movie Ratings')
plt.xlabel('IMDB Rating')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='Genre', y='IMDB Rating', data=imdb)
plt.title('Rating by Genre')
plt.xticks(rotation=45, ha='right')
plt.show()
plt.pause(0.001)

# Correlation matrix
numeric_imdb = imdb.select_dtypes(include=np.number)
correlation_matrix = numeric_imdb.corr()

# Visualize the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
plt.pause(0.001)

# Train/Test separation
from sklearn.model_selection import train_test_split
X = imdb.drop(['Movie Name', 'Gross'],axis=1)
X = pd.get_dummies(X, columns=['Cast', 'Director', 'Genre'], prefix=['Cast', 'Director', 'Genre'])
y = imdb['Gross']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SVM model training
from sklearn.svm import SVR
model_svr = SVR()
model_svr.fit(X_train, y_train)

# Model evaluation
y_pred_svr = model_svr.predict(X_test)
mse_svr = mean_squared_error(y_test, y_pred_svr)
print("Mean Squared Error (SVR):", mse_svr)

# Random forest model training
from sklearn.ensemble import RandomForestRegressor
model_rf = RandomForestRegressor()
model_rf.fit(X_train, y_train) 

# Model evaluation
from sklearn.metrics import mean_squared_error
y_pred = model_rf.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred)
print("Mean Squared Error (RF):", mse_rf)

# Nural network based on the data above
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore

# Define the model
model_nn = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])

# Compile the model
model_nn.compile(optimizer='adam', loss='mse')

# Train the model
model_nn.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

# Plot loss vs epochs
plt.plot(model_nn.history.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()
plt.pause(0.001)

# Evaluate the model
loss = model_nn.evaluate(X_test, y_test)
print("Mean Squared Error:", loss)

plt.ioff()
plt.show()