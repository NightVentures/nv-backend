import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import pickle

def train_model():
    
    df = pd.read_csv('./music_data.csv', index_col = 0)

    X = df.drop(['genre', 'artist', 'title', 'year','duration','dB','speechiness'], axis = 1).values
    y = df['genre'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 21)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    knn = KNeighborsClassifier()
    knn.fit(X_train_scaled, y_train)

    pickle.dump(knn, open('model.pkl', 'wb'))

    return 'Model Trained with Accuracy: ' + str(knn.score(X_test_scaled, y_test))

def predict_model(features):
    knn = pickle.load(open('model.pkl', 'rb'))

    # make sure features is a 2d array
    features = np.array(features).reshape(1, -1)

    # features = [bpm, energy, danceability, liveness, valence, acousticness, popularity]
    
    # test example in swagger api endpoint: 125, 69, 63, 67, 10, 0, 73

    return knn.predict(features)[0]