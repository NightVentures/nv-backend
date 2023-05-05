import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import pickle

def train_model():
    
    df = pd.read_csv('./music_data.csv', index_col = 0)

    X = df.drop(['genre', 'artist', 'title', 'year','duration','dB','speachiness'], axis = 1).values
    y = df['genre'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 21)

    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)

    pickle.dump(knn, open('model.pkl', 'wb'))

    return 'Model Trained with Accuracy: ' + str(knn.score(X_test, y_test))

def predict_model(features):
    knn = pickle.load(open('model.pkl', 'rb'))

    # make sure features is a 2d array
    features = np.array(features).reshape(1, -1)

    # features = [bpm, energy, danceability, liveness, valence, acousticness, popularity]

    return knn.predict(features)[0]