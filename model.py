import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

import pickle

def train_model():
    
    df = pd.read_csv('./music_data.csv', index_col = 0)

    X = df.drop(['genre', 'artist', 'title', 'year','duration','dB','speechiness'], axis = 1).values
    y = df['genre'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    svm_model = SVC(C=100)

    svm_model.fit(X_train_scaled, y_train)

    pickle.dump(svm_model, open('model.pkl', 'wb'))

    return 'Model Trained with Accuracy: ' + str(svm_model.score(X_test_scaled, y_test))

# print(train_model())

def predict_model(features):
    svm_model = pickle.load(open('model.pkl', 'rb'))

    # make sure features is a 2d array
    features = np.array(features).reshape(1, -1) / 100

    # features = [bpm, energy, danceability, liveness, valence, acousticness, popularity]

    return svm_model.predict(features)[0]

# print(predict_model([10,50,60,80,70,80,100]))