# Load data
import numpy as np
from sklearn import datasets
iris_X, iris_y = datasets.load_iris(return_X_y=True)
from google.cloud import storage

# Split iris data in train and test data
# A random permutation, to split the data randomly
np.random.seed(0)
indices = np.random.permutation(len(iris_X))
iris_X_train = iris_X[indices[:-10]]
iris_y_train = iris_y[indices[:-10]]
iris_X_test = iris_X[indices[-10:]]
iris_y_test = iris_y[indices[-10:]]

# Create and fit a nearest-neighbor classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(iris_X_train, iris_y_train)
knn.predict(iris_X_test)

# save model
import pickle
with open('model.pkl', 'wb') as file:
    pickle.dump(knn, file)

storage_client = storage.Client()
bucket = storage_client.bucket('mlops-project-6')
blob = bucket.blob("m22/model/model.pkl")

blob.upload_from_filename('model.pkl')

print(
    "File {} uploaded to {}.".format(
        'model.pkl', "m22/model/model.pkl"
    ))
