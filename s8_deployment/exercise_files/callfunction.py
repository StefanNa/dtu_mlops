import numpy as np
from sklearn import datasets
from google.cloud import storage
import pickle
import requests

url='https://europe-west1-ecstatic-elf-337907.cloudfunctions.net/m22'
BUCKET_NAME = 'mlops-project-6'
MODEL_FILE = 'm22/model/model.pkl'

iris_X, iris_y = datasets.load_iris(return_X_y=True)

client = storage.Client()
bucket = client.get_bucket(BUCKET_NAME)
blob = bucket.get_blob(MODEL_FILE)
my_model = pickle.loads(blob.download_as_string())


np.random.seed(0)
indices = np.random.permutation(len(iris_X))
iris_X_train = iris_X[indices[:-10]]
iris_y_train = iris_y[indices[:-10]]
iris_X_test = iris_X[indices[-10:]]
iris_y_test = iris_y[indices[-10:]]



r=requests.post(url,json={'input_data':[1,2,3,4]})
# r=requests.get(url)
print( r.text)



"""
from google.cloud import storage
import pickle

BUCKET_NAME = 'mlops-project-6'
MODEL_FILE = 'm22/model/model.pkl'

client = storage.Client()
bucket = client.get_bucket(BUCKET_NAME)
blob = bucket.get_blob(MODEL_FILE)
my_model = pickle.loads(blob.download_as_string())

def knn_classifier(request):
   request_json = request.get_json()
   if request_json and 'input_data' in request_json:
         data = request_json['input_data']
         input_data = list(map(int, data.split(',')))
         prediction = my_model.predict([input_data])
         return f'Belongs to class: {prediction}'
   else:
         return 'No input data received'


"""