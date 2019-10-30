import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


from sklearn.pipeline import Pipeline

from sklearn.neural_network import MLPClassifier

train_images = pd.read_pickle('./data/train_max_x')
train_output = pd.read_csv('./data/train_max_y.csv')
test_images = pd.read_pickle('./data/test_max_x')


train_images = train_images.reshape(50000,16384)
test_images = test_images.reshape(10000,16384)


pca = PCA(n_components=500)
pcs = pca.fit_transform(train_images)
pDF = pd.DataFrame(data = pcs)

def accuracy(predicted,true_outcome,num):
    accuracy = 0
    index = 0
    for result in predicted:
        if result == true_outcome[index]:
            accuracy+=1
        index+=1
    print("-----Accuracy:", accuracy/num)

numtrain = 40000

MLP_clf = Pipeline([
        ('clf', MLPClassifier(learning_rate ="adaptive", max_iter = 50)),
        ])

MLP_clf.fit(train_images[numtrain:],train_output['Label'][numtrain:])
MLP_pred = MLP_clf.predict(train_images[:numtrain])
accuracy(MLP_pred,train_output['Label'][:numtrain],numtrain)