import pandas as pd
from sklearn.decomposition import PCA

from sklearn.pipeline import Pipeline

train_images = pd.read_pickle('./data/train_max_x')
train_output = pd.read_csv('./data/train_max_y.csv')
test_images = pd.read_pickle('./data/test_max_x')

pca = PCA(n_components=2)

