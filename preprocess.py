# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 18:22:19 2019

@author: shuha
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

train_img = pd.read_pickle('./data/train_max_x')


#plt.imshow(train_img[103],cmap='gray')


def preprocess(x):


            for t in range(len(x)):
                image = x[t]
                image =(image>220).astype('int32')*255
                x[t] = image


preprocess(train_img)
#plt.imshow(train_img[103],cmap='gray')

res = train_img
res.tofile(r'processed_data', sep="", format="%s")

#np.savetxt(r'g:\test.csv',res,delimiter=',', fmt=('%s, %f'))

    

    

