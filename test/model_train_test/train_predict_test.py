# a jupyter notebook that goes through neural network model in OpenCV

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import glob
import sys
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


print ("OpenCV:",  cv2.__version__)
print ("Numpy : ", np.__version__)
print ("Python:",  sys.version)


# In[3]:


# load training data
dim = 240*320
X = np.empty((0, dim))
y = np.empty((0, 4))
training_data = glob.glob('data_test.npz')

for single_npz in training_data:
    with np.load(single_npz) as data:
        train = data['train']
        train_labels = data['train_labels']
    X = np.vstack((X, train))
    y = np.vstack((y, train_labels))

print ('Image array shape: ', X.shape)
print ('Label array shape: ', y.shape)


# In[4]:


plt.imshow(X[0].reshape(240, 320), cmap='gray')


# In[5]:


# create a neural network
model = cv2.ml.ANN_MLP_create()
layer_sizes = np.int32([dim, 32, 4])
model.setLayerSizes(layer_sizes)
model.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP)
model.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM, 2, 1)
model.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 20, 0.01))


# In[6]:


# training
model.train(np.float32(X), cv2.ml.ROW_SAMPLE, np.float32(y))


# In[7]:


# evaluate on training data
ret, resp = model.predict(X)
prediction = resp.argmax(-1)
true_labels = y.argmax(-1)

train_rate = np.mean(prediction == true_labels)
print (len(prediction))
print (prediction)
print ('Train accuracy: ', "{0:.2f}%".format(train_rate * 100))


# In[8]:


# save model
model.save('model_test.xml')


# In[ ]:





# In[ ]:





# In[9]:


# load model
model = cv2.ml.ANN_MLP_load('model_test.xml')


# In[10]:


# predict
ret, resp = model.predict(X)
print (len(resp))
resp.argmax(-1)


# In[ ]:




