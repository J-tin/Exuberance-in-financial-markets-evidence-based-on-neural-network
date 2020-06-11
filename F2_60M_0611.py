#!/usr/bin/env python
# coding: utf-8

# In[84]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from math import exp, expm1


# ## 1. Load the dataset & label 2 classes

# In[85]:


df_raw = pd.read_csv("https://raw.githubusercontent.com/J-tin/CarryOn/Exuberance-in-Financial-Markets-Evidence-from-Machine-Learning-Algorithms/Data_SVM.csv")

# Delete the second row of the dataframe 
df = df_raw.iloc[1:]

# Rename the first row of the dataframe
df_new = df.rename(columns={'MSCI World ': 'Date', 'NDDUWI': 'monthly_price', 'MSCI WORLD U$ - PRICE/BOOK RATIO':'P/B', 'MSCI WORLD U$ - DIVIDEND YIELD':'div Y'})

# Dividend Yield devides by 100.
for i in range(len(df_new['div Y'])):
    df_new.loc[i+1,'div Y'] = float(df_new.loc[i+1,'div Y'])/100

### Calculate 1-month, 12-month, 60-month returns
lags = [1, 12, 60]

# Convert pandas.Series from object to float using pd.to_numeric(s, errors='coerce')
for lag in lags:
    df_new[f'return_{lag}m']=(pd.to_numeric(df_new['monthly_price'], errors='coerce').
                            pct_change(lag)
                            .add(1)
                            .pow(1/lag)
                            .sub(1))
print(df_new.head(15))    # Reference link: https://books.google.de/books?id=tx2CDwAAQBAJ&pg=PA101&lpg=PA101&dq=compute+returns+for+various+holding+periods+using+python&source=bl&ots=uj8KrNq8ls&sig=ACfU3U0xMVTXnEjGPdLu8oqIRvXzjd3DaQ&hl=zh-CN&sa=X&ved=2ahUKEwijv5Cpzs7pAhWK-6QKHfDBAYwQ6AEwCnoECAkQAQ#v=onepage&q=compute%20returns%20for%20various%20holding%20periods%20using%20python&f=false

df_new.info()


# ### 1.1 Calculate the 25% quantile value in 3 cases

# In[86]:


# Descriptive statistics (excluding NaN values)
stats = df_new.describe()
print(stats)

# Accessing 25% quantile value 
pi_1m = stats.loc['25%', 'return_1m']
pi_12m = stats.loc['25%', 'return_12m']
pi_60m = stats.loc['25%', 'return_60m']

print(pi_1m, pi_12m, pi_60m)


# ### 1.2.1 Create a labelled table for 60-month investment: "df_60m"

# In[87]:


obs_60m = stats.loc['count','return_60m']
df_60m = df_new[['P/B', 'div Y', 'return_60m']]
df_60m = df_60m.loc[0:484,:]    
return_60m_noNaN = df_new['return_60m'].dropna() 
for i in range(len(df_60m['return_60m'])):   
    df_60m.iat[i,2] = return_60m_noNaN[i+61]
print(df_60m.tail())

# Add 'label' column in the datafram 'df_60m'
label = []
for i in range(len(df_60m['return_60m'])):
    return_60m = df_60m.iat[i, 2]
    if return_60m >= pi_60m:
        label.append(0)
    else:
        label.append(1)

df_60m['label'] = label  
df_60m = df_60m.apply(pd.to_numeric, errors='coerce')
df_60m.tail()


# ### 1.2.3 Create a labelled table for 60-month investment: "df_60m"

# In[46]:


obs_60m = stats.loc['count','return_60m']
df_60m = df_new[['P/B', 'div Y', 'return_60m']]
df_60m = df_60m.loc[0:484,:]    
return_60m_noNaN = df_new['return_60m'].dropna() 
for i in range(len(df_60m['return_60m'])):   
    df_60m.iat[i,2] = return_60m_noNaN[i+61]
print(df_60m.tail())

# Add 'label' column in the datafram 'df_60m'
label = []
for i in range(len(df_60m['return_60m'])):
    return_60m = df_60m.iat[i, 2]
    if return_60m >= pi_60m:
        label.append(0)
    else:
        label.append(1)

df_60m['label'] = label  
df_60m = df_60m.apply(pd.to_numeric, errors='coerce')
df_60m.tail()


# # 0. Preprocess the data -- Scaling the data

# In[47]:


# Create a numpy-array (matrix) X that contains 2 features  (x1,x2)
# Create a numpy-array (vector) Y that contains your labels (1,0)

from sklearn import preprocessing           
X_data_60m = df_60m.iloc[:,0:2].values           # (484,2) 
scaler = preprocessing.StandardScaler().fit(X_data_60m)
X_60m = scaler.transform(X_data_60m)
Y_60m = df_60m['label'].iloc[:484].values                 # (484,)

m = X_60m.shape[0]                                  # Training set size
shape_X = X_60m.shape
shape_Y = Y_60m.shape                           # (484,1)

print ('The shape of X is: ' + str(shape_X))
print ('The shape of Y is: ' + str(shape_Y))
print ('I have m = %d training examples!' % (m))


# ##########################################################################################################

# # Model 3: DNN with 2-hidden layers (12, 8)  by different alphas

# In[48]:


import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline


# ## 3.1 Prepare & Split the dataset

# In[49]:


from sklearn import preprocessing           
X_data_60m = df_60m.iloc[:,0:2].values           # (484,2) 
scaler = preprocessing.StandardScaler().fit(X_data_60m)
X_60m = scaler.transform(X_data_60m)
Y_60m = df_60m['label'].iloc[:484].values 
datasets = [(X_60m, Y_60m)]


# ## 3.2 Define the MLPClassifier 

# In[80]:



# Define a 2-hidden-layer NN model based on varying regularization (alpha) in MLPClassifier

clf=MLPClassifier(solver='lbfgs',        #‘lbfgs’ is an optimizer in the family of quasi-Newton methods.
                                     alpha=0.017782,               # L2 penalty (regularization term) parameter   
                                     activation='logistic', # Activation function for the hidden layer. 
                                                            # 'logistic’, the logistic sigmoid function, returns f(x) = 1 / (1 + exp(-x)).
                                     random_state=1,        # Determines random number generation for weights and bias initialization, train-test split if early stopping is used
                                     max_iter=2000,         # Maximum number of iterations. The solver iterates until convergence (determined by ‘tol’) or this number of iterations.
                                     early_stopping=True,   # Whether to use early stopping to terminate training when validation score is not improving.
                                                             # True: it will automatically set aside 10% of training data as validation and terminate training when validation score is not improving by at least tol for n_iter_no_change consecutive epochs.
                                     hidden_layer_sizes=[12, 8]) # We build 2 hidden layers. 12 neurons in the 1st layer; 8 in the 2nd one.



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_60m, Y_60m, test_size=.2, random_state = 42)

clf.fit(X_train, y_train)               # Train the model with automatically 10-fold cross validation which has been specidies in Classifiers
score = clf.score(X_test, y_test)

prob = clf.predict_proba(X_60m)[:,1]
prob.size

results = clf.predict_proba(X_60m)[1]
# gets a dictionary of {'class_name': probability}
prob_per_class_dictionary = dict(zip(clf.classes_, results))

pb = clf.predict_log_proba(X_60m)
pb


# In[ ]:





# In[90]:


##slice date from raw dataframe
dates = pd.to_datetime(df_new['Date'])
date = dates[60:544] 


##merge
data_60m = pd.DataFrame(prob)
  
data_60m.columns = ['Prob']

data_60m['Date']=date


print(data_60m.tail())
print(data_60m.head())


# In[96]:
import matplotlib.pyplot as plt

plt.plot(data_60m['Date'], data_60m['Prob']*10,linewidth=1)
my_y_ticks = np.arange(0,10,1)
plt.yticks = (my_y_ticks)
plt,ylim((0,10))
plt.show()



