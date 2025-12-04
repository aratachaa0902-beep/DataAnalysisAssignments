#!/usr/bin/env python
# coding: utf-8

# # parse_data.ipynb
# 
# This notebook parses the data files used for the FP-2 assignment. 
# 
# <br>
# <br>
# 
# First let's read the attached data file:

# In[1]:


import pandas as pd

df0 = pd.read_csv('student-math.csv')

df0.describe()


# <br>
# <br>
# 
# The dependent and independent variables variables (DVs and IVs) that we are interested in are:
# 
# **DVs**:
# - Quality  (the judged wine quality)
# 
# **IVs**:
# - Acidity ("volatile acidity" column in the CSV file)
# - Density ("density" column in the CSV file)
# - Sugar  ("residual sugar" column in the CSV file)
# 
# 
# <br>
# <br>
# 
# Let's extract the relevant columns:

# In[2]:


df = df0[['G3','Medu','Fedu','studytime','failures','G1','G2']]

df.describe()


# <br>
# <br>
# 
# Next let's use the `rename` function to give the columns simpler variable names:

# In[3]:


df = df.rename( columns={
    'Medu': 'Mother_Education','Fedu': 'Father_Education','G1': 'Grade_1','G2': 'Grade_2','G3': 'Final_Grade'} )

df.describe()


# In[ ]:




