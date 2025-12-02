#!/usr/bin/env python
# coding: utf-8

# # Final Project :  Descriptive Statistics 
# 
# 
# <br>
# <br>
# 

# ___
# 
# <br>
# 
# # Analysis of Socioeconomic and Lifestyle Factors Affecting Student Academic Performance
# 
# 
# * **Name**:  Arata OHORI
# * **Student number**:  0400350126
# 
# <br>

# ### Purpose:
# 
# * The purpose of this Final Project is to analyze the extent to which socioeconomic factors and student lifestyle choices contribute to academic performance variability.
# * The key dependent variable (DV) is the final grade in mathematics ($G3$), which is a score ranging from 0 to 20.
# * Key **independent variables** (IVs) include:
#     * parental education ($Medu$, $Fedu$)
#     * family support ($famsup$)
#     * study time ($studytime$)
#     * failures ($failures$)
#     * alcohol consumption ($Dalc$, $Walc$)
# 
# * This dataset contains 395 cases (i.e., 395 students). There is 1 key DV ($G3$) and a total of 32 IVs (including the selected key IVs) for analysis.
# <br>

# ### Dataset source:
# 
# The data come from the [student academic performance](https://archive.ics.uci.edu/dataset/320/student+performance) dataset from the [UCI Machine Learning repository](https://archive.ics.uci.edu/ml/index.php):
# 
# The data are available for download [here](https://archive.ics.uci.edu/static/public/320/student+performance.zip).
# 
# 
# 
# 
# #### References:
# 
# 

# ___
# 
# ## Descriptive Statistics
# 
# 

# In[7]:


get_ipython().run_cell_magic('capture', '', '%run descriptive.ipynb\n')


# In[2]:


display_central_tendency_table(num=1)
display_dispersion_table(num=2)


# In[8]:


plot_descriptive()


# In[ ]:




