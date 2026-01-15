#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

def load_data():
    if not os.path.exists('student-math.csv'): return None
    df = pd.read_csv('student-math.csv')
    df = df[['G3','Medu','studytime','failures','G1','G2']]
    return df.rename(columns={'G3': 'Final_Grade', 'Medu': 'Mother_Education'})

def run_analysis(X, y, iterations=100):
    scores = []
    scaler = StandardScaler() 
    
    for i in range(iterations):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = DecisionTreeRegressor(max_depth=3)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        scores.append(r2_score(y_test, y_pred))
    return scores

def main():
    df = load_data()
    if df is None: return

    df_low = df[df['Final_Grade'] <= 10]
    df_high = df[df['Final_Grade'] > 10]
    features = ['Mother_Education', 'studytime', 'failures', 'G1', 'G2']

    scores_low = run_analysis(df_low[features], df_low['Final_Grade'])
    scores_high = run_analysis(df_high[features], df_high['Final_Grade'])

    plt.figure(figsize=(8, 5))
    sns.kdeplot(scores_low, label='Low Grade Group (Scaled)', fill=True, color='blue')
    sns.kdeplot(scores_high, label='High Grade Group (Scaled)', fill=True, color='orange')
    plt.title('Figure 5: Improved R2 Scores (StandardScaler + Decision Tree)')
    plt.xlabel('R2 Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('figure5_improved_validation.png')
    plt.close()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    model_full = DecisionTreeRegressor(max_depth=3).fit(X_scaled, df['Final_Grade'])
    
    importance = pd.Series(model_full.feature_importances_, index=features).sort_values()
    plt.figure(figsize=(8, 5))
    importance.plot(kind='barh', color='green')
    plt.title('Figure 6: Feature Importance after Scaling')
    plt.xlabel('Importance')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('figure6_feature_importance.png')
    plt.close()

    print(f"Improved Low Grade Mean R2: {np.mean(scores_low):.4f}")
    print(f"Improved High Grade Mean R2: {np.mean(scores_high):.4f}")

if __name__ == '__main__':
    main()


# In[ ]:




