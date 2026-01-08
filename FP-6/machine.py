#!/usr/bin/env python
# coding: utf-8

# In[1]:


# machine.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def load_and_prepare_data():
    """
    'student-math.csv' を読み込み、カラム名を変更する（parse_data.py と共通）
    """
    file_path = 'student-math.csv'
    if not os.path.exists(file_path):
        print(f"Error: '{file_path}' not found.")
        return None
    
    df0 = pd.read_csv(file_path)
    df = df0[['G3','Medu','Fedu','studytime','failures','G1','G2']]
    df = df.rename(columns={
        'G3': 'Final_Grade', 
        'Medu': 'Mother_Education',
        'Fedu': 'Father_Education',
        'G1': 'Grade_1',
        'G2': 'Grade_2'} 
    )
    return df

def main():

    df = load_and_prepare_data()
    if df is None: return


    features = ['Mother_Education', 'studytime', 'failures', 'Grade_1', 'Grade_2']
    X = df[features]
    y = df['Final_Grade']

 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    model = LinearRegression()
    model.fit(X_train, y_train)


    y_pred = model.predict(X_test)

  
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n--- Machine Learning Results ---")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R-squared (R2) Score: {r2:.2f}")
    print("--------------------------------\n")

    
    plt.figure(figsize=(6, 5))
    plt.scatter(y_test, y_pred, alpha=0.6, color='blue')
    plt.plot([0, 20], [0, 20], '--', color='red', label='Ideal Prediction')
    plt.xlabel('Actual Final Grade')
    plt.ylabel('Predicted Final Grade')
    plt.title('Figure 3: Actual vs Predicted Final Grades')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('figure3_ml_prediction.png')
    plt.close()

   
    importance = pd.Series(model.coef_, index=features).sort_values()
    plt.figure(figsize=(6, 5))
    importance.plot(kind='barh', color='skyblue')
    plt.title('Figure 4: Impact of Each Factor (Model Coefficients)')
    plt.xlabel('Coefficient Value')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('figure4_ml_importance.png')
    plt.close()

    print("Success: figure3_ml_prediction.png and figure4_ml_importance.png have been saved.")

if __name__ == '__main__':
    main()


# In[ ]:





