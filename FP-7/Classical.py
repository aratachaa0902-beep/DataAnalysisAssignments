#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os 

def load_and_prepare_data():
    """
    Loads 'student-math.csv' and renames the columns according to the definitions in parse_data.py.
    """
    file_path = 'student-math.csv'
    if not os.path.exists(file_path):
        print(f"Error: '{file_path}' not found. Please ensure it is in the same directory as classical.py.")
        return None
    
    try:
        df0 = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None

    
    df = df0[['G3','Medu','Fedu','studytime','failures','G1','G2']]


    df = df.rename( columns={
        'G3': 'Final_Grade', 
        'Medu': 'Mother_Education',
        'Fedu': 'Father_Education',
        'G1': 'Grade_1',
        'G2': 'Grade_2'} 
    )
    
    print("Data loaded and prepared successfully.")
    return df

def perform_t_test(data_group, group_name):
    """Medu（High/Low）"""
    
  
    edu_low = data_group[data_group['Mother_Education'] < 3]['Final_Grade']
    edu_high = data_group[data_group['Mother_Education'] >= 3]['Final_Grade']
    
    if len(edu_low) < 2 or len(edu_high) < 2:
        return (
            f"### {group_name} グループにおける母親の学歴による成績比較\n"
            f"Warning: サンプルサイズが小さすぎるため、T検定を実行できませんでした。\n"
        ), None, None
    

    t_stat, p_value = stats.ttest_ind(edu_high, edu_low, equal_var=False)
    
  
    mean_high = edu_high.mean()
    mean_low = edu_low.mean()
    
   
    significance = ""
    if p_value < 0.001:
        significance = "*** (p < 0.001)"
    elif p_value < 0.01:
        significance = "** (p < 0.01)"
    elif p_value < 0.05:
        significance = "* (p < 0.05)"

    
    result = (
        f"### {group_name} グループにおける母親の学歴による成績比較\n"
        f"**高学歴母の平均成績**: {mean_high:.2f} (n={len(edu_high)})\n"
        f"**低学歴母の平均成績**: {mean_low:.2f} (n={len(edu_low)})\n"
        f"T統計量: {t_stat:.2f}, P値: {p_value:.4f} {significance}\n"
    )
    
    return result, t_stat, p_value

def plot_insightful_results(df):
    """main.ipynbに掲載する2つの主要な図を作成し、ファイルに保存する関数"""
    
  
    df_plot = df.copy()
 
    df_plot['Study_Group'] = df_plot['studytime'].apply(lambda x: 'High ST (>=3)' if x >= 3 else 'Low ST (<3)')
   
    df_plot['Mother_Edu_Group'] = df_plot['Mother_Education'].apply(lambda x: 'High Edu (>=3)' if x >= 3 else 'Low Edu (<3)')

    
    plt.figure(figsize=(6, 5))
    sns.boxplot(x='Mother_Edu_Group', y='Final_Grade', 
                data=df_plot[df_plot['Study_Group'] == 'High ST (>=3)'])
    plt.title('Figure 1: High Study Time Group (Final Grade vs Mother\'s Education)')
    plt.xlabel('Mother\'s Education Group')
    plt.ylabel('Final Grade (G3)')
    plt.ylim(0, 20)
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    plt.savefig('figure1_high_st_ttest.png')
    plt.close()

    
    plt.figure(figsize=(6, 5))
    sns.boxplot(x='Mother_Edu_Group', y='Final_Grade', 
                data=df_plot[df_plot['Study_Group'] == 'Low ST (<3)'])
    plt.title('Figure 2: Low Study Time Group (Final Grade vs Mother\'s Education)')
    plt.xlabel('Mother\'s Education Group')
    plt.ylabel('Final Grade (G3)')
    plt.ylim(0, 20)
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    plt.savefig('figure2_low_st_ttest.png')
    plt.close()

    print("\n--- 可視化ファイル作成完了 ---")
    print("以下の2つの画像ファイルが作成されました。これらを main.ipynb に貼り付けてください。")

def main():
    
    df = load_and_prepare_data()
    
    if df is None:
        return

    print("\n--- 仮説検定結果 (classical.py) ---\n")

  
    df_low_st = df[df['studytime'] < 3]
    df_high_st = df[df['studytime'] >= 3]

    
    result_A, t_stat_A, p_value_A = perform_t_test(df_high_st, "勉強時間 高 (Study Time >= 3)")
    print(result_A)
    print("-" * 30)

    
    result_B, t_stat_B, p_value_B = perform_t_test(df_low_st, "勉強時間 低 (Study Time < 3)")
    print(result_B)
    print("-" * 30)

  
    plot_insightful_results(df)
    

if __name__ == '__main__':
    main()


# In[ ]:




