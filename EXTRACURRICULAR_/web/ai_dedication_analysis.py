import numpy as np
import gensim
import nltk
import re
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
from nltk.tokenize import sent_tokenize
import multiprocessing
import os
from pathlib import Path
import io
from gensim.models import Phrases
from textblob import TextBlob
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer


### 웹사이트에서 값을 입력받아야 함 ###

#input
hrs_per_week = 5 #time_spent_hrs_per_week
weeks_per_year = 20 #tiem_spent_weeks_per_year
period_years_of_activity = 3 #period_years_of_activity


def dedication_analysis(hrs_per_week, weeks_per_year, period_years_of_activity):

    input_values = [float(hrs_per_week), float(weeks_per_year), float(period_years_of_activity)] #입력값이 int라면 float로 변환하기
    
    # 비교할 데이터 불러오기
    df = pd.read_csv('dedication_analysis .csv')

    #한주별 활동 시간 강도
    for i, row in df.iterrows(): #한행씩 읽어와서
        if df.at[i, 'hrs_per_week'] == input_values[0]: #입력한 값(12.0)과 같은 값이 hrs_per_week 열의 행값과 같다면,
            ws_score = df.at[i, 'week_strength'] # 그 행에서 컬럼이름 week_strength의 해당 값을 가져온다.
        elif df.at[i, 'hrs_per_week'] > input_values[0]:
            ws_score = df.at[0, 'week_strength'] # 값이 없다면, 그 이상되는 값이기 때문에 최대 점수를 줌 5.0
        else:
            pass
    ##print ('ws_score :', ws_score)

    #연간 몇 주 참여강도
    for k, row in df.iterrows(): #한행씩 읽어와서
        if df.at[k, 'weeks_per_year'] == input_values[1]: #입력한 값(12.0)과 같은 값이 hrs_per_week 열의 행값과 같다면,
            wy_score = df.at[k, 'year_strength'] # 그 행에서 컬럼이름 week_strength의 해당 값을 가져온다.
        elif df.at[k, 'weeks_per_year'] > input_values[1]:
            wy_score = df.at[0, 'year_strength'] # 값이 없다면, 그 이상되는 값이기 때문에 최대 점수를 줌 5.0
        else:
            pass
    #print ('wy_score :', wy_score)

    #Coefficient
    for l, row in df.iterrows(): #한행씩 읽어와서
        if df.at[l, 'period_years_of_activity'] == input_values[2]: #입력한 값(12.0)과 같은 값이 hrs_per_week 열의 행값과 같다면,
            py_score = df.at[l, 'coefficient'] # 그 행에서 컬럼이름 week_strength의 해당 값을 가져온다.
        elif 6 <= input_values[2]: # 6이면 
            py_score = df.at[0, 'coefficient'] # 값이 없다면, 그 이상되는 값이기 때문에 최대 점수를 줌 5.0
        elif 0 == input_values[2]: # 0이면 
            py_score = 0.1
        else:
            pass
    #print ('py_score :', py_score)

    #######   Dedication 계산 (활동별 / 전체적) 

    dedecation_cal = round(py_score * ((ws_score + wy_score)/2),2)


    return dedecation_cal



############## 코드테스트 ###############

# result = dedication_analysis(hrs_per_week, weeks_per_year, period_years_of_activity)

#print ("=================================")
#print ("RESULT :", result)
#print ("=================================")


# ws_score : 3.0
# wy_score : 4.5
# py_score : 0.9
# =================================
# RESULT : 3.375
# =================================