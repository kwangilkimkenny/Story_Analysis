# 가장 Leadership+Dedication+Major Fit 점수 합계가 높은 상위 60% 활동만 계산에 활용 
# (각 활동별로 Dedication level (60%) + Leadership level (40%) + Major Fit 가산점 더해서 가장 높은 순위부터 계산에 활용… 중요한 활동부터)

# dedicaton 값은 leadership, major_activity 값과 합쳐서 한개의 함수로 처리하여 계산할 것

#### 한개의 함수로 처리하여 계산!!!!!!!!!


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

from ai_leadership import leadership_start_here, total_act_lists
from ai_major_activity_fit  import mjr_act_analy, total_actvity
from ai_dedication_analysis  import dedication_analysis


######################## 웹사이트에서 값을 입력받아야 함 ##################################### - start -

#input
hrs_per_week = 5 #time_spent_hrs_per_week
weeks_per_year = 20 #tiem_spent_weeks_per_year
period_years_of_activity = 3 #period_years_of_activity

# 3개의 입력: 전공 3개
majors = """ mechanical engineering, Film Studies, Psychology  """

# 일단 3개의 EXTRACURRICULAR ACTIVITY EXAMPLES  입력, 추가로 활동을 입력할 수 있음. 최대 10개, 그 이상도 가능하지만 비율로 게산
input_text_1 = """ deputy Member (9th/10th) Treasurer (11th/12th) National Honors Society, Ridgefield High School Chapter
We are amongst the highest academically achieving students at our school, who collectively and consistently participate in community service projects.""" # 실제 값은 문장이 입력되어야 함, 현재는 테스트용 단어입력

input_text_2 = """ Leader/Concertmaster (10th-12th)
AMAC Youth Chamber Ensemble (AYCE), audition-based community choir 
Lead ensemble in rehearsal and performance, coordinate rehearsal times, aid younger  """

input_text_3 = """ Number 1 Doubles Starter (9th-12th), Captain (11th-12th)
JV/V Beverly Hills High School Tennis Team
Three year League Champions; planned and hosted team banquet; led team warmups and meetings; Coach's Award Recipient (11th); Team Spirit Award (12th).  """

input_text_4 = """ Peer Advisor (11th-12th)
Erving High School Student Ambassador Program, selective application-based leadership team
Organized and led orientation; served as a year round leader, mentor, tutor, and friend to freshmen; helped with class scheduling."""

input_text_5 = """ Leader (11th)
Books on Global Health Equity and Social Justice, advocacy-focused peer discussion group
Researched global health equity/social justice , assigned weekly readings for group discussion, brainstormed questions to generate input from members.  """

input_text_6 = """ Number 1 Doubles Starter (9th-12th), Captain (11th-12th)
JV/V Beverly Hills High School Tennis Team
Three year League Champions; planned and hosted team banquet; led team warmups and meetings; Coach's Award Recipient (11th); Team Spirit Award (12th).  """

input_text_7 = """ Number 1 Doubles Starter (9th-12th), Captain (11th-12th)
JV/V Beverly Hills High School Tennis Team
Three year League Champions; planned and hosted team banquet; led team warmups and meetings; Coach's Award Recipient (11th); Team Spirit Award (12th).  """

input_text_8 = """Student Coach (9th - 12th)
Middle School MathCounts Team
Taught strategies, selected competitors, hosted weekly practice sessions and lectures. Led team to 2nd place victory at State Mathematics competition (11th). """

input_text_9 = """ Protein Modeling Team Leader (10th)
Science Olympiad, Burke High School Club
Supervised building of protein molecule model, taught peers to use 3D molecular program Jmol; placed in top ten in 2017 regional competition. """

input_text_10 = """""" #이것은 값이 없기 때문에 null로 처리해 보자


## 활동입력 값 리스트에 담기
total_actvity = [input_text_1, input_text_2, input_text_3, input_text_4, input_text_5, input_text_6,input_text_7, input_text_9, input_text_10]

total_activity_num = len(total_actvity)
#print'(total_activity_num :' total_activity_num)

result_mjr_act_ = mjr_act_analy(majors, total_actvity)
print("==================================")
print('RESULT major - activity fit :', result_mjr_act_)


result_leadership_fin = leadership_start_here(total_act_lists)
print ("=================================")
print ('RESULT leadership :', result_leadership_fin)


result = dedication_analysis(hrs_per_week, weeks_per_year, period_years_of_activity)
print ("=================================")
print ("RESULT dedecation :", result)
print ("=================================")






