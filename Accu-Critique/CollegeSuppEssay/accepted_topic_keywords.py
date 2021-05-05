
import nltk
import re
import numpy as np
import pandas as pd
import gensim
from nltk.tokenize import sent_tokenize
import multiprocessing
import io
from gensim.models import Phrases
from textblob import TextBlob
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
from nltk.corpus import stopwords
stop = stopwords.words('english')

from topic_extraction import topic_extraction

def get_topic_extraction():
    emotion_ratio_score_cnt = [] # nums of character 'focus on you'

    path = "./data/accepted_data/ps_essay_evaluated.csv"
    data = pd.read_csv(path)
    #Score를 인덱스로 변환하여 데이터 찾아보기
    data.set_index('Score', inplace=True)
    for i in tqdm(data.index):
        if i is not None and i >= 4:
            get_essay = data.loc[i, 'Essay']

            input_ps_essay = get_essay
            re = topic_extraction(str(input_ps_essay))
            #print('re:', re)
            result = re
            emotion_ratio_score_cnt.append(result)



    e_re = [y for x in emotion_ratio_score_cnt for y in x]
    # 중복 카운팅
    emo_total_count = {}
    for i in e_re:
        try: emo_total_count[i] += 1
        except: emo_total_count[i]=1

    return emo_total_count

print('get_topic_extraction:', get_topic_extraction())



