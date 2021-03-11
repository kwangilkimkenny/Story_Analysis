# Virtual Env : office : py37pytorch

import numpy as np
import re
import gensim
import pandas as pd
from nltk.tokenize import sent_tokenize
# 글속에 감정이 얼마나 표현되어 있는지 분석 - origin (Bert pre trained model 활용)
from transformers import BertTokenizer
from model import BertForMultiLabelClassification
from multilabel_pipeline import MultiLabelPipeline
import multiprocessing
import os
from pathlib import Path
import io
from gensim.models import Phrases
from textblob import TextBlob
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
import pickle
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances


def senti_ays_by_prompt(input_text, question_num):
    tokenizer = BertTokenizer.from_pretrained("monologg/bert-base-cased-goemotions-original")
    model = BertForMultiLabelClassification.from_pretrained("monologg/bert-base-cased-goemotions-original")

    goemotions = MultiLabelPipeline(
    model=model,
    tokenizer=tokenizer,
    threshold=0.3
    )
    
    re_text = input_text.split(".")

    #데이터 전처리 
    def cleaning(datas):

        fin_datas = []

        for data in datas:
            # 영문자 이외 문자는 공백으로 변환
            only_english = re.sub('[^a-zA-Z]', ' ', data)

            # 데이터를 리스트에 추가 
            fin_datas.append(only_english)

        return fin_datas

    texts = cleaning(re_text)

    #분석된 감정만 추출
    emo_re = goemotions(texts)

    emo_all = []
    for list_val in range(0, len(emo_re)):
        emo_all.append((emo_re[list_val]['labels']))

    from pandas.core.common import flatten #이중리스틀 FLATTEN하게 변환
    flat_list = list(flatten(emo_all))
        #중립적인 감정을 제외하고, 입력한 문장에서 다양한 감정을 모두 추출하고 어떤 감정이 있는지 계산
        
    unique = []
    for r in flat_list:
        if r == 'neutral':
            pass
        else:
            unique.append(r)

    #빈도수 계산하여 오름차순 정렬

    from collections import Counter

    count = Counter(unique)
    words = dict(count.most_common())
    # 딕셔너리의 keys 추출하여 리스트로
    senti_re = list(words.keys())
    
    #질문별 감성 키워드 분류
    ques_1_senti_keywords_high_score = ['joy', 'pride', 'approval']
    ques_2_senti_keywords_high_score = ['disappointment', 'fear', 'confusion']
    ques_3_senti_keywords_high_score = ['curiosity', 'disapproval', 'realization']
    ques_4_senti_keywords_high_score = ['gratitude', 'surprise', 'admiration']
    ques_5_senti_keywords_high_score = ['realization', 'pride', 'admiration']
    ques_6_senti_keywords_high_score = ['curiosity', 'excitement', 'confusion']
    ques_7_senti_keywords_high_score = []
    
    ques_1_senti_keywords_low_score = ['curiosity', 'amusement', 'admiration', 'excitement', 'realization']
    ques_2_senti_keywords_low_score = ['anger', 'relief', 'embarrassment', 'disapproval', 'nervousness']
    ques_3_senti_keywords_low_score = ['disappointment', 'anger', 'nervousness', 'confusion', 'approval']
    ques_4_senti_keywords_low_score = ['caring', 'joy', 'love', 'optimism']
    ques_5_senti_keywords_low_score = ['approval', 'curiosity', 'gratitude', 'caring', 'joy']
    ques_6_senti_keywords_low_score = ['desire', 'realization', 'amusement', 'joy', 'surprise']
    ques_7_senti_keywords_low_score = []
    
    # 결과 비교하기
    re_comp_high = []
    re_comp_low = []
    if "ques_1" == question_num:
        selected_pmpt_number = "1" # 선택한 prompt 항목
        for i in senti_re:
            if i in ques_1_senti_keywords_high_score:
                re_comp_high.append(i) # add score
            elif i in ques_1_senti_keywords_low_score:
                re_comp_low.append(i) # minus score
    elif "ques_2" == question_num:
        selected_pmpt_number = "2"
        for i in senti_re:
            if i in ques_2_senti_keywords_high_score:
                re_comp_high.append(i) # add score
            elif i in ques_2_senti_keywords_low_score:
                re_comp_low.append(i) # minus score
    elif "ques_3" == question_num:
        selected_pmpt_number = "3"
        for i in senti_re:
            if i in ques_3_senti_keywords_high_score:
                re_comp_high.append(i) # add score
            elif i in ques_3_senti_keywords_low_score:
                re_comp_low.append(i) # minus score
    elif "ques_4" == question_num:
        selected_pmpt_number = "4" # 선택한 prompt 항목
        for i in senti_re:
            if i in ques_4_senti_keywords_high_score:
                re_comp_high.append(i) # add score
            elif i in ques_4_senti_keywords_low_score:
                re_comp_low.append(i) # minus score
    elif "ques_5" == question_num:
        selected_pmpt_number = "5" # 선택한 prompt 항목
        for i in senti_re:
            if i in ques_5_senti_keywords_high_score:
                re_comp_high.append(i) # add score
            elif i in ques_5_senti_keywords_low_score:
                re_comp_low.append(i) # minus score
    elif "ques_6" == question_num:
        selected_pmpt_number = "6" # 선택한 prompt 항목
        for i in senti_re:
            if i in ques_6_senti_keywords_high_score:
                re_comp_high.append(i) # add score
            elif i in ques_6_senti_keywords_low_score:
                re_comp_low.append(i) # minus score
    elif "ques_7"  == question_num:
        selected_pmpt_number = "7" # 선택한 prompt 항목
        re_comp_high.append(senti_re)
    else :
        pass
    
    #결과비교 및 스코어링
    if not re_comp_high and not re_comp_low: # 각 리스트에 값이 없다면, 매칭되는 것이 없기 때문에 점수 없음
        print("can't not get result, try again")
        score_re = "can't not get result"
    elif not re_comp_low and re_comp_low: # high 값이 없고, low값이 있다면
        result_ = re_comp_high + re_comp_low
        score_re = "get low score!"
        print("get low score!")
    elif not re_comp_low and re_comp_high: # high 값이 있고 low값이 없다면
        result_ = re_comp_high + re_comp_low
        score_re = "get high score!"
        print("get high score!")
    else:
        result_ = re_comp_high # 7번 prompt일 경우 분석결과는 입력한 에세이와 일치할 테니까 있는 그대로 점수를 줌(높은점수)
        score_re = "get high score!"
        
    # 에세이에 표현된 감정과 Prompt 비교분석 결과(높은점수에 해당하는 관련 키워드, 낮은점수에 해당하는 관련 키워드)   
    return result_, score_re , selected_pmpt_number



#### run ####
input_text = """Bloomington Normal is almost laughably cliché for a midwestern city. Vast swathes of corn envelop winding roads and the heady smell of BBQ smoke pervades the countryside every summer. Yet, underlying the trite norms of Normal is the prescriptive force of tradition—the expectation to fulfill my role as a female Filipino by playing Debussy in the yearly piano festival and enrolling in multivariable calculus instead of political philosophy.So when I discovered the technical demand of bebop, the triplet groove, and the intricacies of chordal harmony after ten years of grueling classical piano, I was fascinated by the music's novelty. Jazz guitar was not only evocative and creative, but also strangely liberating. I began to explore different pedagogical methods, transcribe solos from the greats, and experiment with various approaches until my own unique sound began to develop. And, although I did not know what would be the 'best' route for me to follow as a musician, the freedom to forge whatever path I felt was right seemed to be exactly what I needed; there were no expectations for me to continue in any particular way—only the way that suited my own desires.While journeying this trail, I found myself at Interlochen Arts Camp the summer before my junior year. Never before had I been immersed in an environment so conducive to musical growth: I was surrounded by people intensely passionate about pursuing all kinds of art with no regard for ideas of what art 'should' be. I knew immediately that this would be a perfect opportunity to cultivate my sound, unbounded by the limits of confining tradition. On the first day of camp, I found that my peer guitarist in big band was another Filipino girl from Illinois. Until that moment, my endeavors in jazz guitar had been a solitary effort; I had no one with whom to collaborate and no one against whom I could compare myself, much less someone from a background mirroring my own. I was eager to play with her, but while I quickly recognized a slew of differences between us—different heights, guitars, and even playing styles—others seemed to have trouble making that distinction during performances. Some even went as far as calling me 'other-Francesca.' Thus, amidst the glittering lakes and musky pine needles of Interlochen, I once again confronted Bloomington's frustrating expectations.After being mistaken for her several times, I could not help but view Francesca as a standard of what the 'female Filipino jazz guitarist' should embody. Her improvisatory language, comping style and even personal qualities loomed above me as something I had to live up to. Nevertheless, as Francesca and I continued to play together, it was not long before we connected through our creative pursuit. In time, I learned to draw inspiration from her instead of feeling pressured to follow whatever precedent I thought she set. I found that I grew because of, rather than in spite of, her presence; I could find solace in our similarities and even a sense of comfort in an unfamiliar environment without being trapped by expectation. Though the pressure to conform was still present—and will likely remain present in my life no matter what genre I'm playing or what pursuits I engage in—I learned to eschew its corrosive influence and enjoy the rewards that it brings. While my encounter with Francesca at first sparked a feeling of pressure to conform in a setting where I never thought I would feel its presence, it also carried the warmth of finding someone with whom I could connect. Like the admittedly trite conditions of my hometown, the resemblances between us provided comfort to me through their familiarity. I ultimately found that I can embrace this warmth while still rejecting the pressure to succumb to expectations, and that, in the careful balance between these elements, I can grow in a way that feels both like discove"""

re__ = senti_ays_by_prompt(input_text, 'ques_6')
print("sentiment analysis result of essay, score, seletec prompt number : ", re__)