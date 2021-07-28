# update 2021_07_28

# Virtual Env : office : py37pytorch

import numpy as np
import re
import gensim
import pandas as pd
from nltk.tokenize import sent_tokenize
# 글속에 감정이 얼마나 표현되어 있는지 분석 - origin (Bert pre trained model 활용)
from transformers import BertTokenizer
# from accu_ps.data.model import BertForMultiLabelClassification
# from accu_ps.data.multilabel_pipeline import MultiLabelPipeline

# 중복체크
import collections

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

##########  key_value_print
##########  key: value 형식으로 나뉨 
def key_value_print (dictonrytemp) : 
    print("#"*100)
    for key in dictonrytemp.keys() : 
        print(key,": ",dictonrytemp[key])
        print()
    print("#"*100)



def senti_ays_by_prompt(input_text, question_num):
    tokenizer = BertTokenizer.from_pretrained("monologg/bert-base-cased-goemotions-original")
    model = BertForMultiLabelClassification.from_pretrained("monologg/bert-base-cased-goemotions-original")

    goemotions = MultiLabelPipeline(
    model=model,
    tokenizer=tokenizer,
    threshold=0.3
    )
    
    re_text = input_text.split(".")

    result_ = []
    score_re = ""
    

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

    # print("unique_추출한 감성:", unique)
    counter_check = collections.Counter(unique)
    cnt_numb_re = counter_check.most_common()
    print("cnt_numb_re(빈도수 계산):", cnt_numb_re)

    Joy_data = ['joy', 'amusement', 'gratitude', 'optimism', 'pride', 'approval', 'grief', 'love','admiration']
    Sad_data = ['sadness', 'disappointment', 'remorse']
    Disturbed_data = ['confusion', 'anger', 'annoyance', 'disapproval', 'disgust', 'embarrassment', 'nervousness', 'fear']
    Suspenseful_data = ['surprise', 'excitement','desire','curiosity','']

    cal_sent_sum = []
    for each_itm in unique:
        input_sent_value = each_itm
        if input_sent_value in Joy_data:
            conv_input_sent_data = 'joyful'
            cal_sent_sum.append(conv_input_sent_data)
        elif input_sent_value in Sad_data:
            conv_input_sent_data = 'sad'
            cal_sent_sum.append(conv_input_sent_data)
        elif input_sent_value in Disturbed_data:
            conv_input_sent_data = 'disturbed'
            cal_sent_sum.append(conv_input_sent_data)
        elif input_sent_value in Suspenseful_data:
            conv_input_sent_data = 'suspenseful'
            cal_sent_sum.append(conv_input_sent_data)
        else: # input_sent_value in Calm_data:
            conv_input_sent_data = 'calm'
            cal_sent_sum.append(conv_input_sent_data)

    cnt_snt_re = collections.Counter(cal_sent_sum)
    cnt_mst_sent_re = cnt_snt_re.most_common()
    print("sentiments ratio in essay:", cnt_mst_sent_re)
    # [('joy', 9), ('disturbed', 7), ('calm', 6), ('sad', 3), ('suspenseful', 2)]

    emo_sum = 0
    for t in cnt_mst_sent_re:
        get_nums =t[1]
        print('get_nums:',get_nums)
        emo_sum += get_nums

    print('emo_sum:', emo_sum)

    #에세이의 감정 포함 비율 계산
    emo_ration_dic = {}
    for emo_itm  in cnt_mst_sent_re:
        if emo_itm[0] == 'joyful':
            get_joyful_ratio = round(emo_itm[1]/emo_sum * 100, 1)
            emo_ration_dic.setdefault('joyful', get_joyful_ratio)
        elif emo_itm[0] == 'sad':
            get_sad_ratio = round(emo_itm[1]/emo_sum * 100,1)
            emo_ration_dic.setdefault('sad', get_sad_ratio)
        elif emo_itm[0] == 'disturbed':
            get_disturbed_ratio = round(emo_itm[1]/emo_sum * 100, 1)
            emo_ration_dic.setdefault('disturbed', get_disturbed_ratio)
        elif emo_itm[0] == 'suspenseful':
            get_suspenseful_ratio = round(emo_itm[1]/emo_sum * 100, 1)
            emo_ration_dic.setdefault('suspenseful', get_suspenseful_ratio)
        else: # emo_itm[0] == 'calm':
            get_calm_ratio = round(emo_itm[1]/emo_sum * 100,1)
            emo_ration_dic.setdefault('calm', get_calm_ratio)

    print("--------------------------------")
    print("emo_ration_dic:", emo_ration_dic)
    # emo_ration_dic: {'joy': 33.3, 'disturbed': 25.9, 'calm': 22.2, 'sad': 11.1, 'suspenseful': 7.4}
    print("--------------------------------")

    # prompt에 해당하는 값과 emo_ration_dic 값과 비교(웹에 출력) 이격률을 표시하기 위해서 표준편차를 구하기(KJ님 작성 값)
    pmt_1_emo_ratio = {'joyful': 35, 'sad': 26, 'disturbed': 17, 'suspenseful': 17, 'calm': 5}
    pmt_2_emo_ratio = {'joyful': 8, 'sad': 30, 'disturbed': 31, 'suspenseful': 20, 'calm': 11}
    pmt_3_emo_ratio = {'joyful': 16, 'sad': 7, 'disturbed': 30, 'suspenseful': 26, 'calm': 10}
    pmt_4_emo_ratio = {'joyful': 29, 'sad': 15, 'disturbed': 18, 'suspenseful': 22, 'calm': 16}
    pmt_5_emo_ratio = {'joyful': 32, 'sad': 13, 'disturbed': 9, 'suspenseful': 35, 'calm': 11}
    pmt_6_emo_ratio = {'joyful': 54, 'sad': 8, 'disturbed': 6, 'suspenseful': 28, 'calm': 4}


    def std_pmt_essay(pmt_emo_comp_ratio_dic, pmt_emo_ps_ratio_dic):
        std_re = []
        for itm__ in pmt_emo_comp_ratio_dic:
            get_value = pmt_emo_comp_ratio_dic.get(itm__) # 키로 값 가져오기
            print('get_value:', get_value)
            #에세이에서 추출한 값으로 이격률 구하기(표준편차)
            ext_essay_value = pmt_emo_ps_ratio_dic.get(itm__)
            print("essay 추출 값:", ext_essay_value)
            abs_re = abs(round(float(get_value) - float(ext_essay_value), 1)) # 두 값을 빼서 차이를 절대값을 변환
            std_re.append(abs_re)
            print("--------------------------------")
            
        print('std_re:', std_re)
        #이제 이것들의 표준편차를 구해보자
        #질문에 해당하는 감성값들과 에세이에서 추출한 감성값을 비교하여 이격률을 표준편차로 계산
        print('질문에 해당하는 감성값들과 에세이에서 추출한 감성값을 비교하여 이격률을 표준편차로 계산:', round(np.std(std_re),1))

        result_std = round(np.std(std_re),1)
        return result_std


    # 표준편차 계산하기
    if question_num == 'ques_one':

        pmt_emo_comp_ratio = pmt_1_emo_ratio # 질문에 해당하는 표준 비교값 ----> WEB에 출력가능
        pmt_emo_ps_ratio = emo_ration_dic # 에세이에서 추추출한 값 -----------> WEB에 출력가능

        std_result = std_pmt_essay(pmt_emo_comp_ratio, pmt_emo_ps_ratio)
        print("--------------------------------")
        print('std_result:',std_result ) # 표준편차가 예를 들어 5 이상이면 이격이 크다, 작으면 유사하다로 판단해도 됨
        print("--------------------------------")

    elif question_num == 'ques_two':
        
        pmt_emo_comp_ratio = pmt_2_emo_ratio # 질문에 해당하는 표준 비교값 ----> WEB에 출력가능
        pmt_emo_ps_ratio = emo_ration_dic # 에세이에서 추추출한 값 -----------> WEB에 출력가능

        std_result = std_pmt_essay(pmt_emo_comp_ratio, pmt_emo_ps_ratio)
        print("--------------------------------")
        print('std_result:',std_result ) # 표준편차가 예를 들어 5 이상이면 이격이 크다, 작으면 유사하다로 판단해도 됨
        print("--------------------------------")

    elif question_num == 'ques_three':
        
        pmt_emo_comp_ratio = pmt_3_emo_ratio # 질문에 해당하는 표준 비교값 ----> WEB에 출력가능
        pmt_emo_ps_ratio = emo_ration_dic # 에세이에서 추추출한 값 -----------> WEB에 출력가능

        std_result = std_pmt_essay(pmt_emo_comp_ratio, pmt_emo_ps_ratio)
        print("--------------------------------")
        print('std_result:',std_result ) # 표준편차가 예를 들어 5 이상이면 이격이 크다, 작으면 유사하다로 판단해도 됨
        print("--------------------------------")

    elif question_num == 'ques_four':
        
        pmt_emo_comp_ratio = pmt_4_emo_ratio # 질문에 해당하는 표준 비교값 ----> WEB에 출력가능
        pmt_emo_ps_ratio = emo_ration_dic # 에세이에서 추추출한 값 -----------> WEB에 출력가능

        std_result = std_pmt_essay(pmt_emo_comp_ratio, pmt_emo_ps_ratio)
        print("--------------------------------")
        print('std_result:',std_result ) # 표준편차가 예를 들어 5 이상이면 이격이 크다, 작으면 유사하다로 판단해도 됨
        print("--------------------------------")
        
    elif question_num == 'ques_five':
        
        pmt_emo_comp_ratio = pmt_5_emo_ratio # 질문에 해당하는 표준 비교값 ----> WEB에 출력가능
        pmt_emo_ps_ratio = emo_ration_dic # 에세이에서 추추출한 값 -----------> WEB에 출력가능

        std_result = std_pmt_essay(pmt_emo_comp_ratio, pmt_emo_ps_ratio)
        print("--------------------------------")
        print('std_result:',std_result ) # 표준편차가 예를 들어 5 이상이면 이격이 크다, 작으면 유사하다로 판단해도 됨
        print("--------------------------------")
        
    else: # question_num == 'ques_six':
        
        pmt_emo_comp_ratio = pmt_6_emo_ratio # 질문에 해당하는 표준 비교값 ----> WEB에 출력가능
        pmt_emo_ps_ratio = emo_ration_dic # 에세이에서 추추출한 값 -----------> WEB에 출력가능

        std_result = std_pmt_essay(pmt_emo_comp_ratio, pmt_emo_ps_ratio)
        print("--------------------------------")
        print('std_result:',std_result ) # 표준편차가 예를 들어 5 이상이면 이격이 크다, 작으면 유사하다로 판단해도 됨
        print("--------------------------------")
        

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
    selected_pmpt_number = ""
    oriented_sentiment_comment = []
    if "ques_one" == question_num:
        selected_pmpt_number = "prompt 1" # 선택한 prompt 항목
        oriented_sentiment_list = "identity, diversity, inclinations, passion, culture, unique qualities, life story, values, experience"
        
        
        
        for i in senti_re:
            if i in ques_1_senti_keywords_high_score:
                re_comp_high.append(i) # add score
            elif i in ques_1_senti_keywords_low_score:
                re_comp_low.append(i) # minus score
    elif "ques_two" == question_num:
        selected_pmpt_number = "prompt 2"
        oriented_sentiment_list = "obstacle, hardship, challenge, failure, lessons, values, triumph, rebound, courage, initiative, attitude, improvement, development, and more."

        for i in senti_re:
            if i in ques_2_senti_keywords_high_score:
                re_comp_high.append(i) # add score
            elif i in ques_2_senti_keywords_low_score:
                re_comp_low.append(i) # minus score
    elif "ques_three" == question_num:
        selected_pmpt_number = "prompt 3"
        oriented_sentiment_list = "critical thinking, courage, challenging spirit, self-reflection, intellect, action, change, respect, realization, improvement, curiosity, leadership, fight, and more."

        for i in senti_re:
            if i in ques_3_senti_keywords_high_score:
                re_comp_high.append(i) # add score
            elif i in ques_3_senti_keywords_low_score:
                re_comp_low.append(i) # minus score
    elif "ques_four" == question_num:
        selected_pmpt_number = "prompt 4" # 선택한 prompt 항목
        oriented_sentiment_list = "gratitude, altruism, hero, heroine, philanthropy, caring, preconception, realization, maturity, sacrifice, reward, common good, hardship, virtue, and more."

        for i in senti_re:
            if i in ques_4_senti_keywords_high_score:
                re_comp_high.append(i) # add score
            elif i in ques_4_senti_keywords_low_score:
                re_comp_low.append(i) # minus score
    
    elif "ques_five" == question_num:
        selected_pmpt_number = "prompt 5" # 선택한 prompt 항목
        oriented_sentiment_list = "incident, initiative, accomplishment, maturity, perspective change, realization, life story, personal history, triumph, community, team, people, and more."

        for i in senti_re:
            if i in ques_5_senti_keywords_high_score:
                re_comp_high.append(i) # add score
            elif i in ques_5_senti_keywords_low_score:
                re_comp_low.append(i) # minus score
    
    elif "ques_six" == question_num:
        selected_pmpt_number = "prompt 6" # 선택한 prompt 항목
        oriented_sentiment_list = "curiosity, intellectual, social science, STEM, humanities, ideology, question, research, think, logic, reason, depth, academic, intriguing, goal, plan, and more."
        

        for i in senti_re:
            if i in ques_6_senti_keywords_high_score:
                re_comp_high.append(i) # add score
            elif i in ques_6_senti_keywords_low_score:
                re_comp_low.append(i) # minus score
    elif "ques_seven"  == question_num:
        selected_pmpt_number = "prompt 7" # 선택한 prompt 항목
        oriented_sentiment_list = "curiosity, intellectual, social science, STEM, humanities, ideology, question, research, think, logic, reason, depth, academic, intriguing, goal, plan, and more."
        

        re_comp_high.append(senti_re)
    else :
        pass
    
    #결과비교 및 스코어링
    if not re_comp_high and not re_comp_low: # 각 리스트에 값이 없다면, 매칭되는 것이 없기 때문에 점수 없음
        print("can't not get result, try again")
        score_re = "can't not get result"
    elif not re_comp_high and re_comp_low: # high 값이 없고, low값이 있다면
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

    print("result_:",result_)
 
        
    # 에세이에 표현된 감정과 Prompt 비교분석 결과(높은점수에 해당하는 관련 키워드, 낮은점수에 해당하는 관련 키워드)   

    data = {

        "oriented_sentiment_list" : oriented_sentiment_list ,
        
        "detected_result" : result_[0],
        "score_re" : score_re,
        "selected_pmpt_number" : selected_pmpt_number,

        # 각 질문에 해당하는 감성정보 표준값 비교값 ----> WEB에 출력가능
        "pmt_emo_comp_ratio" : pmt_emo_comp_ratio,

        # 에세이에서 추추출한 대표감성정보 분석한 값 -----------> WEB에 출력가능
        "pmt_emo_ps_ratio" : pmt_emo_ps_ratio,

        # prompt별 기준 감성값과 입력한 에세이의 감성정보값을 비교하여 표준편차(이격률) 계산값으로 크면 불일치, 작으면 일치한다는 의미
        "std_result" : std_result

    }
    
    
    return data



#### run ####
# Personal Essay Sample #1 (Prompt #1이나 Prompt #4에 맞는 아름다운 에세이)

input_text = """ My hand lingered on the cold metal doorknob. I closed my eyes as the Vancouver breeze ran its chilling fingers through my hair. The man I was about to meet was infamous for demanding perfection. But the beguiling music that faintly fluttered past the unlatched window’s curtain drew me forward, inviting me to cross the threshold. Stepping into the apartment, under the watchful gaze of an emerald-eyed cat portrait, I entered the sweeping B Major scale. Led by my intrinsic attraction towards music, coupled with the textured layers erupting the instant my fingers grazed the ivory keys, driving the hammers to shoot vibrations up in the air all around me, I soon fell in love with this new extension of my body and mind. My mom began to notice my aptitude for piano when I began returning home with trophies in my arms. These precious experiences fueled my conviction as a rising musician, but despite my confidence, I felt like something was missing.  Back in the drafty apartment, I smiled nervously and walked towards the piano from which the music emanated. Ian Parker, my new piano teacher, eyes-closed and dressed in black glided his hands effortlessly across the keys. I stood beside a leather chair, waiting as he finished the phrase. He stood up. I sat down.  Chopin Black Key Etude — a piece I knew so well I could play it eyes-closed. I took a breath and positioned my right hand in a G-flat 2nd inversion.  Just one measure in, I was stopped. “Start again.”Taken by surprise, I spun left. His eyes were on the score, not me. I started again. Past the first measure, first phrase, then stopped again. What is going on? 	“Are you listening? I nodded. Of course I am. “But are you really listening?” As we slowly dissected each measure, I felt my confidence slip away. The piece was being chipped into fragments. Unlike my previous teachers, who listened to a full performance before giving critical feedback, Ian stopped me every five seconds. One hour later, we only got through half a page. Each consecutive week, the same thing happened. I struggled to meet his expectations. “I’m not here to teach you just how to play. I’m here to teach you how to listen.” I realized what Ian meant — listening involves taking what we hear and asking: is this the sound I want? What story am I telling through my interpretation? Absorbed in the music, I allowed my instincts and muscle memory to take over, flying past the broken tritones or neapolitan chords. But even if I was playing the right notes, it didn’t matter. Becoming immersed in the cascading arpeggio waterfalls, thundering basses, and fairydust trills was actually the easy part, which brought me joy and fueled my love for music in the first place. However, music is not just about me. True artists perform for their audience, and to bring them the same joy, to turn playing into magic-making, they must listen as the audience. The lesson Ian taught me echoes beyond practice rooms and concert halls. I’ve learned to listen as I explore the hidden dialogue between voices, to pauses and silence, equally as powerful as words. Listening is performing as a soloist backed up by an orchestra. Listening is calmly responding during heated debates and being the last to speak in a SPS Harkness discussion. It’s even bouncing jokes around the dining table with family. I’ve grown to envision how my voice will impact the stories of those listening to me. To this day, my lessons with Ian continue to be tough, consisting of 80% discussion and 20% playing. When we were both so immersed in the music that I managed to get to the end of the piece before he looked up to say, “Bravo.” Now, even when I practice piano alone, I repeat my refrain: Are you listening? """

ques_type = "ques_one"

data = senti_ays_by_prompt(input_text,ques_type)
key_value_print(data)

####################################################################################################
# oriented_sentiment_list :  identity, diversity, inclinations, passion, culture, unique qualities, life story, values, experience

# detected_result :  approval

# score_re :  get high score!

# selected_pmpt_number :  prompt 1

# pmt_emo_comp_ratio :  {'joyful': 35, 'sad': 26, 'disturbed': 17, 'suspenseful': 17, 'calm': 5}

# pmt_emo_ps_ratio :  {'joyful': 33.3, 'disturbed': 25.9, 'calm': 22.2, 'sad': 11.1, 'suspenseful': 7.4}

# std_result :  5.4
####################################################################################################
