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
import spacy

nlp = spacy.load('en_core_web_lg')

def lackigIdealOverboard(group_mean, personal_value): # group_mean: 1000명 평균, personal_value|:개인값
        ideal_mean = group_mean
        one_ps_char_desc = personal_value
        #최대, 최소값 기준으로 구간설정. 구간비율 30% => 0.3으로 설정
        min_ = int(ideal_mean-ideal_mean*0.6)
        # #print('min_', min_)
        max_ = int(ideal_mean+ideal_mean*0.6)
        # #print('max_: ', max_)
        div_ = int(((ideal_mean+ideal_mean*0.6)-(ideal_mean-ideal_mean*0.6))/3)
        # #print('div_:', div_)

        #결과 판단 Lacking, Ideal, Overboard
        cal_abs = abs(ideal_mean - one_ps_char_desc) # 개인 - 단체 값의 절대값계산

        # #print('cal_abs 절대값 :', cal_abs)
        compare7 = (one_ps_char_desc + ideal_mean)/6
        compare6 = (one_ps_char_desc + ideal_mean)/5
        compare5 = (one_ps_char_desc + ideal_mean)/4
        compare4 = (one_ps_char_desc + ideal_mean)/3
        compare3 = (one_ps_char_desc + ideal_mean)/2
        # #print('compare7 :', compare7)
        # #print('compare6 :', compare6)
        # #print('compare5 :', compare5)
        # #print('compare4 :', compare4)
        # #print('compare3 :', compare3)



        if one_ps_char_desc > ideal_mean: # 개인점수가 평균보다 클 경우는 overboard
            if cal_abs > compare3: # 37 개인점수가 개인평균차의 절대값보다 클 경우, 즉 차이가 많이 날경우
                # #print("Overboard: 2")
                result = 2 #overboard
                #score = 1
            elif cal_abs > compare4:
                # #print("Overvoard: 2")
                result = 2
                #score = 2
            elif cal_abs > compare5:
                # #print("Overvoard: 2")
                result = 2
                #score = 3
            elif cal_abs > compare6:
                # #print("Overvoard: 2")
                result = 2
                #score = 4
            else:
                # #print("Ideal: 1")
                result = 1
                #score = 5
        elif one_ps_char_desc < ideal_mean: # 개인점수가 평균보다 작을 경우 lacking
            if cal_abs > compare3: # 37 개인점수가 개인평균차의 절대값보다 클 경우, 즉 차이가 많이 날경우
                # #print("Lacking: 2")
                result = 0
                #score = 1
            elif cal_abs > compare4:
                # #print("Lacking: 2")
                result = 0
                #score = 2
            elif cal_abs > compare5:
                # #print("Lacking: 2")
                result = 0
                #score = 3
            elif cal_abs > compare6:
                # #print("Lacking: 2")
                result = 0
                #score = 4
            else:
                # #print("Ideal: 1")
                result = 1
                #score = 5
                
        else:
            # #print("Ideal: 1")
            result = 1
            #score = 5

        return result


# 시간, 공간, 장소를 알려주는 단어 추출하여 카운트
def find_setting_words(text):
    # Create Doc object
    doc2 = nlp(text)
    
    setting_list = []
    # Identify by label FAC(building etc), GPE(countries, cities..), LOC(locaton), TIME
    fac_r = [ent.text for ent in doc2.ents if ent.label_ == 'FAC']
    setting_list.append(fac_r)
    
    gpe_r = [ent.text for ent in doc2.ents if ent.label_ == 'GPE']
    setting_list.append(gpe_r)
    
    loc_r = [ent.text for ent in doc2.ents if ent.label_ == 'LOC']
    setting_list.append(loc_r)
    
    time_r = [ent.text for ent in doc2.ents if ent.label_ == 'TIME']
    setting_list.append(time_r)
    
    #추출된 항목들
    all_setting_words = sum(setting_list, [])
    
    #총 수
    get_setting_list = len(all_setting_words)
    
    # Return all setting words
    return get_setting_list, all_setting_words



##### 이 코드로 실행하는 거임 #####
def get_setting_analysis(input_text):
    result_setting = list(find_setting_words(input_text))
    # ILO mean ideal lacking overboard
    # 1000명의 평균 값을 변경하려면 "1000명의 평균값으로 int "을 적절한 숫자(int)로 바꿔야함!!
    # re_setting_ILO = lackigIdealOverboard("1000명의 평균값으로 int ", "입력한 계산 결과값 int")
    # 0:Not a big factor, 1:Somewhat important, 2:Surroundings matter a lot
    re_setting_ILO = lackigIdealOverboard(4, result_setting[0])
    return re_setting_ILO, result_setting




##### test #####
input_text = """Bloomington Normal is almost laughably cliché for a midwestern city. Vast swathes of corn envelop winding roads and the heady smell of BBQ smoke pervades the countryside every summer. Yet, underlying the trite norms of Normal is the prescriptive force of tradition—the expectation to fulfill my role as a female Filipino by playing Debussy in the yearly piano festival and enrolling in multivariable calculus instead of political philosophy.So when I discovered the technical demand of bebop, the triplet groove, and the intricacies of chordal harmony after ten years of grueling classical piano, I was fascinated by the music's novelty. Jazz guitar was not only evocative and creative, but also strangely liberating. I began to explore different pedagogical methods, transcribe solos from the greats, and experiment with various approaches until my own unique sound began to develop. And, although I did not know what would be the 'best' route for me to follow as a musician, the freedom to forge whatever path I felt was right seemed to be exactly what I needed; there were no expectations for me to continue in any particular way—only the way that suited my own desires.While journeying this trail, I found myself at Interlochen Arts Camp the summer before my junior year. Never before had I been immersed in an environment so conducive to musical growth: I was surrounded by people intensely passionate about pursuing all kinds of art with no regard for ideas of what art 'should' be. I knew immediately that this would be a perfect opportunity to cultivate my sound, unbounded by the limits of confining tradition. On the first day of camp, I found that my peer guitarist in big band was another Filipino girl from Illinois. Until that moment, my endeavors in jazz guitar had been a solitary effort; I had no one with whom to collaborate and no one against whom I could compare myself, much less someone from a background mirroring my own. I was eager to play with her, but while I quickly recognized a slew of differences between us—different heights, guitars, and even playing styles—others seemed to have trouble making that distinction during performances. Some even went as far as calling me 'other-Francesca.' Thus, amidst the glittering lakes and musky pine needles of Interlochen, I once again confronted Bloomington's frustrating expectations.After being mistaken for her several times, I could not help but view Francesca as a standard of what the 'female Filipino jazz guitarist' should embody. Her improvisatory language, comping style and even personal qualities loomed above me as something I had to live up to. Nevertheless, as Francesca and I continued to play together, it was not long before we connected through our creative pursuit. In time, I learned to draw inspiration from her instead of feeling pressured to follow whatever precedent I thought she set. I found that I grew because of, rather than in spite of, her presence; I could find solace in our similarities and even a sense of comfort in an unfamiliar environment without being trapped by expectation. Though the pressure to conform was still present—and will likely remain present in my life no matter what genre I'm playing or what pursuits I engage in—I learned to eschew its corrosive influence and enjoy the rewards that it brings. While my encounter with Francesca at first sparked a feeling of pressure to conform in a setting where I never thought I would feel its presence, it also carried the warmth of finding someone with whom I could connect. Like the admittedly trite conditions of my hometown, the resemblances between us provided comfort to me through their familiarity. I ultimately found that I can embrace this warmth while still rejecting the pressure to succumb to expectations, and that, in the careful balance between these elements, I can grow in a way that feels both like discove"""

result__ = get_setting_analysis(input_text)

print ("ILO Value, Setting Num Words, Words: ", result__)

# 결과

# ILO Value, Setting Num Words, Words:  (1, [3, ['Illinois', 'Interlochen', 'Bloomington']])
# 결과 설명
# 0:Not a big factor, 1:Somewhat important, 2:Surroundings matter a lot  >>> 1
# Setting Num Words >>> 3
# setting words list >>> ['Illinois', 'Interlochen', 'Bloomington']
