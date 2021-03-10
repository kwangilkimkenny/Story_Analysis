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

# 이름으로 인물 수 계산
def find_named_persons(text):
    # Create Doc object
    doc2 = nlp(text)
    # Identify the persons
    persons = [ent.text for ent in doc2.ents if ent.label_ == 'PERSON']
    #총 인물 수
    get_person = len(persons)
    # Return persons
    return get_person


def characters(input_text):
    #소문자로 변환
    input_lower_text = input_text.lower()
    about_doc = nlp(input_lower_text)
    token_list = {}
    for token in about_doc:
        #print (token, token.idx)
        token_list.setdefault(token, token.idx)
    
    li_doc = list(token_list.keys())

    #캐릭터 표현하는 단어들을 리스트에 넣어서 필터로 만들고
    i_character_list = ['i', 'my', 'me', 'mine']
    #하나씩 꺼내서 유사한 단어를 찾아내서 새로운 리스트에 담아서 출력,
    ext_i_characters = []
    for i_itm in i_character_list:
        for k_ in li_doc:
            if i_itm == str(k_):
                ext_i_characters.append(i_itm)
    #I 관련 캐릭터 표현하는 단어들의 총 개수        
    get_i = len(ext_i_characters)

    #캐릭터 표현하는 단어들을 리스트에 넣어서 필터로 만들고
    you_character_list = ['you', 'your', 'they','them',
                    'yours', 'he','him','his' 'she','her','it','someone','their', 'myself', 'aunt',
                    'brother','cousin','daughter','father','grandchild','granddaughter','granddson','grandfather',
                    'grandmother', 'person','great-grandchild','husband','ex-husband','son-in-law', 'daughter-in-law','mother',
                    'niece','nephew','parents','sister','son','stepfather','stepmother','stepdaughter', 'stepson',
                    'twin','uncle','widow','widower','wife','ex-wife']
    #하나씩 꺼내서 유사한 단어를 찾아내서 새로운 리스트에 담아서 출력,
    ext_you_characters = []
    for i_itm in you_character_list:
        for k_ in li_doc:
            if i_itm == str(k_):
                ext_you_characters.append(i_itm)
    #I 관련 캐릭터 표현하는 단어들의 총 개수        
    get_others = len(ext_you_characters)

    return get_i, get_others



def focusOnCharacters(input_text):

    person_num = find_named_persons(input_text)
    charater_num = list(characters(input_text))

    sum_character_num = person_num + charater_num[0] + charater_num[1]
    ratio_i = round((charater_num[0] / sum_character_num),2) * 100

    if ratio_i >= 70: # i 가 70% 이상
        print("Mostly Me")
        result = 1
    elif 40 <= ratio_i < 70: # i가 40~ 70% 
        print("Me & some others")
        result = 2
    else:
        print("Others characters") # i가 40% 이하
        result = 3

    return result #1~3의 결과가 나옴







    ###### Run ######
# input_text = """Bloomington Normal is almost laughably cliché for a midwestern city. Vast swathes of corn envelop winding roads and the heady smell of BBQ smoke pervades the countryside every summer. Yet, underlying the trite norms of Normal is the prescriptive force of tradition—the expectation to fulfill my role as a female Filipino by playing Debussy in the yearly piano festival and enrolling in multivariable calculus instead of political philosophy.So when I discovered the technical demand of bebop, the triplet groove, and the intricacies of chordal harmony after ten years of grueling classical piano, I was fascinated by the music's novelty. Jazz guitar was not only evocative and creative, but also strangely liberating. I began to explore different pedagogical methods, transcribe solos from the greats, and experiment with various approaches until my own unique sound began to develop. And, although I did not know what would be the 'best' route for me to follow as a musician, the freedom to forge whatever path I felt was right seemed to be exactly what I needed; there were no expectations for me to continue in any particular way—only the way that suited my own desires.While journeying this trail, I found myself at Interlochen Arts Camp the summer before my junior year. Never before had I been immersed in an environment so conducive to musical growth: I was surrounded by people intensely passionate about pursuing all kinds of art with no regard for ideas of what art 'should' be. I knew immediately that this would be a perfect opportunity to cultivate my sound, unbounded by the limits of confining tradition. On the first day of camp, I found that my peer guitarist in big band was another Filipino girl from Illinois. Until that moment, my endeavors in jazz guitar had been a solitary effort; I had no one with whom to collaborate and no one against whom I could compare myself, much less someone from a background mirroring my own. I was eager to play with her, but while I quickly recognized a slew of differences between us—different heights, guitars, and even playing styles—others seemed to have trouble making that distinction during performances. Some even went as far as calling me 'other-Francesca.' Thus, amidst the glittering lakes and musky pine needles of Interlochen, I once again confronted Bloomington's frustrating expectations.After being mistaken for her several times, I could not help but view Francesca as a standard of what the 'female Filipino jazz guitarist' should embody. Her improvisatory language, comping style and even personal qualities loomed above me as something I had to live up to. Nevertheless, as Francesca and I continued to play together, it was not long before we connected through our creative pursuit. In time, I learned to draw inspiration from her instead of feeling pressured to follow whatever precedent I thought she set. I found that I grew because of, rather than in spite of, her presence; I could find solace in our similarities and even a sense of comfort in an unfamiliar environment without being trapped by expectation. Though the pressure to conform was still present—and will likely remain present in my life no matter what genre I'm playing or what pursuits I engage in—I learned to eschew its corrosive influence and enjoy the rewards that it brings. While my encounter with Francesca at first sparked a feeling of pressure to conform in a setting where I never thought I would feel its presence, it also carried the warmth of finding someone with whom I could connect. Like the admittedly trite conditions of my hometown, the resemblances between us provided comfort to me through their familiarity. I ultimately found that I can embrace this warmth while still rejecting the pressure to succumb to expectations, and that, in the careful balance between these elements, I can grow in a way that feels both like discove"""

# result = focusOnCharacters(input_text)

# print(result)

# 결과분석

# Mostly Me" : 1
# Me & some others : 2
# Others characters : 3