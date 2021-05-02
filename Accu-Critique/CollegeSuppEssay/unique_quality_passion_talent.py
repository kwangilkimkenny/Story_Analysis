# 문장입력
# 토픽 추출
# topic uniquness 함수 실행 
# collegeSuppy에서 prompt oriented sentiments 분석실행
# enguagement 계산 시행 - google 문서 참고할 것

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
from nltk.corpus import stopwords
stop = stopwords.words('english')

from topic_extraction import topic_extraction
from topic_uniqueness import google_search_result
from prompt_oriented_sentiments import prompt_ori_sentiment
from intellectualEngagement import intellectualEnguagement
from topic_knowledge import google_search_result_tp_knowledge


def unique_quality_passion_talent(select_pmt_type, essay_input):
    # 에세이에서 추출한 모든 토픽
    topic_ext_re = topic_extraction(essay_input) # like', 'usually', 'unique', 'thought', 'led' ...
    #print('topic_ext_re:', topic_ext_re[:10])
    google_search__all_re = []
    for i in topic_ext_re[:10]:
        google_search_re = google_search_result(i) # 토픽중 10개만 검색해서 결과 추출. 일단 1개만 추출(개발 끝난 후 10개로 변경)
        searched_mean_score = google_search_re[1]
        google_search__all_re.append(searched_mean_score)
    pmt_ori_sent_re = prompt_ori_sentiment(select_pmt_type, essay_input)

    google_search__all_result = sum(google_search__all_re) / len(google_search__all_re)

    # Eagagement 30%
    Engagement_result = intellectualEnguagement(essay_input)

    # Topic knowledge 10%
    topic_knowledge_list = []
    for k in topic_ext_re[:10]:
        tp_knwg = google_search_result_tp_knowledge(k)
        topic_knowledge_list.append(tp_knwg)


    # Topic knowledge 점수 계산
    result_topic_knowledge =[]
    for ets_itm in topic_ext_re[:10]: # 추출한 토픽 10개만 분석하기(시간이 많이 걸림)
        result_of_srch = google_search_result_tp_knowledge(ets_itm) # 각 토픽별로 관련 웹검색하여 단어 추출
        result_topic_knowledge.append(result_of_srch) # 추출 리스트 저장
    print('result_topic_knowledge:', result_topic_knowledge)

    # Topic knowledge결과 비교하기 : 전체 추출 리스트와 추출한 토픽들의 포함 비율 계산하기
    match_topic_words = 0
    for ext_itttm in topic_ext_re:
        if ext_itttm in result_topic_knowledge: # 토픽이 리스트안에 있다면! 카운트한다.
            match_topic_words += 1
    print('match_topic_words:', match_topic_words)

    if match_topic_words != 0: # 매칭되는 토픽이 있다면, 검색을 통해 수집된 정보에서 매칭 토픽의 포함 비율을 계산해본다. 예를 들어 일정 기준 이상이면 strong.. 등으로 표현하면 된다.
        get_topic_knowledge_ratio = round(match_topic_words / len(result_topic_knowledge) * 100, 2)
        print('get_topic_knowledge_ratio:', get_topic_knowledge_ratio)
        if get_topic_knowledge_ratio >= 10: #10% 이상이면 ================> 중요! 이 값은 결과값을 보면서 보정해야 함(현재는 임의값 적용)
            fin_topic_knowledge_score = 'Supurb'
        elif get_topic_knowledge_ratio >= 5 and get_topic_knowledge_ratio < 10: #================> 중요! 이 값은 결과값을 보면서 보정해야 함(현재는 임의값 적용)
            fin_topic_knowledge_score = 'Strong'
        elif get_topic_knowledge_ratio >= 3 and get_topic_knowledge_ratio < 5: #================> 중요! 이 값은 결과값을 보면서 보정해야 함(현재는 임의값 적용)
            fin_topic_knowledge_score = 'Good' 
        else:
            fin_topic_knowledge_score = 'Mediocre'
    else: # match_topic_words = 0 매칭하는 값이 0이면=================>>>>>>>>>>>> !!! 결과값 재획인 해야 함!!!
        fin_topic_knowledge_score = 'Lacking'
        get_topic_knowledge_ratio = 0

    #supurb ~ lacking 을 숫자로 된 점수로 변환
    def text_re_to_score(input):
        if input == 'Supurb':
            tp_knowledge_re = 90
        elif input == 'Strong':
            tp_knowledge_re = 75
        elif input == 'Good':
            tp_knowledge_re = 65
        elif input == 'Mediocre':
            tp_knowledge_re = 40
        else: #input == 'Lacking'
            tp_knowledge_re = 10
        return tp_knowledge_re
    # supurb ~ lacking 을 숫자로 된 점수로 변환
    tp_kwlg_result = text_re_to_score(fin_topic_knowledge_score)
    # print('tp_kwlg_result:', tp_kwlg_result)


    def calculate_score(input_scofre):
        if input_scofre >= 80:
            result_pmt_ori_sentiments = 'Supurb'
        elif input_scofre >= 60 and input_scofre < 80:
            result_pmt_ori_sentiments = 'Strong'
        elif input_scofre >= 40 and input_scofre < 60:
            result_pmt_ori_sentiments = 'Good'
        elif input_scofre >= 20 and input_scofre < 40:
            result_pmt_ori_sentiments = 'Mediocre'
        else: #input_scofre < 20
            result_pmt_ori_sentiments = 'Lacking'
        return result_pmt_ori_sentiments


    overall_result = round(google_search__all_result * 0.3 + pmt_ori_sent_re[0] * 0.3 + Engagement_result[0] * 0.3 + tp_kwlg_result * 0.1, 1)

    overall_result_fin = calculate_score(overall_result)

    pmt_ori_sent_re_fin = calculate_score(pmt_ori_sent_re[0])
    google_search__all_re_fin = calculate_score(google_search__all_result)
    Engagement_result_fin = calculate_score(Engagement_result[0])
    tp_kwlg_result_fin = calculate_score(tp_kwlg_result)

    # 문장생성
    fixed_top_comment = """To define a unique quality, passion, or talent, you should find something about you that is not considered cliché. It should be a quality that excites you and makes you proud, while demonstrating your dedication and knowledge towards such passion would be appropriate."""

    def gen_comment(input_score, type):
        if input_score == 'Supurb' or input_score == 'Strong':
            if type == 'Topic_Uniqueness':
                comment_achieve= """The topics in your essay seem very unique and readers may find them intriguing."""
            elif type == 'pmt_ori_sentiment':
                comment_achieve = """Also, you seem to demonstrate strong sentiments that are correlated with the qualities sought by this prompt."""
            elif type == 'Engagement':
                comment_achieve = """You seem to maintain a high level of engagement towards such quality, passion, or talent. """
            else:
                pass
        elif input_score == 'Good':
            if type == 'Topic_Uniqueness':
                comment_achieve= """The topics in your essay seem somewhat unique and you may consider finding more interesting topics."""
            elif type == 'pmt_ori_sentiment':
                comment_achieve = """Also, you seem to demonstrate compatible sentiments that are correlated with the qualities sought by this prompt."""
            elif type == 'Engagement':
                comment_achieve = """You seem to maintain a satisfactory level of engagement towards such quality, passion, or talent. """
            else:
                pass
        else: #input score == 'Mediocre' or input_score == 'Weak'
            if type == 'Topic_Uniqueness':
                comment_achieve= """The topics in your essay seem somewhat common. Hence, you may consider finding less generic topics while adding more detail."""
            elif type == 'pmt_ori_sentiment':
                comment_achieve = """Also, you seem to lacking the sentiments that are correlated with the qualities sought by this prompt."""
            elif type == 'Engagement':
                comment_achieve = """You seem to maintain a somewhat low level of engagement towards such quality, passion, or talent. """
            else:
                pass
        return comment_achieve

    comment_1 = gen_comment(google_search__all_re_fin, 'Topic_Uniqueness')
    comment_2 = gen_comment(pmt_ori_sent_re_fin, 'pmt_ori_sentiment')
    comment_3 = gen_comment(Engagement_result_fin, 'Engagement')

    
    data = { 
        
        'overall_result_score_nums' : overall_result, #Overall Result로 점수로 표현됨
        'overall_result_score' : overall_result_fin, #Overall Result로 Supurb ~ Lacking
        'topic_ext_re' : topic_ext_re, # 에세이에서 추출한 모든 토픽들 ---> 웹에 표시할 것
        'google_search__all_re' : google_search__all_result, # Topic Uniqueness ex)36.0
        'google_search__all_re_fin' : google_search__all_re_fin, # Topic Uniqueness Supurb ~ Lacking
        'pmt_ori_sent_re' : pmt_ori_sent_re[0], #prompt oriented sentiments 점수로 표현(숫자)
        'pmt_ori_sent_re_fin' : pmt_ori_sent_re_fin, # prompt oriented sentiments  Supurb ~ Lacking
        'Engagement_result' : Engagement_result[0], # Engagement ---> 웹에 표시할 것
        'Engagement_result_fin' : Engagement_result_fin, # Engagement Supurb ~ Lacking
        'tp_kwlg_result' : tp_kwlg_result, # topic knowledge ex(10.0
        'tp_kwlg_result_fin' : tp_kwlg_result_fin, # topic knowledge Supurb ~ Lacking
        # comments
        'fixed_top_comment' : fixed_top_comment,
        'comment_1' : comment_1,
        'comment_2' : comment_2,
        'comment_3' : comment_3,
    }

    return data






## run ##

essay_input = """ I inhale deeply and blow harder than I thought possible, tech/engineering pushing the tiny ember from its resting place on the candle out into the air. mit women's technology program (wtp) The room erupts around me, and 'Happy Birthday!' cheers echo through the halls. It's time to make a wish. In my mind, that new Limited Edition Deluxe Ben 10 watch will soon be mine. My parents and the aunties and uncles around me attempt to point me in a different direction. 'Wish that you get to go to the temple every day when you're older! Wish that you memorize all your Sanskrit texts before you turn 6! Wish that you can live in India after college!' My ears listen, but my mind tunes them out, as nothing could possibly compare to that toy watch! What I never realized on my third birthday is that those wishes quietly tell the story of how my family hopes my life will play out. In this version of my life, there wasn't much room for change, personal growth, or 'rocking the boat.' A vital aspect of my family's cultural background is their focus on accepting things as they are. Growing up, I was discouraged from questioning others or asking questions that didn't have definitive yes or no answers. If I innocently asked my grandma why she expected me to touch her feet, my dad would grab my hand in a sudden swoop, look me sternly in the eye, and tell me not to disrespect her like that again. At home, if I mentioned that I had tried eggs for breakfast at a friend's house, I'd be looked at like I had just committed a felony for eating what my parents considered meat. If I asked the priest at the temple why he had asked an Indian man and his white wife to leave, I'd be met with a condescending glare and told that I should also leave for asking such questions.In direct contrast, my curiosity was invited and encouraged at school. After an environmental science lesson, I stayed for a few minutes after class to ask my 4th-grade science teacher with wide eyes how it was possible that Niagara Falls doesn't run out of flowing water. Instead of scolding me for asking her a 'dumb question,' she smiled and explained the intricacy of the water cycle. Now, if a teacher mentions that we'll learn about why a certain proof or idea works only in a future class, I'll stay after to ask more or pour through an advanced textbook to try to understand it. While my perspective was widening at school, the receptiveness to raising complex questions at home was diminishing. After earning my driver's license, I registered as an organ donor. My small checkmark on a piece of paper led to an intense clash between my and my parents' moral platform. I wanted to ensure that I positively contributed to society, while my parents believed that organ donation was an unfamiliar and unnecessary cultural taboo. I would often ask for clarity or for reasons that supported their ideologies. Their response would usually entail feeling a deep, visceral sense that traditions must be followed exactly as taught, without objection. Told in one language to keep asking questions and in another to ask only the right ones, I chose exploring questions that don't have answers, rather than accepting answers that don't get questioned. When it comes to the maze of learning, even when I take a wrong turn and encounter roadblocks that are meant to stop me, I've learned to climb over them and keep moving forward. My curiosity strengthens with each hurdle and has expanded into a pure love of learning new things. I've become someone who seeks to understand things at a fundamental level and who finds excitement in taking on big questions that have yet to be solved. I'm no longer afraid to rock the boat. "},{"index":1,"personal_essay":"Ever since I first held a small foam Spiderman basketball in my tiny hands and watched my idol Kobe Bryant hit every three-pointer he attempted, I've wanted to understand and replicate his flawless jump shot. As my math education progressed in school, I began to realize I had the tools to create a perfect shot formula. After learning about variables for the first time in 5th grade Algebra, I began to treat each aspect of Kobe's jump shot as a different variable, each combination of variables resulting in a unique solution. While in 7th-grade geometry, I graphed the arc of his shot, and after learning about quadratic equations in 8th grade, I expressed his shot as a parabolic function that would ensure a swish when shooting from any spot. After calculus lessons in 10th and 11th grade, I was excited to finally solve for the perfect velocity and acceleration needed on my release. At Brown, I hope to explore this intellectual pursuit through a different lens. What if I could maximize the odds of making shots if I understood the science behind one's mental mindset and focus through CLPS 500: Perception and Action? Or use astrophysics to account for drag and gravitational force anywhere in the universe? Or use data science to break down the analytics of the NBA's best shooters? Through the Open Curriculum, I see myself not only becoming a more complete learner, but also a more complete thinker, applying a flexible mindset to any problem I encounter. Brown's Open Curriculum allows students to explore broadly while also diving deeply into their academic pursuits. Tell us about an academic interest (or interests) that excites you, and how you might use the Open Curriculum to pursue it. I've been playing the Mridangam since I was five years old. It's a simple instrument: A wood barrel covered on two ends by goatskin with leather straps surrounding the hull. This instrument serves as a connection between me and one of the most beautiful aspects of my culture: Carnatic music. As a young child, I'd be taken to the temple every weekend for three-hour-long Carnatic music concerts, where the most accomplished teenagers and young adults in our local Indian community would perform. I would watch in awe as the mridangists' hands moved gracefully, flowing across the goatskin as if they weren't making contact, while simultaneously producing sharp rhythmic patterns that never failed to fall on the beat. Hoping to be like these idols on the stage, I trained intensely with my teacher, a strict man who taught me that the simple drum I was playing had thousands of years of culture behind it. Building up from simple strokes, I realized that the finger speed I'd had been awestruck by wasn't some magical talent, it was instead a science perfected by repeated practice."""


#select_pmt_type = "Unique quality, passion, or talent"
select_pmt_type = "Unique quality, passion, or talent"

print('unique_quality_passion_talent :' , unique_quality_passion_talent(select_pmt_type, essay_input))