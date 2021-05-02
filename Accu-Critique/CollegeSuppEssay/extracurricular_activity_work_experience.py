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
from prompt_oriented_sentiments import prompt_ori_sentiment
from topic_uniqueness import google_search_result
from intellectualEngagement import intellectualEnguagement

def extra_act_work_experi(select_pmt_type, essay_input):
    # 에세이에서 추출한 모든 토픽
    topic_ext_re = topic_extraction(essay_input) # like', 'usually', 'unique', 'thought', 'led' ...
    #print('topic_ext_re:', topic_ext_re[:10])
    google_search__all_re = []
    for i in topic_ext_re[:10]:
        google_search_re = google_search_result(i) # 토픽중 10개만 검색해서 결과 추출. 일단 1개만 추출(개발 끝난 후 10개로 변경)
        searched_mean_score = google_search_re[1]
        google_search__all_re.append(searched_mean_score)
    pmt_ori_sent_re = prompt_ori_sentiment(select_pmt_type, essay_input)


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
        

    pmt_ori_sent_re_fin = calculate_score(pmt_ori_sent_re[0])

    Engagement_result = intellectualEnguagement(essay_input)
    Engagement_result_fin = calculate_score(Engagement_result[0])

    major_fit_re_fin = 



    overall_result = round(pmt_ori_sent_re[0] * 0.4 + Engagement_result[0] * 0.4 + mjr_fit * 0.1, 1)

    # 문장생성
    fixed_top_comment = """To define a unique quality, passion, or talent, you should find something about you that is not considered cliché. It should be a quality that excites you and makes you proud, while demonstrating your dedication and knowledge towards such passion would be appropriate."""

    def gen_comment(input_score, type):
        if input_score == 'Supurb' or input_score == 'Strong':
            if type == 'pmt_ori_sentiment':
                comment_achieve = """"""
            elif type == 'Engagement':
                comment_achieve = """"""
            elif type == 'MajorFit':
                comment_achieve = """"""
            else:
                pass
        elif input_score == 'Good':
            if type == 'pmt_ori_sentiment':
                comment_achieve = """"""
            elif type == 'Engagement':
                comment_achieve = """"""
            elif type == 'MajorFit':
                comment_achieve = """"""
            else:
                pass
        else: #input score == 'Mediocre' or input_score == 'Weak'
            if type == 'pmt_ori_sentiment':
                comment_achieve = """"""
            elif type == 'Engagement':
                comment_achieve = """"""
            elif type == 'MajorFit':
                comment_achieve = """"""
            else:
                pass
        return comment_achieve



    comment_1 = gen_comment(pmt_ori_sent_re_fin, 'pmt_ori_sentiment')
    comment_2 = gen_comment(Engagement_result_fin, 'Engagement')
    comment_3 = gen_comment(major_fit_re_fin, 'pmt_ori_sentiment')

    # 추출한 점수를 5가지 척도로 변환하기.
    if extract_mjr_num  == 5:
        result_sc = 'Supurb'
        result_score = 100
    elif extract_mjr_num == 4:
        result_sc = 'Strong'
        result_score = 80
    elif extract_mjr_num == 3:
        result_sc = 'Good'
        result_score = 60
    elif extract_mjr_num == 2:
        result_sc = 'Mediocre'
        result_score = 40
    else:
        result_sc = 'Lacking'
        result_score = 20


    data = { 
        'pmt_ori_sent_re' : pmt_ori_sent_re[0], #prompt oriented sentiments 점수로 표현(숫자)
        'pmt_ori_sent_re_fin' : pmt_ori_sent_re_fin, # prompt oriented sentiments  Supurb ~ Lacking
        'Engagement_result_fin' : Engagement_result_fin, # Engagement Supurb ~ Lacking
        'Major Score' : result_sc, # 전공접합성의 매칭되는 결과로 점수로 정하기 5점 척도로 표현

        # comments
        'fixed_top_comment' : fixed_top_comment,
        'comment_1' : comment_1,
        'comment_2' : comment_2,
        'comment_3' : comment_3,

        }

    return data


    ## run ##

essay_input = """ I inhale deeply and blow harder than I thought possible, tech/engineering pushing the tiny ember from its resting place on the candle out into the air. mit women's technology program (wtp) The room erupts around me, and 'Happy Birthday!' cheers echo through the halls. It's time to make a wish. In my mind, that new Limited Edition Deluxe Ben 10 watch will soon be mine. My parents and the aunties and uncles around me attempt to point me in a different direction. 'Wish that you get to go to the temple every day when you're older! Wish that you memorize all your Sanskrit texts before you turn 6! Wish that you can live in India after college!' My ears listen, but my mind tunes them out, as nothing could possibly compare to that toy watch! What I never realized on my third birthday is that those wishes quietly tell the story of how my family hopes my life will play out. In this version of my life, there wasn't much room for change, personal growth, or 'rocking the boat.' A vital aspect of my family's cultural background is their focus on accepting things as they are. Growing up, I was discouraged from questioning others or asking questions that didn't have definitive yes or no answers. If I innocently asked my grandma why she expected me to touch her feet, my dad would grab my hand in a sudden swoop, look me sternly in the eye, and tell me not to disrespect her like that again. At home, if I mentioned that I had tried eggs for breakfast at a friend's house, I'd be looked at like I had just committed a felony for eating what my parents considered meat. If I asked the priest at the temple why he had asked an Indian man and his white wife to leave, I'd be met with a condescending glare and told that I should also leave for asking such questions.In direct contrast, my curiosity was invited and encouraged at school. After an environmental science lesson, I stayed for a few minutes after class to ask my 4th-grade science teacher with wide eyes how it was possible that Niagara Falls doesn't run out of flowing water. Instead of scolding me for asking her a 'dumb question,' she smiled and explained the intricacy of the water cycle. Now, if a teacher mentions that we'll learn about why a certain proof or idea works only in a future class, I'll stay after to ask more or pour through an advanced textbook to try to understand it. While my perspective was widening at school, the receptiveness to raising complex questions at home was diminishing. After earning my driver's license, I registered as an organ donor. My small checkmark on a piece of paper led to an intense clash between my and my parents' moral platform. I wanted to ensure that I positively contributed to society, while my parents believed that organ donation was an unfamiliar and unnecessary cultural taboo. I would often ask for clarity or for reasons that supported their ideologies. Their response would usually entail feeling a deep, visceral sense that traditions must be followed exactly as taught, without objection. Told in one language to keep asking questions and in another to ask only the right ones, I chose exploring questions that don't have answers, rather than accepting answers that don't get questioned. When it comes to the maze of learning, even when I take a wrong turn and encounter roadblocks that are meant to stop me, I've learned to climb over them and keep moving forward. My curiosity strengthens with each hurdle and has expanded into a pure love of learning new things. I've become someone who seeks to understand things at a fundamental level and who finds excitement in taking on big questions that have yet to be solved. I'm no longer afraid to rock the boat. "},{"index":1,"personal_essay":"Ever since I first held a small foam Spiderman basketball in my tiny hands and watched my idol Kobe Bryant hit every three-pointer he attempted, I've wanted to understand and replicate his flawless jump shot. As my math education progressed in school, I began to realize I had the tools to create a perfect shot formula. After learning about variables for the first time in 5th grade Algebra, I began to treat each aspect of Kobe's jump shot as a different variable, each combination of variables resulting in a unique solution. While in 7th-grade geometry, I graphed the arc of his shot, and after learning about quadratic equations in 8th grade, I expressed his shot as a parabolic function that would ensure a swish when shooting from any spot. After calculus lessons in 10th and 11th grade, I was excited to finally solve for the perfect velocity and acceleration needed on my release. At Brown, I hope to explore this intellectual pursuit through a different lens. What if I could maximize the odds of making shots if I understood the science behind one's mental mindset and focus through CLPS 500: Perception and Action? Or use astrophysics to account for drag and gravitational force anywhere in the universe? Or use data science to break down the analytics of the NBA's best shooters? Through the Open Curriculum, I see myself not only becoming a more complete learner, but also a more complete thinker, applying a flexible mindset to any problem I encounter. Brown's Open Curriculum allows students to explore broadly while also diving deeply into their academic pursuits. Tell us about an academic interest (or interests) that excites you, and how you might use the Open Curriculum to pursue it. I've been playing the Mridangam since I was five years old. It's a simple instrument: A wood barrel covered on two ends by goatskin with leather straps surrounding the hull. This instrument serves as a connection between me and one of the most beautiful aspects of my culture: Carnatic music. As a young child, I'd be taken to the temple every weekend for three-hour-long Carnatic music concerts, where the most accomplished teenagers and young adults in our local Indian community would perform. I would watch in awe as the mridangists' hands moved gracefully, flowing across the goatskin as if they weren't making contact, while simultaneously producing sharp rhythmic patterns that never failed to fall on the beat. Hoping to be like these idols on the stage, I trained intensely with my teacher, a strict man who taught me that the simple drum I was playing had thousands of years of culture behind it. Building up from simple strokes, I realized that the finger speed I'd had been awestruck by wasn't some magical talent, it was instead a science perfected by repeated practice."""


select_pmt_type = "Extracurricular activity or work experience"

print('unique_quality_passion_talent :' , extra_act_work_experi(select_pmt_type, essay_input))