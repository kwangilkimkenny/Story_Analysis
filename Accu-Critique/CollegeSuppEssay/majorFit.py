# major fit
# 1) get major words from essay
# 2) selected prompt



import re
from difflib import SequenceMatcher
import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
import multiprocessing
from gensim.models import Phrases
import gensim


### sentence_simility.py ###
from sentence_similarity import sent_sim_analysis_with_bert_summarizer

#데이터 전처리 
def cleaning(essay_input):
    #입력한 글을 모두 단어로 쪼개로 리스트로 만들기 - 
    essay_input_corpus_ = str(essay_input) #문장입력
    essay_input_corpus_ = essay_input_corpus_.lower()#소문자 변환

    sentences_  = sent_tokenize(essay_input_corpus_) #문장단위로 토큰화(구분)되어 리스에 담김

    # 문장을 토크큰화하여 해당 문장에 Verbs가 있는지 분석 부분 코드임 

    split_sentences_ = []
    for sentence in sentences_:
        processed = re.sub("[^a-zA-Z]"," ", sentence)
        words = processed.split()
        split_sentences_.append(words)
        
    # 입력한 문장을 모두 리스트로 변환
    input_text_list = [y for x in split_sentences_ for y in x] # 이중 리스트 Flatten
    result = list(set(input_text_list))
    result_include_double = input_text_list # 중복이 저게되지 않은 리스트
    return result, essay_input_corpus_, result_include_double


def mjorfit(select_pmt_type, select_college, select_college_dept, select_major, coll_supp_essay_input_data):
    re_coll_n_dept_fit = sent_sim_analysis_with_bert_summarizer(select_pmt_type, select_college, select_college_dept, select_major, coll_supp_essay_input_data)

    # major fit result  --> 이 값을 intellectualEngagement.py의 def intellectualEnguagement(essay_input, input_mjr_score)에 입력해서 결과를 다시 가져옴
    mjr_fit_result_final = re_coll_n_dept_fit[7] #major fit 일치율 점수
    mjr_fit_result = re_coll_n_dept_fit[1][1] #Major Fit ex)Weak 처럼 결과 추출

    return mjr_fit_result_final, mjr_fit_result








### run ###

essay_input = """ I inhale deeply and blow harder than I thought possible, tech/engineering pushing the tiny ember from its resting place on the candle out into the air. mit women's technology program (wtp) The room erupts around me, and 'Happy Birthday!' cheers echo through the halls. It's time to make a wish. In my mind, that new Limited Edition Deluxe Ben 10 watch will soon be mine. My parents and the aunties and uncles around me attempt to point me in a different direction. 'Wish that you get to go to the temple every day when you're older! Wish that you memorize all your Sanskrit texts before you turn 6! Wish that you can live in India after college!' My ears listen, but my mind tunes them out, as nothing could possibly compare to that toy watch! What I never realized on my third birthday is that those wishes quietly tell the story of how my family hopes my life will play out. In this version of my life, there wasn't much room for change, personal growth, or 'rocking the boat.' A vital aspect of my family's cultural background is their focus on accepting things as they are. Growing up, I was discouraged from questioning others or asking questions that didn't have definitive yes or no answers. If I innocently asked my grandma why she expected me to touch her feet, my dad would grab my hand in a sudden swoop, look me sternly in the eye, and tell me not to disrespect her like that again. At home, if I mentioned that I had tried eggs for breakfast at a friend's house, I'd be looked at like I had just committed a felony for eating what my parents considered meat. If I asked the priest at the temple why he had asked an Indian man and his white wife to leave, I'd be met with a condescending glare and told that I should also leave for asking such questions.In direct contrast, my curiosity was invited and encouraged at school. After an environmental science lesson, I stayed for a few minutes after class to ask my 4th-grade science teacher with wide eyes how it was possible that Niagara Falls doesn't run out of flowing water. Instead of scolding me for asking her a 'dumb question,' she smiled and explained the intricacy of the water cycle. Now, if a teacher mentions that we'll learn about why a certain proof or idea works only in a future class, I'll stay after to ask more or pour through an advanced textbook to try to understand it. While my perspective was widening at school, the receptiveness to raising complex questions at home was diminishing. After earning my driver's license, I registered as an organ donor. My small checkmark on a piece of paper led to an intense clash between my and my parents' moral platform. I wanted to ensure that I positively contributed to society, while my parents believed that organ donation was an unfamiliar and unnecessary cultural taboo. I would often ask for clarity or for reasons that supported their ideologies. Their response would usually entail feeling a deep, visceral sense that traditions must be followed exactly as taught, without objection. Told in one language to keep asking questions and in another to ask only the right ones, I chose exploring questions that don't have answers, rather than accepting answers that don't get questioned. When it comes to the maze of learning, even when I take a wrong turn and encounter roadblocks that are meant to stop me, I've learned to climb over them and keep moving forward. My curiosity strengthens with each hurdle and has expanded into a pure love of learning new things. I've become someone who seeks to understand things at a fundamental level and who finds excitement in taking on big questions that have yet to be solved. I'm no longer afraid to rock the boat. "},{"index":1,"personal_essay":"Ever since I first held a small foam Spiderman basketball in my tiny hands and watched my idol Kobe Bryant hit every three-pointer he attempted, I've wanted to understand and replicate his flawless jump shot. As my math education progressed in school, I began to realize I had the tools to create a perfect shot formula. After learning about variables for the first time in 5th grade Algebra, I began to treat each aspect of Kobe's jump shot as a different variable, each combination of variables resulting in a unique solution. While in 7th-grade geometry, I graphed the arc of his shot, and after learning about quadratic equations in 8th grade, I expressed his shot as a parabolic function that would ensure a swish when shooting from any spot. After calculus lessons in 10th and 11th grade, I was excited to finally solve for the perfect velocity and acceleration needed on my release. At Brown, I hope to explore this intellectual pursuit through a different lens. What if I could maximize the odds of making shots if I understood the science behind one's mental mindset and focus through CLPS 500: Perception and Action? Or use astrophysics to account for drag and gravitational force anywhere in the universe? Or use data science to break down the analytics of the NBA's best shooters? Through the Open Curriculum, I see myself not only becoming a more complete learner, but also a more complete thinker, applying a flexible mindset to any problem I encounter. Brown's Open Curriculum allows students to explore broadly while also diving deeply into their academic pursuits. Tell us about an academic interest (or interests) that excites you, and how you might use the Open Curriculum to pursue it. I've been playing the Mridangam since I was five years old. It's a simple instrument: A wood barrel covered on two ends by goatskin with leather straps surrounding the hull. This instrument serves as a connection between me and one of the most beautiful aspects of my culture: Carnatic music. As a young child, I'd be taken to the temple every weekend for three-hour-long Carnatic music concerts, where the most accomplished teenagers and young adults in our local Indian community would perform. I would watch in awe as the mridangists' hands moved gracefully, flowing across the goatskin as if they weren't making contact, while simultaneously producing sharp rhythmic patterns that never failed to fall on the beat. Hoping to be like these idols on the stage, I trained intensely with my teacher, a strict man who taught me that the simple drum I was playing had thousands of years of culture behind it. Building up from simple strokes, I realized that the finger speed I'd had been awestruck by wasn't some magical talent, it was instead a science perfected by repeated practice."""

# Test 1 (학교, 부서, 전공을 입력했을 경우) : ('Why us', 'Brown', 'Brown_African Studies_dept', 'African Studies', essay_input)
# Test 2 (학교, 부서는 Undecided, 전공도 Undecided를 입력했을 경우) : ('Why us', 'Brown', 'Undecided', 'Undecided', essay_input)
major_fit = mjorfit('Intelectual interest', 'Brown', 'Undecided', 'Undecided', essay_input)

print('major fit:', major_fit)

# major fit: 17.0 % 의 일치율을 보임
# Major Fit ex)Weak 처럼 결과 추출