import re
from difflib import SequenceMatcher
import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize


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
    return result, essay_input_corpus_


def SummerActivity(essay_input):

    cln_re = cleaning(essay_input)
    cln_essay = cln_re[0]
    #print('cln_essay:', cln_essay)
    cnl_sents_ = cln_re[1] # 토큰화하지 않은 문자열(에세이 전체)

    # load summer activities data
    summer_activities = pd.read_csv("./data/SummerPrograms.csv")
    #소문자로 변환
    summer_activities['title'] = summer_activities['title'].str.lower() 
    summer_activities['1st_Major_Category'] = summer_activities['1st_Major_Category'].str.lower()
    summer_activities['2nd_Major_Category'] = summer_activities['2nd_Major_Category'].str.lower()
    summer_activities['3nd_Major_Category'] = summer_activities['3nd_Major_Category'].str.lower()
    #    class	title                                   score	1st_Major_Category	2nd_Major_Category	3nd_Major_Category
    # 0	SUMMER	rsi (research science institute) at mit	5	    math/science	    tech/engineering	NaN
    # 1	SUMMER	mit women's technology program (wtp)	5	    math/science	    tech/engineering
    #  ...

    #title을 인덱스로 변환, 그래야 값을 찾기 쉽다.
    summer_activities.set_index('title', inplace=True)
    # title	                 score_cal_rate	  fin_score	
    # extremely selective	5	              5.0
    # very selective	    4	              4.0
    # ...
    #print(summer_activities)

    # 단어리스트 최기화 설정
    get_score__ = []
    get_score___ = []
    get_word_position = [] # 이것이 중요함! 추출한 단어들의 응집성(위치가 조밀하면 해당 활동일 가능성이 높음)
    for i in cln_essay: # 에세이에서 단어를 하나씩 가져와서
        cnt = 0  # 카운토 초기화
        for j in summer_activities.index: #인덱스의 summer activity 명칭을 하나씩 꺼내와서
            sum_act_wd = j.split() # 단어로 분리
            #print('len__sum_act_wd:', len(sum_act_wd))

            if i in sum_act_wd: # 활동명의 각 단어를 꺼내와서
                cnt += 1
                #essay를 리스로 변환후 포함 여부 비교, 있다면 해당 점수를 가져온다.
                get_wd_posi = cnl_sents_.find(i) # 활동명의 개별 단어가, 에세이에서 어디에 위치해 있는지 파악한다.
                get_word_position.append(get_wd_posi)# 위치값 추출한다.
                if cnt <=  len(sum_act_wd): # 인덱스의 활동 명의 단어 수보다 -4 적은 수가 일치한다면(활동명칭에서 summer, program을 제거했기때문에 숫자 -4를 적음)
                    #print('cnt :', cnt)
                    # 문자열 완전일치 판단
                    if j in cnl_sents_: # 에세이에서 활동정보(j)과 일치하는 문자열이 있는지 확인, 있다면 이하 수행 - 활동명, 스코어 추출하여 리턴
                        get_score = summer_activities.loc[j, 'score']
                        get_score__.append(j)
                        get_score___.append(get_score)
                else:
                    pass

    # 계산한 결과와 에세이 본문의 문장들과의 일치율 계산하기
    gwp_re = list(set(get_word_position)) # 중복제거
    # detect_activities = gwp_re # 활동명의 단어들이 입력에세이의 어떤 위치값을 가지는지 확인

    # 추출한 값의 중복값 제거
    get_score_fin = list(set(get_score__)) # 추출한 summer activity 명칭
    print('get_score_fin:', get_score_fin)
    get_score_fin_re = list(set(get_score___))
    print('get_score_fin_re :', get_score_fin_re) # [5] 로 결과가 리스트 값으로 나오기때문에 [0]번째의 데이터를 꺼내서 비교
    get_score_fin_re = get_score_fin_re[0]

    #추출한 점수를 5가지 척도로 변환하기.
    if get_score_fin_re  == 5:
        result_sc = 'Supurb'
        result_score = 100
    elif get_score_fin_re == 4:
        result_sc = 'Strong'
        result_score = 80
    elif get_score_fin_re == 3:
        result_sc = 'Good'
        result_score = 60
    elif get_score_fin_re == 2:
        result_sc = 'Mediocre'
        result_score = 40
    else:
        result_sc = 'Lacking'
        result_score = 20
    
    data = {
        'result_sc' : result_sc, # 매칭되는 결과로 점수로 정하기
        'get_score_fin' : get_score_fin, # 에세이서 발견한 활동명칭
        #'get_word_position' : detect_activities # 현재 이 분석값을 의미가 없지만 나중에 활용할거임
        'result_score' : result_score # 추출한 활동내역을 점수로 변환 --------> overall 값 계산에 적용할 것
    }

    return data





## run ##

essay_input = """ I inhale deeply and blow harder than I thought possible, pushing the tiny ember from its resting place on the candle out into the air. mit women's technology program (wtp) The room erupts around me, and 'Happy Birthday!' cheers echo through the halls. It's time to make a wish. In my mind, that new Limited Edition Deluxe Ben 10 watch will soon be mine. My parents and the aunties and uncles around me attempt to point me in a different direction. 'Wish that you get to go to the temple every day when you're older! Wish that you memorize all your Sanskrit texts before you turn 6! Wish that you can live in India after college!' My ears listen, but my mind tunes them out, as nothing could possibly compare to that toy watch! What I never realized on my third birthday is that those wishes quietly tell the story of how my family hopes my life will play out. In this version of my life, there wasn't much room for change, personal growth, or 'rocking the boat.' A vital aspect of my family's cultural background is their focus on accepting things as they are. Growing up, I was discouraged from questioning others or asking questions that didn't have definitive yes or no answers. If I innocently asked my grandma why she expected me to touch her feet, my dad would grab my hand in a sudden swoop, look me sternly in the eye, and tell me not to disrespect her like that again. At home, if I mentioned that I had tried eggs for breakfast at a friend's house, I'd be looked at like I had just committed a felony for eating what my parents considered meat. If I asked the priest at the temple why he had asked an Indian man and his white wife to leave, I'd be met with a condescending glare and told that I should also leave for asking such questions.In direct contrast, my curiosity was invited and encouraged at school. After an environmental science lesson, I stayed for a few minutes after class to ask my 4th-grade science teacher with wide eyes how it was possible that Niagara Falls doesn't run out of flowing water. Instead of scolding me for asking her a 'dumb question,' she smiled and explained the intricacy of the water cycle. Now, if a teacher mentions that we'll learn about why a certain proof or idea works only in a future class, I'll stay after to ask more or pour through an advanced textbook to try to understand it. While my perspective was widening at school, the receptiveness to raising complex questions at home was diminishing. After earning my driver's license, I registered as an organ donor. My small checkmark on a piece of paper led to an intense clash between my and my parents' moral platform. I wanted to ensure that I positively contributed to society, while my parents believed that organ donation was an unfamiliar and unnecessary cultural taboo. I would often ask for clarity or for reasons that supported their ideologies. Their response would usually entail feeling a deep, visceral sense that traditions must be followed exactly as taught, without objection. Told in one language to keep asking questions and in another to ask only the right ones, I chose exploring questions that don't have answers, rather than accepting answers that don't get questioned. When it comes to the maze of learning, even when I take a wrong turn and encounter roadblocks that are meant to stop me, I've learned to climb over them and keep moving forward. My curiosity strengthens with each hurdle and has expanded into a pure love of learning new things. I've become someone who seeks to understand things at a fundamental level and who finds excitement in taking on big questions that have yet to be solved. I'm no longer afraid to rock the boat. "},{"index":1,"personal_essay":"Ever since I first held a small foam Spiderman basketball in my tiny hands and watched my idol Kobe Bryant hit every three-pointer he attempted, I've wanted to understand and replicate his flawless jump shot. As my math education progressed in school, I began to realize I had the tools to create a perfect shot formula. After learning about variables for the first time in 5th grade Algebra, I began to treat each aspect of Kobe's jump shot as a different variable, each combination of variables resulting in a unique solution. While in 7th-grade geometry, I graphed the arc of his shot, and after learning about quadratic equations in 8th grade, I expressed his shot as a parabolic function that would ensure a swish when shooting from any spot. After calculus lessons in 10th and 11th grade, I was excited to finally solve for the perfect velocity and acceleration needed on my release. At Brown, I hope to explore this intellectual pursuit through a different lens. What if I could maximize the odds of making shots if I understood the science behind one's mental mindset and focus through CLPS 500: Perception and Action? Or use astrophysics to account for drag and gravitational force anywhere in the universe? Or use data science to break down the analytics of the NBA's best shooters? Through the Open Curriculum, I see myself not only becoming a more complete learner, but also a more complete thinker, applying a flexible mindset to any problem I encounter. Brown's Open Curriculum allows students to explore broadly while also diving deeply into their academic pursuits. Tell us about an academic interest (or interests) that excites you, and how you might use the Open Curriculum to pursue it. I've been playing the Mridangam since I was five years old. It's a simple instrument: A wood barrel covered on two ends by goatskin with leather straps surrounding the hull. This instrument serves as a connection between me and one of the most beautiful aspects of my culture: Carnatic music. As a young child, I'd be taken to the temple every weekend for three-hour-long Carnatic music concerts, where the most accomplished teenagers and young adults in our local Indian community would perform. I would watch in awe as the mridangists' hands moved gracefully, flowing across the goatskin as if they weren't making contact, while simultaneously producing sharp rhythmic patterns that never failed to fall on the beat. Hoping to be like these idols on the stage, I trained intensely with my teacher, a strict man who taught me that the simple drum I was playing had thousands of years of culture behind it. Building up from simple strokes, I realized that the finger speed I'd had been awestruck by wasn't some magical talent, it was instead a science perfected by repeated practice."""


print('SummerActivity result :', SummerActivity(essay_input))

