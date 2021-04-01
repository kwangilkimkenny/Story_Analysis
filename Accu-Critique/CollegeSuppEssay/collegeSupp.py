import numpy as np
import spacy
from collections import Counter
import re
import nltk
nltk.download('averaged_perceptron_tagger')

nlp = spacy.load("en_core_web_sm")


from wordcloud import WordCloud 
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
#%matplotlib inline

import matplotlib
from IPython.display import set_matplotlib_formats
matplotlib.rc('font',family = 'Malgun Gothic')
set_matplotlib_formats('retina')
matplotlib.rc('axes',unicode_minus = False)

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
from gensim import corpora, models, similarities

### sentence_simility.py ###
from sentence_similarity import sent_sim_analysis_with_bert_summarizer



def select_prompt_type(prompt_type):
    
    if prompt_type == 'Why us':
        pmt_typ = [""" 'Why us' school & major interest (select major, by college & department) """]
        pmt_sentiment = ['Admiration', 'Excitement', 'Pride', 'Realization', 'Curiosity']
    elif prompt_type == 'Intellectual interest':
        pmt_typ = [""]
        pmt_sentiment = ['Curiosity', 'Realization']
    elif prompt_type == 'Meaningful experience & lesson learned':
        pmt_typ = [""]
        pmt_sentiment = ['']
    elif prompt_type ==  'Achievement you are proud of':
        pmt_typ = [""]
        pmt_sentiment = ['Realization', 'Approval', 'Gratitude', 'Admiration', 'Pride', 'Desire', 'Optimism']
    elif prompt_type ==  'Social issues: contribution & solution':
        pmt_typ = [""]
        pmt_sentiment = ['Anger', 'Fear', 'Disapproval','Disappointment','Realization','Approval', 'Gratitude', 'Admiration']
    elif prompt_type ==  'Summer activity':
        pmt_typ = [""]
        pmt_sentiment = ['Pride','Realization','Curiosity','Excitement','Amusement','Caring']
    elif prompt_type ==  'Unique quality, passion, or talent':
        pmt_typ = [""]
        pmt_sentiment = ['Pride','Excitement','Amusement','Approval','Admiration','Curiosity']
    elif prompt_type ==  'Extracurricular activity or work experience':
        pmt_typ = [""]
        pmt_sentiment = ['Pride','Realization','Curiosity','Joy','Excitement','Amusement','Caring','Optimism']
    elif prompt_type ==  'Your community: role and contribution in your community':
        pmt_typ = [""]
        pmt_sentiment = ['Admiration','Caring','Approval','Pride','Gratitude','Love']
    elif prompt_type ==  'College community: intended role, involvement, and contribution in college community':
        pmt_typ = [""]
        pmt_sentiment = ['Admiration','Caring','Approval','Excitement','Pride','Gratitude']
    elif prompt_type ==  'Overcoming a Challenge or ethical dilemma':
        pmt_typ = [""]
        pmt_sentiment = ['Anger','Fear','Disapproval','Disappointment','Confusion','Annoyed','Realization', 'Approval','Gratitude','Admiration','Relief','Optimism']
    elif prompt_type ==  'Culture & diversity':
        pmt_typ = [""]
        pmt_sentiment = ['Admiration','Realization','Love','Approval','Pride','Gratitude']
    elif prompt_type ==  'Collaboration & teamwork':
        pmt_typ = [""]
        pmt_sentiment = ['Admiration','Caring','Approval','Optimism','Gratitude','Love']
    elif prompt_type ==  'Creativity/creative projects':
        pmt_typ = [""]
        pmt_sentiment = ['Excitement','Realization','Curiosity','Desire','Amusement','Surprise']
    elif prompt_type ==  'Leadership experience':
        pmt_typ = [""]
        pmt_sentiment = ['Admiration','Caring','Approval','Optimism','Gratitude','Love','Fear','Confusion','Nervousness']
    elif prompt_type ==  'Values, perspectives, or beliefs':
        pmt_typ = [""]
        pmt_sentiment = ['Anger','Fear','Disapproval','Disappointment','Realization','Approval','Gratitude','Admiration']
    elif prompt_type ==  'Person who influenced you':
        pmt_typ = [""]
        pmt_sentiment = ['Realization', 'Approval', 'Gratitude','Admiration','Caring','Love','Curiosity', 'Pride', 'Joy']
    elif prompt_type ==  'Favorite book/movie/quote':
        pmt_typ = [""]
        pmt_sentiment = ['Excitement', 'Realization', 'Curiosity','Admiration','Amusement','Joy']
    elif prompt_type ==  'Write to future roommate':
        pmt_typ = [""]
        pmt_sentiment = ['Admiration','Realization','Love','Excitement','Approval','Pride','Gratitude','Amusement','Curiosity','Joy']
    elif prompt_type ==  'Diversity & Inclusion Statement':
        pmt_typ = [""]
        pmt_sentiment = ['Anger','Fear','Disapproval','Disappointment','Confusion','Annoyed','Realization','Approval','Gratitude','Admiration','Relief','Optimism']
    elif prompt_type ==  'Future goals or reasons for learning':
        pmt_typ = [""]
        pmt_sentiment = ['Realization','Approval','Gratitude','Admiration','Pride','Desire','Optimism']
    elif prompt_type ==  'What you do for fun':
        pmt_typ = [""]
        pmt_sentiment = ['Admiration', 'Excitement', 'Curiosity', 'Amusement', 'Pride','Joy']
    else:
        pass

    # pmt_typ : prompt type 
    # pmt_sentiment : prompy type에 해당하는 sentiment
    return pmt_typ, pmt_sentiment


#데이터 전처리 
def cleaning(data):
    fin_data = []
    for data_itm in data:
        # 영문자 이외 문자는 공백으로 변환
        only_english = re.sub('[^a-zA-Z]', ' ', data_itm)
        lists_re_ = re.sub(r"^\s+|\s+$", "", only_english) # 공백문자 제거
        only_english_ = lists_re_.rstrip('\n')
        # 데이터를 리스트에 추가 
        fin_data.append(only_english_)
    return fin_data


# txt 문서 정보 불러오기 : 대학정보
def open_data(select_college):
    # 폴더 구조, 대학이름 입력 명칭을 통일해야 함
    file_path = "./college_info/colleges_dataset/"
    college_name = select_college
    file_name = "_college_general_info.txt"
    # file = open("./college_info/colleges_dataset/brown_college_general_info.txt", 'r')
    file = open(file_path + college_name + file_name, 'r')
    lists = file.readlines()
    file.close()
    lists_re =  cleaning(lists) # 영어 단어가 아닌 것은 삭제
    result = ' '.join(lists_re) # 문장으로 합치기
    # 소문자 변환
    result_ = result.lower()
    #print("입력문장 불러오기 확인 : ", result_)
    return result_


# txt 문서 정보 불러오기 : 선택한 전공관련 정보 추출 
# 입력값은 대학, 전공 ex) 'Browon', 'AfricanStudies'
def open_major_data(select_college, select_major):
    # 폴더 구조, 대학이름 입력 명칭을 통일해야 함
    file_path = "./major_info/major_dataset/"
    college_name = select_college
    mjr_name = select_major
    file_name = "_major_info.txt"
    # file = open("./major_info/major_dataset/Brown_AfricanStudies_major_info.txt", 'r')
    file = open(file_path + college_name + "_" + mjr_name + file_name, 'r')
    lists = file.readlines()
    file.close()
    doc = ' '.join(lists)
    return lists


# 대학관련 정보의 토픽 키워드 추출하여 WordCloud로 구현
def general_keywords(College_text_data):
    tokenized = nltk.word_tokenize(str(College_text_data))
    #print('tokenized:', tokenized)
    nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if(pos[:2] == 'NN')]
    count = Counter(nouns)
    words = dict(count.most_common())
    #print('words:', words)
    # 가장 많이 등장하는 단어를 추려보자. 
    wordcloud = WordCloud(background_color='white',colormap = "Accent_r",
                            width=1500, height=1000).generate_from_frequencies(words)

    plt.imshow(wordcloud)
    plt.axis('off')
    gk_re = plt.show()

    return gk_re






### 첫번째 방법: college_dept text data와 입력한 에세이의 데이터를 비교하여 TFD-IDF 유사도를 추출한다. ###
# 입력값 #
# College_dept_data : 대학별 수집된 '개별대학의 정보'입력
# Essay_input_data : 학생의 College Supp Essay 입력
# 이 코드는 완벽하게 일치하는 결과로 나오기때문에 사용하지 않고 새로운 코드를 적용해볼 것
def college_n_dept_fit(College_dept_data, Essay_input_data): # ----> 사용하지 말것, 하지만 백업코드

    documents = [College_dept_data]
    
    # remove common words and tokenize them
    stoplist = set('for a of the and to in'.split())

    texts = [[word for word in document.lower().split() if word not in stoplist] for document in documents]

    # remove words those appear only once
    all_tokens = sum(texts, [])

    tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) ==1)
    texts = [[word for word in text if word not in tokens_once]
            for text in texts]
    dictionary = corpora.Dictionary(texts)

    dictionary.save('college_dept.dict')  # save as binary file at the dictionary at local directory
    dictionary.save_as_text('college_dept_text.dict')  # save as text file at the local directory

    #input college supp essay
    text_input = Essay_input_data #문장입력....

    new_vec = dictionary.doc2bow(text_input.lower().split()) # return "word-ID : Frequency of appearance""
    corpus = [dictionary.doc2bow(text) for text in texts]
    corpora.MmCorpus.serialize('college_dept.mm', corpus) # save corpus at local directory
    corpus = corpora.MmCorpus('college_dept.mm') # try to load the saved corpus from local
    dictionary = corpora.Dictionary.load('college_dept.dict') # try to load saved dic.from local
    tfidf = models.TfidfModel(corpus) # step 1 -- initialize a model
    corpus_tfidf = tfidf[corpus]  # map corpus object into tfidf space
    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=20) # initialize LSI #### num_topics=2
    corpus_lsi = lsi[corpus_tfidf] # create a double wrapper over the original corpus
    topic = lsi.print_topics(20)
    lsi.save('model.lsi')  # save output model at local directory
    lsi = models.LsiModel.load('model.lsi') # try to load above saved model

    doc = text_input

    vec_bow = dictionary.doc2bow(doc.lower().split())  # put newly obtained document to existing dictionary object
    vec_lsi = lsi[vec_bow] # convert new document (henceforth, call it "query") to LSI space
    index = similarities.MatrixSimilarity(lsi[corpus]) # transform corpus to LSI space and indexize it
    index.save('college_dept.index') # save index object at local directory
    index = similarities.MatrixSimilarity.load('college_dept.index')
    sims = index[vec_lsi] # calculate degree of similarity of the query to existing corpus

    # print(list(enumerate(sims))) # output (document_number , document similarity)

    sims = sorted(enumerate(sims), key=lambda item: -item[1])  # sort output object as per similarity ( largest similarity document comes first )
    # print(sims) # 가장 유사도가 높은 순서대로 출력
    
    # result_sims = []
    
    sim_dic = {}
    
    for temp in sims : 
        
        sim_dic[temp[0]] = round(float(temp[1]),3) # 여기서 3은 top 3개의 일치도가 높은 순서대로 추출하라는 것, 여기서는 한개만 추출(한개 입력했으니)
        
        # result_sims.append([temp[0],round(float(temp[1]),3)])

    return sim_dic
    # 결과: 're_coll_n_dept_fit': {0: 1.0}}  로 입력한 두개의 문장 일치도가 1.0은 100% 같다는 의미다.


### 두 번째 방법 유사도 측정 ####




# Selected College 외 다양한 것을 계산하는 코드(최종계산코드)
# 입력값:  대학, 전공 ex) ('Why us', 'Brown', 'Brown_African Studies_dept', 'African Studies', essay_input)
def selected_college(select_pmt_type, select_college, select_college_dept, select_major, coll_supp_essay_input_data):

    pmt_sent_etc_re = select_prompt_type(select_pmt_type)
    prompt_type_sentence = pmt_sent_etc_re[0] # prompt 문장 ex) Prompt Type : Intellectual interest 이렇게 20가지가 있음
    pmt_sent_re = list(pmt_sent_etc_re[1]) # prompt 에 해당하는 sentiment list
    intended_mjr = select_major # 희망전공

    if select_college == 'Harvard':
        pass
    elif select_college == 'Princeton':
        pass
    elif select_college == 'Stanford':
        pass
    elif select_college == 'MIT':
        pass
    elif select_college == 'Columbia':
        pass
    elif select_college == 'UPenn':
        pass
    elif select_college == 'Brown':
        College_text_data = open_data(select_college) # 선택한 대학의 정보가 담긴 txt 파일을 불러오고
        re_mjr = open_major_data(select_college, select_major) # 선택한 대학과 전공의 정보를 불러와서
        gen_keywd_college = general_keywords(College_text_data) # 키워드 추출하여 대학정보 WordCloud로 구현
        gen_keywd_college_major = general_keywords(re_mjr) # 키워드 추출하여 대학의 전공 WordCloud로 구현
    elif select_college == 'Cornell':
        pass
    elif select_college == 'Dartmouth':
        pass
    elif select_college == 'UChicago':
        pass
    elif select_college == 'Northwestern':
        pass
    elif select_college == 'Duke':
        pass
    elif select_college == 'Johns Hopkins':
        pass
    elif select_college == 'UCLA':
        pass
    elif select_college == 'UC Berkeley':
        pass
    elif select_college == 'Carnegie Mellon':
        pass
    elif select_college == 'Emory':
        pass
    elif select_college == 'Georgetown':
        pass
    elif select_college == 'UCLA':
        pass
    elif select_college == 'Emory':
        pass
    elif select_college == 'Caltech':
        pass
    elif select_college == 'USC':
        pass
    elif select_college == 'Georgetown':
        pass
    elif select_college == 'Willams':
        pass
    elif select_college == 'Swarthmore':
        pass
    elif select_college == 'Amherst':
        pass
    else:
        pass
    


    # 아래 코드는 사용하지 않음, 대신 sentence_similiarity.py 코드를 사용함
    # college_dept text data와 입력한 에세이의 데이터를 비교하여 유사도를 추출한다. ex) result ==> 're_coll_n_dept_fit': {0: 1.0}}
    # re_coll_n_dept_fit = college_n_dept_fit(College_text_data, coll_supp_essay_input_data)

    #######################################################################################
    # sentence_similiarity.py 코드
    ### 결과해석 ###
    # coll_dept_result : College & Department Fit ex)Weak, 생성한 문장
    # mjr_fit_result : Major Fit ex)Weak, 생성한 문장
    # TopComment : 첫번째 Selected Prompt 에 의한 고정 문장 생성

    # PmtOrientedSentments_result : 감성분석결과
        # counter : 선택한 prompt에 해당하는 coll supp essay의 대표적 감성 5개중 일치하는 상대적인 총 개수
        # matching_sentment : 매칭되는 감성 추출값
        # matching_ratio : 매칭 비율
        # match_result : 감성비교 최종 결과 산출

    # PmtOrientedSentments_result[3] : 최종 감성 상대적 비교 결과
    # overall_drft_sum : overall sum score(계산용 값)
    # overall_reault : Overall 최종 산출값
    #######################################################################################

    re_coll_n_dept_fit = sent_sim_analysis_with_bert_summarizer(select_pmt_type, select_college, select_college_dept, select_major, coll_supp_essay_input_data)

    # 0. gen_keywd_college : 선택한 대학의 General Keywords on college로 wordcloud로 출력됨
    # 1. gen_keywd_college_major : 선택 대학의 전공에 대한 keywords 를 WrodCloud 로 출력
    # 2. intended_mjr : intended major
    # 3. pmt_sent_etc_re : 선택한 prompt 질문
    # 4. prompt_type_sentence : 선택한 prompt에 해당하는 질문 문장 전체
    # 5. pmt_sent_re : 선택한 prompt에 해당하는 sentiment 리스트
    # 6. re_coll_n_dept_fit : sentence_similiarity.py 코드의 결과값임

    data_result = {
        'gen_keywd_college' : gen_keywd_college, # 선택한 대학의 General Keywords on college로 wordcloud로 출력됨
        'gen_keywd_college_major' : gen_keywd_college_major, # 선택 대학의 전공에 대한 keywords 를 WrodCloud 로 출력
        'intended_mjr' : intended_mjr, #intended major
        'pmt_sent_etc_re' : pmt_sent_etc_re, #선택한 prompt 질문
        'prompt_type_sentence' : prompt_type_sentence, #선택한 prompt에 해당하는 질문 문장 전체
        'pmt_sent_re' : pmt_sent_re, # 선택한 prompt에 해당하는 sentiment 리스트
        're_coll_n_dept_fit' : re_coll_n_dept_fit # College & Dept.Fit으로 입력한 Supplyment Essay와 비교하여 적합성  TFD-IDF로 계산해볼 것(우선 lexicon 사용하지 않고 계산해보자)
    }

    return data_result



### 실행 ###
# 입력값은 대학, 전공 ex) 'why_us', 'Browon', 'African Studies', text_input

# input College Supp Essay 
essay_input = """I inhale deeply and blow harder than I thought possible, pushing the tiny ember from its resting place on the candle out into the air. The room erupts around me, and 'Happy Birthday!' cheers echo through the halls. It's time to make a wish. In my mind, that new Limited Edition Deluxe Ben 10 watch will soon be mine. My parents and the aunties and uncles around me attempt to point me in a different direction. 'Wish that you get to go to the temple every day when you're older! Wish that you memorize all your Sanskrit texts before you turn 6! Wish that you can live in India after college!' My ears listen, but my mind tunes them out, as nothing could possibly compare to that toy watch! What I never realized on my third birthday is that those wishes quietly tell the story of how my family hopes my life will play out. In this version of my life, there wasn't much room for change, personal growth, or 'rocking the boat.' A vital aspect of my family's cultural background is their focus on accepting things as they are. Growing up, I was discouraged from questioning others or asking questions that didn't have definitive yes or no answers. If I innocently asked my grandma why she expected me to touch her feet, my dad would grab my hand in a sudden swoop, look me sternly in the eye, and tell me not to disrespect her like that again. At home, if I mentioned that I had tried eggs for breakfast at a friend's house, I'd be looked at like I had just committed a felony for eating what my parents considered meat. If I asked the priest at the temple why he had asked an Indian man and his white wife to leave, I'd be met with a condescending glare and told that I should also leave for asking such questions.In direct contrast, my curiosity was invited and encouraged at school. After an environmental science lesson, I stayed for a few minutes after class to ask my 4th-grade science teacher with wide eyes how it was possible that Niagara Falls doesn't run out of flowing water. Instead of scolding me for asking her a 'dumb question,' she smiled and explained the intricacy of the water cycle. Now, if a teacher mentions that we'll learn about why a certain proof or idea works only in a future class, I'll stay after to ask more or pour through an advanced textbook to try to understand it. While my perspective was widening at school, the receptiveness to raising complex questions at home was diminishing. After earning my driver's license, I registered as an organ donor. My small checkmark on a piece of paper led to an intense clash between my and my parents' moral platform. I wanted to ensure that I positively contributed to society, while my parents believed that organ donation was an unfamiliar and unnecessary cultural taboo. I would often ask for clarity or for reasons that supported their ideologies. Their response would usually entail feeling a deep, visceral sense that traditions must be followed exactly as taught, without objection. Told in one language to keep asking questions and in another to ask only the right ones, I chose exploring questions that don't have answers, rather than accepting answers that don't get questioned. When it comes to the maze of learning, even when I take a wrong turn and encounter roadblocks that are meant to stop me, I've learned to climb over them and keep moving forward. My curiosity strengthens with each hurdle and has expanded into a pure love of learning new things. I've become someone who seeks to understand things at a fundamental level and who finds excitement in taking on big questions that have yet to be solved. I'm no longer afraid to rock the boat. "},{"index":1,"personal_essay":"Ever since I first held a small foam Spiderman basketball in my tiny hands and watched my idol Kobe Bryant hit every three-pointer he attempted, I've wanted to understand and replicate his flawless jump shot. As my math education progressed in school, I began to realize I had the tools to create a perfect shot formula. After learning about variables for the first time in 5th grade Algebra, I began to treat each aspect of Kobe's jump shot as a different variable, each combination of variables resulting in a unique solution. While in 7th-grade geometry, I graphed the arc of his shot, and after learning about quadratic equations in 8th grade, I expressed his shot as a parabolic function that would ensure a swish when shooting from any spot. After calculus lessons in 10th and 11th grade, I was excited to finally solve for the perfect velocity and acceleration needed on my release. At Brown, I hope to explore this intellectual pursuit through a different lens. What if I could maximize the odds of making shots if I understood the science behind one's mental mindset and focus through CLPS 500: Perception and Action? Or use astrophysics to account for drag and gravitational force anywhere in the universe? Or use data science to break down the analytics of the NBA's best shooters? Through the Open Curriculum, I see myself not only becoming a more complete learner, but also a more complete thinker, applying a flexible mindset to any problem I encounter. Brown's Open Curriculum allows students to explore broadly while also diving deeply into their academic pursuits. Tell us about an academic interest (or interests) that excites you, and how you might use the Open Curriculum to pursue it. I've been playing the Mridangam since I was five years old. It's a simple instrument: A wood barrel covered on two ends by goatskin with leather straps surrounding the hull. This instrument serves as a connection between me and one of the most beautiful aspects of my culture: Carnatic music. As a young child, I'd be taken to the temple every weekend for three-hour-long Carnatic music concerts, where the most accomplished teenagers and young adults in our local Indian community would perform. I would watch in awe as the mridangists' hands moved gracefully, flowing across the goatskin as if they weren't making contact, while simultaneously producing sharp rhythmic patterns that never failed to fall on the beat. Hoping to be like these idols on the stage, I trained intensely with my teacher, a strict man who taught me that the simple drum I was playing had thousands of years of culture behind it. Building up from simple strokes, I realized that the finger speed I'd had been awestruck by wasn't some magical talent, it was instead a science perfected by repeated practice."""

sc_re = selected_college('Why us', 'Brown', 'Brown_African Studies_dept', 'African Studies', essay_input)
print('최종결과:', sc_re)


### 결과 ###
# Select_College: {'gen_keywd_college': None,  ---> 이부분은 워드클라우드로 표현
# 'gen_keywd_college_major': None, ---> 이부분은 워드클라우드로 표현
# 'intended_mjr': 'African Studies', 
# 'pmt_sent_etc_re': ([" 'Why us' school & major interest (select major, by college & department) "], 
# ['Admiration', 'Excitement', 'Pride', 'Realization', 'Curiosity']), 
# 'prompt_type_sentence': [" 'Why us' school & major interest (select major, by college & department) "], 
# 'pmt_sent_re': ['Admiration', 'Excitement', 'Pride', 'Realization', 'Curiosity'], 're_coll_n_dept_fit': {0: 1.0}}
# 're_coll_n_dept_fit': ((17.0, 'Weak', 'Your essay seems to be lacking some details about the college and may not demonstrate a strong interest. You may consider doing more research on the college and department you wish to study in.'), (15.0, 'Weak', "Regarding your fit with the intended major, your knowledge of the discipline's intellectual concepts seems lacking."), 'There are two key factors to consider when writing the “why us” school & major interest essay. First, you should define yourself through your intellectual interests, intended major, role in the community, and more. Secondly, you need thorough research about the college, major, and department to show strong interest. After all, it would be best if you created the “fit” between you and the college you are applying to. Meanwhile, it would help show positive sentiments such as admiration, excitement, and curiosity towards the school of your dreams.', (3, ['excitement', 'realization', 'admiration'], 60.0, 'Strong'), 'Strong', 24.8, 'Mediocre')


# Select_College:  {'gen_keywd_college': None,  ---> 이부분은 워드클라우드로 표현
# 'gen_keywd_college_major': None,  ---> 이부분은 워드클라우드로 표현
# 'intended_mjr': 'African Studies', 
# 'pmt_sent_etc_re': ([" 'Why us' school & major interest (select major, by college & department) "], 
# ['Admiration', 'Excitement', 'Pride', 'Realization', 'Curiosity']), 
# 'prompt_type_sentence': [" 'Why us' school & major interest (select major, by college & department) "], 
# 'pmt_sent_re': ['Admiration', 'Excitement', 'Pride', 'Realization', 'Curiosity'], 

# 're_coll_n_dept_fit':  ---> 아래 결과 해석(1) 부분 참고
# ((17.0, 
# 'Weak', 
# 'Your essay seems to be lacking some details about the college and may not demonstrate a strong interest. You may consider doing more research on the college and department you wish to study in.'), 
# (15.0, 
# 'Weak', 
# "Regarding your fit with the intended major, your knowledge of the discipline's intellectual concepts seems lacking."), 
# 'There are two key factors to consider when writing the “why us” school & major interest essay. First, you should define yourself through your intellectual interests, intended major, role in the community, and more. Secondly, you need thorough research about the college, major, and department to show strong interest. After all, it would be best if you created the “fit” between you and the college you are applying to. Meanwhile, it would help show positive sentiments such as admiration, excitement, and curiosity towards the school of your dreams.', 
# (3, 
# ['excitement', 'realization', 'admiration'], 
# 60.0, 'Strong'), 
# 'Strong', 
# 24.8, 
# 'Mediocre')}

   ### re_coll_n_dept_fit : 결과 해석(1) ###
    # coll_dept_result : College & Department Fit ex)Weak, 생성한 문장
    # mjr_fit_result : Major Fit ex)Weak, 생성한 문장
    # TopComment : 첫번째 Selected Prompt 에 의한 고정 문장 생성

    # PmtOrientedSentments_result : 감성분석결과
        # counter : 선택한 prompt에 해당하는 coll supp essay의 대표적 감성 5개중 일치하는 상대적인 총 개수
        # matching_sentment : 매칭되는 감성 추출값
        # matching_ratio : 매칭 비율
        # match_result : 감성비교 최종 결과 산출

    # PmtOrientedSentments_result[3] : 최종 감성 상대적 비교 결과
    # overall_drft_sum : overall sum score(계산용 값)
    # overall_reault : Overall 최종 산출값