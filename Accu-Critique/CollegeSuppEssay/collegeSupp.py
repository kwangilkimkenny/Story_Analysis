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




def select_prompt_type(prompt_type):
    
    if prompt_type == 'why_us':
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






### college_dept text data와 입력한 에세이의 데이터를 비교하여 TFD-IDF 유사도를 추출한다. ###
# 입력값 #
# College_dept_data : 대학별 수집된 '개별대학의 정보'입력
# Essay_input_data : 학생의 College Supp Essay 입력
def college_n_dept_fit(College_dept_data, Essay_input_data):

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







# Selected College
# 입력값:  대학, 전공 ex) 'why_us', 'Browon', 'AfricanStudies'
def selected_college(select_pmt_type, select_college, select_major, coll_supp_essay_input_data):

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
    



    # college_dept text data와 입력한 에세이의 데이터를 비교하여 유사도를 추출한다. ex) result ==> 're_coll_n_dept_fit': {0: 1.0}}
    re_coll_n_dept_fit = college_n_dept_fit(College_text_data, coll_supp_essay_input_data)


    # 0. gen_keywd_college : 선택한 대학의 General Keywords on college로 wordcloud로 출력됨
    # 1. gen_keywd_college_major : 선택 대학의 전공에 대한 keywords 를 WrodCloud 로 출력
    # 2. intended_mjr : intended major
    # 3. pmt_sent_etc_re : 선택한 prompt 질문
    # 4. prompt_type_sentence : 선택한 prompt에 해당하는 질문 문장 전체
    # 5. pmt_sent_re : 선택한 prompt에 해당하는 sentiment 리스트

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

sc_re = selected_college('why_us', 'Brown', 'African Studies', essay_input)
print('Select_College:', sc_re)

# 에세이 입력
#input_text = """Bloomington Normal is almost laughably cliché for a midwestern city. Vast swathes of corn envelop winding roads and the heady smell of BBQ smoke pervades the countryside every summer. Yet, underlying the trite norms of Normal is the prescriptive force of tradition—the expectation to fulfill my role as a female Filipino by playing Debussy in the yearly piano festival and enrolling in multivariable calculus instead of political philosophy.So when I discovered the technical demand of bebop, the triplet groove, and the intricacies of chordal harmony after ten years of grueling classical piano, I was fascinated by the music's novelty. Jazz guitar was not only evocative and creative, but also strangely liberating. I began to explore different pedagogical methods, transcribe solos from the greats, and experiment with various approaches until my own unique sound began to develop. And, although I did not know what would be the 'best' route for me to follow as a musician, the freedom to forge whatever path I felt was right seemed to be exactly what I needed; there were no expectations for me to continue in any particular way—only the way that suited my own desires.While journeying this trail, I found myself at Interlochen Arts Camp the summer before my junior year. Never before had I been immersed in an environment so conducive to musical growth: I was surrounded by people intensely passionate about pursuing all kinds of art with no regard for ideas of what art 'should' be. I knew immediately that this would be a perfect opportunity to cultivate my sound, unbounded by the limits of confining tradition. On the first day of camp, I found that my peer guitarist in big band was another Filipino girl from Illinois. Until that moment, my endeavors in jazz guitar had been a solitary effort; I had no one with whom to collaborate and no one against whom I could compare myself, much less someone from a background mirroring my own. I was eager to play with her, but while I quickly recognized a slew of differences between us—different heights, guitars, and even playing styles—others seemed to have trouble making that distinction during performances. Some even went as far as calling me 'other-Francesca.' Thus, amidst the glittering lakes and musky pine needles of Interlochen, I once again confronted Bloomington's frustrating expectations.After being mistaken for her several times, I could not help but view Francesca as a standard of what the 'female Filipino jazz guitarist' should embody. Her improvisatory language, comping style and even personal qualities loomed above me as something I had to live up to. Nevertheless, as Francesca and I continued to play together, it was not long before we connected through our creative pursuit. In time, I learned to draw inspiration from her instead of feeling pressured to follow whatever precedent I thought she set. I found that I grew because of, rather than in spite of, her presence; I could find solace in our similarities and even a sense of comfort in an unfamiliar environment without being trapped by expectation. Though the pressure to conform was still present—and will likely remain present in my life no matter what genre I'm playing or what pursuits I engage in—I learned to eschew its corrosive influence and enjoy the rewards that it brings. While my encounter with Francesca at first sparked a feeling of pressure to conform in a setting where I never thought I would feel its presence, it also carried the warmth of finding someone with whom I could connect. Like the admittedly trite conditions of my hometown, the resemblances between us provided comfort to me through their familiarity. I ultimately found that I can embrace this warmth while still rejecting the pressure to succumb to expectations, and that, in the careful balance between these elements, I can grow in a way that feels both like discove"""


