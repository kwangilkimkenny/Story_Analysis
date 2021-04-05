# 아나콘다 가상환경 office:  py37TF2
# home : py37Keras

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


def GeneralAcademicKnowledge(text):

    essay_input_corpus = str(text) #문장입력
    essay_input_corpus = essay_input_corpus.lower()#소문자 변환
    #print('essay_input_corpus :', essay_input_corpus)
    
    sentences  = sent_tokenize(essay_input_corpus) #문장 토큰화 > 문장으로 구분
    total_sentences = len(sentences)#토큰으로 처리된 총 문장 수
    total_words = len(word_tokenize(essay_input_corpus))# 총 단어수
    split_sentences = []
    for sentence in sentences:
        processed = re.sub("[^a-zA-Z]"," ", sentence)
        words = processed.split()
        split_sentences.append(words)

    skip_gram = 1
    workers = multiprocessing.cpu_count()
    bigram_transformer = Phrases(split_sentences)

    model = gensim.models.word2vec.Word2Vec(bigram_transformer[split_sentences], workers=workers, sg=skip_gram, min_count=1)

    model.train(split_sentences, total_examples=sum([len(sentence) for sentence in sentences]), epochs=100)
    
    #모델 설계 완료

    #setting을 표현하는 단어들을 리스트에 넣어서 필터로 만들고
    academic_word_list = ['Cybersecurity','E-business','Ethics','Glass ceiling','Online retail','Outsourcing'
                        ,'Sweatshops','White collar crime','Acquaintance rape','Animal rights','Assisted suicide','Campus violence'
                        ,'Capital punishment','Civil rights','Drinking age, legal','Drug legalization','Gun control','Hate crimes'
                        ,'Insanity defense','Mandatory Minimum sentencing','Patriot Act','Police brutality','Prisons and prisoners'
                        ,'Roe vs. Wade','Serial killers','Sex crimes','Sexual harassment','Three Strikes Law','Drugs','Drug Abuse'
                        ,'Alcohol','Cocaine','Doping in sports','Drug testing','Drunk driving','Heroin','Marijuana'
                        ,'Nicotine','Education','Attention deficit disorder','Charter schools','College admission policies'
                        ,'College athletes','College tuition planning','Distance education','Diploma mills'
                        ,'Education and funding','Grade inflation','Greek letter societies','Hazing','Home schooling','Intelligence tests'
                        ,'Learning disabilities','Literacy in America','No Child Left Behind','Plagiarism','Prayer in schools'
                        ,'Sex education','School vouchers','Standardized tests','Environmental','Acid rain','Alternative fuel/hybrid vehicles','Conservation'
                        ,'Climate Change','Deforestation','Endangered species','Energy','Geoengineering','Global warming'
                        ,'Greenhouse effect','Hurricanes','Landfills','Marine pollution','Nuclear energy','Oil spills'
                        ,'Pesticides','Petroleum','Pollution','Population control','Radioactive waste disposal','Recycling'
                        ,'Smog','Soil pollution','Wildlife conservation','Family issues','Battered woman syndrome'
                        ,'Child abuse','Divorce rates','Domestic abuse','Family relationships','Family values'
                        ,'Health','Abortion','AIDS','Attention deficit disorder','Alternative medicine','Alzheimer’s Disease','Anorexia Nervosa'
                        ,'Artificial insemination','Autism','Birth control','Bulimia','Cancer','Coronavirus','COVID-19','Depression','Dietary supplements'
                        ,'Drug abuse','Dyslexia','Exercise and fitness','Fad diets','Fast food','Heart disease','Health Care Reform'
                        ,'HIV infection','In vitro fertilization','Medicaid, Medicare reform','Obesity','Organic foods','Prescription drugs','Plastic surgery'
                        ,'SARS','Sleep','Smoking','Stem cell research','Teen pregnancy','Vegetarianism','Weight loss surgery','Media','Communications'
                        ,'Body image','Censorship','Children’s programming','advertising','Copyright Law','Freedom of speech','Materialism'
                        ,'Media bias','Media conglomerates, ownership','Minorities in mass media','Political correctness','Portrayal of women','Reality television'
                        ,'Stereotypes','Talk radio','Television violence','Political Issue','Affirmative Action','Budget deficit','Electoral College'
                        ,'Election reform','Emigration','Genocide','Illegal aliensIllegal aliens','Immigration','Impeachment','International relations'
                        ,'Medicaid, Medicare reform','Operation Enduring Iraqi Freedom','Partisan politics','Prescription drugs','Social Security Reform'
                        ,'Third parties','Taxes','Psychology','Child abuse','Criminal psychology','Depression','Dreams','Intelligence tests','Learning disabilities'
                        ,'Memory','Physical attraction','Schizophrenia','Religion','Cults','Freedom of religion','Occultism','Prayer in schools','Social Issues'
                        ,'Abortion','Adoption','Airline safety','Airline security','Affirmative Action programs'
                        ,'AIDS','Apartheid','Birth control','Child abuse','Child rearing','Discrimination in education'
                        ,'Employee rights','Gambling, online gaming','Gang identity','Gay, lesbian, bisexual, or transgender'
                        ,'Gay parenting','Gender discrimination','Genetic screening','Homelessness','Identity theft','Interracial marriage'
                        ,'Poverty','Race relations','Reverse discrimination','Suffrage','Suicide','Test biases','Textbook biases','Welfare'
                        ,'Terrorism','Bioterrorism','Homeland Security','September 11','Women and Gender','Abortion','Birth control and Pregnancy'
                        ,'Body image','Cultural expectations and practices','Discrimination','Eating disorders','Education','Feminism','Gay pride'
                        ,'Female genital mutilation','Health','Marriage and Divorce','Media portrayals','Menstruation and Menopause','Parenting'
                        ,'Prostitution','LGBT (lesbian, gay, bisexual, transgender)','Sex and Sexuality','Sports','Stereotypes','Substance abuse'
                        ,'Violence and Rape','Work']

    
    lower_academic_words = []
    for a_itm in academic_word_list:
        lower_a_itm = a_itm.lower()
        lower_academic_words.append(lower_a_itm)

    academic_words_filter_list = academic_word_list + lower_academic_words
    print('academic_words_filter_list :' , academic_words_filter_list)
    ####문장에 ords_filter_list의 단어들이 있는지 확인하고, 있다면 유사단어를 추출한다.
    #우선 토큰화한다.
    retokenize = RegexpTokenizer("[\w]+") #줄바꿈 제거하여 한줄로 만들고
    token_input_text = retokenize.tokenize(essay_input_corpus)
    # print (token_input_text) #토큰화 처리 확인.. 토큰들이 리스트에 담김
    # 리트스로 정리된 개별 토큰을 char_list와 비교해서 존재하는 것만 추출한다.
    filtered_setting_text = []
    for k in token_input_text:
        for j in academic_words_filter_list:
            if k == j:
                filtered_setting_text.append(j)
    
    # print (filtered_chr_text) # 유사단어 비교 추출 완료, 겹치는 단어는 제거하자.
    
    filtered_setting_text_ = set(filtered_setting_text) #중복제거
    filtered_setting_text__ = list(filtered_setting_text_) #다시 리스트로 변환
    # print (filtered_setting_text__) # 중복값 제거 확인
    
   
    # 문장내 모든 academic 단어 추출
    tot_setting_words = filtered_setting_text__
    
    # academic 단어가 포함된 문장을 찾아내서 추출하기
    # if academic단어가 문장에 있다면, 그 문장을 추출(.로 split한 문장 리스트)해서 리스트로 저장한다.
    
    # print('sentences: ', sentences) # .로 구분된 전체 문장
    
    sentence_to_words = word_tokenize(essay_input_corpus) # 총 문장을 단어 리스트로 변환
    # print('sentence_to_words:', sentence_to_words)
    
    # academic단어가 포함된 문장을 찾아내서 추출
    extrace_sentence_and_setting_words = [] # 이것은 "문장", 'academic단어' ... 합쳐서 리스트로 저장
    extract_only_sentences_include_setting_words = [] # academic 단어가 포함된 문장만 리스트로 저장
    for sentence in sentences: # 문장을 하나씩 꺼내온다.
        for item in tot_setting_words: # academic단어를 하나씩 꺼내온다.
            if item in word_tokenize(sentence): # 꺼낸 문장을 단어로 나누고, 그 안에 academic 단어가 있다면
                extrace_sentence_and_setting_words.append(sentence) # academic 단어가 포함된 문장을 별도로 저장한다.
                extrace_sentence_and_setting_words.append(item) # academic 단어도 추가로 저장한다. 
                
                extract_only_sentences_include_setting_words.append(sentence)
                
                
                ## 찾는 단어 수 대로 문장을 모두 별도 저장하기때문에 문장이 중복 저장된다. 한번만 문장이 저장되도록 하자. 
                ## 문장. '단어' , '단어' 이런 식으로다가 수정해야함. 중복리스트를 제거하면 됨.
    # 중복리스트를 제거한다.
    extrace_sentence_with_setting_words_re = set(extrace_sentence_and_setting_words)
    #print('extrace_sentence_and_setting_words(문장+단어)) :', extrace_sentence_with_setting_words_re)
    
    extract_only_sentences_include_setting_words_re = set(extract_only_sentences_include_setting_words) #중복제거
    #print('extract_only_sentences_include_setting_words(오직 academic 포함 문장):', extract_only_sentences_include_setting_words_re)
    
    # 단, 소문자로 문장이 저장되어 있어서, 동일한 원문을 찾을 수 없다. 소문자로 되어있는 문장을 통해서 대문자가 섞여있는 원문을 찾자
    # 방법) 소문자 문장을 단어로로 토크나이즈한 후 리스트로 만든다. 대문자 문장도 단어로 토크나이즈한 후 리스트로 만든다.
    # 두 개의 리스트를 비교해서 같은 단어가 3개 혹은 5개 이상 나오면 대문자 문장의 원문을 매칭한다. 끝!
    
    #아래 메소드에 리스트로된 문장 삽입, set 함수로 처리된것을 다시 list로 변환해야 첫 글자를 대문자로 바꿀 수 있다.
    lower_text_input = list(extract_only_sentences_include_setting_words_re)
    #print('lower_text_input: ', lower_text_input[0])
    
    ######################################################################################
    ###### 소문자 문장으로 대문자 포함원 원문 추출하는 함수 ########
    # essay_input_corpus : 최초의 입력문자를 스트링으로 변환한 원본
    def find_original_sentence(lower_text_input, essay_input_corpus):
        
        #1)원본 전체을 문장으로 토큰화
        sentence_tokenized = sent_tokenize(essay_input_corpus)
        #print("======================================")
        #print('sentence_tokenized:',sentence_tokenized)
        #문장으로 토큰화한 것을 리스트로 묶어서 다시 단어로 토큰화한다. 
        word_tokenized = [] #입력에세이 원본의 토큰화된 리스트화!
        for st_to in sentence_tokenized:
            word_tokenized.append(word_tokenize(st_to))
        #print("======================================")
        # 이렇게 되어 있을 것이다 -> 문장으로 구분되어 리스트로 나뉘고 다시 단어로 분할되어 리스트[['단어','단어', ...], ['단어','단어', ...]...]
        #print('word_tokenized:', word_tokenized)
        
        
        #2)다음으로 계산 추출된 소문자로 변환된 academic단어 포함 문장의 단어에 대해서 첫 글자를 대문자로 만든다.
        capital_text = []
        for lt in lower_text_input:  
            capital_text.append(lt.capitalize())
        #print("======================================")    
        #print('captal_text(첫글자 대문자로 변환되었는지 확인!!!!!!!!!!):', capital_text) # 잘됨!
        
        capital_token_text_list = []
        for cpt_item in capital_text:
            #단어로 분할해서 리스트에 담는다.
            capital_token_text_list.append(word_tokenize(cpt_item))
        #print("======================================")
        # 이렇게 되어 있을 것이다 -> 문장으로 구분되어 리스트로 나뉘고 다시 단어로 분할되어 리스트[['단어','단어', ...], ['단어','단어', ...]
        #print('captal_token_text_list:',capital_token_text_list)
        
        
        # 이제 아래 두개의 리스트를 비교해서 원본을 찾아야 한다.그리고 다시 찾은 원본토큰화된 단어 리스트를 문장으로 복원한다.
        
        # word_tokenized : 입력에세이 원본의 토큰화된 리스트화! (원본문장)
        # capital_token_text_list : 추출된 에세이 결과물 토큰화 (입력문장)
        
        # print('word_tokenized:', word_tokenized)
        # print(('capital_token_text_list:', capital_token_text_list))
        
        # academic 표현이 포함된 최종 문장의 리트스 추출
        count_ct_item = 0
        included_character_exp = []
        for ct in capital_token_text_list:
            for wt in word_tokenized:
                for ct_item in ct:
                    if count_ct_item >= 5: # 겹치는 단어가 4개 이상이면 같은 문장이라고 판단하자 
                        # 같은 문장이기 땜누에 원본 리스트의 단어들을 하나의 문장으로 만들어서 저장하자
                        re_cpt = ' '.join(wt).capitalize()
                        included_character_exp.append(re_cpt)
                    elif ct_item in wt: # 리스트 안에 비교리스트가 있다면, 단어 수 카운트하고 for문 돌림
                        count_ct_item += 1
                        #print('count_ct_item:', count_ct_item)
                    else: # 비교 후 겹치는 값이 없다면 패스
                        pass
                    
        # 최종결과물 첫 글자 대문자로 복원
        
        # 최종 결과물 중복제거
        result_origin = set(included_character_exp) #academic 단어를 사용한 총 문장을 리스트로 출력
        setting_total_sentences_number = len(result_origin) # academic 단어가 발견된 총 문장수를 구하라
        return result_origin, setting_total_sentences_number
    ####################################################################################
    
    
    # academic 단어가 포함된 모든 문장을 추출
    find_origin_result = find_original_sentence(lower_text_input, essay_input_corpus)
    totalSettingSentences = find_origin_result[0]
    #print('totalSettingSentences:', totalSettingSentences)
    
    # academic 단어가 포함된 총 문장 수
    setting_total_sentences_number_re = find_origin_result[1]
    ####################################################################################
    # 합격자들의 평균 academic문장 사용 수(임의로 설정, 나중에 평균값 계산해서 적용할 것)
    setting_total_sentences_number_of_admitted_student = 20
    ####################################################################################
    
    
    # 문장생성 부분  - Overall Emphasis on Setting의 첫 문장값 계산
    
    if setting_total_sentences_number_re > setting_total_sentences_number_of_admitted_student:
        less_more_numb = abs(setting_total_sentences_number_re - setting_total_sentences_number_of_admitted_student)
        over_all_sentence_1 = [less_more_numb, 'more']
    elif setting_total_sentences_number_re < setting_total_sentences_number_of_admitted_student:
        less_more_numb = abs(setting_total_sentences_number_re - setting_total_sentences_number_of_admitted_student)
        over_all_sentence_1 = [less_more_numb, 'fewer']
    elif setting_total_sentences_number_re == setting_total_sentences_number_of_admitted_student: # ??? 두값이 같을 경우
        over_all_sentence_1 = ['a similar number of'] # ??????? 확인할 것
    else:
        pass
        
        
    ext_setting_sim_words_key_list = []
    for i in filtered_setting_text__:
        ext_setting_sim_words_key = model.most_similar_cosmul(i) # 모델적용
        ext_setting_sim_words_key_list.append.ext_setting_sim_words_key
    setting_total_count = len(filtered_setting_text) # 중복이 제거되지 않은 에세이 총 문장에 사용된 setting 표현 수
    setting_count_ = len(filtered_setting_text__) # 중복제거된 setting표현 총 수
        
    result_setting_words_ratio = round(setting_total_count/total_words * 100, 2)
    #return result_setting_words_ratio
    
    ##### Overall Emphasis on Setting : 그래프 표현 부분. #####
    # Setting Indicators 계산으로 문장 전체에 사용된 총 academic표현 값과 합격한 학생들의 academic 평균값을 비교하여 비율로 계산
    # Yours essay 부분
    # setting_total_count : Setting Indicators - Yours essay 부분으로, 중복이 제거되지 않은 에세이 총 문장에 사용된 setting 표현 수
    # setting_total_sentences_number_re : academic 단어가 포함된 총 문장 수 ---> 그래프 표현 부분 * PPT 14page 참고
    ###############################################################################################
    group_total_cnt  = 70 # group_total_cont # Admitted Case Avg. 부분으로 합격학생들의 academic단어 평균값(임의 입력, 계산해서 입력해야 함)
    group_total_setting_descriptors = 20 # Setting Descriptors 합격학생들의 academic 문장수 평균값
    ###############################################################################################
    
    
    # 결과해석 (!! return 값 순서 바꾸면 안됨 !! 만약 값을 추가하려면 맨 뒤에부터 추가하도록! )
    # 0. result_setting_words_ratio : 전체 문장에서 academic관련 단어의 사용비율(포함비율)
    # 1. total_sentences : 총 문장 수
    # 2. total_words : 총 단어 수
    # 3. setting_total_count : # 개인 에세이 중복이 제거되지 않은 에세이 총 문장에 사용된 setting 표현'단어' 수 -----> 그래프로 표현 * PPT 14page 참고
    # 4. setting_count_ : # 중복제거된 setting표현 총 수
    # 5. ext_setting_sim_words_key_list : academic설정과 유사한 단어들 추출
    # 6. totalSettingSentences : academic 단어가 포함된 모든 문장을 추출
    # 7. setting_total_sentences_number_re : 개인 에세이 academic 단어가 포함된 총 '문장' 수 ------> 그래프로 표현 * PPT 14page 참고
    # 8. over_all_sentence_1 : 문장생성 
    # 9. tot_setting_words : 총 문장에서 academic 단어 추출  ---- 웹에 표시할 부분임
    # 10. group_total_cnt : # Admitted Case Avg. 부분으로 합격학생들의 academic'단어' 평균값 ---> 그래프로 표현 * PPT 14page 참고
    # 11. group_total_setting_descriptors : Setting Descriptors 합격학생들의 academic '문장'수 평균값 ---> 그래프로 표현 * PPT 14page 참고
    
    return result_setting_words_ratio, total_sentences, total_words, setting_total_count, setting_count_, ext_setting_sim_words_key_list, totalSettingSentences, setting_total_sentences_number_re, over_all_sentence_1, tot_setting_words, group_total_cnt, group_total_setting_descriptors


###### run #######

# 입력


input_text = """Bloomington Normal is almost laughably cliché for a midwestern city. Vast swathes of corn envelop winding roads and the heady smell of BBQ smoke pervades the countryside every summer. Yet, underlying the trite norms of Normal is the prescriptive force of tradition—the expectation to fulfill my role as a female Filipino by playing Debussy in the yearly piano festival and enrolling in multivariable calculus instead of political philosophy.So when I discovered the technical demand of bebop, the triplet groove, and the intricacies of chordal harmony after ten years of grueling classical piano, I was fascinated by the music's novelty. Jazz guitar was not only evocative and creative, but also strangely liberating. I began to explore different pedagogical methods, transcribe solos from the greats, and experiment with various approaches until my own unique sound began to develop. And, although I did not know what would be the 'best' route for me to follow as a musician, the freedom to forge whatever path I felt was right seemed to be exactly what I needed; there were no expectations for me to continue in any particular way—only the way that suited my own desires.While journeying this trail, I found myself at Interlochen Arts Camp the summer before my junior year. Never before had I been immersed in an environment so conducive to musical growth: I was surrounded by people intensely passionate about pursuing all kinds of art with no regard for ideas of what art 'should' be. I knew immediately that this would be a perfect opportunity to cultivate my sound, unbounded by the limits of confining tradition. On the first day of camp, I found that my peer guitarist in big band was another Filipino girl from Illinois. Until that moment, my endeavors in jazz guitar had been a solitary effort; I had no one with whom to collaborate and no one against whom I could compare myself, much less someone from a background mirroring my own. I was eager to play with her, but while I quickly recognized a slew of differences between us—different heights, guitars, and even playing styles—others seemed to have trouble making that distinction during performances. Some even went as far as calling me 'other-Francesca.' Thus, amidst the glittering lakes and musky pine needles of Interlochen, I once again confronted Bloomington's frustrating expectations.After being mistaken for her several times, I could not help but view Francesca as a standard of what the 'female Filipino jazz guitarist' should embody. Her improvisatory language, comping style and even personal qualities loomed above me as something I had to live up to. Nevertheless, as Francesca and I continued to play together, it was not long before we connected through our creative pursuit. In time, I learned to draw inspiration from her instead of feeling pressured to follow whatever precedent I thought she set. I found that I grew because of, rather than in spite of, her presence; I could find solace in our similarities and even a sense of comfort in an unfamiliar environment without being trapped by expectation. Though the pressure to conform was still present—and will likely remain present in my life no matter what genre I'm playing or what pursuits I engage in—I learned to eschew its corrosive influence and enjoy the rewards that it brings. While my encounter with Francesca at first sparked a feeling of pressure to conform in a setting where I never thought I would feel its presence, it also carried the warmth of finding someone with whom I could connect. Like the admittedly trite conditions of my hometown, the resemblances between us provided comfort to me through their familiarity. I ultimately found that I can embrace this warmth while still rejecting the pressure to succumb to expectations, and that, in the careful balance between these elements, I can grow in a way that feels both like discove"""

result = GeneralAcademicKnowledge(input_text)

print('셋팅 결과 : ', result)







