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
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
stop = stopwords.words('english')
import spacy
nlp = spacy.load('en_core_web_lg')


# 입력문장의 토픽을 추출한다. 대학, 전공관련 정보, 입력한 에세이 등의 문맥상 토픽을 추출하여 다양한 분석에 적용할 수 있음
def extractTopicByLDA(text):
    essay_input_corpus = str(text) #문장입력
    essay_input_corpus = essay_input_corpus.lower()#소문자 변환
    #print('essay_input_corpus :', essay_input_corpus)

    sentences  = sent_tokenize(essay_input_corpus) #문장 토큰화 > 문장으로 구분
    total_sentences = len(sentences)#토큰으로 처리된 총 문장 수
    total_words = len(word_tokenize(essay_input_corpus))# 총 단어수
    #print(total_words)
    split_sentences = []
    for sentence in sentences:
        processed = re.sub("[^a-zA-Z]"," ", sentence)
        words = processed.split()
        split_sentences.append(words)
    #print(split_sentences)

    lemmatizer = WordNetLemmatizer()
    preprossed_sent_all = []
    for i in split_sentences:
        preprossed_sent = []
        for i_ in i:
            if i_ not in stop: #remove stopword
                lema_re = lemmatizer.lemmatize(i_, pos='v') #표제어 추출, 동사는 현재형으로 변환, 3인칭 단수는 1인칭으로 변환
                if len(lema_re) > 3: # 단어 길이가 3 초과단어만 저장(길이가 3 이하는 제거)
                    preprossed_sent.append(lema_re)
        preprossed_sent_all.append(preprossed_sent)

    # 역토큰화
    detokenized = []
    for i_token in range(len(preprossed_sent_all)):
        for tk in preprossed_sent_all:
            t = ' '.join(tk)
            detokenized.append(t)
            
    # print(detokenized)

    #  TF-IDF 행렬로 변환
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(stop_words='english', max_features=None)
    X = vectorizer.fit_transform(detokenized)
    # print(X.shape)

    # LDA 적용
    from sklearn.decomposition import LatentDirichletAllocation
    lda_model = LatentDirichletAllocation(n_components=10, learning_method='online', random_state=777, max_iter=1)
    lda_top = lda_model.fit_transform(X)
    # print(lda_model.components_)
    # print(lda_model.components_.shape)

    terms = vectorizer.get_feature_names() 
    # 단어 집합. 1,000개의 단어가 저장되어있음.
    topics_ext = []
    def get_topics(components, feature_names, n=5):
        for idx, topic in enumerate(components):
            #print("Topic %d :" % (idx+1), [(feature_names[i], topic[i].round(2)) for i in topic.argsort()[:-n -1:-1]])
            topics_ext.append([(feature_names[i], topic[i].round(2)) for i in topic.argsort()[:-n -1:-1]])

    topics_ext.append(get_topics(lda_model.components_, terms))
    # 리스트에 빈 값이 있다면 필터링하자    
    topics_ext = list(filter(None, topics_ext))

    # 단어만 추출하자
    result_ = []
    cnt = 0
    cnt_ = 0
    for ittm in range(len(topics_ext)):
        #print('ittm:', ittm)
        cnt_ = 0
        for t in range(len(topics_ext[ittm])-1):
            
            #print('t:', t)
            add = topics_ext[ittm][cnt_][0]
            result_.append(add)
            #print('result_:', result_)
            cnt_ += 1
    #중복값을 제거하자
    result_fin = list(set(result_))

    # 최종 결과값 반환  ---> 이것은 입력한 문장에 해당하는 토픽을 모두 추출한 단어이다. 
    ##### 활용방안 ####
    # 입력문장의 토픽을 추출한다. 대학, 전공관련 정보, 입력한 에세이 등의 문맥상 토픽을 추출하여 다양한 분석에 적용할 수 있음
    return result_fin


# 본격 분선 코드
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

    #유사한 단어 추출을 위한 모델 설계 완료  --> 비교하고자하는 lexicon이 충분하지 않기 때문에 문맥상에서 유사한 단어도 추출하여 적용해 보자는 의미로 구현하였음
    skip_gram = 1
    workers = multiprocessing.cpu_count()
    bigram_transformer = Phrases(split_sentences)
    model = gensim.models.word2vec.Word2Vec(bigram_transformer[split_sentences], workers=workers, sg=skip_gram, min_count=1)
    model.train(split_sentences, total_examples=sum([len(sentence) for sentence in sentences]), epochs=500)
    
    #academic을 표현하는 단어들을 리스트에 넣어서 필터로 만들고
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

    #소문자 변환하여 정확한 비교를 한다. 대소문자 모두 적용한 글을 합쳐서 필터로 만들거임
    lower_academic_words = []
    for a_itm in academic_word_list:
        lower_a_itm = a_itm.lower()
        lower_academic_words.append(lower_a_itm)

    academic_words_filter_list = academic_word_list + lower_academic_words # 대소문자 모두 포함한 필터 제작
    #print('academic_words_filter_list :' , academic_words_filter_list)
    ####문장에 ords_filter_list의 단어들이 있는지 확인하고, 있다면 유사단어를 추출한다.

    #입력한 에세이처리
    #우선 토큰화한다.
    retokenize = RegexpTokenizer("[\w]+") #줄바꿈 제거하여 한줄로 만들고

    token_input_text = retokenize.tokenize(essay_input_corpus)
    # print (token_input_text) #토큰화 처리 확인.. 토큰들이 리스트에 담김
    # 리트스로 정리된 개별 토큰을 char_list와 비교해서 존재하는 것만 추출한다.
    filtered_academic_words = []
    for k in token_input_text:
        for j in academic_words_filter_list:
            if k == j:
                filtered_academic_words.append(j)
    
    # print (filtered_chr_text) # 유사단어 비교 추출 완료, 겹치는 단어는 제거하자.
    
    filtered_academic_words_ = set(filtered_academic_words) #중복제거
    filtered_academic_words__ = list(filtered_academic_words_) #다시 리스트로 변환
    # print (filtered_setting_text__) # 중복값 제거 확인
    
    # 문장내 모든 academic 단어 추출
    tot_academic_words = filtered_academic_words__
    
    # academic 단어가 포함된 문장을 찾아내서 추출하기
    # if academic단어가 문장에 있다면, 그 문장을 추출(.로 split한 문장 리스트)해서 리스트로 저장한다.
    
    # print('sentences: ', sentences) # .로 구분된 전체 문장
    
    sentence_to_words = word_tokenize(essay_input_corpus) # 총 문장을 단어 리스트로 변환
    # print('sentence_to_words:', sentence_to_words)
    
    # academic단어가 포함된 문장을 찾아내서 추출
    extrace_sentence_and_setting_words = [] # 이것은 "문장", 'academic단어' ... 합쳐서 리스트로 저장
    extract_only_sentences_include_setting_words = [] # academic 단어가 포함된 문장만 리스트로 저장
    for sentence in sentences: # 문장을 하나씩 꺼내온다.
        for item in tot_academic_words: # academic단어를 하나씩 꺼내온다.
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
        
    # 앞서 추출한 Academic words와 유사한 의미의 단어를 모두 추출하여 Academic 관련 포함율 계산
    ext_setting_sim_words_key_list = []
    for i in filtered_academic_words__:
        ext_setting_sim_words_key = model.wv.most_similar(i)# 모델적용
        ext_setting_sim_words_key_list.append(ext_setting_sim_words_key)
    setting_total_count = len(filtered_academic_words) # 중복이 제거되지 않은 에세이 총 문장에 사용된 Academic 표현 수
    setting_count_ = len(filtered_academic_words__) # 중복제거된 Academic 표현 총 수
        
    result_setting_words_ratio = round(setting_total_count/total_words * 100, 2)
 




    ## 문장 유사도 분석 - 에세이에 아카데믹 단어들이 얼마나 포함되어 있는지 추출하여 비교분석해보자, 그리고 합격생기준과 비교하여 수준평가할 것
    # 에세이에 포함된 핵심 키워드 추출 by LDA
    topicEssay = extractTopicByLDA(text)
    print('topicEssay:', topicEssay)



    
    ###############################################################################################
    group_total_cnt  = 70 # group_total_cont # Admitted Case Avg. 부분으로 합격학생들의 academic단어 평균값(임의 입력, 계산해서 입력해야 함)
    group_total_setting_descriptors = 20 # academic Descriptors 합격학생들의 academic 문장수 평균값
    ###############################################################################################
    
    
    # 결과해석 (!! return 값 순서 바꾸면 안됨 !! 만약 값을 추가하려면 맨 뒤에부터 추가하도록! )
    # 0. result_setting_words_ratio : 전체 문장에서 academic관련 단어의 사용비율(포함비율)
    # 1. total_sentences : 총 문장 수
    # 2. total_words : 총 단어 수
    # 3. setting_total_count : # 개인 에세이 중복이 제거되지 않은 에세이 총 문장에 사용된 academic 표현'단어' 수 
    # 4. setting_count_ : # 중복제거된 academic표현 총 수
    # 5. ext_setting_sim_words_key_list : academic설정과 유사한 단어들 추출
    # 6. totalSettingSentences : academic 단어가 포함된 모든 문장을 추출
    # 7. setting_total_sentences_number_re : 개인 에세이 academic 단어가 포함된 총 '문장' 수 
    # 8. tot_academic_words : 총 문장에서 academic 단어 추출  
    # 9. group_total_cnt : # Admitted Case Avg. 부분으로 합격학생들의 academic'단어' 평균값 
    # 10. group_total_setting_descriptors : Setting Descriptors 합격학생들의 academic '문장'수 평균값 
    
    return result_setting_words_ratio, total_sentences, total_words, setting_total_count, setting_count_, ext_setting_sim_words_key_list, totalSettingSentences, setting_total_sentences_number_re, tot_academic_words, group_total_cnt, group_total_setting_descriptors


###### run #######

# 입력


input_text = """This past summer, I had the privilege of participating in the University of Notre Dame’s Research Experience for Undergraduates (REU) program . Under the mentorship of Professor Wendy Bozeman and Professor Georgia Lebedev from the department of Biological Sciences, my goal this summer was to research the effects of cobalt iron oxide cored (CoFe2O3) titanium dioxide (TiO2) nanoparticles as a scaffold for drug delivery, specifically in the delivery of a compound known as curcumin, a flavonoid known for its anti-inflammatory effects. As a high school student trying to find a research opportunity, it was very difficult to find a place that was willing to take me in, but after many months of trying, I sought the help of my high school biology teacher, who used his resources to help me obtain a position in the program. Using equipment that a high school student could only dream of using, I was able to map apoptosis (programmed cell death) versus necrosis (cell death due to damage) in HeLa cells, a cervical cancer line, after treating them with curcumin-bound nanoparticles. Using flow cytometry to excite each individually suspended cell with a laser, the scattered light from the cells helped to determine which cells were living, had died from apoptosis or had died from necrosis. Using this collected data, it was possible to determine if the curcumin and/or the nanoparticles had played any significant role on the cervical cancer cells. Later, I was able to image cells in 4D through con-focal microscopy. From growing HeLa cells to trying to kill them with different compounds, I was able to gain the hands-on experience necessary for me to realize once again why I love science.	Living on the Notre Dame campus with other REU students, UND athletes, and other summer school students was a whole other experience that prepared me for the world beyond high school. For 9 weeks, I worked, played and bonded with the other students, and had the opportunity to live the life of an independent college student. Along with the individually tailored research projects and the housing opportunity, there were seminars on public speaking, trips to the Fermi National Accelerator Laboratory, and one-on-one writing seminars for the end of the summer research papers we were each required to write. By the end of the summer, I wasn’t ready to leave the research that I was doing. While my research didn’t yield definitive results for the effects of curcumin on cervical cancer cells, my research on curcumin-functionalized CoFe2O4/TiO2 core-shell nanoconjugates indicated that there were many unknown factors affecting the HeLa cells, and spurred the lab to expand their research into determining whether or not the timing of the drug delivery mattered and whether or not the position of the binding site of the drugs would alter the results. Through this summer experience, I realized my ambition to pursue a career in research. I always knew that I would want to pursue a future in science, but the exciting world of research where the discoveries are limitless has captured my heart. This school year, the REU program has offered me a year-long job, and despite my obligations as a high school senior preparing for college, I couldn’t give up this offer, and so during this school year, I will be able to further both my research and interest in nanotechnology. """
result = GeneralAcademicKnowledge(input_text)

print('GeneralAcademicKnowledge 분석결과: ', result)







