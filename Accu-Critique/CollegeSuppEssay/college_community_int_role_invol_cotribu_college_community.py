# Prompt Type: College community: intended role, involvement, and contribution in college community

# ppt p.27

# Community Oriented Keywords
# College Community Fit
# Prompt Oriented Sentiments
# Initiative & Engagement


# Community Oriented Keywords
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


from intellectualEngagement import intellectualEnguagement

### sentence_simility.py ###
from sentence_similarity import sent_sim_analysis_with_bert_summarizer




# 토픽 추출
def getTopics(essay_input):

    # step_1
    essay_input_corpus = str(essay_input) #문장입력
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

    # step_2
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

    # step_3 : 역토큰화로 추출된 단어를 문자별 단어리스트로 재구성
    # 역토큰화
    detokenized = []
    for i_token in range(len(preprossed_sent_all)):
        for tk in preprossed_sent_all:
            t = ' '.join(tk)
            detokenized.append(t)

    # step_4 : TF-IDF 행렬로 변환
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(stop_words='english', max_features=None)
    X = vectorizer.fit_transform(detokenized)

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
    # 토픽 추출 시작
    topics_ext.append(get_topics(lda_model.components_, terms))
    # 토픽중 없는 값 제거 필터링
    topics_ext = list(filter(None, topics_ext))


    # 최종 토필 추축
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

    #추출된 토픽들의 중복값 제거
    result_fin = list(set(result_))

    #### 토픽 추출한 결과 반환 ####
    return result_fin




## 연관어 불어오기 ##
#유사단어를 추출하여 리스트로 반환
def get_sim_words(text):

    essay_input_corpus = str(text) #문장입력
    essay_input_corpus = essay_input_corpus.lower()#소문자 변환

    sentences  = sent_tokenize(essay_input_corpus) #문장 토큰화
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

    #비교분석 단어들을 리스트에 넣어서 필터로 만들고

    import pickle

    # load
    with open('./data/data_communityWords.pickle', 'rb') as f:
        words_list = pickle.load(f)

    ####문장에 list의 단어들이 있는지 확인하고, 있다면 유사단어를 추출한다.
    
    #우선 토큰화한다.
    retokenize = RegexpTokenizer("[\w]+") #줄바꿈 제거하여 한줄로 만들고
    token_input_text = retokenize.tokenize(essay_input_corpus)
    #print (token_input_text) #토큰화 처리 확인.. 토큰들이 리스트에 담김
    #리트스로 정리된 개별 토큰을 char_list와 비교해서 존재하는 것만 추출한다.
    filtered_chr_text = []
    for k in token_input_text:
        for j in words_list:
            if k == j:
                filtered_chr_text.append(j)
    
    #print (filtered_chr_text) # 유사단어 비교 추출 완료, 겹치는 단어는 제거하자.
    
    filtered_chr_text_ = set(filtered_chr_text) #중복제거
    filtered_chr_text__ = list(filtered_chr_text_) #다시 리스트로 변환

    ext_sim_words_key = filtered_chr_text__

    return ext_sim_words_key



####  Initiative & Engagement ###
def initiative_engagement(essay_input):
      #입력한 글을 모두 단어로 쪼개로 리스트로 만들기 - 
    essay_input_corpus_ = str(essay_input) #문장입력
    essay_input_corpus_ = essay_input_corpus_.lower()#소문자 변환

    sentences_  = sent_tokenize(essay_input_corpus_) #문장단위로 토큰화(구분)되어 리스에 담김

    # 문장을 토크큰화하여 해당 문장에 Action Verbs가 있는지 분석 부분 코드임 ---> 뒤에서 나옴 아래 777로 표시된 코드부분에서 sentences_ 값 재활용

    split_sentences_ = []
    for sentence in sentences_:
        processed = re.sub("[^a-zA-Z]"," ", sentence)
        words = processed.split()
        split_sentences_.append(words)
        
    # 입력한 문장을 모두 리스트로 변환
    input_text_list = [y for x in split_sentences_ for y in x] # 이중 리스트 Flatten

    #리스로 변환된 값 확인
    #
    #데이터 불러오기
    data_action_verbs = pd.read_csv('./data/actionverbs.csv')
    data_ac_verbs_list = data_action_verbs.values.tolist()
    verbs_list = [y for x in data_ac_verbs_list for y in x]

    def actionverb_sim_words(text):

        essay_input_corpus = str(text) #문장입력
        essay_input_corpus = essay_input_corpus.lower()#소문자 변환

        sentences  = sent_tokenize(essay_input_corpus) #문장 토큰화
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

        # ACTION VERBS 표현하는 단어들을 리스트에 넣어서 필터로 만들고
        ##################################################
        # verbs_list

        ####문장에 list의 단어들이 있는지 확인하고, 있다면 유사단어를 추출한다.
        
        #우선 토큰화한다.
        retokenize = RegexpTokenizer("[\w]+") #줄바꿈 제거하여 한줄로 만들고
        token_input_text = retokenize.tokenize(essay_input_corpus)
        #print (token_input_text) #토큰화 처리 확인.. 토큰들이 리스트에 담김
        #리트스로 정리된 개별 토큰을 char_list와 비교해서 존재하는 것만 추출한다.
        filtered_chr_text = []
        for k in token_input_text:
            for j in verbs_list:
                if k == j:
                    filtered_chr_text.append(j)
        
        #print (filtered_chr_text) # 유사단어 비교 추출 완료, 겹치는 단어는 제거하자.
        
        filtered_chr_text_ = set(filtered_chr_text) #중복제거
        filtered_chr_text__ = list(filtered_chr_text_) #다시 리스트로 변환
        #print (filtered_chr_text__) # 중복값 제거 확인
        
#         for i in filtered_chr_text__:
#             ext_sim_words_key = model.most_similar_cosmul(i) #모델적용
        
#         char_total_count = len(filtered_chr_text) # 중복이 제거되지 않은 에세이 총 문장에 사용된 표현 수
#         char_count_ = len(filtered_chr_text__) #중복제거된  표현 총 수
            
#         result_char_ratio = round(char_total_count/total_words * 100, 2)
        
#         df_conf_words = pd.DataFrame(ext_sim_words_key, columns=['words','values']) #데이터프레임으로 변환
#         df_r = df_conf_words['words'] #words 컬럼 값 추출
#         ext_sim_words_key = df_r.values.tolist() # 유사단어 추출

        #return result_char_ratio, total_sentences, total_words, char_total_count, char_count_, ext_sim_words_key
        ext_sim_words_key = filtered_chr_text__
        return ext_sim_words_key


    # 입력문장에서 맥락상 Aciton Verbs와 유사한 의미의 단어를 추출
    ext_action_verbs = actionverb_sim_words(essay_input)

    #########################################################################
    # 8.이제 입력문장에서 사용용된 Action Verbs 단어를 비교하여 추출해보자.

    # Action Verbs를 모두 모음(직접적인 단어, 문맥상 유사어 포함)
    all_ac_verbs_list = verbs_list + ext_action_verbs

    #입력한 리스트 값을 하나씩 불러와서 데이터프레임에 있는지 비교 찾아내서 해당 점수를 가져오기
    graph_calculation_list =[0]
    get_words__ = []
    counter= 0
    for h in input_text_list: #데이터프레임에서 인덱스의 값과 비교하여
        if h in all_ac_verbs_list: #df에 특정 단어가 있다면, 해당하는 컬럼의 값을 가져오기
            get_words__.append(h) # 동일하면 저장하기
            #print('counter :', counter)
            graph_calculation_list.append(round(graph_calculation_list[counter]+2,2))
            #print ('graph_calculation_list[counter]:', graph_calculation_list[counter])
            #graph_calculation_list.append(random.randrange(1,10))
            counter += 1
        else: #없다면
            #print('counter :', counter)
            graph_calculation_list.append(round(graph_calculation_list[counter]-0.1,2)) 
            counter += 1
    #문장에 Action Verbs 추출확인
    #get_words__ 


    def divide_list(l, n): 
        # 리스트 l의 길이가 n이면 계속 반복
        for i in range(0, int(len(l)), int(n)): 
            yield l[i:i + int(n)] 
        
    # 한 리스트에 몇개씩 담을지 결정 = 20개씩

    n = len(graph_calculation_list)/20

    result_gr = list(divide_list(graph_calculation_list, n))

    gr_cal = []
    for regr in result_gr:
        avg_gr = sum(regr,0.0)/len(regr) #묶어서 평균을 내고 
        gr_cal.append(abs(round(avg_gr,2))) #절대값을 전환해서


    graph_calculation_list = gr_cal  ## 그래프를 위한 최종결과 계산 후, 이것을 딕셔너리로 반환하여 > 그래프로 표현하기
    #########################################################################
    # 9. 그래프 출력 : 문장 전체를 단어로 분리하고, Action verbs가 사용된 부분을 그래프로 표시

    # 전체 글에서 Action verbs가 언급된 부분을 리스트로 계산
    # graph_calculation_list 

    #그래프로 표시됨
    # plt.plot(graph_calculation_list)
    # plt.xlabel('STORY')
    # plt.ylabel('ACTON VERBS')
    # plt.title('USAGE OF ACTION VERBS ANALYSIS')
    # plt.legend(['action verbs'])
    # plt.show()

    #########################################################################
    # 10.입력한 에세이 문장에서 Action Verbs가 얼마나 포함되어 있는지 포함비율 분석
    action_verbs_ratio = round(len(get_words__)/len(input_text_list) *100, 3)

    # print ("ACTION VERBS RATIO :", action_verbs_ratio )
    return action_verbs_ratio


def lackigIdealOverboard(group_mean, personal_value): # group_mean: 1000명 평균, personal_value: 개인값
    ideal_mean = group_mean
    one_ps_char_desc = personal_value
    #최대, 최소값 기준으로 구간설정. 구간비율 30% => 0.3으로 설정
    min_ = int(ideal_mean-ideal_mean*0.6)
    #print('min_', min_)
    max_ = int(ideal_mean+ideal_mean*0.6)
    #print('max_: ', max_)
    div_ = int(((ideal_mean+ideal_mean*0.6)-(ideal_mean-ideal_mean*0.6))/3)
    #print('div_:', div_)

    #결과 판단 Lacking, Ideal, Overboard
    cal_abs = abs(ideal_mean - one_ps_char_desc) # 개인 - 단체 값의 절대값계산

    #print('cal_abs 절대값 :', cal_abs)
    compare7 = (one_ps_char_desc + ideal_mean)/6
    compare6 = (one_ps_char_desc + ideal_mean)/5
    compare5 = (one_ps_char_desc + ideal_mean)/4
    compare4 = (one_ps_char_desc + ideal_mean)/3
    compare3 = (one_ps_char_desc + ideal_mean)/2
    # print('compare7 :', compare7)
    # print('compare6 :', compare6)
    # print('compare5 :', compare5)
    # print('compare4 :', compare4)
    # print('compare3 :', compare3)

    if one_ps_char_desc > ideal_mean: # 개인점수가 평균보다 클 경우는 overboard
        if cal_abs > compare3: # 37 개인점수가 개인평균차의 절대값보다 클 경우, 즉 차이가 많이 날경우
            #print("Overboard: 2")
            result = 2 #overboard
            score = 1
        elif cal_abs > compare4: # 28
            #print("Overvoard: 2")
            result = 2
            score = 2
        elif cal_abs > compare5: # 22
            #print("Overvoard: 2")
            result = 2
            score = 3
        elif cal_abs > compare6: # 18
            #print("Overvoard: 2")
            result = 2
            score = 4
        else:
            #print("Ideal: 1")
            result = 1
            score = 5
    elif one_ps_char_desc < ideal_mean: # 개인점수가 평균보다 작을 경우 lacking
        if cal_abs > compare3: # 37 개인점수가 개인평균차의 절대값보다 클 경우, 즉 차이가 많이 날경우
            #print("Lacking: 2")
            result = 0
            score = 1
        elif cal_abs > compare4: # 28
            #print("Lacking: 2")
            result = 0
            score = 2
        elif cal_abs > compare5: # 22
            #print("Lacking: 2")
            result = 0
            score = 3
        elif cal_abs > compare6: # 18
            #print("Lacking: 2")
            result = 0
            score = 4
        else:
            #print("Ideal: 1")
            result = 1
            score = 5
            
    else: # 같으면 ideal 이지. 가장 높은 점수를 줄 것
        #print("Ideal: 1")
        result = 1
        score = 5

    # 최종 결과 5점 척도로 계산하기
    if score == 5:
        result_ = 'Supurb'
    elif score == 4:
        result_ = 'Strong'
    elif score == 3:
        result_ = 'Good'
    elif score == 2:
        result_ = 'Mediocre'
    else: #score = 1
        result_ = 'Lacking'

    return result_





# key topic을 추출하고, achievement, pride. proud 혹은 관련 단어와 얼마나 일치하는지 비율 계산하기ㅐ
def pmtOrientedKeywords(essay_input):
    get_ext_topics_re = getTopics(essay_input) # 토픽 추출
    ext_topics_num = len(get_ext_topics_re) # 추출한 토픽 수
    # achievement - 유사단어를 문장에서 추출하여 반환
    get_sim_words_result = get_sim_words(essay_input)

    # 비교하기
    cnt = 0
    pmt_ori_keyword =[]
    for i in get_sim_words_result:
        if i in get_ext_topics_re:
            pmt_ori_keyword.append(i)
            cnt += 1
        else:
            pass
    # get ration
    if cnt == 0: # 토픽에서 비교단어와 일치하는 단어가 없다면(카운트가 0이니까)
        print('Achievement you are proud of : Prompt Oriented Keywords 일치율 0')
        get_words_ratio = 0
    else: # 1개 이상이면 일치율 계산하기
        get_words_ratio =  round(cnt / ext_topics_num * 100, 2)

    ##################################################################################
    ############## 합격학생의 평균값 입력하기 (임의 임력, 나중에 계산해서 입력할 것) ################
    # 만약 테스트 결과 출력값이 예상과 다르다면 아래 점수(숫자) 를 변경하여 조율할 것!
    admitted_essay_mean_value_of_promptOrientedSentiments = 20
    ##################################################################################

    # 비교계산 시작 - 상대적 비교로 '합격학생(admitted_essay_mean_value_of_promptOrientedSentiments)에 비교(get_words_ratio)해서 값이 가까우면 높은점수, 멀어지면 낮은점수를 줄 것)
    fin_result = lackigIdealOverboard(admitted_essay_mean_value_of_promptOrientedSentiments, get_words_ratio)


    ##################################################################################
    ############## 합격학생의 평균값 입력하기 (임의 임력, 나중에 계산해서 입력할 것) ################
    # 만약 테스트 결과 출력값이 예상과 다르다면 아래 점수(숫자) 를 변경하여 조율할 것!
    admitted_essay_mean_fin_initiative_enguagement_ratio = 5
    ##################################################################################
    #initiative and enguagement ratio
    fin_initiative_enguagement_ratio = initiative_engagement(essay_input)
    initiative_enguagement_result = lackigIdealOverboard(admitted_essay_mean_fin_initiative_enguagement_ratio, fin_initiative_enguagement_ratio)


    # in_result : 최종 결과로 5가지 척도로 계산됨
    # get_words_ratio : 입력에세이의 토픽과 비교할 단어가 얼마나 일치하는지에 대한 비율 계산 결과
    # pmt_ori_keyword : Prompt Oriented Keywords 추출
    # fin_initiative_enguagement_ratio : initiative_enguagement 가 에세이이 포함된 비율
    # initiative_enguagement_result : initiative_enguagement가 합격생 평균에 비교하여 얻은 최종 값

    # 단, 여기서 원래 계산하기로 했던 Prompt Oriented Sentiments (40%)의 적용값은 collegeSupp.py에서 계산해서 웹에 표시할거임
    # 그리고 각 항목당 비율값도 colleSupp.py에서 계산해서 적용할거임 ( Prompt Oriented Keywords (20%), Prompt Oriented Sentiments (40%), Initiative & Engagement (30%))

    return fin_result, get_words_ratio, pmt_ori_keyword, fin_initiative_enguagement_ratio, initiative_enguagement_result



# 실행 #


# 최종계산
# pmt_ori_key_re = pmtOrientedKeywords(essay_input)
# print('Prompt Oriented Keywords 값 계산 결과:', pmt_ori_key_re[0])
# print('Prompt Oriented Keywords - 입력에세이의 토픽과 비교할 단어가 얼마나 일치하는지에 대한 비율 계산 결과 :', pmt_ori_key_re[1])
# print('Prompt Oriented Keywords 단어들(웹사이트에 표시) :', pmt_ori_key_re[2])
# print('관련 단어가 에세이에 포함된 비율 :', pmt_ori_key_re[3])
# print('합격생 평균에 비교하여 얻은 최종 값 :', pmt_ori_key_re[4])


### 결과 ###
# Prompt Oriented Keywords 값 계산 결과: Mediocre
# Prompt Oriented Keywords - 입력에세이의 토픽과 비교할 단어가 얼마나 일치하는지에 대한 비율 계산 결과 : 52.94
# Prompt Oriented Keywords 단어들(웹사이트에 표시) : ['able', 'image', 'year', 'school', 'cancer', 'microscopy', 'write', 'place', 'opportunity', 'research', 'significant', 'role', 'career', 'data', 'high', 'possible', 'help', 'death']
# 관련 단어가 에세이에 포함된 비율 : 7.531
# 합격생 평균에 비교하여 얻은 최종 값 : Strong



# College Community Fit
def Fit(select_pmt_type, select_college, select_college_dept, select_major, coll_supp_essay_input_data):
    re_coll_n_dept_fit = sent_sim_analysis_with_bert_summarizer(select_pmt_type, select_college, select_college_dept, select_major, coll_supp_essay_input_data)

    # major fit result  --> 이 값을 intellectualEngagement.py의 def intellectualEnguagement(essay_input, input_mjr_score)에 입력해서 결과를 다시 가져옴
    mjr_fit_result_final = re_coll_n_dept_fit[7]

    return mjr_fit_result_final


essay_input = """ I inhale deeply and blow harder than I thought possible, tech/engineering pushing the tiny ember from its resting place on the candle out into the air. mit women's technology program (wtp) The room erupts around me, and 'Happy Birthday!' cheers echo through the halls. It's time to make a wish. In my mind, that new Limited Edition Deluxe Ben 10 watch will soon be mine. My parents and the aunties and uncles around me attempt to point me in a different direction. 'Wish that you get to go to the temple every day when you're older! Wish that you memorize all your Sanskrit texts before you turn 6! Wish that you can live in India after college!' My ears listen, but my mind tunes them out, as nothing could possibly compare to that toy watch! What I never realized on my third birthday is that those wishes quietly tell the story of how my family hopes my life will play out. In this version of my life, there wasn't much room for change, personal growth, or 'rocking the boat.' A vital aspect of my family's cultural background is their focus on accepting things as they are. Growing up, I was discouraged from questioning others or asking questions that didn't have definitive yes or no answers. If I innocently asked my grandma why she expected me to touch her feet, my dad would grab my hand in a sudden swoop, look me sternly in the eye, and tell me not to disrespect her like that again. At home, if I mentioned that I had tried eggs for breakfast at a friend's house, I'd be looked at like I had just committed a felony for eating what my parents considered meat. If I asked the priest at the temple why he had asked an Indian man and his white wife to leave, I'd be met with a condescending glare and told that I should also leave for asking such questions.In direct contrast, my curiosity was invited and encouraged at school. After an environmental science lesson, I stayed for a few minutes after class to ask my 4th-grade science teacher with wide eyes how it was possible that Niagara Falls doesn't run out of flowing water. Instead of scolding me for asking her a 'dumb question,' she smiled and explained the intricacy of the water cycle. Now, if a teacher mentions that we'll learn about why a certain proof or idea works only in a future class, I'll stay after to ask more or pour through an advanced textbook to try to understand it. While my perspective was widening at school, the receptiveness to raising complex questions at home was diminishing. After earning my driver's license, I registered as an organ donor. My small checkmark on a piece of paper led to an intense clash between my and my parents' moral platform. I wanted to ensure that I positively contributed to society, while my parents believed that organ donation was an unfamiliar and unnecessary cultural taboo. I would often ask for clarity or for reasons that supported their ideologies. Their response would usually entail feeling a deep, visceral sense that traditions must be followed exactly as taught, without objection. Told in one language to keep asking questions and in another to ask only the right ones, I chose exploring questions that don't have answers, rather than accepting answers that don't get questioned. When it comes to the maze of learning, even when I take a wrong turn and encounter roadblocks that are meant to stop me, I've learned to climb over them and keep moving forward. My curiosity strengthens with each hurdle and has expanded into a pure love of learning new things. I've become someone who seeks to understand things at a fundamental level and who finds excitement in taking on big questions that have yet to be solved. I'm no longer afraid to rock the boat. "},{"index":1,"personal_essay":"Ever since I first held a small foam Spiderman basketball in my tiny hands and watched my idol Kobe Bryant hit every three-pointer he attempted, I've wanted to understand and replicate his flawless jump shot. As my math education progressed in school, I began to realize I had the tools to create a perfect shot formula. After learning about variables for the first time in 5th grade Algebra, I began to treat each aspect of Kobe's jump shot as a different variable, each combination of variables resulting in a unique solution. While in 7th-grade geometry, I graphed the arc of his shot, and after learning about quadratic equations in 8th grade, I expressed his shot as a parabolic function that would ensure a swish when shooting from any spot. After calculus lessons in 10th and 11th grade, I was excited to finally solve for the perfect velocity and acceleration needed on my release. At Brown, I hope to explore this intellectual pursuit through a different lens. What if I could maximize the odds of making shots if I understood the science behind one's mental mindset and focus through CLPS 500: Perception and Action? Or use astrophysics to account for drag and gravitational force anywhere in the universe? Or use data science to break down the analytics of the NBA's best shooters? Through the Open Curriculum, I see myself not only becoming a more complete learner, but also a more complete thinker, applying a flexible mindset to any problem I encounter. Brown's Open Curriculum allows students to explore broadly while also diving deeply into their academic pursuits. Tell us about an academic interest (or interests) that excites you, and how you might use the Open Curriculum to pursue it. I've been playing the Mridangam since I was five years old. It's a simple instrument: A wood barrel covered on two ends by goatskin with leather straps surrounding the hull. This instrument serves as a connection between me and one of the most beautiful aspects of my culture: Carnatic music. As a young child, I'd be taken to the temple every weekend for three-hour-long Carnatic music concerts, where the most accomplished teenagers and young adults in our local Indian community would perform. I would watch in awe as the mridangists' hands moved gracefully, flowing across the goatskin as if they weren't making contact, while simultaneously producing sharp rhythmic patterns that never failed to fall on the beat. Hoping to be like these idols on the stage, I trained intensely with my teacher, a strict man who taught me that the simple drum I was playing had thousands of years of culture behind it. Building up from simple strokes, I realized that the finger speed I'd had been awestruck by wasn't some magical talent, it was instead a science perfected by repeated practice."""

"Summer activity", 'Brown', 'Brown_African Studies_dept', 'African Studies', essay_input

#result__ = Fit(select_pmt_type, select_college, select_college_dept, select_major, coll_supp_essay_input_data)
# result__ = Fit("Summer activity", 'Brown', 'Brown_African Studies_dept', 'African Studies', essay_input)
# print(' College Community Fit :', result__)



# Prompt Oriented Sentiments  -- 글속에 감정이 얼마나 표현되어 있는지 분석 - origin (Bert pre trained model 활용)
from transformers import BertTokenizer
from model import BertForMultiLabelClassification
from multilabel_pipeline import MultiLabelPipeline
from pprint import pprint

tokenizer = BertTokenizer.from_pretrained("monologg/bert-base-cased-goemotions-original")
model = BertForMultiLabelClassification.from_pretrained("monologg/bert-base-cased-goemotions-original")

goemotions = MultiLabelPipeline(
    model=model,
    tokenizer=tokenizer,
    threshold=0.3
)

def Prompt_Oriented_Sentiments_analysis(essay_input):
    ########## 여기서는 최초 입력 에세이를 적용한다. input_text !!!!!!!!
    re_text = essay_input.split(".")

    #데이터 전처리 
    def cleaning(datas):

        fin_datas = []

        for data in datas:
            # 영문자 이외 문자는 공백으로 변환
            only_english = re.sub('[^a-zA-Z]', ' ', data)
        
            # 데이터를 리스트에 추가 
            fin_datas.append(only_english)

        return fin_datas

    texts = cleaning(re_text)

    #분석된 감정만 추출
    emo_re = goemotions(texts)

    emo_all = []
    for list_val in range(0, len(emo_re)):
        #print(emo_re[list_val]['labels'],emo_re[list_val]['scores'])
        #mo_all.append((emo_re[list_val]['labels'],emo_re[list_val]['scores'])) #KEY, VALUE만 추출하여 리스트로 저장
        #emo_all.append(emo_re[list_val]['scores'])
        emo_all.append((emo_re[list_val]['labels']))
        
    #추출결과 확인 
    # emo_all

    # ['sadness'],
    #  ['anger'],
    #  ['admiration', 'realization'],
    #  ['admiration', 'disappointment'],
    #  ['love'],
    #  ['sadness', 'neutral'],
    #  ['realization', 'neutral'],
    #  ['neutral'],
    #  ['optimism'],
    #  ['neutral'],
    #  ['excitement'],
    #  ['neutral'],
    #  ['neutral'],
    #  ['caring'],
    #  ['gratitude'],
    #  ['admiration', 'approval'], ...

    from pandas.core.common import flatten #이중리스틀 FLATTEN하게 변환
    flat_list = list(flatten(emo_all))

    # ['neutral',
    #  'neutral',
    #  'sadness',
    #  'anger',
    #  'admiration',
    #  'realization',
    #  'admiration',
    #  'disappointment',


    #중립적인 감정을 제외하고, 입력한 문장에서 다양한 감정을 모두 추출하고 어떤 감정이 있는지 계산해보자
    unique = []
    for r in flat_list:
        if r == 'neutral':
            pass
        else:
            unique.append(r)

    #중립감정 제거 및 유일한 감정값 확인
    #unique
    unique_re = set(unique) #중복제거

    ############################################################################
    # 글에 표현된 감정이 얼마나 다양한지 분석 결과!!!¶
    # print("====================================================================")
    # print("에세이에 표현된 다양한 감정 수:", len(unique_re))
    # print("====================================================================")

    #분석가능한 감정 총 감정 수 - Bert origin model 적용시 28개 감정 추출돰
    total_num_emotion_analyzed = 28

    # 감정기복 비율 계산 !!!
    result_emo_swings =round(len(unique_re)/total_num_emotion_analyzed *100,1) #소숫점 첫째자리만 표현
    # print("문장에 표현된 감정 비율 : ", result_emo_swings)
    # print("====================================================================")

    # 결과해서
    # reslult_emo_swings : 전체 문장에서의 감정 비율 계산
    # unique_re : 에세이에서 분석 추출한 감정   ====> 이것이 중요한 값임
    return result_emo_swings, unique_re
    # (32.1, {'disapproval', 'curiosity', 'optimism', 'approval', 'excitement', 'disappointment', 'love', 'desire', 'realization'})
    


# 이 함수는 두 군데 사용중 1)collegeSuppy.py 2)prompt_oriented_sentments.py, 하지만 나중에 분리해야 함. pmp 별로 개별 계산해야 하기떼문
def select_prompt_type(prompt_type):
    
    if prompt_type == 'Why us':
        pmt_typ = [""" 'Why us' school & major interest (select major, by college & department) """]
        pmt_sentiment = ['Admiration', 'Excitement', 'Pride', 'Realization', 'Curiosity']
    elif prompt_type == 'Intellectual interest':
        pmt_typ = [""" Intellectual interest """]
        pmt_sentiment = ['Curiosity', 'Realization']
    elif prompt_type == 'Meaningful experience & lesson learned':
        pmt_typ = ["Meaningful experience & lesson learned"]
        pmt_sentiment = ['Realization', 'Approval', 'Gratitude', 'Admiration']
    elif prompt_type ==  'Achievement you are proud of':
        pmt_typ = ["Achievement you are proud of"]
        pmt_sentiment = ['Realization', 'Approval', 'Gratitude', 'Admiration', 'Pride', 'Desire', 'Optimism']
    elif prompt_type ==  'Social issues: contribution & solution':
        pmt_typ = ["Social issues: contribution & solution"]
        # 0~16 번째는 에세이 입력의 40% 분석 적용, 그 이후부분은 60% 적용  --> 즉 [:16], [17:] 이렇게 나눌 것
        pmt_sentiment = ['Anger', 'annoyance', 'Fear', 'Disapproval', 'disgust', 'Disappointment','grief', 'nervousness', 'sadness', 'surprise', 'remorse', 'curiosity', 'embarrassment', 'Realization','Approval', 'Gratitude', 'Admiration','Admiration','Approval', 'Caring', 'Joy', 'Gratitude', 'Optimism','relief', 'Realization']
    elif prompt_type ==  'Summer activity':
        pmt_typ = ["Summer activity"]
        pmt_sentiment = ['Pride','Realization','Curiosity','Excitement','Amusement','Caring']
    elif prompt_type ==  'Unique quality, passion, or talent':
        pmt_typ = ["Unique quality, passion, or talent"]
        pmt_sentiment = ['Pride','Excitement','Amusement','Approval','Admiration','Curiosity']
    elif prompt_type ==  'Extracurricular activity or work experience':
        pmt_typ = ["Extracurricular activity or work experience'"]
        pmt_sentiment = ['Pride','Realization','Curiosity','Joy','Excitement','Amusement','Caring','Optimism']
    elif prompt_type ==  'Your community: role and contribution in your community':
        pmt_typ = ["Your community: role and contribution in your community"]
        pmt_sentiment = ['Admiration','Caring','Approval','Pride','Gratitude','Love', 'annoyance', 'joy', 'realization', 'relief']
    elif prompt_type ==  'College community: intended role, involvement, and contribution in college community':
        pmt_typ = ["College community: intended role, involvement, and contribution in college community"]
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



def comp_sentiment_both(essay_input, select_pmt_type):
    # prompt 에 해당하는 sentiments와 관련한 감정과, 입력한 에세이에서 추출한 감정들이 얼마나 일치 비율 계산하기
    # 에세이에서 추출 분석한 감정 리스트
    get_sents_from_essay = Prompt_Oriented_Sentiments_analysis(essay_input)
    # 선택한 해당 prompt의 감정 리스트 : pmt_sent_re
    pmt_sent_etc_re = select_prompt_type(select_pmt_type)
    pmt_sent_re = list(pmt_sent_etc_re[1]) # prompt 에 해당하는 sentiment l
    pmt_snet_re_num = len(pmt_sent_re) # 선택한 prompt에서 추출한 감정 수
    cnt = 0
    for i in pmt_sent_re:
        if i in get_sents_from_essay[1]:
            cnt += 1

    cnt_re = cnt # 몇개가 일치하는지 계산결과

    # 일치비율 계산
    sent_comp_ratio_origin = round(cnt_re / pmt_snet_re_num * 100, 2)

    # 일치비율의 score 계산
    def calculate_score(sent_comp_ratio):
        if sent_comp_ratio >= 80:
            result_pmt_ori_sentiments = 'Supurb'
        elif sent_comp_ratio >= 60 and sent_comp_ratio < 80:
            result_pmt_ori_sentiments = 'Strong'
        elif sent_comp_ratio >= 40 and sent_comp_ratio < 60:
            result_pmt_ori_sentiments = 'Good'
        elif sent_comp_ratio >= 20 and sent_comp_ratio < 40:
            result_pmt_ori_sentiments = 'Mediocre'
        else: #sent_comp_ratio < 20
            result_pmt_ori_sentiments = 'Lacking'
        return result_pmt_ori_sentiments

    result_pmt_ori_sentiments = calculate_score(sent_comp_ratio_origin)

    # sent_comp_ratio_origin : 점수 --> overall 계산에 반영
    # result_pmt_ori_sentiments : 5점 척도 점수
    return sent_comp_ratio_origin, result_pmt_ori_sentiments


### run ###

essay_input = """This past summer, I had the privilege of participating in the University of Notre Dame’s Research Experience for Undergraduates (REU) program . Under the mentorship of Professor Wendy Bozeman and Professor Georgia Lebedev from the department of Biological Sciences, my goal this summer was to research the effects of cobalt iron oxide cored (CoFe2O3) titanium dioxide (TiO2) nanoparticles as a scaffold for drug delivery, specifically in the delivery of a compound known as curcumin, a flavonoid known for its anti-inflammatory effects. As a high school student trying to find a research opportunity, it was very difficult to find a place that was willing to take me in, but after many months of trying, I sought the help of my high school biology teacher, who used his resources to help me obtain a position in the program.				
Using equipment that a high school student could only dream of using, I was able to map apoptosis (programmed cell death) versus necrosis (cell death due to damage) in HeLa cells, a cervical cancer line, after treating them with curcumin-bound nanoparticles. Using flow cytometry to excite each individually suspended cell with a laser, the scattered light from the cells helped to determine which cells were living, had died from apoptosis or had died from necrosis. Using this collected data, it was possible to determine if the curcumin and/or the nanoparticles had played any significant role on the cervical cancer cells. Later, I was able to image cells in 4D through con-focal microscopy. From growing HeLa cells to trying to kill them with different compounds, I was able to gain the hands-on experience necessary for me to realize once again why I love science.				
Living on the Notre Dame campus with other REU students, UND athletes, and other summer school students was a whole other experience that prepared me for the world beyond high school. For 9 weeks, I worked, played and bonded with the other students, and had the opportunity to live the life of an independent college student.				
Along with the individually tailored research projects and the housing opportunity, there were seminars on public speaking, trips to the Fermi National Accelerator Laboratory, and one-on-one writing seminars for the end of the summer research papers we were each required to write. By the end of the summer, I wasn’t ready to leave the research that I was doing. While my research didn’t yield definitive results for the effects of curcumin on cervical cancer cells, my research on curcumin-functionalized CoFe2O4/TiO2 core-shell nanoconjugates indicated that there were many unknown factors affecting the HeLa cells, and spurred the lab to expand their research into determining whether or not the timing of the drug delivery mattered and whether or not the position of the binding site of the drugs would alter the results. Through this summer experience, I realized my ambition to pursue a career in research. I always knew that I would want to pursue a future in science, but the exciting world of research where the discoveries are limitless has captured my heart. This school year, the REU program has offered me a year-long job, and despite my obligations as a high school senior preparing for college, I couldn’t give up this offer, and so during this school year, I will be able to further both my research and interest in nanotechnology. """


select_pmt_type = 'College community: intended role, involvement, and contribution in college community'

# print(comp_sentiment_both(essay_input, select_pmt_type))
# 결과 예 : (10.0, 'Lacking')





def calculate_score(input_scofre):
    if input_scofre >= 80:
        result_comm_ori_keywordss = 'Supurb'
    elif input_scofre >= 60 and input_scofre < 80:
        result_comm_ori_keywordss = 'Strong'
    elif input_scofre >= 40 and input_scofre < 60:
        result_comm_ori_keywordss = 'Good'
    elif input_scofre >= 20 and input_scofre < 40:
        result_comm_ori_keywordss = 'Mediocre'
    else: #input_scofre < 20
        result_comm_ori_keywordss = 'Lacking'
    return result_comm_ori_keywordss

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



# initive engagement
def engagement(essay_input):

    Engagement_result = intellectualEnguagement(essay_input)
    Engagement_result_fin = calculate_score(Engagement_result[0])
    engagement_re_final = text_re_to_score(Engagement_result_fin)
    
    return engagement_re_final, Engagement_result[2]


### 종합계산(이것을 실행하면 됨) ###
def coll_comm_int_role_cont_coll_comm(select_pmt_type, select_college, select_college_dept, select_major, coll_supp_essay_input_data):
    comm_ori_keywords = pmtOrientedKeywords(coll_supp_essay_input_data)
    print('comm_ori_keywords[1]:', comm_ori_keywords[1])
    college_comm_fit = Fit(select_pmt_type, select_college, select_college_dept, select_major, coll_supp_essay_input_data)
    print('college_comm_fit:', college_comm_fit)
    pmt_ori_sent = Prompt_Oriented_Sentiments_analysis(coll_supp_essay_input_data)
    print('pmt_ori_sent:', pmt_ori_sent[0])
    initive_engagement = engagement(coll_supp_essay_input_data)
    print('initive_engagement:', initive_engagement)

    overall = round(comm_ori_keywords[1] * 0.35 + college_comm_fit * 0.35 + pmt_ori_sent[0] * 0.15 + initive_engagement[0] * 0.15, 1)


    ori_keywords = calculate_score(comm_ori_keywords[1])
    fitfit = calculate_score(college_comm_fit)
    sent__ = calculate_score(pmt_ori_sent[0])
    engagement_result_ = calculate_score(initive_engagement[0])



      # 문장생성
    fixed_top_comment = """When writing the college community essay, there are two key factors to consider. First, you should define your interest and role in the community as of today. Then, it would be best if you had thorough research about the college’s diverse student groups, facilities, and so on. After all, you want to create the “fit” between you and the college’s community for the admissions officer to imagine your role in it. Meanwhile, it would help to show positive sentiments such as admiration, excitement, and curiosity towards the school of your dreams."""

    def gen_comment(input_score, type):
        if input_score == 'Supurb' or input_score == 'Strong':
            if type == 'comm_ori_keywords':
                comment_achieve = """Your essay indicates a wealth of content associated with your community in terms of social awareness and contribution."""
            elif type == 'coll_community_fit':
                comment_achieve = """It seems that you have an insightful understanding of the given college's community and the opportunities it has to offer."""
            elif type == 'pmt_ori_sentiment':
                comment_achieve = """Moreover, the positive sentiments in your essay show your enthusiasm towards attending the school."""
            elif type == 'engagement':
                comment_achieve = """Lastly, your story shows that you are most likely to be a valued contributor to the campus community."""
            else:
                pass
        elif input_score == 'Good':
            if type == 'comm_ori_keywords':
                comment_achieve = """Your essay indicates some contents associated with your community in terms of social awareness and contribution."""
            elif type == 'coll_community_fit':
                comment_achieve = """It seems that you have a sufficient understanding of the given college's community and the opportunities it has to offer."""
            elif type == 'pmt_ori_sentiment':
                comment_achieve = """Moreover, the positive sentiments in your essay show your interest in attending the school."""
            elif type == 'engagement':
                comment_achieve = """Lastly, your story shows that you are likely to be a contributor to the campus community."""
            else:
                pass
        else: #input score == 'Mediocre' or input_score == 'Weak'
            if type == 'comm_ori_keywords':
                comment_achieve = """Your essay indicates lacking contents associated with your community in terms of social awareness and contribution."""
            elif type == 'coll_community_fit':
                comment_achieve = """It seems that you may need to deepen your understanding of the given college's community and the opportunities it has to offer."""
            elif type == 'pmt_ori_sentiment':
                comment_achieve = """Moreover, your essay may need to demonstrate a more substantial level of enthusiasm towards attending the school."""
            elif type == 'engagement':
                comment_achieve = """Lastly, you may need to demonstrate a higher level of effort and dedication as a community member."""
            else:
                pass
        return comment_achieve


    comment_1 = gen_comment(ori_keywords, 'comm_ori_keywords')
    comment_2 = gen_comment(fitfit, 'coll_community_fit')
    comment_3 = gen_comment(sent__, 'pmt_ori_sentiment')
    comment_4 = gen_comment(engagement_result_, 'engagement')




    data = {
        #'overall' :  overall, #overall Score
        'ori_keywords' : ori_keywords, # Supurb ~ Lacking
        'fitfit' : fitfit, # Supurb ~ Lacking
        'sent__' : sent__, # Supurb ~ Lacking
        'engagement_result_' : engagement_result_, # Supurb ~ Lacking
        'sentments' : pmt_ori_sent[1], # 추출한 감성 단어 - 웹페이지에 반영
        'initive_engagement[1]' : initive_engagement[1], # 추출한 engagement 단어 -- 웹페이지에 반영

        'comm_ori_keywords[2]' : comm_ori_keywords[2], # community oriented keywords 단어들 -- 웹페이지에 반영
        'get_sents_from_essay[1]' : pmt_ori_sent[1], # 리스트 단어들 -- 웹페이지에 반영

        'fixed_top_comment' : fixed_top_comment,
        'comment_1' : comment_1,
        'comment_2' : comment_2,
        'comment_3' : comment_3,
        'comment_4' : comment_4,

    }

    return data



print(coll_comm_int_role_cont_coll_comm("Summer activity", 'Brown', 'Brown_African Studies_dept', 'African Studies', essay_input))
