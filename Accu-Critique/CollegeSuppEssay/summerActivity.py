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
        'Popular Summer Programs' : result_sc, # 매칭되는 결과로 점수로 정하기
        'Name of Popular Summer Programs' : get_score_fin, # 에세이서 발견한 활동명칭
        #'get_word_position' : detect_activities # 현재 이 분석값을 의미가 없지만 나중에 활용할거임
        'Popular Summer Programs score' : result_score # 추출한 활동내역을 점수로 변환 --------> overall 값 계산에 적용할 것
    }

    return data


# 상대적 점수 비교 계산
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
        re__score = 100
    elif score == 4:
        result_ = 'Strong'
        re__score = 80
    elif score == 3:
        result_ = 'Good'
        re__score = 60
    elif score == 2:
        result_ = 'Mediocre'
        re__score = 40
    else: #score = 1
        result_ = 'Lacking'
        re__score = 20

    return result_, re__score


def summer_activity_initiative_engagement(essay_input):
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

    # 데이터 불러오기
    data_action_verbs = pd.read_csv('./data/actionverbs.csv')
    data_ac_verbs_list = data_action_verbs.values.tolist()
    verbs_list_ = [y for x in data_ac_verbs_list for y in x]

    academic_verbs = ['everyone','satisfying','spectacular','rightly','expert','see','unexpected','simply','exceptional','pure','claimed','well','reasonable','light','bet','due','judgment','gratifying','assume','speaking','point','neither','agree','personally','usually','sitting','main','standpoint','truly','mind','tremendous','resembles','pleasurable','adverbs','people','....','safely','definitely','foolishly','honest','confident','heavily','i’d','miraculous','regard','know','predominantly','positive','would','help','understanding','serious','change','highly','disagree','estimation','opinion','phenomenal','i’ll','primarily','solely','reaction','exactly.','undoubtedly',"i'd",'scenic','reservation','likely','concerned','issue','sake','say','shred','exactly','imho','seen','come','book','said','taste','postulate','pretend','view','technically','position','impressive','like','clearly','incredibly','bravely','certainly','surely','hold','given','glorious','consider','unduly','support','maybe','saying','enormously','least','mixed','evidence','extremely','suspect','opposite','imagine','totally','understand','cannot','belief','much','get','perfectly','unlikely','consideration','great','won’t','continually','beautiful','maintain','fair','seems','classic','complete','argued','cleverly','carelessly','suppose','infer','enjoyable','sat','sensational','attractive','strongly','ridiculously','top','according','judgement','surprisingly','clear','wrong','high','surprising','expressed','quality','stunning',"i'm",'convinced','right','really','frank','idiot','find','wish','this.','fact','subject','perspective','sight','remarkable','deny','conclude','certain','observed','experience','course','fortunately','superb','idea','possibly','doubt','fantastic','completely','viewpoint','sheer','assumes','perfect','indeed','typically','unique','situation','delightful','topic','seriously','complicated','think','naturally','suggests','case','part','mainly','unfortunately','generally','generously','particular','actual','..','confidentially','unforeseen','delicious','matter','limited','want','guess','grand','presumably','first-rate','breathtaking','merely','can’t','far','probably','dare','wicked','matter.','doubtless','reckon','repeatedly','gather','...','ask','fabulous','magnificent','perhaps','wonderful','truthfully','may','large','suggest','unbelievably','obviously','purely','must','it’s','dreadfully','majestic','whole','utter','picturesque','wrote','sterling','mostly','pleasant','unpredictable','bitterly','believe','tend','pretty','constantly','alone','prime','appears','read','way','look','sure','positively','deadly','exquisite','conceit','lovely','quite','personal','thoughtfully','either','kindly','could','sufficiently','tell','giving','absolute','noticed','commenting','vantage','plainly','head','theoretically','you’d','argument','eye','notably','familiar','obvious','unbelievable','shadow','feel','amazing','wa','question','fulfilling','i’ve','incredible','mistaken','admit','rather','person','minority',"one's",'standing','one','later','frankly','rewarding','entirely','outta','i’m','initial','disappointingly','even','methinks','argue','brilliant','weighing','consistently','towards','assessment','charming','marvellous','think?','old','imposing','thinking','unusual','precisely','sound','seem','thought','take','stand','go','care','money','superior','always','absolutely','particularly','mean','might','total','although','especially','extraordinary','sit','luckily','increasingly','complex','never','feeling','knowledge','outstanding','summarise','side','conceivably','chiefly','exclusively','presume','reckoning','without','controversial','stupidly','best','excellent','terrific','frequently','amazingly','astonishing','impression','correct','fairly','humble','pleasing','crazy','conviction','conclusion','prof','unforeseeable','awesome','difficult','staggering','wisely']

    #contribution_wd = ['cook','promote','guide','outreach','clean','service','organize','tutor','present','counsel','sacrifice','bake','host','provide','repair','educate','donate','lead','raise','perform','empower','create','mediate','improve','initiate','grant','sponsor','write','enhance','resolve','teach','foster','offer','give','endowment','benefact','enrich','distribute','adopt','develop','oblation','translate','care','share','help','gift','contribute','manage','dedicate','subsidy','volunteer','start','participate']

    #verbs_list = verbs_list_ + academic_verbs + contribution_wd
    verbs_list = verbs_list_ + academic_verbs

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
    # 10.입력한 에세이 문장에서 관련 단어가 얼마나 포함되어 있는지 포함비율 분석
    wd_ratio = round(len(get_words__)/len(input_text_list) *100, 3)

    ##########################################################################
    ##################  합격한 학생들의 평균점수 반영(실제 평균값 적용해야 함)  ###########
    admitted_std_score = 8 # 8%로 가정
    ##########################################################################

    # 합격한 학생들과 비교하여 5가지 구분 결과로 게산하기
    social_awareness_analy_re = lackigIdealOverboard(admitted_std_score, wd_ratio)

    # 추출한 단어 중복제거
    ext_words = list(set(get_words__))

    # print ("ACTION VERBS RATIO :", action_verbs_ratio )

    # rerurn 해석
    # wd_ratio : 입력한 에세이에 비교분석하고자하는 단어가 얼마나 포함되어 있는지에 대한 비율 계산
    # ext_words : 포함된 관련단어 추출 출격 --> 웹에 표시
    # social_awareness_analy_re : 비교 결과값 추출로 2개의 값임 
    #   1) supurb ~ lacking
    #   2) score
    return wd_ratio, ext_words, social_awareness_analy_re


#def summer_act_majorfit(essay_input):
## 여기 개발중......








## run ##

essay_input = """ I inhale deeply and blow harder than I thought possible, pushing the tiny ember from its resting place on the candle out into the air. mit women's technology program (wtp) The room erupts around me, and 'Happy Birthday!' cheers echo through the halls. It's time to make a wish. In my mind, that new Limited Edition Deluxe Ben 10 watch will soon be mine. My parents and the aunties and uncles around me attempt to point me in a different direction. 'Wish that you get to go to the temple every day when you're older! Wish that you memorize all your Sanskrit texts before you turn 6! Wish that you can live in India after college!' My ears listen, but my mind tunes them out, as nothing could possibly compare to that toy watch! What I never realized on my third birthday is that those wishes quietly tell the story of how my family hopes my life will play out. In this version of my life, there wasn't much room for change, personal growth, or 'rocking the boat.' A vital aspect of my family's cultural background is their focus on accepting things as they are. Growing up, I was discouraged from questioning others or asking questions that didn't have definitive yes or no answers. If I innocently asked my grandma why she expected me to touch her feet, my dad would grab my hand in a sudden swoop, look me sternly in the eye, and tell me not to disrespect her like that again. At home, if I mentioned that I had tried eggs for breakfast at a friend's house, I'd be looked at like I had just committed a felony for eating what my parents considered meat. If I asked the priest at the temple why he had asked an Indian man and his white wife to leave, I'd be met with a condescending glare and told that I should also leave for asking such questions.In direct contrast, my curiosity was invited and encouraged at school. After an environmental science lesson, I stayed for a few minutes after class to ask my 4th-grade science teacher with wide eyes how it was possible that Niagara Falls doesn't run out of flowing water. Instead of scolding me for asking her a 'dumb question,' she smiled and explained the intricacy of the water cycle. Now, if a teacher mentions that we'll learn about why a certain proof or idea works only in a future class, I'll stay after to ask more or pour through an advanced textbook to try to understand it. While my perspective was widening at school, the receptiveness to raising complex questions at home was diminishing. After earning my driver's license, I registered as an organ donor. My small checkmark on a piece of paper led to an intense clash between my and my parents' moral platform. I wanted to ensure that I positively contributed to society, while my parents believed that organ donation was an unfamiliar and unnecessary cultural taboo. I would often ask for clarity or for reasons that supported their ideologies. Their response would usually entail feeling a deep, visceral sense that traditions must be followed exactly as taught, without objection. Told in one language to keep asking questions and in another to ask only the right ones, I chose exploring questions that don't have answers, rather than accepting answers that don't get questioned. When it comes to the maze of learning, even when I take a wrong turn and encounter roadblocks that are meant to stop me, I've learned to climb over them and keep moving forward. My curiosity strengthens with each hurdle and has expanded into a pure love of learning new things. I've become someone who seeks to understand things at a fundamental level and who finds excitement in taking on big questions that have yet to be solved. I'm no longer afraid to rock the boat. "},{"index":1,"personal_essay":"Ever since I first held a small foam Spiderman basketball in my tiny hands and watched my idol Kobe Bryant hit every three-pointer he attempted, I've wanted to understand and replicate his flawless jump shot. As my math education progressed in school, I began to realize I had the tools to create a perfect shot formula. After learning about variables for the first time in 5th grade Algebra, I began to treat each aspect of Kobe's jump shot as a different variable, each combination of variables resulting in a unique solution. While in 7th-grade geometry, I graphed the arc of his shot, and after learning about quadratic equations in 8th grade, I expressed his shot as a parabolic function that would ensure a swish when shooting from any spot. After calculus lessons in 10th and 11th grade, I was excited to finally solve for the perfect velocity and acceleration needed on my release. At Brown, I hope to explore this intellectual pursuit through a different lens. What if I could maximize the odds of making shots if I understood the science behind one's mental mindset and focus through CLPS 500: Perception and Action? Or use astrophysics to account for drag and gravitational force anywhere in the universe? Or use data science to break down the analytics of the NBA's best shooters? Through the Open Curriculum, I see myself not only becoming a more complete learner, but also a more complete thinker, applying a flexible mindset to any problem I encounter. Brown's Open Curriculum allows students to explore broadly while also diving deeply into their academic pursuits. Tell us about an academic interest (or interests) that excites you, and how you might use the Open Curriculum to pursue it. I've been playing the Mridangam since I was five years old. It's a simple instrument: A wood barrel covered on two ends by goatskin with leather straps surrounding the hull. This instrument serves as a connection between me and one of the most beautiful aspects of my culture: Carnatic music. As a young child, I'd be taken to the temple every weekend for three-hour-long Carnatic music concerts, where the most accomplished teenagers and young adults in our local Indian community would perform. I would watch in awe as the mridangists' hands moved gracefully, flowing across the goatskin as if they weren't making contact, while simultaneously producing sharp rhythmic patterns that never failed to fall on the beat. Hoping to be like these idols on the stage, I trained intensely with my teacher, a strict man who taught me that the simple drum I was playing had thousands of years of culture behind it. Building up from simple strokes, I realized that the finger speed I'd had been awestruck by wasn't some magical talent, it was instead a science perfected by repeated practice."""


print('SummerActivity result :', SummerActivity(essay_input))

print('initiative_engagement :', summer_activity_initiative_engagement(essay_input))
# initiative_engagement : (10.354, ['personal', 'certain', 'hit', ...