#  Prompt Oriented Keywords  : prompt 별로 관련 키워드가 다르기 때문에 입력값으로 해당 키워드들로 선택작업해야 함

import nltk
import pickle
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
            print("Topic %d :" % (idx+1), [(feature_names[i], topic[i].round(2)) for i in topic.argsort()[:-n -1:-1]])
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
        print('ittm:', ittm)
        cnt_ = 0
        for t in range(len(topics_ext[ittm])-1):
            
            print('t:', t)
            add = topics_ext[ittm][cnt_][0]
            result_.append(add)
            print('result_:', result_)
            cnt_ += 1

    #추출된 토픽들의 중복값 제거
    result_fin = list(set(result_))

    #### 토픽 추출한 결과 반환 ####
    return result_fin

######################################################################


## 연관어 불어오기 ##
#유사단어를 추출하여 리스트로 반환
def get_sim_words(prompt_type, text):

    essay_input_corpus = str(text) #문장입력
    essay_input_corpus = essay_input_corpus.lower()#소문자 변환

    sentences  = sent_tokenize(essay_input_corpus) #문장 토큰화
    # total_sentences = len(sentences)#토큰으로 처리된 총 문장 수
    # total_words = len(word_tokenize(essay_input_corpus))# 총 단어수
    
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
    if prompt_type == 'Why us':
        pmt_typ = [""" 'Why us' school & major interest (select major, by college & department) """]
        words_list = ['Admiration', 'Excitement', 'Pride', 'Realization', 'Curiosity']
    elif prompt_type == 'Intellectual interest':
        pmt_typ = [""" Intellectual interest """]
        words_list = ['Curiosity', 'Realization']
    elif prompt_type == 'Meaningful experience & lesson learned':
        pmt_typ = ["Meaningful experience & lesson learned"]
        words_list = ['Realization', 'Approval', 'Gratitude', 'Admiration']
    elif prompt_type ==  'Achievement you are proud of':
        pmt_typ = ["Achievement you are proud of"]
        words_list = ['bloom','stimulation','sales','competency','Throughput','Parity','specificity','credential','coercion','ideally','propulsion','glory',
                        'improvement','objective','Competence','aspire','Autonomy','statewide','Effective','capstone','Therapeutic','Placement','placement','Acceleration','generate',
                        'meaningful','worst','moderate','letter','able','acclaim','Realization','strategy','Popularity','incentive','copestone','Quality','mastery','win','salvation',
                        'independence','supreme','Penetration','Implementation','competence','Realistic','valiant','ensure','detection','transparency','compensation','Dispersion',
                        'Aspiration','Undefeated','Speed','miracle','Nirvana','promise','improve','Podium','proficiency','extraordinary','achievance','ahead','grade','Maximal',
                        'Statewide','reliability','supremacy','yearly','unity','mark','grammy','salute','calibration','make','Baccalaureate','excellence','hegemony','disarmament',
                        'Equilibrium','Prominence','joint','Amplification','autonomy','glorious','Accuracy','Efficacy','lasting','exam','stone','resolution','Supremacy','Breakthrough',
                        'Harmonious','guinness','tailored','positioning','benchmark','rebirth','modulation','kudos','Eminence','Desirable','Ranking','standardization','platinum',
                        'Accomplishment','Capillary','Perseverance','Attainment','desirable','Stabilization','Calibration','milestone','effort','Satisfactory','reconciliation',
                        'Reconciliation','tangible','compatibility','Resolution','Flexibility','badge','effective','perfection','stability','Titan','Empowerment','finish','strive',
                        'Excellence','technique','Level','talent','Rating','plateau','apex','Grammy','mainstream','Exam','rating','felicitate','motivate','emancipation','Stability',
                        'chart','promising','masterpiece','emulation','famous','saving','equality','Balance','honorary','Equitable','Technique','amplification','Statistically',
                        'stabilization','Unification','renown','ambitious','crowning','Reunification','Precision','problem-solving','Utilization','Triumph','pride','high',
                        'scholarship','Accreditation','Dominance','Fertilization','eminence','league','Hegemony','score','Rebirth','reduction','trophy','motivation','achieve',
                        'Integration','parity','Score','helping','Moderate','Impossible','aim','Distinction','Reduction','Purification','certification','career','satisfactory',
                        'Selective','impossible','Competency','congratulate','speed','accreditation','excel','velocity','efficiently','stakeholder','reward','progress','Saving',
                        'Metre','gain','efficiency','measurable','Strategy','abstinence','feather','Accomplish','Chart','Result','Strive','Perfection','prestige','hurdle',
                        'bandwidth','leading','quality','monument','Incentive','maximum','hardship','therapeutic','greatness','fusion','Transparency','tolerance','Athlete',
                        'Attained','harmonious','uniformity','podium','Mastery','Benchmark','Fame','triumph','Status','High','ability','Grail','deed','renowned','Promotion',
                        'best','distinguished','Renown','Coupling','selective','Placing','penetration','acceptable','Efficiently','ideal','fulfilment','Coherence','Widespread',
                        'statistically','standard','Fusion','consolidation','Efficiency','percent','Acceptable','Decisive','Qualified','balance','emulate','Fixation','luminary',
                        'cum','performance','undefeated','Retention','capillary','implementation','Standardization','Minimum','immortality','Ideal','Consolidation','Equity',
                        'compliance','Qualifying','Yearly','enable','brilliant','Minimize','Standard','integration','throughput','acceptance','Improvement','Ensure','Unity',
                        'optimal','domination','recognition','optimum','Meaningful','Optimize','limitation','resolve','Equality','reunification','superiority','ambition',
                        'Surpass','Consensus','marks','selectivity','Certification','jest','Fulfilling','Feat','overtake','plan','fulfillment','productivity','Output',
                        'qualify','impress','fertilization','Worthwhile','mach','Stimulation','efficient','Consecutive','coherence','Motivate','degree','Independence',
                        'outshine','consecutive','result','amplifier','Effectiveness','emission','Coercion','Consistency','Measurable','Saturation','Transcend','popularity',
                        'pinnacle','Optimal','top','quota','equilibrium','strategic','up','qualifying','precocious','Lasting','Ambition','prosperity','capability','enlightenment',
                        'consistency','success','accountability','Sales','Compliance','sustainability','Uniformity','Disarmament','unbeaten','Reliability','Sustainable','minimum',
                        'baccalaureate','level','help','landmark','peak','recognize','empowerment','comparable','bonding','satisfaction','enrichment','Sustained','Maturity',
                        'Strategic','failed','coup','Unprecedented','Clarity','Quota','output','Greatness','Compatibility','Objective','grandmaster','preen','Enabling',
                        'Compression','titan','Victory','Stakeholder','Tangible','closure','Altering','Success','Guinness','Percent','laude','Attain','unprecedented','credit',
                        'flexibility','Comparable','unification','Domination','Salvation','Productivity','Unbeaten','big','equal','clat','Empower','stature','work','grail',
                        'striving','Finish','Emission','utilization','assessment','Goal','saturation','Enrichment','Acclaim','Recognition','Platinum','golden','Enlightenment',
                        'acceleration','panegyric','victory','widespread','harvest','difficulty','realization','placing','Accountability','Gain','frustrate','sustained',
                        'vanity','society','Milestone','surpass','Peak','inspiration','belt','Capability','summit','coupling','emprise','Aim','Optimum','dispersion','Ideally',
                        'opus','Specificity','Grade','retention','medal','laurel','learner','laureate','target','Velocity','goal','worthwhile','allocation','minimize',
                        'celebrated','build','accomplishment','Immortality','record','Acceptance','Tailored','clarity','fixation','diffraction','statehood','sustainable',
                        'compression','accuracy','perseverance','full','attain','Proficiency','empower','Amplifier','eminent','Bonding','Potential','acme','merit',
                        'reconciliate','attained','qualified','ranking','Outcome','Rank','dominance','Prosperity','Helping','equitable','maximal','Motivation','consummation',
                        'Maximize','metre','crown','adversity','Mach','efficacy','accomplish','Tolerance','optimize','outcome','Emancipation','feat','equity','purification',
                        'rank','Aspire','status','maturity','Grandmaster','potential','Fulfillment','Target','precision','Bandwidth','Superiority','enabling','athlete',
                        'Detection','obstacle','realistic','Abstinence','honor','Failed','Learner','award','effectiveness','decisive','Maximum','promotion','scale','nirvana',
                        'altering','Statehood','cap','rigorous','magnum','attainment','transcend','distinction','Qualification','age','Propulsion','desired','meritocracy',
                        'eclat','Diffraction','accredit','fulfilling','congratulation','contribute','Sustainability','Performance','consensus','hatchment','Equal','create',
                        'Positioning','prominence','coping','fail','Modulation','Rigorous','breakthrough','Selectivity','aspiration','learn','Mainstream','illustrious',
                        'qualification','Desired','stellar','maximize','Allocation','Striving','big-league','Progress','record-breaking','triumphant','Efficient','tops','fame']
    elif prompt_type ==  'Social issues: contribution & solution':
        pmt_typ = ["Social issues: contribution & solution"]
        # 0~16 번째는 에세이 입력의 40% 분석 적용, 그 이후부분은 60% 적용  --> 즉 [:16], [17:] 이렇게 나눌 것
        words_list = ['Anger', 'annoyance', 'Fear', 'Disapproval', 'disgust', 'Disappointment','grief', 'nervousness', 'sadness', 'surprise', 'remorse', 'curiosity', 'embarrassment', 'Realization','Approval', 'Gratitude', 'Admiration','Admiration','Approval', 'Caring', 'Joy', 'Gratitude', 'Optimism','relief', 'Realization']
    elif prompt_type ==  'Summer activity':
        pmt_typ = ["Summer activity"]
        words_list = ['Pride','Realization','Curiosity','Excitement','Amusement','Caring']
    elif prompt_type ==  'Unique quality, passion, or talent':
        pmt_typ = ["Unique quality, passion, or talent"]
        words_list = ['Pride','Excitement','Amusement','Approval','Admiration','Curiosity']
    elif prompt_type ==  'Extracurricular activity or work experience':
        pmt_typ = ["Extracurricular activity or work experience"]
        words_list = ['Pride','Realization','Curiosity','Joy','Excitement','Amusement','Caring','Optimism']
    elif prompt_type ==  'Your community: role and contribution in your community':
        pmt_typ = ["Your community: role and contribution in your community"]
        words_list = ['Admiration','Caring','Approval','Pride','Gratitude','Love']
    elif prompt_type ==  'College community: intended role, involvement, and contribution in college community':
        pmt_typ = ["Admiration','Caring','Approval','Pride','Gratitude','Love"]
        words_list = ['Admiration','Caring','Approval','Excitement','Pride','Gratitude']
    elif prompt_type ==  'Overcoming a Challenge or ethical dilemma':
        pmt_typ = ["Overcoming a Challenge or ethical dilemma"]
        words_list = ['Anger','Fear','Disapproval','Disappointment','Confusion','Annoyed','Realization', 'Approval','Gratitude','Admiration','Relief','Optimism']
    elif prompt_type ==  'Culture & diversity':
        pmt_typ = ["Culture & diversity"]
        # 관련 단어 리스트를 불러옴
        with open('./data/data_communityWords.pickle', 'rb') as f:
            words_list = pickle.load(f)
    elif prompt_type ==  'Collaboration & teamwork':
        pmt_typ = ["Collaboration & teamwork"]
        words_list = ['collaborative','esteem','student','physical','simulation','pioneer','engagement','demonstrate','derail','involve','creative','camping','trust','innovation','combat','innovative','motivation','activity','interpersonal','productivity','achievement','courage','competency','behavior','core','adapt','importance','awareness','element','cooperative','responsibility','task','stress','conflict','assessment','understanding','crew','practice','patient','learning','enable','goal','knowledge','dedication','oriented','display','supervision','judge','completion','cohesion','essential','enhance','loyalty','accomplish','empathy','safety','motto','positive','excellent','encourage','efficiency','strength','attitude','caring','drill','opportunity','aim','necessity','participate','enrich','friendship','crucial','setting','interaction','management','leadership','alternative','excellence','concept','fitness','sport','rowing','math','commitment','obstacle','resource','assess','thinking','youth','aviation','effort','deploy','tactical','exemplify','curiosity','require','self','nato','build','teammate','confidence','skill','success','perseverance','pilot','strengthen','motivate','ability','managerial','flexible','team','improve','outcome','participation','robot','cadet','virtual','help','process','experience','ethic','healthcare','chat','dynamics','employee','counterpart','sense','shooter','consensus','customer','communication','raceway','value','tag','approach','talent','achieve','learn','problem','flexibility','community','reinforce','individual','combine','implement','puzzle','spirit','group','develop','collaboration','prevention','satisfaction','effectiveness','curriculum','researcher','competence','analyze','marketing','effective','dynamic','quality','teach','corporate','organization','toyota','enhanced','resolve','job','technique','lau','cognition','professional','time','mutual','literacy','performance','recruit','component','rely','strategy','competition','outward','integration','integrity','cooperation','exercise','reward','program','evaluate','fairness','mission','decision','coordinate','integrate','overcome','facilitate','screening','designed','discipline','challenge','pike','partnership','complexity','athlete','emphasize','reliance','incorporate','stability','athletic','initiative','endurance','creativity','coordination','gm','training','planning','focus','highlight','promote','participant','project','insight','emphasis','lesson','relationship','topic','constructive','fun','capability','workplace','lack','employer','sharing','organizational','player','tactic','foster','context','evaluation','trio','praise','solve','respect','environment','adventure','airlift','deployment']
    elif prompt_type ==  'Creativity/creative projects':
        pmt_typ = ["Creativity/creative projects"]
        # load
        with open('./data/data_creative_acacdemic_Words.pickle', 'rb') as f:
            words_list = pickle.load(f)
    elif prompt_type ==  'Leadership experience':
        pmt_typ = ["Leadership experience"]
        # load
        with open('./data/data_leadership_Words.pickle', 'rb') as f:
            words_list = pickle.load(f)
    elif prompt_type ==  'Values, perspectives, or beliefs':
        pmt_typ = ["Values, perspectives, or beliefs"]
        # load
        with open('./data/data_coreValue_Words.pickle', 'rb') as f:
            words_list = pickle.load(f)
    elif prompt_type ==  'Person who influenced you':
        pmt_typ = ["Person who influenced you"]
        # load
        with open('./data/data_coreValue_Words.pickle', 'rb') as f:
            words_list = pickle.load(f)
    elif prompt_type ==  'Favorite book/movie/quote':
        pmt_typ = ["Favorite book/movie/quote"]
        # load
        with open('./data/data_coreValue_Words.pickle', 'rb') as f:
            words_list = pickle.load(f)
    elif prompt_type ==  'Write to future roommate':
        pmt_typ = ["Write to future roommate"]
        words_list = ['Admiration','Realization','Love','Excitement','Approval','Pride','Gratitude','Amusement','Curiosity','Joy']
    elif prompt_type ==  'Diversity & Inclusion Statement':
        pmt_typ = ["Diversity & Inclusion Statement"]
        # 관련 단어 리스트를 불러옴
        with open('./data/data_communityWords.pickle', 'rb') as f:
            words_list = pickle.load(f)
    elif prompt_type ==  'Future goals or reasons for learning':
        pmt_typ = ["Future goals or reasons for learning"]
        # load
        with open('./data/data_future_goal.pickle', 'rb') as f:
            words_list = pickle.load(f)
    elif prompt_type ==  'What you do for fun':
        pmt_typ = ["What you do for fun"]
        # load
        with open('./data/data_what_do_you_do_for_fun.pickle', 'rb') as f:
            words_list = pickle.load(f)
    else:
        pass

    
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

    # pmt_typ : 선택한 프람프트 종류
    # ext_sim_words_key : 입력문장과 프람프트 단어리스트와 유사한 단어 추출
    return pmt_typ, ext_sim_words_key



# # 실행 #

# essay_input = """This past summer, I had the privilege of participating in the University of Notre Dame’s Research Experience for Undergraduates (REU) program . Under the mentorship of Professor Wendy Bozeman and Professor Georgia Lebedev from the department of Biological Sciences, my goal this summer was to research the effects of cobalt iron oxide cored (CoFe2O3) titanium dioxide (TiO2) nanoparticles as a scaffold for drug delivery, specifically in the delivery of a compound known as curcumin, a flavonoid known for its anti-inflammatory effects. As a high school student trying to find a research opportunity, it was very difficult to find a place that was willing to take me in, but after many months of trying, I sought the help of my high school biology teacher, who used his resources to help me obtain a position in the program.				
# Using equipment that a high school student could only dream of using, I was able to map apoptosis (programmed cell death) versus necrosis (cell death due to damage) in HeLa cells, a cervical cancer line, after treating them with curcumin-bound nanoparticles. Using flow cytometry to excite each individually suspended cell with a laser, the scattered light from the cells helped to determine which cells were living, had died from apoptosis or had died from necrosis. Using this collected data, it was possible to determine if the curcumin and/or the nanoparticles had played any significant role on the cervical cancer cells. Later, I was able to image cells in 4D through con-focal microscopy. From growing HeLa cells to trying to kill them with different compounds, I was able to gain the hands-on experience necessary for me to realize once again why I love science.				
# Living on the Notre Dame campus with other REU students, UND athletes, and other summer school students was a whole other experience that prepared me for the world beyond high school. For 9 weeks, I worked, played and bonded with the other students, and had the opportunity to live the life of an independent college student.				
# Along with the individually tailored research projects and the housing opportunity, there were seminars on public speaking, trips to the Fermi National Accelerator Laboratory, and one-on-one writing seminars for the end of the summer research papers we were each required to write. By the end of the summer, I wasn’t ready to leave the research that I was doing. While my research didn’t yield definitive results for the effects of curcumin on cervical cancer cells, my research on curcumin-functionalized CoFe2O4/TiO2 core-shell nanoconjugates indicated that there were many unknown factors affecting the HeLa cells, and spurred the lab to expand their research into determining whether or not the timing of the drug delivery mattered and whether or not the position of the binding site of the drugs would alter the results. Through this summer experience, I realized my ambition to pursue a career in research. I always knew that I would want to pursue a future in science, but the exciting world of research where the discoveries are limitless has captured my heart. This school year, the REU program has offered me a year-long job, and despite my obligations as a high school senior preparing for college, I couldn’t give up this offer, and so during this school year, I will be able to further both my research and interest in nanotechnology. """


# print('에세이에서 추출한 토픽:', getTopics(essay_input))

# #유사단어를 문장에서 추출하여 반환한다.
# get_sim_words_result = get_sim_words(essay_input)
# print('입력에세이에서 achievement 관련 유시단어 추출하여 반환', get_sim_words_result)




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
def prompt_oriented_key_wds(prompt_type, essay_input):
    get_ext_topics_re = getTopics(essay_input) # 토픽 추출
    ext_topics_num = len(get_ext_topics_re) # 추출한 토픽 수
    # achievement - 유사단어를 문장에서 추출하여 반환
    get_sim_words_result = get_sim_words(prompt_type, essay_input)

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

essay_input = """This past summer, I had the privilege of participating in the University of Notre Dame’s Research Experience for Undergraduates (REU) program . Under the mentorship of Professor Wendy Bozeman and Professor Georgia Lebedev from the department of Biological Sciences, my goal this summer was to research the effects of cobalt iron oxide cored (CoFe2O3) titanium dioxide (TiO2) nanoparticles as a scaffold for drug delivery, specifically in the delivery of a compound known as curcumin, a flavonoid known for its anti-inflammatory effects. As a high school student trying to find a research opportunity, it was very difficult to find a place that was willing to take me in, but after many months of trying, I sought the help of my high school biology teacher, who used his resources to help me obtain a position in the program.				
Using equipment that a high school student could only dream of using, I was able to map apoptosis (programmed cell death) versus necrosis (cell death due to damage) in HeLa cells, a cervical cancer line, after treating them with curcumin-bound nanoparticles. Using flow cytometry to excite each individually suspended cell with a laser, the scattered light from the cells helped to determine which cells were living, had died from apoptosis or had died from necrosis. Using this collected data, it was possible to determine if the curcumin and/or the nanoparticles had played any significant role on the cervical cancer cells. Later, I was able to image cells in 4D through con-focal microscopy. From growing HeLa cells to trying to kill them with different compounds, I was able to gain the hands-on experience necessary for me to realize once again why I love science.				
Living on the Notre Dame campus with other REU students, UND athletes, and other summer school students was a whole other experience that prepared me for the world beyond high school. For 9 weeks, I worked, played and bonded with the other students, and had the opportunity to live the life of an independent college student.				
Along with the individually tailored research projects and the housing opportunity, there were seminars on public speaking, trips to the Fermi National Accelerator Laboratory, and one-on-one writing seminars for the end of the summer research papers we were each required to write. By the end of the summer, I wasn’t ready to leave the research that I was doing. While my research didn’t yield definitive results for the effects of curcumin on cervical cancer cells, my research on curcumin-functionalized CoFe2O4/TiO2 core-shell nanoconjugates indicated that there were many unknown factors affecting the HeLa cells, and spurred the lab to expand their research into determining whether or not the timing of the drug delivery mattered and whether or not the position of the binding site of the drugs would alter the results. Through this summer experience, I realized my ambition to pursue a career in research. I always knew that I would want to pursue a future in science, but the exciting world of research where the discoveries are limitless has captured my heart. This school year, the REU program has offered me a year-long job, and despite my obligations as a high school senior preparing for college, I couldn’t give up this offer, and so during this school year, I will be able to further both my research and interest in nanotechnology. """

# Prompt Oriented Keywords 최종계산

prompt_type = "Culture & diversity"

pmt_ori_key_re = prompt_oriented_key_wds(prompt_type, essay_input)
print('Prompt Oriented Keywords 값 계산 결과:', pmt_ori_key_re[0])
print('Prompt Oriented Keywords - 입력에세이의 토픽과 비교할 단어가 얼마나 일치하는지에 대한 비율 계산 결과 :', pmt_ori_key_re[1])
print('Prompt Oriented Keywords 단어들(웹사이트에 표시) :', pmt_ori_key_re[2])
print('initiative_enguagement 관련 단어가 에세이에 포함된 비율 :', pmt_ori_key_re[3])
print('initiative_enguagement가 합격생 평균에 비교하여 얻은 최종 값 :', pmt_ori_key_re[4])


# Achievement you are proud of : Prompt Oriented Keywords 일치율 0
# Prompt Oriented Keywords 값 계산 결과: Lacking
# Prompt Oriented Keywords - 입력에세이의 토픽과 비교할 단어가 얼마나 일치하는지에 대한 비율 계산 결과 : 0
# Prompt Oriented Keywords 단어들(웹사이트에 표시) : []
# initiative_enguagement 관련 단어가 에세이에 포함된 비율 : 7.531
# initiative_enguagement가 합격생 평균에 비교하여 얻은 최종 값 : Strong