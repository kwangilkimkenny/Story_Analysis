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



# 1) Social Awareness (30%): 
# List of social issues + Diversity 관련 단어 + activism words 


####  social_awareness_analysis ###
def social_awareness_analysis(essay_input):
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

    #리스로 변환된 값 확인

    #데이터 불러오기
    data_wd_list = ['hong', 'unconscious', 'pronoun', 'destruction', 'idealism', 'sierra', 'autism', 'asexual', 'kazakhs', 'cubans', 'homophobia', 'affirmative', 'nigeriens', 'sindhi', 'kyrgyz', 'uruguayan', 'portuguese', 'são', 'privilege', 'mozambicans', 'anti-semitism', 'ethiopians', 'tactic', 'djiboutians', 'jordanian', 'liberation', 'klan', 'slovak', 'bretons', 'prejudice', 'emotional', 'politics', 'gangs', 'sustainable', 'bulgarian', 'energy', 'native', 'igbo', 'simply', 'engagement', 'lucians', 'champion', 'ageism', 'abuse', 'behavioral', '–', 'judiciary', 'network', 'lankan', 'mauritian', 'mass', 'hate', 'conservative', 'chadian', 'spaniards', 'gun', 'xenophobia', 'inclusion', 'services', 'academic', 'sports', 'sri', 'belizeans', 'lgbtqia', 'wilderness', 'gambians', 'mandatory', 'drought', 'berbers', 'stereotypes', 'uzbeks', 'sinhalese', 'greenlander', 'ugandan', 'swedes', 'gsd', 'optimism', 'nepalese', 'monégasque', 'mongolian', 'panamanians', 'empowerment', 'mexicans', 'transitioning', 'amnesty', 'andorrans', 'guerrilla', 'cute', 'punjabi', 'equatoguinean', 'slovene', 'air', 'tunisian', 'tongans', 'main', 'campaigning', 'higher', 'burkinabé', 'libyans', 'voluntarism', "women's", 'people', 'butch', 'sentencing', 'microadvantages', 'arrested', 'vincentian', 'networking', 'basque', 'capital', 'education', 'macao', 'antiguans', 'prisoner', 'ugandans', 'surinamese', 'outgroup', 'homelessness', 'slavery', 'redefine', 'sweatshops', 'feminist', 'marxist', 'involvement', 'thais', 'color', 'cults', 'heterosexual', 'malay', 'ivoirian', 'microaffirmations', 'credibility', 'bahamian', 'indians', 'tunisians', 'tuvaluan', 'austrians', 'quebecer', 'warming', 'atheism', 'aruban', 'supremacy', 'gandhi', 'colombian', 'latvians', 'romanians', 'liechtensteiner', 'barbudan', 'bangladeshis', 'downsizing', 'malian', 'theorize', 'black', 'omani', 'collective', 'costa', 'cape', 'terrorism', 'critique', 'comorians', 'chileans', 'afghans', 'equatoguineans', 'tuvaluans', 'spending', 'opinion', 'substantive', 'stakeholder', 'montenegrin', 'sociological', 'foods', 'bame', 'kenyan', 'libertarian', 'belarusians', 'liechtensteiners', 'macedonians', 'war', 'luxembourgers', 'internet', 'stance', 'togolese', 'raising', 'peruvians', 'malawians', 'alcohol', 'x', 'djiboutian', 'maltese', 'bulgarians', 'stereotype', 'guyanese', 'issue', 'weapons', 'deforestation', 'participation', 'citizen', 'czechs', 'naacp', 'conscious', 'sprawl', 'east', 'ivoirians', 'honduran', 'infectious', 'heritage', 'réunionnais', 'speech', 'coastal', 'queer', 'somalilanders', 'lithuanians', 'uruguayans', 'crimes', 'space', 'legitimacy', 'irish', 'micronesian', 'afghan', 'seychellois', 'foreign', 'iranian', 'bahrainis', 'religions', 'gender', 'angamis', 'obesity', 'group', 'dictatorship', 'grassroots', 'italians', 'genderqueer', 'somali', 'responsibility', 'georgians', 'tice', 'behalf', 'illegal', 'zimbabweans', 'homo', 'americas', 'politically', 'algerians', 'illness', 'silesian', 'roma', 'indoor', 'gambian', 'elite', 'latina', 'estonians', 'yemeni', 'drc', 'bermudians', 'pornography', 'diversity', 'ghanaians', 'noise', 'engaged', 'gabonese', 'segregate', 'racist', 'somalis', 'poc', 'syndrome', 'armenian', 'syriac', 'bilingualism', 'cypriot', 'naga', 'workers', 'drugs', 'nicaraguans', 'control', 'mainstream', 'maldivians', 'tax', 'support', 'protesting', 'organizer', 'ace', 'greeks', 'boers', 'devote', 'mozambican', 'dissemination', 'chinese', 'mixed', 'ideology', 'canadian', 'german', 'azerbaijanis', 'straight', 'han', 'testing', 'nevis', 'falkland', 'pakistani', 'capitalism', 'malagasy', 'allyship', 'qatari', 'ethnic', 'verdean', 'resurgence', 'lgbtq+', 'bosniaks', 'tibetans', 'minimum', 'computer', 'indigenou', 'ally', 'argentines', 'senegalese', 'tamils', 'zambians', 'innate', 'funding', 'ecuadorians', 'cancer', 'first', 'radical', 'lgbtqi', 'initiative', 'chaldeans', 'hispanic', 'workplace', 'emphasize', 'ecuadorian', 'orientation', 'fuel', 'arubans', 'lankans', 'mental', 'baltic', 'bolivian', 'english', 'israeli', 'homosexuality', 'micronesians', 'kurds', 'anti', 'repression', 'discrimination', 'singaporeans', 'cannabis', 'disease', 'agenda', 'gay', 'revolutionary', 'living', 'prostitution', 'radio', 'kuwaitis', 'superstores', 'vigil', 'intersectionality', 'protection', 'basques', 'privacy', 'index', 'cisgender', 'serbs', 'contributors', 'aids', 'protest', 'america', 'bi-cultural', 'socially', 'gambling', 'disasters', 'ricans', 'censorship', 'divorce', 'biphobia', 'legal', 'telugus', 'militant', 'criticize', 'litigation', 'activist', 'latvian', 'consumer', 'andorran', 'employee', 'belarusian', 'chaldean', 'right', 'eating', 'civic', 'samoans', 'highway', 'foster', 'i-kiribati', 'american', 'depletion', 'rights', 'genetic', 'baby', 'catalan', 'scientific', 'colored', 'transphobia', 'assyrian', 'stress', 'bias', 'campaigner', 'theft', 'constitutional', 'confirmation', 'faroese', 'syriacs', 'engage', 'hondurans', 'pan', 'fijians', 'slovenes', 'attention', 'engineering', 'youth', 'mauritanians', 'political', 'zambian', 'suffrage', 'kodavas', 'hutus', 'kyrgyzs', 'abolition', 'manx', 'tanzanians', 'crime', 'prison', 'transgender', 'research', 'kongers', 'mandela', 'police', 'ghanaian', 'emphasis', 'unions', 'procedural', 'cuban', 'logging', 'sexuality', 'homosexual', 'jews', 'adoption', 'extinction', 'legislation', 'rape', 'threat', 'rotc', 'tobacco', 'cis', 'australian', 'limits', 'kodava', 'suicide', 'term', 'microaggression', 'deficit-hyperactivity', 'salvadoran', 'handicapped', 'dual', 'islamic', 'occupational', 'encourage', 'waste', 'human', 'cheating', 'regime', 'bolivians', 'mobilization', 'pro', 'college', 'tibetan', 'epidemics', 'scholarship', 'water', 'aquifer', 'organic', 'globalization', 'inuit', 'mixed-race', 'governmental', 'housing', 'cameroonians', 'injustice', 'dominicans', 'rooted', 'debt', 'tulus', 'societal', 'consolidation', 'nuclear', 'tobagonians', 'emancipation', 'drug', 'child', 'exchange', 'saint', 'australians', 'mobilize', 'kurd', 'nigerien', 'topic', 'neurodiverse', 'botswana', 'imposter', 'non-binary', 'mentor', 'invasion', 'abolitionist', 'karen', 'pakistanis', 'equality', 'tort', 'trinidadian', 'medical', 'fatigue', 'cajuns', 'urban', 'burundian', 'organized', 'vincentians', 'action', 'japanese', 'public', 'unemployment', 'stem', 'motivate', 'worldview', 'literacy', 'quebecers', 'bengalis', 'assyrians', 'plagiarism', 'cypriots', 'investing', 'socioeconomic', 'tobacco-related', 'poor', 'kosovar', 'humanitarian', 'panamanian', 'namibians', 'jailed', 'bureaucracy', 'deterioration', 'zulu', 'api', 'citizenship', 'transsexual', 'puerto', 'game', 'yemenis', 'disobedience', 'swede', 'single', 'welsh', 'tutsis', 'tung', 'punjabis', 'americans', 'zealanders', 'kenyans', 'nationalism', "children's", 'birth', 'marches', 'separation', 'burmese', 'temperance', 'preface', 'koreans', 'thai', 'diaspora', 'ruling', 'nigerian', 'polling', 'finder', 'bi', 'congestion', 'communist', 'tanzanian', 'accessibility', 'backyard', 'transit', 'barbadian', 'tutsi', 'grenadian', 'mysticism', 'bengali', 'berber', 'rioting', 'empower', 'cover', 'moroccans', 'philanthropy', 'segregation', 'evangelical', 'heart', 'islamist', 'celt', 'escalate', 'guineans', 'multiracial', 'albanians', 'maltes', 'samis', 'attribution', 'street', 'government', 'palauans', 'islander', 'syrians', 'dysphoria', 'biota', 'dominican', 'liberian', 'celts', 'filipino', 'social', 'malpractice', 'church-state', 'species', 'loan', 'z', 'sum', 'zero', 'kosovars', 'intersex', 'community', 'brazilian', 'swazi', 'french', 'conservation', 'cross', 'bankruptcy', 'reassignment', 'mansplain', 'liberians', 'reproductive', 'icelanders', 'danes', 'kitts', 'expression', 'buryat', 'finns', 'bme', 'libyan', 'relief', 'hungarian', 'oppression', 'kazakh', 'malawian', 'finnish', 'iraqis', 'programs', 'saudis', 'socialist', 'focus', 'newsletter', 'chadians', 'chronic', 'shareholder', 'cost', 'opposing', 'slovaks', 'vanuatuan', 'affinity', 'burundians', 'egyptians', 'security', 'census', 'bangladeshi', 'hiv', 'justice', 'disaster', 'corruption', 'indigenous', 'rwandan', 'mauritanian', 'bermudian', 'tajik', 'papua', 'racism', 'deference', 'dominant', 'croats', 'error', 'indian', 'belizean', 'sustainability', 'islanders', 'transplants', 'lucian', 'habitat', 'bahamians', 'algerian', 'nauruan', 'policy', 'seychelloi', 'environmentalist', 'palestinian', 'mexican', "veterans'", 'ethic', 'global', 'standards', 'national', 'authoritarian', 'buryats', 'involved', 'fascist', 'migrant', 'educate', 'solomon', 'lebanese', 'defamation', 'dane', 'namibian', 'malians', 'imprisoned', 'zionist', 'sammarinese', 'inclusive', 'sponsor', 'campaign', 'leader', 'epidemic', 'czech', 'semitism', 'guatemalans', 'lesbian', 'reference', 'advocacy', 'entrepreneurship', 'moldovans', 'poles', 'guatemalan', 'exploration', 'italian', 'nativism', 'medicare', 'vietnam', 'students:', 'corporal', 'korean', 'journalism', 'immigration', 'ethics', 'judicial', 'malaysian', 'riot', 'hallmark', 'boomers', 'groupthink', 'belgian', 'africans', 'philanthropic', 'lao', 'medicine', 'lesbophobia', 'argentine', 'anti-muslim', 'croat', 'malaysians', 'special', 'bosniak', 'marijuana', 'palauan', 'molestation', 'malayali', 'benefit', 'british', 'hacking', 'conservatism', 'greenlanders', 'lobbying', 'disorders', 'jordanians', 'risk', 'advocate', 'fascism', 'bahraini', 'maldivian', 'psychological', 'skepticism', 'secrecy', 'spaniard', 'safety', 'guianese', 'spark', 'femme', 'in-group', 'consciousness', 'chad', 'protestant', 'recycling', 'mixed-ethnicity', 'traffic', 'huaorani', 'technology', 'wetlands', 'fundraising', 'experimentation', 'media', 'disorder', 'nationalist', 'imprisonment', 'russians', 'arson', 'bosnians', 'bhutanese', 'nigerians', 'apartheid', 'commitment', 'profiling', 'rican', 'ecological', 'tenet', 'dei', 'republic', 'malays', 'ethical', 'boycott', 'evolution', 'farm', 'airline', 'gen', 'somalilander', 'solidarity', 'juvenile', 'rwandans', 'identity', 'red-lining', 'forefront', 'lennon', 'parenting', 'animal', 'passionate', 'culture', 'brazilians', 'iranians', 'dutch', 'russian', 'tomé', 'aesthetics', 'toxic', 'jurisprudence', 'trinidadians', 'leonean', 'georgian', 'minority', 'arts', 'congolese', 'euthanasia', 'burkinabés', 'graffito', 'unrest', 'environmentally-induced', 'poverty', 'telugu', 'food', 'ideological', 'skeptical', 'school', 'defense', 'preparedness', 'south', 'marxism', 'sexual', 'genocide', 'cultural', 'medicaid', 'laundering', 'governance', 'angami', 'albanian', 'hepeating', 'sex', 'swedish', 'commonwealth', 'kuwaiti', 'leoneans', 'academia', 'sotho', 'colombians', 'v.', 'aapi', 'cognitive', 'marshallese', 'disposal', 'guinean', 'freedom', 'silesians', 'heighten', 'indonesian', 'herzegovinians', 'timorese', 'courageous', 'indonesians', 'cameroonian', 'racial', 'reclaim', 'beninese', 'spirituality', 'liberal', 'finance', 'cambodian', 'tatars', 'filipinos', 'white', 'turks', 'venezuelans', 'norwegians', 'salvadorans', 'jamaican', 'icelander', 'peruvian', 'vouchers', 'crisis', 'gibraltarian', 'abortion', 'hungarians', 'criminal', 'engaging', 'virgin', 'nevi', 'legalize', 'journalistic', 'reform', 'romanian', 'iraqi', 'natural', 'african', 'tulu', 'israelis', 'deadnaming', 'dissent', 'angolan', 'awareness', 'civil', 'ism', 'socialism', 'automobile', 'métis', 'progressive', 'verdeans', 'democracy', 'advertising', 'punishment', 'serb', 'macedonian', 'accountability', 'ukrainian', 'sudanese', 'eritrean', 'singaporean', 'care', 'catalans', 'abrasion', 'money', 'germans', 'swis', 'emirati', 'wealth', 'moroccan', 'príncipe', 'liberalism', 'conscription', 'fostered', 'ukrainians', 'equity', 'taiwanese', 'comorian', 'guinea-bissau', 'egyptian', 'belgians', 'communism', 'issues', 'underrepresented', 'bruneians', 'environmental', 'cajun', 'grenadians', 'needle', 'property', 'pollution', 'turk', 'caucus', 'aromanian', 'voting', 'organ', 'actively', 'norwegian', 'tamil', 'corporate', 'barbudans', 'violence', 'fit', 'emiratis', 'sindhis', 'privatization', 'volunteering', 'eritreans', 'power', 'nation', 'greek', 'resource', 'kannadigas', 'outspoken', 'latino', 'palestinians', 'new', 'alternative', 'mauritians', 'scots', 'angolans', 'saudi', 'infrastructure', 'armenians', 'paraguayans', 'restraint', 'nauruans', 'barbadians', 'peace', 'ableism', 'estonian', 'greenlandic', 'austrian', 'bruneian', 'cell', 'feminism', 'organization', 'combine', 'liberties', 'welfare', 'konger', 'cambodians', 'lifelong', 'ethnocentrism', 'haitians', 'militia', 'outreach', 'domestic', 'chuvash', 'student', 'gibraltarians', 'tobagonian', 'breton', 'tongan', 'organize', 'zulus', 'kannadiga', 'anarchist', 'pedagogy', 'zimbabwean', 'harassment', 'nations', 'boer', 'creative', 'hutu', 'movement', 'inspire', 'montenegrins', 'uzbek', 'chilean', 'zealander', 'mongolians', 'swiss', 'dependency', 'wages', 'nagas', 'nonprofit', 'asian', 'trans', 'haitian', 'tissue', 'venezuelan', 'leftist', 'imperialism', 'tatar', 'samoan', 'loss:', 'health', 'aromanians', '“mixed”', 'vietnamese', 'postwar', 'ethiopian', 'leadership', 'intellectual', 'demonstration', 'disability', 'canadians', 'labor', 'scot', 'azerbaijani', 'awakening', 'jamaicans', 'syrian', 'campus', 'spearhead', 'vanuatuans', 'gentrification', 'quaker']

    activism = ['criticize','militant','governmental','spark','champion','emphasize','animal','diversity','theorize','newsletter','transgender','gender','rooted','guerrilla','outreach','ism','agenda','abolitionist','workplace','network','graffito','lifelong','nationalist','racist','mobilize','imprisonment','revolutionary','ideology','sexuality','educate','gandhi','skeptical','idealism','palestinian','semitism','disability','marxist','autism','organize','fundraising','spirituality','organization','campaigner','feminism','credibility','libertarian','youth','restraint','deference','movement','passionate','ethics','discrimination','journalism','cultural','tactic','reform','judicial','liberation','dictatorship','networking','bureaucracy','inspire','anti','resurgence','engagement','courageous','legitimacy','jurisprudence','behalf','awakening','amnesty','postwar','caucus','cute','progressive','socioeconomic','emphasis','stakeholder','engaging','anarchist','corporate','campus','ethical','opposing','segregate','mysticism','advocacy','naacp','abolition','nationalism','abortion','minority','societal','segregation','stance','community','spearhead','prostitution','sociological','lesbian','organizer','outspoken','advocate','leftist','devote','homo','political','protesting','liberalism','rights','cannabis','violence','media','constitutional','litigation','elite','islamic','internet','lobbying','engage','censorship','sustainable','environmentalist','accountability','empower','fostered','issue','marxism','latina','reproductive','marijuana','african','shareholder','conscription','homosexuality','mainstream','engaged','mobilization','oppression','islamist','homosexual','skepticism','jailed','socialism','authoritarian','entrepreneurship','motivate','vigil','commitment','activist','sustainability','campaign','equality','colored','protestant','slavery','environmental','tung','regime','emancipation','mandela','leadership','critique','riot','escalate','reclaim','globalization','politically','injustice','genocide','governance','intellectual','legislation','pornography','apartheid','optimism','handicapped','civic','racism','civil','initiative','peace','journalistic','disobedience','consumer','arrested','collective','radical','experimentation','capitalism','dissemination','democracy','legalize','worldview','combine','justice','involved','pro','fascism','affirmative','suffrage','queer','marches','socialist','ethic','evangelical','empowerment','hallmark','scholarship','judiciary','temperance','solidarity','vietnam','involvement','klan','conservatism','humanitarian','citizenship','conservative','student','actively','raising','forefront','ideological','politics','zionist','diaspora','focus','labor','procedural','pedagogy','communism','fascist','ruling','consciousness','epidemic','fuel','hiv','philanthropic','dissent','global','lennon','welfare','campaigning','rape','protest','defamation','communist','investing','philanthropy','substantive','tice','ecological','encourage','grassroots','aesthetics','imperialism','israeli','feminist','awareness','quaker','tenet','demonstration','academia','liberal','aids','unrest','boycott','imprisoned','participation','repression','socially','gay','heighten','nonprofit','indigenous','social','redefine']
    verbs_list = data_wd_list + activism

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
    # 10.입력한 에세이 문장에서 관련 단어가 얼마나 포함되어 있는지 포함비율 분석
    wd_ratio = round(len(get_words__)/len(input_text_list) *100, 3)

    ##########################################################################
    ##################  합격한 학생들의 평균점수 반영(실제 평균값 적용해야 함)  ###########
    admitted_std_score = 5 # 5%로 가정
    ##########################################################################

    # 합격한 학생들과 비교하여 5가지 구분 결과로 게산하기
    ini_enmt_cotri_score = lackigIdealOverboard(admitted_std_score, wd_ratio)

    # 추출한 단어 중복제거
    ext_words = list(set(get_words__))

    # print ("ACTION VERBS RATIO :", action_verbs_ratio )

    # rerurn 해석
    # wd_ratio : 입력한 에세이에 비교분석하고자하는 단어가 얼마나 포함되어 있는지에 대한 비율 계산
    # ext_words : 포함된 관련단어 추출 출격 --> 웹에 표시
    # ini_enmt_cotri_score : Supurb ~ Lacking 으로 결과 출력
    return wd_ratio, ext_words, ini_enmt_cotri_score


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


####  social_awareness_analysis ###
def initiative_engagement_contribution(essay_input):
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

    contribution_wd = ['cook','promote','guide','outreach','clean','service','organize','tutor','present','counsel','sacrifice','bake','host','provide','repair','educate','donate','lead','raise','perform','empower','create','mediate','improve','initiate','grant','sponsor','write','enhance','resolve','teach','foster','offer','give','endowment','benefact','enrich','distribute','adopt','develop','oblation','translate','care','share','help','gift','contribute','manage','dedicate','subsidy','volunteer','start','participate']

    verbs_list = verbs_list_ + academic_verbs + contribution_wd

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


# Topic uniqueness  -------!!!! 이 부분 값을 계산하도록 적용한 것인지 코드 리뷰할 것!
research_wd = ['generously','conclude','sat','resembles','suppose','picturesque','charming','due','scenic','chiefly','eye','personally','imagine','miraculous','brilliant','delicious','never','much','unduly','probably','incredible','especially','complex','everyone','deny','ask','view','regard','beliefis','certain','constantly','issue','pretend','fulfilling','must','mostly','excellent','complicated','conviction','get','lovely','perspective','obviously','seem','spectacular','mistaken','reckoning',"one's",'hold','pretty','thought','believe','naturally','initial','exceptional','right','sound','frequently','gather','unforeseeable','ridiculously','speaking','phenomenal','doubt','actual','top','viewpoint','mind','sterling','always','summarise','shadow','dreadfully','unexpected','position','appears','personal','positive','unforeseen','fair','clear','staggering','may','total','completely','exactly','pure','wicked','foolishly','surprising','highly','enjoyable','reckon','perfectly','sensational','surprisingly','opposite','say','particular','precisely','come','wa','conceivably','convinced','main','serious','stand','far','change','maybe','without','seriously','sight','imho','i’m','impression','bitterly','strongly','vantage','really','presumably','perfect','likely','infer','generally','exclusively','least','maintain','exquisite','wish','feeling','expressed','fairly','attractive','safely','would','large','rewarding','utter','consideration','glorious','one','confident','opinion','fact','idiot','i’d','prime','absolutely','assessment','postulate','increasingly','purely','admit','luckily','tremendous','marvellous','point','usually','go','part','taste','people','wonderful','evidence','particularly','thoughtfully','solely','terrific','suspect','disagree','presume','deadly','look','limited','understanding','giving','first-rate','noticed','feel','bravely','think','pleasing','knowledge','help','beautiful','absolute','suggest','seems','mixed','quality','awesome','idea','honest','difficult','impressive','remarkable','crazy','sheer','although','simply','certainly','methinks','unpredictable','magnificent','confidentially','best','case','head','pleasant','saying','thaㅅ','carelessly','it’s','take','cleverly','want','bet','matter','read','agree','fabulous','clearly','suggests','know','breathtaking','unbelievably','rightly','primarily','standing','extremely','find','theoretically','sitting','frankly','judgement','quite','cannot','truly','subject','plainly','extraordinary','classic','seen','assumes','minority','grand','later','superior','like','disappointingly','could','see','claimed','totally','belief','merely','whole','course','according','side','adverbs','sake','amazingly','possibly','care','reservation','outstanding','majestic','prof','humble','neither','gratifying','reasonable','consider','fantastic','way','alone','mainly','either','great','standpoint','reaction','incredibly','understand','conceit','might','weighing','well','continually','repeatedly','undoubtedly','shred','commenting','can’t','person','notably','judgment',"i'm",'entirely','old','even','expert','positively','perhaps','complete','wrong','money','high','thinking','consistently','pleasurable','said','situation','familiar','unique','indeed','definitely','unlikely','sit','tend','technically','argued','amazing','fortunately','typically','obvious','support','predominantly','argue','unusual','controversial','unbelievable','wisely','rather','book','i’ve','outta','observed','stunning',"i'd",'question','doubtless','frank','tell','towards','sure','heavily','superb','dare','concerned','satisfying','assume','enormously','won’t','experience','correct','conclusion','argument','topic','surely','guess','stupidly','estimation','given','delightful','imposing','mean','light','astonishing', 'prisoner','persuasive','issues','teen','gun','college','media','paternity','maternity','trump','crimes','pharmacy','treatment','lgbtq','work','donald','technology','criminal','fun','community','justice','prisons','reagan','hilary','simple','taxation','east','poverty','bullying','conflict','generational','history','computer','action','farming','pornography','literature','hazing','censorship','aids','interpersonal','psychology','mother','politics','unique','disasters','advertising','relationships','immigration','policy','military','cosmetic','date','debt','education','veterans','internet','religion','explosion','loan','controversial','prostitution','schools','natural','leave','security','rape','business','police','abortion','environment','discrimination','terrorism','copyright','barack','health','energy','ronald','affirmative','animals','family','day','hate','surgery','alcohol','women','obama','control','marketing','care','middle','medical','sociology','science','easy','communication','privacy','foreign','athletes','student','lotteries','population','bill','violence','drugs','gambling','clinton']





# 실행 #

essay_input = """This past summer, I had the privilege of participating in the University of Notre Dame’s Research Experience for Undergraduates (REU) program . Under the mentorship of Professor Wendy Bozeman and Professor Georgia Lebedev from the department of Biological Sciences, my goal this summer was to research the effects of cobalt iron oxide cored (CoFe2O3) titanium dioxide (TiO2) nanoparticles as a scaffold for drug delivery, specifically in the delivery of a compound known as curcumin, a flavonoid known for its anti-inflammatory effects. As a high school student trying to find a research opportunity, it was very difficult to find a place that was willing to take me in, but after many months of trying, I sought the help of my high school biology teacher, who used his resources to help me obtain a position in the program.				
Using equipment that a high school student could only dream of using, I was able to map apoptosis (programmed cell death) versus necrosis (cell death due to damage) in HeLa cells, a cervical cancer line, after treating them with curcumin-bound nanoparticles. Using flow cytometry to excite each individually suspended cell with a laser, the scattered light from the cells helped to determine which cells were living, had died from apoptosis or had died from necrosis. Using this collected data, it was possible to determine if the curcumin and/or the nanoparticles had played any significant role on the cervical cancer cells. Later, I was able to image cells in 4D through con-focal microscopy. From growing HeLa cells to trying to kill them with different compounds, I was able to gain the hands-on experience necessary for me to realize once again why I love science.				
Living on the Notre Dame campus with other REU students, UND athletes, and other summer school students was a whole other experience that prepared me for the world beyond high school. For 9 weeks, I worked, played and bonded with the other students, and had the opportunity to live the life of an independent college student.				
Along with the individually tailored research projects and the housing opportunity, there were seminars on public speaking, trips to the Fermi National Accelerator Laboratory, and one-on-one writing seminars for the end of the summer research papers we were each required to write. By the end of the summer, I wasn’t ready to leave the research that I was doing. While my research didn’t yield definitive results for the effects of curcumin on cervical cancer cells, my research on curcumin-functionalized CoFe2O4/TiO2 core-shell nanoconjugates indicated that there were many unknown factors affecting the HeLa cells, and spurred the lab to expand their research into determining whether or not the timing of the drug delivery mattered and whether or not the position of the binding site of the drugs would alter the results. Through this summer experience, I realized my ambition to pursue a career in research. I always knew that I would want to pursue a future in science, but the exciting world of research where the discoveries are limitless has captured my heart. This school year, the REU program has offered me a year-long job, and despite my obligations as a high school senior preparing for college, I couldn’t give up this offer, and so during this school year, I will be able to further both my research and interest in nanotechnology. """



print('social awareness ratio: ' , social_awareness_analysis(essay_input))
print('initiative_engagement_contribution:', initiative_engagement_contribution(essay_input))

ini_engage_re = initiative_engagement_contribution(essay_input)
ini_engage_5div_re = ini_engage_re[2][0] # supurb ~ lacking 로 결과나옴
print('ini_engage_5div_re : ', ini_engage_5div_re)
ini_engage_words = ini_engage_re[1] # initiative_engagement_contribution 관련 단어들로 웹에 표시
print('ini_engage_words : ', ini_engage_words)
ini_engage_fin_score_re = ini_engage_re[2][1] # 점수로 계산됨 ---> for overall score
print('ini_engage_fin_score_re : ', ini_engage_fin_score_re)



social_aware_re = social_awareness_analysis(essay_input)
social_aware_5div_re = social_aware_re[2][0] # supurb ~ lacking 로 결과나옴
print('social_aware_5div_re:', social_aware_5div_re)
social_aware_words = social_aware_re[1] # Social Awareness 관련 단어들로 웹에 표시
print('social_aware_words:', social_aware_words)
social_aware_fin_score_re = social_aware_re[2][1]
print('social_aware_fin_score_re:', social_aware_fin_score_re)