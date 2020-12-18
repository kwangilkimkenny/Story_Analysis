#Converted to Method...

import collections as coll
import math
import pickle
import string

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
from nltk.corpus import cmudict
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import nltk

nltk.download('cmudict')
nltk.download('stopwords')

style.use("ggplot")
cmuDictionary = None




# 텍스트 한 단락을 취해서 그것을 지정된 수의 문장으로 나눈다.
def slidingWindow(sequence, winSize, step=1):
    try:
        it = iter(sequence)
        print ('slideingWinows it :', it)
        #slideingWinows it : <str_iterator object at 0x7fe51178b910>  #작동할 때 결과
    except TypeError:
        raise Exception("**ERROR** sequence must be iterable.")
    if not ((type(winSize) == type(0)) and (type(step) == type(0))):
        raise Exception("**ERROR** type(winSize) and type(step) must be int.")
    if step > winSize:
        raise Exception("**ERROR** step must not be larger than winSize.")
    if winSize > len(sequence):
        raise Exception("**ERROR** winSize must not be larger than sequence length.")

    sequence = sent_tokenize(sequence)

    # 생략할 사전 계산 청크 수
    numOfChunks = int(((len(sequence) - winSize) / step) + 1)

    l = []
    for i in range(0, numOfChunks * step, step):
        l.append(" ".join(sequence[i:i + winSize]))

    return l



#모음수 세기
def syllable_count_Manual(word):
    word = word.lower()
    count = 0
    vowels = "aeiouy"
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
            if word.endswith("e"):
                count -= 1
    if count == 0:
        count += 1
    print("vowels numbers:", count)
    return count



# COUNTS NUMBER OF 음절을 소문자로 바꾸고 문자열만 가져와서 에세이 내에서 음절이 몇개인지 카운트
def syllable_count(word):
    global cmuDictionary #사전을 사용하여 syl 수를 세어서 가저옴
    d = cmuDictionary
    try:
        syl = [len(list(y for y in x if y[-1].isdigit())) for x in d[word.lower()]][0]
    except:
        syl = syllable_count_Manual(word)
    # print("syllable numbers:", syl)
    return syl




# removing stop words, 구두점 등의 문장기호 제거하고 단어 평균 길이 계사
def Avg_wordLength(str):
    str.translate(string.punctuation)
    tokens = word_tokenize(str, language='english')
    st = [",", ".", "'", "!", '"', "#", "$", "%", "&", "(", ")", "*", "+", "-", ".", "/", ":", ";", "<", "=", '>', "?",
          "@", "[", "\\", "]", "^", "_", '`', "{", "|", "}", '~', '\t', '\n', 'u00e9', '[', ']', 'n','n201', '\u201d' ]
    stop = stopwords.words('english') + st
    words = [word for word in tokens if word not in stop]
    return np.average([len(word) for word in words])




# 문장수 계산
def Avg_SentLenghtByCh(text):
    tokens = sent_tokenize(text)
    return np.average([len(token) for token in tokens])




# 문장의 평균 단어 수 반환
def Avg_SentLenghtByWord(text):
    tokens = sent_tokenize(text)
    return np.average([len(token.split()) for token in tokens])



# 불용어 제거하고 단어당 음절 수를 파악하여 가독성 계산에 반영
def Avg_Syllable_per_Word(text):
    tokens = word_tokenize(text, language='english')
    st = [",", ".", "'", "!", '"', "#", "$", "%", "&", "(", ")", "*", "+", "-", ".", "/", ":", ";", "<", "=", '>', "?",
          "@", "[", "\\", "]", "^", "_", '`', "{", "|", "}", '~', '\t', '\n']
    stop = stopwords.words('english') + st
    words = [word for word in tokens if word not in stop]
    syllabls = [syllable_count(word) for word in words]
    p = (" ".join(words))
    return sum(syllabls) / max(1, len(words))




# 청크 길이에서 정규화된 특수 문자 수
def CountSpecialCharacter(text):
    st = ["#", "$", "%", "&", "(", ")", "*", "+", "-", "/", "<", "=", '>',
          "@", "[", "\\", "]", "^", "_", '`', "{", "|", "}", '~', '\t', '\n']
    count = 0
    for i in text:
        if (i in st):
            count = count + 1
    return count / len(text)




def CountPuncuation(text):
    st = [",", ".", "'", "!", '"', ";", "?", ":", ";"]
    count = 0
    for i in text:
        if (i in st):
            count = count + 1
    return float(count) / float(len(text))



# 기능어 수를 파악, 물론 특수문자 제거
# 온라인 메시지 작성자 식별: 쓰기 유형 특성 및 분류 기법을 용

def CountFunctionalWords(text):
    functional_words = """a between in nor some upon
    about both including nothing somebody us
    above but inside of someone used
    after by into off something via
    all can is on such we
    although cos it once than what
    am do its one that whatever
    among down latter onto the when
    an each less opposite their where
    and either like or them whether
    another enough little our these which
    any every lots outside they while
    anybody everybody many over this who
    anyone everyone me own those whoever
    anything everything more past though whom
    are few most per through whose
    around following much plenty till will
    as for must plus to with
    at from my regarding toward within
    be have near same towards without
    because he need several under worth
    before her neither she unless would
    behind him no should unlike yes
    below i nobody since until you
    beside if none so up your
    """

    functional_words = functional_words.split()
    words = RemoveSpecialCHs(text)
    count = 0

    for i in text:
        if i in functional_words:
            count += 1

    return count / len(words)




# 표현다양성 파악하기 위해 추출해야 값
def hapaxLegemena(text):
    words = RemoveSpecialCHs(text)
    V1 = 0
    # dictionary comprehension . har word kay against value 0 kardi
    freqs = {key: 0 for key in words}
    for word in words: # 단어중 유일한 단어들이 몇개인가 세어서 
        freqs[word] += 1
    for word in freqs:
        if freqs[word] == 1:
            V1 += 1
    N = len(words)
    V = float(len(set(words)))
    R = 100 * math.log(N) / max(1, (1 - (V1 / V))) # 독특한 단어의 비율을 확률로 계산
    h = V1 / N
    return R, h




#동일한 단어가 얼마나 사용되는지 파악
def hapaxDisLegemena(text):
    words = RemoveSpecialCHs(text)
    count = 0

    freqs = coll.Counter()
    freqs.update(words)
    for word in freqs:
        if freqs[word] == 2:
            count += 1

    h = count / float(len(words))
    S = count / float(len(set(words)))
    return S, h




#어휘 풍부성 파악
def AvgWordFrequencyClass(text):
    words = RemoveSpecialCHs(text)
    # dictionary comprehension . har word kay against value 0 kardi
    freqs = {key: 0 for key in words}
    for word in words:
        freqs[word] += 1
    maximum = float(max(list(freqs.values())))
    return np.average([math.floor(math.log((maximum + 1) / (freqs[word]) + 1, 2)) for word in words])




# 총 단어를 토큰화하여 중복되지 않는 토큰의 비율 계산
def typeTokenRatio(text):
    words = word_tokenize(text)
    return len(set(words)) / len(words)




# logW = V-a/log(N)
# N = total words , V = vocabulary richness (unique words) ,  a=0.17
# 문장내 다른 단어들을 파악하기 위해서 설정
def BrunetsMeasureW(text):
    words = RemoveSpecialCHs(text)
    a = 0.17
    V = float(len(set(words)))
    N = len(words)
    B = (V - a) / (math.log(N))
    return B




def RemoveSpecialCHs(text):
    text = word_tokenize(text)
    st = [",", ".", "'", "!", '"', "#", "$", "%", "&", "(", ")", "*", "+", "-", ".", "/", ":", ";", "<", "=", '>', "?",
          "@", "[", "\\", "]", "^", "_", '`', "{", "|", "}", '~', '\t', '\n']

    words = [word for word in text if word not in st]
    return words




# K  10,000 * (M - N) / N**2
# , where M  Sigma i**2 * Vi.
def YulesCharacteristicK(text):
    words = RemoveSpecialCHs(text)
    N = len(words)
    freqs = coll.Counter()
    freqs.update(words)
    vi = coll.Counter()
    vi.update(freqs.values())
    M = sum([(value * value) * vi[value] for key, value in freqs.items()])
    K = 10000 * (M - N) / math.pow(N, 2)
    return K




# -1*sigma(pi*lnpi)
# Shannon과 Simpsons 다양성 지수 활용
def ShannonEntropy(text):
    words = RemoveSpecialCHs(text)
    lenght = len(words)
    freqs = coll.Counter()
    freqs.update(words)
    arr = np.array(list(freqs.values()))
    distribution = 1. * arr
    distribution /= max(1, lenght)
    import scipy as sc
    H = sc.stats.entropy(distribution, base=2)
    # H = sum([(i/lenght)*math.log(i/lenght,math.e) for i in freqs.values()])
    return H




# 1 - (sigma(n(n - 1))/N(N-1)
# N is total number of words
# n is the number of each type of word
def SimpsonsIndex(text):
    words = RemoveSpecialCHs(text)
    freqs = coll.Counter()
    freqs.update(words)
    N = len(words)
    n = sum([1.0 * i * (i - 1) for i in freqs.values()])
    D = 1 - (n / (N * (N - 1)))
    return D




def FleschReadingEase(text, NoOfsentences):
    words = RemoveSpecialCHs(text)
    l = float(len(words))
    scount = 0
    for word in words:
        scount += syllable_count(word)

    I = 206.835 - 1.015 * (l / float(NoOfsentences)) - 84.6 * (scount / float(l))
    return I




def FleschCincadeGradeLevel(text, NoOfSentences):
    words = RemoveSpecialCHs(text)
    scount = 0
    for word in words:
        scount += syllable_count(word)

    l = len(words)
    F = 0.39 * (l / NoOfSentences) + 11.8 * (scount / float(l)) - 15.59
    return F



def dale_chall_readability_formula(text, NoOfSectences):
    words = RemoveSpecialCHs(text)
    difficult = 0
    adjusted = 0
    NoOfWords = len(words)
    #with open('J:\Django\EssayFit_Django\essayfitaiproject\essayai\data\dale-chall.pkl', 'rb') as f:
    #with open('/Users/jongphilkim/Desktop/Django_WEB/01_ESSAYFITAI_11-02/essayfitaiproject/essayai/data/dale-chall.pkl', 'rb') as f:
    with open('dale-chall.pkl', 'rb') as f:
        fimiliarWords = pickle.load(f)
    for word in words:
        if word not in fimiliarWords:
            difficult += 1
    percent = (difficult / NoOfWords) * 100
    if (percent > 5):
        adjusted = 3.6365
    D = 0.1579 * (percent) + 0.0496 * (NoOfWords / NoOfSectences) + adjusted
    return D



def GunningFoxIndex(text, NoOfSentences):
    words = RemoveSpecialCHs(text)
    NoOFWords = float(len(words))
    complexWords = 0
    for word in words:
        if (syllable_count(word) > 2):
            complexWords += 1

    G = 0.4 * ((NoOFWords / NoOfSentences) + 100 * (complexWords / NoOFWords))
    return G


def PrepareData(text1, text2, Winsize):
    chunks1 = slidingWindow(text1, Winsize, Winsize)
    chunks2 = slidingWindow(text2, Winsize, Winsize)
    return " ".join(str(chunk1) + str(chunk2) for chunk1, chunk2 in zip(chunks1, chunks2))




# 글자의 특징들을 모두 추출하여 백터값을 반환
def FeatureExtration(text, winSize, step):
    # cmu dictionary for syllables
    global cmuDictionary
    cmuDictionary = cmudict.dict()

    chunks = slidingWindow(text, winSize, step)

    vector = []

    for chunk in chunks:

        feature = []

        # VOCABULARY 풍부성 특징들 파악
        # 총 단어를 토큰화하여 중복되지 않는 토큰의 비율 계산
        TTratio = round(typeTokenRatio(chunk), 2)
        feature.append(TTratio)
        print("TTratio 표현다양성: ", TTratio)
        # token_ratio.append(TTratio)

        # 표현다양성 파악(전체 단어에서 유일한 단어 사용비율)
        HonoreMeasureR, hapax = hapaxLegemena(chunk)
        feature.append(HonoreMeasureR)
        print('HonoreMeasureR 표현다양성 :', HonoreMeasureR)
        feature.append(hapax)
        print('hapax 표현다양성 :', hapax )


        #다양성 지수 분석
        YuleK = round(YulesCharacteristicK(chunk),2)
        feature.append(YuleK)
        print('YuleK 다양성지수:',YuleK )
        # YuleK__.append(YuleK)

        # Shannon과 Simpsons 다양성 지수 활용
        S = round(SimpsonsIndex(chunk), 2)
        feature.append(S)
        print('SimpsonIndex 다양성지수', S)
        # S__.append(S)

        Shannon = round(ShannonEntropy(text),2)
        feature.append(Shannon)
        # Shannon__.append(Shannon)
        print('Shannon 다양성지수:', Shannon)

        # 가독성
        FR = round(FleschReadingEase(chunk, winSize),2)
        feature.append(FR)
        print("FR_readerablity :", FR)

        vector.append(feature)

    return vector





text_input = """A window into the soul.For most people, this would be the eyes. The eyes cannot lie; they often tell more about a person's emotions than their words. What distinguishes a fake smile from a genuine one? The eyes. What shows sadness? The eyes. What gives away a liar? The eyes.But are the eyes the only window into the soul?Recently, I began painting with watercolors. With watercolors, there is no turning back: if one section is too dark, it is nearly impossible to lighten the area again. Every stroke must be done purposefully, every color mixed to its exact value.I laid my materials before me, preparing myself for the worst. I checked my list of supplies, making sure my setup was perfect.I wet my brush, dipped it into some yellow ochre, and dabbed off the excess paint. Too little water on my brush. I dipped my brush back into my trusty water jar; the colors swirled beautifully, forming an abstract art piece before my eyes. \u2014It's a shame that I couldn't appreciate it.I continued mixing colors to their exact value. More alizarin crimson. More water. More yellow ochre. Less water. More phthalo blue. The cycle continued. Eventually, I was satisfied. The colors looked good, there was enough contrast between facial features, and the watercolors stayed inside the lines.Craving feedback, I posted my art to Snapchat. I got a few messages such as 'wow' and 'pretty,' but one message stood out. 'You were anxious with this one, huh? Anyways, love the hair!'I was caught off guard. Was it a lucky guess? Did they know something I didn't? I immediately responded: 'Haha, how could you tell?' No response.What I didn't know at the time was that my response would come a few months later while babysitting. Since the girl I was babysitting loved art, I took out some Crayola watercolors and some watercolor paper for her to play with. After I went to the bathroom and came back, the watercolors were doused with water. 'You were impatient with this one, huh? Anyways, love the little dog you drew!'The little girl looked up at me, confused. 'How could you tell?' 'You used a lot of water for a brighter color, but you couldn't wait for it to slowly soak in.''Oh.'Now, I would be lying if I said I realized the connection between the two events immediately.Instead, I made the connection when I decided to sit down one day and objectively critique my art. The piece that I once loved now seemed like a nervous wreck: the paper was overworked, the brushstrokes were undecided, the facial features blended together, and each drop of water was bound inside the lines as if it was a prisoner in a cage.From then on, I started noticing pieces of personality in additional creations surrounding me: website designs, solutions to math problems, code written for class, and even the preparation of a meal.When I peer around at people's projects during Code Club, I notice the clear differences between their code. Some people break it up by commenting in every possible section. Others breeze through the project, not caring to comment or organize their code. I could also see clear differences in personalities when our club members began coding the Arduino for the first time. Some followed the tutorials to the letter, while others immediately started experimenting with different colored LEDs and ways of wiring the circuit.It became clear to me that, as humans, we leave pieces of our souls in everything we do, more than we intend to. If we entertain this thought, perhaps the key to better understanding others around us is simply noticing the subtler clues under our noses?Perhaps there are endless windows to the soul, and we simply need to peer through them. I shakily rose my hand. 'We should create workshops of our own,' I suggested.I got a few strange looks. 'It's a good idea, but it's too much work.' 'We just don't have enough free time to make it work.' 'Maybe we could, but I don't know how to make workshops.' My suggestion was shot down. I shuffled in my seat. 'I could make them.' A few people stared at me in disbelief. I glanced over at the club advisor, Mr. C, nervous to hear his response.'If you're willing to take on the work, we can try it.' Mr. C replied. And so I embarked on my quest. I researched different workshops on the internet, learning the information myself at first. Then, I transitioned into creating workshops of my own, making sure that the information was easy to understand for even a beginner. I was exhausted; my first workshop took 16 cumulative hours to create."""



extracted_features_result = FeatureExtration(text_input, winSize=10, step=10) #특징들을 모두 추출하여 출력할 거임

df = pd.DataFrame(extracted_features_result, columns = ['TTratio_WD', 'HonoreMeasureR_WD', 'hapax_WD', 'YuleK_WD', 'SimpsonIndex_WD', 'Shannon_WD', 'FR_readerablity'])
print(df)

#표현의 다양성
TTratio_WD_mean = [df['TTratio_WD'].mean()] * 100
HonoreMeasureR_WD_mean = df['HonoreMeasureR_WD'].mean()
hapax_WD_mean = df['hapax_WD'].mean() * 1000
YuleK_WD_mean = df['YuleK_WD'].mean() * 0.1
SimpsonIndex_WD_mean = df['SimpsonIndex_WD'].mean() * 100
Shannon_WD_mean = df['Shannon_WD'].mean() * 100


# Writing Diversity ratio
wri_div_mean = (TTratio_WD_mean + HonoreMeasureR_WD_mean + 
                hapax_WD_mean + YuleK_WD_mean + HonoreMeasureR_WD_mean + 
                SimpsonIndex_WD_mean + Shannon_WD_mean)/6 
wri_div_result = wri_div_mean[0]
print('Writing Diversity ratio :', wri_div_result)     

# 가독성
FR_readerablity_mean = df['FR_readerablity'].mean()
print('Readablity:', FR_readerablity_mean)  


#==== 비교계산 시작 ==== writing diversity
one_ps_char_desc_wri_div = wri_div_result  # 이하 -13 ~ +17 이상 범위로 Lacking , Ideal, Overboard 
ideal_mean_wri_div = 477.8039583 # 1000명의 평균값 (writing diversity)

#==== 비교계산 시작 ==== readablity
one_ps_char_desc_readablity = FR_readerablity_mean  # 이하 -13 ~ +17 이상 범위로 Lacking , Ideal, Overboard 
ideal_mean_readablity = 50.71931 # 1000명의 평균값 (readablity



def cal_laking_ideal_overboard(one_ps_char_desc, ideal_mean):
    min_ = int(ideal_mean-ideal_mean*0.6)
    print('min_', min_)
    max_ = int(ideal_mean+ideal_mean*0.6)
    print('max_: ', max_)
    div_ = int(((ideal_mean+ideal_mean*0.6)-(ideal_mean-ideal_mean*0.6))/3)
    print('div_:', div_)


    cal_abs = abs(ideal_mean - one_ps_char_desc) # 개인 - 단체 값의 절대값계산

    print('cal_abs 절대값 :', cal_abs)
    compare = (one_ps_char_desc + ideal_mean)/7
    print('compare :', compare)

    if one_ps_char_desc > ideal_mean: # 개인점수가 평균보다 클 경우는 overboard
        if cal_abs > compare: # 개인점수가 개인평균차의 절대값보다 클 경우, 즉 차이가 많이 날경우
            print("Overboard")
            result = 2
        else: #차이가 많이 안나면
            print("Ideal")
            result = 1
            
        
    elif one_ps_char_desc < ideal_mean: # 개인점수가 평균보다 작을 경우 lacking
        if cal_abs > compare: #차이가 많이나면 # 개인점수가  평균보다 작을 경우 Lacking이고 
            print("Lacking")
            result = 0
        else: #차이가 많이 안나면
            print ("Ideal")
            result = 1
            
    else:
        print("Ideal")
        result = 1
    
    return result

#######################################################################################################


# WRITING DIVERSITY  실행  - 결과

writing_diversity_result_fin = cal_laking_ideal_overboard(one_ps_char_desc_wri_div, ideal_mean_wri_div)

print("==================================================")
print("WRITING DIVERSITY :", writing_diversity_result_fin)
print("==================================================")



# READABLITY 실행 - 결과
readablity_cal_result_fin = cal_laking_ideal_overboard(one_ps_char_desc_readablity, ideal_mean_readablity)
print("==================================================")
print("READABLITY :", readablity_cal_result_fin)
print("==================================================")




# Writing Diversity ratio : 479.53898144577875
# Readablity: 67.29
# min_ 191
# max_:  764
# div_: 191
# cal_abs 절대값 : 1.7350231457787686
# compare : 136.76327710653982
# Ideal
# ==================================================
# WRITING DIVERSITY : 1  --------> 0: lacking, 1: ideal, 2: overboard
# ==================================================
# min_ 20
# max_:  81
# div_: 20
# cal_abs 절대값 : 16.570690000000006
# compare : 16.858472857142857
# Ideal
# ==================================================
# READABLITY : 1 --------> 0: lacking, 1: ideal, 2: overboard
# ==================================================