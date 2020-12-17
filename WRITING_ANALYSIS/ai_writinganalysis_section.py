#Converted to Method...

import collections as coll
import math
import pickle
import string

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

    # #문장분석 특징 출력
    # #평균 단어 길이 등등...
    # avgWordLength = []

    # #문장수 계산
    # avgSentLength = []

    # #문장에 단어가 평균적으로 몇개가 있는지 계산
    # avgSentLengthByWord = []

    # # 불용어 제거하고 단어당 음절 수를 파악하여 가독성 계산에 반영  >> 가독성판단
    # avgSyllablePerWord = []

    # # 문장에 특수문자가 얼마나 포함되었는지 계산
    # count_spcial_chr_nums = []

    # # 구두점 수 계산
    # cont_punc_numbers = []

    # # 문법적 구조를 표현하는데 사용하는 function word의 사용 수
    # cont_func_word = []

    # # 총 단어를 토큰화하여 중복되지 않는 토큰의 비율 계산
    # token_ratio = []

    # #표현다양성 파악(전체 단어에서 유일한 단어 사용비율)
    # HonoreMeasureR__ = []
    # hapax__ = []

    # #동일한 단어가 얼마나 사용되는지 파악
    # SichelesMeasureS__ = []

    # dihapax__ = []

    # #Yule's index
    # YuleK__ = []

    # #Shannon과 Simpsons 다양성 지수 활용
    # Shannon__ = []

    # # 문장 표현의 다양성 지수로 0으로 가까우면 다양한 표현이 되고, 1에 가까우면 다양하지 않은, 즉 독창적인 단어들로 구성되 있다는 의미
    # S__ = []

    # # 문장내 다른 단어(unique 단어)들을 파악하기 위해서 설정
    # B__ = []

    # # 가독성 파악 지수
    # FR__ = []

    # #The Flesch Reading Ease formula will output a number from 0 to 100 - a higher score indicates easier reading. 
    # #An average document has a Flesch Reading Ease score between 6 - 70. As a rule of thumb, scores of 90-100 can be understood by an average 5th grader. 
    # #8th and 9th grade students can understand documents with a score of 60-70; and college graduates can understand documents with a score of 0-30.
    # FC__ = []

    # # 표현이 다른 특징들 추출하여 수집
    # # 점수	                    설명
    # # 4.9 이하	평균 4 학년 이하 학생이 쉽게 이해할 수 있음
    # # 5.0 ~ 5.9	평균 5 학년 또는 6 학년 학생이 쉽게 이해할 수 있음
    # # 6.0 ~ 6.9	평균 7 학년 또는 8 학년 학생이 쉽게 이해할 수 있음
    # # 7.0 ~ 7.9	평균 9 학년 또는 10 학년 학생이 쉽게 이해할 수 있음
    # # 8.0 ~ 8.9	평균 11 학년 또는 12 학년 학생이 쉽게 이해할 수 있음
    # # 9.0 ~ 9.9	평균 13 ~ 15 학년 (대학) 학생이 쉽게 이해할 수 있음
    # D__ = []

    # #가독성 분석
    # # Fog Index	Reading level by grade
    # # 17	    College graduate
    # # 16	    College senior
    # # 15	    College junior
    # # 14	    College sophomore
    # # 13	    College freshman
    # # 12	    High school senior
    # # 11	    High school junior
    # # 10	    High school sophomore
    # # 9	    High school freshman
    # # 8	    Eighth grade
    # # 7	    Seventh grade
    # # 6	    Sixth grade
    # G__ = []

    vector = []
    # extracted_features =[]
    for chunk in chunks:
        ###########특징들 별도 저장 후 출력
        # avgWordLength = []
        # avgSentLength = []
        # avgSentLengthByWord = []
        # avgSyllablePerWord = []
        # count_spcial_chr_nums = []
        # cont_punc_numbers = []
        # cont_func_word = []
        # token_ratio = []
        # HonoreMeasureR__ = []
        # hapax__ = []
        # SichelesMeasureS__ = []
        # dihapax__ = []
        # YuleK__ = []
        # Shannon__ = []
        # S__ = []
        # B__ = []
        # FR__ = []
        # FC__ = []
        # D__ = []
        # G__ = []
        ########################
        feature = []

        # LEXICAL 특징들

        # meanwl = (Avg_wordLength(chunk))
        # feature.append(meanwl)
        # # avgWordLength.append(meanwl)

        # meansl = (Avg_SentLenghtByCh(chunk))
        # feature.append(meansl)
        # # avgSentLength.append(meansl)

        # 문장에 단어가 평균적으로 몇개가 있는지 계산
        # mean = (Avg_SentLenghtByWord(chunk))
        # feature.append(mean)
        # # avgSentLengthByWord.append(mean)

        # #불용어 제거하고 단어당 음절 수를 파악하여 가독성 계산에 반영
        # meanSyllable = Avg_Syllable_per_Word(chunk)
        # feature.append(meanSyllable)
        # avgSyllablePerWord.append(meanSyllable)

        # means = CountSpecialCharacter(chunk)
        # feature.append(means)
        # count_spcial_chr_nums.append(means)

        # p = CountPuncuation(chunk)
        # feature.append(p)
        # cont_punc_numbers.append(p)

        # f = CountFunctionalWords(text)
        # feature.append(f)
        # # cont_func_word.append(f)

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
        # hapax__.append(hapax)
        # HonoreMeasureR__.append(HonoreMeasureR)

        # SichelesMeasureS, dihapax = hapaxDisLegemena(chunk)
        # feature.append(dihapax)
        # feature.append(SichelesMeasureS)
        # # SichelesMeasureS__.append(SichelesMeasureS)
        # dihapax__.append(dihapax)

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

        # B = BrunetsMeasureW(chunk)
        # feature.append(B)
        # # B__.append(B)

        Shannon = round(ShannonEntropy(text),2)
        feature.append(Shannon)
        # Shannon__.append(Shannon)
        print('Shannon 다양성지수:', Shannon)

        # 가독성
        FR = round(FleschReadingEase(chunk, winSize),2)
        feature.append(FR)
        print("FR_readerablity :", FR)
        # FR__.append(FR)

        # FC = round(FleschCincadeGradeLevel(chunk, winSize),2)
        # feature.append(FC)
        # # FC__.append(FC)

        # # 표현이 다른 특징들 추출하여 수집
        # D = dale_chall_readability_formula(chunk, winSize)
        # feature.append(D)
        # # D__.append(D)

        # # 다른 특징들 추출하여 수집
        # G = GunningFoxIndex(chunk, winSize)
        # feature.append(G)
        # # G__.append(G)

        vector.append(feature)

    return vector





# # ELBOW METHOD - 분류값을 계산하기 위해서 적용, 현재는 분류하지 않음. 1개의 에세이만 분석할 것
# def ElbowMethod(data):
#     X = data  # <your_data>
#     distorsions = []
#     for k in range(1, 10): #최대 10개의 군집까지 분류할 수 있도록
#         kmeans = KMeans(n_clusters=k)
#         kmeans.fit(X)
#         distorsions.append(kmeans.inertia_)

#     fig = plt.figure(figsize=(15, 5))
#     plt.plot(range(1, 10), distorsions, 'bo-')
#     plt.grid(True)
#     plt.ylabel("Square Root Error")
#     plt.xlabel("Number of Clusters")
#     plt.title('Elbow curve')
#     plt.savefig("ElbowCurve.png")
#     plt.show()


# -----------------------------------------------------------------------------------------
# ANALYSIS PART

# 엘보우 그래프를 이용해서 몇개의 k 값을 넣을 것이지 파악하여 계산에 적용, 현재는 1개의 에세이 스타일만 파악할 것임
# # 1000의 에세이를 파악하여 공통점을 찾을 계획임
# def Analysis(vector, K=1):
#     arr = (np.array(vector))
#     # mean normalization of the data . converting into normal distribution having mean=0 , -0.1<x<0.1
#     sc = StandardScaler()
#     x = sc.fit_transform(arr)

#     # Breaking into principle components
#     pca = PCA(n_components=2)
#     components = (pca.fit_transform(x))
#     # Applying kmeans algorithm for finding centroids

#     kmeans = KMeans(n_clusters=K, n_jobs=-1)
#     kmeans.fit_transform(components)
#     #print("labels: ", kmeans.labels_)
#     centers = kmeans.cluster_centers_

#     # lables are assigned by the algorithm if 2 clusters then lables would be 0 or 1
#     lables = kmeans.labels_
#     colors = ["r.", "g.", "b.", "y.", "c."]
#     colors = colors[:K + 1]

#     return components

#     # for i in range(len(components)):
#     #     plt.plot(components[i][0], components[i][1], colors[lables[i]], markersize=10)

#     # plt.scatter(centers[:, 0], centers[:, 1], marker="x", s=150, linewidths=10, zorder=15)
#     # plt.xlabel("1st Principle Component")
#     # plt.ylabel("2nd Principle Component")
#     # title = "Styles Clusters"
#     # plt.title(title)
#     # plt.savefig("Results" + ".png")
#     # plt.show()


##################################

# if __name__ == '__main__':

#     text = open("/Users/kimkwangil/Documents/001_ESSAYFITAI/01_WEB_2020-10-17/essayfitaiproject/essayai/data/essay_sample.txt").read()

#     vector = FeatureExtration(text, winSize=10, step=10)
#     #ElbowMethod(np.array(vector)) 사용하지 않을 것임(이미 적용함 K=1)
#     Analysis(vector)

##################################
# 모든 분석 항목을 개별 결과로 추출하여 리스트에 담고 이것을 테이블로 result_all.html 에 구현할것

### [
#     [5.417910447761194, 62.0, 10.9, 1.7761194029850746, 0.0, 0.03656597774244833, 0.5493133583021224, 0.6946564885496184, 0.6607142857142857, 471.8498871295094, 0.05357142857142857, 0.06976744186046512, 4529.655612244898, 0.9911518661518661, 18.190107138129314, 7.9451896491259655, 66.30092857142861, 6.794071428571428, 9.549341428571429, 9.48], [5.21875, 49.2, 8.6, 1.625, 0.0, 0.03792415169660679, 0.5493133583021224, 0.6601941747572816, 0.6022727272727273, 447.7336814478207, 0.07954545454545454, 0.1076923076923077, 3959.194214876033, 0.9879832810867294, 14.479589695008315, 7.9451896491259655, 71.96436363636367, 5.40790909090909, 10.711957272727274, 6.247272727272727], 
#     [5.390625, 57.0, 9.9, 1.703125, 0.0, 0.0535405872193437, 0.5493133583021224, 0.6935483870967742, 0.6831683168316832, 461.512051684126, 0.06930693069306931, 0.0875, 4879.913733947652, 0.9914851485148515, 17.297489785735493, 7.9451896491259655, 73.4528069306931, 5.523257425742575, 9.452905544554456, 8.000396039603961], 
#     [5.579710144927536, 125.5, 22.0, 1.7826086956521738, 0.0, 0.03322784810126582, 0.5493133583021224, 0.6126482213438735, 0.5223214285714286, 541.164605185504, 0.08035714285714286, 0.12080536912751678, 3124.6014030612246, 0.9906710442024343, 27.501798634628567, 7.9451896491259655, 57.5766785714286, 10.793321428571428, 9.75240607142857, 13.96], 
#     [5.623762376237623, 97.0, 17.0, 1.7722772277227723, 0.0, 0.03575076608784474, 0.5493133583021224, 0.655, 0.5885714285714285, 516.4785973923515, 0.09714285714285714, 0.13385826771653545, 3934.6938775510203, 0.9919868637110016, 24.556680691194547, 7.9451896491259655, 64.34792857142861, 8.631571428571426, 9.46707142857143, 11.114285714285714], [5.426229508196721, 54.5, 10.0, 1.7704918032786885, 0.0, 0.04332129963898917, 0.5493133583021224, 0.6446280991735537, 0.6039603960396039, 461.512051684126, 0.06930693069306931, 0.0945945945945946, 3911.3812371336144, 0.9889108910891089, 15.997415393722303, 7.9451896491259655, 70.10231188118814, 5.990584158415842, 9.29656891089109, 8.792475247524752]
###  ]

# def ext_each_features(text):
#     my_posts = str(text)
#     ext_features_re = FeatureExtration(text, winSize=10, step=10)
    
#     return ext_each_features


# def writingstyle(text):
#     my_posts = str(text)
#     vector = FeatureExtration(text, winSize=10, step=10)
#     #ElbowMethod(np.array(vector)) 사용하지 않을 것임(이미 적용함 K=1)
#     result = Analysis(vector)
#     return result


# def writestyleResult(input_text__) :

#     #input_text__ = """A window into the soul.For most people, this would be the eyes. The eyes cannot lie; they often tell more about a person's emotions than their words. What distinguishes a fake smile from a genuine one? The eyes. What shows sadness? The eyes. What gives away a liar? The eyes.But are the eyes the only window into the soul?Recently, I began painting with watercolors. With watercolors, there is no turning back: if one section is too dark, it is nearly impossible to lighten the area again. Every stroke must be done purposefully, every color mixed to its exact value.I laid my materials before me, preparing myself for the worst. I checked my list of supplies, making sure my setup was perfect.I wet my brush, dipped it into some yellow ochre, and dabbed off the excess paint. Too little water on my brush. I dipped my brush back into my trusty water jar; the colors swirled beautifully, forming an abstract art piece before my eyes. \u2014It's a shame that I couldn't appreciate it.I continued mixing colors to their exact value. More alizarin crimson. More water. More yellow ochre. Less water. More phthalo blue. The cycle continued. Eventually, I was satisfied. The colors looked good, there was enough contrast between facial features, and the watercolors stayed inside the lines.Craving feedback, I posted my art to Snapchat. I got a few messages such as 'wow' and 'pretty,' but one message stood out. 'You were anxious with this one, huh? Anyways, love the hair!'I was caught off guard. Was it a lucky guess? Did they know something I didn't? I immediately responded: 'Haha, how could you tell?' No response.What I didn't know at the time was that my response would come a few months later while babysitting. Since the girl I was babysitting loved art, I took out some Crayola watercolors and some watercolor paper for her to play with. After I went to the bathroom and came back, the watercolors were doused with water. 'You were impatient with this one, huh? Anyways, love the little dog you drew!'The little girl looked up at me, confused. 'How could you tell?' 'You used a lot of water for a brighter color, but you couldn't wait for it to slowly soak in.''Oh.'Now, I would be lying if I said I realized the connection between the two events immediately.Instead, I made the connection when I decided to sit down one day and objectively critique my art. The piece that I once loved now seemed like a nervous wreck: the paper was overworked, the brushstrokes were undecided, the facial features blended together, and each drop of water was bound inside the lines as if it was a prisoner in a cage.From then on, I started noticing pieces of personality in additional creations surrounding me: website designs, solutions to math problems, code written for class, and even the preparation of a meal.When I peer around at people's projects during Code Club, I notice the clear differences between their code. Some people break it up by commenting in every possible section. Others breeze through the project, not caring to comment or organize their code. I could also see clear differences in personalities when our club members began coding the Arduino for the first time. Some followed the tutorials to the letter, while others immediately started experimenting with different colored LEDs and ways of wiring the circuit.It became clear to me that, as humans, we leave pieces of our souls in everything we do, more than we intend to. If we entertain this thought, perhaps the key to better understanding others around us is simply noticing the subtler clues under our noses?Perhaps there are endless windows to the soul, and we simply need to peer through them. I shakily rose my hand. 'We should create workshops of our own,' I suggested.I got a few strange looks. 'It's a good idea, but it's too much work.' 'We just don't have enough free time to make it work.' 'Maybe we could, but I don't know how to make workshops.' My suggestion was shot down. I shuffled in my seat. 'I could make them.' A few people stared at me in disbelief. I glanced over at the club advisor, Mr. C, nervous to hear his response.'If you're willing to take on the work, we can try it.' Mr. C replied. And so I embarked on my quest. I researched different workshops on the internet, learning the information myself at first. Then, I transitioned into creating workshops of my own, making sure that the information was easy to understand for even a beginner. I was exhausted; my first workshop took 16 cumulative hours to create."""
#     #input_text__ = open("/Users/kimkwangil/Documents/001_ESSAYFITAI/01_WEB_2020-10-17/essayfitaiproject/essayai/data/essay_sample.txt").read()
#     print("input_text__:", input_text__)
#     result__ = writingstyle(input_text__)
#     print (result__)
#     print (type(result__)) #numpy array

#     return result__

# result__  이것으로 그래프를 그리게 되는 거임

# [[-1.53070217 -1.96372503]
#  [-2.97194764  3.56488797]
#  [-3.16646339 -1.8099431 ]
#  [ 5.75572062  0.63072175]
#  [ 3.17393929 -0.5196705 ]
#  [-1.2605467   0.0977289 ]]

text_input = """A window into the soul.For most people, this would be the eyes. The eyes cannot lie; they often tell more about a person's emotions than their words. What distinguishes a fake smile from a genuine one? The eyes. What shows sadness? The eyes. What gives away a liar? The eyes.But are the eyes the only window into the soul?Recently, I began painting with watercolors. With watercolors, there is no turning back: if one section is too dark, it is nearly impossible to lighten the area again. Every stroke must be done purposefully, every color mixed to its exact value.I laid my materials before me, preparing myself for the worst. I checked my list of supplies, making sure my setup was perfect.I wet my brush, dipped it into some yellow ochre, and dabbed off the excess paint. Too little water on my brush. I dipped my brush back into my trusty water jar; the colors swirled beautifully, forming an abstract art piece before my eyes. \u2014It's a shame that I couldn't appreciate it.I continued mixing colors to their exact value. More alizarin crimson. More water. More yellow ochre. Less water. More phthalo blue. The cycle continued. Eventually, I was satisfied. The colors looked good, there was enough contrast between facial features, and the watercolors stayed inside the lines.Craving feedback, I posted my art to Snapchat. I got a few messages such as 'wow' and 'pretty,' but one message stood out. 'You were anxious with this one, huh? Anyways, love the hair!'I was caught off guard. Was it a lucky guess? Did they know something I didn't? I immediately responded: 'Haha, how could you tell?' No response.What I didn't know at the time was that my response would come a few months later while babysitting. Since the girl I was babysitting loved art, I took out some Crayola watercolors and some watercolor paper for her to play with. After I went to the bathroom and came back, the watercolors were doused with water. 'You were impatient with this one, huh? Anyways, love the little dog you drew!'The little girl looked up at me, confused. 'How could you tell?' 'You used a lot of water for a brighter color, but you couldn't wait for it to slowly soak in.''Oh.'Now, I would be lying if I said I realized the connection between the two events immediately.Instead, I made the connection when I decided to sit down one day and objectively critique my art. The piece that I once loved now seemed like a nervous wreck: the paper was overworked, the brushstrokes were undecided, the facial features blended together, and each drop of water was bound inside the lines as if it was a prisoner in a cage.From then on, I started noticing pieces of personality in additional creations surrounding me: website designs, solutions to math problems, code written for class, and even the preparation of a meal.When I peer around at people's projects during Code Club, I notice the clear differences between their code. Some people break it up by commenting in every possible section. Others breeze through the project, not caring to comment or organize their code. I could also see clear differences in personalities when our club members began coding the Arduino for the first time. Some followed the tutorials to the letter, while others immediately started experimenting with different colored LEDs and ways of wiring the circuit.It became clear to me that, as humans, we leave pieces of our souls in everything we do, more than we intend to. If we entertain this thought, perhaps the key to better understanding others around us is simply noticing the subtler clues under our noses?Perhaps there are endless windows to the soul, and we simply need to peer through them. I shakily rose my hand. 'We should create workshops of our own,' I suggested.I got a few strange looks. 'It's a good idea, but it's too much work.' 'We just don't have enough free time to make it work.' 'Maybe we could, but I don't know how to make workshops.' My suggestion was shot down. I shuffled in my seat. 'I could make them.' A few people stared at me in disbelief. I glanced over at the club advisor, Mr. C, nervous to hear his response.'If you're willing to take on the work, we can try it.' Mr. C replied. And so I embarked on my quest. I researched different workshops on the internet, learning the information myself at first. Then, I transitioned into creating workshops of my own, making sure that the information was easy to understand for even a beginner. I was exhausted; my first workshop took 16 cumulative hours to create."""



extracted_features_result = FeatureExtration(text_input, winSize=10, step=10) #특징들을 모두 추출하여 출력할 거임
print("type:>>>>>",type(extracted_features_result))
print ('extracted_features_result', extracted_features_result) 


[[0.69, 471.8498871295094, 0.6607142857142857, 4529.66, 0.99, 7.95, 66.3], 
[0.66, 447.7336814478207, 0.6022727272727273, 3959.19, 0.99, 7.95, 71.96], 
[0.69, 461.512051684126, 0.6831683168316832, 4879.91, 0.99, 7.95, 73.45], 
[0.61, 541.164605185504, 0.5223214285714286, 3124.6, 0.99, 7.95, 57.58], 
[0.66, 516.4785973923515, 0.5885714285714285, 3934.69, 0.99, 7.95, 64.35], 
[0.64, 461.512051684126, 0.6039603960396039, 3911.38, 0.99, 7.95, 70.1]]