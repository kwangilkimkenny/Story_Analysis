# kj 로직 설계 #
# Topic Knowledge (10%) (김광일 대표님 이거 새로 더한거. 고민 요망)
# Wikipedia에서 에세이의 메인 토픽을 검색해 보고, 그 관련 내용과 얼마나 에세이의 컨텐츠가 연관성이 있는지를 본다. 그냥 대입하던지, 또는 에세이의 토픽들 vs. Wikipedia 해당 컨텐츠의 topic들을 추출해서 그들의 vector 연관성을 볼 수도 있다.
# *그러니깐 이것은 에세이에서 정확하게 무슨 토픽을 쓰는지를 detect해서 분석해야 한다… 그냥 diversity가 아니라 글 속에 예를 들면 Black Lives Matter이렇게 들어가 있다던지

# kyle의 방법 #
# 1)에세이의 주요 토픽을 모두 추출한다. 주요 토픽중 가장 대표적인 3개를 추출해본다. 
# 2)3개의 주요 토픽으로 google 검색을 한 후(완료), 검색 페이지를 개별적으로 크롤링하여 문서에 포함된 모든 단어를 추출한다. (완료)
# 3)구글에서 추출한 단어에 에세이의 주요토픽단어가 얼마나 일치하는지,즉 몇개가 일치하는지 카운트를 한다. 
# 4)에세이주요토픽추출단어/구글 검색 추출단어리스트 * 100 을 계산하면 Topic knowledge의 비율이 나올 것임, 이것이 높으면 10% 이상이면 높은 점수를 줄 수있음 (Supurb, Strong, Good, Mediocre, Lacking 중 1개로 계산됨, 점수로도 계산해야 overall 계산 적용할 수 있음)


##### 이 코드는 키워드를 입력하면 1) 1차로 구글검색을 통해서 결과룰 추출 - 연결링크 모두 수집 2) 2차로 링크페이지에 모두 접속하여 text 데이터를 추출하여 리스트로 저장하는 기능


# Chrome 버전을 확인하고 드라이버 버전을 동일하게 해야 함
# 크롬드라이버 다운로드 링크 : https://chromedriver.chromium.org/downloads
# 현재 사용하는 크롬 버전 90.0.4430.85(공식 빌드) (x86_64)
# 적용한 크롬 드라이버 버전은 위와 동일



import re
import pandas as pd
import openpyxl
from urllib.parse import ParseResultBytes, quote_plus
from bs4 import BeautifulSoup
from selenium import webdriver

from tqdm import tqdm

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

from nltk.corpus import stopwords
stop = stopwords.words('english')
stop_words = set(stopwords.words('english')) 

options = webdriver.ChromeOptions()
options.add_argument('headless')
options.add_argument('window-size=1920x1080')
options.add_argument("disable-gpu")
# 혹은 options.add_argument("--disable-gpu")

# UserAgent값을 바꿔줍시다! 서버가 인식하지 못하도록 가상으로 headless 값 추가함ㅠ
options.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36")


def cleaning_data(input_data):
    remove = input_data.replace("  "," ") # 변환
    remove_ = re.sub(r"\t", " ", remove) # 제거
    remove__ = re.sub(r"\n", " ", remove_) # 제거
    remove__ = remove__.replace("   ", " ")
    remove__ = remove__.replace("  ", " ")
    remove__ = remove__.replace(" ", ",")
    remove__ = remove__.replace("…/", " ")
    remove__ = remove__.replace("…", " ")
    remove__ = remove__.replace("/", " ")
    remove__ = remove__.replace(" ", ",")
    remove__ = remove__.replace(")", ",")
    remove__ = remove__.replace("(", ",")
    preprossed = remove__.split(",") # 단어를 리스트로 변환
    #print(preprossed)
    
    # 표제어 추출, 동사는 현재형으로 변환
    lemma_list =[]
    for i in preprossed:
        lema_re = lemmatizer.lemmatize(i, pos='v') #표제어 추출, 동사는 현재형으로 변환
        lemma_list.append(lema_re)
    
    # 표제어 추출
    ext_lema = [lemmatizer.lemmatize(w) for w in preprossed]
    # 중복값을 제거하고
    rm_dupli = set(ext_lema)
    # 다시 리스트로 만들고
    re_li = list(rm_dupli)
    # 빈 값은 제거하고
    get_wd =list(filter(None, re_li))
    # 소문자로 모두 변환
    lower_wd = [i.lower() for i in get_wd]
    
    result = []
    for w in lower_wd: 
        if w not in stop_words: 
            result.append(w)
    return result


def google_search_result_tp_knowledge(input_word):
    baseUrl = 'https://www.google.com/search?q='

    #plusUrl = input('무엇을 검색할까요? :')
    plusUrl = input_word

    # url = baseUrl + quote_plus(plusUrl)
    url = baseUrl + plusUrl
    # 한글을 사용할 경우 :  quote_plus 적용 - URL에 막 %CE%GD%EC 이런 거 생성해줌

    driver = webdriver.Chrome(executable_path= r'./data/chromedriver_mac_ver_90', chrome_options=options)
    driver.get(url)

    html = driver.page_source
    soup = BeautifulSoup(html, features="html.parser")

    v = soup.select('.yuRUbf')

    search_title_result = []
    search_linked_contents_result = []
    for i in v:
        #print(i.select_one('.LC20lb.DKV0Md').text)
        search_title_result.append(i.select_one('.LC20lb.DKV0Md').text)
        print(i.a.attrs['href'])
        search_linked_contents_result.append(i.a.attrs['href'])
        #print()


    # search_linked_contents_result 의 각 링크로 접속하여 해당 내용을 모두 text로 크롤링한 후, body내의 단어들만 추출한다.
    get_all_linked_web_data = []
    for linked_page in tqdm(search_linked_contents_result):
        driver.get(linked_page)
        html = driver.page_source
        get_all_data = BeautifulSoup(html, features="html.parser")
        get_all_linked_web_data.append(get_all_data)
        
    body = re.search('<body.*/body>', html, re.I|re.S)
    if (body is None):
        print ("No <body> in html")
        exit()
            
    body = body.group()
    #print(body)
    
    # 추출된 정보 클린징
    korean = re.compile('[\u3131-\u3163\uac00-\ud7a3]+')#한글제거

    item_extract = str(body).replace('\n', ' ')
    item_extract = re.sub('<span.*?>.*?</span>', ' ', item_extract)
    item_extract = re.sub('<b>.*?</b>', ' ', item_extract)    
    item_extract = re.sub('<.*?>', ' ', item_extract)        
    item_extract = item_extract.replace('\t', ' ')
    item_extract = re.sub(korean, '', item_extract)
    item_extract = re.sub('[-=.#/?:$}]', ' ', item_extract)
    item_extract = re.sub("[-=+,#/\?:^$.@*\"※~&%ㆍ!』;{}()'\\‘|\(\)\[\]\<\>`\'…》]", ' ', item_extract)
    #print (item_extract)
    
    driver.close()

    get_result_str = str(item_extract)
    result_cln = cleaning_data(get_result_str) # 결과값 청소
    result = list(set(result_cln)) # 중복제거

    return result


## run ##

#result = google_search_result(input_word)
result = google_search_result_tp_knowledge("college personal essay")

print(result)

# 결과는 단어에데한 구글 링크 페이지의 모든 텍스트가 추출되어 리스트로 출력됨
# ['←how', 'moved', 'california', 'improved', 'stroke', 'call', 'access', 'hour', ....