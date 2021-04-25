# Chrome 버전을 확인하고 드라이버 버전을 동일하게 해야 함
# 크롬드라이버 다운로드 링크 : https://chromedriver.chromium.org/downloads
# 현재 사용하는 크롬 버전 90.0.4430.85(공식 빌드) (x86_64)
# 적용한 크롬 드라이버 버번은 위와 동일



import re
import pandas as pd
import openpyxl
from urllib.parse import ParseResultBytes, quote_plus
from bs4 import BeautifulSoup
from selenium import webdriver

options = webdriver.ChromeOptions()
options.add_argument('headless')
options.add_argument('window-size=1920x1080')
options.add_argument("disable-gpu")
# 혹은 options.add_argument("--disable-gpu")

# UserAgent값을 바꿔줍시다! 서버가 인식하지 못하도록 가상으로 headless 값 추가함ㅠ
options.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36")


def get_uniquness(search_result):
    # Dim sum  = 197,000,000
    # Macarons  = 68,100,000
    # Churros   = 24,300,000
    # pasta    = 754,000,000
    # ramen    = 164,000,000
    # udon soba = 30,200,000
    # Tom yam kung = 944,000
    if search_result > 100000000:
        uniqueness_re = 'Common'
    elif search_result <= 100000000 and search_result > 30000000:
        uniqueness_re = 'Unique'
    else: # search_result <= 100000000
        uniqueness_re = 'very unique'
    return uniqueness_re


def google_search_result(input_word):
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

    #print(soup.find('div', id='result-stats'))
    get_result = soup.find('div', id='result-stats')

    driver.close()

    result = str(get_result)
    re_ =re.sub("[^()]+$", "", result)
    re__ =re.sub("\([^)]*\)", "", re_) # 괄호안의 문제 제거
    re_d = re.findall("\d+", re__)
    # 검색어로 추출한 결과물을 가지고 topic uniquness 기능을 적용. 검색결과가 평균값(비교 단어로 추정하여 정함)보다 작으면 unique, 크면 ununiqe topis 이다.
    search_re = "".join(re_d)
    #print(search_re)
    input_search_num = int(search_re)
    
    
    # 검색 및 결과데이터 저장기능 (csv로 저장방법)
    # data =[input_word, input_search_num]
    # dataframe = pd.DataFrame(data)
    # dataframe.to_csv("./data/topic_search_result.csv", header=False, index=False)


    # 검색 및 결과데이터 저장기능 (excel로 저장방법) ---> 이 코드를 사용
    # 기존의 저장데이터 불러오기
    wb = openpyxl.load_workbook("./data/topic_search_result.xlsx")

    data =[input_word, input_search_num]
    #wb = openpyxl.Workbook() # 처음 저장할 때 사용
    sheet = wb.active
    sheet.append(data)
    wb.save("./data/topic_search_result.xlsx")

    uniq_result = get_uniquness(input_search_num)

    return uniq_result




## run ##
result = google_search_result('essayfit')
#result = google_search_result('AI')
print('result :' , result)