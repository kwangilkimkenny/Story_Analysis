import numpy as np
import re
import pandas as pd


def major_analysis(input_text_major):

    #소문자로 변환
    major_list_re = []
    for mjl in input_text_major:
        mjl_ = mjl.lower()
        mjl__ = mjl_.strip()
        major_list_re.append(mjl__)

    #print (major_list_re)
    return major_list_re

def award_anaysis(input_text_award):

    # 아래와 같이 입력해야 함, 소문자 처리, 공백처리 완료!
    #    Chemical Engineering, investment,biomedical  
    #awards = """ winner of ABC competition competition, none """
    # 2개의 입력: 수상내역, 성적   carnegie international student essay contest	    winner of ABC competition competition

    #입력 문자를 소문자로 변환
    honor_list_re = []
    for hor in input_text_award:
        hor_ = hor.lower()
        hor__ = hor_.strip()
        honor_list_re.append(hor__)

    #print (honor_list_re)
    return honor_list_re



########## 입력 ######
#####################
# 3개의 입력: 전공 3개를 ,로 구분하여 입력 >> 리스트에 담음
majors = """ Asian Studies, Psychology, Political Science/Government  """ # social_sci 로 전공 적합성은 FIT  점수는 5.0
# 수상내역, 수상성적(점수 등) 2개의 입력이며 , 로 구분 >> 리스트에 담음
#수상내역이 DB에 없을 경우 점수 계산 하기 위한 용  >>>>>>>>> 여기수 중요한 것!!!! none 값을 반드시 입력하도록 해야 함!!!!!!!!!!!!!!!!!

#awards = """ Words of Women,selective """  # 이것은 social_sci 와 관련성이 높다.
#수상내역이 DB에 있을 경우 
#awards = """  carnegie international student essay contest, Extremely Selective	 """
#awards = """ winner of ABC competition competition, none """ # 전공적합성은 not sure, 점수는 5.0 이 결과로 나와야 함
awards = """ WSDC, Extremely Selective """ # 전공적합성은 not fit, 점수는 5.0 이 결과로 나와야 함





def input_values(inp_majors, inp_awards):
    inp_mjr_value = inp_majors.split(',')
    inp_awd_value = inp_awards.split(',')
    #print (inp_mjr_value)
    #print (inp_awd_value )
    mjr_awd_input = inp_mjr_value + inp_awd_value #입력받은 두개의 리스트를 하나로 합친다
    return mjr_awd_input




#  ====== Start! ======
if __name__ == '__main__':
    get_values_re = input_values(majors, awards)
    major_list = major_analysis(get_values_re[:3]) #['asian studies', 'film studies', 'archeology']
    honor_list = award_anaysis(get_values_re[3:]) #['carnegie international student essay contest', 'extremely selective']


# print (major_list)
# print (honor_list)


#Awards 데이터 불러오기
data_awards = pd.read_csv('awards_list.csv')
data_awards['award_title'] = data_awards['award_title'].str.lower() #소문자로 변환해야 타이틀을 비교할 수 있음(매칭되는지)
data_awards['Big Major Category #1'] = data_awards['Big Major Category #1'].str.lower()
data_awards['Big Major Category #2'] = data_awards['Big Major Category #2'].str.lower()
data_awards['Big Major Category #3'] = data_awards['Big Major Category #3'].str.lower()
data_awards['Big Major Category #4'] = data_awards['Big Major Category #4'].str.lower()

is_honor_list = data_awards['award_title'] == honor_list[0] # 입력한 결과중 수상내역 문자열 과 데이터프레임의 타이클 컬럼에서 같은 조건
re_honor = data_awards[is_honor_list] # 데이터 필터링하여 새로운 변수에 저장

#print (re_honor) # 결과 출력(여기서 결과가 나오면, 수상내역 출력, 없으면... )

award_anaysis_re = re_honor[["award_title","prestige_score","Big Major Category #1","Big Major Category #2","Big Major Category #3","Big Major Category #4"]]

#print (award_anaysis_re)



# major fit을 계산하려면 입력한 희망전공 3개를 예: ['mechanical engineering', ' computer science', ' bioligy']

# 각 전공을 다음 리스트에서 검색하여 매칭되는 값이 일치(여기서는 business이면) 하면 'FIT',  아니라면 ''NOT FIT 없으면 '?'로 표시


tech_eng_ = ['mechanical engineering', 'general Engineering', 'mechanical Engineering', 'Industrial engineering', 'bioengineering',
            'operations Research', 'materials Science', 'electrical Engineering', 'computer Science/engineering','computer engineering'
            'civil engineering', 'aerospace Engineering', 'chemical Engineering', 'environmental engineering', 'stem', 'steam', 'robotics', 
            'robots', 'computer', 'computers', 'computer programming', 'programming', 'coding', 'computing', 'mechanical', 'mechanics', 
            'mechanism', 'ai', 'artificial intelligence', 'sensor', 'environmental engineering', 'environmental technology', 
            'alternative energy', 'renewable energy', 'material science', 'biomedical', 'biomaterials', 'nanotech', 'pharmaceutical',
            'biomechanics', 'biomimicking', 'biomimicry', 'machine', 'bioengineering', 'chemical engineering', 'applied physics', 
            'operations research', 'civil engineering', 'aerospace', 'system', 'operations research', 'industrial engineering', 'computer Science'
            'invention', 'electrical', 'computational']

tech_eng = []
for i in tech_eng_:
    i_ = i.lower()
    tech_eng.append(i_)

math_sci_ = ['Agriculture', 'Physics', 'Biology', 'Biophysics', 'Biochemistry/Molecular Biology',
            'Mathematics', 'Applied Mathematics', 'Nutrition/Food Science', 'Genetics', 'Astronomy/Astrophysics',
            'Statistics', 'Chemistry', 'Neuroscience', 'Economics','Nursing','Pre-Med', 'Pre-Veterinary'
            'Pharmacy/Pre-Pharmacy', 'Pre-Dental', 'Pharmacy', 'Pre-Pharmacy', 'STEM', 'STEAM', 'Chemistry', 'physics',
            'biology', 'bio', 'biochem', 'biochemistry', 'science', 'math', 'mathematics', 'algorithm', 'biophysics', 
            'applied math', 'genetics', 'scientific', 'astronomy', 'astrophysics', 'statistics', 'neuroscience', 'dental',
            'doctor', 'physical', 'medical', 'scientific research', 'Biomaterials', 'material science', 'nutrition', 'clinical',
            'earth science', 'environmental science', 'ecology', 'ecological', 'veterinary', 'molecular', 'nanotech', 'biomedical',
            'pharmaceutical', 'food science, biotech', 'brain', 'botany', 'botanical', 'immunology', 'immune', 'zoology', 'biomechanics',
            'biomimicking', 'biomimicry']

math_sci = []
for j in math_sci_:
    j_ = j.lower()
    math_sci.append(j_)
    
social_sci_ = ['Economics Asian Studies', 'African Studies','Psychology', "Women's Studies",'Anthropology', 'Political Science/Government',
                'Political Science', 'Political Government', 'International Relations/Affairs', 'International Relations',
                'International Affairs', 'Public Policy/Affairs', 'Public Policy', 'Public Affairs', 'Public Health',
                'Environmental Science/Studies', 'Environmental Science', 'Environmental Studies', 'Urban Planning', 'Education',
                'Volunteer', 'community', 'community service', 'culture', 'equality', 'service',' social issue', 'social', 
                'society', 'economic', 'economy', 'SDGs', 'environment', 'policy', 'psychology', 'Asian studies', 'race relations', 
                'ethnicity', 'ethnic', 'women', 'anthropology', 'politic', 'international relations', 'international affairs',
                'diplomacy', 'diplomat', 'public policy', 'public health,' 'inequality', 'geography', 'geographic', 'linguistic', 
                 'sociology', 'culture', 'urban studies', 'diversity', 'LGBT', 'LGBTQ', 'sustainable', 'sustainability', 'debate',
                 'Middle East',' Middle Eastern Studies']

social_sci = []
for k in social_sci_:
    k_ = k.lower()
    social_sci.append(k_)

humanities_ = ['Archeology', 'Film Studies','German (Language and Literature)','Art History',
                'French (Language and Literature)','Spanish','Theology/Religious Studies',
                'Linguistics','History','English Language and Literature','Comparative Literature',
                'Philosophy','Classical Studies/Latin/Greek', 'English', 'literature', 'history', 'writing', 
              'heritage', 'essay', 'script', 'culture' ,'humanities', 'poem', 'poet', 'playwright', 'script',
              'poetry', 'literary', 'lit', 'writing', 'Spanish', 'German', 'Chinese', 'Italian', 'French', 'Russian', 'literature', 
              'Latin', 'Greek', 'anthropology', 'archaeology', 'Law', 'ethics', 'film studies']

humanities = []
for l in humanities_:
    l_ = l.lower()
    humanities.append(l_)
    

visualart_ = ['Film&Television', 'Film', 'Television', 'Design (Graphic, Industrial, Computer Graphics)','Graphic Design','Industral Design', 
             'Computer Graphics', 'Fashion Design', 'Photography', 'Architecture', 'Movie', 'visual', 'music', 'art', 'arts', 'performing', 
             'paint', 'painting',' visual arts', 'design', 'draw', 'drawing', 'graphic', 'architecture', 'architect', 'film', 'sculpture',
             'artwork', 'fine art']

visualart = []
for m in visualart_:
    m_ = m.lower()
    visualart.append(m_)
    

business_ = ['Business', 'Finance', 'Accounting', 'Marketing', 'Intl', 'Biz','International Business','Mgmt.', 'Info Systems', 
            'Hotel/Hospitality Management', 'Hotel Management', 'Hospitality Management', 'Business', 'start up', 'start-up',
            'entrepreneur', 'entrepreneurship', 'investment', 'invest', 'business plan', 'finance', 'marketing',' organizational behavior',
            'logistics', 'management', 'managing', 'social entrepreneur', 'social enterprise', 'accounting', 'managerial', 'MIS', 
            'Management Information Systems', 'Hotel management', 'hospitality']

business = []
for o in business_:
    o_ = o.lower()
    business.append(o_)
    

music_ = ['Music (Performance)', 'Music', 'Music Theory']

music = []
for p in music_:
    p_ = p.lower()
    music.append(p_)
    

performing_arts_ = ['performing arts','performance', 'dance', 'Movie', 'music', 'musical', 'theater', 'theatre', 'opera', 'thespian', 'song',
                   'sing', 'vocal', 'drama', 'orchestra', 'band', 'acting', 'acrobatics', 'ballet','circus', 'magic', 'mime', 'puppetry',
                   'ventriloquism', 'spoken word', 'stand-up comedy', 'Chinese opera', 'pansori', 'chamber music', 'cabaret', 
                   'jazz band', 'big band', 'Bharatanatyam', 'Gamelan semar pegulingan', 'Shōmyō', 'Odissi', 'Cantonese opera']

performing_arts = []
for q in performing_arts_:
    q_ = q.lower()
    performing_arts.append(q_)

communication_ = ['Documentary', 'SNS', 'Social Network', 'PR', 'public relations', 'advertisement', 'advertising', 'movie', 'film', 'news',
                 'journalism', 'newspaper', 'debate', 'speech', 'newspaper', 'media', 'communication']

communication = []
for r in communication_:
    r_ = r.lower()
    communication.append(r_)


df_a = pd.DataFrame(tech_eng, columns=["major"])
df_a["main_major"] = "tech_eng"

df_b = pd.DataFrame(math_sci, columns=["major"])
df_b["main_major"] = "math_sci"

df_c = pd.DataFrame(social_sci, columns=["major"])
df_c["main_major"] = "social_sci"

df_d = pd.DataFrame(humanities, columns=["major"])
df_d["main_major"] = "humanities"

df_e = pd.DataFrame(business, columns=["major"])
df_e["main_major"] = "business"

df_f = pd.DataFrame(music, columns=["major"])
df_f["main_major"] = "music"

df_g = pd.DataFrame(performing_arts, columns=["major"])
df_g["main_major"] = "performing_arts"

df_h = pd.DataFrame(communication, columns=["major"])
df_h["main_major"] = "communication"

# 전공별 데이터와 주 전공을 하나의 데이터프레임으로 합친다. 
total_major_df = pd.concat([df_a,df_b,df_c,df_d,df_e,df_f,df_g,df_h])


#검색 결과 추출
major_ctgs_ =[]
major_ctg = list(award_anaysis_re["Big Major Category #1"])
major_ctgs_.append(major_ctg)
major_ctg = list(award_anaysis_re["Big Major Category #2"])
major_ctgs_.append(major_ctg)
major_ctg = list(award_anaysis_re["Big Major Category #3"])
major_ctgs_.append(major_ctg)
major_ctg = list(award_anaysis_re["Big Major Category #4"])
major_ctgs_.append(major_ctg)

major_ctgs = [y for x in major_ctgs_ for y in x]

#print ("major category : ", major_ctgs)

#print (major_list)


# 원하는 전공 리스트의 값을 데이터프레임 total_major_df에서 찾아보자
fit_analysis_list = []
for search_major in major_list:
    
    re__ = total_major_df[total_major_df['major'] == search_major]  # 데이터프레임에 찾고자하는 전공이 있다면

    #print (re__) # 결과물을 리스트에 저장해보자
    fit_analysis_list.append(re__)

#fit_analysis_list 를 문자로 만들어서 개별 리스트로 만들자
s =" ".join(map(str, fit_analysis_list))

fit_list_str = s.split()

#print (fit_list_str)

#드디어 분석 fit analysis!!!!
#선택한 전공 3개, 수상내역 1개를 비교하여 데이터베이스 비교 분석하여 상과 전공이 일치하는지 판단!
fit_result = []


#드디어 분석 fit analysis!!!!
#선택한 전공 3개, 수상내역 1개를 비교하여 데이터베이스 비교 분석하여 상과 전공이 일치하는지 판단!
fit_result = []


# 수상내역이 데이터에 없을 경우 not sure. 그러나 수상점수는 계산이 가능함(first, winner 등의 단어를 확인해서)
if  re_honor.isnull().any().any(): #이 re_honor 이 없으면 false 라면 else 실행되고 not sure로 해야 함
    for category in fit_list_str: #수상내역으로 분석 추출한 결과 내역들을 모두 단어로 쪼개어서 하나씩 추출해서 비교
        for major_item in major_ctgs:# 신청한 전공 3개 꺼내와서 하나씩 비교
            if major_item  == category: # 전공이 있으면 fit
                #print ("fit")
                fit_result.append('fit')

            else: # 원하는 전공 3개와 수상내역과 연관된 전공중 아무것도 매칭되는 것이 없으면 not sure
                    #print ("not fit")
                    fit_result.append('not fit')    
else:
    #print ("not sure")
    fit_result.append('not sure')


#print(fit_result) #데이터프레임으로
df_fit_re = pd.DataFrame(fit_result)
df_fit_re.columns = ['Calssfy_fit']
list_fit_re = df_fit_re.drop_duplicates() #중복값 제거!!!! 

#조건문을 만들어서 결과를 출력해보자.
fit_anaysis_result_fin =[]
if 'fit' in list_fit_re.values : # fit이 하나라도 있다면, FIT 출력
    #print("FIT")
    fit_anaysis_result_fin.append('FIT')
elif 'not fit' in list_fit_re.values : # not fit 이 있다면 , NOT FIT 출력
    #print("NOT FIT")
    fit_anaysis_result_fin.append('NOT FIT')
else:
    #print("NOT SURE")
    fit_anaysis_result_fin.append('NOT SURE')
    
fit_anaysis_result_fin

print ("===============전공과 수상내역을 비교한 '전공 적합성' 분석 결과 ================")
print("Check Major FIT : :", fit_anaysis_result_fin)


# ########################### 2.수상내역에 대한 점수를 계산한다.

#수상리스트의 값을 점수로 계산해보자  extremely selective 같은 경우는 그냥 데이프레임에서 추출하면 됨
#단, winner of ABC competition competition, none  입력을 받았을 경우에는 데이터베이스에 수상내역이 없기 때문에 예외처리를 해야 한다.
is_honor_list = data_awards['award_title'] == honor_list[0] # 입력한 결과중 수상내역 문자열 과 데이터프레임의 타이클 컬럼에서 같은 조건
re_honor = data_awards[is_honor_list] # 데이터 필터링하여 새로운 변수에 저장
re_honor # 결과 출력(여기서 결과가 나오면출력, 만약 수상내역 출력, 없으면... 예외처리를 해야함.. 2번째 방식으로 분석해야 함)
#점수를 추출한다. 그리고 이 점수는 입력한 수상내역의 문자를 확인하여 '예외의 룰'을 적용!!!
award_anaysis_re = re_honor[["prestige_score"]] #이건 데이터프레임
#print(award_anaysis_re) # 5.0 이 나옴
fit_re_fin = sum(award_anaysis_re.values.tolist(),[]) #리스트로 변환
# print ("==== 수상내역 점수 ==================================")
# print ("수상점수_1 : ", fit_re_fin)  #결과값 도출



award_anaysis_re_TF = award_anaysis_re.isnull().values.any()#데이터 프레임에 값의 유무 확인. 부울 값을 반환한다. 값이 있으면  Fasle???이다.

#print(award_anaysis_re_TF) # False가 나옴

print ("==============입력 전공 3개, 수상내역과 수상결과 ===================")
#입력값 확인
print ("Selected 3 Majors : ", majors)
print ("Award Title & Prestige : ", awards)

#input awards : winner of ABC competition competition, none   >>>>>>>>>>> award_anaysis_re_TF == True 로 할 경우 수상점수_2 가 실행됨 <성공>
#input awards : winner of ABC competition competition, none   >>>>>>>>>>> award_anaysis_re_TF == False 로 할 경우 수상점수_1 이 실행됨

#input awards : carnegie international student essay contest, Extremely Selective >>>>> award_anaysis_re_TF == True 로 할 경우 수상점수_2 가 실행됨 

if not fit_re_fin:  ########## 여기 조건문을 고쳐야 함...
    #### 3.수상내역이 DB에 없을 경우에 점수 계산하기
    print ("================== 수상 점수 계산 ===================")
    print ("수상내역에 DB에 없습니다. 그래서 단어 분석을 통해서 점수를 계산해 볼께요.")
    remove_null_text = awards.strip() #앞 뒤 공백 제거한다. 입력할 때 공백을 넣을 수 있기 때문에.. 에러방지
    split_awards_list = remove_null_text.split(" ")
    split_awards_list #결과를 토대로 점수 계산하기위해서 data_awards_score의 데이터 프레임에서 해당 점수값 도출하기

    data_awards_score = pd.read_csv('award_title_score.csv')
    data_awards_score ['title'] = data_awards_score ['title'].str.lower() #소문자로 변환해야 타이틀을 비교할 수 있음(매칭되는지)
    data_awards_score.set_index('title', inplace=True) #title을 인뎃스로 변환, 그래야 값을 찾기 쉽다.

    #입력한 수생내역의 리스트 값을 하나씩 불러와서 데이터프레이에 있는지 비교 찾아내서 해당 점수를 가져오기
    get_score__ = []
    for award_word in split_awards_list: #데이터프레임에서 인덱스의 값과 비교하여
        if award_word in data_awards_score.index: #df에 특정 단어가 있다면, 해당하는 컬럼의 값을 가져오기
            get_score_ = data_awards_score.loc[award_word,'fin_score']
            get_score__.append(get_score_)
            #print ("got it")
            
        else:
            #print ("The result cannot be calculated. Please enter more information about your award.")
            pass
    print ("================== 수상 점수 계산 =======================")
    print ("SCORE : ", get_score__)

else:
    # 이것이 데이터베이스에 있는 수상명에 해당하는 점수 계산 결과다.
    # fit_re_fin

    fit_re_fin = sum(award_anaysis_re.values.tolist(),[])
    print ("================== 수상 내역 점수 =======================")
    print ("SCORE : ", fit_re_fin) 






