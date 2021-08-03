## Author : Grant
## Kyle 대표님 테스트를 위한 Wordcloud 파일입니다.

import os, re, nltk, json

def find_quote(c):
    ## 쌍따옴표 찾는 함수
    return ord(c)==8220 or ord(c)==8221 or ord(c)==34

def Professor_cleaner(json_file):
    ## Major 파일에서 교수 정보 찾아서 전처리하는 함
    f = open(json_file,'r',encoding='utf-8')
    text = ""
    for i in f:
        if(i is not None):
            text = text + i

    essay_text = ""
    professor_flag = 0
    for i in text:
        essay_text = essay_text + i
        if(professor_flag == 0):
            if(find_quote(i)):
                professor_flag = 1
            else:
                professor_flag = 0
        elif(professor_flag == 1):
            if(i == "P"):
                professor_flag = 2
            else:
                professor_flag = 0
        elif(professor_flag == 2):
            if(i == "r"):
                professor_flag = 3
            else:
                professor_flag = 0
        elif(professor_flag == 3):
            if(i == "o"):
                professor_flag = 4
            else:
                professor_flag = 0
        elif(professor_flag == 4):
            if(i == "f"):
                professor_flag = 5
            else:
                professor_flag = 0
        elif(professor_flag == 5):
            if(i == "e"):
                professor_flag = 6
            else:
                professor_flag = 0
        elif(professor_flag == 6):
            if(i == "s"):
                professor_flag = 7
            else:
                professor_flag = 0
        elif(professor_flag == 7):
            if(i == "s"):
                professor_flag = 8
            else:
                professor_flag = 0
        elif(professor_flag == 8):
            if(i == "o"):
                professor_flag = 9
            else:
                professor_flag = 0
        elif(professor_flag == 9):
            if(i == "r"):
                professor_flag = 10
            else:
                professor_flag = 0
        elif(professor_flag == 10):
            if(find_quote(i)):
                professor_flag = 11
            else:
                professor_flag = 0
        elif(professor_flag == 11):
            if(i == ":"):
                professor_flag = 12
            else:
                professor_flag = 0
        elif(professor_flag == 12):
            if(i == " "):
                professor_flag = 13
            else:
                professor_flag = 0
        elif(professor_flag == 13):
            if(find_quote(i)):
                professor_flag = 14
                essay_text = essay_text[:-1] + "\""
            else:
                professor_flag = 0
        else:
            if(find_quote(i)):
                essay_text = essay_text[:-1]
            elif(i == "\n"):
                essay_text = essay_text[:-1] + " "
            elif(i == "}"):
                essay_text = essay_text[:-1] + "\"\n}"

    #for i in text:
        #if(find_quote(i)):
        #    i=""
        #essay_text = essay_text + i
    return essay_text
def college_file_read(college_name, files_path):
    #files_path = "/var/www/html/essayfit/accu_college/grant_lab/DEMO/essayfit/accu_college/college_info/college_dataset/"

    text = ""
    for file_name in os.listdir(files_path):
        if(file_name.split("_")[0] == college_name):
            f = open(files_path + file_name,'r',encoding='utf-8')
            for text_bufer in f:
                if(text_bufer is not None):
                    text = text + text_bufer

    return text

def major_file_read(college_name, major_name, files_path):
    #files_path = "/var/www/html/essayfit/accu_college/grant_lab/DEMO/essayfit/accu_college/college_info/Integral_major_dataset/"
    text = ""
    for file_name in os.listdir(files_path):
        if(file_name.split("_")[0] == college_name):
            if(major_name == "Undecided"):
                essay_text = dict()
                try:
                    ## 파일 읽기
                    ## Professor 데이터가 없을 경우 에러남
                    with open(files_path + file_name, encoding='UTF8') as json_file:
                        essay_text = json.load(json_file, strict=False)
                except:
                    ## 그래서 에러나면 억지로 읽어서 전처리 해야함.
                    ## Jeff가 모든 데이터에 Profeesor 정보 다 넣으면 이 과정 필요없음.
                    essay_text = eval(Professor_cleaner(files_path+file_name))

                Major_General_info = ''.join(essay_text['General_Keyword'])
                Major_Courses_info = ''.join(essay_text['Courses_Concentration'])
                Major_Opportunity_info = ''.join(essay_text['Student_Opportunities'])
                Major_Labs_info = ''.join(essay_text['Facilities_Resources'])
                Major_Professor_info = ''.join(essay_text['Professor'])

                text = text + Major_General_info + Major_Courses_info + Major_Opportunity_info + Major_Labs_info + Major_Professor_info
            else:
                if((file_name.split("_")[1]).split(".")[0] == major_name):
                    essay_text = dict()
                    try:
                        ## 파일 읽기
                        with open(files_path + file_name, encoding='UTF8') as json_file:
                            essay_text = json.load(json_file, strict=False)
                    except:
                        essay_text = eval(Professor_cleaner(files_path+file_name))

                    Major_General_info = ''.join(essay_text['General_Keyword'])
                    Major_Courses_info = ''.join(essay_text['Courses_Concentration'])
                    Major_Opportunity_info = ''.join(essay_text['Student_Opportunities'])
                    Major_Labs_info = ''.join(essay_text['Facilities_Resources'])
                    Major_Professor_info = ''.join(essay_text['Professor'])

                    text = text + Major_General_info + Major_Courses_info + Major_Opportunity_info + Major_Labs_info + Major_Professor_info

    return text

def Convert_others_to_str_english(sentence):
	buffer = ""
	for i in sentence:
		i = re.sub('[^a-zA-Z]','',i) ## We use only english
		buffer = buffer + " " + i

	return nltk.word_tokenize(buffer)

def Sentence_to_word(sentence):
    sentence = (''.join(sentence)).split()
    word_dict = {}
    tagged = nltk.pos_tag(Convert_others_to_str_english(sentence))

    allnoun = [word for word, pos in tagged if pos in ['NN','NNP']]

    for noun in allnoun: ## String to Dictionary
            try:
                    word_dict[noun] += 1
            except:
                    word_dict[noun] = 1

    word_dict = dict(sorted(word_dict.items(), key=lambda x: x[1], reverse = True)) ## DESC Sorted

    result = {}
    ## Limited number of data
    limited_num = 10
    #limited_num = len(word_dict)
    ##############################
    for key in list(word_dict.keys())[:limited_num]:
            result[key] = str(word_dict[key])
    return result


def is_alphabet(c):
    return (ord(c) > 64 and ord(c) < 91) or (ord(c) > 96 and ord(c) < 123)

def major_converter(major_name):
    buffer = ""
    for i in major_name:
        if(is_alphabet(i)):
            buffer = buffer + i
    return buffer.lower()

college = "Caltech"
major = "Astronomy"

major = major_converter(major)

college_file_path = "./college_info/college_dataset/"
major_file_path = "./college_info/Integral_major_dataset/"

college_info = college_file_read(college, college_file_path)
major_info = major_file_read(college, major, major_file_path)
# print('major_info:', major_info)

# extract noun chunk 
import spacy
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English

nlp = spacy.load("en_core_web_lg")

tokenizer = Tokenizer(nlp.vocab)

doc = nlp(college_info)

get_noun_chunk = []
for chunk in doc.noun_chunks:
    # print(chunk)
    get_noun_chunk.append(chunk)

# print('get_noun_chunk:', get_noun_chunk)

noun_chunk_all_set = []


for i in get_noun_chunk:
    token_i = [word.text for word in i]
    # print('token_i:', token_i)
    # 한개의 단어로 구성된 명사 제거
    if len(token_i) > 1:
        noun_chunk_sets = []
        noun_chunk_sets.append(i)
        noun_chunk_sets.append(token_i)
        noun_chunk_all_set.append(noun_chunk_sets)
    else:
        pass

print('noun_chunk_all_set:', noun_chunk_all_set) # [[the physical processes, ['the', 'physical', 'processes']], [the universe, ['the', 'universe']], ...

# 2개 이상으로 구성된 명칭만 추출
get_specific_name = []
for ext_i in noun_chunk_all_set:
    get_n_chunk = ext_i[0]
    get_specific_name.append(get_n_chunk)

print('get_specific_name:', get_specific_name)




####### Word_cloud result ################
college_dict_result = Sentence_to_word(college_info)
major_dict_result = Sentence_to_word(major_info)

# print(college_dict_result)
