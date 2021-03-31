# 이 코드는 잘 작동하지만 학교정보 text 데이터와 개인이 입력한 에세이들의 모든 문장을 개별적으로 비교하기 때문에 매우 많은 시간이 걸린다.
# 그래서 학교정보문장데이터, 학생입력에세이를 요약하는 과정을 거치고, 요약문장에서 두 개의 유사도를 분석하면 될 것이다. 우선 요약을 해보자.
# https://github.com/uoneway/Text-Summarization-Repo    ---> 요약기술 참고

# 학교정보, 희망전공, 개인입력에세이를 각각 요약한다. 

import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk import sent_tokenize
from nltk.tokenize import sent_tokenize
import numpy as np
import re
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
sbert_model = SentenceTransformer('stsb-roberta-large')

from summarizer import Summarizer

# 데이터 전처리 
def cleaning_for_sent_sim(data):
    fin_data = []
    for data_itm in data:
        # 영문자 이외 문자는 공백으로 변환
        only_english = re.sub('[^a-zA-Z]', ' ', data_itm)
        lists_re_ = re.sub(r"^\s+|\s+$", "", only_english) # 공백문자 제거
        only_english_ = lists_re_.rstrip('\n')
        # 데이터를 리스트에 추가 
        fin_data.append(only_english_)
        # str 로 만드릭
        result = ' '.join(fin_data)
    return fin_data


# txt 문서 정보 불러오기 : 대학정보
def open_data_for_sent_sim(select_college):
    # 폴더 구조, 대학이름 입력 명칭을 통일해야 함
    file_path = "./college_info/colleges_dataset/"
    college_name = select_college
    file_name = "_college_general_info.txt"
    # file = open("./college_info/colleges_dataset/brown_college_general_info.txt", 'r')
    file = open(file_path + college_name + file_name, 'r')
    lists = file.readlines()
    file.close()
    
    tokenized_sentences = sent_tokenize(str(lists)) # 문장으로 토큰화
    result = ' '.join(tokenized_sentences)
    return result


# txt 문서 정보 불러오기 : 선택한 전공관련 정보 추출 
# 입력값은 대학, 전공 ex) 'Browon', 'AfricanStudies'
def open_major_data_for_sent_sim(select_college, select_major):
    # 폴더 구조, 대학이름 입력 명칭을 통일해야 함
    file_path = "./major_info/major_dataset/"
    college_name = select_college
    mjr_name = select_major
    file_name = "_major_info.txt"
    # file = open("./major_info/major_dataset/Brown_AfricanStudies_major_info.txt", 'r')
    file = open(file_path + college_name + "_" + mjr_name + file_name, 'r')
    lists = file.readlines()
    file.close()
    tokenized_sentences = sent_tokenize(str(lists)) # 문장으로 토큰화
    result = ' '.join(tokenized_sentences)
    return result
   
########################################
# college 문장을 하나 가져와서, 학생의 에세이 문장들과 모두 비교 후, 평균값으로 유사도 계산 
def sent_sim_analysis(college_info_data, ps_supp_esssay_data):

    sentences = ps_supp_esssay_data
    # sentences = ["I ate dinner.", 
    #     "We had a three-course meal.", 
    #     "Brad came to dinner with us.",
    #     "He loves fish tacos.",
    #     "In the end, we all felt like we ate too much.",
    #     "We all agreed; it was a magnificent evening."]

    # Tokenization of each document
    tokenized_sent = []
    for s in sentences:
        tokenized_sent.append(word_tokenize(s.lower()))
    tokenized_sent


    def cosine(u, v):
        return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

    # from sentence_transformers import SentenceTransformer
    # sbert_model = SentenceTransformer('stsb-roberta-large')
    model = sbert_model
    sentence_embeddings = model.encode(sentences)

    # print('Sample BERT embedding vector - length', len(sentence_embeddings[0]))
    # print('Sample BERT embedding vector - note includes negative values', sentence_embeddings[0])

    sim_sent_of_essay_score = 0
    counter = 0

    for q in tqdm(college_info_data):
        #query = "I had pizza and pasta"
        query = q
        print("query:", query)
        query_vec = model.encode([query])[0]

        for sent in tqdm(sentences):
            sim = cosine(query_vec, model.encode([sent])[0])
            sim_float = float(sim)
            print('sim_float:', sim_float)
            sim_sent_of_essay_score += sim_float
            counter += 1
            print("Sentence = ", sent, "; similarity = ", sim)


    re_mean = round((sim_sent_of_essay_score / counter), 2) * 100 # 일치율 추출하여 총평균 값 추출
    print("re_mean:", re_mean) # 17.0 % 의 일치율이 확률료 표시됨

    return re_mean

# # 위의 코드를 테스트용 샘플데이터 - 이하
# #### ------------start------------- ####
# # Test data
# college_info_data =  ['Founded in 1764, Brown is a leading research university home to world-renowned faculty, and also an innovative educational institution where the curiosity, creativity and intellectual joy of students drives academic excellence.n',
#  "The spirit of the undergraduate Open Curriculum infuses every aspect of the University.",
#   "Brown is a place where rigorous scholarship, complex problem-solving and service to the public good are defined by intense collaboration, intellectual discovery and working in ways that transcend traditional boundaries.", 
#   "Providence, Rhode Island — Brown's home for more than two and a half centuries — is a vibrant place to live, work and study, a stimulating hub for innovation, and a city rich in cultural diversity. Collectively, Brown's researchers are driven by the idea that their work will have a positive impact in the world.", 
#   'At the Institute at Brown for Environment and Society, scholars in the sciences, social sciences and humanities work together to confront the causes and impacts of climate change.',
#   "Brown is a leading research university, where stellar faculty and student researchers deploy deep content knowledge to generate new discoveries on those issues and many more. Nationally recognized as a leader in higher education and an accomplished scholar, she is a professor of economics and public policy.",
#   "The story of Brown is also the story of Providence and Rhode Island. Brown fosters innovative community partnerships and facilitates strong relationships with neighbors, nonprofit organizations, schools, civic organizations and businesses."]

# # Test data
# ps_supp_esssay_data = ['I inhale deeply and blow harder than I thought possible,' 'pushing the tiny ember from its resting place on the candle out into the air',
#  'My parents and the aunties and uncles around me attempt to point me in a different direction.',
#  " Growing up, I was discouraged from questioning others or asking questions that didn't have definitive yes or no answers.",
#  "At home, if I mentioned that I had tried eggs for breakfast at a friend's house, I'd be looked at like I had just committed a felony for eating what my parents considered meat." ,
#  "After an environmental science lesson, I stayed for a few minutes after class to ask my 4th-grade science teacher with wide eyes how it was possible that Niagara Falls doesn't run out of flowing water.",
#  "I wanted to ensure that I positively contributed to society, while my parents believed that organ donation was an unfamiliar and unnecessary cultural taboo. My curiosity strengthens with each hurdle and has expanded into a pure love of learning new things.",
#  "I've become someone who seeks to understand things at a fundamental level and who finds excitement in taking on big questions that have yet to be solved." ,
#  "After learning about variables for the first time in 5th grade Algebra, I began to treat each aspect of Kobe's jump shot as a different variable, each combination of variables resulting in a unique solution." ,
#  "While in 7th-grade geometry, I graphed the arc of his shot, and after learning about quadratic equations in 8th grade, I expressed his shot as a parabolic function that would ensure a swish when shooting from any spot."]

#result_test = sent_sim_analysis(college_info_data, ps_supp_esssay_data)
#print("result :", result_test)
#### ------------ end ------------- ####
########################################


def sent_sim_analysis_with_bert_summarizer(select_pmt_type, select_college, select_major, coll_supp_essay_input_data):
    College_data = open_data_for_sent_sim(select_college) # 선택한 대학의 정보가 담긴 txt 파일을 불러오고
    #print('college_data:', College_data)
    #print('===========================================================================================')
    Mjr_data = open_major_data_for_sent_sim(select_college, select_major) # 선택한 대학과 전공의 정보를 불러와서
    #print('Major_data : ', Mjr_data)
    #print('===========================================================================================')

    # 입력정보를 요약한다. 계산을 빠르게 하기 위해서
    model = Summarizer()

    # 대학정보 요약
    college_result = model(College_data, min_length=60)
    coll_result = ''.join(college_result)
    coll_result = re.sub(r"[\n\\]", "", coll_result) # 공백문자 제거
    coll_result = coll_result.rstrip('\n')
    coll_result = sent_tokenize(coll_result)
    print('===========================================================================================')
    print('대학정보 요약 : ', coll_result)

    # 전공정보 요약
    mjr_result = model(Mjr_data, min_length=60)
    mjr_result = ''.join(mjr_result)
    mjr_result = re.sub(r"[\n\\]", "", mjr_result) # 공백문자 제거
    mjr_result = mjr_result.rstrip('\n')
    mjr_result = sent_tokenize(mjr_result)
    print('===========================================================================================')
    print('전공정보 요약 : ', mjr_result)

    # college info + major info
    coll_mjr_result = coll_result + mjr_result

    # 에세이 요약
    ps_supp_result = model(coll_supp_essay_input_data, min_length=60)
    ps_supp_result_ = ''.join(ps_supp_result)
    ps_supp_result_ = re.sub(r"[\n\\]", "", ps_supp_result_) # 공백문자 제거
    ps_supp_result_ = ps_supp_result_.rstrip('\n')
    ps_supp_result_ = sent_tokenize(ps_supp_result_)
    print('===========================================================================================')
    print('에세이 요약 : ', ps_supp_result_)

    # 대학정보, 입력에세이의 연관성 분석
    # 이 결과는 개발중간 확인 위한 값들임
    # return coll_result, ps_supp_result_


    result = sent_sim_analysis(coll_mjr_result, ps_supp_result_)
    return result






# ## 실행 방법 ##

# input College Supp Essay 
essay_input = """I inhale deeply and blow harder than I thought possible, pushing the tiny ember from its resting place on the candle out into the air. The room erupts around me, and 'Happy Birthday!' cheers echo through the halls. It's time to make a wish. In my mind, that new Limited Edition Deluxe Ben 10 watch will soon be mine. My parents and the aunties and uncles around me attempt to point me in a different direction. 'Wish that you get to go to the temple every day when you're older! Wish that you memorize all your Sanskrit texts before you turn 6! Wish that you can live in India after college!' My ears listen, but my mind tunes them out, as nothing could possibly compare to that toy watch! What I never realized on my third birthday is that those wishes quietly tell the story of how my family hopes my life will play out. In this version of my life, there wasn't much room for change, personal growth, or 'rocking the boat.' A vital aspect of my family's cultural background is their focus on accepting things as they are. Growing up, I was discouraged from questioning others or asking questions that didn't have definitive yes or no answers. If I innocently asked my grandma why she expected me to touch her feet, my dad would grab my hand in a sudden swoop, look me sternly in the eye, and tell me not to disrespect her like that again. At home, if I mentioned that I had tried eggs for breakfast at a friend's house, I'd be looked at like I had just committed a felony for eating what my parents considered meat. If I asked the priest at the temple why he had asked an Indian man and his white wife to leave, I'd be met with a condescending glare and told that I should also leave for asking such questions.In direct contrast, my curiosity was invited and encouraged at school. After an environmental science lesson, I stayed for a few minutes after class to ask my 4th-grade science teacher with wide eyes how it was possible that Niagara Falls doesn't run out of flowing water. Instead of scolding me for asking her a 'dumb question,' she smiled and explained the intricacy of the water cycle. Now, if a teacher mentions that we'll learn about why a certain proof or idea works only in a future class, I'll stay after to ask more or pour through an advanced textbook to try to understand it. While my perspective was widening at school, the receptiveness to raising complex questions at home was diminishing. After earning my driver's license, I registered as an organ donor. My small checkmark on a piece of paper led to an intense clash between my and my parents' moral platform. I wanted to ensure that I positively contributed to society, while my parents believed that organ donation was an unfamiliar and unnecessary cultural taboo. I would often ask for clarity or for reasons that supported their ideologies. Their response would usually entail feeling a deep, visceral sense that traditions must be followed exactly as taught, without objection. Told in one language to keep asking questions and in another to ask only the right ones, I chose exploring questions that don't have answers, rather than accepting answers that don't get questioned. When it comes to the maze of learning, even when I take a wrong turn and encounter roadblocks that are meant to stop me, I've learned to climb over them and keep moving forward. My curiosity strengthens with each hurdle and has expanded into a pure love of learning new things. I've become someone who seeks to understand things at a fundamental level and who finds excitement in taking on big questions that have yet to be solved. I'm no longer afraid to rock the boat. "},{"index":1,"personal_essay":"Ever since I first held a small foam Spiderman basketball in my tiny hands and watched my idol Kobe Bryant hit every three-pointer he attempted, I've wanted to understand and replicate his flawless jump shot. As my math education progressed in school, I began to realize I had the tools to create a perfect shot formula. After learning about variables for the first time in 5th grade Algebra, I began to treat each aspect of Kobe's jump shot as a different variable, each combination of variables resulting in a unique solution. While in 7th-grade geometry, I graphed the arc of his shot, and after learning about quadratic equations in 8th grade, I expressed his shot as a parabolic function that would ensure a swish when shooting from any spot. After calculus lessons in 10th and 11th grade, I was excited to finally solve for the perfect velocity and acceleration needed on my release. At Brown, I hope to explore this intellectual pursuit through a different lens. What if I could maximize the odds of making shots if I understood the science behind one's mental mindset and focus through CLPS 500: Perception and Action? Or use astrophysics to account for drag and gravitational force anywhere in the universe? Or use data science to break down the analytics of the NBA's best shooters? Through the Open Curriculum, I see myself not only becoming a more complete learner, but also a more complete thinker, applying a flexible mindset to any problem I encounter. Brown's Open Curriculum allows students to explore broadly while also diving deeply into their academic pursuits. Tell us about an academic interest (or interests) that excites you, and how you might use the Open Curriculum to pursue it. I've been playing the Mridangam since I was five years old. It's a simple instrument: A wood barrel covered on two ends by goatskin with leather straps surrounding the hull. This instrument serves as a connection between me and one of the most beautiful aspects of my culture: Carnatic music. As a young child, I'd be taken to the temple every weekend for three-hour-long Carnatic music concerts, where the most accomplished teenagers and young adults in our local Indian community would perform. I would watch in awe as the mridangists' hands moved gracefully, flowing across the goatskin as if they weren't making contact, while simultaneously producing sharp rhythmic patterns that never failed to fall on the beat. Hoping to be like these idols on the stage, I trained intensely with my teacher, a strict man who taught me that the simple drum I was playing had thousands of years of culture behind it. Building up from simple strokes, I realized that the finger speed I'd had been awestruck by wasn't some magical talent, it was instead a science perfected by repeated practice."""

# 입력: (prompt, college name, intended major, supp_essay)
re_sent_sim_analy = sent_sim_analysis_with_bert_summarizer('why_us', 'Brown', 'African Studies', essay_input)

print('Result : ', re_sent_sim_analy)

