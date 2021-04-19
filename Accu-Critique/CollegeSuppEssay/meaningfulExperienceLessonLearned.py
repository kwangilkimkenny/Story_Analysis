#### Prompt Type: Meaningful experience & lesson learned 는 이 코드를 실행해야 함 (두개의 코드를 연결되어 있음)####
## 연결코드 key_literary_elements.py, perspective ##

# 1) Key Literary Elements (40%):
# 그냥 PS 평가 로직 그대로 활용해서 총괄 점수

# 2) Prompt Oriented Sentiments (20%):
# Main: realization, approval, admiration, gratitude
# Sub (가산점): confusion, disappointment, caring

# 3)  Originality (Topic detection & 단어 간 vector 거리) (20%)
# Key Academic Topics 간의 vector 거리 계산 해야 해요.
# -너무 가까우면 너무 재미 없는거고
# -너무 멀면 헛소리
# -어느정도 거리가 있으면 (살짝 중간 이상 거리 ?) creative 한거고
# 이거는 잘 고민해 주셔요.
# Creative (문학적) Writing Sample 필요? (linear vs. multi-dimension)

# 4) Perspective (20%) - 나의 관점, 나의 의견 표출
# 로직: Opinion/Belief words (70%) + viewpoint adverbs (30%)  + emphasizing adjectives 

from key_literary_elements import key_literary_element
from perspective import getPerspectiveWD


def MeaningFullExpreenceLessonLearened(essay_input):
    # 1) Key Literary Elements (40%)
    KLE = key_literary_element(essay_input, 'Meaningful experience & lesson learned').get('key_literary_elements')
    print('KLE:', KLE)
    # 2) Prompt Oriented Sentiments (20%)
    PMS = key_literary_element(essay_input, 'Meaningful experience & lesson learned').get('emotion_result')
    PMS_re = PMS[1]
    print('PMS:', PMS_re)
    # 3) Originality (Topic detection & 단어 간 vector 거리 : Cohesion value) (20%)
    Org = key_literary_element(essay_input, 'Meaningful experience & lesson learned').get('originality')
    print('Org:', Org)
    # Perspective(20%)
    PS = getPerspectiveWD(essay_input)
    PS_re = PS[0]
    print('PS:', PS_re)

    # Get MeaningFullExpreenceLessonLearened
    overall_re = round(KLE * 0.4 + PMS_re * 0.2 + Org * 0.2 + PS_re * 0.2, 2)

    def get_score(score_input):
        # 5단계로 계산
        if score_input > 80:
            grade_re = 'Supurb'
            score_re = 100 # Intellectual interest 를 최종 계산하기 위해 변화한 점수
        elif score_input > 60 and score_input <= 80:
            grade_re = 'Strong'
            score_re = 80
        elif score_input > 40 and score_input <= 60:
            grade_re = 'Good'
            score_re = 60
        elif score_input > 20 and score_input <= 40:
            grade_re = 'Mediocre'
            score_re = 40
        else:
            grade_re = 'Lacking'
            score_re = 20
        return grade_re

    overall_score = get_score(overall_re)
    KeyKiterElement_score = get_score(KLE)
    PromptOriented_score = get_score(PMS_re)
    Originality_score = get_score(Org)
    Perspective_score = get_score(PS_re)

    # perspective에서 웹에 표시하는 문장추출 부분
    perspective_analysis_result_for_web = PS[1] + PS[2]


    # 웹에 표시해야 할 부분으로 Prompt Type: Meaningful experience & lesson learned 에서는 key literary elements는 아래 3개 plot & conflict, character, setting 관련 단어가 모두 표시되어야 한다. 
    plot_n_conflict_word_for_web = key_literary_element(essay_input, 'Meaningful experience & lesson learned').get('plot_n_conflict_word_for_web')
    characgter_words_for_web = key_literary_element(essay_input, 'Meaningful experience & lesson learned').get('characgter_words_for_web')
    setting_words_list = key_literary_element(essay_input, 'Meaningful experience & lesson learned').get('setting_words_list')

    # Comments 생성 부분
    fixed_top_comment = """For meaningful experience and lessons learned, you may write about any occasion in your life as long as it had an impact on your life. One can assume that they are looking for a unique story, your own perspective, and a lesson that presented you with a positive outlook in life."""

    # 입력데이터 타입 : keylitElemt, pmtOrigSent, Originality, perspective 중 한개와 get_value (점수)
    def genComment(data_type, get_value):
        if get_value == 'Superb' or get_value == 'Strong':
            if data_type == 'keylitElemt':
                meanFul_Len_comment = """A comprehensive review indicates that your story seems quite strong in terms of the key literary elements in general, such as character, plot & conflict, and setting."""
            elif data_type == 'pmtOrigSent':
                meanFul_Len_comment = """In addition, your writing effectively displays the sentiments closely correlated with positive life lessons."""
            elif data_type == 'Originality':
                meanFul_Len_comment = """Your story seems quite original and versatile since you successfully connect the dots between seemingly distant topics and ideas."""
            elif data_type == "perspective":
                meanFul_Len_comment = """Lastly, your perspective on the meaning and lesson seems very clear, considering your words of emphasis in the essay."""
            else:
                pass
        elif get_value == 'Good':
            if data_type == 'keylitElemt':
                meanFul_Len_comment = """A comprehensive review indicates that your story seems satisfactory in terms of the key literary elements in general, such as character, plot & conflict, and setting."""
            elif data_type == 'pmtOrigSent':
                meanFul_Len_comment = """In addition, your writing seems to display some sentiments correlated with positive life lessons."""
            elif data_type == 'Originality':
                meanFul_Len_comment = """Your story seems original and interesting since you successfully connect the dots between various topics and ideas."""
            elif data_type == "perspective":
                meanFul_Len_comment = """Lastly, your perspective on the meaning and lesson seems clear enough, considering your words of emphasis in the essay."""
            else:
                pass
        elif get_value == 'Mediocore' or 'Weak':
            if data_type == 'keylitElemt':
                meanFul_Len_comment = """A comprehensive review indicates that your story may need reinforcement on the key literary elements in general, such as character, plot & conflict, and setting."""
            elif data_type == 'pmtOrigSent':
                meanFul_Len_comment = """In addition, you may consider expressing the sentiments, such as gratitude, admiration, and realization, which are correlated with positive lessons in life."""
            elif data_type == 'Originality':
                meanFul_Len_comment = """You may consider including various topics and ideas to make your essay sound more original and interesting."""
            elif data_type == "perspective":
                meanFul_Len_comment = """Lastly, you may consider elaborating further on your beliefs and opinions to solidify your viewpoint further."""
            else:
                pass

        return meanFul_Len_comment

    # comment generator
    KLE_comment = genComment('keylitElemt', KeyKiterElement_score)
    keylitElemt_comment = genComment('pmtOrigSent', PromptOriented_score)
    Originality_comment = genComment('Originality', Originality_score)
    perspective_comment = genComment('perspective', Perspective_score)


    ### return 설명 ###
    # 0. overall_result_fin_re : overall score
    # 1. KeyKiterElement_score
    # 2. PromptOriented_score
    # 3. Originality_score
    # 4. Perspective_score

    # 5. plot_n_conflict_word_for_web : 웹에 표시되는 단어 리스트
    # 6. characgter_words_for_web : 웹에 표시되는 단어 리스트
    # 7. setting_words_list : 웹에 표시되는 단어 리스트
    # 8. perspective_analysis_result_for_web :  웹에 표시되는 단어+문장 리스트

    # 9. fixed_top_comment :  코멘트 생성
    # 10. KLE_comment : 코멘트 생성
    # 11. keylitElemt_comment : 코멘트 생성
    # 12. Originality_comment : 코멘트 생성
    # 13. perspective_comment : 코멘트 생성

    data = {
        'overall_score' : overall_score, 
        'KeyKiterElement_score' : KeyKiterElement_score, 
        'PromptOriented_score' : PromptOriented_score, 
        'Originality_score': Originality_score, 
        'Perspective_score': Perspective_score, 
        'plot_n_conflict_word_for_web' : plot_n_conflict_word_for_web, 
        'characgter_words_for_web' : characgter_words_for_web, 
        'setting_words_list' : setting_words_list, 
        'perspective_analysis_result_for_web' : perspective_analysis_result_for_web, 
        'fixed_top_comment' : fixed_top_comment, 
        'KLE_comment' : KLE_comment, 
        'keylitElemt_comment' : keylitElemt_comment, 
        'Originality_comment' : Originality_comment, 
        'perspective_comment' : perspective_comment
    }

    return data





## Run ##

# input College Supp Essay 
essay_input = """I inhale deeply and blow harder than I thought possible, pushing the tiny ember from its resting place on the candle out into the air. The room erupts around me, and 'Happy Birthday!' cheers echo through the halls. It's time to make a wish. In my mind, that new Limited Edition Deluxe Ben 10 watch will soon be mine. My parents and the aunties and uncles around me attempt to point me in a different direction. 'Wish that you get to go to the temple every day when you're older! Wish that you memorize all your Sanskrit texts before you turn 6! Wish that you can live in India after college!' My ears listen, but my mind tunes them out, as nothing could possibly compare to that toy watch! What I never realized on my third birthday is that those wishes quietly tell the story of how my family hopes my life will play out. In this version of my life, there wasn't much room for change, personal growth, or 'rocking the boat.' A vital aspect of my family's cultural background is their focus on accepting things as they are. Growing up, I was discouraged from questioning others or asking questions that didn't have definitive yes or no answers. If I innocently asked my grandma why she expected me to touch her feet, my dad would grab my hand in a sudden swoop, look me sternly in the eye, and tell me not to disrespect her like that again. At home, if I mentioned that I had tried eggs for breakfast at a friend's house, I'd be looked at like I had just committed a felony for eating what my parents considered meat. If I asked the priest at the temple why he had asked an Indian man and his white wife to leave, I'd be met with a condescending glare and told that I should also leave for asking such questions.In direct contrast, my curiosity was invited and encouraged at school. After an environmental science lesson, I stayed for a few minutes after class to ask my 4th-grade science teacher with wide eyes how it was possible that Niagara Falls doesn't run out of flowing water. Instead of scolding me for asking her a 'dumb question,' she smiled and explained the intricacy of the water cycle. Now, if a teacher mentions that we'll learn about why a certain proof or idea works only in a future class, I'll stay after to ask more or pour through an advanced textbook to try to understand it. While my perspective was widening at school, the receptiveness to raising complex questions at home was diminishing. After earning my driver's license, I registered as an organ donor. My small checkmark on a piece of paper led to an intense clash between my and my parents' moral platform. I wanted to ensure that I positively contributed to society, while my parents believed that organ donation was an unfamiliar and unnecessary cultural taboo. I would often ask for clarity or for reasons that supported their ideologies. Their response would usually entail feeling a deep, visceral sense that traditions must be followed exactly as taught, without objection. Told in one language to keep asking questions and in another to ask only the right ones, I chose exploring questions that don't have answers, rather than accepting answers that don't get questioned. When it comes to the maze of learning, even when I take a wrong turn and encounter roadblocks that are meant to stop me, I've learned to climb over them and keep moving forward. My curiosity strengthens with each hurdle and has expanded into a pure love of learning new things. I've become someone who seeks to understand things at a fundamental level and who finds excitement in taking on big questions that have yet to be solved. I'm no longer afraid to rock the boat. "},{"index":1,"personal_essay":"Ever since I first held a small foam Spiderman basketball in my tiny hands and watched my idol Kobe Bryant hit every three-pointer he attempted, I've wanted to understand and replicate his flawless jump shot. As my math education progressed in school, I began to realize I had the tools to create a perfect shot formula. After learning about variables for the first time in 5th grade Algebra, I began to treat each aspect of Kobe's jump shot as a different variable, each combination of variables resulting in a unique solution. While in 7th-grade geometry, I graphed the arc of his shot, and after learning about quadratic equations in 8th grade, I expressed his shot as a parabolic function that would ensure a swish when shooting from any spot. After calculus lessons in 10th and 11th grade, I was excited to finally solve for the perfect velocity and acceleration needed on my release. At Brown, I hope to explore this intellectual pursuit through a different lens. What if I could maximize the odds of making shots if I understood the science behind one's mental mindset and focus through CLPS 500: Perception and Action? Or use astrophysics to account for drag and gravitational force anywhere in the universe? Or use data science to break down the analytics of the NBA's best shooters? Through the Open Curriculum, I see myself not only becoming a more complete learner, but also a more complete thinker, applying a flexible mindset to any problem I encounter. Brown's Open Curriculum allows students to explore broadly while also diving deeply into their academic pursuits. Tell us about an academic interest (or interests) that excites you, and how you might use the Open Curriculum to pursue it. I've been playing the Mridangam since I was five years old. It's a simple instrument: A wood barrel covered on two ends by goatskin with leather straps surrounding the hull. This instrument serves as a connection between me and one of the most beautiful aspects of my culture: Carnatic music. As a young child, I'd be taken to the temple every weekend for three-hour-long Carnatic music concerts, where the most accomplished teenagers and young adults in our local Indian community would perform. I would watch in awe as the mridangists' hands moved gracefully, flowing across the goatskin as if they weren't making contact, while simultaneously producing sharp rhythmic patterns that never failed to fall on the beat. Hoping to be like these idols on the stage, I trained intensely with my teacher, a strict man who taught me that the simple drum I was playing had thousands of years of culture behind it. Building up from simple strokes, I realized that the finger speed I'd had been awestruck by wasn't some magical talent, it was instead a science perfected by repeated practice."""

print("Get MeaningFullExpreenceLessonLearened :", MeaningFullExpreenceLessonLearened(essay_input))


# result: Get MeaningFullExpreenceLessonLearened : 38.01