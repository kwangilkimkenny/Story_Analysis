
Theme 폴더 안에 다양한 파일이 있어야 한다.

실행파일은 ai_theme_section.py 이다.

자세한 설명은 ai_theme_section.py 에 주석을 확인하면 된다.


이하 내용은 주석내용과 같다. 


# ##### 질문을 선택하고, 에세이를 입력한다. #####

# # >>>>>>>>  1명 학생의 입력데이터
# input_text = """Bloomington Normal is almost laughably cliché for a midwestern city. Vast swathes of corn envelop winding roads and the heady smell of BBQ smoke pervades the countryside every summer. Yet, underlying the trite norms of Normal is the prescriptive force of tradition—the expectation to fulfill my role as a female Filipino by playing Debussy in the yearly piano festival and enrolling in multivariable calculus instead of political philosophy.So when I discovered the technical demand of bebop, the triplet groove, and the intricacies of chordal harmony after ten years of grueling classical piano, I was fascinated by the music's novelty. Jazz guitar was not only evocative and creative, but also strangely liberating. I began to explore different pedagogical methods, transcribe solos from the greats, and experiment with various approaches until my own unique sound began to develop. And, although I did not know what would be the 'best' route for me to follow as a musician, the freedom to forge whatever path I felt was right seemed to be exactly what I needed; there were no expectations for me to continue in any particular way—only the way that suited my own desires.While journeying this trail, I found myself at Interlochen Arts Camp the summer before my junior year. Never before had I been immersed in an environment so conducive to musical growth: I was surrounded by people intensely passionate about pursuing all kinds of art with no regard for ideas of what art 'should' be. I knew immediately that this would be a perfect opportunity to cultivate my sound, unbounded by the limits of confining tradition. On the first day of camp, I found that my peer guitarist in big band was another Filipino girl from Illinois. Until that moment, my endeavors in jazz guitar had been a solitary effort; I had no one with whom to collaborate and no one against whom I could compare myself, much less someone from a background mirroring my own. I was eager to play with her, but while I quickly recognized a slew of differences between us—different heights, guitars, and even playing styles—others seemed to have trouble making that distinction during performances. Some even went as far as calling me 'other-Francesca.' Thus, amidst the glittering lakes and musky pine needles of Interlochen, I once again confronted Bloomington's frustrating expectations.After being mistaken for her several times, I could not help but view Francesca as a standard of what the 'female Filipino jazz guitarist' should embody. Her improvisatory language, comping style and even personal qualities loomed above me as something I had to live up to. Nevertheless, as Francesca and I continued to play together, it was not long before we connected through our creative pursuit. In time, I learned to draw inspiration from her instead of feeling pressured to follow whatever precedent I thought she set. I found that I grew because of, rather than in spite of, her presence; I could find solace in our similarities and even a sense of comfort in an unfamiliar environment without being trapped by expectation. Though the pressure to conform was still present—and will likely remain present in my life no matter what genre I'm playing or what pursuits I engage in—I learned to eschew its corrosive influence and enjoy the rewards that it brings. While my encounter with Francesca at first sparked a feeling of pressure to conform in a setting where I never thought I would feel its presence, it also carried the warmth of finding someone with whom I could connect. Like the admittedly trite conditions of my hometown, the resemblances between us provided comfort to me through their familiarity. I ultimately found that I can embrace this warmth while still rejecting the pressure to succumb to expectations, and that, in the careful balance between these elements, I can grow in a way that feels both like discove"""

# #>>>>>> 6개의 질문  ques_one, ques_two, ques_three, ques_four, ques_five, ques_six   중 선택 1개
# question_num = """ques_one""" # 1번째 질문을 선택했을 경우


# #############   실행 테스트   ################

# print("RESULT :", theme_all_section(input_text, question_num))


# 1명의 에세이 결과 계산점수 : [9.224401651139463, 3.97]
# min_ 3
# max_:  15
# div_: 3
# cal_abs 절대값 : 0.6945983488605378
# compare7 : 3.190566941856577
# compare6 : 3.8286803302278924
# compare5 : 4.785850412784866
# compare4 : 6.381133883713154
# compare3 : 9.571700825569732
# Ideal: 1
# min_ 1
# max_:  6
# div_: 1
# cal_abs 절대값 : 0.001000000000000334
# compare7 : 1.3231666666666666
# compare6 : 1.5878
# compare5 : 1.98475
# compare4 : 2.646333333333333
# compare3 : 3.9695
# Ideal: 1

# 셜명  >>>>>> [1, 1, 5.0] 은 순서대로 첫번째 것인 Contextual Semantic Search, Narrativity, Ooverall Theme Rating이다.  그 다음에  {'approval': 7, 'admiration': 3, 'realization': 3, 'amusement': 1, ... 는 워드클라우드다.

# 최종결과 :  ([1, 1, 5.0], {'approval': 7, 'admiration': 3, 'realization': 3, 'amusement': 1, 'confusion': 1, 'excitement': 1, 'annoyance': 1}, {'i': 27, 'filipino': 3, 'expectations': 3, 'way': 3, 'francesca': 3, 'pressure': 3, 'summer': 2, 'tradition': 2, 'expectation': 2, 'piano': 2, 's': 2, 'guitar': 2, 'sound': 2, 'interlochen': 2, 'environment': 2, 'art': 2, 'guitarist': 2, 'jazz': 2, 'one': 2, 'someone': 2, 'presence': 2, 'comfort': 2, 'warmth': 2, "'bloomington": 1, 'city': 1, 'swathes': 1, 'corn': 1, 'envelop': 1, 
# 'roads': 1, 'smell': 1, 'bbq': 1, 'smoke': 1, 'countryside': 1, 'norms': 1, 'force': 1, 'role': 1, 'debussy': 1, 'festival': 1, 'philosophy': 1, 'demand': 1, 'triplet': 1, 'groove': 1, 'intricacies': 1, 'chordal': 1, 'harmony': 1, 'years': 1, 'music': 1, 'novelty': 1, 'methods': 1, 'solos': 1, 'greats': 1, 'experiment': 1, 'approaches': 1, 'route': 1, 'freedom': 1, 'path': 1, 'desires': 1, 'trail': 1, 'arts': 1, 'year': 1, 'growth': 1, 'people': 1, 'kinds': 1, 'regard': 1, 'ideas': 1, 'opportunity': 1, 'limits': 1, 'day': 
# 1, 'camp': 1, 'peer': 1, 'band': 1, 'girl': 1, 'illinois': 1, 'moment': 1, 'endeavors': 1, 'effort': 1, 'background': 1, 'slew': 1, 'differences': 1, 'heights': 1, 'guitars': 1, 'styles': 1, 'others': 1, 'trouble': 1, 'distinction': 1, 'performances': 1, 'glittering': 1, 'lakes': 1, 'pine': 1, 'needles': 1, 'bloomington': 1, 'mistaken': 1, 'times': 1, 'standard': 1, 'language': 1, 'style': 1, 'qualities': 1, 'something': 1, 'pursuit': 1, 'time': 1, 'inspiration': 1, 'precedent': 1, 'spite': 1, 'solace': 1, 'similarities': 
# 1, 'sense': 1, 'life': 1, 'matter': 1, 'm': 1, 'engage': 1, 'influence': 1, 'rewards': 1, 'encounter': 1, 'feeling': 1, 'setting': 1, 'conditions': 1, 'hometown': 1, 'resemblances': 1, 'familiarity': 1, 'balance': 1, 'elements': 1, 'discove': 1, ']': 1}, {'i': 27, 'filipino': 3, 'expectations': 3, 'way': 3, 'francesca': 3, 'pressure': 3, 'summer': 2, 'tradition': 2, 'expectation': 2, 'piano': 2, 's': 2, 'guitar': 2, 'sound': 2, 'interlochen': 2, 'environment': 2, 'art': 2, 'guitarist': 2, 'jazz': 2, 'one': 2, 'someone': 2, 'presence': 2, 'comfort': 2, 'warmth': 2, "'bloomington": 1, 'city': 1, 'swathes': 1, 'corn': 1, 'envelop': 1, 'roads': 1, 'smell': 1, 'bbq': 1, 'smoke': 1, 'countryside': 1, 'norms': 1, 'force': 1, 'role': 1, 'debussy': 1, 'festival': 1, 'philosophy': 1, 'demand': 1, 'triplet': 1, 'groove': 1, 'intricacies': 1, 'chordal': 1, 'harmony': 1, 'years': 1, 'music': 1, 'novelty': 1, 'methods': 1, 'solos': 1, 'greats': 1, 'experiment': 1, 'approaches': 1, 'route': 1, 'freedom': 1, 'path': 1, 'desires': 1, 'trail': 1, 'arts': 1, 
# 'year': 1, 'growth': 1, 'people': 1, 'kinds': 1, 'regard': 1, 'ideas': 1, 'opportunity': 1, 'limits': 1, 'day': 1, 'camp': 1, 'peer': 1, 'band': 1, 'girl': 1, 'illinois': 1, 'moment': 1, 'endeavors': 1, 'effort': 1, 'background': 1, 'slew': 1, 'differences': 1, 'heights': 1, 'guitars': 1, 'styles': 1, 'others': 1, 'trouble': 1, 'distinction': 1, 'performances': 1, 'glittering': 1, 'lakes': 1, 'pine': 1, 'needles': 1, 'bloomington': 1, 'mistaken': 1, 'times': 1, 'standard': 1, 'language': 1, 'style': 1, 'qualities': 1, 'something': 1, 'pursuit': 1, 'time': 1, 'inspiration': 1, 'precedent': 1, 'spite': 1, 'solace': 1, 'similarities': 1, 'sense': 1, 'life': 1, 'matter': 1, 'm': 1, 'engage': 1, 'influence': 1, 'rewards': 1, 'encounter': 1, 'feeling': 1, 'setting': 1, 'conditions': 1, 'hometown': 1, 'resemblances': 1, 'familiarity': 1, 'balance': 1, 'elements': 1, 'discove': 1, ']': 1})



