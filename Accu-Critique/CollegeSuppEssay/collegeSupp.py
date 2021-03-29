import spacy
from collections import Counter
import nltk
nltk.download('averaged_perceptron_tagger')

nlp = spacy.load("en_core_web_sm")


from wordcloud import WordCloud 
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
#%matplotlib inline

import matplotlib
from IPython.display import set_matplotlib_formats
matplotlib.rc('font',family = 'Malgun Gothic')
set_matplotlib_formats('retina')
matplotlib.rc('axes',unicode_minus = False)




def select_prompt_type(prompt_type):
    
    if prompt_type == 'why_us':
        pmt_typ = [""" 'Why us' school & major interest (select major, by college & department) """]
    elif prompt_type == 'Intellectual interest':
        pmt_typ = [""]
    elif prompt_type == 'Meaningful experience & lesson learned':
        pmt_typ = [""]
    elif prompt_type ==  'Achievement you are proud of':
        pmt_typ = [""]
    elif prompt_type ==  'Social issues: contribution & solution':
        pmt_typ = [""]
    elif prompt_type ==  'Summer activity':
        pmt_typ = [""]
    elif prompt_type ==  'Unique quality, passion, or talent':
        pmt_typ = [""]
    elif prompt_type ==  'Extracurricular activity or work experience':
        pmt_typ = [""]
    elif prompt_type ==  'Your community: role and contribution in your community':
        pmt_typ = [""]
    elif prompt_type ==  'College community: intended role, involvement, and contribution in college community':
        pmt_typ = [""]
    elif prompt_type ==  'Overcoming a Challenge or ethical dilemma':
        pmt_typ = [""]
    elif prompt_type ==  'Culture & diversity':
        pmt_typ = [""]
    elif prompt_type ==  'Collaboration & teamwork':
        pmt_typ = [""]
    elif prompt_type ==  'Creativity/creative projects':
        pmt_typ = [""]
    elif prompt_type ==  'Leadership experience':
        pmt_typ = [""]
    elif prompt_type ==  'Values, perspectives, or beliefs':
        pmt_typ = [""]
    elif prompt_type ==  'Person who influenced you':
        pmt_typ = [""]
    elif prompt_type ==  'Favorite book/movie/quote':
        pmt_typ = [""]
    elif prompt_type ==  'Write to future roommate':
        pmt_typ = [""]
    elif prompt_type ==  'Diversity & Inclusion Statement':
        pmt_typ = [""]
    elif prompt_type ==  'Future goals or reasons for learning':
        pmt_typ = [""]
    elif prompt_type ==  'What you do for fun':
        pmt_typ = [""]
    else:
        pass
    

# Selected Colege
def selected_college(select_college):
    if select_college == 'Stanford':
        pass
    elif select_college == 'Cambridge':
        pass
    elif select_college == 'Princeton':
        pass
    elif select_college == 'MIT':
        pass
    elif select_college == 'Yale':
        pass
    elif select_college == 'Penn':
        pass
    elif select_college == 'Chicago':
        pass
    elif select_college == 'Northwestern':
        pass
    elif select_college == 'Duke':
        pass
    elif select_college == 'Johns Hopkins':
        pass
    elif select_college == 'Caltech':
        pass
    elif select_college == 'Dartmouth':
        pass
    elif select_college == 'Brown':
        pass
    elif select_college == 'Notre Dame':
        pass
    elif select_college == 'Vanderbilt':
        pass
    elif select_college == 'Cornell':
        pass
    elif select_college == 'Rice':
        pass
    elif select_college == 'WUSTL':
        pass
    elif select_college == 'UCLA':
        pass
    elif select_college == 'Emory':
        pass
    elif select_college == 'UC Berkeley':
        pass
    elif select_college == 'USC':
        pass
    elif select_college == 'Georgetown':
        pass
    elif select_college == 'Carnegie Mellon':
        pass
    elif select_college == 'UVA':
        pass
    else:
        pass




# 에세이 입력
input_text = """Bloomington Normal is almost laughably cliché for a midwestern city. Vast swathes of corn envelop winding roads and the heady smell of BBQ smoke pervades the countryside every summer. Yet, underlying the trite norms of Normal is the prescriptive force of tradition—the expectation to fulfill my role as a female Filipino by playing Debussy in the yearly piano festival and enrolling in multivariable calculus instead of political philosophy.So when I discovered the technical demand of bebop, the triplet groove, and the intricacies of chordal harmony after ten years of grueling classical piano, I was fascinated by the music's novelty. Jazz guitar was not only evocative and creative, but also strangely liberating. I began to explore different pedagogical methods, transcribe solos from the greats, and experiment with various approaches until my own unique sound began to develop. And, although I did not know what would be the 'best' route for me to follow as a musician, the freedom to forge whatever path I felt was right seemed to be exactly what I needed; there were no expectations for me to continue in any particular way—only the way that suited my own desires.While journeying this trail, I found myself at Interlochen Arts Camp the summer before my junior year. Never before had I been immersed in an environment so conducive to musical growth: I was surrounded by people intensely passionate about pursuing all kinds of art with no regard for ideas of what art 'should' be. I knew immediately that this would be a perfect opportunity to cultivate my sound, unbounded by the limits of confining tradition. On the first day of camp, I found that my peer guitarist in big band was another Filipino girl from Illinois. Until that moment, my endeavors in jazz guitar had been a solitary effort; I had no one with whom to collaborate and no one against whom I could compare myself, much less someone from a background mirroring my own. I was eager to play with her, but while I quickly recognized a slew of differences between us—different heights, guitars, and even playing styles—others seemed to have trouble making that distinction during performances. Some even went as far as calling me 'other-Francesca.' Thus, amidst the glittering lakes and musky pine needles of Interlochen, I once again confronted Bloomington's frustrating expectations.After being mistaken for her several times, I could not help but view Francesca as a standard of what the 'female Filipino jazz guitarist' should embody. Her improvisatory language, comping style and even personal qualities loomed above me as something I had to live up to. Nevertheless, as Francesca and I continued to play together, it was not long before we connected through our creative pursuit. In time, I learned to draw inspiration from her instead of feeling pressured to follow whatever precedent I thought she set. I found that I grew because of, rather than in spite of, her presence; I could find solace in our similarities and even a sense of comfort in an unfamiliar environment without being trapped by expectation. Though the pressure to conform was still present—and will likely remain present in my life no matter what genre I'm playing or what pursuits I engage in—I learned to eschew its corrosive influence and enjoy the rewards that it brings. While my encounter with Francesca at first sparked a feeling of pressure to conform in a setting where I never thought I would feel its presence, it also carried the warmth of finding someone with whom I could connect. Like the admittedly trite conditions of my hometown, the resemblances between us provided comfort to me through their familiarity. I ultimately found that I can embrace this warmth while still rejecting the pressure to succumb to expectations, and that, in the careful balance between these elements, I can grow in a way that feels both like discove"""




# 대학관련 정보의 토픽 키워드
def general_keywords(College_text_data):
    tokenized = nltk.word_tokenize(str(College_text_data))
    print('tokenized:', tokenized)
    nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if(pos[:2] == 'NN')]
    count = Counter(nouns)
    words = dict(count.most_common())
    print('words:', words)
    # 가장 많이 등장하는 단어를 추려보자. 얏 그런데 ii가 많네. 이건 선언문이라 그러함. 이런 단어는 모두 전처리에서 삭제할것!!!
    wordcloud = WordCloud(background_color='white',colormap = "Accent_r",
                            width=1500, height=1000).generate_from_frequencies(words)

    plt.imshow(wordcloud)
    plt.axis('off')
    gk_re = plt.show()

    return gk_re

## run ##
result =  general_keywords(input_text)
print("====================================")
print('result :', result)