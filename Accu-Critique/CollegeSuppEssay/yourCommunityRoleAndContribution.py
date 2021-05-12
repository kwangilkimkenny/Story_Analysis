#Prompt Type: Your community: role and contribution in your community


from yourCommunity_promptOrientedKeywords import pmtOrientedKeywords

from yourCommRoleAndContri_pmpt_ori_sentiment import comp_sentiment_both

from intellectualEngagement import intellectualEnguagement

essay_input = """This past summer, I had the privilege of participating in the University of Notre Dame’s Research Experience for Undergraduates (REU) program . Under the mentorship of Professor Wendy Bozeman and Professor Georgia Lebedev from the department of Biological Sciences, my goal this summer was to research the effects of cobalt iron oxide cored (CoFe2O3) titanium dioxide (TiO2) nanoparticles as a scaffold for drug delivery, specifically in the delivery of a compound known as curcumin, a flavonoid known for its anti-inflammatory effects. As a high school student trying to find a research opportunity, it was very difficult to find a place that was willing to take me in, but after many months of trying, I sought the help of my high school biology teacher, who used his resources to help me obtain a position in the program.				
Using equipment that a high school student could only dream of using, I was able to map apoptosis (programmed cell death) versus necrosis (cell death due to damage) in HeLa cells, a cervical cancer line, after treating them with curcumin-bound nanoparticles. Using flow cytometry to excite each individually suspended cell with a laser, the scattered light from the cells helped to determine which cells were living, had died from apoptosis or had died from necrosis. Using this collected data, it was possible to determine if the curcumin and/or the nanoparticles had played any significant role on the cervical cancer cells. Later, I was able to image cells in 4D through con-focal microscopy. From growing HeLa cells to trying to kill them with different compounds, I was able to gain the hands-on experience necessary for me to realize once again why I love science.				
Living on the Notre Dame campus with other REU students, UND athletes, and other summer school students was a whole other experience that prepared me for the world beyond high school. For 9 weeks, I worked, played and bonded with the other students, and had the opportunity to live the life of an independent college student.				
Along with the individually tailored research projects and the housing opportunity, there were seminars on public speaking, trips to the Fermi National Accelerator Laboratory, and one-on-one writing seminars for the end of the summer research papers we were each required to write. By the end of the summer, I wasn’t ready to leave the research that I was doing. While my research didn’t yield definitive results for the effects of curcumin on cervical cancer cells, my research on curcumin-functionalized CoFe2O4/TiO2 core-shell nanoconjugates indicated that there were many unknown factors affecting the HeLa cells, and spurred the lab to expand their research into determining whether or not the timing of the drug delivery mattered and whether or not the position of the binding site of the drugs would alter the results. Through this summer experience, I realized my ambition to pursue a career in research. I always knew that I would want to pursue a future in science, but the exciting world of research where the discoveries are limitless has captured my heart. This school year, the REU program has offered me a year-long job, and despite my obligations as a high school senior preparing for college, I couldn’t give up this offer, and so during this school year, I will be able to further both my research and interest in nanotechnology. """

# 40%
pmt_ori_key_re = pmtOrientedKeywords(essay_input)
print('your community role and contribution in your community _ Prompt Oriented Keywords 값 계산 결과:', pmt_ori_key_re[0])
print('your community role and contribution in your community _ Prompt Oriented Keywords - 입력에세이의 토픽과 비교할 단어가 얼마나 일치하는지에 대한 비율 계산 결과 :', pmt_ori_key_re[1])
print('your community role and contribution in your community _ Prompt Oriented Keywords 단어들(웹사이트에 표시) :', pmt_ori_key_re[2])
print('your community role and contribution in your community _ 관련 단어가 에세이에 포함된 비율 :', pmt_ori_key_re[3])
print('your community role and contribution in your community _ 합격생 평균에 비교하여 얻은 최종 값 :', pmt_ori_key_re[4])


select_pmt_type = 'Your community: role and contribution in your community'

def your_commu_role_contrib(essay_input, select_pmt_type):
    # 30%
    sentiment_re = comp_sentiment_both(essay_input, select_pmt_type)

    def calculate_score(input_scofre):
        if input_scofre >= 80:
            result_comm_ori_keywordss = 'Supurb'
        elif input_scofre >= 60 and input_scofre < 80:
            result_comm_ori_keywordss = 'Strong'
        elif input_scofre >= 40 and input_scofre < 60:
            result_comm_ori_keywordss = 'Good'
        elif input_scofre >= 20 and input_scofre < 40:
            result_comm_ori_keywordss = 'Mediocre'
        else: #input_scofre < 20
            result_comm_ori_keywordss = 'Lacking'
        return result_comm_ori_keywordss

    #supurb ~ lacking 을 숫자로 된 점수로 변환
    def text_re_to_score(input):
        if input == 'Supurb':
            tp_knowledge_re = 90
        elif input == 'Strong':
            tp_knowledge_re = 75
        elif input == 'Good':
            tp_knowledge_re = 65
        elif input == 'Mediocre':
            tp_knowledge_re = 40
        else: #input == 'Lacking'
            tp_knowledge_re = 10
        return tp_knowledge_re


    Engagement_result = intellectualEnguagement(essay_input)

    Engagement_result_fin = calculate_score(Engagement_result[0])
    engagement_re_final = text_re_to_score(Engagement_result_fin)

    # print('pmt_ori_key_re[1] : ', pmt_ori_key_re[1])
    # print(' sentiment_re[0] :',  sentiment_re[0])
    senti_fin_re = calculate_score(sentiment_re[0]) # 웹 우측 Prompt Oriented Sentiments 결과값 추출
    # print('Engagement_result_fin : ', Engagement_result_fin)
    # print('engagement_re_final:', engagement_re_final)

    overall_result = round(pmt_ori_key_re[1] * 0.4 + sentiment_re[0] * 0.3 + engagement_re_final * 0.3, 1)
    overall_5div_result = calculate_score(overall_result)


    # 문장생성
    fixed_top_comment = """Writing about one’s community involves multiple elements because it is a broad topic. It is also a crucial topic for college admissions to evaluate an applicant’s future contribution to the campus community. Therefore, your essay should demonstrate a sense of affection and pride towards the community you are a part of while defining your role and contribution to it."""

    def gen_comment(input_score, type):
        if input_score == 'Supurb' or input_score == 'Strong':
            if type == 'comm_ori_keywords':
                comment_achieve = """Your essay indicates a wealth of content associated with your community in terms of social awareness and contribution."""
            elif type == 'pmt_ori_sentiment':
                comment_achieve = """Also, your essay seems to contain strongly positive sentiments for being a part of the community."""
            elif type == 'engagement':
                comment_achieve = """All in all, your story seems to demonstrate a high level of effort and dedication for the ones around you."""
            else:
                pass
        elif input_score == 'Good':
            if type == 'comm_ori_keywords':
                comment_achieve = """Your essay indicates some contents associated with your community in terms of social awareness and contribution."""
            elif type == 'pmt_ori_sentiment':
                comment_achieve = """Also, your essay seems to contain sufficient positive sentiments for being a part of the community."""
            elif type == 'engagement':
                comment_achieve = """All in all, your story seems to demonstrate a satisfactory level of effort and dedication for the ones around you."""
            else:
                pass
        else: #input score == 'Mediocre' or input_score == 'Weak'
            if type == 'comm_ori_keywords':
                comment_achieve = """Your essay indicates lacking contents associated with your community in terms of social awareness and contribution."""
            elif type == 'pmt_ori_sentiment':
                comment_achieve = """Your essay may need to include more positive sentiments for being a part of the community."""
            elif type == 'engagement':
                comment_achieve = """All in all, you may need to demonstrate a higher level of effort and dedication for the ones around you."""
            else:
                pass
        return comment_achieve


    comment_1 = gen_comment(pmt_ori_key_re[1], 'comm_ori_keywords')
    comment_2 = gen_comment(sentiment_re[0], 'pmt_ori_sentiment')
    comment_3 = gen_comment(Engagement_result_fin, 'engagement')

    data = {
        'overall_result' : overall_result, # number
        'overall_5div_result' : overall_5div_result, # Supurb ~ Lacking

        'pmt_ori_key_re[0]' : pmt_ori_key_re[0], # Community Oriented Keywords
        'senti_fin_re' : senti_fin_re, # Prompt Oriented Sentiments
        'Engagement_result_fin' : Engagement_result_fin, # Initiative & Engagement 결과값(우측에 표시되는 부분)

        'fixed_top_comment' : fixed_top_comment,
        'comment_1' : comment_1, 
        'comment_2' : comment_2, 
        'comment_3' : comment_3,

        'pmt_ori_key_re[2]' : pmt_ori_key_re[2], # Community Oriented Keywords 본문에 표시할 단어리스트
        # Prompt Oriented Sentiments 본문에 표시할 문장별 분석에 해당하는 감성단어 반영 리스트 추출해야함(시간없어서 아직 안함)
        'Engagement_result[3]' : Engagement_result[3],  # Initiative & Engagement 추출 단어(웹에 표시함)

    }

    return data



### run ###
select_pmt_type = 'Your community: role and contribution in your community'
result_comt_role_contrib = your_commu_role_contrib(essay_input, select_pmt_type)

print('result : ', result_comt_role_contrib)

