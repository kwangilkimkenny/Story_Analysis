#Check GPU
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices()) 

import re

from happytransformer import HappyGeneration
from happytransformer import GENSettings

happy_gen = HappyGeneration("GPT-NEO", "EleutherAI/gpt-neo-2.7B")

def selectFromDict(options, name):

    index = 0
    indexValidList = []
    print('Select a ' + name + ':')
    for optionName in options:
        index = index + 1
        indexValidList.extend([options[optionName]])
        print(str(index) + ') ' + optionName)
    inputValid = False
    while not inputValid:
        inputRaw = input(name + ': ')
        inputNo = int(inputRaw) - 1
        if inputNo > -1 and inputNo < len(indexValidList):
            selected = indexValidList[inputNo]
            print('Selected ' +  name + ': ' + selected)
            inputValid = True
            break
        else:
            print('Please select a valid ' + name + ' number')

    return selected


def GenText(input_txt):
    tmp = [0.45, 0.50, 0.55, 0.75]
    resp = []
    for temp in tmp:
        top_k_sampling_settings = GENSettings(do_sample=True, 
                                                top_k=120, 
                                                temperature=temp,  
                                                max_length=50, 
                                                no_repeat_ngram_size=2)

        result_top_k_sampling = happy_gen.generate_text(input_txt, args=top_k_sampling_settings)

        #전처리 필요함 GenerationResult(text=' 
        result_top_k_sampling_re = result_top_k_sampling.text
        result_top_k_sampling_re.strip("'")
        result_top_k_sampling_re.lstrip("GenerationResult(text=' ")
        result_top_k_sampling_re = re.sub(r"\n", "", result_top_k_sampling_re)


        resp.append(result_top_k_sampling_re)
    return resp


def ExceptionInput(input_txt):
    if (input_txt.isdigit()):
        print("An incorrect input was detected. Try again.")

    elif (len(input_txt) <= 8):
        print("The sentence is too short. Try again.")

    elif (input_txt == None):
        print("Try again.")
    else:
        pass

    return input_txt


def brainStorming_01():
    print("-----------------")
    # Question 1
    print("")
    print("Question 1")
    print("What are your trying to write about among the following? Backgound / Identy / Interest / Talent")
    questions = {}
    questions['Background'] = 'Background'
    questions['Identy'] = 'Identy'
    questions['Interest'] = 'Interest'
    questions['Talent'] = 'Talent'

    try:
        # Let user select a script
        selected_prompt_input = selectFromDict(questions, 'Question')
    except:
        selected_prompt_input = selectFromDict(questions, 'Question')
        ExceptionInput(selected_prompt_input)
        brainStorming_01()
    return selected_prompt_input

def brainStorming_02(selected_pmt_input):

    print("processing...")
    print("     ")
    # Question 2
    print("")
    print("Question 2")
    print("What do you want to tell the reader as the story starts? Who comes out? What happens in the beginning? (100 words)")
    intro_input = input()
    print("processing...")

    # 입력문장 합치기
    input_pmt = selected_pmt_input + ". " + intro_input

    try:
        # 4가지 타입 문장 생성 1
        pmt_re = GenText(input_pmt)
    except:
        pmt_re = GenText(input_pmt)
        ExceptionInput(pmt_re)
        brainStorming_02(selected_pmt_input)

    gen_n1 = pmt_re[0]
    gen_n2 = pmt_re[1]
    gen_n3 = pmt_re[2]
    gen_n4 = pmt_re[3]

    print("-----------------")
    print('Script 1: ', gen_n1)
    print("")
    print('Script 2: ', gen_n2)
    print("")
    print('Script 3: ', gen_n3)
    print("")
    print('Script 4: ', gen_n4)
    print("-----------------")

    scripts = {}
    scripts['Script 1'] = gen_n1
    scripts['Script 2'] = gen_n2
    scripts['Script 3'] = gen_n3
    scripts['Script 4'] = gen_n4

    # Let user select a script
    selected_script = selectFromDict(scripts, 'Script')

    return intro_input, selected_script


def brainStorming_03(selected_script_input):

    print("-----------------")
    # Question 3
    print("")
    print("Question 3")
    print("Did you get some inspirations? So what happens next? How does your story develop? Anyone or something new joins the story? Any issues brewing? (150 words max)")
    body_input = input()
    print("processing...")

    # 입력 + AI 생성 + 입력 
    # input_body_AI = input_pmt + selected_script + body_input
    input_body_AI = selected_script_input + body_input

    try:
        # 4가지 타입 문장 생성 2
        body_re = GenText(input_body_AI)
    except:
        body_re = GenText(input_body_AI)
        ExceptionInput(body_re)
        brainStorming_03(selected_script_input)

    no1_body = body_re[0]
    no2_body = body_re[1]
    no3_body = body_re[2]
    no4_body = body_re[3]

    print("-----------------")
    print('Script 1: ', no1_body)
    print("")
    print('Script 2: ', no2_body)
    print("")
    print('Script 3: ', no3_body)
    print("")
    print('Script 4: ', no4_body)
    print("-----------------")

    scripts_body = {}
    scripts_body['Script 1'] = no1_body
    scripts_body['Script 2'] = no2_body
    scripts_body['Script 3'] = no3_body
    scripts_body['Script 4'] = no4_body

    # Let user select a script
    selected_body_script = selectFromDict(scripts_body, 'script_body')
    return input_body_AI, selected_body_script

def brainStorming_04(input_body_AI_input, selected_body_script_input):

    print("-----------------")
    # Question 4
    print("")
    print("Question 4")
    print("Did you get some inspirations? So what happens at the height of your story? Is there a rising action? An important moment?(150 words max)")
    climax_input = input()
    print("processing...")

    try:
        # 4가지 타입 문장 생성 3
        climax_re = GenText(climax_input)
    except:
        climax_re = GenText(climax_input)
        ExceptionInput(climax_re)
        brainStorming_04(input_body_AI_input, selected_body_script_input)

    no1_climax = climax_re[0]
    no2_climax = climax_re[1]
    no3_climax = climax_re[2]
    no4_climax = climax_re[3]

    print("-----------------")
    print('Script 1: ', no1_climax)
    print("")
    print('Script 2: ', no2_climax)
    print("")
    print('Script 3: ', no3_climax)
    print("")
    print('Script 4: ', no4_climax)
    print("-----------------")

    scripts_climax = {}
    scripts_climax['Script 1'] = no1_climax
    scripts_climax['Script 2'] = no2_climax
    scripts_climax['Script 3'] = no3_climax
    scripts_climax['Script 4'] = no4_climax

    # Let user select a script
    selected_climax_script = selectFromDict(scripts_climax, 'scripts_climax')

    input_body_climax = input_body_AI_input + selected_body_script_input + climax_input + selected_climax_script

    return input_body_climax


def brainStorming_05(intro_input, body_climax_input):
    # Question 5
    print("")
    print("Question 5")
    print("Did you get new ideas? How does the story end and why was it meaningful to you? (150 words max)")
    Conclusion_Lesson_input = input()


    # 최종 문장입력

    finalSentInput = body_climax_input + Conclusion_Lesson_input

    try:
        # 4가지 타입 문장 생성 3
        concluLesson = GenText(finalSentInput)
    except:
        concluLesson = GenText(finalSentInput)
        ExceptionInput(concluLesson)
        brainStorming_05(intro_input, body_climax_input)
        
    re_01 = concluLesson[0]
    re_02 = concluLesson[1]
    re_03 = concluLesson[2]
    re_04 = concluLesson[3]

    print("-----------------")
    print('Script 1: ', re_01)
    print("")
    print('Script 2: ', re_02)
    print("")
    print('Script 3: ', re_03)
    print("")
    print('Script 4: ', re_04)
    print("-----------------")

    scripts_Final= {}
    scripts_Final['Script 1'] = re_01
    scripts_Final['Script 2'] = re_02
    scripts_Final['Script 3'] = re_03
    scripts_Final['Script 4'] = re_04

    # Let user select a script
    selected_final_script = selectFromDict(scripts_Final, 'scripts_Final')

    finalEssay = intro_input + body_climax_input + Conclusion_Lesson_input + selected_final_script

    print("processing...")

    return finalEssay

def brainStorming_Run():
    re_01 = brainStorming_01()
    re_02 = brainStorming_02(re_01)
    re_03 = brainStorming_03(re_02[1])
    re_04 = brainStorming_04(re_03[0], re_03[1])
    f_result = brainStorming_05(re_02[0], re_04)
    return f_result
# run
AIGenEssay = brainStorming_Run()



print("processing...")
print( "")
print("Draft essay : ", AIGenEssay)



##############

# Question 1
# What are your trying to write about among the following? Backgound / Identy / Interest / Talent4

# Q 2 : What do you want to tell the reader as the story starts? Who comes out? What happens in the beginning? (100 words)
# My father is a martial artist and he expects a lot from me.

# Q 3 : Did you get some inspirations? So what happens next? How does your story develop? Anyone or something new joins the story? Any issues brewing? (150 words max)

# Q 4 : Did you get some inspirations? So what happens at the height of your story? Is there a rising action? An important moment?(150 words max)
# You should have seen the other guy.I kicked him back and he started backing off.

# Q 5 : Did you get new ideas? How does the story end and why was it meaningful to you? (150 words max)
# I won the medal and my father began to recognize me.
