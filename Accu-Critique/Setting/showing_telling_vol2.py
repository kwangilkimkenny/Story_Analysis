# Showing & Telling 간단 계산법
# 1) Strong / Descriptive 단어가 나올 때마다 해당 “문장”에 1점 추가
# (e.g. 한 문장에서 strong단어 3개 나오면 일단 3점을 줌)

# 2) Telling / Common 단어가 나올 때 마다 해당 문장에 1점 감점
# (e.g. 이전 문장이 3점인데, telling단어가 2개 나오면 다시 1점이 됨)

# 3) Showing & Telling % 매기기
# -최종점수 2점 이상 나온 모든 문장들의 단어수 = Showing 단어수 %
# -기타 모든 문장들 단어수 (-점수, 0점, 1점 포함) = Telling 단어수 % (에세이 전체단어수 – showing 단어수 임)


import nltk
import re
import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

# Telling / Common 단어들 -> 아래 단어들이 있으면 1점씩 감점을 합니다.    ------------- -1
telling_common_words = ['need','be','right','important','contain','anxious','delicious','call','use','want','nice','owe','take','rich','able','disgust','resemble','tell','lack','include','first','realize','small','look','sorry','try','taste','excited','few','good','disagree','surprise','funny','big','love','long','attractive','relieve','get','public','different','suppose','mind','better','find','sadness','impress','sound','bring','angry','give','own','matter','see','wish','recognize','feel','appreciate','ask','afraid','next','really','satisfy','proud','involve','young','loathe','do','hear','like','mean','last','deny','interesting','joy','think','go','make','appear','doubt','believe','remember','little','understand','amazing','have','imagine','choose','fit','bad','very','work','frustrated','equal','same','fast','high','belong','astonish','happy','kind','old','early','terrible','wonder','large','wrong','sad','adore','seem','true','say','consist','come','anger','fear','know','smell','other','new','hate','concern','cost','posse','dislike','great','agree']

# Descriptive / Showing words (descriptive words)를 모두 합쳐서 활용해 보시면 어떨까 합니다.   -----  +1
showing_descriptive_words = ['harmful','jubilant','vivacious','cyan','precious','concrete','lonesome','sickening','chronic','emotional','sassy','amused','enchanting','bail','zigzag','composed','crumble','efficacious','brooding','perfect','modern','jeering','uptight','zealously','minor','organic','exasperating','methodically','kind','assertively','plateau','sanguine','nauseous','amusingly','clever','livid','professional','kissable','wacky','blissfully','kooky','harmonious','threatening','cadge','warmhearted','babyish','outsize','peculiar','brash','enthusiastic','safe','elegant','selfless','yummy','lavishly','listlessly','melancholy','meagerly','beloved','intricate','delicacy','scintillating','lively','odorous','sedentary','useful','chipper','overprotective','quenched','active','vaulting','coil','immortal','compress','haunt','headstrong','wicked','intentional','elegant','macabre','machiavellian','mammoth','basic','admire','official','jaunty','flustered','queer','outgoing','abrupt','querulous','clean','abruptly','uninterested','sparkling','original','lavishly','fantastical','fester','alert','bluish','free','giant','misty','garner','marvel','delighted','dreaming','isolated','oblivious','immense','sweeping','irritating','frightened','teasingly','nauseated','jostle','abundance','unbelievable','pursuits','quirky','out-of-this-world','quirky','youthfully','eagerly','gleaming','reachable','lawful','kingly','zealful','enraged','feign','out-of-place','questionable','efficient','jangling','rabid','acclaimed','midday','courageous','cloudy','vociferous','hastily','abundant','amazing','fine','opulence','focal','carefree','thoughtless','precious','grieving','bewildered','repulsive','affordable','evanescent','accomplish','lazily','beseeching','exasperation','valorous','cavernous','frivolously','marvellous','condemned','tender','intertwine','angrily','yellow','vast','needy','posied','bitter','bona','vacantly','commissary','dazzlingly','kinetic','xeme','loom','tolerant','menhir','wide-awake','ever-deepening','conclude','dilapidated','creation','jealous','legendary','interior','zazzy','bow','satisfied','fide','meritorious','victorious','elusive','polished','amplify','abusive','intriguing','dredge','mysterious','amused','confounded','intolerable','introverted','racy','zippy','lesser','charming','nice','xenogeneic','tall','lucky','brood','playful','lazy','infuriate','boho','lovable','self-assertive','participant','intense','plea','crenulated','xystus','luxurious','buzz','disheveled','cheerful','seriously','buzzing','bolt','deviate','communal','intrepid','dedicated','thunderous','efface','elegance','balanced','daunt','warm','evidently','funerary','casual','pizzazz','generous','ornate','irksome','paradox','shocking','depressed','handed','qualified','relieved','environs','terrible','inviting','facade','ambition','host','cascade','pleasantly','passionately','perch','linger','colorful','kaleidoscopic','perceptive','embarrassing','cairn','obedient','buoyant','fancy','scrutiny','bleak','motivational','heave','entice','outstanding','comfortable','delirious','fabulous','raspy','dutifully','academic','dazzling','determined','easy-going','validatory','fun-loving','customized','herbaceous','knowledgeable','splendid','inexpensive','irk','quick','naive','narcissistic','inter','impossible','belie','yeasty','pompous','grumpy','effectively','worrisome','careful','ostracize','tenacious','tactfully','worshipful','fervent','aficionado','depressed','riveting','wordy','swiftly','elaborate','alternately','modern-day','grim','decent','delve','ideal','sensitive','utter','radiant','whole','jesting','steeped','idolized','relaxing','hazel','intelligent','openhearted','bright','peaceful','weirdly','obnoxious','helpful','open-minded','motivated','keen','keeled','questioning','metropolitan','attest','romantically','lichen','vigorous','xanthospermous','rapidly','deep','babyish','brainy','thankful','mesmerize','eclectic','obscure','sophisticated','secretly','albeit','curt','gurgle','pristine','nervous','memorable','interweave','point','xerothermic','deafeningly','bleeding','coarse','greedily','flit','zootrophic','vainglorious','kindred','icy','dyed','ugly','mysterious','gifted','impulsively','sarcastic','distressing','valuable','zesty','expertly','grimly','tattered','stuffed','yellowish','eager','lame','passable','distaste','ludicrously','hasten','drift','monumental','ordinary','probe','coyly','baggy','gaping','healthy','radioactive','faux','bright','creative','canopy','endow','thorny','uber','vivid','jeweled','apathetically','arose','heavenly','emerge','fake','keen','hearty','sly','witty','rashly','designate','good-looking','innate','beaming','gust','conifer','unusual','lying','tempt','boisterous','loud','quotable','high-spirited','ulterior','kindhearted','practical','wrinkled','lunged','amend','beaten','masterfully','intricately','defeated','filch','tragically','immune','naughty','limp','babble','embellish','versatile','quarrelsome','strange','brave','humongous','nasty','intersperse','diverse','observant','wild','ooze','trustful','edifice','nestled','zoological','keenly','placid','mutter','hideous','jolly','faithful','depiction','contract','ambitious','powerless','immaculate','baggy','decorated','grandeur','normally','jagged','stunning','intensely','effortless','exemplary','xenial','imbue','expression','gregarious','ultimate','yiddish','outrageous','depth','bookish','exhibition','new','frustrating','yearly','indication','cerulean','pivotal','different','fixated','peppered','baffling','zaftig','xebec','zany','monumentally','patently','yawning','valiant','inclined','disturbing','jaunty','exuberant','quaint','delicious','lazy','malleable','personable','bust','fascinating','eye-catching','buttery','vividly','pugnacious','additionally','natural','talented','talkative','knockout','jarring','devour','luxurious','divergent','nab','narrow','tremendous','pretty','real','lyrical','commitment','panicky','pas','listless','fierce','generic','tempting','mad','improper','chillingly','quiver','gloomy','devouring','equable','neat','intentionally','absorption','broken','feisty','interesting','peacefully','adorable','well-behaved','adorable','begrudgingly','handsome','batter','infuriatingly','necessary','unaffected','radical','licit','limpid','endless','instance','damaged','kindly','old-fashioned','respectable','dynamic','klutzy','congenital','glorious','inquisitive','quiet','beset','rigorous','brittle','crystallized','beneficial','colonnades','calculating','bash','vibrant','petty','behemoth','achy','cautious','humiliating','blue-eyed','fast-moving','tumult','monolith','kaput','jaded','delectable','graceful','bouncy','unbroken','chosen','lonely','xylographic','crackle','myriad','abnormal','radiant','burly','maladapted','xenodochial','glowing','haunch','weathered','lanky','brilliant','juvenile','miserable','deep','dominant','native','keen','crinkle','indelible','rigorously','foreign','xeric','zymotic','musical','hilarious','lascivious','gentle','wealthy','thrilled','launderette','abandoned','eminence','eponymous','ferocious','expansive','foolhardy','tired','dumbfound','erect','lay-by','rustic','arrogant','obtuse','alternate','brilliant','calm','magnanimous','inviting','divine','worried','alter','pointlessly','rakish','lavish','innately','puzzling','unused','pigpen','pariah','engagement','creaky','immensely','unpleasant','emphatic','yodelling','scented','queasy','jam-packed','flawless','opulent','tantalizing','zenithal','exciting','passionate','acclaimed','motionless','nauseating','sprawl','romantic','impulsive','heroic','liberated','gusto','homely','basalt','glamorous','persistent','nocturnal','xanthous','effective','infuriating','plush','factual','jumpy','invitingly','quixotic','immense','questionable','verdant','peaceful','exuberant','boundless','exciting','famous','valid','knotted','patient','confusing','goal-oriented','plunge','intense','dash','wrong','quickly','trustworthy','youthful','mediocre','indelibly','accomplished','neglectfully','exhaustive','quality','silky','yern','rational','karst','plod','hasty','dishevel','earthly','meddlesome','carefree','decorative','salty','complicated','adaptable','battered','rare','young','aimless','boulder','wasteful','extravagant','impulse','malefic','jubilant','overwhelmed','limpidly','universal','ornate','illegal','vigilant','guarded','dismay','beautified','powerful','kindly','battered','chortle','indelible','acrobatic','darkly','idyllic','galore','grim','brilliant','bewitched','exasperate','shriek','boring','handsome','melted','blithe','fashionable','acceptable','furthermore','up-front','tinted','adventurous','bewilder','cresting','guffaw','bewildered','captivating','zestful','keyless','iconic','combative','zealand','unlucky','dainty','zingy','lanky','intricate','dark','mindful','magical','zealous','panic','beauties','dangerous','impervious','jittery','proud','offensively','handy','frescoed','tiling','zionist','colossal','fascinated','gruesome','driven','accomplished','clochán','crisp','charismatic','odd','responsible','young-at-heart','discreet','rampant','peek','conventional','flushed','hoist','loam','intelligently','scared','festive','blushing','innate','grassy','ludicrous','yappy','idealistic','empty','jarring','mellow','adventurous','deluxe','jagged','harrowing','sanctimonious','boost','fearless','brave','faintly','weak','glum']


# List of Strong Verbs (쓰면 좋은 단어들… 많이 있으면 showing 가산점) ------------------ +1
# ***겹치는 단어들 있습니다!!! VERB로만 썼으면 합니다. noun으로 적용 안했으면…
list_of_strong_verb = ['bloom','plant','tiptoe','recoil','seep','saunter','sparkle','breeze','gravitate','drop','snarl','pop','muddle','whirl','scan','club','sway','cease','wrestle','shovel','squeal','trigger','discern','flutter','maul','plod','hover','hit','kidnap','embrace','dismantle','notify','woof','boom','collapse','shuffle','trim','scrawl','snowball','ignite','dash','unfold','drool','glimmer','stare','flit','plummet','skim','unveil','elbow','mimic','extract','sputter','skulk','envelop','sashay','obtain','shimmer','toddle','breathe','deactivate','eyeball','usher','float','titillate','produce','tap','ensnare','shoot','position','flourish','grope','scape','fume','savor','squish','roll','slobber','play','amplify','locate','wring','bump','pause','picture','jig','electrify','compile','clarify','oppress','clash','slap','uncover','sizzle','bolt','build','drip','sip','attack','mushroom','yodel','glimpse','crush','whip','ramble','fly','sink','fling','starve','stretch','hustle','soar','terminate','chip','travel','crave','place','suck','scrape','grasp','prickle','rattle','view','hail','scamper','retreat','enfold','bolt','rub','scarf','withdraw','embolden','cradle','discover','fuse','assemble','cheer','skid','catch','survey','beam','clasp','pocket','remark','consume','swoop','slave','dilly-dally','storm','gallivant','inspect','gape','swig','alter','bellow','wander','shatter','tattle','lash','stroke','curl','gaze','sail','march','bleat','squint','lift','devour','duck','loiter','click-clack','strike','enlarge','slash','fondle','mint','shriek','snack','create','snatch','destroy','wreck','notice','pale','glance','voice','creak','depart','fight','snitch','shower','crank','smash','stroll','illuminate','clog','scrutinize','sabotage','zoom','jolt','declare','ooze','rip','jive','form','polish','puncture','shimmer','traipse','stall','seize','fortify','blemish','modify','blab','gush','compose','dazzle','whisper','cremate','smother','brew','propel','burst','shrivel','shush','demolish','note','babble','brake','gaze','speed','fly','enliven','stomp','hiss','revolve','mention','bark','nestle','shock','throb','poison','impart','howl','kiss','download','incinate','block','study','explode','tickle','multiply','douse','ignite','grasp','brief','power','expose','slump','snarl','nail','drench','penetrate','pet','trail','yelp','catapult','collide','interrupt','fracture','smack','shine','guffaw','flog','scold','growl','whoop','trash','add','trudge','stare','torpedo','drizzle','pilot','intensify','storm','forge','emit','detect','lurch','meow','blubber','swell','confide','scorch','wrench','drag','steal','brust','comment','stumble','probe','wipe','sob','utter','roar','thrive','clank','fragment','thumb','leap','snap','yak','govern','grab','gobble','direct','dig','bleed','poke','huddle','dangle','squeeze','surge','weave','deposit','seize','flood','blast','squall','strangle','scratch','throttle','chap','chew','scamper','plop','raise','swing','transform','grin','troop','wreck','splash','putter','shock','roam','muzzle','state','spray','lead','nose-dive','zing','hobble','launch','galvanize','mystify','chortle','garble','swipe','pilfer','steal','hiccup','bite','hypnotize','zap','shake','jingle','hammer','prance','construct','survey','bat','arouse','crush','sprinkle','brighten','hush','brush','torch','surge','formulate','spring','gobble','flip-flop','eavesdrop','grip','gallop','pluck','amble','skip','stammer','leak','guzzle','pulsate','dance','labor','plop','bolster','engulf','trudge','ruin','dash','scream','ingest','trip','trip','chirp','climb','caper','murmur','finish','devour','ripple','clutch','explode','spout','massage','clap','snicker','jostle','capsize','combust','twitter','slash','dive','bust','sizzle','wobble','park','snuggle','drift','dawdle','growl','croon','unclog','boost','stumble','pluck','model','perceive','muse','whack','choke','gag','tear','nab','holler','turn','advance','wolf','trot','dart','absorb','tango','trickle','devastate','peek','nag','rumble','sweep','spurt','steer','intertwine','mumble','batter','guide','bash','swoon','stab','hose','spin','swirl','patrol','serve','coo','wrestle','relieve','applaud','hearten','blossom','grope','beam','mutter','moan','pat','sneak','capture','quaff','hook','clutch','quiver','slurp','smile','speak','snag','halt','feast','drain','eye','commune','order','bound','lurk','flavor','refine','lutch','reverberate','tinkle','snag','glare','snake','supercharge','moan','invoke','mushroom','skip','goggle','brood','fire','chomp','disentangle','gush','command','journey','squeak','charge','scorch','investigate','hop','pop','expand','engage','wade','cascade','groan','glare','skedaddle','regurgitate','animate','tread','glide','chuckle','glitter','rush','cower','slink','munch','shepherd','barrel','ransack','peer','dip','ruin','drum','burst','wail','drip','paint','nibble','strut','scramble','hoot','gleam','twist','yank','stutter','swallow','smooth','struggle','retreat','sprinkle','split','burp','frown','crash','mosey','agitate','honk','fish','kick','rush','gnaw','reverberate','supersize','groan','conclude','bathe','thunder','hurry','nip','crash','sprint','tumble','hike','handle','broadcast','hug','foster','hack','grip','tee-hee','weave','devise','snort','soar','resonate','dump','chug','balloon','saunter','splinter','loot','chatter','tussle','stroll','slosh','yell','vanish','liberate','tremble','shrill','strip','recite','puke','purr','manufacture','watch','fashion','pinpoint','plough','click','prattle','vibrate','treat','announce','rust','stride','energize','charge','whirr','extend','pour','smite','screech','tantalize','cling','crack','plunder','amend','snowball','stream','take','rise','grumble','buckle','bounce','beat','flick','peck','linger','gargle','answer','whiz','deviate','shout','fizz','instruct','ditch','tackle','observe','pussyfoot','tip-tap','wail','beef','climb','sparkle','invigorate','magnify','stick','suffocate','slip','mouth','button','intertwine','caress','hijack','shatter','remove','finger','berate','bust','wrest','untangle','swoosh','advise','spellbind','unshackle','slide','end','crackle','cuddle','drain','shoulder','inhale','leap','nosh','smash','meander','revitalize','race','clasp','plunge','freeze','release','scan','slide','refashion','envelop','prod','knock','spark','render','conjure','skyrocket','graze','sing','fret','gulp','jump-start','gawk','snigger','vocalize','giggle','boost','plunge','cry','spam','trek','bruise','decimate','heighten','reveal','tail','drop','pinch','escort','glisten','leer','feed','peek','revolutionize','peep','prune','kick-start','zip','sprout','explore','draft','jangle','vomit','wind','demolish','veil','erase','scurry','kindle','topple','blend','examine','unearth','spit','waltz','inspect','strain','slurp','dine','pronounce','peer','shine','pierce','bang','bus','frolic','slump','report','buzz','crunch','stir','glow','treasure','slog','hum','croak','hurry','sharpen','swipe','run','thrill','titter','quit','spill','lurch','transfigure','cackle','schelp','cripple','sport','rotten','devise','glimpse','avuncular','naughty','absorbing','beseech','savor','surmise','query','hopeful','crucial','create','merry','unimaginable','prodigious','sympathetic','phenomenal','well-heeled','brilliant','agile','fetch','courteous','inquire','cheerful','comical','sidesplitting','marvelous','slight','designate','awful','examine','wee','kind','lug','skimpy','overwhelming','wretched','huge','primary','shaky','build','well-to-do','generous','positive','nominate','presume','disgraceful','compelling','despicable','enthusiastic','humorous','select','minute','hysterical','joyful','pick','carry','timid','scrumptious','admire','honestly','droll','extract','whimsical','crappy','bequeath','administer','outrageous','lively','engaging','reckon','gather','gracious','laughable','obtain','solicit','bear','optimistic','thoughtful','benevolent','incredible','encounter','brew','glamorous','elect','absolutely','active','amiable','astonishing','ecstatic','genuinely','usher','fancy','remit','appoint','astounding','award','vital','detect','wicked','lousy','dispense','enthralling','prepare','remarkable','juicy','extraordinary','fabulous','flavorful','locate','riveting','critical','blissful','immense','respect','friendly','surely','deep-pocketed','colossal','concoct','splendid','grant','highly','compassionate','contented','admirable','moneyed','tiny','elated','cat','ascertain','congenial','overjoyed','quick','adopt','horrible','summon','fascinating','momentous','terrified','significant','sense','transport','gripping','decidedly','large','pleasurable','entertaining','brisk','acquire','charmed','gleeful','shell-shocked','giant','luscious','striking','actually','positively','exceedingly','gigantic','petition','determine','delighted','tremendous','pretty','dote','speedy','veritably','ludicrous','delightful','affluent','delectable','entrust','spot','jocular','frightened','cordial','considerate','implore','panicky','great','crummy','spectacular','suspect','gentle','hefty','money','appreciate','discover','pleasant','well-fixed','mammoth','donate','horrified','chief','neat','miniature','breathtaking','cute','deem','gape','swift','verily','stupendous','recognize','precisely','witness','warm-hearted','extremely','invent','fundamental','behold','chicken','excellent','handsome','opulent','captivating','full','attentive','tote','diminutive','honor','prosperous','nonsensical','horror','tremendously','allow','appetizing','haul','cast','bestow','massive','amusing','fearful','towering','flush','urge','farcical','cherish','horror-struck','assume','well-off','diverting','benign','thrilled','deliver','exceptional','eager','tense','essential','delish','forge','fantastic','pleased','particularly','adore','caring','unearth','engrossing','nimble','dreadful','bright','pin','goofy','mouth-watering','observe','fat','outstanding','petite','procure','witty','construct','pleasing','mini','disagreeable','gorgeous','marvelous','staggering','assemble','super',
                       'accelerated','enormous','stunning','beautiful','exultant','intriguing','certainly','foresee','kindhearted','hilarious','teeny-weeny','terrific','charming','paramount','alluring','pygmy','pinpoint']
    


# 점수계산용 단어 리스트

# showing : plus_wds
plus_wds = telling_common_words

# telling : minus_wds
minus_wds = showing_descriptive_words + list_of_strong_verb


#데이터 전처리 
def cleaning(datas):
    
    # 영문자 이외 문자는 공백으로 변환
    only_english = re.sub('[^a-zA-Z, .]', ' ', datas)
    only_english = re.sub('\n', '', only_english)
    
    return only_english


#문장을 단어로 변환
def SentToTokenize(text):
        essay_input_corpus = str(text) #문장입력
        essay_input_corpus_lower = essay_input_corpus.lower()#소문자 변환

        # 문장으로 토큰화
        sentences  = sent_tokenize(essay_input_corpus_lower)
        
        # 단어로 토큰화
        words = word_tokenize(essay_input_corpus_lower)
        
        return sentences, words

#[문장, 단어]들로 한 묶음씩 리스트로 변환
def Group_sent_words(text):
        essay_input_corpus = str(text) #문장입력
        essay_input_corpus = cleaning(essay_input_corpus) # 영어, 특수문자 처리
        essay_input_corpus_lower = essay_input_corpus.lower()#소문자 변환

        # 문장으로 토큰화
        sentences  = sent_tokenize(essay_input_corpus_lower)
        
        # 입력 총 문장 수
        sentence_total_cnt = len(sentences)
        
        Grp = []
        
        for i in sentences:
            Group_sent_wd_list = []
            # 개별 문장을 단어로 토큰화
            words = word_tokenize(i)
            
            # 문장 리스트에 추가
            Group_sent_wd_list.append(i)
            
            # 토큰화된 단어들 리스트에 추가
            Group_sent_wd_list.append(words)
            
            # 문장 + 단어들 그룹으로 리스트에 추가
            Grp.append(Group_sent_wd_list)
        
        # Grp : [문장, 단어]들로 한 묶음씩 리스트로 변환
        # sentence_total_cnt : 입력한 에세이의 총 문장 수
        return Grp, sentence_total_cnt




# plus_wds = telling_common_words
# minus_wds = showing_descriptive_words + list_of_strong_verb

# 점수계산용 리스트를 가지고, 분석 문장을 가져와서 해당 단어가 있으면  + - 점수 계산
# input_gre_re : [['my hand lingered on the cold metal doorknob.',['my', 'hand', 'lingered', 'on', 'the', 'cold', 'metal', 'doorknob', '.']], ...




# total_sent_nums : 입력에세이의 총 문장 수
def ShowTellClassfication(input_gre_re, total_sent_nums):
    showtell_cal = 0
    showing_sents = []
    telling_sents = []
     
    for cal in input_gre_re:
        for pwd in plus_wds:
            if pwd in cal[1]:
                showtell_cal += 1
                # showinG 단어가 발견되어 해당 문장 추출
                showing_sents.append(cal[0])


        for mwd in minus_wds:
            if mwd in cal[1]:
                showtell_cal -= 1
                # telling 단어가 발견되어 해당 문장 추출
                telling_sents.append(cal[0])

    # shotell_cal : 전체 문장에서 Showing vs. telling 사용 비율 카운트, + - 계산하한 최종 값
    # showing_sents : Showing 문장
    # showing_sents_set : 중복제거
    showing_sents_set = list(set(showing_sents))
    show_sent_count = len(showing_sents_set)
    # telling_sents : Telling 문장
    # telling_sents_set : 중복제거
    telling_sents_set = list(set(telling_sents))
    telling_snet_count = len(telling_sents_set)
    
    # Showing ratio : show_sent_count / input_gre_re(전체문장수)
    showingRatio =  round(show_sent_count / total_sent_nums * 100)
    
    # Telling ratio : 100 - showingRatio
    tellingRatio = round(100 - showingRatio)
    return showtell_cal, showing_sents_set, show_sent_count ,telling_sents_set, telling_snet_count, showingRatio, tellingRatio



def ShowingTellingCall(input_text):
    grp_re = Group_sent_words(input_text)
    ShowTellClassResult = ShowTellClassfication(grp_re[0],grp_re[1])
    return ShowTellClassResult


############# Test #############

input_text = """My hand lingered on the cold metal doorknob. I closed my eyes as the Vancouver breeze ran its chilling fingers through my hair. The man I was about to meet was infamous for demanding perfection. But the beguiling music that faintly fluttered past the unlatched window’s curtain drew me forward, inviting me to cross the threshold. Stepping into the apartment, under the watchful gaze of an emerald-eyed cat portrait, I entered the sweeping B Major scale.

Led by my intrinsic attraction towards music, coupled with the textured layers erupting the instant my fingers grazed the ivory keys, driving the hammers to shoot vibrations up in the air all around me, I soon fell in love with this new extension of my body and mind. My mom began to notice my aptitude for piano when I began returning home with trophies in my arms. These precious experiences fueled my conviction as a rising musician, but despite my confidence, I felt like something was missing.

Back in the drafty apartment, I smiled nervously and walked towards the piano from which the music emanated. Ian Parker, my new piano teacher, eyes-closed and dressed in black glided his hands effortlessly across the keys. I stood beside a leather chair, waiting as he finished the phrase. He stood up. I sat down.

Chopin Black Key Etude — a piece I knew so well I could play it eyes-closed. I took a breath and positioned my right hand in a G-flat 2nd inversion. 
Just one measure in, I was stopped. 
“Start again.”
Taken by surprise, I spun left. His eyes were on the score, not me. 
I started again. Past the first measure, first phrase, then stopped again. What is going on? 

“Are you listening?”
I nodded. Of course I am. 
“But are you really listening?”

As we slowly dissected each measure, I felt my confidence slip away. The piece was being chipped into fragments. Unlike my previous teachers, who listened to a full performance before giving critical feedback, Ian stopped me every five seconds. One hour later, we only got through half a page. 

Each consecutive week, the same thing happened. I struggled to meet his expectations. 
“I’m not here to teach you just how to play. I’m here to teach you how to listen.” 
I realized what Ian meant — listening involves taking what we hear and asking: is this the sound I want? What story am I telling through my interpretation? 

Absorbed in the music, I allowed my instincts and muscle memory to take over, flying past the broken tritones or neapolitan chords. But even if I was playing the right notes, it didn’t matter. Becoming immersed in the cascading arpeggio waterfalls, thundering basses, and fairydust trills was actually the easy part, which brought me joy and fueled my love for music in the first place. However, music is not just about me. True artists perform for their audience, and to bring them the same joy, to turn playing into magic-making, they must listen as the audience. 

The lesson Ian taught me echoes beyond practice rooms and concert halls. I’ve learned to listen as I explore the hidden dialogue between voices, to pauses and silence, equally as powerful as words. Listening is performing as a soloist backed up by an orchestra. Listening is calmly responding during heated debates and being the last to speak in a SPS Harkness discussion. It’s even bouncing jokes around the dining table with family. I’ve grown to envision how my voice will impact the stories of those listening to me.

To this day, my lessons with Ian continue to be tough, consisting of 80% discussion and 20% playing. When we were both so immersed in the music that I managed to get to the end of the piece before he looked up to say, “Bravo.” Now, even when I practice piano alone, I repeat my refrain: Are you listening?  """


st_result = ShowingTellingCall(input_text)

print(st_result) 