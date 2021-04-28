import re
from difflib import SequenceMatcher


def SummerActivity(essay_input):

    sentences = """RSI (Research Science Institute) at MIT	5
MIT Women's Technology Program (WTP)	5
MITES (Minority Introduction to Engineering and Sciences)	5
LAUNCH at MIT	5
SSP (Summer Science Program)	5
Garcia MRSEC Summer Research Program	5
SIMONS Summer Research Program	4
High School Summer Science Research Program(HSSSRP) at Baylor U	4
CTY Civic  Leadership Institute Program	4
CTY Global Issues in the 21th century	4
Brown Leadership Institute	3
Brown TheaterBridge	3
Brown Intensive English Program (IEP)	3
Brown Environmental Leadership Lab (BELL)	3
CTD (Northwestern)	3
NASA Women in STEM Program	5
NASA Goddard Space Flight Center Internship	5
NASA National Space Club Scholars Program	5
NASA INSPIRE Program	5
TASP (Telluride Association Summer Program)	5
TASS (Telluride Association Sophomore Seminar)	5
The ROSS Mathematics Program	5
PROMYS (Program in Mathematics for Young Scientists)	5
MathWorks Honors Summer Math Camp (HSMC)	4
MOSP (Mathematics Olympiad Summer Program)	5
Governor's School of New Jersey	3
HSHSP (Michigan State University  High school Honors Sciences Program)	4
The Physics of Atomic Nuclei Program at NSCL (National Superconducting Cyclotron Laboratory)	5
Stanford Institute of Medicine Summer Research	4
SUMaC (Stanford U Mathematics Camp)	5
HCSSiM (Hampshire College Summer Studies in Math)	5
Summer Internship Program in Neurological Science National Institute of Health (NIH)	5
SIP (Summer Internship Program in Biomedical Research)	3
Link Summer Science Explorations at Kopernik Observatory	3
Barnard College Saturday Science Seminars (S-Cubed)	3
NYU Tisch School of the Arts High School Program	4
NYU Summer at Steinhardt School	3
New Jersey Scholars Program	3
LEAD Business (Leadership Education and Development)	3
LEAD Engineering (Leadership Education and Development)	3
LEAD Computer Science (Leadership Education and Development)	3
Bronfman Youth Fellowship in Israel	3
Jackson Laboratory Summer Student Program	3
Davidson Institute Young Scholars Stars Summer	3
Business Leadership Program	3
Summer at Walnut Hill	4
CMU SAMS (The Summer Academy for Math + Science)	3
Architecture Pre-College	3
Carnegie Mellon Drama Pre-College Program	3
Carnegie Mellon Art & Design Pre-College	3
Carnegie Mellon Music Pre-College	3
Carnegie Mellon National High School Game Academy	3
Kenyon College The Young Writers Workshop	3
UPenn SAAST (Summer Academy in Applied Science and Technology)	3
U Penn VETS	3
U Penn Wharton Sports Business Academy	4
U Penn LBW (Leadership in Business)	5
U Penn-Nursing Summer Institute	3
U Penn M&TSI Management & Technology Summer Institute	5
U Penn Art & Architecture Summer Program	4
U Penn TREES Teen Research and Education in Environmental Science	3
U Penn Biomedical Research Academy	4
U Penn Chemistry Research Academy	3
U Penn Experimental Physics Research Academy	3
U Penn Social Justice Research Academy	4
U Penn Art in the City Academy	3
U Penn Startalk Urdo, Hindi, and Chinese Academy	3
U Penn Law Summer Academy	3
U Penn Summer BOOT UP Camp	3
Cornell U Research Apprenticeship in Biological Sciences (RABS)	5
Harvard Debate Council Summer Programs	4
SSEP (Summer Science & Engineering Program) at Smith College	3
Yale Young Global Scholars	4
Yale English for High School Students	3
Summer Journalism Program Princeton University	4
Stanford Summer Humanities Institute	4
Northwestern U Medill Cherub Journalism Program	5
(AMSP) AwesomeMath Summer Program	3
Canada/USA Math Camp	4
iD Tech Summer Computer Camp Programs	3
UC San Diego Academic Connections Research Scholars	3
JSA Junior Statesmen of America Summer School	4
VAMPY Summer Program for Verbally and Pathematically Precocious Youths	3
Startalk Chinese at Brigham Young U	3
Startalk Arabic Brigham Young U	3
Startalk ChineseU of Mississippi	3
Concordia Summer Villages	3
The Great Books Summer Program	3
ISSYP (International Summer School for Young Physicists)	5
RISE (Research Internship in Science and Engineering) Boston U	4
Boston U High School Honors	3
Boston U High School Seminars	3
National Scholars Institute (NSI) U of Iowa	4
Secondary Student Training Program (SSTP) at U of Iowa	4
High School BLIPS Berkeley Lab Internships for Precollegiate Students	3
COSMOS California State Summer School for Mathematics and Science	4
(UF-SSTP) Student Science Training Program U of Florida	3
Young Scholars Program UC Davis	3
MMSS (Michigan Math and Science Scholars)	4
Dartmouth Debate Workshop	4
Dartmouth Debate Institute	4
The Health Careers Institute at Dartmouth	3
University of Texas El Paso High School Law Camp	3
NBA Crump Law Camp	3
(GLW) Girls' Leadership Worldwide, Elenor Roosevelt Center at Val-Kill	3
WashU High School Summer Scholars	3
WashU Architecture Discovery Program	3
WashU Portfolio Plus Program	3
UIUC Discover Architecture Pre-College Program	3
Exeter Foreign Language Summer Programs	3
Phillips Andover MS2 Program ( Math and Science for Minority Students)	3
Choate Rosemary Hall Summer Session for High School Students	3
Hotchkiss School Summer Leadership and Social Change Program	3
Classical Music Summer Program at Curtis Institute	3
Summer Internship for High School Students	3
Indiana U High School Journalism Institute	3
National Youth Science Camp	3
RISD Pre-College	3
Columbia Scholastic Press Association (CSPA) Summer Workshop	3
Aspen Music Festival School	3
Boston University Tanglewood Institute(BUTI)	5
Boston University Visual Arts Summer Institute	3
Boston University Summer Theater Institute	3
Bowdoin International Music Festival	3
Eastern Music Festival Summer Study	3
Interlochen Summer Arts Camp	5
Greenwood Chamber Music Camp	3
Mathlinks	3
Math Zoom Summer Academy	3
Rutgers Young Scholars Program in Discrete Mathematics	3
Stanford Clinical Anatomy Summer Research Scholars Program	4
Stanford Surgical Anatomy for High School Seniors and Pre-Med students	4
Google Computer Science Summer Institute (CSSI)	5
UCSF Biomedical and Health Sciences Internship for High School Students	3
Iowa Young Writers Studio	5
Sewanee Young Writers' Conference	4
Harvard medical school DCP	4
Tufts University Adventures in Veterinary Medicine Program	4
CNSI NanoScience Lab Program	3
Mathematics Summer Camp at Stony Brook University	3
School of Creative and Performing Arts Summer Camp	3
The School of Cinematic Arts Summer Program at USC	4
BU Summer Investigative Journalism Workshop	3
Johns Hopkins Engineering Innovation	4
Rensselaer Robotics Engineering Academy	4
The Rockefeller University Summer Neuroscience Program	3
U Chicago Research in the Biological Sciences (RIBS)	4
High School Field Schools with ArchaeoSpain	3
Center for American Archaeology High School Field School	3
Summer Journalism@NYU	3
U of Iowa Summer Journalism Workshops	3
Hankuk University of Foreign Studies International Summer Session in Korean & East Asian Studies	3
Barnard College Young Women's Leadership Institute	3
Notre Dame Pre-College Summer Scholars Program	3
Notre Dame Global Leadership Seminars	5
Emerson College Summer Journalism Institute	3
Columbia Epidemiology Summer Online Course	3
Startalk U of Mississippi	3
Middlebury Monterey Language Academy	4
The Metropolitan Museum of Art Summer Internships for High School Students	3
Tufts University European Center Summer Program	3
French Heritage Language Program Summer Camp	3
Emory University Youth Theological Initiative Summer Academy	3
Summer Linguistics Institute for Youth Scholars at Ohio State University	3
National Security Language Initiative for Youth (NSLI-Y)	3
Brandeis University Genesis Summer Program in Social Entrepreneurship	3
Wake Forest School of Medicines Camp Med	3
University of Miami Nursing Pre-Entry Program	3
Pharmacy Summer Camp at University of Houston	3
Cleveland Clinic Pharmacy Internship Program	3
The Kansas University School of Pharmacy Summer Camp	3
Julliard Summer Dance Intensive	4
American Dance Festival Program at Duke University	3
Mpulse Summer Dance Institute at U Michigan	3
UCLA Summer Dance/Performing Arts Summer Institutes	3
New York City Dance Alliance Summer Intensive	3
School of Visual Arts Summer Pre-College	3
Parsons Pre College Academy	3
The Fashion Institute of Technology’s Pre-college Programs	3
MPulse Summer Performing Arts Institutes	3
Peabody Preparatory Annapolis Campus Music Theory Workshop	3
UC Berkeley embARC Summer Design Academy	3
Hospitality Management Summer Program at U of New Hampshire	3
Les Roches International School of Hotel Management Summer Program	3
Journey for Juniors at Culinary Institute of America	3
Chicago Culinary, Hospitality and Business Camps for High School Students at Kendall College	3
Institute On Neuroscience (ION/Teach) at Emory University	3
Introduction to Radar for Student Engineers at MIT Lincoln Laboratory	3
The Institute for Speech and Debate	3
Rosetta Institute Cancer/Neuroscience Summer Camps	3
U Penn Game Design Summer	3"""

    sent_list = sentences.split('\n')

    # 문자열 매칭을 비교할 수 있다.  그렇다면 각 문장을 개별 비교하여 매칭되는 결과중 가장 높은 것이 하나라도 있다면 그것이 바로 일치하는 문자열임
    all_comp = {}
    for itm in sent_list:
        ratio = SequenceMatcher(None, itm, essay_input).ratio()
        ratio_ = ratio * 100
        all_comp[itm] = ratio_

    # 오름차순 정렬
    sorted_data = sorted(all_comp.items(), key=lambda x: x[1], reverse=True)

    # 가장 큰 값 찾기
    # dic_max = max(all_comp.values())

    # 2.2 이상의 값을 모두 찾기(일치확률이 높은 모든 값 추출)
    for std in sorted_data:
        if std[1] > 1.5:
            print('Extracted summer activities expected to match: ', std)
        else:
            pass

    # 조건문으로 2 이상의 결과가 나오면 이것은 summer activities의 내용과 입력한 에세이에 언급한 활동과 일치한다고 보면 됨
    for key, value in all_comp.items():
        if max(all_comp.values()) > 2.2: # 2.2 이상의 결과값 중에서
            if value == max(all_comp.values()): # 결과 중 가장 큰 값이 일치한다면
                detected_summer_activity = key[:-2]
                result = key[-1:]
                break
        else:
            detected_summer_activity = "Not found"
            # 일치하지 않기 때문에 점수가 없음
            result = 0
                
            
    return detected_summer_activity, result


## run ##

essay_input = """This past summer, I had the privilege of participating in the University of Notre Dame’s Research Experience for Undergraduates (REU) program . Under the mentorship of Professor Wendy Bozeman and Professor Georgia Lebedev from the department of Biological Sciences, my goal this summer was to research the effects of cobalt iron oxide cored (CoFe2O3) titanium dioxide (TiO2) nanoparticles as a scaffold for drug delivery, specifically in the delivery of a compound known as curcumin, a flavonoid known for its anti-inflammatory effects. As a high school student trying to find a research opportunity, it was very difficult to find a place that was willing to take me in, but after many months of trying, I sought the help of my high school biology teacher, who used his resources to help me obtain a position in the program.				
Using equipment that a high school student could only dream of using, The Rockefeller University Summer Neuroscience Program, I was able to map apoptosis (programmed cell death) versus necrosis (cell death due to damage) in HeLa cells, a cervical cancer line, after treating them with curcumin-bound nanoparticles. Using flow cytometry to excite each individually suspended cell with a laser, the scattered light from the cells helped to determine which cells were living, had died from apoptosis or had died from necrosis. Using this collected data, it was possible to determine if the curcumin and/or the nanoparticles had played any significant role on the cervical cancer cells. Later, I was able to image cells in 4D through con-focal microscopy. From growing HeLa cells to trying to kill them with different compounds, I was able to gain the hands-on experience necessary for me to realize once again why I love science.				
Living on the Notre Dame campus with other REU students, UND athletes, and other summer school students was a whole other experience that prepared me for the world beyond high school. For 9 weeks, I worked, played and bonded with the other students, and had the opportunity to live the life of an independent college student.				
Along with the individually tailored research projects and the housing opportunity, there were seminars on public speaking, trips to the Fermi National Accelerator Laboratory, and one-on-one writing seminars for the end of the summer research papers we were each required to write. By the end of the summer, I wasn’t ready to leave the research that I was doing. While my research didn’t yield definitive results for the effects of curcumin on cervical cancer cells, my research on curcumin-functionalized CoFe2O4/TiO2 core-shell nanoconjugates indicated that there were many unknown factors affecting the HeLa cells, and spurred the lab to expand their research into determining whether or not the timing of the drug delivery mattered and whether or not the position of the binding site of the drugs would alter the results. Through this summer experience, I realized my ambition to pursue a career in research. I always knew that I would want to pursue a future in science, but the exciting world of research where the discoveries are limitless has captured my heart. This school year, the REU program has offered me a year-long job, and despite my obligations as a high school senior preparing for college, I couldn’t give up this offer, and so during this school year, I will be able to further both my research and interest in nanotechnology. """


print('SummerActivity result :', SummerActivity(essay_input))