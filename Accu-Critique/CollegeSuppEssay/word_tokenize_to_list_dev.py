import re
#Academic Verbs
academicVerbs = """
AAPI
API
Ableism
Accessibility
Ace
Affinity bias
Affinity groups
Affirmative action
Ageism
Ally
Allyship
Asexual
Atheism
Attribution error
Behavioral diversity
Bi
Bias
Bi-cultural
Biphobia
Black
BME
BAME
Butch
Cisgender or Cis
Cognitive diversity
Confirmation bias
Conscious prejudice
Corporate social responsibility
Cover
Creative abrasion
Culture fit
Deadnaming
DEI
Diaspora
Disability


Ethnic groups
Ethnocentrism
Femme
Gay
Gender
Gender dysphoria
Gender expression
Gender identity
Gender privilege
Gender reassignment
Genderqueer
Groupthink
GSD
Hepeating
Heterosexual privilege
Heterosexual/Straight
Homophobia
Homosexual
Imposter Syndrome
Inclusion
Inclusive Leader
In-group bias
Innate diversity
Intersectionality
Intersex
Lesbian
Lesbophobia
LGBTQ+
LGBTQI
LGBTQIA
Mansplain
Mentor
Microadvantages
Microaffirmations
Microaggression
Multiracial, mixed heritage, dual heritage, mixed-race, mixed-ethnicity – or simply “mixed”
Pan
People of color (PoC)
Prejudice
Privilege
Pronoun
Psychological Safety
Queer
Racism
Sex
Sexual orientation
Socioeconomic privilege
Sponsor
Stereotype threat
Stereotypes
Straight
Trans or transgender
Transitioning
Transphobia
Transsexual
Unconscious bias
Underrepresented groups
White privilege
White supremacy
Workplace inclusion
Xenophobia
Zero sum game
Outgroup bias
Equity
Neurodiverse
Non-binary
Oppression
Discrimination
Diversity
Dominant Culture
Emotional tax
Employee Resource Group
Equality
AAPI  — AAPI is an acronym for Asian Americans & Pacific Islanders. Other similar acronyms are APA which means Asian-Pacific American and API which means Asian-Pacific Islander. These acronyms replace a derogatory term, “Oriental” in the 1960s.
AAVE — AAVE is an acronym for African American Vernacular English. AAVE is a dialect of American English characterized by pronunciations and vocabulary used by some North American Black people and is a variation of Standard American English.
Ableism — Ableism means the practices or dominant attitudes by a society that devalue or limit the potential for people with disabilities. Ableism is the act of giving inferior value or worth to people who have different types of disabilities (physical, emotional, developmental, or psychiatric).
Accessibility — Accessibility is the term for making a facility usable by people with physical disabilities. Examples of accessibility include self-opening doors, elevators for multiple levels, raised lettering on signs and entry ramps
Accountability — Accountability refers to ways individuals and communities hold themselves to their goals and actions, while acknowledging the values and groups to which they are responsible.
Acculturation — Acculturation means a process when members of a cultural group adopt the patterns, beliefs, languages, and behaviors of another group’s culture.
ADA  — ADA is an abbreviation for the Americans with Disabilities Act. The ADA is a civil rights law that prohibits discrimination against people with disabilities.
ADOS — ADOS means American Descendants of Slavery. ADOS is a group that seeks to reclaim and restore the critical national character of the African American identity and experience in the United States.
Agender — Agender means a person who does not identify themselves as having a particular gender.
Affinity Groups — Affinity Groups are a collection of individuals with similar interests or goals. Affinity Groups promote inclusion, diversity, and other efforts that benefit employees from underrepresented groups.
Affirmative Action — Affirmative Action is the practice of favoring groups of people who have been discriminated against in the past.
African American — The term African American refers to people in the United States who have ethnic origins to Africa.
Alaska Native — Alaska Native is a term for the indigenous people of Alaska. Alaska Natives consist of over 200 federally recognized tribes who speak 20 different languages.
Ally — Ally is a term for people who advocate for individuals from underrepresented or marginalized groups in a society.
Allyship — Allyship is the process in which people with privilege and power work to develop empathy towards to advance the interests of an oppressed or marginalized outgroup. Allyship is part of the anti-oppression or anti-racist conversation, which puts into use social justice theories and ideals. The goal of allyship is to create a culture in which the marginalized group feels supported.
Amplification — Amplification is a term used for the techniques a person uses to give a member of a less dominant group more credit by repeating their message.
Androgyne — Androgyne is a term for a person identifying or expressing gender outside of the gender binary.
Anglo  — Anglo or Anglo-Saxon means to be related to the descendants of Germanic people who reigned in Britain until the Norman conquest in 1066. Anglo often refers to white English-speaking persons of European descent in England or North America, not of Hispanic or French origin.
Anti-Black — Anti-Black refers to the marginalization of Black People and the unethical disregard for anti-Black institutions and policies.
Anti-Racism — Anti-Racism means to actively oppose racism by advocating for political, economic, and social change.
Anti-Racist Ideas — Anti-Racist ideas refer to the assumption that racial groups are equals despite their differences.
Arab — Arab refers to people who have ethnic roots in the following Arabic‐speaking lands: Algeria, Bahrain, Egypt, Iraq, Jordan, Kuwait, Lebanon, Libya, Morocco, Oman, Palestine, Qatar, Saudi Arabia, Sudan, Syria, Tunisia, the United Arab Emirates, and Yemen.
Asexual — An “asexual person’ is used to describe people who do not experience sexual attraction.
Asian-American  — Asian-American is a term that means to have origins in Asia or the Indian subcontinent. Asian-American includes people who live in the United States and indicate their race as:
Asian
Indian
Chinese
Filipino
Korean
Japanese
Vietnamese
Other Asian
Assimilation — Assimilation is a term for the concept where an individual, family, or group gives up certain aspects of their culture to adapt to the beliefs, language, patterns, and behaviors of a new host country.



BAME — BAME meaning “Black, Asian and Minority Ethnic” is an acronym used mostly in the United Kingdom to identify Black and Asian people.
Belonging — Belonging is a term used to define the experience of being accepted and included by those around you. Belonging means to have a sense of social connection and identification with others.
Bias — Bias means to have a prejudice against groups that are not similar to you or to have show preference for people that are similar to you.
Bicultural — Bicultural is a term that refers to people who possess the values, beliefs, languages, and behaviors of two distinct ethnic or racial groups.
Bigotry — Bigotry means to glorify a person’s own group and have prejudices against members of other groups.
BIPOC — What does BIPOC mean? The BIPOC acronym stands for Black, Indigenous, People of Color. Read BIPOC: The Hottest (Controversial) Word in Diversity?
Biphobia — Biphobia means to have an irrational fear, hatred, or intolerance for people who identify as bisexual.
Biracial — Biracial is a term for mixed race. Biracial is used to describe a person who identities as being of two races, or whose parents are from two different race groups.
Birth Assigned Sex — Birth Assigned Sex refers to a person’s biological, hormonal, and genetic composition at the time of their birth.
Bisexual — Bisexual, commonly known as Bi, is a term for individuals who are attracted to people of two genders.
Black — Black means to be related to people who have ethnic origins in Africa, or not of white European descent. Black is often used
interchangeably with African American in the United States.
Black-American — Black-American is a term used by Black people born in the United States who do not identify with having ethnic roots in Africa or other nations.
Black ethnic group — A phrase used in the UK to describe a person who identifies as Black. Other accepted terms are “people from a Black Caribbean background” and “Black people”.
Black Lives Matter — Black Lives Matter is a movement that addresses systemic racism and violence against African Americans and other groups with ties to Black culture.
Block list — An inclusive replacement phrase in the U.S. and the UK for “blacklist” or “black list”.
BME — What is BME? BME stands for Black [and Asian] & Minority Ethnic and is commonly used in the UK, interchangeably with BAME. (See GOV.UK’s style guide on the use of BME, BAME, and people of colour)



Caren Act — “CAREN Act” (Caution Against Racially Exploitative Non-Emergencies). The ordinance is similar to the statewide AB 1550 bill introduced by California Assemblyman Rob Bonta, making it unlawful and accountable for a caller to “fabricate false racially-biased emergency reports.”.
Caucuses — Caucuses are groups that provide spaces for people to work within their own racial or ethnic groups.
CD&I — Acronym for Culture, Diversity and Inclusion. Walmart, the U.S. Navy and others use CD&I to describe their overall diversity initiatives.
Chicanx — Chicanx means a person related to Mexican Americans or their culture. Chicanx is a gender-neutral term used in the place of Chicano or Chicana.
Cisgender (CIS) — Cisgender means a person whose gender identity matches the sex they were assigned at birth. The abbreviation for Cisgeneder is CIS.
Cissexual — Cissexual is a term that refers to a person who identifies with the same biological sex that they were assigned at birth.
Classism — Classism is a term that means to have prejudicial thoughts or to discriminate against a person or group based on differences in socioeconomic status and income level.
Code-Switching — Code-switching means when a person changes the way they express themselves culturally and linguistically based on different parts of their identity and how they are represented in the group they’re with.
Color Blind(ness) — Color Blind(ness) or being Color Blind means treating people as equally as possible without regard to race, culture, or ethnicity.
Collusion — Collusion is when a person acts to perpetuate oppression or prevent people from working to eliminate oppression.
Colonization — Colonization refers to forms of invasion, dispossession, or controlling an underrepresented group.
Color Brave — Color Brave is when a person has conversations about race that can help people better understand each other’s perspectives and experiences to improve inclusiveness in future generations.
Coming Out — Coming Out is a phrase used to define the process of making others aware of one’s sexual orientation, and is also known as Coming Out of the Closet.
Communities of Color — Communities of Color is used in the United States to describe groups of people who are not identified as White, with emphasis on common experiences of racism.
Corporate Social Responsibility — Corporate Social Responsibility means to practice positive corporate citizenship to make a positive impact on communities, not just focusing on maximizing profits.
Covert Racism — Covert Racism is an indirect behavior used to express racist attitudes or ideas in hidden or subtle forms.
Cross-Dresser — Cross-Dresser refers to people who wear clothing that is traditionally associated with a different gender than the one they identify with.
Cultural Appropriation — Cultural Appropriation means the act of stealing cultural elements for a person’s own use or profit.
Cultural Identity — Cultural Identity means the identity or feeling of belonging to a group based on nationality, ethnicity, religion, social class, generation, locality, or other types of social groups that have their own distinct culture.
Culture — Culture is defined as a social system of customs that are developed by a group of people to ensure its survival and adaptation.
Culture Add — Culture Add refers to people who value company culture and standards, as well as bringing an aspect of diversity that positively contributes to the organization.
Culture Fit — Culture Fit refers to a person’s attitudes, values, behaviors, and beliefs being in line with the values and culture of an organization. Culture Add, defined above, is becoming a preferred alternative to Culture Fit.




D&I — D&I stands for “diversity and inclusion” and is often a catch-all for diversity initiatives.
The phrase “Diversity and Inclusion” (D&I) is not always used in the same order. For example, in the social media ad space, Facebook uses “Diversity and Inclusion (D&I)” while Twitter uses “Inclusion and Diversity (I&D)”.
What is the difference between “diversity” versus “inclusion”? As written about in Top Diversity Job Titles,
“Diversity is the what (the characteristics of the people you work with such as gender, ethnicity, age, disability and education). Inclusion is the how (the behaviors and social norms that ensure people feel welcome).”
Some companies also use the words “equity” (Slack) and “equality” (Salesforce) in their diversity titles. Equity and equality are usually alternatives to “inclusion”.
Decolonization — Decolonization refers to the active resistance against colonial powers from indigenous culture groups.
DEI — What is DEI? DEI is an acronym that stands for Diversity, Equity & Inclusion.
Demisexual — A sexual orientation where people experience sexual attraction only to people they are emotionally close to.
Disabled People — An inclusive replacement phrase used in the UK for “the disabled” or “people with disabilities”.
Disability — Disability is a term used to describe people who have a mental or physical impairment which has a long-term effect on their ability to carry out day-to-day activities. What is the politically correct term for disabled? “Person with a Disability” is a more inclusive, less biased term to describe someone who is disabled.
Disablism — Disablism means promoting the unequal or differential treatment of people with actual or presumed disabilities; either consciously or unconsciously.
Diaspora — Diaspora is either voluntary or forcible movement of people from their homelands into new regions.
Discrimination — What is Discrimination? Discrimination is a term used to describe the unequal treatment of individuals or groups based on race, gender, social class, sexual
orientation, physical ability, religion, national origin, age, physical or mental abilities, and other categories that may result in differences.
Diversity — Diversity is defined as individual differences between groups based on such things as:
abilities
age
disability
learning styles
life experiences
neurodiversity
race/ethnicity
class
gender
sexual orientation
country of origin
cultural, political or religious affiliation
any other difference
Which term is a synonym for diversity? Popular synonyms are “mixture”, “variance”, “difference”, and “un-alike”.
Dominant Culture — Dominant Culture is a term that refers to the cultural beliefs, values, and traditions that are based on those of a dominant society. Practices in a Dominant Culture practices are considered “normal” and “right.”



EDI — EDI Is an acroynm that stands for Equality, Diversity and Inclusion. It’s used by such organizations as University of California Los Angeles (UCLA), American Library Association and the National Institutes of Health.
EEO — EEO stands for Equal Employment Opportunity. EEO is part of the United States Civil Rights Act of 1964 which prohibits discrimination in any aspect of employment based on an individual’s race, color, religion, sex, or national origin.
Emotional Tax — Emotional Tax refers to the effects of being on guard to protect against bias at work because of gender, race, and/or ethnicity. Emotional Tax has effects on a person’s health, well-being, and the ability to be successful at work.
Enby — Enby is an abbreviation used for a nonbinary person in the LGBTQ community. It’s a phonetic pronunciation of NB, short for nonbinary, or people who do not identify their gender as male or female.
Equality — The term “Equality” (in the context of diversity) is typically defined as treating everyone the same and giving everyone access to the same opportunities. It is sometimes used as an alternative to “inclusion”. The company Salesforce, for example, uses Chief Equality Officer as the job title for the top diversity and inclusion executive.
Equity — The term “equity” (in the context of diversity) refers to proportional representation (by race, class, gender, etc.) in employment opportunities. The company Slack, for example, uses “Equity” in some job titles (e.g. Senior Technical Recruiter, Diversity Equity Inclusion Lead).
ERG — ERG stands for Employee Resource Group. ERGs are employee identity or experience-based groups that are meant to build the feeling of community in the workplace. ERGS are sometimes known as Affinity Groups or Diversity Groups. Google has over 16 Employee Resource Groups with more than 250 chapters globally that support professional development opportunities for underrepresented communities. Creating ERGs are often a part of a company’s Steps to Implement a Diversity and Inclusion Strategy.
Ethnic Diversity  — The term Ethnic Diversity refers to the presence of different ethnic backgrounds or identities.
Ethnic Minorities — Used in the UK when referring to all ethnic groups except the “White British Group. Ethnic minorities include White minorities, such as Gypsy, Roma, and Irish Traveller groups.”
Ethnicity — Ethnicity, or Ethnic Group, is a way to divide people into smaller social groups based on characteristics like:
cultural heritage
values
behavioral patterns
language
political and economic interests
ancestral geographical base
ESL — ESL is an acronym for English as a Second Language. ESL refers to individuals who do not speak English as their first language but may still be proficient in speaking English.
Exclusion — Exclusion means leaving someone out based on their differences. These differences can be related to race, gender, sexual orientation, age, disability, class, or other social groups.



Femme — Femme is a gender identity where a person has an awareness of cultural standards of femininity and actively carries out a feminine appearance or role.
Filipinx — Filipinx means a person who is a national of the Philippines, or a person of Filipino descent. Filipinx is a gender-neutral term used in the place of Filipino or Filipina.
First Nations — First Nations is a term used to describe indigenous people from Canada who are not Inuit or Métis. Many First Nations people prefer to define or identify themselves by their specific tribal affiliations.
Folx — Folx is an umbrella term for people with non-normative sexual orientation or identity.
FTM — FTM is an acronym for the Female-to-Male Spectrum. FTM is used by people who are assigned female at birth but identify with or express their gender as a male part of the time.




Gay — Gay is an umbrella term used to refer to people who experience a same-sex or same-gender attraction. Gay is also an identity term used to describe a male-identified person who is attracted to other male-identified people in a romantic, sexual, and/or emotional sense.
Gender — Gender is a term used to describe socially constructed roles, behaviors, activities, and attributes that society considers “appropriate” for men and women. It is separate from ‘sex’, which is the biological classification of male or female based on physiological and biological features.
Gender Binary — Gender Binary is a term used to describe the classification system consisting of two genders, male and female.
Gender Dysphoria — Gender Dysphoria is a phrase used to describe a feeling of discomfort that occurs in people whose gender identity differs from their birth assigned sex.
Gender Expression — Gender Expressions means that a person shows external displays of gender (masculine or feminine) based on one or more of the following:
dress
demeanor
social behavior
Gender Fluid — What do you call someone who is Gender Fluid? A person who is gender fluid changes their gender over time or may switch between dressing as male or female day-to-day.
Gender Identity — Gender Identity means a person’s perception of their gender. Gender Identity may or may not correspond with their birth assigned sex.
Gender Neutral — Gender Neutral, or Gender Neutrality, means that policies, language, and other social institutions should avoid distinguishing roles based on sex or gender in order to avoid discrimination.
Gender Non-Conforming  — Gender Non-Conforming sometimes called Gender-Variant is a term used to describe a person who does not conform to society’s expectations of gender expression.
Gender Policing — Gender Policing means the enforcement of normative gender expressions on a person who is perceived as not participating in behavior that aligns with their assigned gender at birth.
Gender Queer — Gender Queer, or Genderqueer, is a catch-all term for people who have non-binary gender identities. What do you call a non-binary person? Calling them by their preferred pronouns is preferred.
Gender Role — A Gender Role is a socially assigned expectation or cultural norm related to behavior, mannerisms, dress, etc. based on gender.
Gender Spectrum — Gender Spectrum refers to the idea that there are many different genders, besides male and female.
Groupthink — Groupthink is when people discourage a person from thinking a certain way or making decisions using individual creativity.
Gypsies (Gypsy Travellers) — A recognized ethnic group in the UK under the Race Relations Act.



HBCU — HBCU is an acronym that stands for “Historically Black Colleges and Universities”. HBCUs were established, post-American Civil War, in the United States to primarily serve the Black community, although they allow admission to students of all races.
Hepeating — Hepeating is when a man repeats a woman’s comments to takes them as his own to gain credit or praise for the idea.
Heteronormativity — Heteromnormativity is the assumption that heterosexuality is natural, ideal, or superior to other sexual preferences. Examples of Heteronormativity include:
the lack of same-sex couples in media or advertising
laws against same-sex marriage
Heterosexism — Heterosexism is a term used to describe the belief that heterosexuality is superior or “normal”  compared to other forms of sexuality or sexual orientation.
Heterosexual — Heterosexual a term used to identify a female-identified person who is attracted to a male-identified person, or a male-identified person who is attracted to a female-identified person.
Hispanic — Hispanic is a term used to describe people who speak Spanish and/or are descended from Spanish-speaking populations.
Homophobia — Homophobia means to have an irrational fear or intolerance of people who are homosexual or having feelings of homosexuality.
Host Culture — Host Culture refers to the dominant culture in a place people live in after leaving their home country.



Implicit Bias — What is Implicit Bias? Implicit Bias, or hidden bias, refers to the attitudes or stereotypes that affect a person’s understanding, actions, or decisions unconsciously as it relates to people from different groups. Also known as Unconcious Bias.
Imposter Syndrome — Imposter Syndrome is common in members of underrepresented groups. Imposter Syndrome is present when high-achieving individuals are in constant fear of being exposed as a fraud and are unable to internalize their accomplishments.
Inclusion — The term Inclusion refers to the process of bringing people that are traditionally excluded into decision making processes, activities, or positions of power. Inclusion is sometimes called Inclusiveness and allows individuals or groups to feel safe, respected, motivated, and engaged.
Inclusive Language — Inclusive Language refers to the use of gender non-specific language to avoid assumptions around sexual orientation and gender identity.
Indigenous People  — Indigenous People is a term used to identify ethnic groups who are the earliest known inhabitants of an area, also known as First People in some regions.
Individual Racism — Individual Racism is when a person acts to perpetuate or support racism without knowing that is what they are doing. For example, racists jokes, avoiding people of color or accepting racist acts.
In-Group Bias — In-Group Bias is when people respond more positively to people from their “in-groups” than they do for people from “out-groups”.
Institutional Racism — Institutional Racism means that institutional practices and policies create different outcomes for different racial groups. These policies may not specifically target any racial group, but their effect creates advantages for white people and oppression or disadvantages for people of color. Often used interchangeably with Structural Racism.
Integration — Integration is when an individual maintains their own cultural identity while also becoming a participant in a host culture.
Intersectionality — Intersectionality means to intertwine social identities like gender, race, ethnicity, social class, religion, sexual orientation, or gender identity which causes unique opportunities, barriers, experiences, or social inequality.
Intersex — Intersex means to be born with a combination of male and female biological traits.
Inuit — Inuit is a term used to describe a member of an indigenous group from northern Canada and parts of Greenland and Alaska.
Irish Traveller — A recognized ethnic group in the UK under the Race Relations Act.




Karen — Karen is a common stereotype of white women who use privilege to demand something out of the scope of what is necessary. Wikipedia says a Karen would:
“demand to “speak to the manager”, anti-vaccination beliefs, being racist, or sporting a particular bob cut hairstyle. As of 2020, the term was increasingly being used as a general-purpose term of disapproval for middle-aged white women”
DIVERSITY TERMS STARTING WITH L

Latino — Latino is a term used to describe people who are from or descended from people from Latin America.
Latinx — What is Latinx? Latinx is a gender-neutral term used to replace Latino or Latina when referring to a person of Latin-American descent.
Lesbian — Lesbian is a term that refers to a female-identified person who is attracted emotionally, physically, or sexually to other female-identified people.
Lesbophobia — Lesbophobia is an irrational fear or hatred of, and discrimination against lesbians or lesbian behavior.
LGBT — Abbreviation for lesbian, gay, bisexual, and transgender (often used to encompass sexual preference and gender identities that do not correspond to heterosexual norms).
LGBT is an acronym with multiple variations such as:
LGBTQ — Lesbian, gay, bisexual, transgender, and queer (or questioning).
LGBTQIA — Lesbian, gay, bisexual, transgender, queer (or questioning), intersex, and asexual (or allies).
LGBTA — Lesbian, gay, bisexual, transgender, and asexual/aromantic/agender.
LGBTIQQ — Lesbian, Gay, Bisexual, Transgender, Intersex, Queer, and Questioning.
LGBTQ2+ — Lesbian, gay, bisexual, transgender, queer (or sometimes questioning), and two-spirited. The “+”  signifies a number of other identities and is used to keep the abbreviation brief when written out.  Some write out the full abbreviation which is LGBTTTQQIAA.




Mansplain — Mansplain is a word used to describe when are men explaining something to a person in a condescending or patronizing manner, typically a woman.
Marginalization — Marginalization means to exclude, ignore, or relegate a group of people to an unimportant or powerless position in society.
Melting Pot — Melting Pot is a metaphor people use to describe a society where various types of people blend together as one.
Métis — Métis is a French word that refers to someone with mixed ancestry. Métis is a common term referring to a multiancestral indigenous group whose homeland is in Canada and parts of the United States between the Great Lakes region and the Rocky Mountains.
Metrosexual — Metrosexual means refers to a well-groomed style for non-queer men that is a mix of the words “heterosexual” and “metropolitan”.
Mexican American — Mexican American refers to the group of Americans of full or partial Mexican descent in the United States. What do you call a Mexican American? Chicano, Chicana, or Chicanx or accepted terms for Mexican Americans.
Microaggression — Microaggression is a term that describes daily behavior (verbal or nonverbal) that communicates hostile or negative insults towards a group, either intentionally or unintentionally, particularly culturally marginalized groups.
Minority — Minority is a term used to describe racially, ethnically, or culturally distinct groups that are usually subordinate to more dominant groups. These groups are called Minority Groups.
Misgender — To refer to someone using a word (especially a pronoun or form of address) that does not correctly reflect the gender with which they identify.
Mixed Race — What do you call a person of Mixed Race? Mixed Race means a person who has parents that belong to different racial or ethnic groups.
Model Minority — According to the Racial Equity Tools Glossary, Model Minority is:
“A term created by sociologist William Peterson to describe the Japanese community, whom he saw as being able to overcome oppression because of their cultural values.”
Movement Building — Movement Building refers to an effort to address systemic problems or injustices while promoting alternative solutions or visions.
MTF — MTF is an acronym for the Male-to-Female Spectrum. MTF is used to describe people who are assigned the male gender at birth but identifies or express their gender as a female all or part of the time.
Multicultural — Multicultural means pertaining to more than one culture.
Multicultural Competency — Multicultural Competency refers to the process of learning about other cultures and becoming allies with people from different backgrounds.
Multiethnic — Multiethnic describes a person who comes from more than one ethnicity.
Multi-ethnic — A commonly used term in the UK that means consisting of, or relating to various different races.
Multiracial — Multiracial describes a person who comes from more than one race.

Native American  — Native American is a broad term that refers to people of North and South America but is generally used to describe the indigenous people from the United States. Native American is often used interchangeably with American Indian, although many Native Americans find the word “Indian” offensive and prefer to identify themselves by their specific tribe.
Some companies and sports teams are even changing their names or mascots, because of racial bias related to the word “Indian”.
Neurodivergent — Neurodivergent, sometimes known as ND, means having a brain that works in a way that diverges significantly from the dominant societal standards of “normal.”
Neurodiversity — Neurodiversity is a relatively new term coined in 1998 by autistic, Australian sociologist Judy Singer in 1998. The neurodiversity definition began as a way to describe people on the Autistic spectrum. Neurodiversity has since broadened to include people with:
Autism
Dyslexia
ADHD (Attention Deficit Hyperactivity Order)
Dyscalculia
DSD (Dyspraxia)
Dysgraphia
Tourette Syndrome
and other neurological differences
What does it mean to be on the spectrum? Check out Neurodiversity: The Definitive Guide for more about the meaning of neurodiversity and examples of it in the workplace.
Neurodiversity Movement — The Neurodiversity Movement is a social justice movement that is seeking equality, respect, inclusion, and civil rights for people with Neurodiversity.
Neurotypical — Neurotypical is often abbreviated as NT and it means to have a style of neurocognitive functioning that falls within the dominant societal standards of “normal.” Neurotypical can be used as either an adjective (“They’re neurotypical”) or a noun (“They are a neurotypical”).
Neurominority — Neurominority refers to an underrepresented group of Neurodiverse people who may face challenges or bias from society.
Non-Binary — What do you call a gender-neutral person? The preferred term is Non-Binary. What does it mean to be non-binary? Non-Binary is a term used to describe people who identify with a gender that is not exclusively male or female or is in between both genders.
Non-White — Using this phrase in the UK is not recommended according to the GOV.UK Writing About Ethnicity Style Guide “because defining groups in relation to the White majority was not well received in user research.”



Oppositional Sexism — Oppositional Sexism is the belief that femininity and masculinity are rigid and exclusive categories.
Oppression — Oppression refers to systemic and institutional abuse of power by a dominant or privileged group at the expense of targeted, less privileged groups.
Outgroup Bias — Outgroup Bias is when people view people from outside their “group” as less similar and have negative bias against them.




Pacific Islander — Pacific Islander, or Pasifika, is a term that  refers to the indigenous inhabitants of the Pacific Islands, specifically people with origins whose origins from the following sub-regions of Oceania:
Polynesia
Melanesia
Micronesia
Pansexual — Pansexual is a term used to describe a person who has an attraction to a person regardless of where they fall on the gender or sexuality spectrum.
Patriarchy — Patriarchy refers to a social system where power and authority are held by men.
People-First Language (PFL) — People-first language puts a person before a diagnosis or way of being. It describes what a person “has” rather than saying what a person “is”. (e.g., “person with a disability” vs. “disabled”)
People of Color — People of Color, or Person of Color, is a phrase used in the United States to describe people who are not white and is meant to be inclusive of non-white groups, with emphasis on common experiences of racism.
POC — POC stands for People of Color. It is commonly used as an acronym in the United States to describe people who are not white.
Power — Power (in the context of diversity) is considered to be unequally distributed globally due to the following things:
wealth
whiteness
citizenship
patriarchy
heterosexism
education
Prejudice — Prejudice means to pre-judge or have a negative attitude towards one type of person or group because of stereotypes or generalizations.
Privilege — Privilege (in the context of diversity) means an unearned social power for members of a dominant group of society including benefits, entitlements, or a set of advantages in society.
Pronouns — Pronouns (in the context of diversity) are consciously chosen phrases that people use to represent their gender identity. There are certain pronouns to avoid like “he” or “she”, especially during the hiring process or in the workplace.



QPOC — What is QPOC? QPOC is an acronym for Queer People of Color used in the UK and Canada. Another similar acronym is QTIPOC which stands for Queer, Transgender, and Intersex People of Color.
Queer — What does it mean to be queer? The term Queer is an umbrella term that allows non-heterosexual people to identify their sexual orientation without stating who they are attracted to. The term Queer includes gay men, lesbians, bisexuals, and transgendered people.



Race — What does race mean? Race is a social term that is used to divide people into distinct groups based on characteristics like:
physical appearance (mainly skin color)
cultural affliction
cultural history
ethnic classification
social, economic, and political needs
When was the term race first used? According to Wikipedia:
“The word “race”, interpreted to mean an identifiable group of people who share a common descent, was introduced into English in about 1580, from the Old French rasse (1512), from Italian razza.”
Racism — Racism is the oppression of people of color based on a socially constructed racial hierarchy that gives privilege to white people.
Racial and Ethnic Identity — Racial and Ethnic Identity refers to a person’s experience of being a member of an ethnic and racial group. Racial and Ethnic Identity is based on what a person chooses to describe themselves as based on the following:
biological heritage
physical appearance
cultural affiliation
early socialization
personal experience
Racial Justice — Racial Justice means to reinforce policies, practices, actions, and attitudes that produce equal treatment and opportunities for all groups of people.
Reclaimed Language — Reclaimed Language is language that has traditionally been used to degrade certain groups, but members of the community have reclaimed and used as their own. For example, “queer” or “queen”.
Religion — Religion is a system of beliefs that are spiritual and part of a formal, organized institution.
Restorative Justice — Restorative Justice is an effort to repair the harm caused by crime and conflict related to bias or racism.
Reverse Racism — Reverse Racism is perceived discrimination against a dominant group or majority.
Roma Traveller — A recognized ethnic group in the UK under the Race Relations Act.




Safe Space — Safe Space means a place people can be comfortable expressing themselves without fear as it relates to their cultural background, biological sex, religion, race, gender identity or expression, age, physical or mental ability.
Segregation — Segregation is a systemic separation of people into racial or ethnic groups during the activities of daily life.
Separation — Separation is when an individual or group rejects a host culture and maintains their cultural identity.
Sex — Sex, as it relates to diversity, means the biological classification of male or female based on the physical and biological features of a person. A person’s sex may vary from their gender identity.
Sexual Orientation — Sexual Orientation refers to the sex(es) or gender(es) a person is connected to emotionally, physically, sexually, or romantically. Examples of sexual orientation include:
gay
lesbian
bisexual
heterosexual
asexual
pansexual
queer
“Sexual orientation” is considered more politically correct thatn “sexual preference” since “preference” implies a conscious choice.
Sponsorship — Sponsorship is an action by allies that are taken to advance the career of members of marginalized groups. These may include mentoring, protecting, or promoting.
Stereotype — A Stereotype is an over-generalized belief about a particular group or category of people. A Stereotype represents the expectation that something is true about every member of that group.
Straight — Straight refers to a person who is attracted to a person of a different gender to their own.
Structural Racism — Structural Racism, sometimes called Institutional Racism, refers to institutional practices or policies that create different outcomes for various racial groups. The effects of Structural Racism usually create advantages for white people and oppression or disadvantages for people of color.



TERFs — “trans-exclusionary radical feminists”, TERFs constitute “a minority of a minority of feminists,” says Grace Lavery, a UC Berkeley literature professor and writer.
Third Gender  — Third Gender refers to a category of people who do not identify as male or female, but rather as neither, both, or a combination of male and female genders.
Tokenism — Tokenism is a practice of including one or a few members of an underrepresented group in a team or company.
Transfeminine — Transfeminine describes a person who identifies as “trans” but identifies their gender expression as feminine.
Transgender — What do you call a man that becomes a woman? Or a woman that becomes a man? Transgender is an umbrella term for people whose gender expression or identity is different from their assigned sex at birth.
Transmasculine — Transmasculine means a person who identifies as “trans” but identifies their gender expression as masculine.
Transition — Transition, in terms of diversity, is a process that people go through to change their physical appearance or gender expression through surgery or using hormones to align with their gender identity.
Transphobia — Transphobia means fear, hatred, or discrimination towards people who identify as Transgender.
Transvestite — A person who dresses as the binary opposite gender expression. What is the politically correct term for a transvestite? The more politically correct term is “Cross-dresser”.
Two-Spirit — Two-Spirit is a phrase that refers to a person who is Native American that embodies both masculine and feminine genders.



Unconscious Bias — Unconscious Bias, also known as Implicit Bias, refers to attitudes or stereotypes about certain groups which are often based on mistaken or inaccurate information.
Underrepresented Group — An Underrepresented Group refers to a subset of a population with a smaller percentage than the general population. For example, women, people of color, or indigenous people.
Unity — Unity in Diversity is an expression of harmony between dissimilar individuals or groups. Who coined the term unity in diversity? Toppr.com says:
“The phrase ‘unity in diversity’ is coined by Jawaharlal Nehru.”
URM — Acronym for underrepresented minorities.

Values Fit — Values Fit is being used in the place of Culture Fit to identify the connection of shared goals rather than viewpoints or background.

White Privilege — White Privilege represents the unearned set of advantages, privileges, or benefits given to people based solely on being white.
White Supremacy — White Supremacy refers to the exploitation or oppression of nations or people of color by white people for the purpose of maintaining and defending a system of wealth, privilege, and power.
Wimmin — Wimmin is a nonstandard spelling of the word “women” used by feminists in an effort to avoid the word ending “-men”.
Womxn — Womxn is a term used to replace the word women in an attempt to get away from patriarchal language. Womxn is also meant to be inclusive of trans women, and some non-binary people.
Womyn — Womyn is a nonstandard spelling of the word “women” used by feminists in an effort to avoid the word ending “-men”.
Workforce Diversity —  Workforce Diversity means having a group of employees with similarities and differences like age, cultural background, physical abilities and disabilities, race, religion, gender, and sexual orientation. During which era was the term workforce diversity first used? Workforce Diversity came into the business scene in the early 1980s.
Work-Life Effectiveness — Work-Life Effectiveness is a talent management strategy that focuses on doing the best work with the best talent regardless of the diverse aspects of individuals.
Workplace Inclusion — Workplace inclusion is an intentional effort to create an atmosphere of belonging where all parties can contribute and thrive regardless of their age, gender, race, ethnicity, gender, or sexual orientation.

Xenophobia — Xenophobia is prejudice or a dislike for people from other countries.

Zi/Hir — Zi/Hir are gender-inclusive pronouns used to avoid relying on gender binary based language or making assumptions about people’s gender.

A
Able-ism | The belief that disabled individuals are inferior to non-disabled individuals, leading to discrimination toward and oppression of individuals with disabilities and physical differences.
Accessibility | The extent to which a facility is readily approachable and usable by individuals with disabilities, particularly such areas as the residence halls, classrooms, and public areas.
Accomplice(s) | The actions of an accomplice are meant to directly challenge institutionalized racism, colonization, and white supremacy by blocking or impeding racist people, policies and structures.
Acculturation | The general phenomenon of persons learning the nuances of or being initiated into a culture. It may also carry a negative connotation when referring to the attempt by dominant cultural groups to acculturate members of other cultural groups into the dominant culture in an assimilation fashion.
Actor [Actions] | Do not disrupt the status quo, much the same as a spectator at a game, both have only a nominal effect in shifting an overall outcome.
Adult-ism | Prejudiced thoughts and discriminatory actions against young people, in favor of the older person(s).
Advocate | Someone who speaks up for themselves and members of their identity group; e.g. a person who lobbies for equal pay for a specific group.
Age-ism | Prejudiced thoughts and discriminatory actions based on differences in age; usually that of younger persons against older.
A-Gender | Not identifying with any gender, the feeling of having no gender.
Agent | The perpetrator of oppression and/or discrimination; usually a member of the dominant, non‐target identity group.
Ally | A person of one social identity group who stands up in support of members of another group. Typically, member of dominant group standing beside member(s) of targeted group; e.g., a male arguing for equal pay for women.
Androgyne | A person whose biological sex is not readily apparent, whether intentionally or unintentionally.
Androgynous | A person whose identity is between the two traditional genders.
Androgyny | A person who rejects gender roles entirely.
Androgynous | Someone who reflects an appearance that is both masculine and feminine, or who appears to be neither or both a male and a female.
Anti‐Semitism | The fear or hatred of Jews, Judaism, and related symbols.
A-Sexuality | Little or no romantic, emotional and/or sexual attraction toward other persons. Asexual could be described as non-sexual, but asexuality is different from celibacy, which is a choice to not engage in sexual behaviors with another person.
Assigned Sex | What a doctor determines to be your physical sex birth based on the appearance of one's primary sex characteristics.
Assimilation | A process by which outsiders (persons who are others by virtue of cultural heritage, gender, age, religious background, and so forth) are brought into, or made to take on the existing identity of the group into which they are being assimilated. The term has had a negative connotation in recent educational literature, imposing coercion and a failure to recognize and value diversity. It is also understood as a survival technique for individuals or groups.

B
Bias | Prejudice; an inclination or preference, especially one that interferes with impartial judgment.
Bigotry | An unreasonable or irrational attachment to negative stereotypes and prejudices.
Bi-Phobia | The fear or hatred of homosexuality (and other non‐heterosexual identities), and persons perceived to be bisexual.
Bi-Racial | A person who identifies as coming from two races. A person whose biological parents are of two different races.
Bi-Sexual | A romantic, sexual, or/and emotional attraction toward people of all sexes. A person who identifies as bisexual is understood to have attraction to male and female identified persons. However, it can also mean female attraction and non-binary, or other identifiers. It is not restricted to only CIS identifiers.
Brave Space | Honors and invites full engagement from folks who are vulnerable while also setting the expectation that there could be an oppressive moment that the facilitator and allies have a responsibility to address.

C
Categorization | The natural cognitive process of grouping and labeling people, things, etc. based on their similarities. Categorization becomes problematic when the groupings become oversimplified and rigid (e.g. stereotypes).
Cis-Gender | A person who identifies as the gender they were assigned at birth.
Cis-Sexism | Oppression based assumption that transgender identities and sex embodiments are less legitimate than cis-gender ones.
Class-ism | Prejudiced thoughts and discriminatory actions based on a difference in socioeconomic status, income, class; usually by upper classes against lower.
Coalition | A collection of different people or groups, working toward a common goal.
Codification | The capture and expression of a complex concept in a simple symbol, sign or prop; for example, symbolizing “community” (equity, connection, unity) with a circle.
Collusion | Willing participation in the discrimination against and/or oppression of one’s own group (e.g., a woman who enforces dominant body ideals through her comments and actions).
Colonization | The action or process of settling among and establishing control over the indigenous people of an area. The action of appropriating a place or domain for one's own use.
Color Blind | The belief in treating everyone “equally” by treating everyone the same; based on the presumption that differences are by definition bad or problematic, and therefore best ignored (i.e., “I don’t see race, gender, etc.”).
Color-ism | A form of prejudice or discrimination in which people are treated differently based on the social meanings attached to skin color.
Co-Option | A process of appointing members to a group, or an act of absorbing or assimilating.
Co-Optation | Various processes by which members of the dominant cultures or groups assimilate members of target groups, reward them, and hold them up as models for other members of the target groups. Tokenism is a form of co-optation.
Conscious Bias (Explicit Bias) | Refers to the attitudes and beliefs we have about a person or group on a conscious level. Much of the time, these biases and their expression arise as the direct result of a perceived threat. When people feel threatened, they are more likely to draw group boundaries to distinguish themselves from others.
Critical Race Theory | Critical race theory in education challenges the dominant discourse on race and racism as they relate to education by examining how educational theory, policy, and practice are used to subordinate certain racial and ethnic groups. There are at least five themes that form the basic perspectives, research methods, and pedagogy of critical race theory in education:
1. The centrality and intersectionality of race and racism
2. The challenge to dominant ideology
3. The commitment to social justice
4. The centrality of experiential knowledge
5. The interdisciplinary perspective
Culture | Culture is the pattern of daily life learned consciously and unconsciously by a group of people. These patterns can be seen in language, governing practices, arts, customs, holiday celebrations, food, religion, dating rituals, and clothing.
Cultural Appropriation | The adoption or theft of icons, rituals, aesthetic standards, and behavior from one culture or subculture by another. It is generally applied when the subject culture is a minority culture or somehow subordinate in social, political, economic, or military status to appropriating culture. This “appropriation” often occurs without any real understanding of why the original culture took part in these activities, often converting culturally significant artifacts, practices, and beliefs into “meaningless” pop-culture or giving them a significance that is completely different/less nuanced than they would originally have had.
Culturally Responsive Pedagogy | Culturally responsive pedagogy facilitates and supports the achievement of all students. In a culturally responsive classroom, reflective teaching and learning occur in a culturally supported, learner-centered context, whereby the strengths students bring to school are identified, nurtured and utilized to promote student achievement.

D
D.A.C.A (Deferred Action for Childhood Arrivals) | An American immigration policy that allows some individuals who were brought to the United States without inspection as children to receive a renewable two-year period of deferred action from deportation and become eligible for a work permit in the U.S.
Drag Queen / King | A man or woman dressed as the opposite gender, usually for the purpose of performance or entertainment. Many times, overdone or outrageous and may present a “stereotyped image.”
Dialogue | "Communication that creates and recreates multiple understandings” (Wink, 1997). It is bi-directional, not zero‐sum and may or may not end in agreement. It can be emotional and uncomfortable, but is safe, respectful and has greater understanding as its goal.
Disability | An impairment that may be cognitive, developmental, intellectual, mental, physical, sensory, or some combination of these. It substantially affects a person's life activities and may be present from birth or occur during a person's lifetime.
Discrimination | The denial of justice and fair treatment by both individuals and institutions in many areas, including employment, education, housing, banking, and political rights. Discrimination is an action that can follow prejudiced thinking.
Diversity | The wide variety of shared and different personal and group characteristics among human beings.
Domestic Partner | Either member of an unmarried, cohabiting, straight and same-sex couple that seeks benefits usually available only to spouses.
Dominant Culture | The cultural values, beliefs, and practices that are assumed to be the most common and influential within a given society.

E
Ethnicity | A social construct which divides individuals into smaller social groups based on characteristics such as a shared sense of group membership, values, behavioral patterns, language, political and economic interests, history and ancestral geographical base.
Examples of different ethnic groups are but not limited to:
Haitian
African American (Black)
Chinese
Korean
Vietnamese (Asian)
Cherokee, Mohawk
Navajo (Native American)
Cuban
Mexican
Puerto Rican (Latino)
Polish
Irish
Swedish (White)
Ethnocentricity | Considered by some to be an attitude that views one’s own culture as superior. Others cast it as “seeing things from the point of view of one’s own ethnic group” without the necessary connotation of superiority.
Euro-Centric | The inclination to consider European culture as normative. While the term does not imply an attitude of superiority (since all cultural groups have the initial right to understand their own culture as normative), most use the term with a clear awareness of the historic oppressiveness of Eurocentric tendencies in U.S and European society.
Equality | A state of affairs in which all people within a specific society or isolated group have the same status in certain respects, including civil rights, freedom of speech, property rights and equal access to certain social goods and services.
Equity | Takes into consideration the fact that the social identifiers (race, gender, socioeconomic status, etc.) do, in fact, affect equality. In an equitable environment, an individual or a group would be given what was needed to give them equal advantage. This would not necessarily be equal to what others were receiving. It could be more or different. Equity is an ideal and a goal, not a process. It insures that everyone has the resources they need to succeed.

F
Feminism | The advocacy of women's rights on the ground of the equality of the sexes.
Femme | A person who expresses and/or identifies with femininity.
First Nation People | Individuals who identify as those who were the first people to live on the Western Hemisphere continent. People also identified as Native Americans.
Fundamental Attribution Error | A common cognitive action in which one attributes their own success and positive actions to their own innate characteristics ('I’m a good person') and failure to external influences ('I lost it in the sun'), while attributing others' success to external influences ('He had help and got lucky') and failure to others’ innate characteristics ('They’re bad people'). This operates on group levels as well, with the in-group giving itself favorable attributions, while giving the out-group unfavorable attributions, as a way of maintaining a feeling of superiority, i.e. “double standard.”.

G
Gay | A person who is emotionally, romantically or sexually attracted to members of the same gender.
Gender | The socially constructed concepts of masculinity and femininity; the “appropriate” qualities accompanying biological sex.
Gender Bending | Dressing or behaving in such a way as to question the traditional feminine or masculine qualities assigned to articles of clothing, jewelry, mannerisms, activities, etc.
Gender Dysphoria (Gender Identity Disorder) | Significant, clinical distress caused when a person’s assigned birth gender is not the same as the one with which they identify. The American Psychiatric Association’s Diagnostic and Statistical Manual of Mental Disorders (DSM) consider Gender Identity Disorder as “intended to better characterize the experiences of affected children, adolescents, and adults.”
Gender Expression | External manifestations of gender, expressed through a person's name, pronouns, clothing, haircut, behavior, voice, and/or body characteristics.
Gender Fluid | A person who does not identify with a single fixed gender; of or relating to a person having or expressing a fluid or unfixed gender identity.
Gender Identity | Your internal sense of self; how you relate to your gender(s).
Gender Non-Conforming | A broad term referring to people who do not behave in a way that conforms to the traditional expectations of their gender, or whose gender expression does not fit into a category.
Gender Queer | Gender queer people typically reject notions of static categories of gender and embrace a fluidity of gender identity and often, though not always, sexual orientation. People who identify as “gender queer” may see themselves as both male or female aligned, neither male or female or as falling completely outside these categories.

H
Hate Crime | Hate crime legislation often defines a hate crime as a crime motivated by the actual or perceived race, color, religion, national origin, ethnicity, gender, disability, or sexual orientation of any person.
Hermaphrodite | An individual having the reproductive organs and many of the secondary sex characteristics of both sexes. (Not a preferred term. See: Intersex)
Hetero-sexism | The presumption that everyone is, and should be, heterosexual.
Heterosexuality | An enduring romantic, emotional and/or sexual attraction toward people of the other sex. The term “straight” is commonly used to refer to heterosexual people.
Heterosexual | Attracted to members of other or the opposite sex.
Homophobia | The fear or hatred of homosexuality (and other non‐heterosexual identities), and persons perceived to be gay or lesbian.
Homosexual | Attracted to members of the same sex. (Not a preferred term. See Gay, Lesbian)
Humility | A modest or low view of one's own importance; humbleness.

I
Impostor Syndrome | Refers to individuals' feelings of not being as capable or adequate as others. Common symptoms of the impostor phenomenon include feelings of phoniness, self-doubt, and inability to take credit for one's accomplishments. The literature has shown that such impostor feelings influence a person's self-esteem, professional goal directed-ness, locus of control, mood, and relationships with others.
Inclusion | Authentically bringing traditionally excluded individuals and/or groups into processes, activities, and decision/policy making in a way that shares power.
Inclusive Language | Refers to non-sexist language or language that “includes” all persons in its references. For example, “a writer needs to proofread his work” excludes females due to the masculine reference of the pronoun. Likewise, “a nurse must disinfect her hands” is exclusive of males and stereotypes nurses as females.
In-Group Bias ( Favoritism ) | The tendency for groups to “favor” themselves by rewarding group members economically, socially, psychologically, and emotionally in order to uplift one group over another.
Institutional Racism | It is widely accepted that racism is, by definition, institutional. Institutions have greater power to reward and penalize. They reward by providing career opportunities for some people and foreclosing them for others. They reward as well by the way social goods are distributed, by deciding who receives institutional benefits.
Intercultural Competency | A process of learning about and becoming allies with people from other cultures, thereby broadening our own understanding and ability to participate in a multicultural process. The key element to becoming more culturally competent is respect for the ways that others live in and organize the world and an openness to learn from them.
Inter-Group Conflict | Tension and conflict which exists between social groups and which may be enacted by individual members of these groups.
Internalized Homophobia | Among lesbians, gay men, and bisexuals, internalized sexual stigma (also called internalized homophobia) refers to the personal acceptance and endorsement of sexual stigma as part of the individual's value system and self-concept. It is the counterpart to sexual prejudice among heterosexuals.
Internalized Oppression | The process whereby individuals in the target group make oppression internal and personal by coming to believe that the lies, prejudices, and stereotypes about them are true. Members of target groups exhibit internalized oppression when they alter their attitudes, behaviors, speech, and self-confidence to reflect the stereotypes and norms of the dominant group. Internalized oppression can create low self-esteem, self-doubt, and even self-loathing. It can also be projected outward as fear, criticism, and distrust of members of one’s target group.
Internalized Racism | When individuals from targeted racial groups internalize racist beliefs about themselves or members of their racial group. Examples include using creams to lighten one’s skin, believing that white leaders are inherently more competent, asserting that individuals of color are not as intelligent as white individuals, believing that racial inequality is the result of individuals of color not raising themselves up “by their bootstraps”. (Jackson & Hardiman, 1997)
Intersectionality | An approach largely advanced by women of color, arguing that classifications such as gender, race, class, and others cannot be examined in isolation from one another; they interact and intersect in individuals’ lives, in society, in social systems, and are mutually constitutive. Exposing [one’s] multiple identities can help clarify the ways in which a person can simultaneously experience privilege and oppression. For example, a Black woman in America does not experience gender inequalities in exactly the same way as a white woman, nor racial oppression identical to that experienced by a Black man. Each race and gender intersection produces a qualitatively distinct life.
Intersex | An umbrella term describing people born with reproductive or sexual anatomy and/or chromosome pattern that can't be classified as typically male or female.
ISM | A social phenomenon and psychological state where prejudice is accompanied by the power to systemically enact it.

L
Lesbian | A woman who is attracted to other women. Also used as an adjective describing such women.
LGBTQIA+ | Acronym encompassing the diverse groups of lesbians, gay, bisexual, transgender, intersex, and asexual and/or corresponding queer alliances/associations. It is a common misconception that the "A" stands for allies/ally. The full acronym is "Lesbian, Gay, Bisexual, Transgender, Queer, Intersex, Asexual, with all other queer identities that are not encompassed by the letters themselves being represented by the "+".
Lines of Difference | A person who operates across lines of difference is one who welcomes and honors perspectives from others in different racial, gender, socioeconomic, generational, regional groups than their own. [Listing is not exhaustive]
Look-ism | Discrimination or prejudice based upon an individual’s appearance.

M
Marginalized | Excluded, ignored, or relegated to the outer edge of a group/society/community.
Micro-Aggressions | Commonplace daily verbal, behavioral, or environmental indignities, whether intentional or unintentional, that communicate hostile, derogatory racial slights. These messages may be sent verbally, ("You speak good English"), non-verbally (clutching one's purse more tightly around people from certain race/ethnicity) or environmentally (symbols like the confederate flag or using Native American mascots). Such communications are usually outside the level of conscious awareness of perpetrators.
Micro-Insults | Verbal and nonverbal communications that subtly convey rudeness and insensitivity and demean a person's racial heritage or identity. An example is an employee who asks a colleague of color how she got her job, implying she may have landed it through an affirmative action or quota system.
Micro-Invalidation | Communications that subtly exclude, negate or nullify the thoughts, feelings or experiential reality of a person of color. For instance, white individuals often ask Asian-Americans where they were born, conveying the message that they are perpetual foreigners in their own land.
Model Minority | Refers to a minority ethnic, racial, or religious group whose members achieve a higher degree of success than the population average. This success is typically measured in income, education, and related factors such as low crime rate and high family stability.
Mono-Racial | To be of only one race (composed of or involving members of one race only; (of a person) not of mixed race.)
Multi-Cultural | This term is used in a variety of ways and is less often defined by its users than terms such as multiculturalism or multicultural education.
One common use of the term refers to the raw fact of cultural diversity: “multicultural education … responds to a multicultural population.” Another use of the term refers to an ideological awareness of diversity: “[multicultural theorists] have a clear recognition of a pluralistic society.” Still others go beyond this and understand multicultural as reflecting a specific ideology of inclusion and openness toward “others.” Perhaps the most common use of this term in the literature is in reference simultaneously to a context of cultural pluralism and an ideology of inclusion or “mutual exchange of and respect for diverse cultures.”

When the term is used to refer to a group of persons (or an organization or institution), it most often refers to the presence of and mutual interaction among diverse persons (in terms of race, class, gender, and so forth) of significant representation in the group. In other words, a few African Americans in a predominantly European American congregation would not make the congregation “multicultural.” Some, however, do use the term to refer to the mere presence of some non-majority persons somewhere in the designated institution (or group or society), even if there is neither significant interaction nor substantial numerical representation.
Multi-Cultural Feminism | The advocacy of women's rights on the ground of the equality of the sexes within cultural/ethnic groups within a society.
Multi-Ethnic | An individual that comes from more than one ethnicity. An individual whose parents are born with more than one ethnicity.
Multiplicity | The quality of having multiple, simultaneous social identities (e.g., being male and Buddhist and working-class).
Multi-Racial | An individual that comes from more than one race.

N
Naming | When one articulates a thought that traditionally has not been discussed.
National Origin | The political state from which an individual hails; may or may not be the same as that person's current location or citizenship.
Neo-Liberalism | A substantial subjugation and marginalization of policies and practices informed by the values of social justice and equity.
Non-Binary/Gender Queer/Gender Variant | Terms used by some people who experience their gender identity and/or gender expression as falling outside the categories of man and woman.
Non-White | Used at times to reference all persons or groups outside of the white culture, often in the clear consciousness that white culture should be seen as an alternative to various non-white cultures and not as normative.

O
Oppression | Results from the use of institutional power and privilege where one person or group benefits at the expense of another. Oppression is the use of power and the effects of domination.

P
Pan-Sexual | A term referring to the potential for sexual attractions or romantic love toward people of all gender identities and biological sexes. The concept of pan-sexuality deliberately rejects the gender binary and derives its origin from the transgender movement.
Persons of Color | A collective term for men and women of Asian, African, Latin and Native American backgrounds; as opposed to the collective "White" for those of European ancestry.
Personal Identity | Our identities as individuals including our personal characteristics, history, personality, name, and other characteristics that make us unique and different from other individuals.
Prejudice | A prejudgment or preconceived opinion, feeling, or belief, usually negative, often based on stereotypes, that includes feelings such as dislike or contempt and is often enacted as discrimination or other negative behavior; OR, a set of negative personal beliefs about a social group that leads individuals to prejudge individuals from that group or the group in general, regardless of individual differences among members of that group.
Privilege | Unearned access to resources (social power) only readily available to some individuals as a result of their social group.
Privileged Group Member | A member of an advantaged social group privileged by birth or acquisition, i.e. Whites, men, owning class, upper-middle-class, heterosexuals, gentiles, Christians, non-disabled individuals.
Post-Racial | A theoretical term to describe an environment free from racial preference, discrimination, and prejudice.

Q
Queer | An umbrella term that can refer to anyone who transgresses society's view of gender or sexuality. The definition indeterminacy of the word Queer, its elasticity, is one of its constituent characteristics: "A zone of possibilities."
Questioning | A term used to refer to an individual who is uncertain of their sexual orientation or identity.

R
Race | A social construct that artificially divides individuals into distinct groups based on characteristics such as physical appearance (particularly skin color), ancestral heritage, cultural affiliation or history, ethnic classification, and/or the social, economic, and political needs of a society at a given period of time. Scientists agree that there is no biological or genetic basis for racial categories.
Racial Equity | Racial equity is the condition that would be achieved if one's racial identity is no longer predicted, in a statistical sense, how one fares. When this term is used, the term may imply that racial equity is one part of racial justice, and thus also includes work to address the root causes of inequities, not just their manifestations. This includes the elimination of policies, practices, attitudes and cultural messages that reinforce differential outcomes by race or fail to eliminate them.
Racial Profiling | The use of race or ethnicity as grounds for suspecting someone of having committed an offense.
Racism | Prejudiced thoughts and discriminatory actions based on a difference in race/ethnicity; usually by white/European descent groups against persons of color. Racism is racial prejudice plus power. It is the intentional or unintentional use of power to isolate, separate and exploit others. The use of power is based on a belief in superior origin, the identity of supposed racial characteristics. Racism confers certain privileges on and defends the dominant group, which in turn, sustains and perpetuates racism.
Rainbow Flag | The Rainbow Freedom Flag was designed in 1978 by Gilbert Baker to designate the great diversity of the LGBTIQ community. It has been recognized by the International Flag Makers Association as the official flag of the LGBTIQ civil rights movement.
Re-Fencing (Exception-Making) | A cognitive process for protecting stereotypes by explaining any evidence/example to the contrary as an isolated exception.
Religion | A system of beliefs, usually spiritual in nature, and often in terms of a formal, organized denomination.
Resilience | The ability to recover from some shock or disturbance

S
Safe Space | Refers to an environment in which everyone feels comfortable expressing themselves and participating fully, without fear of attack, ridicule or denial of experience.
Safer Space | A supportive, non-threatening environment that encourages open-mindedness, respect, a willingness to learn from others, as well as physical and mental safety.
Saliency | The quality of a group identity in which an individual is more conscious, and plays a larger role in that individual's day‐to‐day life; for example, a man's awareness of his "maleness" in an elevator with only women.
Scapegoating | The action of blaming an individual or group for something when, in reality, there is no one person or group responsible for the problem. It targets another person or group as responsible for problems in society because of that person’s group identity.
Sex | Biological classification of male or female (based on genetic or physiological features); as opposed to gender.
Sexism | Prejudiced thoughts and discriminatory actions based on a difference in sex/gender; usually by men against women.
Sexual Orientation | One's natural preference in sexual partners; examples include homosexuality, heterosexuality, or bisexuality. Sexual orientation is not a choice, it is determined by a complex interaction of biological, genetic, and environmental factors.
Social Identity | Involves the ways in which one characterizes oneself, the affinities one has with other people, the ways one has learned to behave in stereotyped social settings, the things one values in oneself and in the world, and the norms that one recognizes or accepts governing everyday behavior.
Social Identity Development | The stages or phases that a person's group identity follows as it matures or develops.
Social Justice | A broad term for action intended to create genuine equality, fairness, and respect among peoples.
Social Oppression | This condition exists when one social group, whether knowingly or unconsciously, exploits another group for its own benefit.
Social Self-Esteem | The degree of positive/negative evaluation an individual holds about their particular situation in regard to their social identities.
Social Self-View | An individual's perception about which social identity group(s) they belong.
Stereotype | Blanket beliefs and expectations about members of certain groups that present an oversimplified opinion, prejudiced attitude, or uncritical judgment. They go beyond necessary and useful categorizations and generalizations in that they are typically negative, are based on little information and are highly generalized.
System of Oppression | Conscious and unconscious, non‐random, and organized harassment, discrimination, exploitation, discrimination, prejudice and other forms of unequal treatment that impact different groups.

T
Tolerance | Acceptance, and open‐mindedness to different practices, attitudes, and cultures; does not necessarily mean agreement with the differences.
Token-ism | Hiring or seeking to have representation such as a few women and/or racial or ethnic minority persons so as to appear inclusive while remaining mono-cultural.
Transgender/Trans | An umbrella term for people whose gender identity differs from the sex they were assigned at birth. The term transgender is not indicative of gender expression, sexual orientation, hormonal makeup, physical anatomy, or how one is perceived in daily life.
Transgressive | Challenging the accepted expectations and/or rules of the appropriateness of “polite society”.
Trans Misogyny | The negative attitudes, expressed through cultural hate, individual and state violence, and discrimination directed toward trans women and transfeminine people.
Transphobia | Fear or hatred of transgender people; transphobia is manifested in a number of ways, including violence, harassment, and discrimination. This phobia can exist in LGB and straight communities.
Transexual | One who identifies as a gender other than that of their biological sex.
Two Spirit | An umbrella term for a wide range of non-binary culturally recognized gender identities and expressions among Indigenous people.
A Native American term for individuals who identify both as male and female. In western culture, these individuals are identified as lesbian, gay, bi‐sexual or trans-gendered.

U
Unconscious Bias (Implicit Bias) | Social stereotypes about certain groups of people that individuals form outside their own conscious awareness. Everyone holds unconscious beliefs about various social and identity groups, and these biases stem from one’s tendency to organize social worlds by categorizing.
Undocumented | A foreign-born person living in the United States without legal citizenship status.
Undocumented Student | School-aged immigrants who entered the United States without inspection/overstayed their visas and are present in the United States with or without their parents. They face unique legal uncertainties and limitations within the United States educational system.

V
Veteran Status | Whether or not an individual has served in a nation's armed forces (or other uniformed service).

W
Whiteness | A broad social construction that embraces the white culture, history, ideology, racialization, expressions, and economic, experiences, epistemology, and emotions and behaviors and nonetheless reaps material, political, economic, and structural benefits for those socially deemed white.
White Fragility | Discomfort and defensiveness on the part of a white person when confronted by information about racial inequality and injustice.
White Privilege | White Privilege is the spillover effect of racial prejudice and White institutional power. It means, for example, that a White person in the United States has privilege, simply because one is White. It means that as a member of the dominant group a White person has greater access or availability to resources because of being White. It means that White ways of thinking and living are seen as the norm against which all people of color are compared. Life is structured around those norms for the benefit of White people. White privilege is the ability to grow up thinking that race doesn’t matter. It is not having to daily think about skin color and the questions, looks, and hurdles that need to be overcome because of one’s color. White Privilege may be less recognizable to some White people because of gender, age, sexual orientation, economic class or physical or mental ability, but it remains a reality because of one’s membership in the White dominant group.
White Supremacy | White supremacy is a historically based, institutionally perpetuated system of exploitation and oppression of continents, nations and individuals of color by white individuals and nations of the European continent for the purpose of maintaining and defending a system of wealth, power and privilege.
Worldview | The perspective through which individuals view the world; comprised of their history, experiences, culture, family history, and other influences.

X
Xenophobia | Hatred or fear of foreigners/strangers or of their politics or culture.


Diversity terms #4 (이거는 PDF에서 긁어와서 좀 enter가 이상하게 쳐진것 같이 보여요.
https://www.pacificu.edu/life-pacific/support-safety/office-equity-diversity-inclusion/glossary-terms

Acculturation Difficulty - A problem stemming from an inability to appropriately
adapt to a different culture or environment. The problem is not based on any coexisting
mental disorder.
Achieved Status - Social status and prestige of an individual acquired as a result of
individual accomplishments (cf. ascribed status).
Adaptation - is a process of reconciliation and of coming to terms with a changed sociocultural environment by making "adjustments" in one's cultural identity. It is also a stage
of intercultural sensitivity, which may allow the person to function in a bicultural
capacity. In this stage, a person is able to take the perspective of another culture and
operate successfully within that culture. The person should know enough about his or her
own culture and a second culture to allow a mental shift into the value scheme of the other
culture, and an evaluation of behaviour based on its norms, rather than the norms of the
individual's culture of origin. This is referred to as "cognitive adaptation." The more
advanced form of adaptation is "behavioural adaptation," in which the person can
produce behaviours appropriate to the norms of the second culture. Adaptation may also
refer to patterns of behavior which enable a culture to cope with its surroundings.
Adaptation Level - Individual standards of comparison for evaluating properties of
physical and social environment such as crowding and noise.
Advocacy View - of applied anthropology is the belief that as anthropologists have
acquired expertise on human problems and social change, and because they study,
understand, and respect cultural values, they should be responsible for making policies
affecting people.
Affirmative Action - "Affirmative action" refers to positive steps taken to increase the
representation of minorities (racial, ethnic minorities and women in general) in areas of
employment, education, and business from which they have been historically excluded.
Age Discrimination - is discrimination against a person or group on the basis of age. Age
discrimination usually comes in one of two forms: discrimination against youth, and
discrimination against the elderly.
Age Set - Group uniting all men or women born during a certain historical time span.
Aggregate - Any collection of individuals who do not interact with one another.
Alternative Medicine - Any form of medicine or healthcare practices which are not
within the jurisdiction of the official health care delivery system nor legally sanctioned.
Ambient Environment - Changeable aspects of an individual's immediate surroundings,
e.g., light, sounds, air quality, humidity, temperature etc.
Ambient Stressors - Factors in the environment that contributes to the experience of
stress.
Anchor - A reference point for making judgments. In social judgment theory, anchor is
the point corresponding to the centre of the latitude of acceptance.
Animism - Is the belief that souls inhabit all or most objects. Animism attributes
personalized souls to animals, vegetables, and minerals in a manner that the material 
object is also governed by the qualities which compose its particular soul. Animistic
religions generally do not accept a sharp distinction between spirit and matter.
Anthropology - The study of the human species and its immediate ancestors.
Anthropology is the comparative study of past and contemporary cultures, focusing on
the ways of life, and customs of all peoples of the world. Main sub-disciplines are
physical anthropology, archaeology, linguistic anthropology, ethnology (which is also
called social or cultural anthropology) and theoretical anthropology, and applied
anthropology.
Apartheid - was a system of racial segregation used in South Africa from 1948 to the
early 1990s. Though first used in 1917 by Jan Smuts, the future Prime Minister of South
Africa, apartheid was simply an extension of the segregationist policies of previous white
governments in South Africa. The term originates in Afrikaans or Dutch, where it means
separateness. Races, classified by law into White, Black, Indian, and Coloured groups,
were separated, each with their own homelands and institutions. This prevented nonwhite people from having a vote or influence on the governance. Education, medical care
and other public services available to non-white people were vastly inferior and nonwhites were not allowed to run businesses or professional practices in those areas
designated as 'White South Africa'.
Arbitration - Third-party assistance to two or more groups for reaching an agreement,
where the third party or arbitrary has the power to force everyone to accept a particular
solution.
Arranged Marriage - Any marriage in which the selection of a spouse is outside the
control of the bride and groom. Usually parents or their representatives select brides or
grooms by trying to match compatibility rather than relying on romantic attraction.
Ascribed Status - Social status which is the re A concept that originated with the Maori
of New Zealand, that focuses on culturally ?appropriate health care services, as well as
improving healthcare access, inequalities in health, unequal power relations, and the social,
political, and historical context of care
Assimilation - is a process of consistent integration whereby members of an ethnocultural group, typically immigrants, or other minority groups, are "absorbed" into an
established larger community. If a child assimilates into a new culture, he/she gives up
his/her cultural values and beliefs and adopts the new cultural values in their place.
Originates from a Piagetian (Swiss Developmental Psychologist JEAN PIAGET, 1896-
1980) term describing a person's ability to comprehend and integrate new experiences.
Assimilation Effects - Shifts in judgments towards an anchor point in social judgment
theory.
Attachment Theory - A theory of the formation and characterization of relationships
based on the progress and outcome of an individual's experiences as an infant in relation to
the primary caregiver.
Attitude - Evaluation of people, objects, or issues about which an individual has some
knowledge.
Availability Heuristic - The tendency to be biased by events readily accessible in our
memory.

B
Baak Gwai - A derogatory term meaning "White devil" or "white ghost" used by the
Chinese in Mainland China and Hong Kong to refer to Caucasians.
Banana - Derogatory term for an East Asian person who is "yellow on the outside, white
on the inside" used by other Asian Americans to indicate someone who has lost touch
with their cultural identity and have over-assimilated in white, American culture.
Band - Basic unit of social organization among foragers. A band includes fewer than 100
people; it often splits up seasonally.
Belief System - is the way in which a culture collectively constructs a model or
framework for how it thinks about something. A religion is a particular kind of belief
system. Other examples of general forms of belief systems are ideologies, paradigms and
world-views also known by the German word Weltanschauung. In addition to governing
almost all aspects of human activity, belief systems have a significant impact on what a
culture deems worthy of passing down to following generations as its cultural heritage.
This also influences how cultures view the cultural heritage of other cultures. Many
people today recognize that there is no one corrects belief system or way of thinking.
This is known as relativism or conceptual relativism. This contrasts with objectivism and
essentialism, both of which posit a reality that is independent of the way in which people
conceptualize. A plurality of belief systems is a hallmark of postmodernism.
Belief in a Just World - The tendency of people to want to believe that the world is
just so that when they witness an otherwise inexplicable injustice they will rationalize
it by searching for things that the victim might have done to deserve it. Also called the
just-world theory, just-world fallacy, just-world effect, or just-world hypothesis, Famous
proponent is Melvin Lerner.
Biculturalism - The simultaneous identification with two cultures when an individual
feels equally at home in both cultures and feels emotional attachment with both cultures.
The term started appearing in the 1950s.
Biethnic - Of two ethnic groups: belonging or relating to two different ethnic groups.
Usually, used in reference to a person. For example: if a person's father is French and
mother English, she is biethnic though not biracial. See also biracial.
Bilingual Education - teaching a second language by relying heavily on the native
language of the speaker. The background theory claims that a strong sense of one's one
culture and language is necessary to acquire another language and culture.
Bilateral Kinship Calculation - is a system in which kinship ties are calculated equally
through both sexes: mother and father, sister and brother, daughter and son, and so on.
Biological Determinists - are those who argue that human behaviour and social
organization are biologically determined and not learnt.
Biracial - Of two races. Usually, used to refer to people whose parents come from two
different races, e.g., father is Chinese and mother English.
Bottom-up Development - Economic and social changes brought about by activities of
individuals and social groups in society rather than by the state and its agents.
30 | P a g e
Bride Price - is the payment made by a man to the family from whom he takes a
daughter in marriage.
C
Complementary Medicine - Traditional or alternative health beliefs or practices which
are brought into a healing practice to enhance the dominant healthcare modality.
Corporate Culture - The fundamental philosophy of an organization is determined by
its corporate culture. The behavior and actions of individuals within a corporation
illustrate the existing culture of that organization.
Capital - Wealth or resources invested in business, with the intent of producing a profit
for the owner of the capital.
Capitalist World Economy- The single world system, committed to production for sale,
with the object of maximizing profits rather than supplying domestic needs. The term
was launched by the US historical social scientist, Immanuel Wallenstein.
Capitalism - Economic or socio-economic system in which production and distribution
are designed to accumulate capital and create profit. A characteristic feature of the system
is the separation of those who own the means of production and those who work for
them. The Communist Manifesto by Karl Marx and Friedrich Engels first used the term
Kapitalist in 1848. The first use of the word capitalism is by novelist William Thackeray
in 1854.
Caste System - Hereditary system of stratification. Hierarchical social status is ascribed
at birth and often dictated by religion or other social norms. Today, it is most commonly
associated with the Indian caste system and the Varna in Hinduism.
Charlie - Non-derogatory slang term used by American troops during the Vietnam War
as a shorthand term for Vietnamese guerrillas. Shortened from "Victor Charlie", the
phonetic alphabet for Viet Cong, or VC. It was also a mildly derogatory term used by
African Americans, in the 1960s and 1970s, for a white person (from James Baldwin's
novel, Blues for Mr. Charlie).
Chiefdom - Kin-based form of sociopolitical organization between the tribe and the state.
It comes with differential access to resources and a permanent political structure. The
relations among villages as well as among individuals are unequal, with smaller villages
under the authority of leaders in larger villages; it has a two-level settlement hierarchy.
Clan - Form of unilateral descent group based on stipulated descent. A clan is a group of
people united by kinship and descent, which is defined by perceived descent from a
common ancestor. As kinship based bonds can be merely symbolical in nature some clans
share a "stipulated" common ancestor.
Clash of Civilizations - is a hotly debated theory publicized by Samuel P. Huntington
with his 1996 book The Clash of Civilizations and the Remaking of World Order. He
argues that the world has cultural fault lines similar to the physical ones that cause
earthquakes and that people's cultural/religious identity will be the primary agent of
conflict in the post-Cold War world. Bernard Lewis first used the term in an article in the
September 1990 issue of The Atlantic Monthly called "The Roots of Muslim Rage."
31 | P a g e
Collateral Household - is a type of expanded family household including siblings and
their spouses and children.
Collectivism - Individualism/Collectivism is one of the Hofstede dimensions in
intercultural communication studies. "Collectivism pertains to societies in which people
from birth onwards are integrated into strong, cohesive in-groups, which throughout
people's lifetime continue to protect them in exchange for unquestioning loyalty."
(Hofstede, G. (1991).
Colonialism - The political, social, economic, and cultural domination of a territory and
its people by a foreign power for an extended time.
Communism - A political theory of Karl Marx and Friedrich Engels. Communism is
characterized by the common ownership of the means of production contra private
ownership in capitalism. The Soviet Union was the first communist state and lasted from
1917 to 1991.
Complex Societies - are usually nation states; large and populous, with social
stratification and centralized forms of governments.
Consanguineal Kin - A blood relative. An individual related by common descent from
the same individual. In most societies of the world, kinship can be traced both by
common descent and through marriage, although a distinction is usually made between the
two categories. The degree of consanguinity between any two people can be calculated as
the percentage of genes they share through common descent.
Contact Zone - The space in which transculturation takes place - where two different
cultures meet and inform each other, often in highly asymmetrical ways.
Core Values - Basic, or central values that integrate a culture and help distinguish it from
others.
Cosmology - Ideas and beliefs about the universe as an ordered system, its origin and the
place of humans in the universe through which, people in that culture understand the
makeup and the workings of all things.
Counterculture - is a sociological term used to describe a cultural or social group whose
values and norms are at odds with those of the social mainstream. The term became
popular during the youth rebellion and unrest in the USA and Western Europe in the
1960s as a reaction against the conservative social norms of the 1950s. The Russian term
Counterculture has a different meaning and is used to define a cultural movement that
promotes acting outside the usual conventions of Russian culture - using explicit language,
graphical description of sex, violence and illicit activities. Counterculture in an Asian
context as launched by Dr. Sebastian Kappen, an Indian Theologian very influential in the
third world, means an approach for navigating between the two opposing cultural
phenomena in modern Asian countries: (1) invasion by western capitalist culture and (2)
the emergence of revivalist movements in reaction. Identification with the first requires
losing own identity and with the second results in living in a world of obsolete myths and
phantoms of the dead past. Thus discovering one's own cultural roots in a creative and
yet critical fashion while being open to the positive facets of the other. (Adapted from
http://www.wikipedia.org)
Cross Cousins - Children of a brother and a sister.
32 | P a g e
Cross Cultural - Interaction between individuals from different cultures. The term crosscultural is generally used to describe comparative studies of cultures. Inter cultural is also
used for the same meaning.
Cross Cultural Awareness - develops from cross-cultural knowledge as the learner
understands and appreciates the deeper functioning of a culture. This may also be
followed by changes in the learner's own behaviour and attitudes and a greater flexibility
and openness becomes visible.
Cross-Cultural Communication - (also referred to as Intercultural Communication) is a
field of study that looks at how people from differing cultural backgrounds try to
communicate. As a science, Cross-cultural communication tries to bring together such
seemingly unrelated disciplines as communication, information theory, learning theories
and cultural anthropology. The aim is to produce increased understanding and some
guidelines, which would help people from different cultures to better, communicate with
each other.
Cross-Cultural Communication Skills - refers to the ability to recognize cultural
differences and similarities when dealing with someone from another culture and also the
ability to recognize features of own behaviour, which are affected by culture.
Cross Cultural Competence - is the final stage of cross-cultural learning and signals the
individual's ability to work effectively across cultures. Cross cultural competency
necessitates more than knowledge,
Cross Cultural Knowledge - refers to a surface level familiarization with cultural
characteristics, values, beliefs and behaviours. It is vital to basic cross-cultural
understanding and without it cross-cultural competence cannot develop.
Cross Cultural Sensitivity - refers to an individual's ability to read into situations,
contexts and behaviours that are culturally rooted and consequently the individual is able
to react to them suitably. A suitable response necessitates that the individual no longer
carries his/her own culturally predetermined interpretations of the situation or behaviour
(i.e. good/bad, right/wrong).
Cultural Alienation - is the process of devaluing or abandoning one's own culture or
cultural background in favour of another.
Cultural Anthropology - The study of contemporary and recent historical cultures
among humans all over the world. The focus is on social organization, culture change,
economic and political systems and religion. Cultural anthropologists argue that culture is
human nature, and that all people have a capacity to classify experiences, encode
classifications symbolically and teach such abstractions to others. They believe that
humans acquire culture through learning and people living in different places or different
circumstances may develop different cultures because it is through culture that people can
adapt to their environment in non-genetic ways. Cultural anthropology is also referred to
as social or socio-cultural anthropology. Key theorists: Franz Boas, Emile Durkheim,
Clifford Geertz, Marvin Harris, Claude Levi-Strauss, Karl Marx.
Cultural Boundaries - Cultural Boundaries can be defined as those invisible lines, which
divide territories, cultures, traditions, practices, and worldviews. Typically they are not
aligned with the physical boundaries of political entities such as nation states.
33 | P a g e
Cultural Components - Attributes that vary from culture to culture, including religion,
language, architecture, cuisine, technology, music, dance, sports, medicine, dress, gender
roles, laws, education, government, agriculture, economy, grooming, values, work ethic,
etiquette, courtship, recreation, and gestures.
Culturally Competent Healthcare - Healthcare practice which recognizes the
importance of cultural beliefs and practices in restoration and maintenance of health, and
thus adapts, modifies and reorients perceptions and practices within a bio-medical setting
in response to the cultural background of the patient.
Cultural Competency - The ability to respond respectfully and effectively to people of
all cultures, classes, ethnic background and religions in a manner that recognizes and
values cultural differences and similarities.
Cultural Construct - the idea that the characteristics people attribute to social categories
such as gender, illness, death, status of women, and status of men is culturally defined.
Cultural Convergence - is an idea that increased communication among the peoples of
the world via the Internet will lead to the differences among national cultures becoming
smaller over time, eventually resulting in the formation of a single global culture. One
outcome of this process is that unique national identities will disappear, replaced by a
single transnational identity. Henry Jenkins, a professor at the Massachusetts Institute of
Technology, USA coined the term in 1998.
Cultural Cringe - refers to an internalized inferiority complex of an entire culture. This
leads people of that culture to dismiss their own culture as inferior to the cultures of other
countries. In 1950 the Melbourne critic A.A.Philips coined the term Cultural cringe to
show how Australians widely assumed that anything produced by local artists,
dramatists, actors, musicians and writers was inferior to the works of the British and
European counterparts. The term cultural cringe is very close to "cultural alienation" or
the process of devaluing or abandoning one's own culture or cultural background in favour
of another.
Cultural Determinists - are those who relate behaviour and social organization to
cultural or environmental factors. The focus is on variation rather than on universals and
stresses learning and the role of culture in human adaptation.
Cultural Diffusion - The spreading of a cultural trait (e.g., material object, idea, or
behaviour pattern) from one society to another.
Cultural Dissonance - Elements of discord or lack of agreement within a culture.
Cultural Diversity - Differences in race, ethnicity, language, nationality or religion.
Cultural diversity refers to the variety or multiformity of human social structures, belief
systems, and strategies for adapting to situations in different parts of the world.
Cultural Evolution - Theories that have developed since the mid-19th century, which
attempt to explain processes and patterns of cultural change. Often such theories have
presented such change as "progress," from "earlier" forms ("primitive", "less
developed," "less advanced" etc.) to "later" forms ("more developed," "more
advanced"). These schemes usually have reflected the ethnocentrism of the theorists, as
they frequently put their own societies at the pinnacle of "progress."
34 | P a g e
Cultural Identity - is the identity of a group or culture, or of an individual as her/his
belonging to a group or culture affects her/his view of her/him. People who feel they
belong to the same culture share a common set of norms.
Cultural Imperialism - is the rapid spread or advance of one culture at the expense of
others, or its imposition on other cultures, which it modifies, replaces, or destroysusually due to economic or political reasons.
Cultural Landscape- The natural landscape as modified by human activities and bearing
the imprint of a culture group or society including buildings, shrines, signage, sports and
recreational facilities, economic and agricultural structures, transportation systems, etc.
Cultural Materialism - Is a theoretical approach in Cultural Anthropology that
explores and examines culture as a reflection or product of material conditions in a
society. Cultural materialism is a variation on basic materialist approaches to
understanding culture. The Anthropologist Marvin Harris is a famous representative.
Cultural Norms - are behaviour patterns that are typical of specific groups, which have
distinct identities, based on culture, language, ethnicity or race separating them from other
groups. Such behaviours are learned early in life from parents, teachers, peers and other
human interaction. Norms are the unwritten rules that govern individual behaviour. Norms
assume importance especially when broken or when an individual finds him/herself in a
foreign environment dealing with an unfamiliar culture where the norms are different.
Cultural Relativism - The position that the values, beliefs and customs of cultures
differ and deserve recognition and acceptance. This principle was established by the
German anthropologist Franz Boas (1858-1942) in the first few decades of the 20th
century. Cultural relativism as a movement was in part a response to Western
ethnocentrism. Between World War I and World War II, "Cultural relativism" was the
central tool for American anthropologists in their refusal of Western claims to
universality.
Cultural Resource Management (CRM) - is the branch of applied archaeology which
aims to preserve archeological sites threatened by prospective dams, highways, and other
projects.
Cultural Rights - is the idea that certain rights are vested not in individuals but in larger
identifiable groups, such as religious and ethnic minorities and indigenous societies.
Cultural rights include a group's ability to preserve its culture, to raise its children in the
ways of its ancestors, to continue practicing its language, and not to be deprived of its
economic base by the nation-state or large political entity in which it is located.
Cultural Safety - A concept that originated with the Maori of New Zealand, that
focuses on culturally ?appropriate health care services, as well as improving healthcare
access, inequalities in health, unequal power relations, and the social, political, and
historical context of care
Cultural Sensitivity - is a necessary component of cultural competence, meaning that
we make an effort to be aware of the potential and actual cultural factors that affect our
interactions with others.
Cultural Traits - Distinguishing features of a culture such as language, dress, religion,
values, and an emphasis on family; these traits are shared throughout that culture.
35 | P a g e
Cultural Universality- General cultural traits and features found in all societies of the
world. Some examples are organization of family life; roles of males, females, children and
elders; division of labour; religious beliefs and practices; birth and death rituals; stories of
creation and myths for explaining the unknown; "rights" and "wrongs" of behaviour etc.
Cultural Universalism - Cultural Universalism is the assertion that there exist values,
which transcend cultural and national differences. Universalism claims that more
primitive cultures will eventually evolve to have the same system of law and rights as
Western cultures. Cultural relativists on the other hand hold an opposite viewpoint, that a
traditional culture is unchangeable. In universalism, an individual is a social unit,
possessing inalienable rights, and driven by the pursuit of self-interest. In the cultural
relativist model, a community is the basic social unit where concepts such as
individualism, freedom of choice, and equality are absent.
Cultural values: The individual's desirable or preferred way of acting or knowing
something that is sustained over time and that governs actions
Culture - The shared values, norms, traditions, customs, arts, history, folklore and
institutions of a group of people. "Integrated pattern of human knowledge, belief, and
behaviour that is both a result of an integral to the human capacity for learning and
transmitting knowledge to succeeding generations." The etymological root of the word is
from the Latin 'colere' which means to cultivate, from which is derived 'cultus', that which
is cultivated or fashioned. In comparison of words such as "Kultur" and "Zivilisation" in
German, "culture" and civilization" in English, and "culture" and "civilization" in French
the concepts reveal very different perspectives. The meaning of these concepts is
however, converging across languages as a result of international contacts, cultural
exchanges and other information processes.
Quotation from source http://www.britannica.com
Culture Shock - A state of distress and tension with possible physical symptoms after
a person relocates to an unfamiliar cultural environment. This term was used by social
scientists in the 1950s to describe, the difficulties of a person moving from the country to
a big city but now the meaning has changed to mean relocating to a different culture or
country. One of the first recorded uses of the term was in 1954 by the anthropologist Dr.
Kalervo Oberg who was born to Finnish parents in British Columbia, Canada. While
giving a talk to the Women's Club of Rio de Janeiro, August 3, 1954, he identified four
stages of culture shock-the honeymoon of being a newcomer and guest, the hostility and
aggressiveness of coming to grips with different way of life, working through feelings of
superiority and gaining ability to operate in the culture by learning the language and
finally acceptance of another way of living and worldview. (Source: American
Anthropologist June, 1974 Vol.76 (2): 357-359.
D
Daughter Languages - are languages developing out of the same parent language; for
example, French and Spanish are daughter languages of Latin or Bengali and Hindi are
daughter languages of Sanskrit.
36 | P a g e
Debriefing - Open discussion at the end of a study or experiment when the researcher
reveals the complete procedure and background to the subject and explains the reasons for
any possible deceptions that may have taken place and were necessary for the success.
Demarginalization - The process which facilitates a marginal or stigmatized space
becoming 'normalized' so that its population is incorporated into the mainstream.
Descent Group - is a permanent social unit whose members claim common ancestry.
Usually this is fundamental to tribal society.
Differential Access - refers to unequal access to resources, which is the basic attribute
of different social structures from chiefdoms and states.
Diffuse - Diffuse/Specific is one of the value dimensions proposed by Trompenaars &
Hampden-Turner (1997). It shows "how far we choose to get involved". In a very diffuse
culture, a large part of the life is regarded as "private", where other persons without
explicit consent have no access.
Diffusion - is the borrowing of cultural traits between societies, either directly or through
intermediaries.
Dimensions of Diversity - Dimensions of diversity in humans includes, but is not
limited to: culture, gender, age, ethnicity, nationality, geography, lifestyle, education,
income, health, physical appearance, pigmentation, language, personality, beliefs, faith,
dreams, interests, aspirations, skills, professions, perceptions, and experiences.
Discrimination - Treatment or consideration based on class or category defined by
prejudicial attitudes and beliefs rather than individual merit. The denial of equal treatment,
civil liberties and opportunities to education, accommodation, health care, employment
and access. In many countries discrimination by law consists of making unjust
distinctions based on:
? Religion, political affiliation, marital or family status
? Age, sexual orientation, gender, race, colour, nationality
? Physical, developmental or mental disability
Diversity - The concept of diversity means understanding that each individual is unique,
and recognizing individual differences along the dimensions of race, ethnicity, gender,
sexual orientation, socio-economic status, age, physical abilities, religious beliefs, political
beliefs, or other ideologies. Primary dimensions are those that cannot be changed e.g., age,
ethnicity, gender, physical abilities/qualities, race and sexual orientation. Secondary
dimensions of diversity are those that can be changed, e.g., educational background,
geographic location, income, marital status, parental status, religious beliefs, and work
role/experiences. Diversity or diversity management includes, therefore, knowing how to
relate to those qualities and conditions that are different from our own and outside the
groups to which we belong.
¶ Diversity Initiative - Sets of policy, definitions, action-plans and steps to map out,
support and protect diversity in different dimensions such as age, gender ethnicity etc in
any organization, society or area.
Dominant Culture - There is usually one "dominant" culture in each area that forms the
basis for defining that culture. This is determined by power and control in cultural
institutions (church, government, education, mass media, monetary systems, and 
37 | P a g e
economics). Often, those in the dominant culture do not see the privilege that accrues to
them by being dominant "norm" and do not identify themselves as being the dominant
culture. Rather, they believe that their cultural norm.
Dowry - A marital exchange in which the wife's family provides substantial gifts of
money, goods or property to the husband's family. The opposite direction, property
given to the bride by the groom, is called dower.
E
Egalitarianism - Affirming, promoting, or characterized by belief in equal political,
economic, social, and civil rights for all people. One of the seven fundamental value
dimensions of Shalom Schwartz measuring how other people are recognized as moral
equals.
Embeddedness - One of the seven fundamental value dimensions of Shalom Schwartz
describing people as part of a collective.
Enculturation - is the process whereby an established culture teaches an individual its
accepted norms and values, by establishing a context of boundaries and correctness that
dictates what is and is not permissible within that society's framework. Enculturation is
learned through communication by way of speech, words, action and gestures. The six
components of culture learnt are: technological, economic, political, interactive, and
ideological and world-view. It is also called socialization. (Conrad Phillip Kottack,
Cultural Anthropology)
Endogamy - is the practice of marrying within one's own social group. Cultures that
practice endogamy require marriage between specified social groups, classes, or
ethnicities. Strictly endogamous communities like the Jews, the Parsees of India and the
Yazidi of Iraq claim that endogamy helps minorities to survive over a long time in societies
with other practices and beliefs. The opposite practice is exogamy.
Equity, Increased - is a reduction in absolute poverty and a fairer or more even
distribution of wealth in a particular society or nation state.
Ethnic Competence - The capacity to function effectively in more than one culture,
requiring the ability to appreciate and understand features of other ethnic groups and
further to interact with people of ethnic groups other than one's own.
Ethnic Group - Group characterized by cultural similarities (shared among members of
that group) and differences (between that group and others). Members of an ethnic group
share beliefs, values, habits, customs, norms, a common language, religion, history,
geography, kinship, and/or race.
Ethnic Slur - Is a term used to insult someone on the basis of ethnicity, race or
nationality. Some derogatory examples are Flip (Western derogatory term used for
Filipinos), Ginzo in US (for Italian Americans), Gweilo ("Foreign devil" or "white
ghost", term used by the Chinese to refer to Westerners), Paki (UK for a South Asian)
etc.
Ethnicity - Belonging to a common group with shared heritage, often linked by race,
nationality and language.
38 | P a g e
Ethnocentrism - Belief in the superiority of one's own ethnic group. Seeing the world
through the lenses of one's own people or culture so that own culture always looks best
and becomes the pattern everyone else should fit into.
Ethnography - A research methodology associated with anthropology and sociology that
systematically tries to describe the culture of a group of people by trying to understand
the natives'/insiders' view of their own world (an emic view of the world).
Ethnology - Cross-cultural comparison or the comparative study of ethnographic data, of
society and of culture
Ethnomusicology - is the comparative study of the music’s of different places of the
world and of music as a central aspect of culture and society.
Ethnosemantics - is the study of meaning attached to specific terms used by members of
a group. Ethnosemantics concentrates on the meaning of categories of reality and folk
taxonomies to the people who use them. (Source: Cultural Anthropology.
A.R.N.Srivastava. Prentice-Hall)
Exogamy - is the custom of marrying outside a specific group to which one belongs.
Some experts hold that the custom of exogamy originated from a scarcity of women,
which forced men to seek wives from other groups, e.g., marriage by capture. Another
viewpoint ascribes the origin of exogamy to totemism, and claim that a religious respect
for the blood of a totemic clan, led to exogamy. The opposite of exogamy is endogamy.
Expatriate - Someone who has left his or her home country to live and work in another
country. When we go to another country to live, we become expatriates or expats for
short.
Extended Family - The relatives of an individual, both by blood and by marriage, other
than its immediate family, such as aunts, uncles, grandparents and cousins, who live in
close proximity and often under one roof. Extended families are very common in
collectivistic cultures. This is the opposite of the nuclear family.
Family of Orientation - Nuclear family in which one is born and grows up.
Family of Procreation - Nuclear family established when one marries and has children.
Feminity - Masculinity/Feminity is one of the Hofstede dimensions. Hofstede defines
this dimension as follows: "femininity pertains to societies in which social agenda roles
overlap (i.e., men and women are supposed be modest, tender, and concerned with the
quality of life)." (Hofstede, 1991, p. 83)
Feudalism - Hierarchical social and political system common in Europe during the
medieval period. The majority of the population was engaged in subsistence agriculture
while simultaneously having an obligation to fulfill certain duties for the landholder. At
the same time the landholder owed various obligations called fealty to his overlord.
First Nation - The indigenous population of Canada, excepting the Inuit or Metis
people. The term came into common usage in the 1980s to refer mostly to Canada's
aboriginal people, most of who live around Ontario and British Columbia.
Flip - Is a Western derogatory term used for Filipinos.
Folk - means 'Of the people', originally coined for European peasants. It refers to the art,
music, and lore of ordinary people, as contrasted with the "high" art or "classic" art of
the European elites.
39 | P a g e
G
Gender Discrimination - Gender discrimination is any action that allows or denies
opportunities, privileges or rewards to a person on the basis of their gender alone. The
term 'glass ceiling' describes the process by which women are barred from promotion by
means of an invisible barrier. In the United States, the Glass Ceiling Commission has
stated that women represent 1.1 per cent of inside directors (those drawn from top
management of the company) on the boards of Fortune 500 companies.
Gender Roles - The tasks and activities that a culture assigns to each sex.
Gender Stereotypes - are oversimplified but strongly held ideas about the
characteristics, roles and behaviour models of males and females.
Gender Stratification - Unequal distribution of rewards (socially valued resources,
power, prestige, and personal freedom) between men and women, depending on their
different positions in a social hierarchy.
Generalized Reciprocity - is the principle that characterizes exchanges between closely
related individuals. As social distance increases, reciprocity becomes balanced and finally
negative.
Genetic Marker - Is a known DNA sequence of the human DNA. Genetic markers can
be used to study the relationship between an inherited disease and its likely genetic cause.
Genitor - Biological father of a child.
Ginzo - Is a US derogatory term to refer to Italian Americans.
Global Culture - One world culture. The earth's inhabitants will lose their individual
cultural diversity and one culture will remain for all the people.
Globalization - A disputed term relating to transformation in the relationship between
space, economy and society. The International Monetary Fund defines globalization as
"the growing economic interdependence of countries worldwide through increasing
volume and variety of cross-border transactions in goods and services, free international
capital flows, and more rapid and widespread diffusion of technology. Meanwhile, The"
International Forum on Globalization defines it as "the present worldwide drive toward a
globalized economic system dominated by supranational corporate trade and banking
institutions that are not accountable to democratic processes or national governments."
Gweilo - A derogatory term meaning "Foreign devil" or "white ghost" used by the
Chinese in South of Mainland China and Hong Kong to refer to Westerners.
H
Helping Behaviour - Prosocial behaviour that benefits others more than the person.
Different from prosocial cooperation, in which mutual benefit is gained.
Hierarchy - One of the seven fundamental value dimensions of Shalom Schwartz
measuring the unequal distribution of power in a culture.
High Context and Low Context Cultures - According to E.T. Hall (1981), all
communication (verbal as well as nonverbal) is contextually bound. What we do or do not 
40 | P a g e
pay attention to is largely dictated by cultural contexting. In low-context cultures, the
majority of the information is explicitly communicated in the verbal message. In highcontext cultures the information is embedded in the context. High- and low-context
cultures also differ in their definition of social and power hierarchies, relationships, work
ethics, business practices, time management. Low-context cultures tend to emphasize the
individual while high-context cultures places more importance on the collective.
Historical Linguistics - also called diachronic linguistics, is the study of how and why
languages change.
Holistic - Emphasizing the importance of the whole and the interdependence of its parts.
Interested in the whole of the human condition: past, present, and future; biology,
society, language, and culture.
Holocultural Analysis - A paradigm of research for testing hypotheses "by means of
correlations found in a worldwide, comparative study whose units of study are entire
societies or cultures, and whose sampling universe is either (a) all known cultures... or (b)
all known primitive tribes" (Naroll, Michik, & Naroll, 1976).
Human Rights - Human rights refers to the basic rights and freedoms to which all
humans irrespective of countries, cultures, politics, languages, skin colour and religions are
entitled. Examples of human rights are the right to life and liberty, freedom of expression,
and equality before the law, the right to participate in culture, the right to work, the right
to hold religious beliefs without persecution, and to not be enslaved, or imprisoned
without charge and the right to education.
Hybridity - Refers to groups as a mixture of local and non-local influences; their character
and cultural attributes is a product of contact with the world beyond a local place. The
term originates from agriculture and has for a long time been strongly related to pejorative
concepts of racism and racial purity from western colonial history.
Hyperdescent - is the practice of determining the lineage of a child of mixed race ancestry
by assigning the child the race of his more socially dominant parent (opposite of
Hypodescent).
Hypodescent - A social rule that automatically places the children of a union or mating
between members of different socioeconomic groups in the less-privileged group. In its
most extreme form in the United States, hypodescent came to be known as the "one drop
rule," meaning that if a person had one drop of black blood, he was considered black. The
opposite of hypodescent is hyperdescent.
I
Imaginary Geographies - The ideas and representations that divide the world into
spaces and areas with specific meanings and associations. These can exist on different
scales e.g. the imaginaries that divide the world into a developed core and less developed
peripheries or the imagined divide between the deprived inner city and the affluent
suburbs. (Sibley)
Imperialism - A policy of extending the rule of a nation or empire over foreign nations or
of taking and holding foreign colonies by forceful conquest.
41 | P a g e
Independent Invention - Appearance of the same cultural trait or pattern in separate
cultures as a result of comparable needs and circumstances.
Indigenized - Adapted or modified to fit the local culture.
Indigenous Peoples - Those peoples native to a particular territory that was later
colonized, particularly by Europeans. Other terms for indigenous peoples include
aborigines, native peoples, first peoples, Fourth World, first nations and autochthonous
(this last term having a derivation from Greek, meaning "sprung from the earth"). The UN
Permanent Forum on Indigenous Issues estimates range from 300 million to 350 million as
of the start of the 21st century or just fewer than 6 per cent of the total world
population. This includes at least 5000 distinct peoples in over 72 countries.
Individualism - Individualism/Collectivism is one of the Hofstede dimensions in
intercultural communication studies. He defines this dimension as: "individualism pertains
to societies in which the ties between individuals are loose: everyone is expected to look
after himself or herself and his or her immediate family." (Hofstede, 1991, p.51)
International Culture - Cultural traditions that extend beyond the boundaries of nation
states.
Integration - The bringing of people of different racial or ethnic groups into unrestricted
and equal association, as in society or an organization; desegregation. An individual
integrates when s/he becomes a part of the existing society.
Interpretive Approach in Cultural Anthropology - Regards culture as "texts," to be
read and translated for their "thick" meaning. Clifford Geertz is an example of those who
represents this approach.
Islamophobia - Fear and dread of Islam, which has been increasing particularly since
September 11th 2001. The Runnymede Trust in 1997 identified 'closed' and 'open' views
of Islam. Closed views see Islam as static and unchanging, as primitive, sexist, aggressive,
and threatening. Closed views of Islam see hostility towards Muslims as 'normal' and are
used to justify discrimination because no common values with other religions are
admitted. Central to closed views, or 'Islamophobia', and propagated by the Western
media, is the assumption that all Muslims support all actions taken in the name of Islam.
Terrorists are called 'Islamic Fundamentalists' although Muslims see them as breaking
Islamic law and they suffer from being associated with terrorists and murderers. Open
views see Islam as a diverse and progressive faith with internal differences, debates and
developments. Recognizing shared values with other faiths and cultures Islam is perceived
to be equally worthy of respect. Criticisms by the West are considered and differences
and disagreements do not diminish efforts to combat discrimination while care is taken
that critical views of Islam are not unfair and inaccurate.
J
Jati - A local subcastes in Hindu India.
Joint Family Household - Is a complex family unit formed through polygyny or
polyandry or through the decision of married siblings to live together with or without
their parents.
42 | P a g e
Jook Sing - A Chinese term used to refer to "American Born Chinese" of either U.S. or
Canadian birth. Meaning "hollow bamboo" in Cantonese, it suggests that the target of the
remark may be Chinese on the outside, but lacks the cultural beliefs and values that would
make them "truly" Chinese.
K
Kinesics - The study of non-linguistic bodily movements, such as gestures, stances and
facial expressions as a systematic mode of communication.
Kinship Calculation - The system by which people in a particular society reckon kin
relationships.
Kike or Kyke - Derogatory term in the U.S. for a Jew. From kikel, in Yiddish for
circle. Probably came from the practice that early immigrant Jews signed legal
documents with an "O" (rather than an "X")
L
Language - is the primary means of communication for humans. It may be spoken or
written and features productivity and displacement and is culturally transmitted.
Levirate - Custom by which a widow marries the brother of her deceased husband.
Life Expectancy - is the length of time that a person can, on the average, expect to live.
Life History - provides a personal cultural portrait of existence or change in a culture.
Liminality - The critically important marginal or in-between phase of a rite of passage.
Lineage - Unilineal descent group based on demonstrated descent.
Lineal Relative - Any of ego's or principal subject’s ancestors or descendants (e.g.,
parents, grandparents, children, grandchildren) on the direct line of descent that leads to
and from ego.
Linguistic Anthropology - The descriptive, comparative, and historical study of
language and of linguistic similarities and differences in time, space, and society.
M
Magic - Use of supernatural techniques to accomplish specific aims. Common in many
societies. Example: Folk magic, Witchcraft or Voodoo.
Mana - Sacred impersonal force in Melanesian and Polynesian religions.
Masculinity - One of the Hofstede dimensions. Hofstede defines this dimension as
follows: "masculinity pertains to societies in which social roles are clearly distinct (i.e.,
men are supposed to be assertive, tough and focused on material success whereas women
are supposed to be more modest, tender and concerned with the quality of life)."
(Hofstede, 1991, p. 83)
Mater - Socially recognized mother of a child.
Matriarchy - A society ruled by women. There is consensus among modern
anthropologists and sociologists that a strictly matriarchal society never existed, but there 
43 | P a g e
are examples of matrifocal societies. There exist many matriarchal animal societies
including bees, elephants, and killer whales. The word matriarchy is coined as the
opposite of Patriarchy.
Matrifocal - Mother-centered society. It often refers to a household with no resident
husband-father.
Matrilineage - Line of descent as traced through women on the maternal side of a family.
In some cultures, membership of a specific group is inherited matrilineally. For example
one is a Jew if one's mother (rather than one's father) is a Jew. The Nairs of Kerala, India
are also matrilineal.
Matrilocality - Customary residence with the wife's relatives after marriage, so that
children grow up in their mother's community. The Nair community in Kerala in South
India and the Mosuo of Yunnan and Sichuan in southwestern China are contemporary
examples.
Meritocracy - A system of government based on rule by ability or merit rather than by
wealth, race or other determinants of social position. Nowadays this term refers to
openly competitive societies like the USA where large inequalities of income and wealth
accrued by merit rather than birth is accepted. In contrast egalitarian societies like the
Scandinavian countries aim to reduce such disparities of wealth.
Mestizo - A term used to refer to people of partly Native American descent. From
Spanish.
Minority Group - A group that occupies a subordinate position in a society. Minorities
may be separated by physical or cultural traits disapproved of by the dominant group
and as a result often experience discrimination. Minorities may not always be defined
along social, ethnic, religious or sexual lines but could be broad based e.g. non-citizens or
foreigners.
Monoethnic - Belonging to the same ethnic group.
Monotheism - Worship of an eternal, omniscient, omnipotent, and omnipresent supreme
being. Judaism and Islam are examples.
Morphology - The study of form. It is used in linguistics (the study of morphemes and
word construction).
Monochronic - E.T.Hall introduced the concept of Polychronic/Monochronic cultures.
According to him, in monochronic cultures, people try to sequence actions on the "one
thing at a time" principle. Interpersonal relations are subordinate to time schedules and
deadlines.
Mulato - A term used for people of partly African descent. Originates from Spanish.
Multiculturalism - A belief or policy that endorses the principle of cultural diversity of
different cultural and ethnic groups so that they retain distinctive cultural identities. The
United States is understood as a "mosaic" of various and diverse cultures, as opposed to
the single monolithic culture that results from the "melting pot" or assimilation model.
Pluralism tends to focus on differences within the whole, while multiculturalism
emphasizes the individual groups that make up the whole. The term multiculturalism is
also used to refer to strategies and measures intended to promote diversity. According to 
44 | P a g e
Wikipedia, the word was first used in 1957 to describe Switzerland, but came into
common currency in Canada in the late 1960s.
Multiracial - The terms multiracial and mixed-race describe people whose parents are
not the same race. Multiracial is more commonly used to describe a society or group of
people from more than one racial or ethnic group. Mulato (for people of partly African
descent) and mestizo (people of partly Native American descent) in Spanish and metis in
Canadian French (for people of mixed white and original inhabitants of Canada descent)
are also used in English.
Myth - Story told in one's culture to explain things like the creation of the world, and the
behaviour of its inhabitants.
N
Nation - Earlier a synonym for "ethnic group," designating a single culture sharing a
language, religion, history, territory, ancestry, and kinship. Now usually a synonym for
state or nation-state.
National Culture - Cultural experiences, beliefs, learned behavior patterns, and values
shared by citizens of the same nation.
Nationalities - Ethnic groups that have, once had, or wish to have or regain, autonomous
political status (their own country).
Nation-State - A symbolic system of institutions claiming sovereignty over a bounded
territory. The Oxford English Dictionary defines "nation-state": a sovereign state of which
most of the citizens or subjects are united also by factors which define a nation, such as
language or common descent. Japan and Iceland could be two examples of near ideal
nation-states.
Negritude - Black association and identity. It is an idea developed by dark-skinned
intellectuals in Francophone (French-speaking) West Africa and the Caribbean.
Negro - Negro usually refers to people of Black African ancestry. Originates from
Spanish negro meaning black. The term Negro is considered offensive nowadays. Modern
synonyms in common use: "Black", "Dark-skinned", "African", "African American" in
the US.
Neolocality - Postmarital residence pattern in which a couple establishes a new place of
residence rather than living with or near either set of parents.
Nigga - Term used in African American vernacular English to refer to a person of Black
African ancestry living in the US. The use of the term by persons not of African descent
is still widely viewed as unacceptable and hostile even if there is no intention to slander.
Nigger - Extremely offensive term to refer to people of Black African ancestry in the
USA.
Nigrew - In the U.S. it is a derogatory term for a Jew of African-American descent
(shortened version of Nigger and Jew.)
Nuclear Family - is a household consisting of two heterosexual parents and their
children as distinct from the extended family. Nuclear families are typical in societies
where people must be relatively mobile e.g., hunter-gatherers and industrial societies. 
45 | P a g e
O
One-World Culture - A belief that the future will bring development of a single
homogeneous world culture through advances and links created by modern
communication, transportation and trade.
Open Class System - Stratification system that facilitates social mobility, with
individual achievement and personal merit determining social rank.
Oreo Cookies - US racial slur to refer to a person perceived as black on the outside
and white on the inside, hinted by the appearance of an Oreo cookie.
Osenbei (Senbei) - Derogatory term in the US and the UK used to refer to a half Asian,
half Caucasian person. It means "rice cracker" in Japanese. Its use derives from the US
slang "cracker" for a white person, and "rice" to refer to an Asian.
Overinnovation - Characteristic of projects that require major changes in the daily lives
of the natives in the target community, especially ones that interfere with customary
subsistence pursuits.
P
Paradigm - is the set of fundamental assumptions that influence how people think and
how they perceive the world.
Paradigmatic view - is an approach to science, developed by Thomas Kuhn, which holds
that science develops from a set of assumptions (paradigm) and that revolutionary science
ends with the acceptance of a new paradigm which ushers in a period of normal science.
Parallel Cousins - Children of two brothers or two sisters.
Particularity - Distinctive or unique culture trait, pattern, or integration.
Participant Observation - Technique for cross-cultural adjustment. This entails keeping
a detailed record of your observations, interactions and interviews while living in a culture
that is not your own. Participant observation is also a fundamental method of research
used in cultural anthropology. A researcher lives within a given culture for an extended
period of time, to take part in its daily life in all its richness and diversity. The
anthropologist in this approach tries to experience a culture "from within," as a person
native to that culture is presumed to.
Participative competence - The ability to interact on equal terms in multicultural
environments so that knowledge is shared and the learning experience is professionally
enhancing for all involved. Even when using a second language, people with high
participative competence are able to contribute equitably to the common task under
discussion and can also share knowledge, communicate experience, and stimulate group
learning to benefit all parties. (Adapted from source: Holden, Nigel 2001, Cross-Cultural
Management: a Knowledge Management Perspective) Financial Times Management
Particularism - One of the value dimensions as proposed by Trompenaars & HampdenTurner (1997). It reflects the preference for rules over relationships (or vice versa). 
46 | P a g e
Particularist societies tend to be more flexible with rules, and acknowledge the unique
circumstances around a particular rule.
Pater - Socially recognized father of a child though not necessarily the genitor or
biological father.
Patriarchy - Political system ruled by men in which women have inferior social and
political status, including basic human rights.
Patrilineage - Line of descent as traced through men on the paternal side of a family each
of whom is related to the common ancestor through males. Synonym is agnation and
opposite is matrilineage.
Patrilocality - Customary residence with the husband's relatives after marriage, so that
children grow up in their father's community.
Peers Pressure - the influences that people of the same rank, age or group have on each
other. Under peer pressure a group norm of attitudes and/or behaviours may override
individual moral inhibitions, sexual personal habits or individual attitudes or behavioural
patterns.
Periphery - is the weakest structural position in the world system.
Personal Space - Humans desire to have a pocket of space around them and into which
they tend to resent others intruding. Personal space is highly variable. Those who live in a
densely populated environment tend to have smaller personal space requirements. Thus a
resident of a city in India or China may have a smaller personal space than someone who
lives in Northern Lapland. See also Proxemics.
Phonetics - The study of speech sounds in general; what people actually say in various
languages.
Phylogenetic tree - is a graphic representation of evolutionary relationships among
animal species.
Plural Society - A society that combines ethnic contrasts and economic interdependence
of the ethnic groups.
Polyandry - A variety of plural marriage in which a woman has more than one husband.
Tibet is the well-documented cultural domain within which polyandry is practiced,
though it has recently been outlawed.
Polytheism - Belief in several deities who control aspects of nature. The ancient Greeks
believed that their gods were independent deities who weren't aspects of a great deity.
Polychronic - The concept of Polychronic/Monochronic cultures was introduced by E.T.
Hall. He suggested that in Polychronic cultures, multiple tasks are handled at the same
time, and time is subordinate to interpersonal relations.
Postcolonial - Refers to interactions between European nations and the societies they
colonized (mainly after 1800). "Postcolonial" may be used to signify a position against
imperialism and Eurocentrism
Postmodern - Describes the blurring and breakdown of established canons (rules,
standards), categories, distinctions, and boundaries.
Postmodernity - Refers to the condition of a world in flux, with people on the move, in
which established groups, boundaries, identities, contrasts, and standards are breaking
down.
47 | P a g e
Post-Partum Sex Taboo - is the prohibition of a woman from having sexual intercourse
for a specified period of time following the birth of a child.
Power Distance - One of the Hofstede dimensions of national cultures. "The extent to
which the less powerful members of institutions and organizations within a country expect
and accept that power is distributed unequally" (Hofstede, 1991 p.27)
Power Geometry - The notion of Power Geometry is a product of globalization and
refers to the ways that different groups of individuals interact at different scales, linking
local development to national, international, and global processes.
Prejudice - Over-generalized, oversimplified or exaggerated beliefs associated with a
category or group of people. These beliefs are not easily changed, even in the fact of
contrary evidence. Example: A French woman is in an elevator alone. She grabs her purse
tight when an African young man enters. Prejudice can also be devaluing (looking down
on) a group because of its assumed behavior, values, capabilities, attitudes, or other
attributes.
Progeny Price - A gift from the husband and his kin to the wife and her kin before, at, or
after marriage. It legitimizes children born to the woman as members of the husband's
descent group.
Protoculture - is the simplest or beginning aspects of culture as seen in some nonhuman
primates.
Proto-language - refers to a language ancestral to several daughter languages. Example:
Latin or Sanskrit.
Proxemics - is the study of human "perception and use of space" (Hall 1959). Proxemics
tries to identify the distance and the way the space around persons are "organized". In
some cultures, people are comfortable with being very close, or even touching each other
as a normal sign of friendship. In other cultures, touching and sitting/standing very close
can cause considerable discomfort.
Purdah - is the Muslim or Hindu practice of keeping women hidden from men outside
their own family; or, a curtain, veil, or the like used for such a purpose.
Q
Qualitative Research - Qualitative research involves the gathering of data through
methods that involve observing forms of behaviour e.g. conversations, non-verbal
communication, rituals, displays of emotion, which cannot easily be expressed in terms of
quantities or numbers.
Quantitative Research - Quantitative research is the systematic scientific investigation
of quantitative or measurable properties and phenomena and interrelationships.
Quantitative research aims to develop and employ hypotheses, theories and models,
which can be verified scientifically.
Questionnaire - Survey research technique in which the researcher supplies written
questions to the subject, who gives written answers to the questions asked.
R
48 | P a g e
Racism - Theories, attitudes and practices that display dislike or antagonism towards
people seen as belonging to particular ethnic groups. Social or political significance is
attached to culturally constructed ideas of difference.
Ranked Society - A society in which there is an unequal division of status and power
between its members, where such divisions are based primarily on such factors as family
and inherited social position. This is in contrast with egalitarian society, which aims to
minimize such unequal divisions.
Reciprocity - One of the three principles of exchange. It governs exchange between social
equals and is a major exchange mode in band and tribal societies. Since virtually all humans
live in some kind of society and have at least a few possessions, reciprocity is common to
every culture. Reciprocity is the basis of most non-market economies.
Religious Discrimination - Religious discrimination is treating someone differently
because of what they do or don't believe. Religious discrimination is closely related to
racism, but there are differences in how it is expressed and how it is treated in law. An
example of religious discrimination by the state is non-Muslims being discriminated
against in some Islamic states. In many countries legislation specifically prohibits
employers from discriminating against individuals because of their religion in relation to
hiring, firing and other terms and conditions of employment. Today, many western states
forbid discrimination based on religion, though this is not always enforced. For example,
since the terrorist attacks of September 11, 2001 in the United States of America, research
conducted by the Level Playing Field Institute and the Center for Survey Research and
Analysis at the University of Connecticut revealed that Muslims were rated very low
relative to other racial, ethnic, and religious groups in terms of their fit in the American
workplace. Adapted from source: http://en.wikipedia.org
Relativism - A willingness to consider other persons' or groups' theories and values as
equally reasonable as one's own.
Rites of Passage - Culturally defined activities (rituals) that mark a person's transition
from one stage of life to another. These aim to help participants move into new social
roles, positions or statuses. Puberty, wedding, childbirth are examples.
Ritual - Behaviour that is formal, stylized, repetitive, and stereotyped. A ritual is
performed earnestly as a social act. Rituals are held at set times and places and have
liturgical orders.
S
Sample - A smaller study group chosen to represent a larger population.
Sapir - Sapir?Whorf hypothesis (SWH) (also known as the "linguistic relativity
hypothesis") is a theory that different languages produce different ways of thinking. It
postulates a systematic relationship between the grammatical categories of the language a
person speaks and how that person both understands the world and behaves in it.
Scapegoating - The directing of hostility towards less powerful groups when the actual
source of frustration or anger cannot be attacked or is unavailable.
49 | P a g e
Schema - An organized pattern of knowledge, acquired from past experience, humans use
to interpret current experience.
Script - A conceptual representation of a stereotyped sequence of events.
Self-awareness - A psychological state in which individuals focus their attention on and
evaluate different aspects of their self-concepts. These can vary from physical
experiences to differences between "Ideal" self and "Real" self.
Self-categorization - The process of an individual spontaneously including herself or
himself as a member of a group.
Self-schema - Cognitive generalizations about own self. These guide and organize the
processing of self-related information.
Semantic differential technique - A method of measuring attitude in which test
subjects rate a concept on a series of bipolar scales of adjectives.
Sexism - Discrimination or prejudice against some people because of their gender.
Sexual Orientation - A person's habitual sexual attraction to, and activities with:
persons of the opposite sex, heterosexuality; the same sex, homosexuality; or both
sexes, bisexuality.
Sexual Orientation Discrimination - Sexual orientation discrimination is
discrimination against individuals, couples or groups based on sexual orientation or
perceived sexual orientation. Usually, this means the discrimination of a person who has a
same-sex sexual orientation, whether or not they identify as gay, lesbian or bisexual.
Acceptability of sexual orientation varies greatly from society to society. The Republic of
South Africa is the first nation on earth to integrate freedom from discrimination based on
sexual orientation into its constitution.
Simulation - A research method that tries to imitate crucial aspects some real-world
situation in order to understand the underlying mechanism of that situation.
Slavery - is the most extreme, coercive, abusive, and inhumane form of legalized
inequality where people are treated as things or someone's property.
Social Distance - The degree of physical, social or psychological closeness or intimacy
to members of a group like ethnic, racial or religious groups.
Social Exclusion - The various ways in which people are excluded from the accepted
norms within a society. Exclusion can be economic, social, religious or political.
Social Inhibition - Happens when the presence of other people causes a decline in a
person's performance. Also called Social Impairment.
Social Judgment Theory - A theory of attitude change which emphasizes the
individual's perception and judgment of a persuasive communication. Central concepts in
this theory are anchors, assimilation and contrast effects, and latitudes of acceptance,
rejection and noncommitment.
Social Learning Theory - A theory that proposes that social behaviour develops as a
result of observing others and of being reinforced for certain behaviours.
Social Race - A group assumed to have a biological basis but actually perceived and
defined in a social context, by a particular culture rather than by scientific criteria. The
term "social race" has been used in the past as well as in today's American societies.
Terms as "Negro", "white", "Indian", or "mulatto" do not have any genetic meanings in 
most of the American societies - in one society they may be classifications based on real
or imaginary physical characteristics, in another they may refer more to criteria of social
status such as education, wealth, language and custom, or in yet another society they may
indicate near or distant ancestry.
Social Support - Help and resourced provided by others for coping.
Socialization - A process of behaviours accepted by society.
Sociofugal Space - Settings created to discourage conversation among people by making
eye contact difficult. E.g. side by side seating in waiting rooms.
Sociolinguistics - is the study of relationships between social and linguistic variation or
the study of language (performance) in its social context.
Sociopetal Space - Setting that encourage interpersonal interaction through increased eye
contact. E.g. cafes, cocktail lounges.
Stereotypes - Stereotypes (or "characterizations") are generalizations or assumptions
that people make about the characteristics of all members of a group, based on an
inaccurate image about what people in that group are like. For example, Americans are
generally friendly, generous, and tolerant, but also arrogant, impatient, and domineering.
Asians are humble, shrewd and alert, but reserved. Stereotyping is common and causes
most of the problems in cross-cultural conflicts.
Stigma - A term describing the condition of possessing an identity which has been
branded 'spoiled' or discredited identity by others. Examples of negative social stigmas are
physical or mental handicaps and disorders, as well as homosexuality or affiliation with a
specific nationality, religion or ethnicity.
Stratification - Characteristic of a system with socioeconomic strata, sharp social
divisions based on unequal access to wealth and power.
Stratified Society - A society where there is an unequal division of material wealth
between its members.
Strength - Power, status or resources associated with a social influence agent in social
impact theory.
Stress - An imbalance between environmental demands and an organism's response
capabilities. Also the human body's response to excessive change.
Structuralism - There has been a number of forms of "structuralism" in the history of
anthropology.
Structural-functionalism approaches the basic structures of a given society as serving
key functions in meeting basic human needs. Another form of structuralism, developed by
Claude Levi-Strauss, argues that social/cultural structures are actually rooted in the
fundamental structure of the human brain, which generates basic building-blocks of
social/cultural systems. In this approach, culture is studied for its deeper meaning to be
discovered in the careful structural analysis of meaning in myth and ritual.
Sub-Culture - A part or subdivision of a dominant culture or an enclave within it with a
distinct integrated network of behaviour, beliefs and attitudes. The subculture may be
distinctive because of the race, ethnicity, social class, gender or age of its members.
Symbolic Racism - A blend of negative affect and traditional moral values embodied in
e.g., the Protestant ethic; underlying attitudes that support racist positions.
Syncretism - Blending traits from two different cultures to form a new trait. Also called
fusion. This occurs when a subordinate group moulds elements of a dominant culture to
fit its own traditions.
Syntax - The arrangement and order of words in phrases and sentences.
T
Taboo - is a strong social prohibition with grave consequences about certain areas of
human activity or social custom. The term originally came from the Tongan language. The
first recorded usage in English was by Captain James Cook in 1777. Some examples of
taboo are dietary restrictions such as halal or kosher, restrictions on sexual activities such
as incest, bestiality or animal-human sex, necrophilia or sex with the dead etc.
Third World - A very vague term used to describe those regions of the world in which
levels of development, applying such measures as GDP, are significantly below those of
the economically more advanced regions. The term is increasingly seen as an inadequate
description of the prevailing world situation since it fails to describe a significant amount
of internal differentiation and development.
Traditional Medicine- Medicine and healthcare practices which originated in a particular
culture, and have been practiced by an ethnic or cultural group centuries in the country of
origin or of emigration
Trait - Describes regularities in behaviour, especially with reference to an individual's
personality.
Transculturation - is a term coined by Fernando Ortiz in the 1940s to describe the
phenomenon of merging and converging of different cultures. It argues that the natural
tendency of people is to resolve conflicts over time, rather than aggravating them. Global
communication and transportation technology nowadays replaces the ancient tendency of
cultures drifting or remaining apart by bringing cultures more into interaction. The term
Ethnoconvergence is sometimes used in cases where tranculturation affects ethnic
issues.
Tribe - A type of social formation usually considered to arise from the development of
agriculture. Tribes tend to have a higher population density than bands and are also
characterized by common descent or ancestry.
U
Uncertainty Avoidance - is one of the Hofstede dimensions, which he defines as "the
extent to which the members of a culture feel threatened by uncertain or unknown
situations." (Hofstede, 1991)
Uncertainty of Approval - Measures how much any member of a group is concerned
about getting acceptance from other group members.
Underdifferentiation - In developmental anthropology it refers to planning fallacy of
viewing less-developed countries as an undifferentiated group. Ignoring cultural diversity
and adopting a uniform approach (often ethnocentric) for very different types of project 
beneficiaries. In Linguistics it is the representation of two or more phonemes, syllables, or
morphemes with a single symbol.
Unilineal Descent - Matrilineal or patrilineal descent.
Unilineal Descent Group - is a kin group in which membership is inherited only
through either the paternal or the maternal line.
Universal - Something that exists in every culture.
Universalism - One of the Trompenaars & Hampden-Turner (1997) dimensions
describing the preference for rules over relationships (or vice versa). In a Universalist
culture, a rule cannot be broken and is a "hard fact", no matter what the relationship with
the person is. People in universalistic cultures share the belief that general rules, codes,
values and standards take precedence over particular needs and claims of friends and
relations.
V
Validity - The extent to which a measure represents accurately what it is supposed to
represent.
Variables - Attributes (e.g., sex, age, height, weight) that differ from one person or case
to the next.
Vertical Mobility - Upward or downward change in a person's social status.
Visual dominance behaviour - Is the tendency of high-status positions to look more
fixedly at lower-status people when speaking than when listening.
Vividness - The intensity or emotional interest of a stimulus.
W
Wealth - All a person's material assets, including income, land, and other types of
property. It is the basis of economic and often social status.
Westernization - The acculturative influence of Western expansion on native cultures.
Wetback - Derogatory US term used to describe Mexican illegal immigrants, who
allegedly entered the country by swimming the Rio Grande.
White Nigger / Wigger / Whigger / Wigga - Derogatory term used in 19th-century
United States to describe the Irish. Nowadays used mainly to demean any White person
as being White Trash or to describe white youth that imitate urban black youth by
means of clothing style, mannerisms, and slang speech.
Worldview - Is the English translation of the German word Weltanschaung. Also called
World View.
X
Xenophile - is a person attracted to everything that is foreign, especially to foreign
peoples, manners, or cultures.
Xenophile - The belief that people and things from other countries must be superior. 
Xenophobe - is a person who is fearful or contemptuous of anything foreign, especially
of strangers or foreign peoples or cultures.
Xenophobia - The belief that people and things from other countries are dangerous and
always have ulterior motives. Xenophobia is an irrational fear or hatred of anything
foreign or unfamiliar.
Y
Yang - Yin and Yang are two opposing and complementing aspects of phenomena in
Chinese philosophy. Yin qualities are hot, fire, restless, hard, dry, excitement, nonsubstantial, rapidity, and correspond to the day.
Yin - Yin and Yang are two opposing and complementing aspects of phenomena in
Chinese philosophy. Yin qualities are characterized as soft, substantial, water, cold,
conserving, tranquil, gentle, and corresponds to the night.
Ability: Having the mental and/or physical condition to engage in one or more major life activities (e.g., seeing, hearing, speaking, walking, breathing, performing manual tasks, learning or caring for oneself).
Ableism: Prejudice and/or discrimination against people with mental and/or physical disabilities.
American Sign Language: A means of communication that uses hand gestures to represent letters and words, and the primary sign language used by people with hearing disability in the United States and Canada (devised in part by Thomas Hopkins Gallaudet on the basis of sign language in France).
Assistive Technology: A device or piece of equipment used to maintain or improve the functional facility of people with disabilities (e.g., brace, crutches, descriptive video, hearing aid, prosthetic device, walker, wheelchair).
Attention Deficit (Hyperactivity) Disorder: Attention-deficit (hyperactivity) disorder is a condition affecting children and adults that is characterized by problems with attention, impulsivity, and overactivity. Science recognizes three subtypes of ADD or ADHD: inattentive, hyperactive-impulsive, and combined. A diagnosis of one type or another depends on the specific symptoms that person has.
Blindness: Partial or “legal” visual impairment based on standard vision being defined as 20/20 visual acuity and an average range of 180 degrees in peripheral vision; thus, people are defined as being legally blind if after methods of correction, such as glasses or contact lenses, they have a visual acuity of 20/200 or higher, or a range of peripheral vision under 20 degrees.
Cerebral Palsy: A functional disorder caused by damage to a child’s brain during pregnancy, delivery, or shortly after birth. Cerebral Palsy is characterized by one or more movement disorders, such as spasticity (tight limb muscles), purposeless movements, rigidity (severe form of spasticity), or a lack of balance. People with cerebral palsy may also experience seizures, speech, hearing and/or visual impairments, and/or mental retardation.
Closed Captioning: An on-screen system that allows people with a hearing disability to view television with spoken words written across the bottom of the screen.
Deaf-Blindness: A hearing and visual disability, the combination of which can cause severe communication and other developmental and educational difficulties.
Deafness: A total or partial inability to hear, which can be genetic or also acquired through disease, most commonly from meningitis in childhood or rubella in a woman during pregnancy.
Descriptive Video: Film media designed for people with visual disability that provides additional narration detailing the visual elements of a film (the action of the characters, locations, costumes, etc.) without interfering with the actual dialogue and sound effects.
Developmental Disability: A long lasting cognitive disability occurring before age 22 that limits one or more major life activities (self-care, independent living, learning, mobility, etc.), and is likely to continue indefinitely (e.g., Autism).
Disability: A mental or physical condition that restricts an individual's ability to engage in one or more major life activities (e.g., seeing, hearing, speaking, walking, communicating, sensing, breathing, performing manual tasks, learning, working or caring for oneself).
Down Syndrome: A chromosomal condition (trisomy 21) caused by the presence of one extra chromosome, and characterized by delayed physical and mental development, and often identifiable by certain physical characteristics, such as a round face, slanting eyes, and a small stature.
Dwarfism: A genetic condition resulting in short stature.
Emotional Disability: One or more psychiatric disabilities exhibited over a long period of time and to a marked degree, e.g., an inability to build or maintain satisfactory interpersonal relationships with others; inappropriate types of behavior or feelings under ordinary circumstances; a generally pervasive mood of unhappiness or depression; or a tendency to develop physical symptoms or fears associated with personal problems.
Epilepsy: A physical condition that occurs when there is a sudden, brief disturbance in the function of the brain, and alters an individual's consciousness, movements or actions. Most individuals with epilepsy can reduce or eliminate the risk of seizures through the regular use of appropriate medication.
Handicap: Any obstacle that decreases a person’s opportunity for success (e.g., discriminatory practices, inaccessible buildings/public places/transportation, insufficient insurance/training/resources, negative attitudes).
Health Disability: A temporary or permanent health impairment that affects one or more major life activities (e.g., AIDS, arthritis, cancer, diabetes, drug addiction, heart disease).
Hearing Disability: Partial or full hearing loss due to either a decibel loss (person hears all sounds much more softly than a person with complete hearing), or a frequency loss (person hears a pitch of a sound better than others, thus a person with frequency loss would hear all of some words, some parts of other words, and would not hear some words at all).
Inclusion: An environment and commitment to support, represent and embrace diverse social groups and identities; an environment where all people feel they belong. (In K-12 learning environments, inclusion can sometimes also refer a set of practices and beliefs that all people should be educated, regardless of disability, in an age appropriate, local, general education setting with appropriate supports and services.)
Learning Disability: A cognitive impairment in comprehension or in using language, spoken or written, that manifests itself in a person’s ability to listen, think, speak, read, write, spell, or to do mathematical calculations (e.g., Dyslexia, Dysnomia, Dysgraphia). The term does not include persons who have learning difficulties that are primarily the result of mental retardation, emotional disability, or environmental, cultural or economic disadvantage.
Little Person: A person with short-stature. In general, people with short-stature prefer the term “Little Person” to describe their physical condition. The term “dwarf” is considered derogatory.
Mental Illness: Refers to any illness or impairment that has significant psychological or behavioral manifestations, is associated with painful or distressing symptoms and impairs an individual’s level of functioning in certain areas of life (e.g., Anxiety Disorder, Depression, Bipolar disorder, Obsession-Compulsion, Schizophrenia).
Mental Retardation: Consistent demonstration of general cognitive functioning that is determined to be 1.5 standard deviations or more below the mean of the general population on the basis of a comprehensive evaluation.
Orthopedic Impairment: Physical disability caused by a congenital anomaly (e.g., club foot), impairments caused by disease (e.g., poliomyelitis), and impairment from other causes (e.g., cerebral palsy, fractures or burns which cause contractures.) 
Paraplegia: The paralysis of the legs and lower part of the body and is usually caused by injury or disease in the lower spinal cord, or by brain disorders such as cerebral palsy.
Parkinson’s Disease: A progressive disorder caused by the brain’s inability to manufacture a chemical that signals the muscles to move. Symptoms include involuntary tremors, stiff movements, and/or lack of balance.
People First: Acknowledging the personhood of individuals with disabilities before their disability (e.g., “people with disabilities”, “person who uses a wheelchair”, “person with cerebral palsy”, “person has a physical disability”, etc.).
Physical Disability: One or more physical impairments that substantially limit one or more major life activities (e.g., seeing, hearing, speaking, walking, breathing, performing manual tasks, learning, or caring for oneself).
Post-Polio Syndrome: A condition that affects a person who has had poliomyelitis (polio) after recovery, and is characterized by muscle weakness, joint and muscle pain and fatigue.
Prosthesis: An artificial device used to replace a missing body part, such as a limb, tooth, eye or heart valve.
Quadriplegia: The paralysis of a person’s four limbs.
Reasonable Accommodation: A modification made in facilities, a job restructuring or rescheduling, or a modification of equipment and devices to make an environment accessible and useable by people with disabilities.
Speech Impairment: A communication disorder characterized by impaired articulation, language impairment or voice impairment (e.g., Dysfluency, Stuttering).
Tourette Syndrome: A genetic, neurological disorder characterized by repetitious, involuntary body movements and uncontrollable vocal sounds.
Visual Disability: A form of eyesight impairment that varies in severity and in more acute cases cannot be corrected by glasses or contact lenses.

Topic Finder
Cross Reference Index
Contributors
Preface
Academic Freedom
Adoption
Advertising, children's 
Affirmative Action
Ageism
AIDS/HIV
Air Pollution
Airline Issues
Alcohol Abuse
Animal rights
Anti-Muslim Discrimination and Violence
Anti-Semitism
Arson
Arts Funding and Censorship
At Risk Students: Higher Education
Attention Deficit-Hyperactivity Disorder
Autism
Automobile and Highway Safety
Bilingualism
Birth Control
Campaign Finance Reform
Cancer
Capital Punishment
Census Issues
Cheating, academic
Child Abuse and Molestation
Child Labor
Chronic Fatigue Syndrome
Church-State Separation
Civil Liberties
Civil Rights
Coastal Pollution and Wetlands Protection
College Sports
Computer Crime, Hacking
Consumer Debt and Bankruptcy
Corporal Punishment


Drugs, War on
Eating Disorders
Energy Dependency
Environmental Justice
Environmentally-induced Illness
Euthanasia
Evolution Education
Extinction and Species Loss: Biota Invasion and Habitat Destruction
Farm crisis
Food and Drug Safety
Foster Care
Gambling
Gangs
Gay and Lesbian Rights
Genetic Engineering
Gentrification
Global Warming
Gun violence and gun control
Hate Crimes
Hate Internet and Radio
Hate Speech
Health Care Reform
Heart Disease
Homelessness
Housing costs
Human experimentation
Identity Theft
Immigration
Immigration, Illegal
Indoor Pollution
Infectious Disease and Epidemics
Infrastructure Deterioration
Intellectual Property Rights
Journalistic Ethics
Judicial Reform
Juvenile Justice
Legal Services for the Poor
Literacy
Mandatory Sentencing
Marijuana
Mass Transit
Media Bias
Media Consolidation
Media Sex and Violence



Natural Disasters and Disaster Relief
Needle Exchange Programs
Noise Pollution
Nuclear Power and Waste
Nuclear Weapons
Obesity
Occupational Safety and Health
Organ and Tissue Transplants
Organic Foods
Organized Crime
Plagiarism
Police Abuse and Corruption
Pornography
Poverty and Wealth
Prison Reform and Prisoner Rights
Privacy
Prostitution
Public Opinion Polling
Racial Profiling
Rape
Recycling and Conservation
Red-lining and loan discrimination
Reproductive Rights and Technology
Rioting
School Standards and Testing
School Violence
School Vouchers and Privatization
Scientific Research Ethics
Secrecy, Governmental
Sex Education
Sexual Harassment
Single Parenting
Social Security Reform
Space Exploration, costs and benefits
Special Education
Stem Cell Research
Stress
Student Rights
Suicide
Superstores v. Main Street
Sweatshops
Tax Reform
Term Limits



Voting Issues
Waste Disposal
Water Pollution
Weapons of Mass Destruction
Welfare and welfare reform
Wilderness Protection
Women's Rights
Xenophobia and Nativism
Mental Illness
Migrant Workers
Militia Movement
Minimum and Living Wages
Money Laundering
Not In My Backyard Issue
Native Americans and Government Policy
Tobacco and tobacco-related health issues
Tort Reform
Toxic Waste
Traffic Congestion
Unemployment
Unions
Urban Sprawl
Veterans' Issues
Voluntarism and Volunteering
Corporate Crime
Crime
Criminal Rights
Cults and Alternative Religions
Defense Spending and Preparedness
Deforestation and Logging
Disability Rights
Divorce and Child Support
Domestic Violence
Downsizing, corporate
Drought and aquifer depletion
Drug Abuse
Medicine, alternative
Medical Malpractice
Medicare and Medicaid Reform
Terrorism, Domestic
Terrorism, Foreign
Terrorism, War on



Feminism
Feminist
Advocacy
Apartheid
Philanthropy
Shareholder
Aids
Naacp
Activist
Lobbying
Empowerment
Anarchist
Campaigning
Lesbian
Restraint
Suffrage
Disobedience
Racism
Islamist
Boycott
Rights
Spirituality
Protest
Globalization
Abortion
Abolitionist
Environmentalist
Awareness
Forefront
Anti
Mobilization
Solidarity
Gay
Collective
Resurgence
Entrepreneurship
Involvement
Politics
Sexuality
Klan
Oppression
Movement
Marxism
Equality
Idealism
Journalism
Hiv
Marches
Accountability
Segregation
Behalf
Outreach
Social
Workplace
Academia
Injustice
Commitment
Conservatism
Deference
Gender
Homosexuality
Socialism
Progressive
Advocate
Imperialism
Focus
Ruling
Unrest
Leadership
Stance
Communism
Genocide
Ism
Mandela
Cannabis


Intellectual
Consumer
Labor
Ethic
Radical
Investing
Defamation
Pedagogy
Organization
Bureaucracy
Scholarship
Youth
Quaker
Sustainability
African
Reform
Violence
Evangelical
Socialist
Latina
Conscription
Fascist
Riot
Justice
Tenet
Student
Civil
Nonprofit
Engaging
Palestinian
Nationalist
Ecological
Authoritarian
Involved
Community
Governmental
Corporate
Substantive
Quaker
Journalistic
Societal
Mainstream
Indigenous
Imprisoned
Homosexual
Collective
Lifelong
Welfare
Courageous
Campus
Epidemic
Procedural
Cute
Communist
Postwar
Ethical
Media
Handicapped
Empower
Spearhead
Motivate
Advocate
Network
Homosexual
Disability
Liberation
Censorship
Demonstration
Governance
Organizer
Prostitution
Participation
Palestinian
Fascism
Colored


Reproductive
Engaged
Cultural
Guerrilla
Opposing
Judiciary
Socioeconomic
Sociological
Intellectual
Passionate
Liberal
Rooted
Fostered
Constitutional
Sustainable
Revolutionary
Arrested
Global
Protestant
Awakening
Skeptical
Animal
Islamic
Escalate
Devote
Inspire
Emphasize
Theorize
Segregate
Criticize
Educate
Redefine
Encourage
Champion
Actively
Rape
Citizenship
Slavery
Graffito
Newsletter
Legislation
Optimism
Emancipation
Consciousness
Israeli
Regime
Vigil
Experimentation
Credibility
Community
Ethics
Organize
Fuel
Legalize
Reclaim
Heighten
Focus
Socially
Politically
Engagement
Issue
Emphasis
Pro
Gandhi
Worldview
Imprisonment
Dissemination
Legitimacy
Diversity
Lennon
Aesthetics
Peace
Temperance
Internet
Vietnam


Tactic
Jurisprudence
Dictatorship
Semitism
Minority
Mysticism
Initiative
Raising
Homo
Caucus
Hallmark
Libertarian
Campaign
Communist
Autism
Networking
Litigation
Tung
Skepticism
Ideology
Pornography
Grassroots
Feminist
Judicial
Leftist
Lesbian
Militant
Transgender
Political
Activist
Rights
Gay
Radical
Civic
Environmental
Anti
Social
Queer
Outspoken
Zionist
Affirmative
Protesting
Abolition
Socialist
Humanitarian
Philanthropic
Jailed
Fascist
Racist
Marxist
Conservative
Evangelical
Ideological
Mobilize
Spark
Engage
Amnesty
Capitalism
Judiciary
Agenda
Repression
Stakeholder
Dissent
Discrimination
Tice
Liberalism
Campaigner
Critique
Democracy
Fundraising
Marijuana
Nationalism
Diaspora
Combine
Elite

Ableism	Accomplice	Ageism	Ally	Anti-Semitism Anti- Jewish Oppression	Asexual	Biphobia	Bisexual	Birth Assigned Sex	Cisgender	Cissexism	Classism	Collusion	Coming Out	Cultural Appropriation	Cultural Competence	Discrimination	Empathy	Ethnocentrism	Equality	Equity	Gay	Gender	Gender Binary	Gender Expression	Gender Identity	Genderqueer/ also termed Gender Non Binary	Gender Non Conforming	Gender Pronoun	Gender Neutral or Gender Inclusive Pronoun	Hate Group	Heterosexism	Heterosexual	Homophobia	Internalized Oppression	Intersectionality	Intersex	Islamophobia	Lesbian	Oppression	Pansexual	Power	Prejudice	Privilege	Queer	Racism	Racial Profiling	Religious Oppression	Sexism	Sexual Orientation	Social Justice	Stereotype	Transgender	Transphobia	Privilege	Xenophobia


animal rights
civil rights
education
free trade
fair trade
gay rights
gun control
illegal arms
immigration
nuclear energy
refugee
oil price
poverty
unemployment
First Amendment
human rights
minority
minorities


terrorism
domestic violence
Me too
Don't Ask, Don't Tell
gender
gender roles
misogyny
rape
sexism
sexual discrimination
sexual harassment
sexual slavery
transgender movement
LGBTQ
LGBT
wage gap
women's rights
gentrification


censorship
democratic party
election
political action committee
political party
republican
democrat
revolution
voting
militia
child soldier
child labor
exploitation
affirmative action
sex education
Taliban
abortion
birth control
pro-choice
pro-life


apartheid
civil rights movement
discrimination
genocide
Holocaust
NAACP
race
race relations
racial
racism
slavery
abortion
health insurance
medicaid
medicare
stem cell research
health care

Accountability	Ally	Anti-Black	Anti-Racism	Anti-Racist	Anti-Racist Ideas	Assimilationist	Bigotry	Black Lives Matter	Caucusing (Affinity Groups)	Collusion	Colonization	Critical Race Theory	Cultural Appropriation	Cultural Misappropriation	Cultural Racism	Culture	Decolonization	Diaspora	Discrimination	Diversity	Ethnicity	Implicit Bias	Inclusion	Indigeneity	Individual Racism	Institutional Racism	Internalized Racism	Interpersonal Racism	Intersectionality	Microaggression	Model Minority	Movement Building	Multicultural Competency	Oppression	People of Color	Power	Prejudice	Privilege	Race	Racial and Ethnic Identity	Racial Equity	Racial Healing	Racial Identity Development Theory	Racial Inequity	Racial Justice	Racial Reconciliation	Racialization	Racism	Racist	Racist Ideas	Racist Policies	Reparations	Restorative Justice	Settler Colonialism	Structural Racialization	Structural Racism	Targeted Universalism	White Fragility	White Privilege	White Supremacy	White Supremacy Culture	Whiteness

Environmental Issues (Wikipedia)
Agriculture 	irrigation	meat production	cocoa production	palm oil	Energy industry 	biofuels	biodiesel	coal	nuclear power	oil shale	petroleum	reservoirs	Genetic pollution	Industrialisation	Land use	Manufacturing 	cleaning agents	concrete	plastics	nanotechnology	paint	paper	pesticides	pharmaceuticals and personal care	Marine life 	fishing	fishing down the food web	marine pollution	overfishing	Mining	Overdrafting	Overexploitation	Overgrazing	Overpopulation	Particulates	Pollution	Quarrying	Reservoirs	Tourism	Transport 	aviation	roads	shipping	Urbanization 	urban sprawl	War	Effects
Biodiversity threats 	biodiversity loss	decline in amphibian populations	decline in insect populations	Climate change 	global warming	runaway climate change	Coral reefs	Deforestation	Defaunation	Desertification	 Ecocide	Erosion	Environmental degradation	Freshwater cycle	Habitat destruction	Holocene extinction	Nitrogen cycle	Land degradation	Land consumption	Land surface effects on climate	Loss of green belts	Phosphorus cycle	Ocean acidification	Ozone depletion	Resource depletion	Water degradation	Water scarcity	Alternative fuel vehicle propulsion	Birth control	Cleaner production	Climate change mitigation	Climate engineering	Community resilience	Decoupling	Ecological engineering	Environmental engineering	Environmental mitigation	Industrial ecology	Mitigation banking	Organic farming	Recycling	Reforestation 	urban	Restoration ecology	Sustainable consumption	Waste minimization
Greenhouse Effect	deforestation	endangered species	environmentalism	global warming	overfishing	overpopulation	ozone	petroleum	rainforest	urban sprawl	suburbanization	whaling	air pollution	fine dust	yellow dust	Chernobyl	Love Canal	marine pollution	nuclear waste	oil spill	pollution	smog	water pollution	 clean water	American Clean Energy and Security Act	Biodiesel	conservation	Environmental Protection Agency	Global Environment Facility	hybrid electric cars	national parks and reserves	renewable energy	solar energy	wind power	wind turbine	alternative energy


Global Issues (Wikipedia)
Africa	Ageing	Agriculture	AIDS	Atomic energy	Children	Decolonization	Demining	Democracy	Development	Disarmament	Environment	Family	Food	Governance	Health	Human rights	Human settlements	Humanitarian assistance (s.a. Refugees)	International law	Oceans / Law of the Sea (s.a. Water)	Peace and security	People with disabilities	Population	Refugees (s.a. Humanitarian Assistance)
poverty, diseases, desertification, malnutrition, regional conflict	ageing population, demographic transition	sustainable agriculture, food security	Prevention of HIV/AIDS, HIV and pregnancy, HIV/AIDS denialism	nuclear weapons, nuclear waste	Child poverty, Child labour, Child abuse, Child mortality, Global education	exploitation	land mines	democratization	social transformation, economic development	weapons of mass destruction, chemical and biological weapons, conventional weapons, landmines and small arms	pollution, deforestation, desertification, etc., see Global environment issues below	socialisation of children, s.a. Ageing, Children	missing food security and safety, food riots, world hunger	lack of equity, participation, pluralism, transparency, accountability, rule of law	maternal health, extreme poverty	human rights violations	slums, urbanization, sanitation	humanitarian crisis, human migration, displacement	war crimes, discrimination, state-corporate crime	marine pollution, ocean governance
Terrorism	Volunteerism	Water (s.a. Ocean trash)	Women	2030 Agenda for Sustainable Development	Center for Global Food Issues	Chicago Council on Global Affairs	Climate change	Cybersecurity	Developing country	Earth Economics	Earth system science	Ecological footprint	Ecological collapse	Ecosystem collapse	Energy crisis	Environmental social science	Financial crisis	Global catastrophic risk	Global Challenges Foundation	Global change	Global Goals	Global governance	Global health	Antimicrobial resistance	Global justice	Global Rights	Global warming controversy	Human impact on the environment	Human security	Intergovernmental organization	List of United Nations peacekeeping missions	Liu Institute for Global Issues	Mass surveillance	Ozone depletion and climate change	Pandemic	Peak oil	Social justice	Species extinction	UN Sustainable Development Goals	Washington consensus	World Community Grid	World-systems theory	World War


NGO’s
Office of the Special Adviser on Africa, African Union, New Partnership for Africa’s Development, United Nations–African Union Mission in Darfur	Vienna International Plan of Action on Ageing, United Nations Principles for Older Persons, Proclamation on Ageing, International Year of Older Persons	Food and Agriculture Organization (FAO)	Joint United Nations Programme on HIV/AIDS, The Global Fund to Fight AIDS, Tuberculosis and Malaria	International Atomic Energy Agency, Treaty on the Non-Proliferation of Nuclear Weapons, Comprehensive Nuclear-Test-Ban Treaty	Education First, United Nations Children’s Fund (UNICEF), World Food Programme, Global Education First Initiative[1]	United Nations Special Committee on Decolonization, United Nations Trust Territories, International Decade for the Eradication of Colonialism	Mine Action Coordination Center, Ottawa Treaty	Universal Declaration of Human Rights, International Covenant on Civil and Political Rights, UNDP, UNDEF, DPKO, DPA, OHCHR, UN Women	Social protection floor	United Nations Office for Disarmament Affairs	United Nations Conference on the Human Environment, World Environment Day, United Nations Environment Programme (UNEP), Framework Convention on Climate Change, Montreal Protocol, Convention to Combat Desertification, Convention on Biological Diversity	UNFPA, UNICEF, International Year of the Family	FAO, World Food Programme		Millennium Development Goals	Universal Declaration of Human Rights	UN-HABITAT, Millennium Development Goal	World Food Programme, Office for the Coordination of Humanitarian Affairs, International Organization for Migration	International Law Commission, Convention on the Prevention and Punishment of the Crime of Genocide (1948), International Convention on the Elimination of All Forms of Racial Discrimination (1965), International Covenant on Civil and Political Rights (1966), International Covenant on Economic, Social and Cultural Rights (1966), Convention on the Elimination of All Forms of Discrimination against Women (1979), United Nations Convention on the Law of the Sea (1982), Convention on the Rights of the Child (1989), Comprehensive Nuclear-Test-Ban Treaty (1996), International Convention for the Suppression of the Financing of Terrorism (1999), Convention on the Rights of Persons with Disabilities (2006)	United Nations Conference on the Human Environment, Convention on the Prevention of Marine Pollution by Dumping of Wastes and Other Matter	United Nations peacekeeping, List of United Nations peacekeeping missions, Peacebuilding Commission	Convention on the Rights of Persons with Disabilities	UNFPA	United Nations Relief and Rehabilitation Administration (UNRRA),	United Nations High Commissioner for Refugees (UNHCR)	Comprehensive Convention on International Terrorism	United Nations Volunteers	UN-Water, System of Environmental and Economic Accounting for Water, Water for Life Decade, International Recommendations on Water Statistics, United Nations Water Conference, Millennium Development Goals, International Conference on Water and the Environment (1992), Earth Summit (1992)	Commission on the Status of Women, International Women's Year


3D printing
Acroyoga
Acting
Airbrushing
Amateur radio
Animation
Aquascaping
Astrology
Art
Babysitting
Baking
Baton twirling
Beatboxing
Beer tasting
Binge-watching
Blogging
Board/tabletop games
Book discussion clubs
Book restoration
Bowling
Brazilian jiu-jitsu
Breadmaking
Building
Bullet journaling
Cabaret
Calligraphy
Candle making
Candy making
Car fixing & building
Card games
Cardistry
Camgirl
Ceramics
Chatting
Cheesemaking
Chess
Cleaning
Clothesmaking
Coffee roasting
Collecting
Coloring
Communication
Community activism
Computer programming
Confectionery
Construction
Cooking
Cosplaying
Couch surfing
Couponing
Craft
Creative writing
Crocheting
Cross-stitch
Crossword puzzles
Cryptography
Cue sports
Dance
Decorating
Digital arts
Dining
Diorama
Distro Hopping
Djembe
DJing
Diving
Do it yourself
Drama
Drawing
Drink mixing
Drinking
Electronic games
Electronics
Embroidery
Engraving
Entertaining
Experimenting
Fantasy sports
Fashion
Fashion design
Feng shui decorating
Fishfarming
Fishkeeping
Filmmaking
Fingerpainting
Flower arranging
Fly tying
Foreign language learning
Furniture building
Gaming (tabletop games, role-playing games, Electronic games)
Genealogy
Gingerbread house making
Giving advice
Glassblowing
Graphic design
Gunsmithing
Gymnastics
Hacking
Hardware
Herp keeping
Home improvement
Homebrewing
Houseplant care
Hula hooping
Humor
Hydroponics
Ice skating
Inventing
Jewelry making
Jigsaw puzzles
Journaling
Juggling
Karaoke
Karate
Kendama
Knife making
Knitting
Knot tying
Kombucha brewing
Kung fu
Lace making
Lapidary
Leather crafting
Lego building
Livestreaming
Massaging
Listening to music
Listening to podcasts
Lock picking
Machining
Macrame
Magic
Makeup
Mazes (indoor/outdoor)
Mechanics
Meditation
Memory training
Metalworking
Miniature art
Minimalism
Model building
Model engineering
Music
Nail art
Needlepoint
Origami
Painting
Performance
Planning
Palmistry
Pet
Pet adoption & fostering
Pet sitting
Philately
Air sports
Airsoft
Amateur geology
Amusement park visiting
Archery
Auto detailing
Automobilism
Astronomy
Backpacking
Badminton
BASE jumping
Baseball
Basketball
Beachcombing
Beekeeping
Birdwatching
Blacksmithing
BMX
Board sports
Bodybuilding
Bonsai
Butterfly watching
Bus riding
Camping
Canoeing
Canyoning
Car riding
Car tuning
Caving
City trip
Composting
Cycling
Dandyism
Dog training
Dog walking
Dowsing
Driving
Farming
Fishing
Flag football
Flower growing
Flying
Flying disc
Flying model planes
Foraging
Fossicking
Freestyle football
Fruit picking
Gardening
Geocaching
Ghost hunting
Gold prospecting
Graffiti
Groundhopping
Guerrilla gardening
Handball
Herbalism
Herping
High-power rocketry
Hiking
Hobby horsing
Hobby tunneling
Hooping
Horseback riding
Hunting
Inline skating
Jogging
Jumping rope
Karting
Kayaking
Kite flying
Kitesurfing
Lacrosse
LARPing
Letterboxing
Lomography
Longboarding
Martial arts
Metal detecting
Modelism
Motorcycling
Meteorology
Motor sports
Mountain biking
Mountaineering
Museum visiting
Mushroom hunting/mycology
Netball
Noodling
Nordic skating
Orienteering
Paintball
Paragliding
Parkour
Photography
Picnicking
Podcast hosting
Polo
Public transport riding
Qigong
Radio-controlled model playing
Rafting
Railway journeys
Rappelling
Renaissance fair
Renovating
Road biking
Rock climbing
Rock painting
Roller skating
Rugby
Running
Sailing
Sand art
Scouting
Scuba diving
Sculling or rowing
Shooting
Shopping
Shuffleboard
Skateboarding
Skiing
Skimboarding
Skydiving
Slacklining
Sledding
Snorkeling
Snowboarding
Snowmobiling
Snowshoeing
Soccer
Stone skipping
Storm chasing
Sun bathing
Surfing
Survivalism
Swimming
Taekwondo
Tai chi
Tennis
Thru-hiking
Topiary
Tourism
Trade fair visiting
Travel
Urban exploration
Vacation
Vegetable farming
Vehicle restoration
Videography
Volunteering
Walking
Water sports
Zoo visiting


Archaeology
Astronomy
Biology
Chemistry
Electrochemistry
Geography
History
Mathematics
Medical science
Microbiology
Philosophy
Physics
Psychology
Railway studies
Research
Science and technology studies
Social studies
Sports science
Teaching
Web design
Action figure
Antiquing
Ant-keeping
Art collecting
Book collecting
Button collecting
Cartophily (card collecting)
Coin collecting
Comic book collecting
Compact discs
Credit card
Deltiology (postcard collecting)
Die-cast toy
Digital hoarding
Dolls
Element collecting
Ephemera collecting
Films
Fingerprint collecting
Fusilately (phonecard collecting)
Knife collecting
Lotology (lottery ticket collecting)
Movie memorabilia collecting
Perfume
Philately
Phillumeny
Pin (lapel)
Radio-controlled model collecting
Rail transport modelling
Record collecting
Rock tumbling
Scutelliphily
Shoes
Slot car
Sports memorabilia
Stamp collecting
Stuffed toy collecting
Tea bag collecting
Ticket collecting
Toys
Transit map collecting
Video game collecting
Vintage cars
Vintage clothing
Vinyl Records
Antiquities
Auto audiophilia
Flower collecting and pressing
Fossil hunting
Insect collecting
Leaves
Magnet fishing
Metal detecting
Mineral collecting
Rock balancing
Sea glass collecting
Seashell collecting
Stone collecting
Animal fancy
Axe throwing
Backgammon
Badminton
Baton twirling
Beauty pageants
Billiards
Bowling
Boxing
Bridge
Checkers (draughts)
Cheerleading
Chess
Color guard
Cribbage
Curling
Dancing
Darts
Debate
Dominoes
Eating
Esports
Fencing
Go
Gymnastics
Ice hockey
Ice skating
Judo
Jujitsu
Kabaddi
Knowledge/word games
Laser tag
Longboarding
Magic
Mahjong
Marbles
Martial arts
Model racing
Model United Nations
Poker
Pole dancing
Pool
Radio-controlled model playing
Role-playing games
Shogi
Slot car racing
Speedcubing
Sport stacking
Table football
Table tennis
Volleyball
VR Gaming
Weightlifting
Wrestling
Airsoft
Archery
Association football
Australian rules football
Auto racing
Baseball
Beach volleyball
Breakdancing
Climbing
Cornhole
Cricket
Croquet
Cycling
Disc golf
Dog sport
Equestrianism
Exhibition drill
Field hockey
Figure skating
Fishing
Fitness
Footbag


Photography
Pilates
Plastic art
Playing musical instruments
Poetry
Poi
Pole dancing
Pottery
Powerlifting
Practical jokes
Pressed flower craft
Proofreading and editing
Proverbs
Public speaking
Puppetry
Puzzles
Pyrography
Quilling
Quilting
Quizzes
Radio-controlled model playing
Rail transport modeling
Rapping
Reading
Recipe creation
Refinishing
Reiki
Reviewing Gadgets
Robot combat
Rubik's Cube
Scrapbooking
SCUBA Diving
Sculpting
Sewing
Shoemaking
Singing
Sketching
Skipping rope
Slot car
Soapmaking
Social media
Spreadsheets
Stand-up comedy
Stamp collecting
Storytelling
Stripping
Sudoku
Table tennis playing
Tapestry
Tarot
Tatebanko
Tattooing
Taxidermy
Thrifting
Telling jokes
Upcycling
Video editing
Video game developing
Video gaming
Video making
VR Gaming
Watch documentaries
Watching movies
Watching television
Waxing
Weaving
Webtooning
Weight training
Welding
Whittling
Wikipedia editing
Winemaking
Wine tasting
Wood carving
Woodworking
Word searches
Worldbuilding
Writing
Writing music
Yo-yoing
Yoga
Zumba
Frisbee
Golfing
Handball
Horseback riding
Horsemanship
Horseshoes
Iceboat racing
Jukskei
Kart racing
Knife throwing
Lacrosse
Longboarding
Long-distance running
Marching band
Mini Golf
Model aircraft
Orienteering
Pickleball
Powerboat racing
Quidditch
Race walking
Racquetball
Radio-controlled car racing
Radio-controlled model playing
Roller derby
Rugby league football
Sculling or rowing
Shooting sport
Skateboarding
Skiing
Sled dog racing
Softball
Speed skating
Squash
Surfing
Swimming
Table tennis
Tennis
Tennis polo
Tether car
Tour skating
Tourism
Trapshooting
Triathlon
Ultimate frisbee
Volleyball
Water polo
Audiophile
Fishkeeping
Learning
Meditation
Microscopy
Reading
Research
Shortwave listening
Aircraft spotting
Amateur astronomy
Benchmarking
Birdwatching
Bus spotting
Geocaching
Gongoozling
Herping
Hiking/backpacking
Meteorology
Photography
Satellite watching
Trainspotting
Whale watching






"""

remove = academicVerbs.replace("  "," ") # 변환
remove_ = re.sub(r"\t", " ", remove) # 제거
remove__ = re.sub(r"\n", " ", remove_) # 제거
remove__ = remove__.replace("   ", " ")
remove__ = remove__.replace("  ", " ")
remove__ = remove__.replace(" ", ",")
remove__ = remove__.replace("…/", " ")
remove__ = remove__.replace("…", " ")
remove__ = remove__.replace("/", " ")
remove__ = remove__.replace(" ", ",")
remove__ = remove__.replace(")", ",")
remove__ = remove__.replace("(", ",")
preprossed = remove__.split(",") # 단어를 리스트로 변환

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# 방법 2: 표제어 추출
ext_lema = [lemmatizer.lemmatize(w) for w in preprossed]
# 중복값을 제거하고
rm_dupli = set(ext_lema)
# 다시 리스트로 만들고
re_li = list(rm_dupli)
# 빈 값은 제거하고
get_wd =list(filter(None, re_li))
# 소문자로 모두 변환
lower_wd = [i.lower() for i in get_wd]



# 불용어 제거

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
stop = stopwords.words('english')
stop_words = set(stopwords.words('english')) 

result = []
for w in lower_wd: 
    if w not in stop_words: 
        result.append(w) 


print(result)
