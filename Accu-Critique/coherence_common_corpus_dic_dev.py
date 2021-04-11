from gensim.test.utils import common_corpus, common_dictionary
from gensim.models.coherencemodel import CoherenceModel
topics = [
    ['human', 'computer', 'system', 'interface']
]

print('common_corpus:', common_corpus)

print('common_dictionary:', common_dictionary)
dic_tuple_list = list(zip(common_dictionary.keys(),common_dictionary.values()))
print('dict_list:', dic_tuple_list)