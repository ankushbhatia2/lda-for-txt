from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
from math import log

tokenizer = RegexpTokenizer(r'\w+')

en_stop = get_stop_words('en')


p_stemmer = PorterStemmer()

#Add root folder path here
root = ''


f = open(root+'descriptions.txt', 'r')
doc_set = {}
i = 0
names = []
for r in f.readlines():
    if i%2 == 0:
        doc_set[r.strip()] = {}
        doc_set[r.strip()]['id'] = i
        names.append(r.strip())
        i+=1
    else:
        doc_set[names[int(i/2)]]['description'] = r.strip()
        i+=1

#Cleaning data
for k in doc_set:
    doc_set[k]['description'] = list(doc_set[k]['description'])
    for i in range(0, len(doc_set[k]['description'])):
        if doc_set[k]['description'][i] == '\\' and doc_set[k]['description'][i+1] == 'x':
            doc_set[k]['description'][i] = ' '
            doc_set[k]['description'][i+1] = ' '
            doc_set[k]['description'][i+2] = ' '
            doc_set[k]['description'][i+3] = ' '
        elif doc_set[k]['description'][i] == '\\' and doc_set[k]['description'][i+1] == 'n':
            doc_set[k]['description'][i] = ' '
            doc_set[k]['description'][i+1] = ' '
        elif doc_set[k]['description'][i] in '.,()-!@#$%^&*_+=~`"1234567890:/\\\';':
            doc_set[k]['description'][i] = ' '
    doc_set[k]['description'] = ''.join(doc_set[k]['description'])

for i in names:
    print(i)
    print(doc_set[i]['description'])

#Getting data
doc_set_l = []
for i in range(len(names)):
    for k in doc_set:
        if doc_set[k]['id'] == i:
            doc_set_l.append(doc_set[k]['description'])

#list for tokenized documents in loop
texts = []

#loop through document list
for i in doc_set:
    # clean and tokenize document string
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)
    stopped_tokens = [i for i in tokens if not i in en_stop]

    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]

    texts.append(stemmed_tokens)

dictionary = corpora.Dictionary(texts)

corpus = [dictionary.doc2bow(text) for text in texts]

ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=2, id2word=dictionary, passes=10)

for i in range(ldamodel.num_topics):
    print(ldamodel.print_topic(i))

for i in ldamodel[corpus]:
    print(type(i))
