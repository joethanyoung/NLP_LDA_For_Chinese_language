# NLP_LDA_ANAlysis_主题分析

# 
import jieba
import pandas as pd
import os
import re
from gensim import corpora
from gensim.models import LdaModel
from gensim.models import CoherenceModel
from gensim.models import TfidfModel
import pyLDAvis.gensim_models

# set working envs
os.chdir('C:/Users/zhuoxun.yang001/Downloads/第七期常规/江西分公司S2332373')

# Load data
data = pd.read_excel(r"L06103基础表投诉处理清单.xlsx", skiprows=1)

# Load stop words
with open('stop_words.txt', 'r', encoding='utf-8') as f:
    stopwords = [line.strip() for line in f]

# Clean text
data['cleaned'] = data['投诉事由'].apply(lambda x: re.sub('[^\u4e00-\u9fa5]', '', x))

# Tokenize and remove stop words
data['words'] = data['cleaned'].apply(lambda x: [word for word in jieba.cut(x) if word not in stopwords and len(word) > 2])

# Create dictionary
dictionary = corpora.Dictionary(data['words'])

# Create corpus
corpus = [dictionary.doc2bow(text) for text in data['words']]

# Apply TF-IDF
tfidf = TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

# Train LDA model
lda = LdaModel(corpus_tfidf, id2word=dictionary, num_topics=10)

# Print topics
topics = lda.print_topics()
for topic in topics:
    print(topic)

# Evaluate model
coherence_model_lda = CoherenceModel(model=lda, texts=data['words'], dictionary=dictionary, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)

# Visualize topics
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim_models.prepare(lda, corpus, dictionary)

