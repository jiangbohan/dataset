import numpy as np
import scipy.io as sio
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
import string
import re

df = pd.read_csv("D:/pycharm/dataset/yelp-dataset/train_400.csv", names = ['id', 'label', 'statement', 'subject', 'speaker', 'job', 'state', 'party', 'barely_true_c', 'false_c', 'half_true_c', 'mostly_true_c', 'pants_on_fire_c', 'venue'])
# result.csv is the preprocessed dict(hyperedge collection) from raw data with 143 subject * id , e.g. economy {1,2,3,4,5}, {education: 2,4,6,8,10}
df1 = pd.read_csv("D:/pycharm/dataset/yelp-dataset/result_400.csv", names = ['subject', 'edge'])

# Create the adjacency matrix with zeros

A_hypergraph = np.zeros((8645, 142))
A_simple_graph = np.zeros((8645, 8645))
G_T = np.zeros((8645, 1))
tfs_array = np.zeros((8645,4223))
rate_list = np.zeros((8645,5))

id_index = {}
def get_index(id):
    if id not in id_index:
        id_index[id] = df.loc[df.id == id].index[0]
    return id_index[id]

# Generate A_Simple_Graph: 10240*10240
for index, row in df1.iterrows():
    IDlist = row.values[1].replace("[", "").replace("]","").replace(" ","").replace("'","").split(",")
    # print(index)
    for k, i in enumerate(IDlist):
        if i in df.id.values:
            id_i = get_index(i)
            for j in IDlist[k:]:
                id_j = get_index(j)
                A_simple_graph[id_i, id_j] = 1
                A_simple_graph[id_j, id_i] = 1

print('simple_graph AM finished!')

# Generate Adjacency Matrix A: 10240*143
for index, row in df1.iterrows():
    IDlist = row.values[1].replace("[", "").replace("]","").replace(" ","").replace("'","").split(",")

    for i in IDlist:
        if i in df.id.values:
            id_i = get_index(i)
            A_hypergraph[id_i, index] = 1

print('hyper_graph AM finished!')


# Generate Ground Truth Matrix gnd: 10240*1
false_list = df.index[df['label']== 'false'].tolist()
true_list1 = df.index[df['label']== 'half-true'].tolist()
true_list2 = df.index[df['label']== 'mostly-true'].tolist()
true_list3 = df.index[df['label']== 'true'].tolist()
true_list4 = df.index[df['label']== 'barely-true'].tolist()


# print(kk)
for i in false_list:
    G_T[i,0] = 1

for i in true_list1:
    G_T[i,0] = 2

for i in true_list2:
    G_T[i,0] = 3

for i in true_list3:
    G_T[i,0] = 4

for i in true_list4:
    G_T[i,0] = 5


print('Label matrix finished!')

# tf-idf feature matrix 10240* 4622
def tokenize(text):
    text = text.translate(str.maketrans("", ""))
    tokens = word_tokenize(text)
    sents_no_punct = [s for s in tokens if s not in string.punctuation]
    stems = []
    pattern = '[0-9]'
    lst = [re.sub(pattern, '', i) for i in sents_no_punct]
    for item in lst:
        stems.append(PorterStemmer().stem(item))
    return stems

en_stop = set()
for line in open('stopwords_long.txt'):
    en_stop.add(line.strip())

X = {x: "" for x in range(8645)}
# count = 0
for index, row in df.iterrows():
    X[index] = X[index] + " " + row.statement

vectorizer = TfidfVectorizer(token_pattern="[a-zA-Z]+", min_df=2, max_df=0.7, lowercase=True, use_idf=True,
                             stop_words=en_stop, tokenizer=tokenize, max_features=5000)
tfs = vectorizer.fit_transform(X.values())
tfs_array = tfs.toarray()

lst = vectorizer.get_feature_names()

print('tf-idf Feature Matrix finished!')

# rating features
rate_list = df['barely_true_c'].to_frame().join(df['false_c']).join(df['half_true_c']).join(df['mostly_true_c']).join(df['pants_on_fire_c'])
a = rate_list.to_numpy()
print('rating Feature matrix finished!')
# df2 = np.concatenate((A_hypergraph).T, (G_T).T, (A_simple_graph).T, (tfs_array).T)
#

sG_t = sparse.csr_matrix(G_T)
sA_simple_graph = sparse.csr_matrix(A_simple_graph)
sa = sparse.csr_matrix(a)
stfs_array = sparse.csr_matrix(tfs_array)


sio.savemat('Liar_400.mat', {'Label': G_T, 'Network': sA_simple_graph, 'Attributes': stfs_array})
print('Liar.mat finished!')