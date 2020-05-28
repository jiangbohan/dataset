import csv
import pandas as pd
import halp

# with open('/Users/chen/Desktop/train_csv_version(1).csv') as csv_file:
#     csv_reader = csv.reader(csv_file, delimiter=',')
#     line_count = 0
#     cat = {}
#     for row in csv_reader:
#         split = row[3].split(",")
#         for i in split:
#             if i not in cat:
#                 cat[i] = [row[0]]
#             else:
#                 cat[i].append(row[0])
#     CAT, ID = zip(*cat.items())
#     for i in range(len(CAT)):
#         print(CAT[i], "--", ID[i])
#
# with open('/Users/chen/Desktop/result.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     for i in range(len(CAT)):
#         writer.writerow([CAT[i],ID[i]])

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from halp.undirected_hypergraph import UndirectedHypergraph

df = pd.read_csv("C:/Users/user/PycharmProjects/dataset/yelp-dataset/train_csv_version.csv", names = ['id', 'label','statement',  'subject', 'speaker',     'job',     'state',   'party',   'barely_true_c',   'false_c', 'half_true_c', 'mostly_true_c',   'pants_on_fire_c', 'venue'])
df1 = pd.read_csv("C:/Users/user/PycharmProjects/dataset/yelp-dataset/result.csv", names = ['subject','edge'])

# df.info()
# print(df.head(10))

# Initialize an empty hypergraph
H = UndirectedHypergraph()

# NODE
node_list = df['id']
node_list1 = tuple(node_list)

# EDGE
node_subject = df1['edge']

# attribute_list = {"f1": df['speaker'],
#                   "f2": df['job'],
#                   "f3": df['state']}
# economy = df.loc[df['subject'] == 'economy']
# # list = economy['id']
hyperedge_list = (node_subject)
edge_list = list(zip(hyperedge_list,hyperedge_list.index))

# print(hyperedge_list)
node = H.add_nodes(node_list1)
hyperedge_ids = H.add_hyperedges(edge_list)
print(node)