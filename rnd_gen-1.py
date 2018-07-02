import pandas as pd
from pandas.core.frame import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random

'''generate network using one of the following type of component with the'''
'''same network density'''
rnd_g1=nx.random_graphs.erdos_renyi_graph(n=50, p=0.08)
rnd_g2=nx.random_graphs.erdos_renyi_graph(n=50, p=0.08)

#rnd_g1=nx.random_graphs.barabasi_albert_graph(n=50, m=2)
#rnd_g2=nx.random_graphs.barabasi_albert_graph(n=50, m=2)

#rnd_g1=nx.random_graphs.watts_strogatz_graph(n=50, k=4, p=0.08)
#rnd_g2=nx.random_graphs.watts_strogatz_graph(n=50, k=4, p=0.08)

for (u, v) in rnd_g1.edges():
	rnd_g1.edge[u][v]['weight'] = int(np.random.chisquare(2))
for (u, v) in rnd_g2.edges():
	rnd_g2.edge[u][v]['weight'] = int(np.random.chisquare(2))


#take a look at the network
g_layout = nx.fruchterman_reingold_layout(rnd_g1)
e_weights=[rnd_g1[u][v]['weight'] for u,v in rnd_g1.edges()]
nx.draw(rnd_g1, g_layout, width=e_weights)
plt.show(block=False)

#connect the two components with randomly generated ties (relatively weak)
adj_mat1=nx.adjacency_matrix(rnd_g1).toarray()+2
adj_mat2=nx.adjacency_matrix(rnd_g2).toarray()+2
adj_mat1[adj_mat1==2]=0
adj_mat2[adj_mat2==2]=0
adj_mat_padding=np.random.chisquare(1,size=adj_mat1.shape)
adj_mat_padding[adj_mat_padding<5]=0#match the network density of the above two components
#regenerate weights for selected ties
for i_ in range(adj_mat_padding.shape[0]):
	for j_ in range(adj_mat_padding.shape[0]):
		if(adj_mat_padding[i_,j_]>0):
			adj_mat_padding[i_,j_]=int(np.random.chisquare(2))

#combine
adj_mat_full=np.vstack([np.hstack([adj_mat1,adj_mat_padding]),np.hstack([np.transpose(adj_mat_padding),adj_mat2])])

full_g=nx.from_numpy_matrix(adj_mat_full)
full_e_color=np.zeros(100)
full_e_color[0:50]=1
full_layout = nx.fruchterman_reingold_layout(full_g)
nx.draw(full_g,full_layout, node_color=full_e_color)
plt.show(block=False)

#generate association matrix based on tie strength
exp_adj=np.exp(2*adj_mat_full)
#avoid 0 rows
exp_adj[exp_adj==1]=0.01
sft_mx_adj=np.transpose(np.transpose(exp_adj)/np.sum(exp_adj,axis=1))

#initialize the two components with different values
attr_arr_last=np.hstack([np.random.uniform(low=0,high=1,size=50)+10,np.random.uniform(low=0,high=1,size=50)+1])
gap_iter=1
nb_iter=0
attr_arr=np.zeros(50)
while(gap_iter>0.001):
	attr_arr=0.5*np.dot(sft_mx_adj,attr_arr_last)+0.5*attr_arr_last
	gap_iter=np.mean(np.abs(attr_arr-attr_arr_last))
	attr_arr_last=attr_arr
	nb_iter=nb_iter+1
	if (nb_iter%1000==0):
		print(nb_iter,gap_iter)
	if (nb_iter>100000):
		break

#save the fully synced networks
np.savetxt(r'...\eds\adj_gen_smw.csv',adj_mat_full,delimiter=',')
np.savetxt(r'...\eds\nw_gen_smw.csv',attr_arr,delimiter=',')










