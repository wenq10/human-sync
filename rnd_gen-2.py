import pandas as pd
from pandas.core.frame import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random


#generate
for i_ in range(100):
	'''generate network using one of the following type of component with the'''
	'''same network density'''
	#rnd_g1=nx.random_graphs.erdos_renyi_graph(n=50, p=0.08)
	#rnd_g2=nx.random_graphs.erdos_renyi_graph(n=50, p=0.08)

	#rnd_g1=nx.random_graphs.barabasi_albert_graph(n=50, m=2)
	#rnd_g2=nx.random_graphs.barabasi_albert_graph(n=50, m=2)

	rnd_g1=nx.random_graphs.watts_strogatz_graph(n=50, k=4, p=0.15)
	rnd_g2=nx.random_graphs.watts_strogatz_graph(n=50, k=4, p=0.15)

	#weight the network randomly
	for (u, v) in rnd_g1.edges():
		rnd_g1.edge[u][v]['weight'] = int(np.random.chisquare(2)+2)
	for (u, v) in rnd_g2.edges():
		rnd_g2.edge[u][v]['weight'] = int(np.random.chisquare(2)+2)

	#take a look at the network
	#g_layout = nx.fruchterman_reingold_layout(rnd_g1)
	#e_weights=[rnd_g1[u][v]['weight'] for u,v in rnd_g1.edges()]
	#nx.draw(rnd_g1, g_layout, width=e_weights)
	#plt.show(block=False)

	adj_mat1=nx.adjacency_matrix(rnd_g1).toarray()+2
	adj_mat2=nx.adjacency_matrix(rnd_g2).toarray()+2
	adj_mat_padding=np.random.chisquare(1,size=adj_mat1.shape)
	adj_mat_padding[adj_mat_padding<5]=0#match the network density of the above two components
	#regenerate weights for selected ties
	for i_ in range(adj_mat_padding.shape[0]):
		for j_ in range(adj_mat_padding.shape[0]):
			if(adj_mat_padding[i_,j_]>0):
				adj_mat_padding[i_,j_]=int(np.random.chisquare(2))
	#add links between two blocks
	adj_mat_full=np.vstack([np.hstack([adj_mat1,adj_mat_padding]),np.hstack([np.transpose(adj_mat_padding),adj_mat2])])

	#generate association matrix based on tie strength
	exp_adj=np.exp(2*adj_mat_full)
	#avoid 0 rows
	exp_adj[exp_adj==1]=0.01
	sft_mx_adj=np.transpose(np.transpose(exp_adj)/np.sum(exp_adj,axis=1))

	#initialize the two components with different values
	attr_arr_last=np.zeros(100)
	attr_arr_last[0:50]=5
	attr_arr_last[50:100]=-5
	gap_iter=1
	nb_iter=0
	attr_arr=np.zeros(100)
	#find behavior values that are fully synchronized under this network topology
	while(gap_iter>0.00001):
		attr_arr=0.5*np.dot(sft_mx_adj,attr_arr_last)+0.5*attr_arr_last
		gap_iter=np.mean(np.abs(attr_arr-attr_arr_last))
		attr_arr_last=attr_arr
		nb_iter=nb_iter+1
		if (nb_iter%1000==0):
			print(nb_iter,gap_iter)
		if (nb_iter>100000):
			break

	#inspect values of nodes
	#nx.draw(full_g,full_layout, node_color=attr_arr)
	#plt.show(block=False)
	#export to one of the folder
	#the data have already been generated and can be used directly
	np.savetxt(r'...\scf\network\adj_gen_smw'+str(i_)+'.csv',adj_mat_full,delimiter=',',fmt='%u')
	#record the fully synchronized behavior
	np.savetxt(r'...\scf\behavior\100sync\nw_gen_smw'+str(i_)+'.csv',attr_arr,delimiter=',')
	print(i_)

	#
	#noise level 0.1-0.9
	for j_ in range(1,10):
		R_noise=0.1*j_
		R_peer=1-R_noise

		attr_arr=attr_arr-np.mean(attr_arr)

		kesi=np.random.uniform(size=100)
		kesi=kesi-np.mean(kesi)

		#solve for the scaling factor alpha that add intended level of noise
		A=np.dot(attr_arr,attr_arr)
		B=np.dot(kesi,attr_arr)
		S_kesi=np.dot(sft_mx_adj,kesi)
		S_kesi=S_kesi-np.mean(S_kesi)
		C=np.dot(attr_arr,S_kesi)
		D=np.dot(kesi,S_kesi)
		E=np.dot(kesi,kesi)
		F=np.dot(S_kesi,S_kesi)
		alpha0=0
		alpha1=10
		nb_itr=0
		F_alpha0=((A+B*alpha0+C*alpha0+D*(alpha0**2))**2)-R_peer*(A+2*B*alpha0+E*(alpha0**2))*(A+2*C*alpha0+F*(alpha0**2))
		F_alpha1=((A+B*alpha1+C*alpha1+D*(alpha1**2))**2)-R_peer*(A+2*B*alpha1+E*(alpha1**2))*(A+2*C*alpha1+F*(alpha1**2))
		while(F_alpha1>0):
			alpha1=alpha1+10
			F_alpha1=((A+B*alpha1+C*alpha1+D*(alpha1**2))**2)-R_peer*(A+2*B*alpha1+E*(alpha1**2))*(A+2*C*alpha1+F*(alpha1**2))
			if(nb_itr%1000==999):
				#show potential non-convergence problem
				print(nb_itr,F_alpha1,'really long')
				time.sleep(5)
		nb_itr=0
		while((alpha1-alpha0)>0.01):
			#alpha_new=F_alpha0*(alpha1-alpha0)/(F_alpha0+abs(F_alpha1))+alpha0
			alpha_new=0.5*alpha0+0.5*alpha1
			F_alpha_new=((A+B*alpha_new+C*alpha_new+D*(alpha_new**2))**2)-R_peer*(A+2*B*alpha_new+E*(alpha_new**2))*(A+2*C*alpha_new+F*(alpha_new**2))
			if(F_alpha_new>0):
				F_alpha0=F_alpha_new
				alpha0=alpha_new
			else:
				F_alpha1=F_alpha_new
				alpha1=alpha_new
			nb_itr=nb_itr+1
			if(nb_itr%1000==0):
				print(nb_itr,np.abs(F_alpha0-F_alpha1))
				time.sleep(5)
		alpha_kesi=kesi*alpha1
		y_ns=alpha_kesi+attr_arr
		#save generated behvaior data (with intended noise level)
		np.savetxt(r'scf/behavior/'+str(10-j_)+'0sync/nw_gen_smw'+str(i_)+'.csv',y_ns*np.std(attr_arr)/np.std(y_ns),delimiter=',')






