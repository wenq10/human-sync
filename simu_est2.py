import tensorflow as tf
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
import matplotlib.pyplot as plt
import networkx as nx
import time

#configure optimizor
Bat_Siz=50
X_input=tf.placeholder(dtype=tf.float32,shape=[Bat_Siz,None,2])
y_input=tf.placeholder(dtype=tf.float32,shape=[Bat_Siz,1])

W_inner=tf.Variable(np.array([0]).astype(np.float32).reshape(1,1),name='coefs_w_inner')

inner_mul=tf.reshape(tf.matmul(tf.reshape(X_input[:,:,1:2],[-1,1]),W_inner),[Bat_Siz,-1])

inner_softmax=tf.nn.softmax(inner_mul)

inner_out=tf.matmul(tf.reshape(X_input[:,:,0],[Bat_Siz,1,-1]),tf.reshape(inner_softmax,[Bat_Siz,-1,1]))

y_hat=tf.reshape(inner_out,[Bat_Siz,1])

loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_input - y_hat),reduction_indices=[1]))

train_step=tf.train.RMSPropOptimizer(0.01).minimize(loss)

init = tf.global_variables_initializer()


#estimated individually
nb_folds=100
corr_coef=np.zeros([nb_folds,10])
for i_ in range(10,nb_folds):

	#load simulated network (weighted by tie strength) data from one of the folds: smw, scf or eds
	drt_mat_sym=np.loadtxt(r".../network/adj_gen_smw"+str(i_)+".csv", delimiter=",")
	#delete (possible) isolators
	drt_rowsums=np.sum(drt_mat_sym,axis=1)
	del_indcs_sum=np.argwhere(drt_rowsums==0).reshape(-1)
	drt_mat_sym=np.delete(drt_mat_sym,del_indcs_sum,axis=1)
	drt_mat_sym=np.delete(drt_mat_sym,del_indcs_sum,axis=0)

	for j_ in range(10):
		#load simulated behavior data from one of the folds: smw, scf or eds
		night_wk_arr=np.loadtxt(r".../behavior/"+str(j_+1)+"0sync/nw_gen_smw"+str(i_)+".csv", delimiter=",")
		night_wk_arr=np.delete(night_wk_arr,del_indcs_sum,axis=0)

		#convert to training data
		x_data=np.zeros([drt_mat_sym.shape[0],5,2])
		peer_lth=night_wk_arr.copy()
		for i in range(drt_mat_sym.shape[0]):
			tempt_dumy=drt_mat_sym[i,:].copy()
			peer_indcs=np.argwhere(tempt_dumy>0).reshape(-1)
			if (peer_indcs.shape[0]>=5):
				peer_indcs=np.argsort(tempt_dumy)[-5:]
			tempt_drt=np.zeros(5)
			peer_nw=np.zeros(5)
			tempt_drt[-peer_indcs.shape[0]:]=tempt_dumy[peer_indcs].copy()+2
			peer_nw[-peer_indcs.shape[0]:]=peer_lth[peer_indcs].copy()
			x_data[i,:,:]=np.transpose(np.vstack([peer_nw,tempt_drt]))

		##convert to input data
		x_data_use=np.vstack([x_data for i in range(50)])

		#behavior data
		y_data = night_wk_arr.copy()
		y_data = y_data.reshape([-1,1])

		#behavior data used for training
		y_data_use=np.vstack([y_data for i in range(50)])

		#run
		sess=tf.Session()
		sess.run(init)
		last_wi=np.zeros(1)+1
		nb_epoch=100
		for j in range(5000):#maximum 5000 iterations
			sess.run(train_step,feed_dict={X_input:x_data_use[((j%nb_epoch)*50):((j%nb_epoch)*50+50),:,:],y_input:y_data_use[((j%nb_epoch)*50):((j%nb_epoch)*50+50),:]})
			if (j%nb_epoch==0):
				tempt_wi=sess.run(W_inner)
				#print(j,tempt_wi)
				if(np.mean(np.abs(last_wi-tempt_wi))/np.mean(np.abs(last_wi))<0.01):
					print(j,tempt_wi)
					break
				else:
					last_wi=tempt_wi

		#obtain predictions
		ful_outcom=np.zeros(100)
		for i in range(2):
			ful_outcom[(i*50):(i*50+50)]=sess.run(y_hat,feed_dict={X_input:x_data_use[(i*50):(i*50+50),:,:],y_input:y_data_use[(i*50):(i*50+50),:]}).reshape(-1)
		ful_outcom=ful_outcom[0:x_data.shape[0]]

		sess.close()
		#record correlation between real and predicted values
		corr_coef[i_,j_]=np.corrcoef(y_data.reshape(-1),ful_outcom)[0,1]
	print(i_)

np.savetxt(r'.../smw_corrcoef.csv',corr_coef,delimiter=',')
















