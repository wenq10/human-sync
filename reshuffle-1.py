#model estimation
import tensorflow as tf
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
import matplotlib.pyplot as plt
import networkx as nx
import os
from matplotlib import colors
from scipy.stats import kstest
from scipy.stats import ks_2samp
from scipy.stats import mannwhitneyu
from scipy.stats import chi2_contingency
import random


#combined full dataset
x_data_full=np.vstack(x_data)
y_data_full=np.vstack(y_data)

#reindex before training
reindex=random.sample(xrange(y_data_full.shape[0]),y_data_full.shape[0])
x_data_use=x_data_full[reindex,:,:]
y_data_use=y_data_full[reindex,:]


#build model with tf
Bat_Siz=50
X_input=tf.placeholder(dtype=tf.float32,shape=[Bat_Siz,None,5])
y_input=tf.placeholder(dtype=tf.float32,shape=[Bat_Siz,1])

W_inner=tf.Variable(np.array([0,0,0]).astype(np.float32).reshape(4,1),name='coefs_w_inner')

#X*beta
inner_mul=tf.reshape(tf.matmul(tf.reshape(X_input[:,:,1:5],[-1,4]),W_inner),[Bat_Siz,-1])

#generate weights based on X*beta
peer_weights=tf.nn.softmax(inner_mul)

#weighted sum of peers' behaviors
inner_out=tf.matmul(tf.reshape(X_input[:,:,0],[Bat_Siz,1,-1]),tf.reshape(peer_weights,[Bat_Siz,-1,1]))

#reshape predicted values
y_hat=tf.reshape(inner_out,[Bat_Siz,1])

loss = tf.reduce_mean(tf.square(y_input - y_hat))

train_step=tf.train.RMSPropOptimizer(0.01).minimize(loss)

init = tf.global_variables_initializer()


#=====================================reshuffle test====================================#
#reshuffle for 20000 folds
nb_folds=20000

# day level full metrics
cor_full=np.zeros(nb_folds)
acc_full=np.zeros(nb_folds)
F1_full=np.zeros(nb_folds)
avg_err_full=np.zeros(nb_folds)
chisq_full=np.zeros(nb_folds)

# day * fold metrics for each day
cor_day=np.zeros([nb_folds,len(x_data)])
acc_day=np.zeros([nb_folds,len(x_data)])
F1_day=np.zeros([nb_folds,len(x_data)])
avg_err_day=np.zeros([nb_folds,len(x_data)])
chisq_day=np.zeros([nb_folds,len(x_data)])

night_wk_full=[]
for day_j in range(len(y_data)):
	night_wk_tempt=pd.read_csv(os.path.join(path_nw,files_nw[day_j]))
	night_wk_full.append(np.array(night_wk_tempt.nightwork_durations).reshape([-1,1]))


for fd_i in range(nb_folds):
	y_data_rsf=[]

	for day_k in range(len(y_data)):
		y_data_tempt_rsf=night_wk_full[day_k]
		rsf_reindex=random.sample(xrange(y_data_tempt_rsf.shape[0]),y_data_tempt_rsf.shape[0])#reshuffle for each day
		y_data_rsf.append(y_data_tempt_rsf[rsf_reindex,:][day_k,:])

	y_data_full_rsf=np.vstack(y_data_rsf)
	y_data_use_rsf=y_data_full_rsf[reindex,:]

	# start training
	sess=tf.Session()
	sess.run(init)
	last_wi=np.zeros(3)+1
	for j in range(50000):#in case of non-convergence with randomly reshuffled data, have a large upper bound of 50000 iterations
		sess.run(train_step,feed_dict={X_input:x_data_use[((j%nb_epochs)*50):((j%nb_epochs)*50+50),:,:],y_input:y_data_use_rsf[((j%nb_epochs)*50):((j%nb_epochs)*50+50),:]})
		if (j%nb_epochs==0):
			tempt_wi=sess.run(W_inner).reshape(-1)
			print(j,sess.run(W_inner),sess.run(loss,feed_dict={X_input:x_data_use[((j%nb_epochs)*50):((j%nb_epochs)*50+50),:,:],y_input:y_data_use_rsf[((j%nb_epochs)*50):((j%nb_epochs)*50+50),:]}))
			delta=np.mean(np.abs(last_wi-tempt_wi))/np.mean(np.abs(last_wi))
			if(delta<0.01):
				#relative change less than 1 percent
				break

	ful_outcom_rsf=np.zeros(y_data_rsf.shape[0])
	for i in range(nb_epochs):
		ful_outcom_rsf[(i*50):(i*50+50)]=sess.run(y_hat,feed_dict={X_input:x_data_full[(i*50):(i*50+50),:,:],y_input:y_data_full_rsf[(i*50):(i*50+50),:]}).reshape(-1)
		sup_outcome_rsf=sess.run(y_hat,feed_dict={X_input:x_data_full[-50:,:,:],y_input:y_data_full_rsf[-50:,:]})[-16:]
	ful_outcom_rsf=np.vstack([ful_outcom_rsf.reshape([-1,1]),sup_outcome_rsf]).reshape(-1)

	cor_full[fd_i]=np.corrcoef(y_data_full_rsf.reshape(-1),ful_outcom_rsf)[0,1]

	bchm_rsf=DataFrame({'real':y_data_full_rsf.reshape(-1),'pdct':ful_outcom_rsf})

	tp_full_rsf=np.argwhere((bchm_rsf.real>=6)&(bchm_rsf.pdct>=6)).shape[0]
	fp_full_rsf=np.argwhere((bchm_rsf.real<6)&(bchm_rsf.pdct>=6)).shape[0]
	tn_full_rsf=np.argwhere((bchm_rsf.real<6)&(bchm_rsf.pdct<6)).shape[0]
	fn_full_rsf=np.argwhere((bchm_rsf.real>=6)&(bchm_rsf.pdct<6)).shape[0]

	chisq_full[fd_i]=chi2_contingency(np.array([[tn_full_rsf,fn_full_rsf],[fp_full_rsf,tp_full_rsf]]))[0]
	F1_full[fd_i]=tp_full_rsf*2.0/(2*tp_full_rsf+fp_full_rsf+fn_full_rsf)
	acc_full[fd_i]=(tp_full_rsf+tn_full_rsf)*1.0/bchm_rsf.shape[0]
	avg_err_full[fd_i]=np.mean(np.abs(bchm_rsf.real-bchm_rsf.pdct))

	cur_sum=0
	y_pdct_rsf=[]
	for day_l in range(len(y_data_rsf)):
		y_pdct_rsf.append(ful_outcom_rsf[cur_sum:(cur_sum+len(y_data_rsf[day_l]))])
		cur_sum=cur_sum+len(y_data_rsf[day_j])

	for day_m in range(len(y_data)):
		bchm_tempt=DataFrame({'real':y_data_rsf[day_m].reshape(-1),'pdct':y_pdct_rsf[day_m].reshape(-1)})
		tp=bchm_tempt[(bchm_tempt.real>=6)&(bchm_tempt.pdct>=6)].shape[0]
		tn=bchm_tempt[(bchm_tempt.real<6)&(bchm_tempt.pdct<6)].shape[0]
		fp=bchm_tempt[(bchm_tempt.real<6)&(bchm_tempt.pdct>=6)].shape[0]
		fn=bchm_tempt[(bchm_tempt.real>=6)&(bchm_tempt.pdct<6)].shape[0]
		try:
			F1_day[fd_i,day_m]=(tp*2.0)/(tp*2+fp+fn)
		except:
			F1_day[fd_i,day_m]=0
		acc_day[fd_i,day_m]=(tp+tn*1.0)/bchm_tempt.shape[0]
		try:
			chisq_day[fd_i,day_m],pval_rsf,dof,ex=chi2_contingency(np.array([[tn,fn],[fp,tp]]))
		except:
			chisq_day[fd_i,day_m]=-1
		cor_day[fd_i,day_m]=np.corrcoef(bchm_tempt.real,bchm_tempt.pdct)[0,1]
		avg_err_day[fd_i,day_m]=np.mean(np.abs(bchm_tempt.real-bchm_tempt.pdct))

	sess.close()
	print(fd_i,cor_full[fd_i])

# reshuffle upper bounds p=0.05
cor_full[np.argsort(cor_full)[-500]]
acc_full[np.argsort(acc_full)[-500]]
F1_full[np.argsort(F1_full)[-500]]
avg_err_full[np.argsort(avg_err_full)[500]]
chisq_full[np.argsort(chisq_full)[-500]]


#reshuffle upper bounds for each day
upper_cor=np.zeros(len(x_data))
med_cor=np.zeros(len(x_data))
lower_cor=np.zeros(len(x_data))
upper_acc=np.zeros(len(x_data))
med_acc=np.zeros(len(x_data))
lower_acc=np.zeros(len(x_data))
upper_F1=np.zeros(len(x_data))
med_F1=np.zeros(len(x_data))
lower_F1=np.zeros(len(x_data))
upper_avgerr=np.zeros(len(x_data))
med_avgerr=np.zeros(len(x_data))
lower_avgerr=np.zeros(len(x_data))
upper_chisq=np.zeros(len(x_data))
med_chisq=np.zeros(len(x_data))
lower_chisq=np.zeros(len(x_data))

for day_t in range(len(x_data)):
	tempt_cor=cor_day[:,day_t]
	tempt_acc=acc_day[:,day_t]
	tempt_F1=F1_day[:,day_t]
	tempt_avgerr=avg_err_day[:,day_t]
	tempt_chisq=chisq_day[:,day_t]
	# with p=0.05 significance level
	upper_cor[day_t]=tempt_cor[np.argsort(tempt_cor)[-500]]
	med_cor[day_t]=tempt_cor[np.argsort(tempt_cor)[10000]]
	lower_cor[day_t]=tempt_cor[np.argsort(tempt_cor)[500]]
	upper_acc[day_t]=tempt_acc[np.argsort(tempt_acc)[-500]]
	med_acc[day_t]=tempt_acc[np.argsort(tempt_acc)[10000]]
	lower_acc[day_t]=tempt_acc[np.argsort(tempt_acc)[500]]
	upper_F1[day_t]=tempt_F1[np.argsort(tempt_F1)[-500]]
	med_F1[day_t]=tempt_F1[np.argsort(tempt_F1)[10000]]
	lower_F1[day_t]=tempt_F1[np.argsort(tempt_F1)[500]]
	upper_avgerr[day_t]=tempt_avgerr[np.argsort(tempt_avgerr)[-500]]
	med_avgerr[day_t]=tempt_avgerr[np.argsort(tempt_avgerr)[10000]]
	lower_avgerr[day_t]=tempt_avgerr[np.argsort(tempt_avgerr)[500]]
	upper_chisq[day_t]=tempt_chisq[np.argsort(tempt_chisq)[-500]]
	med_chisq[day_t]=tempt_chisq[np.argsort(tempt_chisq)[10000]]
	lower_chisq[day_t]=tempt_chisq[np.argsort(tempt_chisq)[500]]














