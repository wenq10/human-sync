import tensorflow as tf
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
import os
import random



#read data
#x_data_full: email network tie strength measures
#y_data_full: overtime work
x_data_full=np.vstack(x_data)
y_data_full=np.vstack(y_data)

#reindex before
reindex=random.sample(xrange(y_data_full.shape[0]),y_data_full.shape[0])

x_data_use=x_data_full[reindex,:,:]
y_data_use=y_data_full[reindex,:]


#build optimizor
Bat_Siz=60

X_input=tf.placeholder(dtype=tf.float32,shape=[Bat_Siz,None,8])

y_input=tf.placeholder(dtype=tf.float32,shape=[Bat_Siz,1])

W_inner=tf.Variable(np.array([0,0,0,0,0,0,0]).astype(np.float32).reshape(7,1),name='coefs_w_inner')

inner_mul=tf.reshape(tf.matmul(tf.reshape(X_input[:,:,1:8],[-1,7]),W_inner),[Bat_Siz,-1])

peer_weight=tf.nn.softmax(inner_mul)

inner_out=tf.matmul(tf.reshape(X_input[:,:,0],[Bat_Siz,1,-1]),tf.reshape(peer_weight,[Bat_Siz,-1,1]))

y_hat=tf.reshape(inner_out,[Bat_Siz,1])

loss = tf.reduce_mean(tf.abs(y_input - y_hat))
#loss = tf.reduce_mean(tf.square(y_input - y_hat))

train_step=tf.train.RMSPropOptimizer(0.01).minimize(loss)

init = tf.global_variables_initializer()



#reshuffle y_data
nb_folds=20000

F1_rsf=np.zeros([nb_folds,len(x_data)])
Acc_rsf=np.zeros([nb_folds,len(x_data)])
chisq_rsf=np.zeros([nb_folds,len(x_data)])
pval_rsf=np.zeros([nb_folds,len(x_data)])
#full model evaluation
corr_all=np.zeros(nb_folds)
F1_all=np.zeros(nb_folds)
chisq_all=np.zeros(nb_folds)

night_wk_full=[]
for day_k in range(len(y_data)):
	night_wk_tempt=pd.read_csv(os.path.join(path_nw,files_nw[day_k]),delimiter=",")
	night_wk_arr_tempt=np.array(night_wk_tempt.night_drt)
	night_wk_full.append(night_wk_arr_tempt)

for rsf_i in range(nb_folds):
	y_data_rsf=[]
	for day_r in range(len(x_data)):
		rsf_inds=random.sample(xrange(night_wk_full[day_r].shape[0]),night_wk_full[day_r].shape[0])
		nw_rsf_tempt=night_wk_full[day_r][rsf_inds]#reshuffled full local
		y_data_rsf.append(nw_rsf_tempt.reshape([-1,1]))

	y_data_rsf_out=np.vstack(y_data_rsf)
	y_data_rsf_use=y_data_rsf_out[reindex,:]

	#run the model
	sess=tf.Session()
	sess.run(init)
	last_wi=np.zeros(7)+1
	for j in range(50000):
		sess.run(train_step,feed_dict={X_input:x_data_use[((j%nb_epochs)*60):((j%nb_epochs)*60+60),:,:],y_input:y_data_rsf_use[((j%nb_epochs)*60):((j%nb_epochs)*60+60),:]})
		if (j%nb_epochs==0):
			tempt_wi=sess.run(W_inner).reshape(-1)
			if(np.mean(np.abs(last_wi-tempt_wi))/np.mean(np.abs(last_wi))<0.01):
				print(j)
				break
			else:
				last_wi=tempt_wi

	ful_outcom_rsf=np.zeros(y_data_rsf.shape[0])
	for i in range(nb_epochs):
		ful_outcom_rsf[(i*60):(i*60+60)]=sess.run(y_hat,feed_dict={X_input:x_data_full[(i*60):(i*60+60),:,:],y_input:y_data_rsf_out[(i*60):(i*60+60),:]}).reshape(-1)
	sup_outcom_rsf=sess.run(y_hat,feed_dict={X_input:x_data_full[-60:,:,:],y_input:y_data_rsf_out[-60:,:]}).reshape(-1)[-7:]
	ful_outcom_rsf=np.concatenate([ful_outcom_rsf,sup_outcom_rsf],axis=0)
	
	#full model evaluation
	corr_all[rsf_i]=np.corrcoef(y_data_rsf_out.reshape(-1),ful_outcom_rsf)[0,1]
	bchm_rsf=DataFrame({'real':y_data_rsf_out.reshape(-1),'pdct':ful_outcom_rsf})
	tp_all=np.argwhere((bchm_rsf.real>0.5)&(bchm_rsf.pdct>=0.5)).shape[0]
	tn_all=np.argwhere((bchm_rsf.real<0.5)&(bchm_rsf.pdct<0.5)).shape[0]
	fp_all=np.argwhere((bchm_rsf.real<0.5)&(bchm_rsf.pdct>=0.5)).shape[0]
	fn_all=np.argwhere((bchm_rsf.real>0.5)&(bchm_rsf.pdct<0.5)).shape[0]
	chisq_all[rsf_i],pval_all,dof_all,ex_all=chi2_contingency(np.array([[tn_all,fn_all],[fp_all,tp_all]]))
	F1_all[rsf_i]=2.0*tp_all/(2*tp_all+fp_all+fn_all)

	#split predictions
	cur_sum=0
	y_pdct_rsf=[]
	for day_j in range(len(x_data)):
		y_pdct_rsf.append(ful_outcom_rsf[cur_sum:(cur_sum+len(y_data_rsf[day_j]))])
		cur_sum=cur_sum+len(y_data_rsf[day_j])

	for day_i in range(len(y_data)):
		bchm_tempt=DataFrame({'real':y_data_rsf[day_i].reshape(-1),'pdct':y_pdct_rsf[day_i].reshape(-1)})
		tp=bchm_tempt[(bchm_tempt.real>=0.5)&(bchm_tempt.pdct>=0.5)].shape[0]
		tn=bchm_tempt[(bchm_tempt.real<0.5)&(bchm_tempt.pdct<0.5)].shape[0]
		fp=bchm_tempt[(bchm_tempt.real<0.5)&(bchm_tempt.pdct>=0.5)].shape[0]
		fn=bchm_tempt[(bchm_tempt.real>=0.5)&(bchm_tempt.pdct<0.5)].shape[0]
		try:
			F1_rsf[rsf_i,day_i]=(tp*2.0)/(tp*2+fp+fn)
		except:
			F1_rsf[rsf_i,day_i]=0
		Acc_rsf[rsf_i,day_i]=(tp+tn*1.0)/bchm_tempt.shape[0]
		try:
			chisq_rsf[rsf_i,day_i],pval_rsf[rsf_i,day_i],dof,ex=chi2_contingency(np.array([[tn,fn],[fp,tp]]))
		except:
			chisq_rsf[rsf_i,day_i]=-1
			pval_rsf[rsf_i,day_i]=1

	sess.close()
	print(rsf_i,F1_all[rsf_i])

#store the full reshuffle-simulation results



#whole model evaluation, compare to 0.0001
corr_all[np.argsort(corr_all)[-1]]
F1_all[np.argsort(F1_all)[-1]]
chisq_all[np.argsort(chisq_all)[-1]]

#===================================day-by-day sync strength
upperF1=np.zeros(len(x_data))
lowerF1=np.zeros(len(x_data))
medF1=np.zeros(len(x_data))
upperchisq=np.zeros(len(x_data))
lowerchisq=np.zeros(len(x_data))
medchisq=np.zeros(len(x_data))
for f in range(len(x_data)):
	tempt_F1=F1_rsf[:,f]
	tempt_chisq=chisq_rsf[:,f]
	upperF1[f]=tempt_F1[np.argsort(tempt_F1)[-500]]
	lowerF1[f]=tempt_F1[np.argsort(tempt_F1)[500]]
	medF1[f]=tempt_F1[np.argsort(tempt_F1)[10000]]
	upperchisq[f]=tempt_chisq[np.argsort(tempt_chisq)[-500]]
	lowerchisq[f]=tempt_chisq[np.argsort(tempt_chisq)[500]]
	medchisq[f]=tempt_chisq[np.argsort(tempt_chisq)[10000]]

rslt_eval=DataFrame({'F1':F1,'Acc':Acc,'chi2':chisq,'p-val':pval,'F1_crit':F1_criterion,'chisq_crit':chisq_criterion,'nb_nw':nb_nw,
	'upperF1':upperF1,'lowerF1':lowerF1,'upperchisq':upperchisq,'lowerchisq':lowerchisq})

rslt_eval.to_csv(r'.../nw_result_evaluation_indices.csv')




















