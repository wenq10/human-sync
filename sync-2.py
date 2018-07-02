import tensorflow as tf
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
import os
import random
from scipy.stats import pearsonr
import networkx as nx

#/.../ is the current file path
path_xdata=r".../x_data/"
path_nw=r".../y_data/"


files_xdata=os.listdir(path_xdata)
files_xdata.sort()

files_nw=os.listdir(path_nw)
files_nw.sort()


#====================================retrain model with only significantly synced days========
#load the org-level sync indicators generated after reshuffling 
rslt_eval=pd.read_csv(r'.../result_evaluation_indices.csv')
#select the significantly synced days, strict chisq criterion
chisq_criterion=np.array(rslt_eval.chisq_crit)

x_data_sy=[]
y_data_sy=[]

for day_i in range(len(files_nw)):
	if (chisq_criterion[day_i]==1):
		night_wk=pd.read_csv(os.path.join(path_nw,files_nw[day_i]),delimiter=",")
		night_wk_arr=np.array(night_wk)[:,0]
		x_data_tempt=np.loadtxt(os.path.join(path_xdata,files_xdata[day_i]), delimiter=",")
		y_data_tempt=night_wk_arr.reshape(night_wk_arr.shape[0],1)
		x_data_sy.append(x_data_tempt)
		y_data_sy.append(y_data_tempt)
		if(day_i%10==0): print(day_i)

#combine
x_data_full_sy=np.vstack(x_data_sy)
y_data_full_sy=np.vstack(y_data_sy)

reindex_sy=random.sample(xrange(y_data_full_sy.shape[0]),y_data_full_sy.shape[0])

x_data_use_sy=x_data_full_sy[reindex_sy,:,:]
y_data_use_sy=y_data_full_sy[reindex_sy,:]



#sgd estimation
Bat_Siz=60
X_input=tf.placeholder(dtype=tf.float32,shape=[Bat_Siz,None,8])
y_input=tf.placeholder(dtype=tf.float32,shape=[Bat_Siz,1])

W_inner=tf.Variable(np.array([0,0,0,0,0,0,0]).astype(np.float32).reshape(7,1),name='coefs_w_inner')

inner_mul=tf.reshape(tf.matmul(tf.reshape(X_input[:,:,1:8],[-1,7]),W_inner),[Bat_Siz,-1])

inner_softmax=tf.nn.softmax(inner_mul)

inner_out=tf.matmul(tf.reshape(X_input[:,:,0],[Bat_Siz,1,-1]),tf.reshape(inner_softmax,[Bat_Siz,-1,1]))

y_hat=tf.reshape(inner_out,[Bat_Siz,1])

loss = tf.reduce_mean(tf.reduce_sum(tf.abs(y_input - y_hat),reduction_indices=[1]))

train_step=tf.train.RMSPropOptimizer(0.01).minimize(loss)

init = tf.global_variables_initializer()





sess=tf.Session()
sess.run(init)
last_wi=np.zeros(7)+1
for j in range(30000):
	sess.run(train_step,feed_dict={X_input:x_data_use_sy[((j%nb_epochs)*60):((j%nb_epochs)*60+60),:,:],y_input:y_data_use_sy[((j%nb_epochs)*60):((j%nb_epochs)*60+60),:]})
	if (j%nb_epochs==0):
		tempt_wi=sess.run(W_inner).reshape(-1)
		if(np.mean(np.abs(last_wi-tempt_wi))/np.mean(np.abs(last_wi))<0.01):
			print(j)
			break
		else:
			last_wi=tempt_wi





#coefficients
wi=sess.run(W_inner)

outcome_softmax=sess.run(inner_softmax,feed_dict={X_input:x_data_full_sy[0:60,:,:],y_input:y_data_full_sy[0:60,:]})
for bt in range(1,nb_epochs):
	outcome_softmax=np.vstack([outcome_softmax,sess.run(inner_softmax,feed_dict={X_input:x_data_full_sy[(60*bt):(60*bt+60),:,:],y_input:y_data_full_sy[(60*bt):(60*bt+60),:]})])
outcome_softmax=np.vstack([outcome_softmax,sess.run(inner_softmax,feed_dict={X_input:x_data_full_sy[-60:,:,:],y_input:y_data_full_sy[-60:,:]})[-32:,:]])

ful_outcom=np.zeros(y_data_full_sy.shape[0])
for i in range(nb_epochs):
	ful_outcom[(i*60):(i*60+60)]=sess.run(y_hat,feed_dict={X_input:x_data_full_sy[(i*60):(i*60+60),:,:],y_input:y_data_full_sy[(i*60):(i*60+60),:]}).reshape(-1)
sup_outcom=sess.run(y_hat,feed_dict={X_input:x_data_full_sy[-60:,:,:],y_input:y_data_full_sy[-60:,:]}).reshape(-1)[-32:]
ful_outcom=np.concatenate([ful_outcom,sup_outcom],axis=0)


plt.scatter(y_data_full_sy,ful_outcom)
plt.show()

np.corrcoef(y_data_full_sy.reshape(-1),ful_outcom)
bchm=DataFrame({'real':y_data_full_sy.reshape(-1),'pdct':ful_outcom})

tp=np.argwhere((bchm.real>0.5)&(bchm.pdct>=0.5)).shape[0]
tn=np.argwhere((bchm.real<0.5)&(bchm.pdct<0.5)).shape[0]
fp=np.argwhere((bchm.real<0.5)&(bchm.pdct>=0.5)).shape[0]
fn=np.argwhere((bchm.real>0.5)&(bchm.pdct<0.5)).shape[0]

#full model
chi2_contingency(np.array([[tn,fn],[fp,tp]]))[0]
#F1-score
tp*2.0/(2*tp+fp+fn)



#==================================match pairs to sync strength=============================
#split softmax sync weights
pdct_pwr=[]
y_pdct=[]
cur_sum=0
for day_i in range(len(x_data_sy)):
	pdct_pwr.append(outcome_softmax[cur_sum:(cur_sum+x_data_sy[day_i].shape[0]),:])
	y_pdct.append(ful_outcom[cur_sum:(cur_sum+x_data_sy[day_i].shape[0])])
	cur_sum=cur_sum+x_data_sy[day_i].shape[0]

#edge list with sync weights
pdct_edg_lst=[]

for day_i in range(len(x_data_sy)):
	for nd_i in range(len(x_peers_sy[day_i])):
		tempt_pdctee=local2glb_sy[day_i][nd_i]
		tempt_pdcters=local2glb_sy[day_i][x_peers_sy[nd_i]]
		tempt_pdctpwr=pdct_pwr[day_i][nd_i,-len(tempt_pdcters):]
		org_pdctee=nd_attrs[nd_attrs.id==tempt_pdctee]
		pdct_edg_lst.append([tempt_pdctee,tempt_pdcters[pdcter_i],tempt_pdctpwr[pdcter_i],sm_org])
	print(day_i)

edg_lst_df=DataFrame(pdct_edg_lst).rename(columns={0:'pdctee',1:'pdctor',2:'pdct_pwr'})

#mean pdct power
np.mean(edg_lst_df.pdct_pwr)


#====================================predictability v.s. predicting-ability===================================
#predicting-ability
unique_pdctors=edg_lst_df.pdctor.drop_duplicates().tolist()
#1. number of reach-predictees
nb_pdctees=np.zeros(len(unique_pdctors),dtype=np.int)
#2. intensity of predictive power
mean_pdctpwr=np.zeros(len(unique_pdctors))
for pdctor_i in range(len(unique_pdctors)):
	tempt_pdctee_df=edg_lst_df[edg_lst_df.pdctor==unique_pdctors[pdctor_i]]
	if(tempt_pdctee_df.shape[0]>0):
		nb_pdctees[pdctor_i]=tempt_pdctee_df.shape[0]
		mean_pdctpwr[pdctor_i]=np.mean(tempt_pdctee_df.pdct_pwr)


#predictability
pdctability_df=DataFrame({'real':y_data_full_sy.reshape(-1),'pdct':ful_outcom})
pdctability_gid=np.zeros(ful_outcom.shape[0],dtype=np.int)
cur_pos=0
for day_i in range(len(x_data_sy)):
	for nd_i in range(len(x_kept_sy[day_i])):
		pdctability_gid[cur_pos]=local2glb_sy[day_i][x_kept_sy[day_i][nd_i]]
		cur_pos=cur_pos+1
pdctability_df['gid']=pdctability_gid


# mean error when being predicted
mean_err=np.zeros(len(unique_pdctors))

# accuracy of being predicted
acc_pdct=np.zeros(len(unique_pdctors))

# number of times being predicted
nb_pdcts=np.zeros(len(unique_pdctors),dtype=np.int)

pdctor_del_indcs=[]

for pdctor_i in range(len(unique_pdctors)):
	tempt_pdctability_df=pdctability_df[pdctability_df.gid==unique_pdctors[pdctor_i]]
	if(tempt_pdctability_df.shape[0]>0):
		mean_err[pdctor_i]=np.mean(np.abs(tempt_pdctability_df.real-tempt_pdctability_df.pdct))
		trup=tempt_pdctability_df[(tempt_pdctability_df.real>=0.5)&(tempt_pdctability_df.pdct>=0.5)].shape[0]
		trun=tempt_pdctability_df[(tempt_pdctability_df.real<0.5)&(tempt_pdctability_df.pdct<0.5)].shape[0]
		acc_pdct[pdctor_i]=(trup*1.0+trun)/tempt_pdctability_df.shape[0]
		if(trun<tempt_pdctability_df.shape[0]):
			F1_pdct[pdctor_i]=(2.0*trup)/(trup+tempt_pdctability_df.shape[0]-trun)
		else:
			F1_pdct[pdctor_i]=1
		nb_pdcts[pdctor_i]=tempt_pdctability_df.shape[0]
	else:
		pdctor_del_indcs.append(pdctor_i)


plt.scatter(nb_pdctees,nb_pdcts)
plt.show(block=False)

plt.scatter(mean_pdctpwr,mean_err)

#predicting vs predicted
pVped_df=DataFrame({'nb_out':nb_pdctees,'mean_pwr':mean_pdctpwr,'mean_err':mean_err,'acc_pdcted':acc_pdct,'F1_pdcted':F1_pdct,'nb_in':nb_pdcts})
pVped_df.to_csv(r'.../nw_result_sync.csv')











