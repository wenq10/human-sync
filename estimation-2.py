import tensorflow as tf
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
import os
import random
from scipy.stats import chi2_contingency
from scipy.stats import mannwhitneyu
from scipy.stats import wilcoxon
import networkx as nx



#read data
#x_data: email network tie strength measures
#y_data: overtime work
path_xdata=r"/xdata/"
path_ydata=r"/ydata/nw/"

#load files from the paths
files_xdata=os.listdir(path_xdata)
files_xdata.sort()

files_ydata=os.listdir(path_ydata)
files_ydata.sort()


#load data
x_data=[]
y_data=[]

for day_i in range(len(files_xdata)):
	#overtime work behavior
	night_wk=pd.read_csv(os.path.join(path_ydata,files_nw[day_i]),delimiter=",")
	night_wk_arr=np.array(night_wk)[:,0]
	#tie strength
	x_data_tempt=np.loadtxt(os.path.join(path_xdata,files_xdata[day_i]), delimiter=",")
	#convert to model input
	y_data_tempt = night_wk_arr.reshape(night_wk_arr.shape[0],1)
	x_data_tempt = x_data_tempt.reshape([y_data_tempt.shape[0],-1,8])
	x_data.append(x_data_tempt)
	y_data.append(y_data_tempt)
	if(day_i%10==0): print(day_i)

#reindex before training
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
#alternatives attempted but they turn out to be less effective than L1 loss
#loss = tf.reduce_mean(tf.square(y_input - y_hat))
#loss = tf.losses.hinge_loss(labels=y_input,logits=y_hat)
#loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_input,logits=y_hat))

train_step=tf.train.RMSPropOptimizer(0.01).minimize(loss)

init = tf.global_variables_initializer()

#run
sess=tf.Session()
sess.run(init)
last_wi=np.zeros(7)+1
nb_epochs=y_data_use.shape[0]/Bat_Siz
for j in range(50000):
	sess.run(train_step,feed_dict={X_input:x_data_use[((j%nb_epochs)*60):((j%nb_epochs)*60+60),:,:],y_input:y_data_use[((j%nb_epochs)*60):((j%nb_epochs)*60+60),:]})
	if (j%nb_epochs==0):
		tempt_wi=sess.run(W_inner).reshape(-1)
		#relative change in weights less than 1%
		if(np.mean(np.abs(last_wi-tempt_wi))/np.mean(np.abs(last_wi))<0.01):
			print(j)
			break
		else:
			last_wi=tempt_wi
			print(j,sess.run(loss,feed_dict={X_input:x_data_use[((j%nb_epochs)*60):((j%nb_epochs)*60+60),:,:],y_input:y_data_use[((j%nb_epochs)*60):((j%nb_epochs)*60+60),:]}))




wi=sess.run(W_inner)

#obtain sync strength
outcome_softmax=sess.run(inner_softmax,feed_dict={X_input:x_data_full[0:60,:,:],y_input:y_data_full[0:60,:]})
for bt in range(1,nb_epochs):
	outcome_softmax=np.vstack([outcome_softmax,sess.run(inner_softmax,feed_dict={X_input:x_data_full[(60*bt):(60*bt+60),:,:],y_input:y_data_full[(60*bt):(60*bt+60),:]})])
outcome_softmax=np.vstack([outcome_softmax,sess.run(inner_softmax,feed_dict={X_input:x_data_full[-60:,:,:],y_input:y_data_full[-60:,:]})[-7,:]])

#obtain y predictions
ful_outcom=np.zeros(nb_epochs*Bat_Siz)
for i in range(nb_epochs):
	ful_outcom[(i*60):(i*60+60)]=sess.run(y_hat,feed_dict={X_input:x_data_full[(i*60):(i*60+60),:,:],y_input:y_data_full[(i*60):(i*60+60),:]}).reshape(-1)
sup_outcom=sess.run(y_hat,feed_dict={X_input:x_data_full[-60:,:,:],y_input:y_data_full[-60:,:]}).reshape(-1)[-7:]
ful_outcom=np.concatenate([ful_outcom,sup_outcom],axis=0)

#initial visualization
plt.scatter(y_data_full,ful_outcom)
plt.show()

#correlation between prediction and real values
np.corrcoef(y_data_full.reshape(-1),ful_outcom)
bchm=DataFrame({'real':y_data_full.reshape(-1),'pdct':ful_outcom})

#calculating F1-score for binary prediction
tp=np.argwhere((bchm.real>0.5)&(bchm.pdct>=0.5)).shape[0]
tn=np.argwhere((bchm.real<0.5)&(bchm.pdct<0.5)).shape[0]
fp=np.argwhere((bchm.real<0.5)&(bchm.pdct>=0.5)).shape[0]
fn=np.argwhere((bchm.real>0.5)&(bchm.pdct<0.5)).shape[0]

#chi-square test
chi2_contingency(np.array([[tn,fn],[fp,tp]]))
#F1-score
tp*2.0/(2*tp+fp+fn)



cur_sum=0
y_pdct=[]
for day_j in range(len(y_data)):
	y_pdct.append(ful_outcom[cur_sum:(cur_sum+len(y_data[day_j]))])
	cur_sum=cur_sum+len(y_data[day_j])

#calculate predictability measures
F1=np.zeros(210)
Acc=np.zeros(210)
chisq=np.zeros(210)
pval=np.zeros(210)
nb_nw=np.zeros(210,dtype=np.int)
mean_err=np.zeros(210)
for day_i in range(210):
	bchm_tempt=DataFrame({'real':y_data[day_i].reshape(-1),'pdct':y_pdct[day_i].reshape(-1)})
	mean_err[day_i]=np.mean(np.abs(bchm_tempt.real-bchm_tempt.pdct))
	tp=bchm_tempt[(bchm_tempt.real>=0.5)&(bchm_tempt.pdct>=0.5)].shape[0]
	tn=bchm_tempt[(bchm_tempt.real<0.5)&(bchm_tempt.pdct<0.5)].shape[0]
	fp=bchm_tempt[(bchm_tempt.real<0.5)&(bchm_tempt.pdct>=0.5)].shape[0]
	fn=bchm_tempt[(bchm_tempt.real>=0.5)&(bchm_tempt.pdct<0.5)].shape[0]
	try:
		F1[day_i]=(tp*2.0)/(tp*2+fp+fn)
	except:
		F1[day_i]=0
	Acc[day_i]=(tp+tn*1.0)/bchm_tempt.shape[0]
	try:
		chisq[day_i],pval[day_i],dof,ex=chi2_contingency(np.array([[tn,fn],[fp,tp]]))
	except:
		chisq[day_i]=-1
		pval[day_i]=1
	nb_nw[day_i]=tp+fn

#
rslt_eval=DataFrame({'F1':F1,'Acc':Acc,'chi2':chisq,'p-val':pval,'nb_nw':nb_nw,'mean_err':mean_err})






















