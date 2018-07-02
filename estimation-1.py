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

#read data from these file paths day-by-day
#x_data: top 5 collaborators' tie strength measures
#y_data: each individuals' overtime work durations
#the same approach was used to test the robustness of the findings with alternative cutoff time points

path_xdata=r".../x_data/"
path_ydata=r".../y_data/"

#load all the data from the path
files_xdata=os.listdir(path_xdata)
files_xdata.sort() #sort to ensure that the two sets of data correspond

files_ydata=os.listdir(path_ydata)
files_ydata.sort()


x_data=[]
y_data=[]


for day_i in range(len(files_xdata)):
	time_tic=time.time()
	#tie strength data for each day
	x_data_tempt=np.loadtxt(os.path.join(path_xdata,files_xdata[day_i]), delimiter=",")
	#behavior data for each day
	y_data_tempt=np.loadtxt(os.path.join(path_ydata,files_ydata[day_i]), delimiter=",")
	#convert to model input
	x_data_tempt=x_data_tempt.reshape([y_data_tempt.shape[0],-1,5])
	y_data_tempt = y_data_tempt.reshape(y_data_tempt.shape[0],1)
	x_data.append(x_data_tempt)
	y_data.append(y_data_tempt)
	if(day_i%10==0): print(day_i,time.time()-time_tic)

#combine to full dataset
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

#================================estimation========================
sess=tf.Session()
sess.run(init)
#record last model weights for comparison
last_wi=np.zeros(4)+1
nb_epochs=y_data_use.shape[0]/Bat_Siz
j=0
while True:
	sess.run(train_step,feed_dict={X_input:x_data_use[((j%nb_epochs)*50):((j%nb_epochs)*50+50),:,:],y_input:y_data_use[((j%nb_epochs)*50):((j%nb_epochs)*50+50),:]})
	j=j+1
	if (j%nb_epochs==0):
		tempt_wi=sess.run(W_inner).reshape(-1)
		print(j,sess.run(W_inner),sess.run(loss,feed_dict={X_input:x_data_use[((j%nb_epochs)*50):((j%nb_epochs)*50+50),:,:],y_input:y_data_use[((j%nb_epochs)*50):((j%nb_epochs)*50+50),:]}))
		delta=np.mean(np.abs(last_wi-tempt_wi))/np.mean(np.abs(last_wi))
		if(delta<0.01):
			#relative change less than 1 percent
			break

#generate prediction
nb_itg=y_data_use.shape[0]-(y_data_use.shape[0]%50)
ful_outcom=np.zeros(nb_itg)
for i in range(nb_epochs):
	ful_outcom[(i*50):(i*50+50)]=sess.run(y_hat,feed_dict={X_input:x_data_full[(i*50):(i*50+50),:,:],y_input:y_data_full[(i*50):(i*50+50),:]}).reshape(-1)
sup_outcome=sess.run(y_hat,feed_dict={X_input:x_data_full[-50:,:,:],y_input:y_data_full[-50:,:]})[-(y_data_use.shape[0]%50):]
ful_outcom=np.vstack([ful_outcom.reshape([-1,1]),sup_outcome]).reshape(-1)

#full correlation
np.corrcoef(y_data_full.reshape(-1),ful_outcom)

#F1-score in predicting heavy and light/non- overtime work
bchm=DataFrame({'real':y_data_full.reshape(-1),'pdct':ful_outcom})
tp=np.argwhere((bchm.real>=6)&(bchm.pdct>=6)).shape[0]
fp=np.argwhere((bchm.real<6)&(bchm.pdct>=6)).shape[0]
tn=np.argwhere((bchm.real<6)&(bchm.pdct<6)).shape[0]
fn=np.argwhere((bchm.real>=6)&(bchm.pdct<6)).shape[0]

#full chi-square
chi2_contingency(np.array([[tn,fn],[fp,tp]]))[0]
#full F1
tp*2.0/(2*tp+fp+fn)
#full accuracy
(tp+tn)*1.0/bchm.shape[0]
#full mean prediction error
np.mean(np.abs(y_data_full.reshape(-1)-ful_outcom.reshape(-1)))


#split prediction for each day
bchm_full=[]
cur_sum=0
for day_i in range(len(y_data)):
	tempt_bchm=DataFrame({'real':y_data[day_i].reshape(-1),'pdct':ful_outcom[cur_sum:(cur_sum+len(y_data[day_i]))]})
	cur_sum=cur_sum+len(y_data[day_i])
	bchm_full.append(tempt_bchm)

#calculate 5 predictability indicators for each day
cor_fit=np.zeros(len(y_data))
acc_fit=np.zeros(len(y_data))
F1_fit=np.zeros(len(y_data))
nb_worker=np.zeros(len(y_data),dtype=np.int)
avg_err_fit=np.zeros(len(y_data))
chisq_fit=np.zeros(len(y_data))

for day_i in range(len(y_data)):
	tempt_bchm=bchm_full[day_i]
	nb_worker[day_i]=tempt_bchm.shape[0]
	avg_err_fit[day_i]=np.mean(np.abs(tempt_bchm.real-tempt_bchm.pdct))
	cor_fit[day_i]=np.corrcoef(tempt_bchm.real,tempt_bchm.pdct)[0,1]
	tp=np.argwhere((tempt_bchm.real>=6)&(tempt_bchm.pdct>=6)).shape[0]
	tn=np.argwhere((tempt_bchm.real<6)&(tempt_bchm.pdct<6)).shape[0]
	fp=np.argwhere((tempt_bchm.real<6)&(tempt_bchm.pdct>=6)).shape[0]
	fn=np.argwhere((tempt_bchm.real>=6)&(tempt_bchm.pdct<6)).shape[0]
	acc_fit[day_i]=(tp+tn)*1.0/tempt_bchm.shape[0]
	F1_fit[day_i]=(2*tp*1.0)/(2*tp+fp+fn)
	chisq_fit[day_i]=chi2_contingency(np.array([[tn,fn],[fp,tp]]))[0]


