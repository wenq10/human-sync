#run after model estimation
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame



#calculate peer sync strength
ful_sync=np.zeros([nb_itg,5])
for i in range(nb_epoch):
	ful_sync[(i*50):(i*50+50),:]=sess.run(peer_weights,feed_dict={X_input:x_data_full[(i*50):(i*50+50),:,:],y_input:y_data_full[(i*50):(i*50+50),:]})
sup_sync=sess.run(peer_weights,feed_dict={X_input:x_data_full[-50:,:,:],y_input:y_data_full[-50:,:]})[-16:,:]
ful_sync=np.vstack([ful_sync,sup_sync])


#split for each day
ful_sync_split=[]
cur_sum=0
for day_i in range(len(x_data)):
	ful_sync_split.append(ful_sync[cur_sum:(cur_sum+len(x_data[day_i])),:])
	cur_sum=cur_sum+len(x_data[day_i])


#edge list with sync strength
sync_edglst=[]
for day_i in range(len(x_data)):
	tempt_sync=ful_sync_split[day_i]
	tempt_peer_indcs=x_peer_indcs[day_i]
	local2global=nw_cardid[day_i]
	for ego_i in range(tempt_sync.shape[0]):
		ego_glb_id=[local2global[ego_i] for _ in range(len(tempt_peer_indcs[ego_i]))]
		altr_glb_id=local2global[tempt_local_indcs[ego_i]]
		altrONego=tempt_sync[ego_i,-len(tempt_local_indcs[ego_i]):]
		sync_edglst.append(DataFrame({'ego':ego_glb_id,'altr':altr_glb_id,'prd':altrONego}))
	if(day_i%10==0): print(day_i)


sync_edglst=pd.concat(sync_edglst)


staffs=sync_edglst.altr.drop_duplicates().tolist()

pdctability_df=[]
pdctability_gid=[]
cur_pos=0
non_sync_list=np.argwhere(F1_fit<0.6).reshape(-1)
for day_i in range(len(x_data)):
	if (day_i not in non_sync_list):
		pdctability_df.append(bchm_full[day_i])
		pdctability_gid.append(nw_cardid[day_i][keep_indcs[day_i]])
	if(day_i%10==0): print(day_i)

pdctability_df=pd.concat(pdctability_df,axis=0)

pdctability_gid=np.concatenate(pdctability_gid,axis=0)

pdctability_df['gid']=pdctability_gid


#1. number of reach-predictees
nb_pdctees=np.zeros(len(staffs),dtype=np.int)
#2. intensity of predictive power
mean_pdctpwr=np.zeros(len(staffs))

#3. mean error when being predicted
mean_err=np.zeros(len(staffs))

#individual
for stf_i in range(len(staffs)):
	tempt_in=pdctability_df[pdctability_df.gid==staffs[stf_i]]
	tempt_out=sync_edglst[sync_edglst.altr==staffs[stf_i]]
	if (tempt_in.shape[0]>0):
		#predicting ability
		nb_pdctees[stf_i]=tempt_out.shape[0]
		mean_pdctpwr[stf_i]=np.mean(tempt_out.prd)
		#predictability
		mean_err[stf_i]=np.mean(np.abs(tempt_in.pdct-tempt_in.real))
		nb_pdcts[stf_i]=tempt_in.shape[0]


nb_pdctees=np.delete(nb_pdctees,axis=0)

mean_pdctpwr=np.delete(mean_pdctpwr,axis=0)

mean_err=np.delete(mean_err,axis=0)

nb_pdcts=np.delete(nb_pdcts,axis=0)










