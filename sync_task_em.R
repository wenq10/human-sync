library(corrplot)
library(ggplot2)

#load sync calculation result and organization events
sync_df=read.table('.../task/em_task.csv',sep=';',header=TRUE)

prevs3=rep(0,dim(sync_df)[1])
prevs2=rep(0,dim(sync_df)[1])
prevs1=rep(0,dim(sync_df)[1])
focal=rep(0,dim(sync_df)[1])
flw1=rep(0,dim(sync_df)[1])
flw2=rep(0,dim(sync_df)[1])
flw3=rep(0,dim(sync_df)[1])

for (i in 4:(dim(sync_df)[1]-3)){
	if(sync_df$havevent[i-3]==1){
		prevs3[i]=1
		}
	if(sync_df$havevent[i-2]==1){
		prevs2[i]=1
		}
	if(sync_df$havevent[i-1]==1){
		prevs1[i]=1
		}
	if(sync_df$havevent[i]==1){
		focal[i]=1
		}
	if(sync_df$havevent[i+1]==1){
		flw1[i]=1
		}
	if(sync_df$havevent[i+2]==1){
		flw2[i]=1
		}
	if(sync_df$havevent[i+3]==1){
		flw3[i]=1
		}
	}

analyz_df=cbind(sync_df$F1,prevs3,prevs2,prevs1,focal,flw1,flw2,flw3)
analyz_df=analyz_df[4:(dim(sync_df)[1]-3),]
analyz_df=data.frame(analyz_df)
names(analyz_df)[1]="F1"

corM=cor(analyz_df)
corrplot(corM,method="square")

# linear additive model
reg_rslt=lm(F1 ~ ., data=data.frame(analyz_df))
summary(reg_rslt)


























