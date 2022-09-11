library(pec) ##验证模型
library(rms)  ##拟合生存分析模型
library(survival)  ##生存分析包
library(glmnet)  ##Lasso回归包
dt<-read.csv('data/final_selected_data_for_cox_MTXnonMTX_delete_light_stemcell.csv',na = c("", "NA"))
transData <- function(df,vartype){
  for (i in 1:length(vartype)) {
    if(vartype[i]==0){df[,i]<-as.numeric(df[,i])}
    else if(vartype[i]!=0){df[,i]<-as.factor(df[,i])}
  }
  return(df)
}

m<-c(1,rep(0,31),rep(1,68),0,rep(1,2),0,0,1)
df<- transData(dt,m)
df[,106]<-factor(df[,106], levels = c( "MTX"，"nonMTX"))
str(df$medicine) 
dt <- na.omit(df) 

factor_col<-c(33:100,102:103)
x.factors <- model.matrix(~ dt$Sex+dt$bloodRT.1.IsNormal+
                            dt$bloodRT.2.IsNormal+dt$bloodRT.3.IsNormal+
                            dt$bloodRT.4.IsNormal+
                            dt$bloodRT.5.IsNormal+dt$bloodRT.6.IsNormal+
                            dt$UrineRT.1+dt$UrineRT.1.IsNormal+
                            dt$UrineRT.2+
                            dt$UrineRT.3+dt$UrineRT.3.IsNormal+
                            dt$UrineRT.4+
                            dt$LiverRenal.1.IsNormal+
                            dt$LiverRenal.2.IsNormal+
                            dt$LiverRenal.3.IsNormal+
                            dt$LiverRenal.4.IsNormal+dt$LiverRenal.4.ClinicalSignificant+
                            dt$LiverRenal.5.IsNormal+dt$LiverRenal.5.ClinicalSignificant+
                            dt$LiverRenal.6.IsNormal+dt$LiverRenal.7.IsNormal+
                            dt$LiverRenal.8.IsNormal+
                            dt$LiverRenal.9.IsNormal+
                            dt$LiverRenal.10.IsNormal+
                            dt$BloodLipid.1.IsNormal+
                            dt$BloodLipid.2.IsNormal+
                            dt$BloodLipid.3.IsNormal+
                            dt$BloodLipid.4.IsNormal+
                            dt$X25.OH.D3.1.IsNormal+
                            dt$ESR.1.IsNormal+dt$ESR.1.IsNormal+dt$Rheumatism.2.IsNormal+
                            dt$the.stage.of.your.illness+
                            dt$onset.seasons+dt$medicine
                          ,dt)[,-1]

x.factors <- model.matrix(~ dt$bloodRT.6.IsNormal+
                            dt$UrineRT.4+dt$medicine
                          ,dt)[,-1]
#将矩阵的因子变量与其它定量边量合并成数据框，定义了自变量。
x=as.matrix(data.frame(x.factors,dt[,c(2:4,8,14:18,23,26,27,30,31)]))
#设置应变量，生存时间和生存状态（生存数据）
y <- data.matrix(Surv(dt$days,dt$type))
#调用glmnet包中的glmnet函数，注意family那里一定要制定是“cox”，如果是做logistic需要换成"binomial"。
fit <-glmnet(x,y,family = "cox",alpha = 1)
plot(fit,label=T)
plot(fit,xvar="lambda",label=T) ##见图一
#主要在做交叉验证,lasso
fitcv <- cv.glmnet(x,y,family="cox", alpha=1,nfolds=10)
plot(fitcv)  ## 见图2
print(fitcv) ## 1个标准差对应变量少，选此收缩系数
##      Lambda Measure     SE Nonzero
## min 0.02632   11.96 0.2057      21
## 1se 0.11661   12.15 0.1779       5
coef(fitcv, s="lambda.min") 

### cox2 诶筛选变量搭建的模型

library(boot)

c_index <- function(formula, data, indices) {
  tran.data <- data[indices,]
  vali.data <- data[-indices,]
  
  fit <- coxph(formula, data=tran.data)
  result<-survConcordance(Surv(vali.data$days,vali.data$type)~predict(fit,vali.data))
  index<-as.numeric(result$concordance)
  return(index)
}


set.seed(1234)
as.numeric(dt[,104])

results <- boot(data=dt, statistic=c_index, R=10000, formula=Surv(time = days,event = type)~medicine)


mean(results$t)
boot.ci(results, conf=0.95,type = "bca")
alpha <- 0.05
quantile(results$t, c(alpha/2, 1-alpha/2))
res.cox <- coxph(Surv(time = days,event = type) ~ Age, data = dt)
res.cox
summary(res.cox)



####all features univariate analysis
#假设我们要对如下5个特征做单因素cox回归分析
covariates <- c("Sex","bloodRT.1.IsNormal",   "bloodRT.2.IsNormal", 
                "bloodRT.3.IsNormal",
                "bloodRT.4.IsNormal","bloodRT.5.IsNormal",
                "bloodRT.6.IsNormal","UrineRT.1","UrineRT.1.IsNormal","UrineRT.2",
                "UrineRT.3","UrineRT.3.IsNormal","UrineRT.4","LiverRenal.1.IsNormal",
                "LiverRenal.2.IsNormal","LiverRenal.3.IsNormal","LiverRenal.4.IsNormal",
                "LiverRenal.4.ClinicalSignificant","LiverRenal.5.IsNormal",
                "LiverRenal.5.ClinicalSignificant","LiverRenal.6.IsNormal","LiverRenal.7.IsNormal",
                "LiverRenal.8.IsNormal","LiverRenal.9.IsNormal","LiverRenal.10.IsNormal",
                "BloodLipid.1.IsNormal","BloodLipid.2.IsNormal",
                "BloodLipid.3.IsNormal","BloodLipid.4.IsNormal",
                "X25.OH.D3.1.IsNormal","ESR.1.IsNormal",
                "Rheumatism.2.IsNormal",
                "the.stage.of.your.illness","onset.seasons","Age","PASI","BSA","Height","Weight",
                "W","H","WHR","BMI","SBP","DBP","PASI.1.A","PASI.2.A","PASI.3.A","PASI.4.A","PASI.4.E",
                "PASI.4.D","PASI.4.I","BSA.1.A","BSA.2.A","BSA.3.A","BSA.4.A","BSA.5.A","BSA.6.A","BSA.7.A","BSA.8.A",
                "BSA.9.A","BSA.10.A","BSA.11.A","BSA.12.A","BSA.13.A","DLQI","medicine")
#分别对每一个变量，构建生存分析的公式
univ_formulas <- sapply(covariates,
                        function(x) as.formula(paste('Surv(time = days,event = type)~', x)))
#循环对每一个特征做cox回归分析
univ_models <- lapply( univ_formulas, function(x){coxph(x, data = dt)})

#提取HR，95%置信区间和p值
univ_results <- lapply(univ_models,
                       function(x){ 
                         x <- summary(x)
                         #获取p值
                         p.value<-signif(x$wald["pvalue"], digits=2)
                         #获取HR
                         HR <-signif(x$coef[2], digits=2);
                         #获取95%置信区间
                         HR.confint.lower <- signif(x$conf.int[,"lower .95"], 2)
                         HR.confint.upper <- signif(x$conf.int[,"upper .95"],2)
                         HR <- paste0(HR, " (", 
                                      HR.confint.lower, "-", HR.confint.upper, ")")
                         res<-c(p.value,HR)
                         names(res)<-c("p.value","HR (95% CI for HR)")
                         return(res)
                       })
#转换成数据框，并转置
res <- t(as.data.frame(univ_results, check.names = FALSE))
as.data.frame(res)
write.table(file="final_fig/univariate_cox_result.txt",as.data.frame(res),quote=F,sep="\t")


###multi-variates
library(survminer)
library(survival)
res.cox <- coxph(Surv(time = days,event = type) ~ medicine,
                 data =  dt)
x <- summary(res.cox)
x
ggforest(res.cox,data = dt)
ggcoxdiagnostics(res.cox)
ftest<-cox.zph(res.cox)
ftest
ggcoxzph(ftest)
fit<-survfit(Surv(time = days,event = type)~medicine,data=dt)
ggsurvplot(fit,data=dt,conf.int = TRUE,fun='pct',pval = TRUE,risk.table = TRUE,linetype = 'strata',
           palette = c("#5CACD0","#DA6C6C"),legend="bottom",legend.title='Drug')
summary( fit)
surv_summary(fit)
ggcoxadjustedcurves(res.cox)
data.survdiff <- survdiff(Surv(time = days,event = type) ~ medicine,data=dt)
p.val = 1 - pchisq(data.survdiff$chisq, length(data.survdiff$n) - 1)
HR = (data.survdiff$obs[2]/data.survdiff$exp[2])/(data.survdiff$obs[1]/data.survdiff$exp[1])
up95 = exp(log(HR) + qnorm(0.975)*sqrt(1/data.survdiff$exp[2]+1/data.survdiff$exp[1]))
low95 = exp(log(HR) - qnorm(0.975)*sqrt(1/data.survdiff$exp[2]+1/data.survdiff$exp[1]))

results <- boot(data=dt, statistic=c_index, R=10000, 
                formula=Surv(time = days,event = type)~ medicine+
                  Age+H+UrineRT.4+bloodRT.6.IsNormal+
                PASI.2.A+PASI.4.D+BSA.4.A+BSA.7.A+BSA.12.A)
results <- boot(data=dt, statistic=c_index, R=10000, 
                formula=Surv(time = days,event = type)~ 
                  Age+H+UrineRT.4+bloodRT.6.IsNormal+
                  PASI.2.A+PASI.4.D+BSA.4.A+BSA.7.A+BSA.12.A)

mean(results$t)
boot.ci(results, conf=0.95,type = "bca")
alpha <- 0.05
quantile(results$t, c(alpha/2, 1-alpha/2))

pvalue=signif(as.matrix(x$coefficients)[,5],2)
HR=signif(as.matrix(x$coefficients)[,2],2)
low=signif(x$conf.int[,3],2)
high=signif(x$conf.int[,4],2)
multi_res=data.frame(p.value=pvalue,
                     HR=paste(HR," (",low,"-",high,")",sep=""),
                     stringsAsFactors = F
)
multi_res
write.table(file="final_fig/multivariate_cox_result.csv",multi_res,quote=F,sep="\t")




