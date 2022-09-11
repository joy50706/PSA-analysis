library(ggplot2)
library(ggthemes)
library(ggsignif)
library(tidyverse)
library(dplyr)
library(ggpubr)
library(devEMF)

library(dplyr)
library(forcats)

library("ggsignif")
library(gridExtra)
library(tibble)#rownames_to_column
library(ggplot2)
library(grid)
library(reshape2)
library(ggpubr)
library(dplyr)
library(tidyr)
library(pheatmap)
library(egg)
library(patchwork)
library(pROC)

data <- read.csv(file='final_fig/new_data_risk_prediction_mrmr_adaboost_feature_importance_delete_unimportant.csv', header=T)

new_data <- data[order(data$PImean,decreasing = TRUE),]

new_data$features<-factor(new_data$features,levels = unique(new_data$features),ordered = T)
p <- ggplot()+ 
  geom_bar(data=new_data,mapping=aes(x=features,y=PImean,fill=features),
           stat="identity", # 数据格式
           width = 0.7)+
  scale_fill_manual(values = c("#FFC1C2", "#FAE2C2","#CCDCAE",'#A6A6A6',
                               '#907DFF','#F9C7C7', '#F27A7A','#F10505',
                               '#2301FF','#666666',"#FFC1C2", "#FAE2C2"))+
  geom_errorbar(data=new_data,mapping=aes(x = features,ymin = PImean-PIstd, ymax = PImean+PIstd), # 误差线添加
                width = 0.1, #误差线的宽度
                color = 'black', #颜色
                size=0.8)+
  scale_y_continuous(expand = c(0,0))+theme_classic()+theme(legend.position="none")+
  theme(axis.text.x = element_text(
    angle = 45,
    hjust = 1,
    vjust = 1
  ))+
  labs(title="",x="",y="")
p

