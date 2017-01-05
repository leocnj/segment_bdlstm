#
# Process dfs to generate csv files needed for NN experiments.
#
library(dplyr)
library(magrittr)

df <- read.csv('csv/all.tsv', sep = '\t')
score <- read.csv('csv/hs_out.csv')
# score$subj <- factor(score$subj)

# obtain subj's BARS score (sum)
out <- score %>%
  group_by(subj) %>%
  summarise(sum = sum(BARS, na.rm=T))

out<-data.frame(out)
out.sorted <- out[order(out$sum),]

# check out.sorted distribution to identify dev set (n=6) to
# cover score range
#
# toal 36 subj.
#
# <40     3    1  subj=24
# 40-50   4    1  subj=18
# 50-60  12    2  subj=10,17
# 60-70  17    2  subj=13,31

df_sc <- merge(df, score[, c('videoID', 'subj', 'BARS')], by.x='vid', by.y='videoID')
df_sc <- df_sc[complete.cases(df_sc),] # remove NA row.

# based on the above subj-split plan, divide data
subj <- unique(score$subj)
dev <- c('24', '18', '10', '17', '13', '31')
train <- setdiff(subj, dev)

df.dev<-subset(df_sc, subj %in% dev, select = vid:BARS)
df.train<-subset(df_sc, subj %in% train, select = vid:BARS)

# now we can save both new dfs for param tweaking
# later, from df.train will generate 10-fold CV from train (n=30)
write.csv(df.dev, file='csv/param_dev.csv', row.names = F)
write.csv(df.train, file='csv/param_train.csv', row.names = F)
