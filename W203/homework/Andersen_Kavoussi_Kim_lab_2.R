# Load the ANES survey data, noting the absolute path to which the working directory is set
setwd("C:/Users/Brad/work/git-repos/mids/W203/homework/lab-2")
ANES <- read.csv("./master/anes_pilot_2018.csv")
D <- A <- ANES

#########################################################
################### Question 1 ##########################
#########################################################

#Q1.b EDA

#sanity check1
na_p = nrow(subset(D,is.na(D$ftpolice)))
na_j = nrow(subset(D,is.na(D$ftjournal)))
paste('First we want to see how many values are missing for P and J, for P there is ',na_p,'null values and for J there is ',na_j,' null values')

#create dataframe
q1df <- data.frame(police = D$ftpolice, journalists = D$ftjournal)
head(q1df, n=5)

# change outlier (-7) to the boundary value, zero.
q1df$journalists[q1df$journalists<0] <- 0
nrow((q1df$journalists<0) == TRUE)
summary(q1df)

# new column for difference of respects
q1df$dif <- with(q1df, police - journalists)
summary(q1df$dif)
paste('total number of observation is ',length(q1df$dif))

# sanity check for boundary values
q1df[q1df$dif == 100,]
q1df[q1df$dif == -99,]

# create histogram
x<-q1df$dif
h<-hist(x, breaks=250, xlab='respect(police - journalists)', main='magnitude of respect towards police minus journalists')
xfit<-seq(min(x),max(x),length=40) 
yfit<-dnorm(xfit,mean=mean(x),sd=sd(x)) 
yfit <- yfit*diff(h$mids[1:2])*length(x) 
lines(xfit, yfit, col="blue", lwd=2)

# scatterplot
ggplot(q1df, aes(x=police, y=dif))+geom_boxplot(color="red")+geom_point(position=position_dodge(width=0.5))

#########################################################

#Q1.c Test

t.test(q1df$dif, alternative = "two.sided", mu=0.0)

#check cohen's d
Mu = 0.0
Sd <- sd(q1df$dif)
cohenD = (mean(q1df$dif) - Mu)/Sd
cohenD


#########################################################
################### Question 2 ##########################
#########################################################

#Q2.b EDA

# create dataframe
q2df <- data.frame(birthyear = D$birthyr, party = D$pid1d, party2 = D$pid1r, strongness = D$pidstr, closeness = D$pidlean)
head(q2df, n=5)

#sanity check
sapply(q2df, function(x) sum(is.na(x)))
paste('we found no missing value for these varaibles')
unique(q2df$party)
unique(q2df$party2)
unique(q2df$strongness)
unique(q2df$closeness)
q2df[q2df$party == -7,]
q2df[q2df$party2 == -7,]
abnormal = q2df[q2df$party == -1 & (q2df$party2 == 1 | q2df$party2 == 2),]
head(abnormal, n=5)
abn2 = q2df[q2df$party != q2df$party2,]
head(abn2, n=10)
paste('within voters who skipped either of the party identity questions, the number of those who skipped both and so cannot identify the party id is ',nrow(q2df[q2df$party == -1 & q2df$party2 == -1,]))

# remove "no answer" 
q2df <- q2df[which(q2df$party != -7 & q2df$party2 != -7),]
q2df.new <- q2df
q2df.new$party[q2df.new$party<1] <- 0
q2df.new$party2[q2df.new$party2<1] <- 0
tail(q2df.new, n=5)

# consolidate pid1d & pid1r into 'partyid'
q2df.new$partyid <- with(q2df.new, party + party2)
tail(q2df.new, n=5)
u <- c(unique(q2df.new$partyid))
paste('the consolidated variable "partyid" has unique values of', u, 'which falls in the range of orignal party(pid1d & pid1r) variable values')

# Sanity check for 'partyid'
check1= nrow(q2df.new[(q2df.new$party ==1 & q2df.new$partyid ==2),])
check2= nrow(q2df.new[(q2df.new$party2 ==1 & q2df.new$partyid ==2),])
check3= nrow(q2df.new[(q2df.new$party ==2 & q2df.new$partyid ==3),])
check4= nrow(q2df.new[(q2df.new$party2 ==2 & q2df.new$partyid ==3),])
paste('sanity check for new column "partyid" shows no abnormal value for 1(Democrat) and 2(Republican), abnormality counts=',sum(check1,check2,check3,check4))


# subset Democrat and Republican by partyid, 1 & 2 respectively
q2_D <- q2df.new[(q2df.new$partyid == 1),]
q2_R <- q2df.new[(q2df.new$partyid == 2),]

# create ggplot
q2_D$party <- 'Democrat'
q2_R$party <- 'Republican'
combined <- rbind(q2_D, q2_R)
library(ggplot2)
ggplot(combined,aes(x = birthyear)) + geom_histogram(bins=50, fill = "white", color = "black") + facet_grid(party ~ .)
ggplot(combined,aes(x = birthyear, fill=party)) + geom_histogram(alpha= 0.5, bins=50, position = "identity")+ theme_classic()

bp <- ggplot(combined, aes(x = party, y = birthyear))
bp+ geom_boxplot(fill="lightblue") +geom_violin(alpha=0.5, aes(color=party))

######################################################

#Q2.c Test

# Shapiro test (normality check)
shapiro.test(q2_D$birthyear)
shapiro.test(q2_R$birthyear)
var(q2_D$birthyear)
var(q2_R$birthyear)

# Welch Two Sample t-test
t.test(q2_D$birthyear, q2_R$birthyear, alternative = "two.sided", var.equal = FALSE)

# Mann-Whitney-wilcox U test
wilcox.test(q2_D$birthyear, q2_R$birthyear)

# Cohen's d
x<-q2_D$birthyear
y<-q2_R$birthyear

cohens_d <- function(x, y) {
    lx <- length(x)- 1
    ly <- length(y)- 1
    md  <- abs(mean(x) - mean(y))   
    csd <- lx * var(x) + ly * var(y)
    csd <- csd/(lx + ly)
    csd <- sqrt(csd)   
    cd <- md/csd
    }

print(cohens_d(x,y))

# strongness subset
x2 <- q2_D[(q2_D$strongness == 1),] 
y2 <- q2_R[(q2_R$strongness == 1),]
head(x2,n=5)
nrow(x2)
nrow(y2)

# Strong identity group test & cohen's d
wilcox.test(x2$birthyear, y2$birthyear)
t.test(x2$birthyear, y2$birthyear, alternative = "two.sided", var.equal = FALSE)
print(cohens_d(x2$birthyear, y2$birthyear))

#########################################################
################### Question 3 ##########################
#########################################################

#3.b perform an EDA

#create independent voters sample
new1 = A[A$pid1d == 3, ]
new2 = A[A$pid1r == 3, ]
ind = rbind(new1, new2)

#create strong independent voter sample
new3 = new1[new1$pidlean == 3,]
new4 = new2[new2$pidlean == 3,]
str_ind = rbind(new3, new4)

#check to make sure it is correct, no values are -7, -4, -1
table(A$pid1d)
table(A$pid1r)
table(ind$pid1d)
table(ind$pid1r)
table(ind$pidlean)
table(str_ind$pidlean)

####################################################

#3.d run the t-test
t.test(ind$russia16, mu=1.5, alternative="two.sided",conf.level = 0.95)
t.test(str_ind$russia16, mu=1.5, alternative="two.sided",conf.level = 0.95)

#calculate cohen's d for independents
a<-mean(ind$russia16)
b<-sd(ind$russia16)
c<-1.5
d<-(c-a)/b
d

#calculate cohen's d for strong independents
e<-mean(str_ind$russia16)
f<-sd(str_ind$russia16)
g<-1.5
h<-(g-e)/f
h

#########################################################
################### Question 4 ##########################
#########################################################

#4.b perform an EDA 
library(ggplot2)

#filter out values with -7 or -1 for all three battery questions, create total voter sample
total1 = A[A$geangry != -7 & A$geafraid != -7, ]
total2 = A[A$dtangry != -7 & A$dtafraid != -7 & A$dtangry != -1 & A$dtafraid != -1, ]
total3 = A[A$imangry != -7 & A$imafraid != -7 & A$imangry != -1 & A$imafraid != -1, ]

#create 2018 voter samples
v18_1 = total1[total1$turnout18 == 1 | total1$turnout18 == 2 | total1$turnout18 == 3 | total1$turnout18ns == 1,]
v18_16 = total1[total1$turnout16 == 1 | total1$turnout16 == 2 | total1$turnout16b == 1,]
v18_2 = v18_1[v18_1$dtangry != -7 & v18_1$dtafraid != -7 & v18_1$dtangry != -1 & v18_1$dtafraid != -1, ]
v18_3 = v18_1[v18_1$imangry != -7 & v18_1$imafraid != -7 & v18_1$imangry != -1 & v18_1$imafraid != -1, ]

#create vote in 2018, not in 2016 samples
new5 = v18_1[v18_1$turnout16 == 2 | v18_1$turnout16b == 2, ]
v18n16 = new5[new5$birthyr > 1998, ]
v18n16_2 = v18n16[v18n16$dtangry != -1 & v18n16$dtafraid != -1, ]

#tests to make sure everything looks correct
#table(v18_1$turnout18)
#table(v18_16$turnout16)
#table(total1$geangry)
#table(total1$geafraid)
#table(total2$dtangry)
#table(total2$dtafraid)
#table(total3$imangry)
#table(total3$imafraid)
#table(v18_1$geangry)
#table(v18_1$geafraid)
#table(v18_2$dtangry)
#table(v18_2$dtafraid)
#table(v18_3$imangry)
#table(v18_3$imafraid)
#table(v18n16$geangry)
#table(v18n16$geafraid)
#table(v18n16$imangry)
#table(v18n16$imafraid)

#create histograms
ggplot(total1,aes(x = geangry)) + geom_histogram(bins=5, fill = "white", color = "black") + labs(title = "Histogram of Total Voters")
ggplot(total1,aes(x = geafraid)) + geom_histogram(bins=5, fill = "white", color = "black") + labs(title = "Histogram of Total Voters")

#4.d conduct tests

#wilcox tests for each sample, print means and number of observations
wilcox.test(total1$geangry, total1$geafraid, paired=TRUE, conf.level=0.95)
mean(total1$geangry)
mean(total1$geafraid)
NROW(total1$geangry)
NROW(total1$geafraid)

wilcox.test(total2$dtangry, total2$dtafraid, paired=TRUE, conf.level=0.95)
mean(total2$dtangry)
mean(total2$dtafraid)
NROW(total2$dtangry)
NROW(total2$dtafraid)

wilcox.test(total3$imangry, total3$imafraid, paired=TRUE, conf.level=0.95)
mean(total3$imangry)
mean(total3$imafraid)
NROW(total3$imangry)
NROW(total3$imafraid)

wilcox.test(v18_1$geangry, v18_1$geafraid, paired=TRUE, conf.level=0.95)
mean(v18_1$geangry)
mean(v18_1$geafraid)
NROW(v18_1$geangry)
NROW(v18_1$geafraid)

wilcox.test(v18_2$dtangry, v18_2$dtafraid, paired=TRUE, conf.level=0.95)
mean(v18_2$dtangry)
mean(v18_2$dtafraid)
NROW(v18_2$dtangry)
NROW(v18_2$dtafraid)

wilcox.test(v18_3$imangry, v18_3$imafraid, paired=TRUE, conf.level=0.95)
mean(v18_3$imangry)
mean(v18_3$imafraid)
NROW(v18_3$imangry)
NROW(v18_3$imafraid)

wilcox.test(v18n16$geangry, v18n16$geafraid, paired=TRUE, conf.level=0.95, exact=F)
mean(v18n16$geangry)
mean(v18n16$geafraid)
NROW(v18n16$geangry)
NROW(v18n16$geafraid)

#calculate effect size (r)
#total general
aa = 2.2e-16
ab <- qnorm(aa)
print(ab/(sqrt(2494)))

#total dt
ac <- 1.291e-13
ad <- qnorm(ac)
print(ad/(sqrt(1251)))

#total im
ae <- 0.001032
af <- qnorm(ae)
print(af/(sqrt(1237)))

#v18 general
ag <- 2.2e-16
ah <- qnorm(ag)
print(ah/(sqrt(1856)))

#v18 dt
ai <- 2.575e-10
aj <- qnorm(ai)
print(aj/(sqrt(942)))

#v18 im
ak <- 0.002854
al <- qnorm(ak)
print(al/(sqrt(907)))

#v18n16 general
am <- 0.4918
an <- qnorm(am)
print(an/(sqrt(16)))

#########################################################
################### Question 5 ##########################
#########################################################


#Create a data frame with only columns required for the analysis, providing greater ease with which to analyze the data as well as facilitating visibility into data using functionality associated with data.frame objects
q5_df <- data.frame(race = A$race, birthyr = A$birthyr, turnout16 = A$turnout16, turnout18 = A$turnout18, reg = A$reg, whenreg = A$whenreg)

#Because we're using race as a Boolean value -- white or not white -- introduce a column in the data frame accordingly
q5_df$white = q5_df$race == 1

#Remove rows that have data associated with no response from the subject
q5_df <- q5_df[q5_df$race != -7 & q5_df$reg != -7 & q5_df$whenreg != -1,]

#Remove rows of those who answered "not completely sure" if they voted in the 2016 election
q5_df <- q5_df[q5_df$turnout16 != 3,]

#Remove rows of those who answered "not completely sure" if they voted in the 2018 election
q5_df <- q5_df[q5_df$turnout18 != 5,]

#Because we're concerned that we may be excluding a large number of subjects who could have voted in the 2016 election because their birthdays fell before or on the day of the election, identify the number of subjects born in 1998 that potentially could have registered to vote in 2016.  Remove these rows.
# nrow(q5_df[q5_df$birthyr == 1998 & (q5_df$whenreg == 2 | q5_df$whenreg == 3),])
# q5_df <- q5_df[q5_df$birthyr != 1998 | (q5_df$birthyr == 1998 & (q5_df$whenreg != 2 & q5_df$whenreg != 3)),]

#Remove rows of those not registered to vote
q5_df <- q5_df[q5_df$reg != 3,]

#Data frame q5_df should now represent the sample of ANES respondents we must use in our analysis.
paste("Total number of observations included in analysis:", length(q5_df$white))
summary(q5_df)

#Identify only those respondents that registered within the last two years and did not vote in the 2016 election.  These are the individuals in the ANES sample we refer to as "beginning to participate in the United States' electoral process and registering to vote."
registered_post2016_df <- q5_df[
  (q5_df$whenreg == 1 | q5_df$whenreg == 2) &
  q5_df$turnout16 == 2
,]
registered_post2016_white <- length(registered_post2016_df$white[registered_post2016_df$white == TRUE])
registered_post2016_nonwhite <- length(registered_post2016_df$white[registered_post2016_df$white == FALSE])

#Perform the proportion test to compare the two sub-samples
prop.test(
  x = c(registered_post2016_white, registered_post2016_nonwhite), 
  n = c(length(A$race[A$race == 1]), length(A$race[A$race > 1])), 
  alternative = "two.sided", 
  conf.level = 0.95
)

