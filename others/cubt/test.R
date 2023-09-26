install.packages('/home/sfy/Documents/VScodeProject/Thesis/algorithms/cubt_3.2.tar.gz',repos <- NULL,type <- "source")
library('clue')

library('partitions')
library('cubt')
# library('libpango1.0-dev')
installed.packages()

# start import data and create model


# model
# genrate data from the first modele
dd <- gendata(1)
Don <- as.matrix(dd[,-1])

# construct the maximal tree
aa <- cubt(Don)
# prune it 
toto <- prune.cubt(aa,Don)
# join leaves
ooo <- join.cubt(toto,Don)

# plot the tree

plot.cubt(aa,type = 'u')
text.cubt(aa)

capabilities()
help(cubt)
###############
print(aa, "var", "label")



aa['frame']

# check if decision tree has same result

# another way to plot
library(ISLR)
library(rpart)
library(rpart.plot)

#build the initial decision tree
tree <- rpart(Salary ~ Years + HmRun, data=Hitters, control=rpart.control(cp=.0001))

#identify best cp value to use
best <- tree$cptable[which.min(tree$cptable[,"xerror"]),"CP"]

#produce a pruned tree based on the best cp value
pruned_tree <- prune(tree, cp=best)

#plot the pruned tree
prp(pruned_tree)


install.packages("flametree")
flametree_plot(aa['frame'])

# another trainning process
# training.data = Don
# result.cubt<-cubt(as.matrix(training.data))
# vv<-prune.cubt(result.cubt,training.data)
# join.cubt(vv,training.data,nclass = 3)
# plot(result.cubt,type="u")
# text(result.cubt)
