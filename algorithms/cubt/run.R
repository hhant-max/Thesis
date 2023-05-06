library(ISLR)
library(rpart)
library(rpart.plot)
# install.packages('rpart.plot')

#build the initial decision tree
tree <- rpart(Salary ~ Years + HmRun, data=Hitters, control=rpart.control(cp=.0001))

#identify best cp value to use
best <- tree$cptable[which.min(tree$cptable[,"xerror"]),"CP"]

#produce a pruned tree based on the best cp value
pruned_tree <- prune(tree, cp=best)

#plot the pruned tree
plot(pruned_tree)

