install.packages('/home/sfy/Documents/VScodeProject/Thesis/algorithms/cubt_3.2.tar.gz',repos = NULL,type = "source")
library('clue')

library('partitions')
library('cubt')
installed.packages()

# start import data and create model


# model
# genrate data from the first modele
dd = gendata(1)
Don = as.matrix(dd[,-1])

# construct the maximal tree
aa = cubt(Don)
# prune it 
toto = prune.cubt(aa,Don)
# join leaves
ooo = join.cubt(toto,Don)
# plot the tree
plot(aa,type="u")
text(aa)

