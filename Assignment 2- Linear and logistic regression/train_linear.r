#!/usr/bin/env Rscript
# your code must have this first line.

# Train code for linear regression part goes here
args<-commandArgs()
df	<- read.table(args[6], header = FALSE, sep = ",")
XY	<- data.matrix(df,rownames.force = NA)
X   <- XY[,-ncol(XY)]
Y   <- XY[,ncol(XY)]
m   <- nrow(X)		
n   <- ncol(X)
ones<- c(rep(1,n))
X   <- cbind(ones,X)
n<-n+1
X_T <- t(X)
theta <- (solve((X_T%*%X)))%*%X_T%*%Y
write.table(theta,file=args[7],row.names=FALSE,col.names=FALSE)