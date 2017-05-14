#!/usr/bin/env Rscript
# your code must have this first line.

# Test code for logistic regression part goes here
args<- commandArgs()
x_df 	 <- read.table(args[6], header = FALSE, sep = ",")
theta_df <- read.table(args[7], header = FALSE, sep = ",")
X <- data.matrix(x_df,rownames.force = NA)
X <- as.matrix(X)
n <- ncol(X)
m <- nrow(X)
ones<- c(rep(1,n))
X   <- cbind(ones,X)
n<-n+1
thetas <- data.matrix(theta_df,rownames.force = NA)
num_classes <- nrow(thetas)

sigmoid <- function(z){
	g <- 1/(1+exp(-z))
	return(g)
}

classification<- function(){
	y <- c(0,m)
	for(i in 1:m){
		cat("m = ",i,"\n")
		maxVal <- 0
		maxArg <- 1
		for(j in 1:num_classes){
			h <- sigmoid(X[i,]%*%as.matrix(thetas[j,]))
			if(h>maxVal){
				maxVal <- h
				maxArg <- j
			}
		}
		y[i] <- maxArg
	}
	return(y)
}

y<-classification()
write.table(y,file=args[8],sep=",",row.names=FALSE,col.names=FALSE)

