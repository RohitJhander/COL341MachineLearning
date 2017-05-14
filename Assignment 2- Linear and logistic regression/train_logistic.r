#!/usr/bin/env Rscript
# your code must have this first line.

# Train code for logistic regression part goes here
args<- commandArgs()
df	<- read.table(args[6], header = FALSE, sep = ",")
XY	<- data.matrix(df,rownames.force = NA)
X   <- XY[,-ncol(XY)]
Y   <- XY[,ncol(XY)]
m   <- nrow(X)		
n   <- ncol(X)
ones<- c(rep(1,n))
X   <- cbind(ones,X)
n<-n+1
class <- sort(unique(as.vector(Y)))
num_class <- length(class)

cat("Size of training data (m): ",m,"\n")
cat("Number of features (n): ",n,"\n")
cat("Number of classes :",num_class,"\n")
cat("Classes: ",class,"\n")

sigmoid <- function(z){
	g <- 1/(1+exp(-z))
	return(g)
}

gradientDescent<- function(alpha,y,iterations){
	theta <- c(rep(0,n))
	for(it in 1:iterations){
		#cat("Iteration: ",it,"\n")
		for(i in 1:m){
			h <- sigmoid(X[i,]%*%as.matrix(theta))
			theta <- theta + alpha*(y[i]-h[1])*X[i,]
		}
	}
	return(theta)
}

multiThetaLearn <- function(alpha,iterations){
	thetas<-matrix(0,nrow=num_class,ncol=n,byrow=TRUE)
	for(i in 1:num_class){
		c <- class[i]
		#cat("----------------------- theta ",c,"-----------------------","\n")
		y <- Y
		for( j in 1:m){
			if(Y[j]==c){
				y[j] = 1
			}else{
				y[j] = 0
			}
		}
		thetas[i,] <- gradientDescent(alpha,as.matrix(y),iterations)
	}
	return(thetas)
}

thetas<-multiThetaLearn(0.001,15)
write.table(thetas,file=args[7],sep=",",row.names=FALSE,col.names=FALSE)
