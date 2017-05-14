#!/usr/bin/env Rscript
# your code must have this first line.

# Test code for linear regression part goes here
args<-commandArgs()
x_df 	 <- read.table(args[6], header = FALSE, sep = ",")
theta_df <- read.table(args[7], header = FALSE)
x 		 <- data.matrix(x_df,rownames.force = NA)
n <- ncol(x)
m <- nrow(x)
ones<- c(rep(1,n))
x   <- cbind(ones,x)
n<-n+1
theta 	<- data.matrix(theta_df,rownames.force = NA)
h_x 	<- x%*%theta
#output()
write.table(h_x,file=args[8],row.names=FALSE,col.names=FALSE)