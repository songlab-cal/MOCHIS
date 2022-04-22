## Install from tar ball
install.packages("mochisR_0.0.0.tar.gz", repos=NULL, type = "source")

## Load mochisR
library(mochisR)
library(tidyverse)

## Look at function documentation
?mochis.test

## Tests
set.seed(2022)

# Mann-Whitney
# EXAMPLE 1
x1 <- abs(rnorm(10))
y1 <- abs(rnorm(40))

df.1 <- data.frame(POINTS = c(x1,y1),
                   LABELS = c(rep("x",10),rep("y",40)))

# Visualize the stochastic equality!
ggplot(df.1, aes(x = POINTS)) +
  geom_histogram(aes(y=..density.., fill = LABELS), position = "dodge2") +
  theme_bw()

# 1. Base R's implementation
mochis.test(x=x1,y=y1,p=1,wList=length(x1):0,approx="resample")
# 2. Our implementation
mochis.test(x=x1,y=y1,p=1,wList=length(x1):0,approx="resample",use_base_r=FALSE)

# EXAMPLE 2
x2 <- abs(rnorm(50))
y2 <- abs(rnorm(100, mean = 1.5))

df.2 <- data.frame(POINTS = c(x2,y2),
                   LABELS = c(rep("x",50),rep("y",100)))

# Visualize the mean shift!
ggplot(df.2, aes(x = POINTS)) +
  geom_histogram(aes(y=..density.., fill = LABELS), position = "dodge2") +
  theme_bw() 

# 1. Base R's implementation
mochis.test(x=x2,y=y2,p=1,wList=length(x2):0,approx="resample")
# 2. Our implementation
mochis.test(x=x2,y=y2,p=1,wList=length(x2):0,approx="resample",use_base_r=FALSE)

# Dispersion Change
# EXAMPLE 1
x3 <- abs(rnorm(35, sd=2.5))
y3 <- abs(rnorm(420))

df.3 <- data.frame(POINTS = c(x3,y3),
                   LABELS = c(rep("x",35),rep("y",420)))

ggplot(df.3, aes(x = POINTS)) +
  geom_histogram(aes(y=..density.., fill = LABELS), position = "dodge2") +
  theme_bw() 

# 1. Quadratic kernel
quad_w_vec_3 <- sapply(1:(length(x3)+1), function(x) {(x/(length(x3)+1)-0.5)^4})
mochis.test(x=x3,y=y3,p=1,wList=quad_w_vec_3,approx="resample", alternative="less")
# 2. Gaussian kernel
gauss_w_vec_3 <- sapply(1:(length(x3)+1), function(x) {qnorm(x/(length(x3)+1.5))^4})
mochis.test(x=x3,y=y3,p=1,wList=gauss_w_vec_3,approx="resample",alternative="less")

# EXAMPLE 2
x4 <- abs(rnorm(213, sd=3))
y4 <- abs(rnorm(381))

df.4 <- data.frame(POINTS = c(x4,y4),
                   LABELS = c(rep("x",213),rep("y",381)))

ggplot(df.4, aes(x = POINTS)) +
  geom_histogram(aes(y=..density.., fill = LABELS), position = "dodge2") +
  theme_bw() 

# 1. Quadratic kernel
quad_w_vec_4 <- sapply(1:(length(x4)+1), function(x) {(x/(length(x4)+1)-0.5)^2})
mochis.test(x=x4,y=y4,p=1,wList=quad_w_vec_4,approx="resample")
# 2. Gaussian kernel
gauss_w_vec_4 <- sapply(1:(length(x4)+1), function(x) {qnorm(x/(length(x4)+2))^2})
mochis.test(x=x4,y=y4,p=1,wList=gauss_w_vec_4,approx="resample")
