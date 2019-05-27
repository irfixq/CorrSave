# Code from https://github.com/apache/incubator-mxnet/blob/master/example/gan/CGAN_mnist_R/CGAN_train.R
library(reticulate)
library(imager)
library(dplyr)
library(readr)
library(EBImage)
library(BiocGenerics)
library(scales)
library(glcm)
library(wvtool)

getFeatures <- function (images, im_x = 28, im_y = 28) {
  
  # resize image to im_x and im_y dimension
  images <- lapply(images, imager:::resize, size_x = im_x, size_y = im_y, interpolation_type = 2 )
  
  # initialize features vector
  features = c();
  
  for (i in 1:length(images)){
    
    # get colour features all channel from every images
    for (j in 1:3)
    {
      channel <- images[[i]][,,,j]
      
      m = dim(channel)[1]
      n = dim(channel)[2]
      # mean = (sum(channel))/(m*n)
      mean = mean(channel)
      # sd = ((sum((channel-mean)**2)/(m*n))**(1/2))
      sd = sd(channel)
      
      # skewness =((sum((channel-mean)**3)/(m*n))**(1/3))
      
      # https://alstatr.blogspot.com/2013/06/measures-of-skewness-and-kurtosis.html
      library(moments)
      
      f_channel = channel
      dim(f_channel) = c(m*n, 1) 
      sk = skewness(f_channel)
      features <- append(features, c (mean,sd,sk))
    }
    
    # convert to grayscale and take only the 1st two matrix
    temp = grayscale(images[[i]])[,,1,1]
    
    # package used to calculate the texture features
    # https://cran.r-project.org/web/packages/RTextureMetrics/RTextureMetrics.pdf
    library(RTextureMetrics)
    
    mask <- array(TRUE, dim=dim(temp))
    glcms <- EBImage:::.haralickMatrix(mask, temp, nc=255)
    mat <- EBImage:::.haralickFeatures(mask, temp, nc=255)
    
    # entropy = calcENT(glcms[,,1])
    # contrast = calcCON(glcms[,,1])
    homogeneity = calcHOM(glcms[,,1])
    energy = sum(glcms[,,1]^2)
    
    features <- append(features, c(mat[,"h.ent"],energy,mat[,"h.con"],homogeneity))
  }
  
  dim(features) = c(13, length(images))
  
  row.names(features) <- c("r_mean", "r_sd", "r_skewness",
                           "g_mean", "g_sd", "g_skewness",
                           "b_mean", "b_sd", "b_skewness",
                           "entrophy", "energy", "contrast", "homogeneity")
  
  colnames(features) < as.character(c(1:length(images)))
  
  return (features)
}



