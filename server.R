#
# This is the server logic of a Shiny web application. You can run the 
# application by clicking 'Run App' above.
#
# Find out more about building applications with Shiny here:
# 
#    http://shiny.rstudio.com/
#
#install.packages("BiocManager")
#BiocManager::install("EBImage")
#library(tensorflow)
#install_tensorflow(method = c("auto", "virtualenv", "conda", "system"),
                   #conda = "auto", version = "default", envname = "r-tensorflow",
                   #extra_packages = c("keras", "tensorflow-hub"),
                   #restart_session = TRUE)
library(reticulate)
library(shiny)
library(keras)
#install_keras()
library(imager)
library(shinyjs)
library(ggplot2)
library(DT)
library(imager)
library(dplyr)
library(readr)
library(EBImage)
library(scales)
library(glcm)
library(wvtool)
library(pROC)
library(caret)

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
# source("texture features.R")

# Define server logic required to draw a histogram
shinyServer(function(input, output, session) {
  
  opt <- list( lengthChange = TRUE ,
               autoWidth = TRUE,
               scrollX=TRUE
  )
  
  use_session_with_seed(1,disable_parallel_cpu = FALSE)
  
  variable <- reactiveValues(
    node = 0,
    layer = 0,
    batch_size = 0,
    epoch = 0,
    dropout = 0,
    train_ratio = 0,
    model = keras_model_sequential(),
    img_list_c = c(),
    img_list_nc = c(),
    img_list = c(),
    feature_mat = c(),
    feature_mat_flag = FALSE,
    label = c(),
    
    fit = "",
    plot_flag = FALSE,
    
    k = "",
    cv_fit = list(),
    cv_roc_obj = list(),
    cv_pred_class = list(),
    cv_pred_prob = list(),
    cv_roc_plot = list(),
    cv_flag = FALSE,
    
    confusion_matrix_flag = FALSE,
    performance_report = "",
    roc_obj = "",
    
    img_single = "",
    scale_attr = c(),
    
    image_url_flag = FALSE,
    image_url = "",
    
    img_list_t = ""
  )
  
  observeEvent(input$fileIn, {
    inFiles <- input$fileIn
    variable$img_list_c <- lapply(inFiles$datapath, load.image)
    variable$img_list_c <- lapply(variable$img_list_c, rm.alpha)
  })
  
  observeEvent(input$fileIn2, {
    inFiles <- input$fileIn2
    variable$img_list_nc <- lapply(inFiles$datapath, load.image)
    variable$img_list_nc <- lapply(variable$img_list_nc, rm.alpha)
  })
  
  observeEvent(input$save_param, {
    variable$node <- as.integer(input$node)
    variable$layer <- as.integer(input$layer)
    variable$batch_size <- as.integer(input$batch_size)
    variable$epoch <- as.integer(input$epoch)
    variable$dropout <- as.double(input$dropout)
    variable$train_ratio <- as.integer(input$train_ratio)
    
    variable$model = keras_model_sequential()
    
    variable$model %>%
      layer_dense(input_shape = 13, units = variable$node, activation = "relu") %>%
      layer_dropout(variable$dropout) 
    
    for (i in 1:(variable$layer)){
      variable$model %>%
        layer_dense(units = variable$node, activation = "relu") %>%
        layer_dropout(variable$dropout)
    }
    
    variable$model %>%
      # output layer
      layer_dense(units = 1, activation = "sigmoid")
    
    # add a loss function and optimizer
    variable$model %>%
      compile(
        loss = "binary_crossentropy",
        optimizer = "Adam",
        metrics = "accuracy"
      )
    
    output$value <- renderPrint({ variable$model })
  })
  
  observeEvent(input$calculate_Features, {
    variable$img_list = c(variable$img_list_c, variable$img_list_nc)
    
    # 0 = corroded images
    # 1 = non-corroded images
    variable$label = c(rep(0, length(variable$img_list_c)), rep(1, length(variable$img_list_nc)))
    
    # implement own function to get image features
    variable$feature_mat <- ""
    
    variable$feature_mat <-getFeatures(variable$img_list)
    
    variable$feature_mat <- as.data.frame(variable$feature_mat)
    
    variable$feature_mat <- rbind(variable$feature_mat, as.integer(variable$label))
    
    variable$feature_mat <- t.data.frame(variable$feature_mat)
    
    rownames(variable$feature_mat) <- c(1:dim(variable$feature_mat)[1])
    
    colnames(variable$feature_mat)[14] <- "label"
    
    variable$feature_mat_flag = TRUE
  })
  
  observeEvent(input$start_training, {
    
    set.seed(1)
    
    rate <- round(variable$train_ratio/100, digits = 2)
    n <- nrow(variable$feature_mat)
    shuffled_df <- variable$feature_mat[sample(n), ]
    train_indices <- 1:round(rate * n)
    train <- shuffled_df[train_indices, ]
    test_indices <- (round(rate * n) + 1):n
    test <- shuffled_df[test_indices, ]
    
    perceptron = variable$node
    dropout = variable$dropout
    batch_size = round((variable$batch_size/100) * dim(train)[1])
    epoch = variable$epoch
    
    test_x <- test[,1:13]
    test_y <- test[,"label"]
    # test_x = as.matrix(apply(test_x, 2, function(test_x) (test_x-min(test_x))/(max(test_x) - min(test_x))))
    
    y = train[,"label"]
    names(y) <- NULL
    
    x = train[,1:13]
    
    # scale to [0,1]
    # x = as.matrix(apply(x, 2, function(x) (x-min(x))/(max(x) - min(x))))
    maxs <- apply(x, 2, max)    
    mins <- apply(x, 2, min)
    variable$scale_attr = rbind(mins, maxs)
    write.csv(variable$scale_attr, file="www/Scale Attribute.csv", row.names = FALSE)
    
    variable$scale_attr <- read.csv("www/Scale Attribute.csv")
    
    x_train_scaled = scale(x, center = mins, scale = maxs - mins)
    
    test_x_scaled = scale(test_x, center = as.numeric(mins), scale = as.numeric(maxs - mins))
    
    
    progress <- Progress$new(session, min=1, max=10)
    on.exit(progress$close())
    
    progress$set(message = 'Training in progress',
                 detail = 'This may take a while...', value = 5)
    
    # fit model with our training data set, training will be done for 200 times data set
    fit = variable$model %>%
      fit(
        x = x_train_scaled,
        y = y,
        shuffle = T,
        batch_size = batch_size,
        validation_split = 0.3,
        epochs = epoch,
        view_metrics = TRUE,
        callbacks = callback_csv_logger("www/log.csv", separator = ",", append = FALSE) 
      )
    variable$fit <- fit
    variable$plot_flag <- TRUE
    
    # Save the model to be used later
    name = paste("www/model", Sys.time())
    
    name = gsub(":", ".", name)
    save_model_hdf5(variable$model, name, overwrite = TRUE, include_optimizer = TRUE)
    model2 = load_model_hdf5(name, custom_objects = NULL, compile = TRUE)
    variable$performance_report <- c()
    pred_class <- predict_classes(object = variable$model, x = as.matrix(test_x_scaled)) %>%
      as.vector()
    
    # Predicted class probability
    pred_prob <- predict_proba(object = variable$model, x = as.matrix(test_x_scaled)) %>%
      as.vector()
    
    # show confusion matrix
    # require(caret)
    # cm <- confusionMatrix(data = as.factor(as.vector(test_y)), as.factor(pred_class), mode = "everything")
    
    # plot ROC and calculate AUC
    par(mfrow = c(1, 1), mar = c(0.1, 0.1, 0.1, 0.1))
    variable$roc_obj = roc(as.numeric(test_y), as.numeric(predict_proba(variable$model, test_x_scaled)))
    # plot.roc(roc_obj)
    # auc(roc_obj)
    
    variable$performance_report <- rbind(test_y,pred_class, pred_prob)
    variable$confusion_matrix_flag <- TRUE
    
    progress$set(message = 'Completed!',
                 detail = 'Test the model with images.', value = 10)
    
  })
  
  output$ui.trainPerformance <- renderUI({
    if(variable$fit != ""){
      tagList(
        h2("Model Performance for Training and Validation Process"),
        plotlyOutput('acc_plot'),
        br(),
        plotlyOutput('loss_plot'),
        br(),
        h3("Validation on Test data"),
        verbatimTextOutput("perfMetric"),
        br(),
        h3("ROC Graph"),
        plotOutput("rocPlot")
      )
    }
  })
  
  output$rocPlot <- renderPlot({
    if (variable$confusion_matrix_flag == TRUE){
      plot.roc(variable$roc_obj, print.auc=TRUE)
    }
    else {
      return ()
    }
    
  })
  
  
  output$perfMetric = renderPrint({
    if (variable$confusion_matrix_flag == TRUE){
      row.names(variable$performance_report) <- c("Actual Class", "Predicted Class", "Predicted Probability")
      variable$performance_report
    }
  })
  
  output$acc_plot <- renderPlotly({
    if(variable$plot_flag == TRUE){
      
      ti = 1:length(variable$fit$metrics$val_loss)
      mva = lm(variable$fit$metrics$val_acc~ti+I(ti^2)+I(ti^3))
      ma = lm(variable$fit$metrics$acc~ti+I(ti^2)+I(ti^3))
      
      line.fmt = list(dash="solid", width = 1.5, color=NULL)
      
      plot_ly(data = as.data.frame(variable$fit$metrics), x = 1:length(variable$fit$metrics$val_loss)) %>%
        add_trace(y = variable$fit$metrics$acc, name = 'Accuracy', mode = 'lines+markers') %>%
        add_trace(y = variable$fit$metrics$val_acc, name = 'Validation Accuracy', mode = 'lines+markers') %>%
        add_lines(y =  predict(ma), line=line.fmt, name="Accuracy Trend Line") %>%
        add_lines(y =  predict(mva), line=line.fmt, name="Validation Accuracy Trend Line") %>%
        layout(title = "Accuracy for Train and Validation")
      
    }
    else{
      return ()
    }
  })
  
  output$loss_plot <- renderPlotly({
    if(variable$plot_flag == TRUE){
      
      ti = 1:length(variable$fit$metrics$val_loss)
      mvl = lm(variable$fit$metrics$val_loss~ti+I(ti^2)+I(ti^3))
      ml = lm(variable$fit$metrics$loss~ti+I(ti^2)+I(ti^3))
      
      line.fmt = list(dash="solid", width = 1.5, color=NULL)
      
      plot_ly(data = as.data.frame(variable$fit$metrics), x = 1:length(variable$fit$metrics$val_loss)) %>%
        add_trace(y = variable$fit$metrics$loss, name = 'Loss', mode = 'lines+markers')%>%
        add_trace(y = variable$fit$metrics$val_loss, name = 'Validation Loss', mode = 'lines+markers')%>%
        add_lines(y =  predict(ml), line=line.fmt, name="Loss Trend Line") %>%
        add_lines(y =  predict(mvl), line=line.fmt, name="Validation Loss Trend Line") %>%
        layout(title = "Losses for Train and Validation")
      
    }
    else{
      return ()
    }
  })
  
  observeEvent(input$loadModel, {
    inFile <- input$loadModel
    
    if (is.null(inFile))
      return(NULL)
    
    variable$model <- keras_model_sequential()
    variable$model <- load_model_hdf5(inFile$datapath, custom_objects = NULL, compile = TRUE)
    
    output$value <- renderPrint({ variable$model })
  })
  
  observeEvent(input$loadScaleAttr, {
    inFile <- input$loadScaleAttr
    
    if (is.null(inFile))
      return(NULL)
    
    variable$scale_attr <- read.csv(inFile$datapath)
    
  })
  
  observeEvent(input$loadImage, {
    inFile <- input$loadImage
    
    if (is.null(inFile))
      return(NULL)
    
    variable$img_single <- load.image(inFile$datapath)
    
    output$imgDetail <- renderPrint({ variable$img_single })
  })
  
  observeEvent(input$test_single_image, {
    
    img = rm.alpha(variable$img_single)
    im_x = 28
    im_y = 28
    # resize image to im_x and im_y dimension
    r_img = imager:::resize(img, size_x = im_x, size_y = im_y, interpolation_type = 2)
    
    # initialize features vector
    features = c();
    
    # get colour features all channel from every images
    for (j in 1:3)
    {
      channel <- r_img[,,,j]
      
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
    temp = grayscale(r_img)[,,1,1]
    
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
    
    
    dim(features) = c(13, 1)
    
    row.names(features) <- c("r_mean", "r_sd", "r_skewness",
                             "g_mean", "g_sd", "g_skewness",
                             "b_mean", "b_sd", "b_skewness",
                             "entrophy", "energy", "contrast", "homogeneity")
    
    feature_mat_v <- as.data.frame(features)
    
    feature_mat_v <- t.data.frame(feature_mat_v)
    
    
    ### ================= Evaluation ===================
    
    n <- nrow(feature_mat_v)
    test_x_v <- feature_mat_v
    maxs <- variable$scale_attr[2,]    
    mins <- variable$scale_attr[1,]
    # test_x_v = as.matrix((test_x_v-min(test_x_v))/(max(test_x_v) - min(test_x_v)))
    test_x_v_scaled = scale(test_x_v, center = as.numeric(mins), scale = as.numeric(maxs - mins))
    
    rownames(test_x_v_scaled) <- c(1:n)
    
    pred_class <- predict_classes(object = variable$model, x = as.matrix(test_x_v_scaled)) %>%
      as.vector()
    
    # Predicted class probability
    pred_prob <- predict_proba(object = variable$model, x = as.matrix(test_x_v_scaled)) %>%
      as.vector()
    
    output$predResult1 <- renderPrint({ pred_class })
    output$predResult2 <- renderPrint({ pred_prob })
  })
  
  observeEvent(input$imageIn, {
    inFiles <- input$imageIn
    image_name <- input$imageIn$name
    img_list <- lapply(inFiles$datapath, load.image)
    img_list <- lapply(img_list, rm.alpha)
    variable$img_list_t <- cbind(image_name, img_list)
  })
  
  observeEvent(input$test_multi_image, {
    
    # implement own function to get image features
    feature_mat <-getFeatures(variable$img_list_t[,2])
    
    feature_mat <- as.data.frame(feature_mat)
    
    feature_mat <- t.data.frame(feature_mat)
    
    rownames(feature_mat) <- c(1:dim(feature_mat)[1])
    
    test_x_t <- feature_mat
    maxs <- variable$scale_attr[2,]    
    mins <- variable$scale_attr[1,]
    # test_x_v = as.matrix((test_x_v-min(test_x_v))/(max(test_x_v) - min(test_x_v)))
    test_x_t_scaled = scale(test_x_t, center = as.numeric(mins), scale = as.numeric(maxs - mins))
    
    pred_class <- predict_classes(object = variable$model, x = as.matrix(test_x_t_scaled)) %>%
      as.vector()
    
    # Predicted class probability
    pred_prob <- predict_proba(object = variable$model, x = as.matrix(test_x_t_scaled)) %>%
      as.vector()
    
    res_output <- rbind(variable$img_list_t[,1], pred_class, pred_prob)
    
    res_output <- as.data.frame(t(res_output))
    
    output$predResult1 <- renderPrint({ res_output })
    output$predResult2 <- renderPrint({  })
    
  })
  
  
  observeEvent(input$load_imageFromUrl, {
    if (input$imageURL != ""){
      download.file(input$imageURL,'www/temp.jpg', mode = 'wb')
      variable$image_url_flag <- TRUE
    }
    
    else {
      variable$image_url_flag <- FALSE
    }
  })
  
  output$ui.imageFromUrl <- renderUI({
    
    # check whether the dataset is loaded in the application
    if(variable$image_url_flag == TRUE) {
      
      # return radio button UI
      htmlOutput('image')
    }
    
    else {
      return ()
    }
  })
  
  output$image = renderUI({
    tags$div(
      tags$br(),
      tags$img(src = input$imageURL, width = "100%")
    )
    
  })
  
  observeEvent(input$test_single_image_from_url, {
    
    variable$img_single <- load.image("www/temp.jpg")
    
    img = rm.alpha(variable$img_single)
    im_x = 28
    im_y = 28
    # resize image to im_x and im_y dimension
    r_img = imager:::resize(img, size_x = im_x, size_y = im_y, interpolation_type = 2)
    
    # initialize features vector
    features = c();
    
    # get colour features all channel from every images
    for (j in 1:3)
    {
      channel <- r_img[,,,j]
      
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
    temp = grayscale(r_img)[,,1,1]
    
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
    
    
    dim(features) = c(13, 1)
    
    row.names(features) <- c("r_mean", "r_sd", "r_skewness",
                             "g_mean", "g_sd", "g_skewness",
                             "b_mean", "b_sd", "b_skewness",
                             "entrophy", "energy", "contrast", "homogeneity")
    
    feature_mat_v <- as.data.frame(features)
    
    feature_mat_v <- t.data.frame(feature_mat_v)
    
    
    ### ================= Evaluation ===================
    
    n <- nrow(feature_mat_v)
    test_x_v <- feature_mat_v
    maxs <- variable$scale_attr[2,]    
    mins <- variable$scale_attr[1,]
    # test_x_v = as.matrix((test_x_v-min(test_x_v))/(max(test_x_v) - min(test_x_v)))
    test_x_v_scaled = scale(test_x_v, center = as.numeric(mins), scale = as.numeric(maxs - mins))
    
    rownames(test_x_v_scaled) <- c(1:n)
    
    pred_class <- predict_classes(object = variable$model, x = as.matrix(test_x_v_scaled)) %>%
      as.vector()
    
    # Predicted class probability
    pred_prob <- predict_proba(object = variable$model, x = as.matrix(test_x_v_scaled)) %>%
      as.vector()
    
    output$predResult1 <- renderPrint({ pred_class })
    output$predResult2 <- renderPrint({ pred_prob })
  })
  
  output$summTable <- DT::renderDataTable({ 
    if (variable$feature_mat_flag == TRUE){
      name <- c(input$fileIn$name, input$fileIn2$name)
      path <- c(input$fileIn$datapath, input$fileIn2$datapath)
      
      img_path <- c()
      
      for (i in 1:length(variable$img_list)){
        save.image(variable$img_list[i][[1]], file = paste("www/temp/img",i,".png", sep = ""))
        img_path <- append(img_path,paste("<img src='",paste("temp/img",i,".png'", sep = "")," height=52></img>", sep=''))
      }
      dt <- cbind(name, img_path, variable$feature_mat)
      
      datatable(dt, escape = FALSE, option = opt)
    }
  })
  
  observeEvent(input$start_CV, {
    
    variable$k <- input$k
    
    feature_mat2 <- as.data.frame(variable$feature_mat)
    
    dropout = variable$dropout
    
    folds <- createFolds(y = variable$feature_mat[,"label"], k = variable$k, list = F) 
    feature_mat2$folds <- folds
    
    idx <- 1
    variable$cv_flag <- FALSE
    
    for(f in unique(feature_mat2$folds)){
      cat("\n Fold: ", f)
      ind <- which(feature_mat2$folds == f) 
      
      train_df <- feature_mat2[-ind,1:13]
      y_train <- as.matrix(feature_mat2[-ind, "label"])
      
      valid_df <- as.matrix(feature_mat2[ind,1:13])
      y_valid <- as.matrix(feature_mat2[ind, "label"])
      
      
      maxs <- apply(train_df, 2, max)    
      mins <- apply(train_df, 2, min)
      scale_attr = rbind(mins, maxs)
      train_df = scale(train_df, center = mins, scale = maxs - mins)
      
      valid_df = scale(valid_df, center = mins, scale = maxs - mins)
      
      # create sequential model
      model = keras_model_sequential()
      
      # add layers, first layer needs input dimension
      model %>%
        # 1- 10th layer
        layer_dense(input_shape = ncol(train_df), units = variable$node, activation = "relu") %>%
        layer_dropout(dropout) 
      
      for (j in 1:variable$layer){
        model %>%
          # 1- 10th layer
          layer_dense(units = variable$node, activation = "relu") %>%
          layer_dropout(dropout)
      }
      
      model %>%
        # output layer
        layer_dense(units = 1, activation = "sigmoid")
      
      # add a loss function and optimizer
      model %>%
        compile(
          loss = "binary_crossentropy",
          optimizer = "Adam",
          metrics = "accuracy"
        )
      
      fit <- model %>% fit(
        x = as.matrix(train_df), 
        y = y_train,
        shuffle = T,
        batch_size = floor(dim(train_df)[1]*variable$batch_size/100),
        epochs = variable$epoch,
        validation_split = 0.2
      )
      
      l = list("val_loss" = fit$metrics$val_loss, "val_acc" = fit$metrics$val_acc,
               "loss" = fit$metrics$loss, "acc" = fit$metrics$acc)
      variable$cv_fit[[idx]] <- l
      
      # Predicted class
      variable$cv_pred_class[[idx]] <- predict_classes(object = model, x = as.matrix(valid_df[,1:13])) %>%
        as.vector()
      
      # Predicted class probability
      variable$cv_pred_prob[[idx]] <- predict_proba(object = model, x = as.matrix(valid_df[,1:13])) %>%
        as.vector()
      
      print(variable$cv_pred_class[[idx]])
      print(variable$cv_pred_prob[[idx]])
      print(evaluate(object = model, x = valid_df[,1:13], y = y_valid))
      
      # plot ROC and calculate AUC
      # par(mfrow = c(1, 1), mar = c(0.1, 0.1, 0.1, 0.1))
      variable$cv_roc_obj[[idx]] = roc(as.numeric(y_valid), as.numeric(variable$cv_pred_class[[idx]]))
      
      idx <- idx + 1
      
      
      # y <- rbind(variable$cv_pred_class[[idx]], t(y_valid))
      # write.csv(y, paste0("label","_fold_",f,".csv"), row.names = F) # "saves/cv/",
    } 
    
    variable$cv_flag <- TRUE
    
  })
  
  
  
  output$cv_acc_plot <- renderPlotly({
    if(variable$cv_flag == TRUE){
      
      p = plot_ly(x = 1:length(variable$cv_fit[[1]]$acc))
      
      for (i in 1:variable$k){
        p = p %>% add_trace(y = variable$cv_fit[[i]]$acc, name = paste('Accuracy cv-', i, sep=""), mode = 'lines+markers')
      }
      
      p = p %>% layout(title = "Cross Validation - Accuracy for Training Data")
        
      return (p)  
        
      # ti = 1:length(variable$fit$metrics$val_loss)
      # mva = lm(variable$fit$metrics$val_acc~ti+I(ti^2)+I(ti^3))
      # ma = lm(variable$fit$metrics$acc~ti+I(ti^2)+I(ti^3))
      # 
      # line.fmt = list(dash="solid", width = 1.5, color=NULL)
      # 
      # plot_ly(data = as.data.frame(variable$fit$metrics), x = 1:length(variable$fit$metrics$val_loss)) %>%
      #   add_trace(y = variable$fit$metrics$acc, name = 'Accuracy', mode = 'lines+markers') %>%
      #   add_trace(y = variable$fit$metrics$val_acc, name = 'Validation Accuracy', mode = 'lines+markers') %>%
      #   add_lines(y =  predict(ma), line=line.fmt, name="Accuracy Trend Line") %>%
      #   add_lines(y =  predict(mva), line=line.fmt, name="Validation Accuracy Trend Line") %>%
      #   layout(title = "Accuracy for Train and Validation")
      
    }
    else{
      return ()
    }
  })
  
  output$cv_val_acc_plot <- renderPlotly({
    if(variable$cv_flag == TRUE){
      
      p = plot_ly(x = 1:length(variable$cv_fit[[1]]$acc))
      
      for (i in 1:variable$k){
        p = p %>% add_trace(y = variable$cv_fit[[i]]$val_acc, name = paste('Validation Accuracy CV-', i, sep=""), mode = 'lines+markers')
      }
      
      p = p %>% layout(title = "Cross Validation - Accuracy for Validation Data")
      
      return (p)  
    }
    else{
      return ()
    }
  })
  
  output$cv_loss_plot <- renderPlotly({
    if(variable$cv_flag == TRUE){
      
      p = plot_ly(x = 1:length(variable$cv_fit[[1]]$acc))
      
      for (i in 1:variable$k){
        p = p %>% add_trace(y = variable$cv_fit[[i]]$loss, name = paste('Loss CV-', i, sep=""), mode = 'lines+markers')
      }
      
      p = p %>% layout(title = "Losses in Cross Validation for Train Data")
      
      return (p)  
    }
    else{
      return ()
    }
  })
  
  output$cv_val_loss_plot <- renderPlotly({
    if(variable$cv_flag == TRUE){
      
      p = plot_ly(x = 1:length(variable$cv_fit[[1]]$acc))
      
      for (i in 1:variable$k){
        p = p %>% add_trace(y = variable$cv_fit[[i]]$val_loss, name = paste('Validation Loss CV-', i, sep=""), mode = 'lines+markers')
      }
      
      p = p %>% layout(title = "Losses in Cross Validation for Validation Data")
      
      return (p)  
    }
    else{
      return ()
    }
  })
  
  output$cv_plot <- renderUI({
    tagList(
      plotlyOutput("cv_acc_plot"),
      tags$br(),
      plotlyOutput("cv_val_acc_plot"),
      tags$br(),
      plotlyOutput("cv_loss_plot"),
      tags$br(),
      plotlyOutput("cv_val_loss_plot"),
      tags$br(),
      h3("ROC plot for Cross Validation Traning Process"),
      plotOutput("rocPlot_cv")
    )
  })
  
  output$rocPlot_cv <- renderPlot({
    if (variable$cv_flag == TRUE){
      
      if (variable$k <= 3){
        par(mfrow = c(3, 1), mar = c(0.1, 0.1, 0.1, 0.1))
      }
      else if (variable$k <= 6){
        par(mfrow = c(3, 2), mar = c(0.1, 0.1, 0.1, 0.1))
      }
      else {
        par(mfrow = c(3, 3), mar = c(0.1, 0.1, 0.1, 0.1))
      }
      
      for (i in 1:variable$k){
        plot.roc(variable$cv_roc_obj[[i]], print.auc=TRUE, main=paste("ROC curve for CV-", i, sep=""))
      }
    }
    else {
      return ()
    }
    
  })
  
})
