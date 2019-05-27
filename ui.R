#
# This is the user-interface definition of a Shiny web application. You can
# run the application by clicking 'Run App' above.
#
# Find out more about building applications with Shiny here:
# 
#    http://shiny.rstudio.com/
#

library(reticulate)
library(shiny)
library(shinydashboard)
library(DT)
library(plotly)

dbHeader <- dashboardHeader(title = "CorrSave")

ui <- dashboardPage(  
  dbHeader,
  dashboardSidebar(
    sidebarMenu(
      # First Tab Menu
      menuItem("Image Classification", tabName = "image_classification", icon = icon("dashboard"))
    )
  ),
  dashboardBody(
    tags$style(type="text/css",
               ".shiny-output-error { visibility: hidden; }",
               ".shiny-output-error:before { visibility: hidden; }"
      ),
      tabItems(
        tabItem(
          tabName = "image_classification",
          fluidPage(
            titlePanel("Corrosion Classification"),
            tabsetPanel(
              tabPanel("Model Initialization",
                sidebarLayout(
                  sidebarPanel(
                    wellPanel(
                      h2("Initialize Model Parameter"),
                      helpText("Please fill in the neural network parameter to initialize the model"),
                      numericInput("node", label = h4("Number of node per layer"), value = 10, min = 5, max = NA, step = NA, width = NULL),
                      numericInput("layer", label = h4("Number of layer"), value = 5, min = NA, max = NA, step = NA, width = NULL),
                      numericInput("epoch", label = h4("Number of Passes (Epoch)"), value = 1000, min = 50, max = NA, step = NA, width = NULL),
                      numericInput("dropout", label = h4("Dropout Rate"), value = 0.5, min = 0, max = 1, step = 0.1, width = NULL),
                      sliderInput("batch_size", label = h4("Batch Size Percentage"), min = 10, max = 100, value = 60),
                      sliderInput("train_ratio", label = h4("Train-Test Split Percentage"), min = 50, max = 90, value = 70, step = 5),
                      actionButton("save_param", label = "Save Parameter")
                    )
                 ),
                 mainPanel(
                   fluidRow(
                       br(),
                       h2("Network Architecture"),
                       verbatimTextOutput("value")
                    )
                  )
                )
              ),
              tabPanel("Train Model",
                       sidebarLayout(
                         sidebarPanel(
                           wellPanel(
                               h2("Train Model"),
                               helpText("Please select two folder contains different image classes"),
                               
                               tags$div(class="form-group shiny-input-container", 
                                        tags$div(tags$label("Class 0 Folder")),
                                        tags$div(tags$label("Choose Class 0 folder", class="btn btn-primary",
                                                            tags$input(id = "fileIn", webkitdirectory = TRUE, type = "file", style="display: none;", onchange="pressed()"))),
                                        tags$label("No folder choosen", id = "noFile"),
                                        tags$div(id="fileIn_progress", class="progress progress-striped active shiny-file-input-progress",
                                                 tags$div(class="progress-bar")
                                        )     
                               ),
                               
                               tags$div(class="form-group shiny-input-container", 
                                        tags$div(tags$label("Class 1 Folder")),
                                        tags$div(tags$label("Choose Class 1 folder", class="btn btn-primary",
                                                            tags$input(id = "fileIn2", webkitdirectory = TRUE, type = "file", style="display: none;", onchange="pressed2()"))),
                                        tags$label("No folder choosen", id = "noFile2"),
                                        tags$div(id="fileIn2_progress", class="progress progress-striped active shiny-file-input-progress",
                                                 tags$div(class="progress-bar")
                                        )     
                               ),
                               actionButton("calculate_Features", label = "Calculate Features", width = "50%"),
                               tags$br(),
                               tags$br(),
                               actionButton("start_training", label = "Start Training", width = "50%"),
                               
                               HTML("<script type='text/javascript' src='getFolders.js'></script>"),
                               HTML("<script type='text/javascript' src='text.js'></script>")
                           ),
                           wellPanel(
                             h2("Pre-trained Model"),
                             helpText("Load existing model if available"),
                             fileInput("loadModel", "Choose Model File"),
                             fileInput("loadScaleAttr", "Choose Scale Attribute File")
                           )
                           
                         ),
                         mainPanel(
                           fluidRow(
                              br(),
                              h2("Features Summary"),
                              DT::dataTableOutput ("summTable"),
                              uiOutput('ui.trainPerformance')
                             
                           )
                         )
                       )),

              tabPanel("Test Model",
                       sidebarLayout(
                         sidebarPanel(
                           wellPanel(
                             h2("Single Image Test"),
                             h3("Load Local Image"),
                             helpText("Choose an image to test the classifier"),
                             fileInput("loadImage", "Choose an Image"),
                             actionButton("test_single_image", label = "Classify Image"),
                             tags$hr(),
                             h3("Load Image from URL"),
                             helpText("Add Image URL to test the classifier"),
                             textInput("imageURL", label = "ImageURL", placeholder = "Paste the URL here.."),
                             actionButton("load_imageFromUrl", label = "Load Image"),
                             uiOutput('ui.imageFromUrl'),
                             tags$br(),
                             actionButton("test_single_image_from_url", label = "Classify Image")
                           ),
                           wellPanel(
                             h2("Multiple Image Test"),
                             helpText("Choose folder contain images to test the classifier"),
                             tags$div(class="form-group shiny-input-container", 
                                      tags$div(tags$label("Image Folder")),
                                      tags$div(tags$label("Choose Image folder", class="btn btn-primary",
                                                          tags$input(id = "imageIn", webkitdirectory = TRUE, type = "file", style="display: none;", onchange="pressed3()"))),
                                      tags$label("No folder choosen", id = "noFile3"),
                                      tags$div(id="imageIn_progress", class="progress progress-striped active shiny-file-input-progress",
                                               tags$div(class="progress-bar")
                                      )     
                             ),
                             actionButton("test_multi_image", label = "Classify Image")
                           )
                           
                           
                         ),
                         mainPanel(
                           fluidRow(
                              br(),
                              h2("Image Detail"),
                              verbatimTextOutput("imgDetail"),
                              h2("Test Result"),
                              verbatimTextOutput("predResult1"),
                              verbatimTextOutput("predResult2")
                             
                           )
                         )
                       )
              )
            )
          )
          )
    )
  )
)

