library(reticulate)
py_install(".", pip = TRUE)

pathlib <- import("pathlib", convert=FALSE)
warnings <- import("warnings", convert=FALSE)
autoprognosis <- import("autoprognosis", convert=FALSE)

warnings$filterwarnings('ignore')

Path = pathlib$Path
ClassifierStudy = autoprognosis$studies$classifiers$ClassifierStudy
load_model_from_file = autoprognosis$utils$serialization$load_model_from_file
evaluate_estimator = autoprognosis$utils$tester$evaluate_estimator

data("iris")


workspace <- Path("workspace")
study_name <- "example"

study <- ClassifierStudy(
	dataset = iris, 
	target = "Species",
	study_name=study_name,  
	num_iter=as.integer(10), 
	num_study_iter=as.integer(2), 
	timeout=as.integer(60), 
	classifiers=list("logistic_regression", "lda", "qda"), 
	workspace=workspace
)

study$run()

output <- sprintf("%s/%s/model.p", workspace, study_name)

model <- load_model_from_file(output)
# The model is not fitted yet here

targets <- c("Species")
X <- iris[ , !(names(iris) %in% targets)]
Y = iris[, "Species"]

metrics <- evaluate_estimator(model, X, Y)

# Fit the model
model$fit(X, Y)

sprintf("Performance metrics %s", metrics["str"])
