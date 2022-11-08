library(reticulate)
install.packages("AppliedPredictiveModeling")
library(AppliedPredictiveModeling)

py_install(".", pip = TRUE)

pathlib <- import("pathlib", convert=FALSE)
warnings <- import("warnings", convert=FALSE)
autoprognosis <- import("autoprognosis", convert=FALSE)

warnings$filterwarnings('ignore')

Path = pathlib$Path
RegressionStudy = autoprognosis$studies$regression$RegressionStudy
load_model_from_file = autoprognosis$utils$serialization$load_model_from_file
evaluate_regression = autoprognosis$utils$tester$evaluate_regression

data("abalone")


workspace <- Path("workspace")
study_name <- "example"

target <- "Rings"

study <- RegressionStudy(
	dataset = abalone, 
	target = target,
	study_name=study_name,  
	num_iter=as.integer(10), 
	num_study_iter=as.integer(2), 
	timeout=as.integer(60), 
	regressors=list("linear_regression", "kneighbors_regressor"), 
	workspace=workspace
)

study$run()

output <- sprintf("%s/%s/model.p", workspace, study_name)

model <- load_model_from_file(output)
# The model is not fitted yet here

targets <- c(target)
X <- abalone[ , !(names(abalone) %in% targets)]
Y = abalone[, target]

metrics <- evaluate_regression(model, X, Y)

# Fit the model
model$fit(X, Y)

sprintf("Performance metrics %s", metrics["str"])
