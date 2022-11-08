library(reticulate)
py_install(".", pip = TRUE)

pathlib <- import("pathlib", convert=FALSE)
warnings <- import("warnings", convert=FALSE)
autoprognosis <- import("autoprognosis", convert=FALSE)

warnings$filterwarnings('ignore')

Path = pathlib$Path
RegressionStudy = autoprognosis$studies$regression$RegressionStudy
load_model_from_file = autoprognosis$utils$serialization$load_model_from_file
evaluate_regression = autoprognosis$utils$tester$evaluate_regression

workspace <- Path("workspace")
study_name <- "example"

# Load dataset
airfoil <- read.csv(
        url("https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat"),
        sep = "\t",
        header = FALSE,
)

target <- "V6"

# Create AutoPrognosis Study
study <- RegressionStudy(
	dataset = airfoil, 
	target = target,
	study_name=study_name,  
	num_iter=as.integer(10), 
	num_study_iter=as.integer(2), 
	timeout=as.integer(60), 
	regressors=list("linear_regression", "kneighbors_regressor"), 
	workspace=workspace
)

study$run()

# Load the optimal model - if exists
output <- sprintf("%s/%s/model.p", workspace, study_name)

model <- load_model_from_file(output)
# The model is not fitted yet here

# Benchmark the model
targets <- c(target)
X <- airfoil[ , !(names(iris) %in% targets)]
Y = airfoil[, target]

metrics <- evaluate_regression(model, X, Y)

sprintf("Performance metrics %s", metrics["str"])

# Fit the model
model$fit(X, Y)

# Predict 
model$predict(X)

