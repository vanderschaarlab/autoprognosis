library(reticulate)
library(survival)

# geomloss bug
py_install("numpy", pip = TRUE)
py_install("torch", pip = TRUE)

# install autoprognosis
py_install(".", pip = TRUE)

pathlib <- import("pathlib", convert=FALSE)
warnings <- import("warnings", convert=FALSE)
autoprognosis <- import("autoprognosis", convert=FALSE)
np <- import("numpy", convert=FALSE)

warnings$filterwarnings('ignore')

Path = pathlib$Path
RiskEstimationStudy = autoprognosis$studies$risk_estimation$RiskEstimationStudy
load_model_from_file = autoprognosis$utils$serialization$load_model_from_file
evaluate_survival_estimator = autoprognosis$utils$tester$evaluate_survival_estimator

workspace <- Path("workspace")
study_name <- "example_risk_estimation"

# Load the data
data(cancer, package="survival")

targets <- c("dtime", "death")
df <- rotterdam

X <- df[ , !(names(df) %in% targets)]
Y <- df[, "death"]
T <- df[, "dtime"]

eval_time_horizons <- list(2000)

# Create the AutoPrognosis Study
study <- RiskEstimationStudy(
	dataset = df,
	target = "death",
    time_to_event="dtime",
    time_horizons = eval_time_horizons,
	study_name=study_name,
	num_iter=as.integer(10),
	num_study_iter=as.integer(2),
	timeout=as.integer(60),
	risk_estimators=list("cox_ph", "survival_xgboost"),
	workspace=workspace
)

study$run()

# Load the optimal model - if exists
output <- sprintf("%s/%s/model.p", workspace, study_name)

model <- load_model_from_file(output)
# The model is not fitted yet here

# Benchmark the model
metrics <- evaluate_survival_estimator(model, X, T, Y, eval_time_horizons)

# Fit the model
model$fit(X, T, Y)

sprintf("Performance metrics %s", metrics["str"])

# Predict using the model
model$predict(X)
