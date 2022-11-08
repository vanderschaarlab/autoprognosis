library(reticulate)
py_install("autoprognosis", pip = TRUE)

pathlib <- import("pathlib", convert=FALSE)
warnings <- import("warnings", convert=FALSE)
autoprognosis <- import("autoprognosis", convert=FALSE)

warnings$filterwarnings('ignore')

Path = pathlib$Path
ClassifierStudy = autoprognosis$studies$classifiers$ClassifierStudy
load_model_from_file = autoprognosis$utils$serialization$load_model_from_file
evaluate_estimator = autoprognosis$utils$tester$evaluate_estimator
workspace <- Path("workspace")
study_name <- "example_classifier_with_miss"

# Load the data
adult <- read.table('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data', 
                    sep = ',', fill = F, strip.white = T)
colnames(adult) <- c('age', 'workclass', 'fnlwgt', 'educatoin', 
                     'educatoin_num', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 
                     'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income')
adult[adult == "?"] <- NA
adult <- adult[ , !(names(adult) %in% c("native_country"))]

df <- adult

target <- "income"

# Create the AutoPrognosis Study
study <- ClassifierStudy(
	dataset = df, 
	target = target,
	study_name=study_name,  
	num_iter=as.integer(10), 
	num_study_iter=as.integer(2), 
	timeout=as.integer(60), 
	classifiers=list("logistic_regression", "lda", "qda"), 
	workspace=workspace,
    nan_placeholder='NA'
)

study$run()

# Load the optimal model - if exists
output <- sprintf("%s/%s/model.p", workspace, study_name)

model <- load_model_from_file(output)
# The model is not fitted yet here

# Benchmark the model
targets <- c(target)
X <- df[ , !(names(df) %in% targets)]
Y = df[, target]

metrics <- evaluate_estimator(model, X, Y)

# Fit the model
model$fit(X, Y)

sprintf("Performance metrics %s", metrics["str"])

# Predict using the model
model$predict_proba(X)
