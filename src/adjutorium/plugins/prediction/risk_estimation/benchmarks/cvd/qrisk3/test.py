# third party
from model import inference

score = inference(
    gender="F",
    age=44,  # age value
    b_AF=1,  # bool, Atrial fibrillation
    b_atypicalantipsy=1,  # bool, On atypical antipsychotic medication
    b_corticosteroids=1,  # Are you on regular steroid tablets?
    b_impotence2=False,
    b_migraine=1,  # bool, Do you have migraines?
    b_ra=0,  # Rheumatoid arthritis?
    b_renal=0,  # Chronic kidney disease (stage 3, 4 or 5)?
    b_semi=0,  # Severe mental illness?
    b_sle=1,  # bool, Systemic lupus erythematosus
    b_treatedhyp=1,  # bool, On blood pressure treatment?
    b_type1=0,  # Diabetes status: type 1
    b_type2=0,  # Diabetes status: type 2
    bmi=25,  # Body mass index = kg/m^2
    ethrisk=0,  # ethnic risk
    fh_cvd=0,  # Angina or heart attack in a 1st degree relative < 60?
    rati=5,  # Cholesterol/HDL ratio
    sbp=180,  # Systolic blood pressure
    sbps5=20,  # Standard deviation of at least two most recent systolic blood pressure readings (mmHg)
    smoke_cat=0,  # smoking category: non-smoker, ex-smoker, light-smoker(less than 10/), moderate smoker(10- 19), heavy smoker(20 or over)
    town=0,  # Townsend deprivation score
)

print(score)
