# third party
from model import inference

score = inference(
    gender="F",
    age=64,  # age value
    bmi=24,  # Body mass index = kg/m^2
    waist=80,
    b_daily_exercise=1,
    b_daily_vegs=1,
    b_treatedhyp=1,  # Do you have high blood pressure requiring treatment?
    b_ever_had_high_glucose=1,
    fh_diab=0,  # Do immediate family (mother, father, brothers or sisters) have diabetes?
)

print(score)
