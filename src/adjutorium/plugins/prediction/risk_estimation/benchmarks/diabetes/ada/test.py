# third party
from model import inference

score = inference(
    gender="F",
    age=64,  # age value
    fh_diab=0,  # Do immediate family (mother, father, brothers or sisters) have diabetes?
    b_treatedhyp=1,  # Do you have high blood pressure requiring treatment?
    b_daily_exercise=1,
    bmi=24,  # Body mass index = kg/m^2
)

print(score)
