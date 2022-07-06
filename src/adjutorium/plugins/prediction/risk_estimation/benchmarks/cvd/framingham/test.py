# third party
from model import inference

score = inference(
    sex="F",
    age=60,  # age value
    total_cholesterol=204,
    hdl_cholesterol=38.67,
    systolic_blood_pressure=160,  # Systolic blood pressure
    smoker=True,
    blood_pressure_med_treatment=True,
)

print(score)
