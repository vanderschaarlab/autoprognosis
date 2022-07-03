# third party
from model import inference

score = inference(
    gender="M",
    age=40,
    tchol=160,
    hdlc=40,
    sbp=180,
    smoking=0,
    diab=0,
    ht_treat=1,
    race="W",
)
print(score)
