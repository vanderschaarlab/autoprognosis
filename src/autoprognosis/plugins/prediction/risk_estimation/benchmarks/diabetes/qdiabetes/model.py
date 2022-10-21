# stdlib
from typing import Any

# third party
import numpy as np
import pandas as pd

# autoprognosis absolute
import autoprognosis.logger as log
from autoprognosis.utils.metrics import evaluate_auc

# Model A


def type2_female_model_a(
    age: float,
    b_atypicalantipsy: bool,
    b_corticosteroids: bool,
    b_cvd: bool,
    b_gestdiab: bool,
    b_learning: bool,
    b_manicschiz: bool,
    b_pos: bool,
    b_statin: bool,
    b_treatedhyp: bool,
    bmi: float,
    ethrisk: int,
    fh_diab: int,
    smoke_cat: int,
    surv: int,
    town: float,
) -> float:
    surv = 10
    survivor = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.986227273941040]

    # /* The conditional arrays */

    Iethrisk = [
        0,
        0,
        1.0695857881565456000000000,
        1.3430172097414006000000000,
        1.8029022579794518000000000,
        1.1274654517708020000000000,
        0.4214631490239910100000000,
        0.2850919645908353000000000,
        0.8815108797589199500000000,
        0.3660573343168487300000000,
    ]
    Ismoke = [
        0,
        0.0656016901750590550000000,
        0.2845098867369837400000000,
        0.3567664381700702000000000,
        0.5359517110678775300000000,
    ]

    # /* Applying the fractional polynomial transforms */
    # /* (which includes scaling)                      */

    dage = age
    dage = dage / 10
    age_2 = pow(dage, 3)
    age_1 = pow(dage, 0.5)
    dbmi = bmi
    dbmi = dbmi / 10
    bmi_1 = dbmi
    bmi_2 = pow(dbmi, 3)

    # /* Centring the continuous variables */

    age_1 = age_1 - 2.123332023620606
    age_2 = age_2 - 91.644744873046875
    bmi_1 = bmi_1 - 2.571253299713135
    bmi_2 = bmi_2 - 16.999439239501953
    town = town - 0.391116052865982

    # /* Start of Sum */
    a = 0.0

    # /* The conditional sums */

    a += Iethrisk[ethrisk]
    a += Ismoke[smoke_cat]

    # /* Sum from continuous values */

    a += age_1 * 4.3400852699139278000000000
    a += age_2 * -0.0048771702696158879000000
    a += bmi_1 * 2.9320361259524925000000000
    a += bmi_2 * -0.0474002058748434900000000
    a += town * 0.0373405696180491510000000

    # /* Sum from boolean values */

    a += b_atypicalantipsy * 0.5526764611098438100000000
    a += b_corticosteroids * 0.2679223368067459900000000
    a += b_cvd * 0.1779722905458669100000000
    a += b_gestdiab * 1.5248871531467574000000000
    a += b_learning * 0.2783514358717271700000000
    a += b_manicschiz * 0.2618085210917905900000000
    a += b_pos * 0.3406173988206666100000000
    a += b_statin * 0.6590728773280821700000000
    a += b_treatedhyp * 0.4394758285813711900000000
    a += fh_diab * 0.5313359456558733900000000

    # /* Sum from interaction terms */

    a += age_1 * b_atypicalantipsy * -0.8031518398316395100000000
    a += age_1 * b_learning * -0.8641596002882057100000000
    a += age_1 * b_statin * -1.9757776696583935000000000
    a += age_1 * bmi_1 * 0.6553138757562945200000000
    a += age_1 * bmi_2 * -0.0362096572016301770000000
    a += age_1 * fh_diab * -0.2641171450558896200000000
    a += age_2 * b_atypicalantipsy * 0.0004684041181021049800000
    a += age_2 * b_learning * 0.0006724968808953360200000
    a += age_2 * b_statin * 0.0023750534194347966000000
    a += age_2 * bmi_1 * -0.0044719662445263054000000
    a += age_2 * bmi_2 * 0.0001185479967753342000000
    a += age_2 * fh_diab * 0.0004161025828904768300000

    # /* Calculate the score itself */
    score = 100.0 * (1 - pow(survivor[surv], np.exp(a)))
    return score


def type2_male_model_a(
    age: float,
    b_atypicalantipsy: bool,
    b_corticosteroids: bool,
    b_cvd: bool,
    b_learning: bool,
    b_manicschiz: bool,
    b_statin: bool,
    b_treatedhyp: bool,
    bmi: float,
    ethrisk: int,
    fh_diab: int,
    smoke_cat: int,
    surv: int,
    town: float,
) -> float:
    surv = 10
    survivor = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.978732228279114]

    # /* The conditional arrays */

    Iethrisk = [
        0,
        0,
        1.1000230829124793000000000,
        1.2903840126147210000000000,
        1.6740908848727458000000000,
        1.1400446789147816000000000,
        0.4682468169065580600000000,
        0.6990564996301544800000000,
        0.6894365712711156800000000,
        0.4172222846773820900000000,
    ]
    Ismoke = [
        0,
        0.1638740910548557300000000,
        0.3185144911395897900000000,
        0.3220726656778343200000000,
        0.4505243716340953100000000,
    ]

    # /* Applying the fractional polynomial transforms */
    # /* (which includes scaling)                      */

    dage = age
    dage = dage / 10
    age_2 = pow(dage, 3)
    age_1 = np.log(dage)
    dbmi = bmi
    dbmi = dbmi / 10
    bmi_2 = pow(dbmi, 3)
    bmi_1 = pow(dbmi, 2)

    # /* Centring the continuous variables */

    age_1 = age_1 - 1.496392488479614
    age_2 = age_2 - 89.048171997070313
    bmi_1 = bmi_1 - 6.817805767059326
    bmi_2 = bmi_2 - 17.801923751831055
    town = town - 0.515986680984497

    # /* Start of Sum */
    a = 0.0

    # /* The conditional sums */

    a += Iethrisk[ethrisk]
    a += Ismoke[smoke_cat]

    # /* Sum from continuous values */

    a += age_1 * 4.4642324388691348000000000
    a += age_2 * -0.0040750108019255568000000
    a += bmi_1 * 0.9512902786712067500000000
    a += bmi_2 * -0.1435248827788547500000000
    a += town * 0.0259181820676787250000000

    # /* Sum from boolean values */

    a += b_atypicalantipsy * 0.4210109234600543600000000
    a += b_corticosteroids * 0.2218358093292538400000000
    a += b_cvd * 0.2026960575629002100000000
    a += b_learning * 0.2331532140798696100000000
    a += b_manicschiz * 0.2277044952051772700000000
    a += b_statin * 0.5849007543114134200000000
    a += b_treatedhyp * 0.3337939218350107800000000
    a += fh_diab * 0.6479928489936953600000000

    # /* Sum from interaction terms */

    a += age_1 * b_atypicalantipsy * -0.9463772226853415200000000
    a += age_1 * b_learning * -0.9384237552649983300000000
    a += age_1 * b_statin * -1.7479070653003299000000000
    a += age_1 * bmi_1 * 0.4514759924187976600000000
    a += age_1 * bmi_2 * -0.1079548126277638100000000
    a += age_1 * fh_diab * -0.6011853042930119800000000
    a += age_2 * b_atypicalantipsy * -0.0000519927442172335000000
    a += age_2 * b_learning * 0.0007102643855968814100000
    a += age_2 * b_statin * 0.0013508364599531669000000
    a += age_2 * bmi_1 * -0.0011797722394560309000000
    a += age_2 * bmi_2 * 0.0002147150913931929100000
    a += age_2 * fh_diab * 0.0004914185594087803400000

    # /* Calculate the score itself */
    score = 100.0 * (1 - pow(survivor[surv], np.exp(a)))
    return score


# Model B


def type2_female_model_b(
    age: float,
    b_atypicalantipsy: bool,
    b_corticosteroids: bool,
    b_cvd: bool,
    b_gestdiab: bool,
    b_learning: bool,
    b_manicschiz: bool,
    b_pos: bool,
    b_statin: bool,
    b_treatedhyp: bool,
    bmi: float,
    ethrisk: int,
    fbs: float,  # fasting blood glucose
    fh_diab: int,
    smoke_cat: int,
    surv: int,
    town: float,
) -> float:
    surv = 10
    survivor = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.990905702114105]

    # /* The conditional arrays */

    Iethrisk = [
        0,
        0,
        0.9898906127239111000000000,
        1.2511504196326508000000000,
        1.4934757568196120000000000,
        0.9673887434565966400000000,
        0.4844644519593178100000000,
        0.4784214955360102700000000,
        0.7520946270805577400000000,
        0.4050880741541424400000000,
    ]
    Ismoke = [
        0,
        0.0374156307236963230000000,
        0.2252973672514482800000000,
        0.3099736428023662800000000,
        0.4361942139496417500000000,
    ]

    # /* Applying the fractional polynomial transforms */
    # /* (which includes scaling)                      */

    dage = age
    dage = dage / 10
    age_1 = pow(dage, 0.5)
    age_2 = pow(dage, 3)
    dbmi = bmi
    dbmi = dbmi / 10
    bmi_2 = pow(dbmi, 3)
    bmi_1 = dbmi
    dfbs = fbs

    fbs_2 = pow(dfbs, -1) * np.log(dfbs)
    fbs_1 = pow(dfbs, -1)

    # /* Centring the continuous variables */

    age_1 = age_1 - 2.123332023620606
    age_2 = age_2 - 91.644744873046875
    bmi_1 = bmi_1 - 2.571253299713135
    bmi_2 = bmi_2 - 16.999439239501953
    fbs_1 = fbs_1 - 0.208309367299080
    fbs_2 = fbs_2 - 0.326781362295151
    town = town - 0.391116052865982

    # /* Start of Sum */
    a = 0.0

    # /* The conditional sums */

    a += Iethrisk[ethrisk]
    a += Ismoke[smoke_cat]

    # /* Sum from continuous values */

    a += age_1 * 3.7650129507517280000000000
    a += age_2 * -0.0056043343436614941000000
    a += bmi_1 * 2.4410935031672469000000000
    a += bmi_2 * -0.0421526334799096420000000
    a += fbs_1 * -2.1887891946337308000000000
    a += fbs_2 * -69.9608419828660290000000000
    a += town * 0.0358046297663126500000000

    # /* Sum from boolean values */

    a += b_atypicalantipsy * 0.4748378550253853400000000
    a += b_corticosteroids * 0.3767933443754728500000000
    a += b_cvd * 0.1967261568066525100000000
    a += b_gestdiab * 1.0689325033692647000000000
    a += b_learning * 0.4542293408951034700000000
    a += b_manicschiz * 0.1616171889084260500000000
    a += b_pos * 0.3565365789576717100000000
    a += b_statin * 0.5809287382718667500000000
    a += b_treatedhyp * 0.2836632020122907300000000
    a += fh_diab * 0.4522149766206111600000000

    # /* Sum from interaction terms */

    a += age_1 * b_atypicalantipsy * -0.7683591642786522500000000
    a += age_1 * b_learning * -0.7983128124297588200000000
    a += age_1 * b_statin * -1.9033508839833257000000000
    a += age_1 * bmi_1 * 0.4844747602404915200000000
    a += age_1 * bmi_2 * -0.0319399883071813450000000
    a += age_1 * fbs_1 * 2.2442903047404350000000000
    a += age_1 * fbs_2 * 13.0068388699783030000000000
    a += age_1 * fh_diab * -0.3040627374034501300000000
    a += age_2 * b_atypicalantipsy * 0.0005194455624413476200000
    a += age_2 * b_learning * 0.0003028327567161890600000
    a += age_2 * b_statin * 0.0024397111406018711000000
    a += age_2 * bmi_1 * -0.0041572976682154057000000
    a += age_2 * bmi_2 * 0.0001126882194204252200000
    a += age_2 * fbs_1 * 0.0199345308534312550000000
    a += age_2 * fbs_2 * -0.0716677187529306680000000
    a += age_2 * fh_diab * 0.0004523639671202325400000

    # /* Calculate the score itself */
    score = 100.0 * (1 - pow(survivor[surv], np.exp(a)))
    return score


def type2_male_model_b(
    age: float,
    b_atypicalantipsy: bool,
    b_corticosteroids: bool,
    b_cvd: bool,
    b_learning: bool,
    b_manicschiz: bool,
    b_statin: bool,
    b_treatedhyp: bool,
    bmi: float,
    ethrisk: int,
    fbs: float,  # fasting blood glucose
    fh_diab: int,
    smoke_cat: int,
    surv: int,
    town: float,
) -> float:
    surv = 10
    survivor = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.985019445419312]

    # /* The conditional arrays */

    Iethrisk = [
        0,
        0,
        1.0081475800686235000000000,
        1.3359138425778705000000000,
        1.4815419524892652000000000,
        1.0384996851820663000000000,
        0.5202348070887524700000000,
        0.8579673418258558800000000,
        0.6413108960765615500000000,
        0.4838340220821504800000000,
    ]
    Ismoke = [
        0,
        0.1119475792364162500000000,
        0.3110132095412204700000000,
        0.3328898469326042100000000,
        0.4257069026941993100000000,
    ]

    # /* Applying the fractional polynomial transforms */
    # /* (which includes scaling)                      */

    dage = age
    dage = dage / 10
    age_1 = np.log(dage)
    age_2 = pow(dage, 3)
    dbmi = bmi
    dbmi = dbmi / 10
    bmi_1 = pow(dbmi, 2)
    bmi_2 = pow(dbmi, 3)
    dfbs = fbs
    fbs_1 = pow(dfbs, -0.5)
    fbs_2 = pow(dfbs, -0.5) * np.log(dfbs)

    # /* Centring the continuous variables */

    age_1 = age_1 - 1.496392488479614
    age_2 = age_2 - 89.048171997070313
    bmi_1 = bmi_1 - 6.817805767059326
    bmi_2 = bmi_2 - 17.801923751831055
    fbs_1 = fbs_1 - 0.448028832674026
    fbs_2 = fbs_2 - 0.719442605972290
    town = town - 0.515986680984497

    # /* Start of Sum */
    a = 0.0

    # /* The conditional sums */

    a += Iethrisk[ethrisk]
    a += Ismoke[smoke_cat]

    # /* Sum from continuous values */

    a += age_1 * 4.1149143302364717000000000
    a += age_2 * -0.0047593576668505362000000
    a += bmi_1 * 0.8169361587644297100000000
    a += bmi_2 * -0.1250237740343336200000000
    a += fbs_1 * -54.8417881280971070000000000
    a += fbs_2 * -53.1120784984813600000000000
    a += town * 0.0253741755198943560000000

    # /* Sum from boolean values */

    a += b_atypicalantipsy * 0.4417934088889577400000000
    a += b_corticosteroids * 0.3413547348339454100000000
    a += b_cvd * 0.2158977454372756600000000
    a += b_learning * 0.4012885027585300100000000
    a += b_manicschiz * 0.2181769391399779300000000
    a += b_statin * 0.5147657600111734700000000
    a += b_treatedhyp * 0.2467209287407037300000000
    a += fh_diab * 0.5749437333987512700000000

    # /* Sum from interaction terms */

    a += age_1 * b_atypicalantipsy * -0.9502224313823126600000000
    a += age_1 * b_learning * -0.8358370163090045300000000
    a += age_1 * b_statin * -1.8141786919269460000000000
    a += age_1 * bmi_1 * 0.3748482092078384600000000
    a += age_1 * bmi_2 * -0.0909836579562487420000000
    a += age_1 * fbs_1 * 21.0117301217643340000000000
    a += age_1 * fbs_2 * 23.8244600447469740000000000
    a += age_1 * fh_diab * -0.6780647705291665800000000
    a += age_2 * b_atypicalantipsy * 0.0001472972077162874300000
    a += age_2 * b_learning * 0.0006012919264966409100000
    a += age_2 * b_statin * 0.0016393484911405418000000
    a += age_2 * bmi_1 * -0.0010774782221531713000000
    a += age_2 * bmi_2 * 0.0001911048730458310100000
    a += age_2 * fbs_1 * -0.0390046079223835270000000
    a += age_2 * fbs_2 * -0.0411277198058959470000000
    a += age_2 * fh_diab * 0.0006257588248859499300000

    # /* Calculate the score itself */
    score = 100.0 * (1 - pow(survivor[surv], np.exp(a)))
    return score


# Model C
def type2_female_model_c(
    age: float,  # age value
    b_atypicalantipsy: bool,  # bool, On atypical antipsychotic medication
    b_corticosteroids: bool,  # Are you on regular steroid tablets?
    b_cvd: bool,  # Have you had a heart attack, angina, stroke or TIA?
    b_gestdiab: bool,  # Women: Do you have gestational diabetes ?
    b_learning: bool,  # Learning disabilities?
    b_manicschiz: bool,  # Manic depression or schizophrenia?
    b_pos: bool,  # Do you have polycystic ovaries?
    b_statin: bool,  # Are you on statins?
    b_treatedhyp: bool,  # Do you have high blood pressure requiring treatment?
    bmi: float,  # Body mass index = kg/m^2
    ethrisk: int,  # ethnic risk
    fh_diab: int,  # Do immediate family (mother, father, brothers or sisters) have diabetes?
    hba1c: float,  # HBA1c (mmol/mol)
    smoke_cat: int,  # smoking category: non-smoker, ex-smoker, light-smoker(less than 10/), moderate smoker(10-      19), heavy smoker(20 or over)
    surv: int,  # 10-year risk
    town: float,  # Townsend deprivation score
) -> float:
    surv = 10
    survivor = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.988788545131683]

    # /* The conditional arrays */

    Iethrisk = [
        0,
        0,
        0.5990951599291540800000000,
        0.7832030965635389300000000,
        1.1947351247960103000000000,
        0.7141744699168143300000000,
        0.1195328468388768800000000,
        0.0136688728784904270000000,
        0.5709226537693945500000000,
        0.1709107628106929200000000,
    ]
    Ismoke = [
        0,
        0.0658482585100006730000000,
        0.1458413689734224000000000,
        0.1525864247480118700000000,
        0.3078741679661397600000000,
    ]

    # /* Applying the fractional polynomial transforms */
    # /* (which includes scaling)                      */

    dage = age
    dage = dage / 10
    age_1 = pow(dage, 0.5)
    age_2 = pow(dage, 3)
    dbmi = bmi
    dbmi = dbmi / 10
    bmi_2 = pow(dbmi, 3)
    bmi_1 = dbmi
    dhba1c = hba1c
    dhba1c = dhba1c / 10
    hba1c_1 = pow(dhba1c, 0.5)
    hba1c_2 = dhba1c

    # /* Centring the continuous variables */

    age_1 = age_1 - 2.123332023620606
    age_2 = age_2 - 91.644744873046875
    bmi_1 = bmi_1 - 2.571253299713135
    bmi_2 = bmi_2 - 16.999439239501953
    hba1c_1 = hba1c_1 - 1.886751174926758
    hba1c_2 = hba1c_2 - 3.559829950332642
    town = town - 0.391116052865982

    # /* Start of Sum */
    a = 0.0

    # /* The conditional sums */

    a += Iethrisk[ethrisk]
    a += Ismoke[smoke_cat]

    # /* Sum from continuous values */

    a += age_1 * 3.5655214891947722000000000
    a += age_2 * -0.0056158243572733135000000
    a += bmi_1 * 2.5043028874544841000000000
    a += bmi_2 * -0.0428758018926904610000000
    a += hba1c_1 * 8.7368031307362184000000000
    a += hba1c_2 * -0.0782313866699499700000000
    a += town * 0.0358668220563482940000000

    # /* Sum from boolean values */

    a += b_atypicalantipsy * 0.5497633311042200400000000
    a += b_corticosteroids * 0.1687220550638970400000000
    a += b_cvd * 0.1644330036273934400000000
    a += b_gestdiab * 1.1250098105171140000000000
    a += b_learning * 0.2891205831073965800000000
    a += b_manicschiz * 0.3182512249068407700000000
    a += b_pos * 0.3380644414098174500000000
    a += b_statin * 0.4559396847381116400000000
    a += b_treatedhyp * 0.4040022295023758000000000
    a += fh_diab * 0.4428015404826031700000000

    # /* Sum from interaction terms */

    a += age_1 * b_atypicalantipsy * -0.8125434197162131300000000
    a += age_1 * b_learning * -0.9084665765269808200000000
    a += age_1 * b_statin * -1.8557960585560658000000000
    a += age_1 * bmi_1 * 0.6023218765235252000000000
    a += age_1 * bmi_2 * -0.0344950383968044700000000
    a += age_1 * fh_diab * -0.2727571351506187200000000
    a += age_1 * hba1c_1 * 25.4412033227367150000000000
    a += age_1 * hba1c_2 * -6.8076080421556107000000000
    a += age_2 * b_atypicalantipsy * 0.0004665611306005428000000
    a += age_2 * b_learning * 0.0008518980139928006500000
    a += age_2 * b_statin * 0.0022627250963352537000000
    a += age_2 * bmi_1 * -0.0043386645663133425000000
    a += age_2 * bmi_2 * 0.0001162778561671208900000
    a += age_2 * fh_diab * 0.0004354519795220774900000
    a += age_2 * hba1c_1 * -0.0522541355885925220000000
    a += age_2 * hba1c_2 * 0.0140548259061144530000000

    # /* Calculate the score itself */
    score = 100.0 * (1 - pow(survivor[surv], np.exp(a)))
    return score


def type2_male_model_c(
    age: float,  # age value
    b_atypicalantipsy: bool,  # bool, On atypical antipsychotic medication
    b_corticosteroids: bool,  # Are you on regular steroid tablets?
    b_cvd: bool,  # Have you had a heart attack, angina, stroke or TIA?
    b_learning: bool,  # Learning disabilities?
    b_manicschiz: bool,  # Manic depression or schizophrenia?
    b_statin: bool,  # Are you on statins?
    b_treatedhyp: bool,  # bool, On blood pressure treatment?
    bmi: float,  # Body mass index = kg/m^2
    ethrisk: int,  # ethnic risk
    fh_diab: int,  # Do immediate family (mother, father, brothers or sisters) have diabetes?
    hba1c: float,  # HBA1c (mmol/mol)
    smoke_cat: int,  # smoking category: non-smoker, ex-smoker, light-smoker(less than 10/), moderate smoker(10-  19), heavy smoker(20 or over)
    surv: int,  # 10-year risk
    town: float,  # Townsend deprivation score
) -> float:
    surv = 10
    survivor = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.981181740760803]

    # /* The conditional arrays */

    Iethrisk = [
        0,
        0,
        0.6757120705498780300000000,
        0.8314732504966345600000000,
        1.0969133802228563000000000,
        0.7682244636456048200000000,
        0.2089752925910850200000000,
        0.3809159378197057900000000,
        0.3423583679661269500000000,
        0.2204647785343308300000000,
    ]
    Ismoke = [
        0,
        0.1159289120687865100000000,
        0.1462418263763327100000000,
        0.1078142411249314200000000,
        0.1984862916366847400000000,
    ]

    # /* Applying the fractional polynomial transforms */
    # /* (which includes scaling)                      */

    dage = age
    dage = dage / 10
    age_1 = np.log(dage + 1e-8)
    age_2 = pow(dage, 3)
    dbmi = bmi
    dbmi = dbmi / 10
    bmi_1 = pow(dbmi, 2)
    bmi_2 = pow(dbmi, 3)
    dhba1c = hba1c
    dhba1c = dhba1c / 10
    hba1c_1 = pow(dhba1c, 0.5)
    hba1c_2 = dhba1c

    # /* Centring the continuous variables */

    age_1 = age_1 - 1.496392488479614
    age_2 = age_2 - 89.048171997070313
    bmi_1 = bmi_1 - 6.817805767059326
    bmi_2 = bmi_2 - 17.801923751831055
    hba1c_1 = hba1c_1 - 1.900265336036682
    hba1c_2 = hba1c_2 - 3.611008167266846
    town = town - 0.515986680984497

    # /* Start of Sum */
    a = 0.0

    # /* The conditional sums */

    a += Iethrisk[ethrisk]
    a += Ismoke[smoke_cat]

    # /* Sum from continuous values */

    a += age_1 * 4.0193435623978031000000000
    a += age_2 * -0.0048396442306278238000000
    a += bmi_1 * 0.8182916890534932500000000
    a += bmi_2 * -0.1255880870135964200000000
    a += hba1c_1 * 8.0511642238857934000000000
    a += hba1c_2 * -0.1465234689391449500000000
    a += town * 0.0252299651849007270000000

    # /* Sum from boolean values */

    a += b_atypicalantipsy * 0.4554152522017330100000000
    a += b_corticosteroids * 0.1381618768682392200000000
    a += b_cvd * 0.1454698889623951800000000
    a += b_learning * 0.2596046658040857000000000
    a += b_manicschiz * 0.2852378849058589400000000
    a += b_statin * 0.4255195190118552500000000
    a += b_treatedhyp * 0.3316943000645931100000000
    a += fh_diab * 0.5661232594368061900000000

    # /* Sum from interaction terms */

    a += age_1 * b_atypicalantipsy * -1.0013331909079835000000000
    a += age_1 * b_learning * -0.8916465737221592700000000
    a += age_1 * b_statin * -1.7074561167819817000000000
    a += age_1 * bmi_1 * 0.4507452747267244300000000
    a += age_1 * bmi_2 * -0.1085185980916560100000000
    a += age_1 * fh_diab * -0.6141009388709716100000000
    a += age_1 * hba1c_1 * 27.6705938271465650000000000
    a += age_1 * hba1c_2 * -7.4006134846785434000000000
    a += age_2 * b_atypicalantipsy * 0.0002245597398574240700000
    a += age_2 * b_learning * 0.0006604436076569648200000
    a += age_2 * b_statin * 0.0013873509357389619000000
    a += age_2 * bmi_1 * -0.0012224736160287865000000
    a += age_2 * bmi_2 * 0.0002266731010346126000000
    a += age_2 * fh_diab * 0.0005060258289477209100000
    a += age_2 * hba1c_1 * -0.0592014581247543300000000
    a += age_2 * hba1c_2 * 0.0155920894851499880000000

    # /* Calculate the score itself */
    score = 100.0 * (1 - pow(survivor[surv], np.exp(a)))
    return score


def inference(
    model: str,  # A, B, C
    gender: str,  # M/F
    age: float,  # age value
    b_atypicalantipsy: bool,  # bool, On atypical antipsychotic medication
    b_corticosteroids: bool,  # Are you on regular steroid tablets?
    b_cvd: bool,  # Have you had a heart attack, angina, stroke or TIA?
    b_gestdiab: bool,  # Women: Do you have gestational diabetes ?
    b_learning: bool,  # Learning disabilities?
    b_manicschiz: bool,  # Manic depression or schizophrenia?
    b_pos: bool,  # Do you have polycystic ovaries?
    b_statin: bool,  # Are you on statins?
    b_treatedhyp: bool,  # Do you have high blood pressure requiring treatment?
    bmi: float,  # Body mass index = kg/m^2
    ethrisk: int,  # ethnic risk
    fbs: float,  # fasting blood glucose
    fh_diab: int,  # Do immediate family (mother, father, brothers or sisters) have diabetes?
    hba1c: float,  # HBA1c (mmol/mol)
    smoke_cat: int,  # smoking category: non-smoker, ex-smoker, light-smoker(less than 10/), moderate smoker(10-      19), heavy smoker(20 or over)
    town: float,  # Townsend deprivation score
    surv: int = 10,  # 10-year risk
) -> float:

    if model == "A":
        if gender == "M":
            pct = type2_male_model_a(
                age=age,
                b_atypicalantipsy=b_atypicalantipsy,
                b_corticosteroids=b_corticosteroids,
                b_cvd=b_cvd,
                b_learning=b_learning,
                b_manicschiz=b_manicschiz,
                b_statin=b_statin,
                b_treatedhyp=b_treatedhyp,
                bmi=bmi,
                ethrisk=ethrisk,
                fh_diab=fh_diab,
                smoke_cat=smoke_cat,
                surv=surv,
                town=town,
            )
        else:
            pct = type2_female_model_a(
                age=age,
                b_atypicalantipsy=b_atypicalantipsy,
                b_corticosteroids=b_corticosteroids,
                b_cvd=b_cvd,
                b_gestdiab=b_gestdiab,
                b_learning=b_learning,
                b_manicschiz=b_manicschiz,
                b_pos=b_pos,
                b_statin=b_statin,
                b_treatedhyp=b_treatedhyp,
                bmi=bmi,
                ethrisk=ethrisk,
                fh_diab=fh_diab,
                smoke_cat=smoke_cat,
                surv=surv,
                town=town,
            )
    elif model == "B":
        if gender == "M":
            pct = type2_male_model_b(
                age=age,
                b_atypicalantipsy=b_atypicalantipsy,
                b_corticosteroids=b_corticosteroids,
                b_cvd=b_cvd,
                b_learning=b_learning,
                b_manicschiz=b_manicschiz,
                b_statin=b_statin,
                b_treatedhyp=b_treatedhyp,
                bmi=bmi,
                ethrisk=ethrisk,
                fbs=fbs,
                fh_diab=fh_diab,
                smoke_cat=smoke_cat,
                surv=surv,
                town=town,
            )
        else:
            pct = type2_female_model_b(
                age=age,
                b_atypicalantipsy=b_atypicalantipsy,
                b_corticosteroids=b_corticosteroids,
                b_cvd=b_cvd,
                b_gestdiab=b_gestdiab,
                b_learning=b_learning,
                b_manicschiz=b_manicschiz,
                b_pos=b_pos,
                b_statin=b_statin,
                b_treatedhyp=b_treatedhyp,
                bmi=bmi,
                ethrisk=ethrisk,
                fbs=fbs,
                fh_diab=fh_diab,
                smoke_cat=smoke_cat,
                surv=surv,
                town=town,
            )
    elif model == "C":
        if gender == "M":
            pct = type2_male_model_c(
                age=age,
                b_atypicalantipsy=b_atypicalantipsy,
                b_corticosteroids=b_corticosteroids,
                b_cvd=b_cvd,
                b_learning=b_learning,
                b_manicschiz=b_manicschiz,
                b_statin=b_statin,
                b_treatedhyp=b_treatedhyp,
                bmi=bmi,
                ethrisk=ethrisk,
                fh_diab=fh_diab,
                hba1c=hba1c,
                smoke_cat=smoke_cat,
                surv=surv,
                town=town,
            )
        else:
            pct = type2_female_model_c(
                age=age,
                b_atypicalantipsy=b_atypicalantipsy,
                b_corticosteroids=b_corticosteroids,
                b_cvd=b_cvd,
                b_gestdiab=b_gestdiab,
                b_learning=b_learning,
                b_manicschiz=b_manicschiz,
                b_pos=b_pos,
                b_statin=b_statin,
                b_treatedhyp=b_treatedhyp,
                bmi=bmi,
                ethrisk=ethrisk,
                fh_diab=fh_diab,
                hba1c=hba1c,
                smoke_cat=smoke_cat,
                surv=surv,
                town=town,
            )

    return pct / 100.0


class QDiabetesModel:
    def __init__(self, model_type: str) -> None:
        self.model_type = model_type

    def fit(self, *args: Any, **kwargs: Any) -> "QDiabetesModel":
        return self

    def score(self, X: pd.DataFrame, Y: pd.DataFrame) -> float:
        local_preds = self.predict(X).to_numpy()[:, 0]
        local_surv_pred = 1 - local_preds

        full_proba = []
        full_proba.append(local_surv_pred)
        full_proba.append(local_preds)
        full_proba = pd.DataFrame(full_proba).T

        return evaluate_auc(Y, full_proba)[0]

    def predict(
        self, df: pd.DataFrame, times: list = []
    ) -> Any:  # times is considered always ten years
        def qdiabetes_inference(row: pd.DataFrame) -> Any:
            score = inference(
                model=self.model_type,
                gender=row["sex"],  # M/F
                age=row["age"],  # age value
                b_atypicalantipsy=row[
                    "b_antipsychotic_use"
                ],  # bool, On atypical antipsychotic medication
                b_corticosteroids=row[
                    "b_steroid_treat"
                ],  # Are you on regular steroid tablets?
                b_cvd=row[
                    "b_cvd"
                ],  # Have you had a heart attack, angina, stroke or TIA?
                b_gestdiab=row[
                    "b_gestdiab"
                ],  # Women: Do you have gestational diabetes ?
                b_learning=row["b_learning"],  # Learning disabilities?
                b_manicschiz=row["b_manicschiz"],  # Manic depression or schizophrenia?
                b_pos=row["b_pos"],  # Do you have polycystic ovaries?
                b_statin=row["b_statin"],  # Are you on statins?
                b_treatedhyp=row["ht_treat"],  # On blood pressure treatment?
                bmi=row["bmi"],  # Body mass index = kg/m^2
                ethrisk=row["ethrisk"],  # ethnic risk
                fbs=row["fbs"],  # fasting blood glucose
                fh_diab=row[
                    "fh_diab"
                ],  # Do immediate family (mother, father, brothers or sisters) have diabetes?
                hba1c=row["hba1c"],  # HBA1c (mmol/mol)
                smoke_cat=row[
                    "smoker"
                ],  # smoking category: non-smoker, ex-smoker, light-smoker(less than 10/), moderate smoker(10-      19), heavy smoker(20 or over)
                town=row["town_depr_index"],  # Townsend deprivation score
            )
            return score

        expected_cols = [
            "sex",
            "age",
            "b_antipsychotic_use",
            "b_steroid_treat",
            "b_cvd",
            "b_gestdiab",
            "b_learning",
            "b_manicschiz",
            "b_pos",
            "b_statin",
            "ht_treat",
            "bmi",
            "ethrisk",
            "fh_diab",
            "hba1c",
            "smoker",
            "town_depr_index",
        ]
        for col in expected_cols:
            if col not in df.columns:
                log.error(f"[QDiab] missing {col}")
                df[col] = 0

        scores = df.apply(lambda row: qdiabetes_inference(row), axis=1)

        return pd.DataFrame(scores, columns=[10 * 365])
