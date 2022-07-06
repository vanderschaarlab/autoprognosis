# Female
# /*
# * Copyright 2017 ClinRisk Ltd.
# *
# * This file is part of QRISK3-2017 (https://qrisk.org).
# *
# * QRISK3-2017 is free software: you can redistribute it and/or modify
# * it under the terms of the GNU Lesser General Public License as published by
# * the Free Software Foundation, either version 3 of the License, or
# * (at your option) any later version.
# *
# * QRISK3-2017 is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU Lesser General Public License for more details.
# *
# * You should have received a copy of the GNU Lesser General Public License
# * along with QRISK3-2017.  If not, see http://www.gnu.org/licenses/.
# *
# * Additional terms
# *
# * The following disclaimer must be held together with any risk score score generated by this code.
# * If the score is displayed, then this disclaimer must be displayed or otherwise be made easily accessible, e.g. by a prominent link alongside it.
# *   The initial version of this file, to be found at http://svn.clinrisk.co.uk/opensource/qrisk2, faithfully implements QRISK3-2017.
# *   ClinRisk Ltd. have released this code under the GNU Lesser General Public License to enable others to implement the algorithm faithfully.
# *   However, the nature of the GNU Lesser General Public License is such that we cannot prevent, for example, someone accidentally
# *   altering the coefficients, getting the inputs wrong, or just poor programming.
# *   ClinRisk Ltd. stress, therefore, that it is the responsibility of the end user to check that the source that they receive produces the same
# *   results as the original code found at https://qrisk.org.
# *   Inaccurate implementations of risk scores can lead to wrong patients being given the wrong treatment.
# *
# * End of additional terms
# *
# */
# stdlib
from typing import Any

# third party
import numpy as np
import pandas as pd

# adjutorium absolute
import adjutorium.logger as log
from adjutorium.utils.metrics import evaluate_auc


def cvd_female_raw(
    age: float,  # age value
    b_AF: int,  # bool, Atrial fibrillation
    b_atypicalantipsy: int,  # bool, On atypical antipsychotic medication
    b_corticosteroids: int,  # Are you on regular steroid tablets?
    b_migraine: int,  # bool, Do you have migraines?
    b_ra: int,  # Rheumatoid arthritis?
    b_renal: int,  # Chronic kidney disease (stage 3, 4 or 5)?
    b_semi: int,  # Severe mental illness?
    b_sle: int,  # bool, Systemic lupus erythematosus
    b_treatedhyp: int,  # bool, On blood pressure treatment?
    b_type1: int,  # Diabetes status: type 1
    b_type2: int,  # Diabetes status: type 2
    bmi: float,  # Body mass index = kg/m^2
    ethrisk: int,  # ethnic risk
    fh_cvd: int,  # Angina or heart attack in a 1st degree relative < 60?
    rati: float,  # Cholesterol/HDL ratio
    sbp: float,  # Systolic blood pressure
    sbps5: float,  # Standard deviation of at least two most recent systolic blood pressure readings (mmHg)
    smoke_cat: int,  # smoking category: non-smoker, ex-smoker, light-smoker(less than 10/), moderate smoker(10- 19), heavy smoker(20 or over)
    surv: int,  # 10-year risk
    town: float,  # Townsend deprivation score
) -> float:
    survivor = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.988876402378082, 0, 0, 0, 0, 0]

    # /* The conditional arrays */

    Iethrisk = [
        0,
        0,
        0.2804031433299542500000000,
        0.5629899414207539800000000,
        0.2959000085111651600000000,
        0.0727853798779825450000000,
        -0.1707213550885731700000000,
        -0.3937104331487497100000000,
        -0.3263249528353027200000000,
        -0.1712705688324178400000000,
    ]
    Ismoke = [
        0,
        0.1338683378654626200000000,
        0.5620085801243853700000000,
        0.6674959337750254700000000,
        0.8494817764483084700000000,
    ]

    # /* Applying the fractional polynomial transforms */
    # /* (which includes scaling)                      */

    dage = age
    dage = dage / 10
    age_1 = pow(dage, -2)
    age_2 = dage
    dbmi = bmi
    dbmi = dbmi / 10
    bmi_1 = pow(dbmi, -2)
    bmi_2 = pow(dbmi, -2) * np.log(dbmi)

    # /* Centring the continuous variables */

    age_1 = age_1 - 0.053274843841791
    age_2 = age_2 - 4.332503318786621
    bmi_1 = bmi_1 - 0.154946178197861
    bmi_2 = bmi_2 - 0.144462317228317
    rati = rati - 3.476326465606690
    sbp = sbp - 123.130012512207030
    sbps5 = sbps5 - 9.002537727355957
    town = town - 0.392308831214905

    # /* Start of Sum */
    a = 0.0

    # /* The conditional sums */

    a += Iethrisk[ethrisk]
    a += Ismoke[smoke_cat]

    # /* Sum from continuous values */

    a += age_1 * -8.1388109247726188000000000
    a += age_2 * 0.7973337668969909800000000
    a += bmi_1 * 0.2923609227546005200000000
    a += bmi_2 * -4.1513300213837665000000000
    a += rati * 0.1533803582080255400000000
    a += sbp * 0.0131314884071034240000000
    a += sbps5 * 0.0078894541014586095000000
    a += town * 0.0772237905885901080000000

    # /* Sum from boolean values */

    a += b_AF * 1.5923354969269663000000000
    a += b_atypicalantipsy * 0.2523764207011555700000000
    a += b_corticosteroids * 0.5952072530460185100000000
    a += b_migraine * 0.3012672608703450000000000
    a += b_ra * 0.2136480343518194200000000
    a += b_renal * 0.6519456949384583300000000
    a += b_semi * 0.1255530805882017800000000
    a += b_sle * 0.7588093865426769300000000
    a += b_treatedhyp * 0.5093159368342300400000000
    a += b_type1 * 1.7267977510537347000000000
    a += b_type2 * 1.0688773244615468000000000
    a += fh_cvd * 0.4544531902089621300000000

    # /* Sum from interaction terms */

    a += age_1 * (smoke_cat == 1) * -4.7057161785851891000000000
    a += age_1 * (smoke_cat == 2) * -2.7430383403573337000000000
    a += age_1 * (smoke_cat == 3) * -0.8660808882939218200000000
    a += age_1 * (smoke_cat == 4) * 0.9024156236971064800000000
    a += age_1 * b_AF * 19.9380348895465610000000000
    a += age_1 * b_corticosteroids * -0.9840804523593628100000000
    a += age_1 * b_migraine * 1.7634979587872999000000000
    a += age_1 * b_renal * -3.5874047731694114000000000
    a += age_1 * b_sle * 19.6903037386382920000000000
    a += age_1 * b_treatedhyp * 11.8728097339218120000000000
    a += age_1 * b_type1 * -1.2444332714320747000000000
    a += age_1 * b_type2 * 6.8652342000009599000000000
    a += age_1 * bmi_1 * 23.8026234121417420000000000
    a += age_1 * bmi_2 * -71.1849476920870070000000000
    a += age_1 * fh_cvd * 0.9946780794043512700000000
    a += age_1 * sbp * 0.0341318423386154850000000
    a += age_1 * town * -1.0301180802035639000000000
    a += age_2 * (smoke_cat == 1) * -0.0755892446431930260000000
    a += age_2 * (smoke_cat == 2) * -0.1195119287486707400000000
    a += age_2 * (smoke_cat == 3) * -0.1036630639757192300000000
    a += age_2 * (smoke_cat == 4) * -0.1399185359171838900000000
    a += age_2 * b_AF * -0.0761826510111625050000000
    a += age_2 * b_corticosteroids * -0.1200536494674247200000000
    a += age_2 * b_migraine * -0.0655869178986998590000000
    a += age_2 * b_renal * -0.2268887308644250700000000
    a += age_2 * b_sle * 0.0773479496790162730000000
    a += age_2 * b_treatedhyp * 0.0009685782358817443600000
    a += age_2 * b_type1 * -0.2872406462448894900000000
    a += age_2 * b_type2 * -0.0971122525906954890000000
    a += age_2 * bmi_1 * 0.5236995893366442900000000
    a += age_2 * bmi_2 * 0.0457441901223237590000000
    a += age_2 * fh_cvd * -0.0768850516984230380000000
    a += age_2 * sbp * -0.0015082501423272358000000
    a += age_2 * town * -0.0315934146749623290000000

    # /* Calculate the score itself */
    score = 100.0 * (1 - survivor[surv] ** np.exp(a))
    return score


# Male
# /*
# * Copyright 2017 ClinRisk Ltd.
# *
# * This file is part of QRISK3-2017 (https://qrisk.org).
# *
# * QRISK3-2017 is free software: you can redistribute it and/or modify
# * it under the terms of the GNU Lesser General Public License as published by
# * the Free Software Foundation, either version 3 of the License, or
# * (at your option) any later version.
# *
# * QRISK3-2017 is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU Lesser General Public License for more details.
# *
# * You should have received a copy of the GNU Lesser General Public License
# * along with QRISK3-2017.  If not, see http://www.gnu.org/licenses/.
# *
# * Additional terms
# *
# * The following disclaimer must be held together with any risk score score generated by this code.
# * If the score is displayed, then this disclaimer must be displayed or otherwise be made easily accessible, e.g. by a prominent link alongside it.
# *   The initial version of this file, to be found at http://svn.clinrisk.co.uk/opensource/qrisk2, faithfully implements QRISK3-2017.
# *   ClinRisk Ltd. have released this code under the GNU Lesser General Public License to enable others to implement the algorithm faithfully.
# *   However, the nature of the GNU Lesser General Public License is such that we cannot prevent, for example, someone accidentally
# *   altering the coefficients, getting the inputs wrong, or just poor programming.
# *   ClinRisk Ltd. stress, therefore, that it is the responsibility of the end user to check that the source that they receive produces the same
# *   results as the original code found at https://qrisk.org.
# *   Inaccurate implementations of risk scores can lead to wrong patients being given the wrong treatment.
# *
# * End of additional terms
# *
# */


def cvd_male_raw(
    age: float,  # age value
    b_AF: bool,  # bool, Atrial fibrillation
    b_atypicalantipsy: bool,  # bool, On atypical antipsychotic medication
    b_corticosteroids: bool,  # Are you on regular steroid tablets?
    b_impotence2: bool,  # A diagnosis of or treatment for erectile disfunction?
    b_migraine: bool,  # bool, Do you have migraines?
    b_ra: bool,  # Rheumatoid arthritis?
    b_renal: bool,  # Chronic kidney disease (stage 3, 4 or 5)?
    b_semi: bool,  # Severe mental illness?
    b_sle: bool,  # Systemic lupus erythematosus
    b_treatedhyp: bool,  # On blood pressure treatment?
    b_type1: bool,  # Diabetes status: type 1
    b_type2: bool,  # Diabetes status: type 2
    bmi: float,  # Body mass index = kg/m^2
    ethrisk: int,  # ethnic risk
    fh_cvd: int,  # Angina or heart attack in a 1st degree relative < 60?
    rati: float,  # Cholesterol/HDL ratio
    sbp: float,  # Systolic blood pressure
    sbps5: float,  # Standard deviation of at least two most recent systolic blood pressure readings (mmHg)
    smoke_cat: int,  # smoking category: non-smoker, ex-smoker, light-smoker(less than 10/), moderate smoker(10-      19), heavy smoker(20 or over)
    surv: int,  # 10-year risk
    town: float,  # Townsend deprivation score
) -> float:
    survivor = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.977268040180206, 0, 0, 0, 0, 0]

    # /* The conditional arrays */

    Iethrisk = [
        0,
        0,
        0.2771924876030827900000000,
        0.4744636071493126800000000,
        0.5296172991968937100000000,
        0.0351001591862990170000000,
        -0.3580789966932791900000000,
        -0.4005648523216514000000000,
        -0.4152279288983017300000000,
        -0.2632134813474996700000000,
    ]
    Ismoke = [
        0,
        0.1912822286338898300000000,
        0.5524158819264555200000000,
        0.6383505302750607200000000,
        0.7898381988185801900000000,
    ]

    # /* Applying the fractional polynomial transforms */
    # /* (which includes scaling)                      */

    dage = age
    dage = dage / 10
    age_1 = pow(dage, -1)
    age_2 = pow(dage, 3)
    dbmi = bmi
    dbmi = dbmi / 10
    bmi_2 = pow(dbmi, -2) * np.log(dbmi)
    bmi_1 = pow(dbmi, -2)

    # /* Centring the continuous variables */

    age_1 = age_1 - 0.234766781330109
    age_2 = age_2 - 77.284080505371094
    bmi_1 = bmi_1 - 0.149176135659218
    bmi_2 = bmi_2 - 0.141913309693336
    rati = rati - 4.300998687744141
    sbp = sbp - 128.571578979492190
    sbps5 = sbps5 - 8.756621360778809
    town = town - 0.526304900646210

    # /* Start of Sum */
    a = 0.0

    # /* The conditional sums */

    a += Iethrisk[ethrisk]
    a += Ismoke[smoke_cat]

    # /* Sum from continuous values */

    a += age_1 * -17.8397816660055750000000000
    a += age_2 * 0.0022964880605765492000000
    a += bmi_1 * 2.4562776660536358000000000
    a += bmi_2 * -8.3011122314711354000000000
    a += rati * 0.1734019685632711100000000
    a += sbp * 0.0129101265425533050000000
    a += sbps5 * 0.0102519142912904560000000
    a += town * 0.0332682012772872950000000

    # /* Sum from boolean values */

    a += b_AF * 0.8820923692805465700000000
    a += b_atypicalantipsy * 0.1304687985517351300000000
    a += b_corticosteroids * 0.4548539975044554300000000
    a += b_impotence2 * 0.2225185908670538300000000
    a += b_migraine * 0.2558417807415991300000000
    a += b_ra * 0.2097065801395656700000000
    a += b_renal * 0.7185326128827438400000000
    a += b_semi * 0.1213303988204716400000000
    a += b_sle * 0.4401572174457522000000000
    a += b_treatedhyp * 0.5165987108269547400000000
    a += b_type1 * 1.2343425521675175000000000
    a += b_type2 * 0.8594207143093222100000000
    a += fh_cvd * 0.5405546900939015600000000

    # /* Sum from interaction terms */

    a += age_1 * (smoke_cat == 1) * -0.2101113393351634600000000
    a += age_1 * (smoke_cat == 2) * 0.7526867644750319100000000
    a += age_1 * (smoke_cat == 3) * 0.9931588755640579100000000
    a += age_1 * (smoke_cat == 4) * 2.1331163414389076000000000
    a += age_1 * b_AF * 3.4896675530623207000000000
    a += age_1 * b_corticosteroids * 1.1708133653489108000000000
    a += age_1 * b_impotence2 * -1.5064009857454310000000000
    a += age_1 * b_migraine * 2.3491159871402441000000000
    a += age_1 * b_renal * -0.5065671632722369400000000
    a += age_1 * b_treatedhyp * 6.5114581098532671000000000
    a += age_1 * b_type1 * 5.3379864878006531000000000
    a += age_1 * b_type2 * 3.6461817406221311000000000
    a += age_1 * bmi_1 * 31.0049529560338860000000000
    a += age_1 * bmi_2 * -111.2915718439164300000000000
    a += age_1 * fh_cvd * 2.7808628508531887000000000
    a += age_1 * sbp * 0.0188585244698658530000000
    a += age_1 * town * -0.1007554870063731000000000
    a += age_2 * (smoke_cat == 1) * -0.0004985487027532612100000
    a += age_2 * (smoke_cat == 2) * -0.0007987563331738541400000
    a += age_2 * (smoke_cat == 3) * -0.0008370618426625129600000
    a += age_2 * (smoke_cat == 4) * -0.0007840031915563728900000
    a += age_2 * b_AF * -0.0003499560834063604900000
    a += age_2 * b_corticosteroids * -0.0002496045095297166000000
    a += age_2 * b_impotence2 * -0.0011058218441227373000000
    a += age_2 * b_migraine * 0.0001989644604147863100000
    a += age_2 * b_renal * -0.0018325930166498813000000
    a += age_2 * b_treatedhyp * 0.0006383805310416501300000
    a += age_2 * b_type1 * 0.0006409780808752897000000
    a += age_2 * b_type2 * -0.0002469569558886831500000
    a += age_2 * bmi_1 * 0.0050380102356322029000000
    a += age_2 * bmi_2 * -0.0130744830025243190000000
    a += age_2 * fh_cvd * -0.0002479180990739603700000
    a += age_2 * sbp * -0.0000127187419158845700000
    a += age_2 * town * -0.0000932996423232728880000

    # /* Calculate the score itself */
    score = 100.0 * (1 - survivor[surv] ** np.exp(a))
    return score


def inference(
    gender: str,  # M/F
    age: float,  # age value
    b_AF: bool,  # bool, Atrial fibrillation
    b_atypicalantipsy: bool,  # bool, On atypical antipsychotic medication
    b_corticosteroids: bool,  # Are you on regular steroid tablets?
    b_impotence2: bool,  # A diagnosis of or treatment for erectile disfunction?
    b_migraine: bool,  # bool, Do you have migraines?
    b_ra: bool,  # Rheumatoid arthritis?
    b_renal: bool,  # Chronic kidney disease (stage 3, 4 or 5)?
    b_semi: bool,  # Severe mental illness?
    b_sle: bool,  # Systemic lupus erythematosus
    b_treatedhyp: bool,  # On blood pressure treatment?
    b_type1: bool,  # Diabetes status: type 1
    b_type2: bool,  # Diabetes status: type 2
    bmi: float,  # Body mass index = kg/m^2
    ethrisk: int,  # ethnic risk
    fh_cvd: int,  # Angina or heart attack in a 1st degree relative < 60?
    rati: float,  # Cholesterol/HDL ratio
    sbp: float,  # Systolic blood pressure
    sbps5: float,  # Standard deviation of at least two most recent systolic blood pressure readings (mmHg)
    smoke_cat: int,  # smoking category: non-smoker, ex-smoker, light-smoker(less than 10/), moderate smoker(10-      19), heavy smoker(20 or over)
    town: float,  # Townsend deprivation score
    surv: int = 10,  # 10-year risk
) -> float:

    if gender == "M":
        pct = cvd_male_raw(
            age=age,
            b_AF=b_AF,
            b_atypicalantipsy=b_atypicalantipsy,
            b_corticosteroids=b_corticosteroids,
            b_impotence2=b_impotence2,
            b_migraine=b_migraine,
            b_ra=b_ra,
            b_renal=b_renal,
            b_semi=b_semi,
            b_sle=b_sle,
            b_treatedhyp=b_treatedhyp,
            b_type1=b_type1,
            b_type2=b_type2,
            bmi=bmi,
            ethrisk=ethrisk,
            fh_cvd=fh_cvd,
            rati=rati,
            sbp=sbp,
            sbps5=sbps5,
            smoke_cat=smoke_cat,
            surv=surv,
            town=town,
        )
    else:
        pct = cvd_female_raw(
            age=age,
            b_AF=b_AF,
            b_atypicalantipsy=b_atypicalantipsy,
            b_corticosteroids=b_corticosteroids,
            b_migraine=b_migraine,
            b_ra=b_ra,
            b_renal=b_renal,
            b_semi=b_semi,
            b_sle=b_sle,
            b_treatedhyp=b_treatedhyp,
            b_type1=b_type1,
            b_type2=b_type2,
            bmi=bmi,
            ethrisk=ethrisk,
            fh_cvd=fh_cvd,
            rati=rati,
            sbp=sbp,
            sbps5=sbps5,
            smoke_cat=smoke_cat,
            surv=surv,
            town=town,
        )

    return pct / 100.0


def mmolL_to_mgdl(val: float) -> float:
    return val * 18.0182


class QRisk3Model:
    def __init__(self) -> None:
        pass

    def fit(self, *args: Any, **kwargs: Any) -> "QRisk3Model":
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
        def qrisk3_inference(row: pd.DataFrame) -> Any:
            score = inference(
                gender=row["sex"],  # M/F
                age=row["age"],  # age value
                b_AF=row["b_atrial_fibr"],  # bool, Atrial fibrillation
                b_atypicalantipsy=row[
                    "b_antipsychotic_use"
                ],  # bool, On atypical antipsychotic medication
                b_corticosteroids=row[
                    "b_steroid_treat"
                ],  # Are you on regular steroid tablets?
                b_impotence2=row[
                    "b_erectile_disf"
                ],  # A diagnosis of or treatment for erectile disfunction?
                b_migraine=row["b_had_migraine"],  # bool, Do you have migraines?
                b_ra=row["b_rheumatoid_arthritis"],  # Rheumatoid arthritis?
                b_renal=row["b_renal"],  # Chronic kidney disease (stage 3, 4 or 5)?
                b_semi=row["b_mental_illness"],  # Severe mental illness?
                b_sle=row["b_sle"],  # Systemic lupus erythematosus
                b_treatedhyp=row["ht_treat"],  # On blood pressure treatment?
                b_type1=row["b_diab_type1"],  # Diabetes status: type 1
                b_type2=row["b_diab_type2"],  # Diabetes status: type 2
                bmi=row["bmi"],  # Body mass index = kg/m^2
                ethrisk=row["ethrisk"],  # ethnic risk
                fh_cvd=row[
                    "family_cvd"
                ],  # Angina or heart attack in a 1st degree relative < 60?
                rati=row["chol_ratio"],  # Cholesterol/HDL ratio
                sbp=row["sbp"],  # Systolic blood pressure
                sbps5=row[
                    "sbps5"
                ],  # Standard deviation of at least two most recent systolic blood pressure readings (mmHg)
                smoke_cat=row[
                    "smoker"
                ],  # smoking category: non-smoker, ex-smoker, light-smoker(less than 10/), moderate smoker(10-      19), heavy smoker(20 or over)
                town=row["town_depr_index"],  # Townsend deprivation score
            )
            return score

        expected_cols = [
            "sex",
            "age",
            "b_atrial_fibr",
            "b_antipsychotic_use",
            "b_steroid_treat",
            "b_erectile_disf",
            "b_had_migraine",
            "b_rheumatoid_arthritis",
            "b_renal",
            "b_mental_illness",
            "b_sle",
            "ht_treat",
            "b_diab_type1",
            "b_diab_type2",
            "bmi",
            "ethrisk",
            "family_cvd",
            "sbps5",
            "chol_ratio",
            "sbp",
            "smoker",
        ]
        df = df.copy()
        for col in expected_cols:
            if col not in df.columns:
                log.error(f"[QRisk3] missing {col}")
                df[col] = 0

        scores = df.apply(lambda row: qrisk3_inference(row), axis=1)

        return pd.DataFrame(scores, columns=[10 * 365])
