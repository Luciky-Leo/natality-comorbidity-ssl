# SINASC to U.S. Natality variable mapping feasibility

## Data parsed

- SINASC 2023: entry `SINASC_2023.csv`, delimiter `;`, 62 columns.
- SINASC 2024: entry `SINASC_2024.csv`, delimiter `;`, 62 columns.
- SINASC 2023: 2,537,576 rows scanned for mapped candidate-variable missingness/examples.
- SINASC 2024: 2,389,325 rows scanned for mapped candidate-variable missingness/examples.
- SINASC header union across years: 62 variables.
- Official PDF dictionary context extracted for 60 of 62 header variables.
- Candidate mapped SINASC variables profiled for missingness/examples: 32.

## Feasibility summary

- derived: 3.
- direct: 4.
- direct-derived: 1.
- direct_or_partial: 8.
- not_available: 10.
- partial: 5.

The SINASC public files can support an independent registry stress-test for demographics, prenatal care, parity, plurality, delivery mode, gestational age, birthweight, Apgar, and congenital anomaly markers. They cannot directly reproduce the current U.S. Natality SSL input space because major feature families are absent, especially maternal BMI/weight gain, smoking, diabetes, hypertensive disorders, infections, infertility/ART, WIC/payment, and the current maternal morbidity endpoint.

## Mappable input families

- Maternal age: `IDADEMAE` (direct).
- Maternal race/ethnicity: `RACACORMAE; RACACOR` (partial).
- Marital status: `ESTCIVMAE` (partial).
- Maternal education: `ESCMAE; ESCMAE2010; ESCMAEAGR1; SERIESCMAE` (partial).
- Maternal residence geography: `CODMUNRES; CODPAISRES` (partial).
- Live-birth order / parity: `QTDFILVIVO; QTDGESTANT; PARIDADE; QTDPARTNOR; QTDPARTCES` (partial).
- Previous cesarean: `QTDPARTCES` (direct-derived).
- Month prenatal care began: `MESPRENAT` (direct_or_partial).
- Number of prenatal visits: `CONSULTAS; CONSPRENAT` (direct_or_partial).
- Plurality: `GRAVIDEZ` (direct_or_partial).
- Infant sex: `SEXO` (direct).

## Major missing current-model families

- Previous preterm birth: U.S. variables `RF_PPTERM`.
- WIC receipt / payment source: U.S. variables `WIC; PAY_REC`.
- Smoking before/during pregnancy: U.S. variables `CIG_0; CIG_1; CIG_2; CIG_3; CIG_REC`.
- Maternal height/BMI/prepregnancy weight/weight gain: U.S. variables `M_Ht_In; BMI; BMI_R; PWgt_R; WTGAIN; WTGAIN_REC`.
- Pregestational or gestational diabetes: U.S. variables `RF_PDIAB; RF_GDIAB`.
- Pregestational hypertension, gestational hypertension, eclampsia: U.S. variables `RF_PHYPE; RF_GHYPE; RF_EHYPE`.
- Infertility treatment / fertility drugs / ART: U.S. variables `RF_INFTR; RF_FEDRG; RF_ARTEC`.
- Maternal infections: U.S. variables `IP_GON; IP_SYPH; IP_CHLAM; IP_HEPB; IP_HEPC`.
- NICU admission, ventilation, seizures: U.S. variables `AB_NICU; AB_VENT; AB_VENT6; AB_SEIZ`.
- Maternal transfusion/ICU/rupture/hysterectomy/laceration: U.S. variables `MM_MTR; MM_UHYST; MM_AICU; MM_RUPT; MR_LAC`.

## Recommended use

Use SINASC as an independent registry workflow stress-test, not as direct external validation of the trained U.S. Natality encoder. The defensible design is to build a harmonized SINASC matrix from overlapping variables, train/develop/test a separate masked tabular SSL model within SINASC years, and evaluate phenotype enrichment for preterm birth, low birthweight, very low birthweight, low 5-minute Apgar, and congenital anomaly markers.

## Output files

- Crosswalk: `E:\Reserch\AI\Natality_Comorbidity_SSL_20260525\results\tables\sinasc_natality_variable_crosswalk.csv`
- SINASC header/dictionary inventory: `E:\Reserch\AI\Natality_Comorbidity_SSL_20260525\results\tables\sinasc_header_dictionary_inventory.csv`
- Candidate variable missingness/examples: `E:\Reserch\AI\Natality_Comorbidity_SSL_20260525\results\tables\sinasc_candidate_variable_missingness_examples.csv`
- Metadata: `E:\Reserch\AI\Natality_Comorbidity_SSL_20260525\results\tables\sinasc_crosswalk_metadata.json`
