# Variable Plan

This file is a working variable map. Exact variable names, positions and coding must be extracted from the official year-specific user guides before analysis.

## Core Principle

Build the primary model from variables plausibly available before the delivery outcome is known. Avoid including outcome proxies or post-outcome variables in the primary prediction/enrichment model.

## Candidate Input Families

### Maternal Demographics

- maternal age;
- race and Hispanic origin;
- education;
- marital status;
- nativity if available;
- payment source.

### Reproductive and Obstetric History

- parity or live-birth order;
- previous cesarean delivery;
- previous preterm birth;
- birth interval or pregnancy interval where available;
- plurality.

### Anthropometry and Behavior

- prepregnancy BMI;
- height;
- prepregnancy weight;
- delivery weight;
- gestational weight gain;
- smoking before pregnancy;
- smoking by trimester.

### Endocrine and Cardiometabolic Conditions

- prepregnancy diabetes;
- gestational diabetes;
- prepregnancy hypertension;
- gestational hypertension;
- eclampsia.

### Infection and Pregnancy Risk Factors

- hepatitis B;
- hepatitis C;
- chorioamnionitis;
- other infections present or treated during pregnancy if consistently available.

### Reproductive Medicine

- infertility treatment;
- fertility-enhancing drugs;
- assisted reproductive technology.

### Prenatal Care and Utilization

- prenatal care start month/trimester;
- number of prenatal visits;
- WIC receipt if used as contextual social/health-service variable.

## Primary Outcome Candidate: Neonatal Risk Composite

Candidate components, subject to file confirmation:

- preterm birth;
- low birthweight;
- very low birthweight;
- low 5-minute Apgar;
- assisted ventilation immediately after delivery or more than 6 hours;
- NICU admission;
- infant transfer;
- neonatal seizures or serious abnormal newborn conditions if available and well reported.

Important: If gestational age or birthweight are used as outcome components, do not include them as model inputs for that same primary outcome.

## Secondary Outcome Candidate: Maternal Morbidity Composite

Candidate components:

- maternal transfusion;
- ruptured uterus;
- unplanned hysterectomy;
- maternal ICU admission;
- third- or fourth-degree perineal laceration.

Important: delivery mode, trial of labor, induction, augmentation, and labor characteristics may be clinically important but can act as post-baseline or intrapartum variables. Use them in a secondary delivery-context model, not the primary antepartum model, unless the target use case is explicitly intrapartum.

## Fairness and Subgroup Variables

Candidate strata:

- maternal age group;
- race/ethnicity;
- BMI group;
- diabetes status;
- hypertension status;
- payment source;
- ART/infertility treatment;
- plurality;
- year.

## Data Quality Checks

For each study year:

- confirm variable existence and coding;
- quantify missingness and unknown categories;
- compare event rates against CDC WONDER or published aggregate rates where possible;
- flag variables with inconsistent coding across years;
- preserve missingness indicators rather than silently dropping large groups.
