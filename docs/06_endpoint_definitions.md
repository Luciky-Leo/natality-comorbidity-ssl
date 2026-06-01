# Endpoint Definitions

Created: 2026-05-25

## Primary Maternal Endpoint

`outcome_maternal_morbidity_core`

Components:

- maternal transfusion;
- ruptured uterus;
- unplanned hysterectomy;
- maternal ICU admission.

Rationale: this endpoint captures severe maternal delivery morbidity while avoiding perineal laceration as a primary component. Perineal laceration is clinically important, but it is tightly linked to delivery mode and obstetric mechanics, so it is better handled as sensitivity/context.

## Maternal Sensitivity Endpoint

`outcome_maternal_morbidity_extended`

Components:

- maternal transfusion;
- ruptured uterus;
- unplanned hysterectomy;
- maternal ICU admission;
- perineal laceration.

Rationale: this approximates a broader birth-certificate maternal morbidity composite and helps test whether findings are driven by endpoint definition.

## Primary Severe Neonatal Endpoint

`outcome_severe_neonatal_no_nicu`

Components:

- very low birthweight, defined as birthweight <1500 g;
- 5-minute Apgar score <7;
- assisted ventilation for more than 6 hours;
- newborn seizure.

Rationale: this is more clinically severe than the broad feasibility composite and avoids making NICU admission the main driver.

## Neonatal Sensitivity Endpoint

`outcome_severe_neonatal_plus_nicu`

Components:

- very low birthweight, defined as birthweight <1500 g;
- 5-minute Apgar score <7;
- assisted ventilation for more than 6 hours;
- newborn seizure;
- NICU admission.

Rationale: NICU admission is clinically relevant but may vary by local policy, facility capacity, gestational age management and coding practice. It is therefore better as a sensitivity component rather than the primary neonatal endpoint.

## Broad Neonatal Secondary Endpoint

`outcome_broad_neonatal_composite`

Components:

- preterm birth <37 weeks;
- low birthweight <2500 g;
- 5-minute Apgar score <7;
- assisted ventilation for more than 6 hours;
- NICU admission;
- newborn seizure.

Rationale: useful for feasibility and secondary modelling, but too broad for the main claim because it mixes common and severe events.

## Leakage Boundary

Gestational age, birthweight, Apgar, ventilation, NICU admission and newborn seizure variables are outcomes or outcome components. They must not be included as inputs in the primary antepartum risk-enrichment model.

Delivery-context variables can be included only in secondary delivery-context models and must be labelled accordingly.
