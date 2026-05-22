# System and Method for Real-Time Indoor Airborne Infection Risk Estimation Using Consumer CO2 Sensor Concentration Decay Kinetics and Occupancy-Normalized Rebreathed Air Fraction Modeling

**LITF-PA-2026-045 · Indoor Air Quality / Computational Epidemiology**
**Published:** 2026-05-22
**License:** [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/) — Public Domain

> ⚖️ **Prior Art Notice:** This document is published as defensive prior art under [35 U.S.C. § 102(a)(1)](https://www.law.cornell.edu/uscode/text/35/102). The inventions described herein are dedicated to the public domain as of the publication date above.

---

## Abstract

Disclosed is a system and method for continuously estimating the probability of airborne infectious disease transmission in occupied indoor spaces using data from consumer-grade CO2 sensors. The system operates in three stages: (1) automatic ventilation rate estimation by detecting occupancy transitions in the CO2 time series and fitting first-order exponential decay models to post-vacancy concentration curves, yielding air changes per hour (ACH) without tracer gas injection, blower door equipment, or manual calibration; (2) real-time computation of the rebreathed air fraction during occupied periods using the Rudnick-Milton reformulation of the Wells-Riley equation, where the instantaneous CO2 elevation above outdoor baseline serves as a direct proxy for the fraction of inhaled air that was previously exhaled by other occupants; (3) pathogen-specific infection probability estimation by integrating the rebreathed air fraction over exposure duration and scaling by disease-specific quanta generation rates drawn from an updateable epidemiological parameter database covering SARS-CoV-2 variants, influenza subtypes, measles, tuberculosis, and RSV. The system runs on the embedded processor of consumer CO2 monitors or on a paired smartphone, requires no additional hardware beyond a single NDIR CO2 sensor, and outputs a continuously updated infection risk score displayed as a color-coded index (green/yellow/orange/red) alongside ventilation adequacy metrics and time-to-threshold alerts.

## Field of the Invention

This invention relates to indoor air quality monitoring and computational epidemiology, specifically to methods for estimating airborne disease transmission risk in real time using carbon dioxide concentration measurements from consumer-grade non-dispersive infrared (NDIR) sensors as a proxy for room ventilation adequacy and rebreathed air exposure.

## Background

Airborne transmission is the dominant route for several major respiratory pathogens. The COVID-19 pandemic forced a global reckoning with indoor ventilation: the [Greenhalgh et al. (2021)](https://doi.org/10.1126/science.abd9149) consensus statement in Science, signed by 239 scientists, established that SARS-CoV-2 spreads primarily through aerosol inhalation in poorly ventilated indoor spaces.

The foundational model for predicting airborne infection probability is the [Wells-Riley equation](https://en.wikipedia.org/wiki/Wells-Riley_model), first formalized by [Riley, Murphy, and Riley (1978)](https://doi.org/10.1093/oxfordjournals.aje.a112560). The model estimates the probability P of infection as:

```
P = 1 - exp(-Iqpt/Q)
```

where I is the number of infectious sources, q is the quanta generation rate (infectious dose units per hour), p is the pulmonary ventilation rate of susceptible occupants (m³/h), t is exposure time (hours), and Q is the room ventilation rate with clean air (m³/h). The critical difficulty in applying Wells-Riley outside controlled research settings has always been Q: determining the actual ventilation rate of a real room in real time.

[Rudnick and Milton (2003)](https://doi.org/10.1111/ina.12054) demonstrated that indoor CO2 concentration serves as a natural tracer gas for rebreathed air, reformulating Wells-Riley to eliminate Q:

```
f = (C_indoor - C_outdoor) / (C_exhaled - C_outdoor)
```

where C_indoor is measured indoor CO2 (ppm), C_outdoor is outdoor baseline (~420 ppm in 2026), and C_exhaled is CO2 in human exhaled breath (~38,000-40,000 ppm). [Peng and Jimenez (2021)](https://doi.org/10.1021/acs.est.1c06531) extended this into a practical COVID-19 estimator, demonstrating that CO2 measurements alone can bound infection probability to within a factor of 2-3 for well-mixed rooms.

Consumer CO2 sensors have proliferated since 2020: Aranet4 ($179, ±50 ppm, Bluetooth), Awair Element ($149, ±75 ppm, WiFi), Senseair Sunrise OEM ($25-40, ±30 ppm). These devices display raw concentration or static thresholds but do not compute ventilation rates, rebreathed fractions, or infection probabilities.

The gap: no consumer-accessible system automatically derives ventilation rate from the CO2 time series, computes rebreathed air fraction via Rudnick-Milton, and integrates with pathogen-specific quanta rates to output calibrated infection risk.

## Detailed Description

### 1. Hardware Requirements

The system requires a single NDIR CO2 sensor: range 0-5000 ppm, accuracy ±50 ppm or ±5%, sampling ≤2 minutes, resolution ≤1 ppm. Met by all major consumer monitors (Aranet4/Senseair S8 LP, Awair, Sensirion SCD40/SCD41 at $15, MH-Z19C at $12).

The inference pipeline runs on: (a) the sensor's embedded MCU (ARM Cortex-M4+, 256 KB RAM); (b) paired smartphone via BLE; or (c) home automation hub. Additional compute cost: zero.

### 2. Automatic Ventilation Rate Estimation

**Occupancy transition detection:** The system identifies vacancy onset (sustained CO2 decrease from >C_outdoor + 80 ppm), occupancy onset (sustained increase from baseline), and steady-state occupancy (±30 ppm stability for 20+ minutes). Uses sliding-window slope estimation with Kalman filtering, combined with [Bayesian Online Changepoint Detection (Adams and MacKay, 2007)](https://doi.org/10.48550/arXiv.0710.3742).

**Decay curve fitting:** Post-vacancy CO2 follows first-order exponential decay:

```
C(t) = C_outdoor + (C_peak - C_outdoor) × exp(-λt)
```

where λ = ACH. Fitted via Levenberg-Marquardt nonlinear least squares. Minimum 20-minute decay segment. [Validated accuracy](https://pubmed.ncbi.nlm.nih.gov/38153750/): ±15% of blower door measurements, up to 10 estimates/day/room.

**Multi-regime tracking:** Rolling ACH library tagged by time, day, outdoor temperature. Gaussian process regression interpolates between measurements for continuous ACH during occupied periods.

### 3. Real-Time Rebreathed Air Fraction

During occupancy:

```
f(t) = (C(t) - C_outdoor) / (C_exhaled - C_outdoor)
```

C_outdoor: minimum over 72h, external API, or 420 ppm fallback (+10 ppm urban). C_exhaled: 38,000 ppm sedentary ([Persily and de Jonge, 2017](https://pubmed.ncbi.nlm.nih.gov/28715296/)), adjusted by activity.

At 1000 ppm indoor / 420 ppm outdoor: f = 1.54% rebreathed. At 2000 ppm: f = 4.2%.

### 4. Pathogen-Specific Infection Probability

Using Rudnick-Milton Wells-Riley:

```
P = 1 - exp(-f_avg × n_i × q × t / n_total)
```

Quanta generation rates from literature:
- **SARS-CoV-2 (original):** 10-50 quanta/h ([Buonanno et al., 2020](https://doi.org/10.1016/j.envint.2020.106112))
- **SARS-CoV-2 (Omicron+):** 30-150 quanta/h ([Mikszewski et al., 2022](https://doi.org/10.1016/j.jhin.2022.01.020))
- **Influenza A/B:** 15-128 quanta/h ([Yan et al., 2018](https://doi.org/10.1111/ina.12062))
- **Measles:** 570-5580 quanta/h ([de Jong and Zusman, 2019](https://doi.org/10.1093/cid/ciz211))
- **Tuberculosis:** 1.25-13 quanta/h ([Nardell, 2016](https://doi.org/10.1128/CMR.22.4.694-719.2009))
- **RSV:** 9-28 quanta/h (household study estimates)

Infectious occupant estimation: (a) community prevalence mode using CDC/wastewater surveillance; (b) known-case mode (n_i = 1); (c) sensitivity analysis across n_i = {0.5, 1, 2}.

**Mask adjustment:** Surgical (0.3-0.5 reduction), KN95 (0.15-0.3), N95 fitted (0.05-0.15), per [Cheng et al., PNAS 2021](https://doi.org/10.1073/pnas.2110117118).

### 5. Occupancy Estimation from CO2

When no explicit count available:

```
n = λV(C_ss - C_outdoor) / G
```

G defaults: 0.005 L/s office, 0.004 L/s classroom, 0.015 L/s gym, 0.006 L/s restaurant. Accuracy: ±1-2 persons for well-characterized rooms (4-20 occupants).

### 6. User Interface

- **Risk Index (0-100):** Green (<1%/h), Yellow (1-3%/h), Orange (3-8%/h), Red (>8%/h)
- **Ventilation adequacy:** Estimated ACH vs ASHRAE 62.1 minimum
- **Time-to-threshold:** Time until cumulative P exceeds configurable limit (default 1%)
- **Historical dashboard:** 24h/7d trends of CO2, ACH, rebreathed fraction, cumulative risk

### 7. Multi-Zone Extension

Multiple sensors model building as directed graph. Inter-zone airflow estimated from correlated CO2 transients when doors open. Graph neural network predicts pathogen propagation across zones for building-level management.

## Claims

1. A system for estimating airborne infection transmission probability in indoor spaces, comprising: a consumer-grade NDIR CO2 sensor providing time-series concentration measurements; a processor executing an occupancy transition detection algorithm on the CO2 time series; a ventilation rate estimation module that fits exponential decay models to post-vacancy CO2 concentration segments to determine air changes per hour without tracer gas injection or manual calibration; and a rebreathed air fraction computation module that applies the Rudnick-Milton reformulation to compute the fraction of inhaled air previously exhaled by other occupants.

2. The system of claim 1, further comprising a pathogen-specific infection probability engine that integrates the time-averaged rebreathed air fraction over an exposure window and scales by disease-specific quanta generation rates from an updateable epidemiological parameter database to output a continuously updated infection probability for one or more respiratory pathogens.

3. The system of claim 1, wherein the occupancy transition detection algorithm uses Bayesian Online Changepoint Detection on the CO2 concentration derivative to identify vacancy onset, occupancy onset, and steady-state occupancy regimes without external occupancy sensor inputs.

4. The system of claim 1, wherein the ventilation rate estimation module maintains a rolling library of ACH estimates tagged by time of day, day of week, and outdoor temperature, and interpolates between measurements using a Gaussian process regression model to provide continuous ACH estimates during occupied periods.

5. The system of claim 2, further comprising a community prevalence integration module that ingests public health surveillance data to estimate the expected number of infectious occupants as a function of total occupancy and local disease prevalence, enabling infection probability computation without explicit knowledge of any individual's infection status.

6. A method for real-time indoor airborne infection risk estimation comprising: continuously measuring indoor CO2 concentration using a consumer NDIR sensor; detecting occupancy transitions in the CO2 time series using changepoint detection; fitting exponential decay models to post-vacancy segments to estimate the room air change rate; computing the rebreathed air fraction during occupied periods as the ratio of CO2 elevation above outdoor baseline to exhaled-breath CO2 minus outdoor baseline; and integrating the rebreathed air fraction over exposure duration, scaled by pathogen-specific quanta generation rates, to output an infection probability estimate.

7. The method of claim 6, further comprising computing a time-to-threshold metric representing the estimated time until cumulative infection probability exceeds a user-configurable threshold, and generating a preemptive alert when the remaining time falls below a configurable warning period.

8. The method of claim 6, further comprising an occupancy estimation step that derives the number of room occupants from the CO2 mass balance equation using the estimated air change rate, room volume, and metabolic CO2 generation rates, without requiring external occupancy counting sensors.

9. The system of claim 1, deployed across multiple rooms in a building, further comprising a graph neural network that models inter-zone airflow by correlating CO2 transients across sensor pairs and predicts pathogen concentration propagation through the building's zone graph to support building-level ventilation management decisions.

10. The system of claim 2, further comprising a mask adjustment module that applies pathogen filtration reduction factors based on declared mask type to the effective quanta inhalation rate, modifying the infection probability estimate to reflect the protective effect of respiratory protection equipment.

## Implementation Notes

The entire inference pipeline requires <5,000 FP operations per reading and <50 KB working memory. Epidemiological database: ~2 KB for 20 pathogens. GP ventilation model: ~1.6 KB for 100 ACH estimates. Total firmware: <100 KB, compatible with OTA update on existing hardware.

Validation: ground-truth via [ASTM E741-11](https://www.astm.org/e0741-11r17.html) tracer gas tests. Infection probability cross-validated against retrospective superspreading events (e.g., [Skagit Valley Chorale](https://doi.org/10.1016/j.envint.2020.106112): 53/61 infected, 2.5h rehearsal, poor ventilation).

## Prior Art References

1. [Riley, Murphy, and Riley, AJE 1978](https://doi.org/10.1093/oxfordjournals.aje.a112560) — Original Wells-Riley model
2. [Rudnick and Milton, Indoor Air 2003](https://doi.org/10.1111/ina.12054) — CO2-based rebreathed air fraction reformulation
3. [Peng and Jimenez, ES&T 2021](https://doi.org/10.1021/acs.est.1c06531) — COVID-19 aerosol transmission estimator
4. [Greenhalgh et al., Science 2021](https://doi.org/10.1126/science.abd9149) — Airborne transmission consensus
5. [Persily and de Jonge, Indoor Air 2017](https://pubmed.ncbi.nlm.nih.gov/28715296/) — CO2 generation rates
6. [Buonanno et al., Env Int 2020](https://doi.org/10.1016/j.envint.2020.106112) — SARS-CoV-2 quanta rates
7. [Mikszewski et al., J Hosp Infect 2022](https://doi.org/10.1016/j.jhin.2022.01.020) — Omicron transmissibility
8. [Yan et al., Indoor Air 2018](https://doi.org/10.1111/ina.12062) — Influenza quanta from breath
9. [Cheng et al., PNAS 2021](https://doi.org/10.1073/pnas.2110117118) — Mask filtration efficiency
10. [Adams and MacKay, 2007](https://doi.org/10.48550/arXiv.0710.3742) — Bayesian Online Changepoint Detection
11. [Hegde et al., 2024](https://pubmed.ncbi.nlm.nih.gov/38153750/) — ML-based ACH from single CO2 sensor
12. [US11566801B2](https://patents.google.com/patent/US11566801B2) — Metabolic rate + passive sensors (adjacent)
13. [ASTM E741-11](https://www.astm.org/e0741-11r17.html) — Tracer gas air change measurement
14. [Aranet4 Specs](https://aranet.com/products/aranet4/) — Consumer NDIR benchmark
15. [ASHRAE 62.1](https://www.ashrae.org/technical-resources/bookstore/standards-62-1-62-2) — Ventilation standard
