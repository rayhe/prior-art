# System and Method for Continuous Building Envelope Air Leakage Estimation Using Differential Barometric Pressure Spectral Analysis from Consumer IoT Devices Under Natural Wind Loading

**LITF-PA-2026-121 · Building Science / IoT / Energy Efficiency**
**Published:** 2026-07-26
**License:** [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/) — Public Domain

> ⚖️ **Prior Art Notice:** This document is published as defensive prior art under [35 U.S.C. § 102(a)(1)](https://www.law.cornell.edu/uscode/text/35/102). The inventions described herein are dedicated to the public domain as of the publication date above.

---

## Abstract

Disclosed is a system and method for continuously estimating building envelope air leakage rates without conventional blower door testing. The system exploits the fact that natural wind loading creates measurable differential pressure across a building envelope, and that the ratio of indoor barometric pressure fluctuation to outdoor wind-induced pressure fluctuation in the 0.01–2 Hz frequency band is a function of the building's equivalent leakage area (ELA). Distributed consumer barometric pressure sensors already present in smartphones, smart speakers, smart thermostats, and weather stations sample indoor pressure at 1–10 Hz. Outdoor wind speed, direction, and barometric pressure are obtained from nearby personal weather stations or public meteorological APIs. A convolutional neural network (CNN) regression model, trained on paired blower door test results and concurrent barometric time series from 50,000+ residential buildings, estimates the building's air changes per hour at 50 Pa (ACH50) with a target mean absolute error below 1.5 ACH50. The system enables passive, zero-cost air leakage screening at population scale, prioritizing weatherization investment toward the leakiest buildings without dispatching technicians.

## Field of the Invention

This invention relates to building performance diagnostics, specifically to the passive estimation of air infiltration rates using barometric pressure measurements from consumer IoT devices and machine learning regression against wind-induced pressure differentials.

## Background

Residential building envelope air leakage accounts for 25–40% of heating and cooling energy consumption in typical U.S. homes (DOE). The standard diagnostic is the blower door test, codified in ASTM E779 and ASTM E1827, which pressurizes or depressurizes a building to 50 Pa using a calibrated fan mounted in a doorway. The test yields ACH50 (air changes per hour at 50 Pa), the primary metric for envelope tightness. A typical existing U.S. home measures 5–15 ACH50; energy codes increasingly require 3–5 ACH50 for new construction (2021 IECC).

Blower door testing has significant limitations:

- **Cost:** $300–500 per test for residential, $1,000–5,000 for commercial. Approximately 130 million U.S. housing units (EIA RECS 2020) would require $40–65 billion to test universally.
- **Access:** Requires a technician on-site for 1–2 hours. All exterior doors and windows must be closed. HVAC must be disabled. Occupant scheduling is required.
- **Episodic measurement:** A blower door test captures a single snapshot. Envelope performance degrades over time due to settling, weatherstripping failure, caulk deterioration, and thermal cycling, but retesting is rare.
- **Artificial conditions:** The 50 Pa test pressure is roughly equivalent to a 20 mph wind on all surfaces simultaneously. Natural wind loading produces directional, time-varying pressure that more closely reflects real operating conditions.

Consumer barometric pressure sensors are now ubiquitous. The Bosch BMP390 (used in smartphones and smart home devices) provides absolute pressure accuracy of ±0.5 hPa and relative accuracy of ±0.03 hPa, with a noise floor of 0.008 hPa RMS at the highest oversampling setting. At 1 Hz sampling, this sensor can resolve pressure fluctuations of approximately 0.01 Pa, well below the wind-induced indoor pressure variations of interest (0.1–5 Pa in typical residential buildings under moderate wind).

The relationship between wind-induced pressure and building air leakage has been studied extensively. Shaw and Tamura (Building and Environment, 1977) established the power-law model Q = C·(ΔP)^n relating airflow Q through the envelope to pressure differential ΔP, where C is the flow coefficient (proportional to ELA) and n is the pressure exponent (typically 0.6–0.7 for residential buildings). Orme (2001) showed that natural infiltration rates under wind loading can be estimated from blower door results using the Alberta Air Infiltration Model (AIM-2), but this requires knowing local wind exposure, terrain class, and shielding.

The gap in the art is a system that reverses this relationship: given continuous measurements of indoor barometric pressure fluctuations and outdoor wind conditions, the system infers the building's equivalent leakage area and ACH50 rating without any physical test, using machine learning to account for building geometry, shielding, terrain, and sensor placement effects that are difficult to model analytically.

## Detailed Description

### 1. Physical Basis

Wind creates pressure on a building exterior described by the wind pressure coefficient Cp: P_surface = Cp · 0.5 · ρ · V², where ρ is air density (~1.2 kg/m³) and V is wind speed at building height. Cp varies from +0.8 (windward wall) to -0.5 (leeward wall and roof) and is a function of wind angle, building geometry, and surrounding terrain.

For a building with a total equivalent leakage area ELA (in cm²), the indoor pressure responds to outdoor wind pressure changes with a time constant τ = V_building / (ELA · √(2·ΔP/ρ)), where V_building is the building volume. A typical 250 m³ home with ELA = 500 cm² (moderately leaky, ~10 ACH50) and ΔP = 2 Pa has τ ≈ 15 seconds. A tight home (ELA = 150 cm², ~3 ACH50) has τ ≈ 50 seconds. This means that in the frequency domain, the transfer function from outdoor wind pressure to indoor pressure acts as a low-pass filter whose cutoff frequency is inversely proportional to building tightness.

The key insight of this disclosure is that the spectral transfer function H(f) = S_indoor(f) / S_outdoor(f), where S denotes the power spectral density of barometric pressure fluctuations, encodes information about the building's equivalent leakage area. A leaky building transmits more high-frequency pressure fluctuations from wind gusts; a tight building attenuates them. By measuring H(f) across the 0.01–2 Hz band and fitting a parametric model or training a neural network regressor, ELA and ACH50 can be estimated.

### 2. Sensor Infrastructure

The system requires at minimum one indoor barometric pressure sensor and one outdoor wind/pressure reference. Indoor sensors are consumer devices already present in the home:

- **Smartphones:** Virtually all modern smartphones contain a barometric pressure sensor (e.g., Bosch BMP380/BMP390, STMicroelectronics LPS22HH). Sampling via mobile OS APIs at 1–10 Hz. The phone must be stationary during measurement windows (detected via accelerometer stillness).
- **Smart thermostats:** Devices such as the Ecobee SmartThermostat and Google Nest contain barometric sensors for altitude compensation. These are permanently wall-mounted, providing stable, continuous indoor measurements.
- **Smart speakers:** Some smart speakers contain barometric sensors for voice assistant altitude context. These provide fixed indoor measurement points.
- **Consumer weather stations:** Indoor base units of personal weather stations (e.g., Ambient Weather WS-2902, Davis Vantage Vue) sample barometric pressure at 1–10 second intervals with ±0.05 hPa accuracy.

Outdoor references are obtained from:

- **Personal weather stations (PWS):** Over 300,000 personal weather stations report to Weather Underground in the U.S. alone, providing hyperlocal wind speed, direction, gusts, and barometric pressure. Typical PWS density in suburban areas: 2–5 stations per km².
- **Outdoor sensors belonging to the same home:** Consumer weather stations often include an outdoor unit with anemometer and barometer.
- **National Weather Service ASOS/AWOS:** Airport weather stations provide high-quality 1-minute wind and pressure data, though at lower spatial density.

### 3. Data Acquisition and Preprocessing

Indoor barometric pressure is sampled at a minimum of 1 Hz (10 Hz preferred for smartphones during active measurement sessions). Each sample is timestamped with NTP-synchronized device time (typical accuracy ±50 ms). The raw pressure time series undergoes:

1. **Trend removal:** A 10-minute moving average is subtracted to remove synoptic weather-scale pressure changes, isolating the wind-induced fluctuation component.
2. **HVAC artifact rejection:** Forced-air HVAC systems create indoor pressure fluctuations of 0.5–3 Pa when the air handler operates. These are quasi-periodic at the HVAC cycling frequency (typically 3–8 cycles/hour) and are identified and masked using a notch filter at the detected cycling frequency. Alternatively, the system preferentially selects measurement windows during HVAC off-cycles.
3. **Door/window event detection:** Opening an exterior door or window creates a transient pressure equalization event (rapid indoor pressure change of 0.5–5 Pa over 1–3 seconds). These events are detected by a wavelet-based transient detector and excluded from the analysis window. A minimum of 20 minutes of continuous, undisturbed data is required per estimation epoch.
4. **Multi-sensor fusion:** When multiple indoor sensors are available, their signals are averaged after cross-correlation alignment to reduce uncorrelated noise. The spatial variance across sensors provides additional information about internal compartmentalization and leakage distribution.

### 4. Spectral Transfer Function Estimation

The core measurement is the spectral transfer function H(f) between outdoor wind-induced pressure and indoor pressure fluctuations. The outdoor wind-induced pressure at the building envelope is estimated from wind speed measurements using P_wind(t) = Cp_eff · 0.5 · ρ · V(t)², where Cp_eff is an effective wind pressure coefficient treated as a fitted parameter or estimated from building orientation.

H(f) is computed using Welch's method with Hanning-windowed segments of 256 seconds (frequency resolution 0.004 Hz), 50% overlap, and averaging over the full measurement epoch (minimum 20 minutes, preferred 2+ hours).

The system extracts the following features from H(f):

- **Cutoff frequency f_c:** Estimated by fitting a first-order low-pass model to |H(f)|. Primary predictor of ELA.
- **Roll-off slope:** The slope of |H(f)| in dB/decade above f_c. Deviations from -20 dB/decade indicate multi-zone leakage.
- **Low-frequency gain:** |H(f)| at f < 0.01 Hz, which approaches 1.0 for all buildings. Deviations indicate stack effect or mechanical ventilation.
- **Phase lag at f_c:** Should be approximately -45° for a single-zone model. Larger phase lags indicate distributed leakage pathways.
- **Coherence γ²(f):** Magnitude-squared coherence between indoor pressure and wind speed. High coherence (γ² > 0.5) in 0.05–0.5 Hz indicates adequate SNR.

### 5. Machine Learning Estimation Model

A CNN regression model is trained on paired measurements: spectral transfer function features from consumer barometric sensor data during windy periods, paired with blower door test results (ACH50, ELA, flow coefficient, pressure exponent) from the same building within 30 days.

Training data sources include DOE Weatherization Assistance Program (~35,000 annual audits), RESNET HERS ratings (~150,000 annual), utility weatherization programs, and voluntary homeowner contributions via companion app.

Model architecture: 1D CNN with 5 convolutional layers (32/64/128/128/64 filters, kernel size 5, ReLU, batch normalization) on log-magnitude and phase of H(f) at 128 frequency bins. Auxiliary inputs: building floor area, stories, year built, climate zone, mean outdoor temperature, mean wind speed. Output: ln(ACH50) regression. Separate classification head for confidence (high/medium/low).

Target: MAE < 1.5 ACH50, sufficient for population-scale screening (>85% accuracy distinguishing tight/moderate/leaky categories).

### 6. Continuous Monitoring and Degradation Detection

The system tracks envelope performance over time via a Bayesian state estimator maintaining running ACH50 with uncertainty bounds. Alerts trigger when:

- Estimated ACH50 increases by >2 units from baseline (new leakage pathway)
- Spectral shape of H(f) changes qualitatively (concentrated leak)
- Coherence suddenly increases across all frequencies (large new opening)

### 7. Population-Scale Applications

- **Weatherization targeting:** Identify leakiest 10% of buildings for prioritized outreach
- **Code compliance screening:** Flag new construction for targeted blower door verification
- **Portfolio energy modeling:** Compare envelope performance across multiple properties
- **Real estate disclosure:** Estimated air leakage as part of home energy profile
- **Climate policy monitoring:** Track aggregate building stock improvement over time

## Claims

1. A system for estimating building envelope air leakage rates without blower door testing, comprising: one or more indoor barometric pressure sensors in consumer IoT devices; outdoor wind speed and barometric pressure references; a signal processing module computing spectral transfer functions between outdoor wind-induced pressure and indoor barometric pressure fluctuations in the 0.01–2 Hz band; and a machine learning regression model estimating ACH50 from the spectral transfer function and building metadata.

2. The system of claim 1, wherein the spectral transfer function is computed using Welch's method with a minimum 20-minute measurement epoch during wind speeds exceeding a configurable threshold.

3. The system of claim 1, further comprising HVAC artifact rejection via notch filtering or selective measurement during HVAC off-cycles.

4. The system of claim 1, further comprising door/window event detection that excludes transient pressure equalization events from spectral analysis.

5. The system of claim 1, wherein building metadata inputs include building floor area, stories, year built, climate zone, outdoor temperature, and building orientation.

6. The system of claim 1, wherein the ML model is a CNN trained on paired spectral transfer functions and blower door results from weatherization programs and energy rating assessments.

7. A method for continuous envelope monitoring comprising: passively collecting indoor barometric data at ≥1 Hz; obtaining concurrent outdoor wind data; computing spectral transfer functions; estimating ACH50; maintaining Bayesian state estimates over time; and alerting on degradation.

8. The method of claim 7, detecting degradation via upward ACH50 trends, spectral shape changes, or coherence increases.

9. A method for population-scale weatherization prioritization using passive barometric air leakage estimates from consumer IoT devices across a geographic region.

10. The system of claim 1, wherein multiple indoor sensors are fused by cross-correlation alignment, with spatial variance providing compartmentalization information.

## Prior Art References

1. [DOE Energy Saver — Air Sealing Your Home](https://www.energy.gov/energysaver/air-sealing-your-home)
2. [ASTM E779-19](https://www.astm.org/e0779-19.html) — Fan pressurization air leakage test
3. [ASTM E1827-11(2017)](https://www.astm.org/e1827-11r17.html) — Orifice blower door test
4. [EIA RECS 2020](https://www.eia.gov/consumption/residential/) — U.S. housing unit count
5. [DOE WAP](https://www.energy.gov/scep/wap/about-weatherization-assistance-program) — Annual audit count
6. [RESNET HERS](https://www.resnet.us/about/hers-index/) — Annual rating count
7. Shaw & Tamura, Building and Environment (1977) — Power-law infiltration model
8. Orme, AIVC Technical Note 55 (2001) — AIM-2 natural infiltration model
9. [Bosch BMP390](https://www.bosch-sensortec.com/products/environmental-sensors/pressure-sensors/bmp390/) — MEMS barometric sensor specs
10. [Microsoft US Building Footprints](https://github.com/microsoft/USBuildingFootprints) — 130M+ footprints
11. [Weather Underground PWS](https://www.wunderground.com/pws/overview) — 300K+ U.S. stations
12. [2021 IECC](https://www.energycodes.gov/) — Air leakage code requirements
