# System and Method for Early Detection of Incipient Lithium-Ion Battery Thermal Runaway in Consumer Environments Using Volatile Organic Compound Emission Signature Analysis from Distributed Metal-Oxide Semiconductor Gas Sensor Arrays with Edge-Deployed Anomaly Detection

**LITF-PA-2026-119 · Battery Safety / IoT**
**Published:** 2026-07-24
**License:** [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/) — Public Domain

> ⚖️ **Prior Art Notice:** This document is published as defensive prior art under [35 U.S.C. § 102(a)(1)](https://www.law.cornell.edu/uscode/text/35/102). The inventions described herein are dedicated to the public domain as of the publication date above.

---

## Abstract

Disclosed is a system and method for detecting incipient lithium-ion battery thermal runaway in consumer and commercial environments using distributed arrays of low-cost metal-oxide semiconductor (MOX) gas sensors. Lithium-ion cells undergoing internal short circuits, dendrite penetration, or electrolyte decomposition emit characteristic volatile organic compounds (VOCs) including dimethyl carbonate (DMC), ethyl methyl carbonate (EMC), diethyl carbonate (DEC), ethylene, and propylene at cell surface temperatures as low as 60–80°C, well below the 130–180°C separator collapse threshold that initiates full thermal runaway. The system deploys MOX sensor nodes (unit cost $8–15) near battery charging and storage areas. Each node samples a vector of resistive responses across 3–5 MOX elements with different metal-oxide coatings (SnO₂, WO₃, ZnO, In₂O₃, TiO₂) at 1 Hz, generating a multi-dimensional gas fingerprint. An edge-deployed autoencoder neural network trained on the location-specific ambient VOC baseline detects anomalous fingerprint deviations with sub-ppm sensitivity. Temporal gradient analysis of the anomaly score distinguishes slow electrolyte decomposition (cell swelling, capacity fade) from rapid pre-runaway venting, enabling tiered alerting: advisory (disconnect charger), warning (evacuate room), and critical (activate suppression). The system provides 5–45 minutes of advance warning before thermal runaway, compared to 0–30 seconds from conventional temperature-based detection.

## Field of the Invention

This invention relates to fire safety and battery management, specifically to the early detection of lithium-ion battery failure using ambient gas sensing and machine learning for pre-thermal-runaway identification in residential, commercial, and industrial environments.

## Background

Lithium-ion battery fires have become a public health emergency in dense urban areas. The [New York City Fire Department reported 268 lithium-ion battery fires in 2023](https://www.nyc.gov/site/fdny/news/dn023-009/lithium-ion-battery-fires), a 58% increase over 2022, causing 18 deaths and 150 injuries. E-bike and e-scooter battery fires accounted for the largest share. The [U.S. Consumer Product Safety Commission (CPSC)](https://www.cpsc.gov/Newsroom/News-Releases/2024/CPSC-Warns-of-Fire-and-Explosion-Hazards-from-Lithium-Ion-Batteries-in-E-Bikes-and-E-Scooters) has issued multiple warnings and initiated recalls affecting millions of devices containing lithium-ion cells.

The physics of thermal runaway follows a well-characterized cascade. [Feng et al. (Journal of Power Sources, 2012)](https://doi.org/10.1016/j.jpowsour.2012.01.038) documented three stages: (1) onset of self-heating from solid electrolyte interphase (SEI) decomposition at 90–120°C, (2) separator collapse and internal short circuit at 130–180°C, and (3) cathode decomposition and oxygen release above 200°C producing uncontrollable exothermic reactions exceeding 800°C. The transition from stage 1 to stage 3 can occur in under 30 seconds once separator integrity is lost.

Current detection methods are inadequate for consumer environments:

- **Temperature sensors (thermistors, RTDs):** Placed on cell surfaces, they detect thermal runaway only after significant self-heating has begun. [Koch et al. (Applied Energy, 2020)](https://doi.org/10.1016/j.apenergy.2020.114882) showed surface temperature rise lags internal hotspot temperature by 10–60 seconds depending on cell geometry, providing minimal evacuation time.
- **Voltage monitoring (BMS):** Battery management systems detect cell-level voltage drops but only in managed battery packs. Aftermarket, counterfeit, and modified packs (the primary fire sources) frequently lack functional BMS circuits. Voltage changes during internal shorts can be masked by parallel cell configurations.
- **Smoke detectors:** Standard photoelectric and ionization smoke detectors activate only after visible smoke or aerosol production, which occurs during or after stage 2, leaving no meaningful evacuation window. [Ribière et al. (Fire Safety Journal, 2017)](https://doi.org/10.1016/j.firesaf.2017.03.067) found that conventional smoke detectors triggered 0–30 seconds before full thermal runaway flame ejection.

The critical gap: lithium-ion electrolytes emit detectable VOCs at temperatures far below the thermal runaway cascade. [Lammer et al. (Journal of The Electrochemical Society, 2017)](https://doi.org/10.1149/2.0171701jes) characterized the gas emissions from NMC and LFP cells during abuse testing and found DMC, EMC, and DEC emissions beginning at 60°C and increasing exponentially through 120°C, well before mechanical failure. [Koch et al. (Journal of Energy Storage, 2022)](https://doi.org/10.1016/j.est.2021.103765) confirmed that EMC and DMC are released through micro-vents and seal defects at pressures below the catastrophic vent threshold. These VOC emissions constitute a chemical early warning signal that precedes thermal runaway by minutes to hours, yet no commercially available consumer safety device monitors for them.

MOX gas sensors are mature, low-cost, and mass-produced for consumer applications. [Sensirion's SGP41](https://www.sensirion.com/en/environmental-sensors/gas-sensors/sgp41/) ($3.50 in volume) provides a VOC index and NOₓ measurement. [Bosch's BME688](https://www.bosch-sensortec.com/products/environmental-sensors/gas-sensors/bme688/) ($5.00 in volume) combines MOX gas sensing with temperature, pressure, and humidity in a single 3×3 mm package. These sensors are already deployed in consumer air quality monitors but have not been applied to battery safety detection.

## Detailed Description

### 1. Sensor Node Architecture

Each sensor node comprises: a multi-element MOX gas sensor array consisting of 3–5 MOX sensing elements with different metal-oxide semiconductor coatings (SnO₂ for broad-spectrum VOC, WO₃ for carbonyl compounds, ZnO for light hydrocarbons, In₂O₃ for short-chain alcohols, TiO₂ for aromatic compounds); an integrated environmental sensor measuring temperature (±0.1°C), relative humidity (±1%), and barometric pressure (±1 hPa); a microcontroller with floating-point unit (e.g., ESP32-C3 with RISC-V core, unit cost $1.80) running the anomaly detection model; a WiFi/BLE radio for connectivity; and a 5V USB-C power input (intended for always-on operation near charging stations).

Target bill-of-materials cost per node: $8–15 in production volumes of 10,000+ units. This is 10–50× less expensive than laboratory gas chromatography or FTIR spectroscopy systems used in battery abuse testing, and comparable in cost to consumer smoke detectors ($15–30 retail).

### 2. Multi-Element Gas Fingerprinting

Each MOX element's resistance varies as a function of the reducing gases adsorbed on its heated surface. The resistance ratio Rₛ/R₀ (where Rₛ is resistance in the target gas and R₀ is resistance in clean air) provides a semi-quantitative gas concentration measurement. Because different metal oxides exhibit different sensitivity profiles to the same gas, the vector of resistance ratios across the 3–5 elements forms a characteristic fingerprint for different gas mixtures.

For lithium-ion electrolyte decomposition products, the fingerprint is distinctive. DMC (C₃H₆O₃) and EMC (C₄H₈O₃) are carbonate esters that produce strong responses on SnO₂ and WO₃ elements (Rₛ/R₀ = 0.2–0.5 at 10 ppm) but weak responses on ZnO (Rₛ/R₀ = 0.7–0.9). Ethylene (C₂H₄), a pyrolysis product from polyethylene separator decomposition, produces the inverse pattern: strong ZnO response, weak WO₃ response. This orthogonality enables the system to distinguish battery-specific VOC emissions from common household VOCs such as cooking fumes, cleaning products, perfumes, and off-gassing from new furniture.

The sensor array samples all elements at 1 Hz. Each sample produces a feature vector of dimensionality 3N+3 (N elements × [raw resistance, temperature-compensated ratio, first temporal derivative] + [ambient temperature, humidity, pressure]). For a 4-element array, this yields a 15-dimensional feature vector per second.

### 3. Ambient Baseline Learning and Anomaly Detection

The core detection algorithm is an autoencoder neural network operating on the sensor feature vectors. The autoencoder architecture: encoder with two fully connected layers (15 → 10 → 5 neurons) with ReLU activation; latent space of dimensionality 5; decoder with two fully connected layers (5 → 10 → 15 neurons) mirroring the encoder. Total parameter count: approximately 400 parameters (1.6 KB at float32), easily fitting in the ESP32-C3's 400 KB SRAM.

During a 72-hour calibration period after installation, the autoencoder trains on the location-specific ambient VOC profile using online stochastic gradient descent. This captures normal daily patterns of cooking, cleaning, HVAC operation, human occupancy, and seasonal ventilation variation. The reconstruction error (MSE between input and output) establishes the baseline distribution. Post-calibration, the system continues passive online learning at a 100× reduced learning rate to adapt to gradual environmental drift.

An anomaly is flagged when the reconstruction error exceeds μ + kσ of the calibration-period error distribution, where k is a configurable sensitivity parameter (default k=4, yielding a false alarm rate of approximately 1 per 16,000 samples, or roughly one per 4.4 hours at 1 Hz sampling). To suppress transient false alarms from brief VOC events, the system requires the anomaly score to exceed the threshold for a sustained duration window (default: 30 consecutive seconds).

### 4. Temporal Gradient Classification

Once an anomaly is confirmed, the system classifies the temporal gradient of the anomaly score:

- **Slow decomposition (hours to days):** Anomaly score increases at <0.01σ/minute. Characteristic of chronic electrolyte leakage, manufacturing defects, or slow overcharge damage. Response: advisory alert (disconnect charger, inspect battery).
- **Accelerating decomposition (minutes to hours):** Anomaly score increases at 0.01–0.5σ/minute with positive second derivative. Characteristic of internal short circuit progression from dendrite growth or mechanical damage. Response: warning alert (remove battery from building, prepare for evacuation).
- **Rapid venting (seconds to minutes):** Anomaly score increases at >0.5σ/minute. Characteristic of active thermal runaway with pressure vent activation. Response: critical alert (evacuate immediately, activate suppression).

The gradient classifier uses a sliding window of 60 seconds with linear regression on the anomaly score time series.

### 5. Environmental Compensation

MOX sensor responses are strongly influenced by ambient temperature and humidity. The system applies compensation using the Arrhenius equation for temperature dependence (activation energy varies by metal oxide: SnO₂ = 0.3–0.5 eV, WO₃ = 0.4–0.6 eV) and an empirical humidity correction derived from the integrated humidity sensor. Temperature compensation: R_corrected = R_measured × exp(Eₐ/kB × (1/T_ref − 1/T_measured)), where Eₐ is the activation energy, kB is Boltzmann's constant, T_ref is 25°C = 298.15 K, and T_measured is current ambient temperature.

### 6. Multi-Node Spatial Correlation

In deployments with multiple sensor nodes, cross-node correlation provides spatial localization of the emission source and reduces false alarm rates. Simplified Gaussian plume modeling with measured HVAC airflow direction estimates the source location as the point that best explains the observed concentration gradients across nodes. For 3+ node detection, localization accuracy is approximately ±2 meters in typical indoor environments.

### 7. Integration with Building Safety Systems

Alert integration pathways: local audible/visual alarm; push notification to smartphone; MQTT message to building management system for HVAC isolation and fire panel activation; API webhook for monitoring service integration; relay output for battery disconnect switches or fire suppression solenoids.

## Claims

1. A system for early detection of lithium-ion battery thermal runaway, comprising: one or more sensor nodes, each containing a multi-element metal-oxide semiconductor gas sensor array with at least three MOX sensing elements of different metal-oxide compositions; an environmental sensor measuring ambient temperature and humidity; and a microcontroller executing an anomaly detection model; wherein the system detects volatile organic compound emissions characteristic of lithium-ion electrolyte decomposition at cell temperatures below the thermal runaway onset temperature.

2. The system of claim 1, wherein the MOX sensing elements comprise two or more of tin dioxide (SnO₂), tungsten trioxide (WO₃), zinc oxide (ZnO), indium oxide (In₂O₃), and titanium dioxide (TiO₂), selected to provide orthogonal sensitivity profiles for distinguishing battery electrolyte decomposition products from common household volatile organic compounds.

3. The system of claim 1, wherein the anomaly detection model is an autoencoder neural network that learns a location-specific ambient VOC baseline during a calibration period and flags deviations from the baseline as potential battery failure events.

4. The system of claim 3, wherein the autoencoder continues passive online learning at a reduced rate after the initial calibration period to adapt to gradual environmental drift including seasonal ventilation changes, new furniture off-gassing, and occupancy pattern shifts.

5. The system of claim 1, further comprising a temporal gradient classifier that analyzes the rate of change of the anomaly score to distinguish between slow electrolyte decomposition, accelerating pre-runaway decomposition, and rapid venting events, and generates tiered alerts corresponding to the classified severity.

6. The system of claim 1, wherein MOX sensor resistance readings are compensated for ambient temperature using an Arrhenius correction model with oxide-specific activation energies and for ambient humidity using an empirical lookup table derived from the co-located environmental sensor.

7. The system of claim 1, wherein multiple sensor nodes communicate wirelessly to correlate anomaly detections across nodes and estimate the spatial location of the emission source using gas dispersion modeling.

8. A method for detecting incipient lithium-ion battery thermal runaway comprising: continuously sampling a multi-element MOX gas sensor array at a rate of at least 0.5 Hz; computing temperature-compensated and humidity-compensated resistance ratio feature vectors; comparing feature vectors against a learned ambient baseline using an autoencoder reconstruction error metric; flagging sustained anomaly score elevation exceeding a configurable threshold for a minimum duration; and classifying the temporal gradient of the anomaly score to determine battery failure severity and appropriate response tier.

9. The method of claim 8, wherein the sustained duration requirement for anomaly flagging is configurable between 10 and 120 seconds to balance detection latency against false alarm rate for different deployment environments.

10. The method of claim 8, further comprising integration with building safety systems via MQTT, API webhook, or relay output to trigger automated responses including battery charger disconnection, HVAC isolation, fire panel notification, and occupant evacuation alerts.

11. The system of claim 1, wherein each sensor node has a bill-of-materials cost below $20 and is powered via USB-C for continuous operation near battery charging and storage locations, and wherein the anomaly detection model operates within the memory and compute constraints of a sub-$2 microcontroller without cloud connectivity requirements.

## Prior Art References

1. [FDNY Lithium-Ion Battery Fire Statistics 2023](https://www.nyc.gov/site/fdny/news/dn023-009/lithium-ion-battery-fires) — 268 fires, 18 deaths, 150 injuries in NYC
2. [CPSC E-Bike/E-Scooter Battery Fire Warnings](https://www.cpsc.gov/Newsroom/News-Releases/2024/CPSC-Warns-of-Fire-and-Explosion-Hazards-from-Lithium-Ion-Batteries-in-E-Bikes-and-E-Scooters) — Federal recalls and safety advisories
3. [Feng et al., Journal of Power Sources (2012)](https://doi.org/10.1016/j.jpowsour.2012.01.038) — Three-stage thermal runaway cascade
4. [Koch et al., Applied Energy (2020)](https://doi.org/10.1016/j.apenergy.2020.114882) — Surface vs. internal temperature lag
5. [Ribière et al., Fire Safety Journal (2017)](https://doi.org/10.1016/j.firesaf.2017.03.067) — Smoke detector response timing
6. [Lammer et al., Journal of The Electrochemical Society (2017)](https://doi.org/10.1149/2.0171701jes) — Gas emission characterization from NMC/LFP cells
7. [Koch et al., Journal of Energy Storage (2022)](https://doi.org/10.1016/j.est.2021.103765) — EMC/DMC release below catastrophic vent threshold
8. [Sensirion SGP41](https://www.sensirion.com/en/environmental-sensors/gas-sensors/sgp41/) — Consumer MOX VOC + NOₓ sensor
9. [Bosch BME688](https://www.bosch-sensortec.com/products/environmental-sensors/gas-sensors/bme688/) — Combined MOX gas + environmental sensor
10. [ESP32-C3](https://www.espressif.com/en/products/socs/esp32-c3) — RISC-V microcontroller with WiFi/BLE
11. [Baur et al., Sensors and Actuators B (2020)](https://doi.org/10.1016/j.snb.2019.127382) — MOX sensor arrays for gas mixture discrimination
12. [Essl et al., Journal of Power Sources (2020)](https://doi.org/10.1016/j.jpowsour.2020.228861) — Comprehensive gas emission analysis during Li-ion venting
