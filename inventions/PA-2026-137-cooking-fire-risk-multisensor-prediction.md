# PA-2026-137: System and Method for Residential Cooking Fire Risk Prediction Using Multi-Sensor Fusion of Stove Current Signature, Range Hood Particulate Density, and Ambient Acoustic Anomaly Detection with Edge-Deployed Temporal Convolutional Network

**Filing:** LITF-PA-2026-137  
**Domain:** Fire Safety / Edge AI / Sensor Fusion  
**Published:** August 11, 2026  
**Type:** Defensive Prior Art Disclosure  

---

## Abstract

Disclosed is a system and method for predicting residential cooking fires 2-5 minutes before ignition using edge-deployed multi-sensor fusion. The system comprises three sensing modalities: (1) a smart plug with high-frequency current sampling (4 kHz) on the stove circuit, performing non-intrusive load monitoring (NILM) to classify cooking method, estimate surface temperature from resistive heating element impedance drift, and detect unattended cooking via temporal activity patterns; (2) a compact particulate and volatile organic compound (VOC) sensor module integrated into or adjacent to the range hood, measuring PM2.5 concentration, total VOC level, and their first and second time derivatives to detect the characteristic exponential particulate ramp that precedes grease ignition by 90-300 seconds; and (3) a MEMS microphone array positioned in the kitchen that classifies the acoustic signature transition from normal cooking sounds (sizzling, bubbling, boiling) to pre-fire conditions (carbonization crackling, oil decomposition popping, dry-pan thermal stress) using a mel-spectrogram convolutional classifier. A temporal convolutional network (TCN) running on a low-power edge processor (e.g., Coral Edge TPU or ESP32-S3) fuses 60-second sliding windows from all three modalities, outputting a fire risk score (0-100) with configurable alert thresholds. At risk score 70, the system issues a smartphone notification; at 85, it triggers audible alarm and optional smart plug shutoff. The system operates entirely on-device with no cloud dependency, preserving privacy and ensuring sub-second response latency.

## Field of the Invention

This invention relates to residential fire safety, specifically to predictive fire risk assessment in cooking environments using multi-modal sensor fusion and edge-deployed machine learning for pre-ignition intervention.

## Background

Cooking is the leading cause of residential fires in the United States, responsible for 49% of all home structure fires, 42% of home fire injuries, and 21% of home fire deaths (NFPA, Ahrens & Maheshwari 2024, based on 2017-2021 NFIRS data). The U.S. Fire Administration reports approximately 187,600 cooking fires per year, causing 550 deaths, 4,820 injuries, and $1.2 billion in direct property damage annually. Unattended cooking is the leading contributing factor in 31% of these fires.

Current cooking fire prevention and detection technologies have fundamental limitations:

- **Smoke detectors (ionization and photoelectric):** Detect airborne combustion particles only after visible smoke production, which occurs simultaneously with or after ignition. False alarm rates of 85-97% in kitchen environments (Cleary, NIST TN 1629, 2010) lead to chronic disabling, with 25% of U.S. homes having at least one non-functional smoke alarm (NFPA, Ahrens 2023). By design, these devices alert to fire, not to fire risk.
- **Stove guard systems (e.g., FireAvert, CookStop):** Monitor smoke detector activation or stove surface temperature and cut power when thresholds are exceeded. Response occurs at or after ignition, not before. FireAvert activates only when an existing smoke detector sounds. Temperature-only systems cannot distinguish between intentional high-heat cooking (searing, wok stir-fry at 260-315°C) and a hazardous unattended pan.
- **Stove-top timers and auto-shutoff:** Fixed-duration shutoffs (e.g., 8 or 12 hours) prevent overnight hazards but do not address the 2-15 minute window in which most cooking fires develop from normal cooking conditions.
- **Computer vision monitoring (e.g., Wallflower, SensorFlow):** Camera-based systems can detect flame and smoke visually but raise significant privacy concerns in residential kitchens. These require persistent video capture of living spaces and are rejected by a substantial fraction of potential users on privacy grounds. They also struggle with steam/smoke disambiguation.

The gap in the art is a pre-ignition prediction system that: (a) fuses multiple non-visual sensing modalities to detect the thermochemical precursors of cooking fires before ignition occurs; (b) distinguishes normal high-heat cooking from genuinely hazardous conditions using contextual multi-modal analysis rather than single-threshold triggers; (c) operates without cameras, preserving kitchen privacy; (d) runs entirely on-device with no cloud dependency; and (e) provides graduated risk scoring with configurable intervention levels from notification through automated shutoff.

## Detailed Description

### 1. Stove Current Signature Analysis Module

The first sensing modality is a smart plug or inline current monitor installed on the stove's electrical circuit (240V/50A for U.S. electric ranges, or 120V/20A for countertop appliances). The module samples current at 4 kHz with 12-bit resolution using a split-core current transformer (e.g., SCT-013-030, $3.50) and a dedicated ADC (ADS1115, $5.00).

The high-frequency current waveform enables several inferences:

- **Cooking method classification via NILM:** Non-intrusive load monitoring disaggregates the total stove current into individual heating element contributions. Electric coil and radiant elements exhibit characteristic duty-cycling patterns (on/off at 15-60 second intervals for thermostat-regulated burners). Induction elements produce high-frequency switching signatures (20-75 kHz fundamental, aliased into the 4 kHz sample stream). The system classifies active cooking mode: boiling (steady state, low power oscillation), searing (high power, brief duration), simmering (low power, long duration), oven bake (cyclic with period matching oven thermostat hysteresis), and idle-hot (element cooling curve with no new input).
- **Surface temperature estimation from impedance drift:** Resistive heating elements (Nichrome alloy) exhibit a positive temperature coefficient of resistance (TCR) of approximately +0.0004/°C. At 4 kHz sampling, the system measures the ratio of voltage (inferred from known supply voltage and phase angle) to current, tracking the slow impedance increase as the element heats. A calibration curve maps impedance drift to surface temperature with ±15°C accuracy between 100-400°C, sufficient to detect a pan approaching the 360°C autoignition temperature of common cooking oils without requiring a direct temperature sensor on the cooking surface.
- **Unattended cooking detection:** The system maintains a state machine tracking human interaction indicators: element power changes (manual knob adjustments), rapid current transients (placing/removing cookware, which changes the magnetic coupling on induction elements), and correlated activity from other kitchen circuits if available (refrigerator door, microwave, kitchen lights). Absence of interaction indicators for a configurable duration (default: 5 minutes at high heat, 15 minutes at medium, 30 minutes at low) triggers an "unattended" flag that increases the risk model's sensitivity.

For gas ranges, the current module is replaced by a gas flow sensor (thermal mass flow meter, e.g., Sensirion SFM3003, $18) on the gas supply line, providing equivalent cooking state inference from flow rate patterns. Gas igniter current draw (400-600 mA pulse on the igniter circuit) provides an additional timing signal.

### 2. Range Hood Particulate and VOC Sensor Module

The second sensing modality is a compact multi-sensor module mounted inside or directly below the range hood, in the exhaust airflow path above the cooking surface. The module comprises:

- **Laser-scattering particulate sensor:** Plantower PMS5003 ($12) or Sensirion SPS30 ($28), measuring PM1.0, PM2.5, and PM10 concentrations at 1 Hz update rate with ±10% accuracy above 100 μg/m³. The sensor is positioned in the range hood capture zone where cooking aerosols are concentrated before exhaust, providing 10-50x higher signal-to-noise ratio compared to room-level placement.
- **Metal oxide VOC sensor:** Sensirion SGP41 ($5) measuring total VOC index and NOx index at 1 Hz. Cooking oil thermal decomposition begins producing acrolein (2-propenal) at 200-230°C, well below the smoke point, and increases exponentially as the oil approaches its smoke point (e.g., 232°C for refined canola, 252°C for refined safflower). Acrolein is a potent irritant and a specific chemical marker of oil decomposition that the VOC sensor detects as a rise in the raw VOC signal 60-120 seconds before visible smoke production.
- **Temperature and humidity sensor:** Sensirion SHT41 ($2), providing exhaust air temperature (correlates with cooking intensity) and humidity (distinguishes steam from smoke; steam produces high humidity with low particulate, while oil smoke produces high particulate with stable or falling humidity).

The critical signal is the PM2.5 time derivative profile. Normal cooking produces PM2.5 levels of 50-500 μg/m³ at the range hood inlet with gradual, bounded increases. Pre-fire conditions produce a characteristic exponential ramp: PM2.5 concentration doubles every 15-30 seconds as oil approaches autoignition, driven by accelerating pyrolysis of fatty acids. The system computes the first derivative (rate of PM2.5 increase) and second derivative (acceleration of increase). A positive second derivative sustained for >30 seconds, combined with PM2.5 exceeding 800 μg/m³, indicates the oil is in the runaway decomposition regime between smoke point and flash point. This signal precedes visible flame by 90-300 seconds depending on oil type, volume, and heat input rate.

The humidity-to-particulate ratio (H/P ratio) serves as a critical disambiguation feature. Steam-heavy cooking (boiling pasta, steaming vegetables) produces H/P > 5.0 (high humidity, low particulate). Grease decomposition produces H/P < 0.3 (rising particulate, stable or declining humidity). Normal frying occupies the middle range (H/P 0.5-3.0). The H/P ratio prevents false alarms from steam-intensive cooking that would trigger a particulate-only system.

### 3. Acoustic Anomaly Detection Module

The third sensing modality is a MEMS microphone array (2 microphones, e.g., Knowles SPH0645LM4H, $1.50 each) positioned 0.5-2.0 m from the cooking surface, typically integrated into the range hood housing or a countertop-mounted sensor puck.

Cooking produces distinctive acoustic signatures across the 200 Hz - 8 kHz range:

- **Normal sizzling (oil-food contact):** Broadband noise centered at 2-6 kHz, produced by rapid steam generation as moisture in food contacts hot oil. Spectral envelope is relatively stable over minutes. RMS amplitude correlates with cooking intensity but remains bounded.
- **Boiling/simmering:** Low-frequency bubble collapse events (200-800 Hz) with quasi-periodic structure. Spectral centroid below 1 kHz.
- **Pre-fire carbonization crackling:** Impulsive, high-amplitude transients (>15 dB above ambient cooking noise floor) at 1-4 kHz, produced by thermal decomposition of organic material on the cooking surface or pan walls. These events are distinguishable from normal sizzle by their impulsive temporal profile (rise time <2 ms, decay time 5-15 ms vs. continuous noise for sizzle) and their increasing repetition rate as carbonization progresses.
- **Oil decomposition popping:** Irregular, sharp transients at 3-8 kHz caused by volatile decomposition products bursting through the oil surface as the oil temperature exceeds its smoke point. Distinguished from food-moisture sizzle by higher frequency content and increasing irregularity (decreasing inter-event interval variance, indicating a shift from stochastic moisture-driven events to thermally-driven decomposition).
- **Dry-pan thermal stress:** High-frequency tonal components (4-12 kHz) from thermal expansion of empty or near-empty cookware. Empty pan heating produces characteristic frequency sweeps as the metal expands and internal stresses develop.

Audio is processed in 1-second frames with 50% overlap. Each frame is converted to a 64-bin mel-frequency spectrogram. A lightweight 1D convolutional classifier (3 layers, 16/32/64 filters, model size ~120 KB after INT8 quantization) outputs probability vectors over classes: normal_sizzle, boil_simmer, carbonization, oil_decomposition, dry_pan, background, and other. The classifier does not perform speech recognition or record intelligible audio; it processes only spectral features in the cooking-relevant frequency bands, and raw audio buffers are overwritten every 2 seconds, preserving privacy.

### 4. Temporal Convolutional Network Fusion

The three sensing modalities are fused by a temporal convolutional network (TCN) running on a low-power edge processor. The TCN architecture uses dilated causal convolutions (Bai et al., 2018) with an exponentially increasing dilation schedule (1, 2, 4, 8, 16, 32), providing a 64-second receptive field from a 60-second sliding input window.

Input features at each 1-second timestep (60 timesteps per window):

- **Current module (8 features):** Active power (W), estimated element temperature (°C), cooking mode class probabilities (5 classes), unattended duration (seconds, log-scaled), power change rate (W/s).
- **Particulate module (8 features):** PM2.5 (μg/m³, log-scaled), PM2.5 first derivative, PM2.5 second derivative, total VOC index, VOC first derivative, exhaust temperature (°C), humidity (%), H/P ratio.
- **Acoustic module (9 features):** Acoustic class probabilities (7 classes), RMS amplitude (dB), spectral centroid (Hz).
- **Cross-modal (2 features):** Power-particulate correlation coefficient (rolling 30s window), acoustic-particulate correlation coefficient (rolling 30s window).

Total: 27 input features x 60 timesteps = 1,620 values per inference. The TCN comprises 6 residual blocks with 32 filters each, batch normalization, and dropout (0.1). Output: single scalar fire risk score (0-100) via sigmoid activation scaled to [0, 100]. Total model parameters: approximately 45,000 (INT8 quantized size: ~50 KB). Inference time: <50 ms on ESP32-S3, <10 ms on Coral Edge TPU.

The TCN is trained on a dataset combining: (a) controlled cooking experiments spanning 15 cooking methods x 8 oil types x 3 cookware materials x attended/unattended conditions (estimated 500+ hours of labeled multi-modal data from purpose-built test kitchens); (b) real kitchen ambient data collected from consenting beta users with all three sensors deployed (thousands of hours of normal cooking with no fire events, providing negative class data); and (c) synthetic fire escalation trajectories generated by physics-based simulation of oil heating curves, particulate generation models (Wallace et al., Building and Environment 2019), and acoustic models of carbonization. Actual fire events are simulated in controlled settings (outdoor test kitchens with fire suppression) to generate positive class training data without endangering occupied structures.

### 5. Graduated Alert and Intervention Protocol

The system implements a four-level graduated response:

- **Level 0 (risk score 0-39, Green):** Normal cooking. No action. System logs sensor data for model improvement (if user opts in).
- **Level 1 (risk score 40-69, Yellow):** Elevated risk. Subtle indicator (LED color change on sensor puck). No audible alert. Logged for pattern analysis. Typical triggers: high-heat cooking approaching upper normal bounds, unattended cooking at medium heat entering 10+ minute window.
- **Level 2 (risk score 70-84, Orange):** High risk. Smartphone push notification via local network (no cloud required if phone is on same WiFi; cloud-optional for remote notification). Notification includes: "Kitchen sensors detect elevated cooking risk. Check your stove." No false specificity about flame or fire.
- **Level 3 (risk score 85-100, Red):** Critical risk. Audible alarm from sensor puck (85 dB at 1 m). Smartphone notification. If configured, smart plug cuts stove power. If integrated with smart home system (e.g., via Matter/Thread), triggers additional responses: kitchen lights to full brightness, HVAC to exhaust mode, smart speaker verbal alert.

The intervention at Level 3 (power cutoff) is user-configurable and disabled by default. The system is designed as a warning system first, an automatic shutoff second. Users who enable auto-shutoff accept a higher false-positive intervention rate in exchange for maximum protection. A physical override button on the sensor puck or smart plug allows immediate re-energization after a shutoff event.

### 6. Calibration and Personalization

The system runs a 7-day calibration period after installation, during which it learns the baseline cooking patterns of the household: typical cooking durations, preferred heat levels, common cooking methods, and the specific particulate and acoustic signatures of the installed stove, cookware, and range hood configuration. During calibration, alerts are suppressed and the system records data to adjust per-household thresholds.

Personalization includes:

- **Wok cooking mode:** Wok stir-frying on high BTU burners produces extreme heat (>315°C), heavy smoke, and intense acoustic transients that would trigger false alarms in a generalized model. The system detects wok cooking via its distinctive rapid power cycling (gas adjustment), very short high-temperature episodes (30-90 seconds), and broadband high-amplitude acoustic signatures, and applies a wok-specific risk model with elevated thresholds.
- **Deep frying adjustment:** Deep frying maintains oil at 175-190°C with large thermal mass, producing steady high particulate with slow temperature changes. The system distinguishes controlled deep frying from uncontrolled oil heating by the presence of temperature stability (low first derivative) and periodic food-insertion transients.
- **Range hood exhaust rate compensation:** High-powered range hoods (600+ CFM) clear particulates rapidly, reducing the PM2.5 signal amplitude. The system learns the installed hood's clearance rate during calibration and adjusts PM2.5 thresholds and derivative expectations accordingly. A hood that is off or on low speed produces higher-amplitude signals and tighter thresholds.

### 7. Hardware Integration Options

The system can be deployed in three configurations:

- **Retrofit kit:** Three discrete modules (smart plug with CT sensor, range hood clip-on sensor puck, countertop acoustic sensor puck) communicating via BLE Mesh or Thread to a central hub (Raspberry Pi Zero 2W or dedicated ESP32-S3 board). Estimated BOM: $65-90. Installation: 10 minutes, no tools, no wiring.
- **Range hood integrated:** All three sensing modalities integrated into a replacement range hood control board. The range hood already contains a motor, lighting, and a grease filter positioned directly above the cooking surface. Adding a PM2.5 sensor, VOC sensor, microphone, and CT clamp on the stove circuit (passed through the hood's wiring chase) creates a single-device solution. Target incremental manufacturing cost: $15-25 above standard range hood electronics.
- **Smart stove OEM integration:** Stove manufacturers (e.g., GE, Samsung, LG) integrate current sensing, a surface-mount particulate sensor near the cooktop vent, and a MEMS microphone into the stove's existing control board. The stove's built-in processor runs the TCN model. Zero additional devices. Target incremental BOM: $8-12.

## Claims

1. A system for predicting residential cooking fire risk prior to ignition, comprising: a current sensing module on a stove electrical circuit that samples current waveforms and classifies cooking method, estimates cooking surface temperature from heating element impedance drift, and detects unattended cooking states; a particulate and volatile organic compound sensing module positioned in a range hood exhaust path that measures PM2.5 concentration and VOC levels and computes their temporal derivatives; an acoustic sensing module comprising at least one MEMS microphone that classifies cooking acoustic signatures into categories including normal sizzle, carbonization crackling, and oil decomposition popping using spectral feature analysis; and an edge-deployed machine learning model that fuses time-series data from all three modules and outputs a fire risk score representing the probability of ignition within a prediction horizon.

2. The system of claim 1, wherein the current sensing module estimates cooking surface temperature by measuring the impedance drift of resistive heating elements due to their positive temperature coefficient of resistance, without requiring a direct temperature sensor on the cooking surface.

3. The system of claim 1, wherein the particulate sensing module computes a humidity-to-particulate ratio to distinguish steam-intensive cooking from grease decomposition, preventing false alarms from boiling or steaming operations that produce high particulate levels with simultaneously high humidity.

4. The system of claim 1, wherein the particulate sensing module detects a pre-fire condition by identifying a sustained positive second derivative of PM2.5 concentration exceeding a configurable duration threshold, indicating the exponential particulate ramp characteristic of oil approaching autoignition temperature.

5. The system of claim 1, wherein the acoustic sensing module distinguishes carbonization crackling from normal cooking sizzle by detecting impulsive transients with rise times below a threshold, increasing repetition rate, and frequency content above the normal sizzle spectral centroid.

6. The system of claim 1, wherein the edge-deployed machine learning model is a temporal convolutional network with dilated causal convolutions processing a sliding window of multi-modal features, quantized for execution on a low-power microcontroller or edge accelerator without cloud connectivity.

7. The system of claim 1, further comprising a graduated alert protocol with at least three severity levels, wherein lower levels issue visual or smartphone notifications and the highest level triggers an audible alarm and optionally cuts electrical power to the stove via a smart plug or relay.

8. A method for predicting cooking fire risk comprising: continuously monitoring electrical current on a stove circuit to classify cooking state and estimate surface temperature; continuously measuring particulate concentration and volatile organic compound levels in the range hood exhaust airflow and computing their first and second time derivatives; continuously classifying kitchen acoustic signatures to detect transitions from normal cooking sounds to pre-fire indicators including carbonization and oil decomposition; fusing time-series features from all three sensing modalities using a temporal convolutional network to produce a fire risk score; and issuing graduated alerts at configurable risk thresholds before ignition occurs.

9. The method of claim 8, further comprising a calibration period during which the system learns household-specific cooking patterns, range hood exhaust performance, and baseline sensor signatures, and adjusts detection thresholds accordingly.

10. The method of claim 8, further comprising a cooking-mode-specific risk model that applies elevated alert thresholds for wok cooking and deep frying to reduce false alarm rates during intentionally high-heat cooking operations.

11. The system of claim 1, wherein all raw audio data is processed as spectral features only, with raw audio buffers overwritten within a configurable short duration, and no speech recognition or audio recording capability is present, preserving occupant privacy.

12. The system of claim 1, implemented as a retrofit kit comprising three discrete wireless sensor modules communicating via a low-power mesh protocol to a central edge processor, installable without tools or electrical wiring modification.

## Prior Art References

1. [NFPA Cooking Equipment Fire Statistics](https://www.nfpa.org/education-and-research/research/nfpa-research/fire-statistical-reports/cooking-equipment) (Ahrens & Maheshwari, 2024): 49% of U.S. home fires, 187,600/year
2. [USFA Cooking Fire Statistics](https://www.usfa.fema.gov/statistics/residential-fires/cooking.html): 550 deaths, 4,820 injuries, $1.2B damage annually
3. [Cleary, NIST TN 1629, 2010](https://doi.org/10.6028/NIST.TN.1629): Kitchen smoke detector false alarm rates: 85-97%
4. [NFPA Smoke Alarms Report](https://www.nfpa.org/education-and-research/research/nfpa-research/fire-statistical-reports/smoke-alarms-in-us-home-fires) (Ahrens, 2023): 25% of homes have non-functional smoke alarms
5. [FireAvert](https://fireavert.com/): Stove guard system triggered by smoke detector activation
6. [Wallace et al., Building and Environment 2019](https://doi.org/10.1016/j.buildenv.2019.106272): Residential cooking particulate emission characterization
7. [Bai et al., 2018](https://doi.org/10.48550/arXiv.1803.01271): Temporal convolutional networks for sequence modeling
8. [Babrauskas, Fire Safety Journal 2003](https://doi.org/10.1016/j.firesaf.2017.03.003): Cooking oil ignition temperatures and fire behavior
9. [TensorFlow Lite for Microcontrollers](https://www.tensorflow.org/lite/microcontrollers): On-device ML runtime for edge deployment
10. [ESP32-S3 SoC](https://www.espressif.com/en/products/socs/esp32-s3): Low-power microcontroller with vector DSP extensions
11. [Knowles SPH0645LM4H](https://www.knowles.com/docs/default-source/default-document-library/sph0645lm4h-1.pdf): MEMS microphone datasheet
12. [Sensirion SGP41](https://sensirion.com/products/catalog/SGP41): Multi-gas VOC and NOx sensor
13. [Plantower PMS5003](https://www.plantower.com/en/products_33/74.html): Laser scattering particulate matter sensor
14. [Hart, 1992; Zoha et al., IEEE TPEL 2012](https://doi.org/10.1109/TPEL.2012.2211481): Non-intrusive load monitoring (NILM) for appliance disaggregation

---

*Published as [LITF-PA-2026-137](https://liveinthefuture.org/priorart/cooking-fire-risk-multisensor-prediction.html) on liveinthefuture.org*
