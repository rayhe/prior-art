# System and Method for Continuous Non-Invasive Hydration Assessment Using Wearable Multi-Wavelength Photoplethysmography and Bioelectric Impedance Sensor Fusion

**LITF-PA-2026-123 · Wearables / HealthTech**
**Published:** 2026-07-29
**License:** [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/) — Public Domain

> ⚖️ **Prior Art Notice:** This document is published as defensive prior art under [35 U.S.C. § 102(a)(1)](https://www.law.cornell.edu/uscode/text/35/102). The inventions described herein are dedicated to the public domain as of the publication date above.

---

## Abstract

Disclosed is a system and method for continuous, non-invasive estimation of whole-body hydration status using a consumer wearable device that fuses two complementary sensing modalities: multi-wavelength photoplethysmography (PPG) operating at green (525 nm), red (660 nm), and near-infrared (940 nm) wavelengths, and tetrapolar bioelectric impedance spectroscopy (BIS) performed through skin-contact electrodes on the device caseback. The PPG subsystem extracts hydration-sensitive features including pulse transit time variability, AC/DC ratio shifts across wavelengths (indicating plasma volume changes), and peripheral perfusion index trends. The BIS subsystem sweeps excitation frequencies from 5 kHz to 1 MHz, fitting measured impedance spectra to a Cole-Cole model to separate intracellular (ICW) and extracellular (ECW) water compartments. A multimodal fusion network, implemented as a lightweight temporal attention transformer running on-device, integrates both sensor streams with contextual inputs (ambient temperature from an onboard thermistor, accelerometer-derived activity state, and time-of-day circadian phase) to produce a continuous hydration index calibrated against clinical serum osmolality. The system maintains a 14-day personalized baseline and employs a hybrid personalization approach combining on-device low-rank adaptation with periodic cloud-based Bayesian posterior updates, adapting to individual physiology, body composition, and behavioral patterns without requiring laboratory calibration.

## Field of the Invention

This invention relates to non-invasive physiological monitoring using consumer wearable devices, specifically to the fusion of optical and electrical biosensing modalities for continuous hydration status estimation with personalized machine learning models.

## Background

Dehydration affects an estimated [17-28% of older adults](https://pubmed.ncbi.nlm.nih.gov/30109950/) in developed countries (Hooper et al., Cochrane Database of Systematic Reviews 2015; El-Sharkawy et al., Age and Ageing 2015) and is implicated in [up to 1.8 million emergency department visits annually](https://pubmed.ncbi.nlm.nih.gov/31657610/) in the United States (Liamis et al., European Journal of Internal Medicine 2016). Mild dehydration (1-2% body mass loss) impairs cognitive performance by 10-15%, reduces physical work capacity by 25-30%, and increases injury risk in athletic and occupational settings ([Ganio et al., British Journal of Nutrition 2011](https://pubmed.ncbi.nlm.nih.gov/21736786/)). The gold standard for hydration assessment remains serum osmolality measurement via venous blood draw (reference range: 275-295 mOsm/kg), which requires laboratory equipment, trained phlebotomy staff, and yields only a point-in-time snapshot.

Current non-invasive hydration assessment methods each suffer significant limitations:

- **Urine specific gravity / urine color:** Inexpensive and widely used but episodic (only at voiding events), confounded by dietary solute load (protein, sodium), diuretic use, and medications that alter urine concentration. Sensitivity for detecting mild dehydration is only 40-60% ([Cheuvront & Kenefick, Nutrition Reviews 2014](https://pubmed.ncbi.nlm.nih.gov/26200017/)).
- **Body mass change:** Accurate for acute fluid loss in controlled exercise settings but requires a precise baseline weight, is confounded by food intake, clothing, and bladder/bowel status, and cannot distinguish water loss from other mass changes.
- **Single-frequency bioimpedance analysis (BIA):** Consumer scales (InBody, Tanita) and wrist-worn devices estimate total body water from impedance at 50 kHz. Samsung Galaxy Watch 4 and later models (shipping since August 2021) perform single-frequency BIA through caseback electrodes combined with a side-button return path, reporting body composition including total body water percentage. However, single-frequency BIA at 50 kHz cannot differentiate ICW from ECW compartments because cell membranes remain partially capacitive at this frequency. The technique is highly sensitive to electrode placement, skin contact quality, and postural fluid redistribution, with a typical error of ±2-5 L for total body water ([Earthman, Nutrition in Clinical Practice 2015](https://pubmed.ncbi.nlm.nih.gov/27465459/)). No shipping consumer wearable performs multi-frequency bioelectric impedance spectroscopy (BIS) across the full 5 kHz to 1 MHz range required for Cole-Cole modeling and ICW/ECW separation.
- **PPG-based approaches:** [US20200170525A1](https://patents.google.com/patent/US20200170525A1) (Valencell) describes a wearable hydration sensor using PPG amplitude ratios but relies on a single wavelength pair and does not incorporate impedance data or personalized modeling. The PPG signal is sensitive to hydration through its dependence on blood volume and peripheral vasomotor tone, but PPG alone lacks specificity: vasoconstriction from cold exposure, sympathetic activation, and caffeine produces similar waveform changes to dehydration.
- **Sweat-based biosensors:** [Gao et al., Nature 2016](https://pubmed.ncbi.nlm.nih.gov/26783950/) demonstrated a flexible sweat sensor array measuring sodium, potassium, glucose, and lactate. However, these devices require active sweating (limiting use to exercise contexts), are consumable (enzyme degradation), and measure sweat electrolyte concentration rather than systemic hydration status.

The gap in the art is a wearable system that: (a) performs multi-frequency bioelectric impedance spectroscopy (5 kHz to 1 MHz) at the wrist, enabling Cole-Cole model fitting and separation of intracellular from extracellular water compartments, which single-frequency BIA devices (including Samsung Galaxy Watch) cannot achieve, (b) fuses this multi-frequency BIS data with multi-wavelength PPG features to overcome the specificity limitations of either modality alone, (c) operates continuously during normal daily activities without requiring active sweating or user intervention, and (d) adapts to individual physiological variation through personalized online learning with modality-concordance self-supervision.

## Detailed Description

### 1. Multi-Wavelength PPG Subsystem

The PPG subsystem employs three LED wavelengths on the device caseback in direct skin contact at the dorsal wrist: green (525 nm, penetration depth ~1 mm, primarily arterial plethysmography in the dermal capillary bed), red (660 nm, penetration depth ~3 mm, sensitive to venous oxygen saturation and blood volume in deeper vessels), and near-infrared (940 nm, penetration depth ~5 mm, sensitive to tissue water absorption through the 970 nm water absorption peak shoulder). Each LED is driven in a time-multiplexed sequence at 100 Hz per channel (33.3 Hz effective per wavelength), with a photodiode (e.g., OSRAM SFH 2201, spectral range 400-1100 nm) capturing reflected light.

From the raw PPG waveforms, the system extracts hydration-sensitive features in sliding 30-second windows with 50% overlap:

- **Peripheral Perfusion Index (PPI):** Ratio of pulsatile (AC) to non-pulsatile (DC) components of the green-wavelength PPG signal. PPI decreases with dehydration-induced peripheral vasoconstriction. Baseline range: 1-20% in healthy adults; clinically significant dehydration correlates with PPI decline >40% from personal baseline ([Lima & Bakker, Revista Brasileira de Terapia Intensiva 2005](https://pubmed.ncbi.nlm.nih.gov/19399024/)).
- **Multi-Wavelength AC/DC Ratio Matrix:** The ratios of AC amplitude to DC baseline at each wavelength pair (green/red, green/NIR, red/NIR) form a 3-element vector that shifts predictably with plasma volume changes. Dehydration reduces plasma volume by 5-10% at 2-3% body mass loss ([Sawka et al., Medicine & Science in Sports & Exercise 1998](https://pubmed.ncbi.nlm.nih.gov/9694412/)), increasing hemoglobin concentration and altering optical absorption ratios across wavelengths. The NIR channel is particularly sensitive because the 940 nm absorption of water in tissue decreases as water content falls.
- **Pulse Transit Time Variability (PTTV):** Estimated from the green-wavelength PPG waveform foot-to-peak interval as a proxy for arterial stiffness changes. Dehydration increases blood viscosity by 3-8% per 1% body mass loss, accelerating pulse wave velocity. PTTV is computed as the coefficient of variation of pulse arrival times over 60-second epochs.
- **Dicrotic Notch Position Index (DNPI):** Relative timing and amplitude of the dicrotic notch in the red-wavelength PPG waveform, reflecting aortic valve closure and arterial compliance. Dehydration shifts the notch earlier in the cardiac cycle and reduces its amplitude, correlating with decreased stroke volume.

### 2. Tetrapolar Bioelectric Impedance Spectroscopy Subsystem

The BIS subsystem uses four stainless steel electrodes embedded in the device caseback in a tetrapolar configuration (two drive electrodes, two sense electrodes) spaced 8-12 mm apart. A constant-current source injects a sinusoidal excitation signal of 200 μA (peak-to-peak, well below the [IEC 60601-1 perception threshold of 500 μA](https://www.iec.ch/dml/info_iec60601-1%7Bed3.2%7Den.pdf)) at logarithmically spaced frequencies across the range 5 kHz to 1 MHz (typically 32 frequency points per sweep). The sense electrodes measure the resulting voltage, and a synchronous demodulator (implemented in the device's analog front-end IC, e.g., Analog Devices AD5940) extracts the complex impedance (magnitude and phase) at each frequency.

The measured impedance spectrum is fit to a Cole-Cole model parameterized by four variables: R₀ (impedance at zero frequency, reflecting total body water resistance), R∞ (impedance at infinite frequency, reflecting intracellular pathway resistance), τ (time constant of the characteristic frequency), and α (dispersion broadening factor). From these parameters:

- **Extracellular water (ECW) estimate:** Proportional to 1/R₀, dominated by the low-frequency impedance where current flows primarily through extracellular fluid.
- **Intracellular water (ICW) estimate:** Derived from the parallel combination of R₀ and R∞, corresponding to the high-frequency impedance path through both fluid compartments.
- **ECW/ICW ratio:** Shifts in this ratio indicate the type of fluid imbalance. Isotonic dehydration (e.g., sweat loss during exercise) reduces ECW preferentially (ECW/ICW ratio decreases). Hypertonic dehydration (insufficient water intake) depletes ICW as cells lose water osmotically (ECW/ICW ratio increases). This distinction is clinically relevant and cannot be made by PPG alone.

The wrist-local BIS measurement captures segmental impedance of the forearm, not whole-body impedance. The system applies a body segment-to-whole-body scaling model trained on paired measurements from a multi-electrode whole-body BIS device (e.g., ImpediMed SFB7) and the wearable. This scaling model is personalized during the calibration period using the user's height, weight, age, and sex as anthropometric inputs.

### 3. Contextual Signal Integration

Both PPG and BIS signals are confounded by non-hydration physiological states. The system mitigates these confounders by integrating contextual signals:

- **Ambient temperature:** An NTC thermistor on the device PCB measures skin-adjacent temperature. Cold-induced vasoconstriction mimics dehydration in PPG features; the model applies a temperature-dependent correction using a piecewise linear mapping trained on controlled thermal exposure data (15-40°C range).
- **Activity state:** The onboard 3-axis accelerometer (e.g., Bosch BMA456, 16-bit, ±8g) classifies the user's activity into sedentary, walking, vigorous exercise, and sleep states using a lightweight random forest classifier. BIS measurements are only considered valid during sedentary and sleep states (minimal motion artifact). PPG features are normalized by activity state, as exercise-induced vasodilation confounds perfusion index interpretation.
- **Circadian phase:** Hydration status follows a diurnal pattern driven by renal concentrating cycles, overnight insensible loss, and meal timing. The system models circadian hydration dynamics using the user's historical 24-hour hydration index profile, enabling separation of normal diurnal variation from pathological dehydration trends.
- **Postural state:** Derived from the accelerometer's static gravitational component. Transitioning from supine to upright causes 10-15% plasma volume redistribution to the lower extremities within 10 minutes ([Thompson et al., Journal of Applied Physiology 1991](https://pubmed.ncbi.nlm.nih.gov/3572864/)), affecting both PPG amplitude and segmental BIS. The fusion model includes postural state and time-since-transition as features.

### 4. On-Device Multimodal Fusion Architecture

The fusion model is a lightweight temporal attention transformer designed for edge deployment on wearable-class processors (e.g., Ambiq Apollo4 Blue Plus, ARM Cortex-M4F at 192 MHz, 2 MB SRAM). Architecture: 2-layer transformer encoder with 4 attention heads, 64-dimensional embeddings, and feed-forward dimension 128. Input sequence: 10-minute windows of PPG features (20 × 4 features), BIS parameters (2 × 4 Cole-Cole parameters, sampled every 5 minutes during valid windows), and contextual signals (temperature, activity, circadian phase, posture). Total model size: 380 KB (INT8 quantized). Inference latency: <50 ms per 10-minute window.

The model outputs a continuous Hydration Index (HI) on a 0-100 scale, calibrated against serum osmolality via a sigmoidal mapping: HI = 100 corresponds to serum osmolality ≤275 mOsm/kg (overhydrated), HI = 50 corresponds to 290 mOsm/kg (euhydrated), and HI = 0 corresponds to ≥310 mOsm/kg (severely dehydrated). The model also outputs a confidence score (0-1) reflecting signal quality and the model's epistemic uncertainty, estimated via Monte Carlo dropout with 5 forward passes.

### 5. Personalized Adaptation

Individual variation in skin pigmentation, subcutaneous fat thickness, forearm muscle mass, baseline blood viscosity, and autonomic reactivity produces substantial inter-person variability in both PPG and BIS signals for the same hydration state. The system addresses this through a two-phase personalization approach:

- **Calibration phase (days 1-14):** During the initial wear period, the system collects continuous sensor data while the user is prompted to self-report fluid intake events (via a companion app notification triggered by detected drinking gestures from the accelerometer — see Section 8 below). The system establishes a personal baseline distribution for all features during assumed-euhydrated states (morning after sleep with urine specific gravity <1.020 if available from companion app input, post-meal periods with confirmed fluid intake).
- **Adaptation phase (day 15+):** A personalization module continuously refines model predictions using two complementary mechanisms. First, on-device adaptation: the final two dense layers of the fusion transformer are fine-tuned using a low-rank update (LoRA, rank 4, adding only ~2 KB of trainable parameters) with gradient-free evolutionary optimization (CMA-ES with population size 8), avoiding backpropagation entirely and requiring <500 KB peak RAM. The fitness function combines modality concordance (agreement between PPG-derived and BIS-derived hydration estimates as a self-supervision signal) with physiological plausibility constraints (hydration index should correlate with time since last fluid intake event, inversely correlate with exercise duration, and follow expected circadian patterns). Second, periodic cloud synchronization (weekly, when the device is charging and connected to WiFi): the companion app uploads anonymized feature-label pairs to a cloud service that performs full Bayesian posterior updates using variational inference, returning updated model weights to the device. This hybrid approach keeps the computationally expensive Bayesian inference off the constrained MCU while enabling continuous on-device adaptation between cloud syncs.

The on-device adaptation rate is governed by a surprise-modulated schedule: rapid adaptation occurs when sensor readings deviate significantly from the current model's predictions (measured by the KL divergence between predicted and observed feature distributions exceeding 2 standard deviations of the historical divergence), while the model remains stable when predictions are well-calibrated. Model parameters are snapshotted weekly and can be rolled back if adaptation drift is detected (measured by increasing divergence between PPG and BIS modality predictions over a 48-hour window).

### 6. Alert and Intervention System

- **Mild dehydration warning (HI 30-40):** Passive notification on the wearable display recommending fluid intake, with estimated fluid deficit in mL computed from the personal baseline and body mass.
- **Moderate dehydration alert (HI 15-30):** Active haptic alert with companion app notification, including estimated time to clinical concern if current trajectory continues (linear extrapolation of HI trend over the past 2 hours).
- **Severe dehydration alarm (HI <15):** Persistent alert with recommendation to seek medical attention. Triggers emergency contact notification if enabled and HI remains below 15 for 30 minutes with declining trajectory.
- **Overhydration warning (HI >85 sustained >4 hours):** Alert for potential hyponatremia risk, particularly relevant for endurance athletes. Computed from the ECW/ICW ratio shift toward ECW expansion combined with high HI.
- **Rehydration tracking:** After a dehydration event, the system tracks the HI recovery trajectory and estimates time to euhydration, enabling the user to titrate fluid intake appropriately rather than overcorrecting.

### 7. Figures Description

- **Figure 1:** System architecture showing the dual-modality sensor hardware (multi-wavelength PPG + tetrapolar BIS on wearable caseback), contextual signal inputs, on-device fusion transformer, and personalization pipeline with output Hydration Index scale.
- **Figure 2:** Cole-Cole impedance spectra at three hydration states (euhydrated, mildly dehydrated at 2% body mass loss, moderately dehydrated at 4% body mass loss) showing characteristic shifts in R₀, R∞, and dispersion frequency, with ECW/ICW ratio derivation.
- **Figure 3:** Multi-wavelength PPG feature response to progressive dehydration during controlled heat exposure, showing differential sensitivity of green, red, and NIR channels to plasma volume reduction.
- **Figure 4:** 24-hour Hydration Index timeline for a representative subject showing circadian variation, exercise-induced dehydration, rehydration events detected from drinking gesture recognition, and alert threshold crossings with corresponding intervention notifications.
- **Figure 5:** On-device adaptation convergence curves showing prediction error reduction over the 14-day calibration period and subsequent stability during the adaptation phase, with modality concordance as a self-supervision metric and weekly cloud synchronization events marked.
- **Figure 6:** Drinking gesture accelerometer signature showing the characteristic gravitational axis rotation during fluid intake, with classifier decision boundaries separating drinking from confounding wrist motions (time-checking, head-scratching, phone-answering).

### 8. Drinking Gesture Recognition

The system includes an accelerometer-based drinking gesture recognition module that detects fluid intake events without manual logging. The module monitors the 3-axis accelerometer stream for a characteristic wrist motion signature: an upward rotation of the forearm (supination + elbow flexion, producing a distinctive gravitational axis shift from approximately [0, 0, -1g] to [0.7g, 0, -0.7g] in the device frame), sustained for 2-15 seconds (the drinking duration), followed by a return to the pre-drink orientation. A lightweight 1D CNN classifier (3 convolutional layers, 8/16/32 filters, ~12 KB INT8) processes 3-second sliding windows of accelerometer data at 50 Hz, distinguishing drinking gestures from confounders including checking the time (shorter duration, no sustained tilt), scratching the head (different rotation axis), and answering a phone call (similar tilt but typically held longer than 15 seconds and accompanied by speech-correlated micro-vibrations detectable in the accelerometer). Classification confidence threshold: 0.8. Validated drinking events are timestamped and used as implicit recalibration signals: the hydration model expects a positive HI trajectory following a confirmed drink event, and persistent negative trajectories despite frequent drinking events trigger a model confidence reduction and accelerated adaptation.

## Claims

1. A wearable system for continuous non-invasive hydration assessment, comprising: a multi-wavelength photoplethysmography (PPG) subsystem with at least three LED wavelengths including green (520-530 nm), red (650-670 nm), and near-infrared (930-950 nm); a tetrapolar bioelectric impedance spectroscopy (BIS) subsystem performing frequency sweeps across 5 kHz to 1 MHz through skin-contact electrodes; and an on-device multimodal fusion model that integrates features from both subsystems to produce a continuous Hydration Index calibrated against serum osmolality.

2. The system of claim 1, wherein the BIS subsystem fits measured impedance spectra to a Cole-Cole model to extract R₀, R∞, τ, and α parameters, and derives separate estimates of extracellular water (ECW) and intracellular water (ICW) volume from the frequency-dependent impedance, enabling differentiation of isotonic from hypertonic dehydration.

3. The system of claim 1, wherein the PPG subsystem extracts hydration-sensitive features including peripheral perfusion index, multi-wavelength AC/DC ratio matrix, pulse transit time variability, and dicrotic notch position index from simultaneous multi-wavelength recordings.

4. The system of claim 1, further comprising contextual signal integration including ambient temperature measurement, accelerometer-derived activity and postural state classification, and circadian phase estimation, used as additional inputs to the fusion model to mitigate non-hydration confounders.

5. The system of claim 1, wherein the multimodal fusion model is a temporal attention transformer quantized to INT8 and executing on a microcontroller-class processor, processing sequential 10-minute windows of PPG features, BIS parameters, and contextual signals.

6. A method for personalized hydration monitoring comprising: collecting continuous multi-wavelength PPG and multi-frequency bioelectric impedance data from a wrist-worn device; establishing a 14-day personalized physiological baseline during assumed-euhydrated states; and applying a hybrid personalization approach combining on-device low-rank adaptation using gradient-free optimization with periodic cloud-based Bayesian posterior updates, using modality concordance as a self-supervision signal and physiological plausibility constraints.

7. The method of claim 6, wherein the on-device adaptation uses low-rank updates to the fusion model's final layers with evolutionary optimization, avoiding backpropagation on the constrained microcontroller, and a surprise-modulated adaptation rate that increases when sensor readings exhibit high divergence from predictions and decreases when predictions are well-calibrated.

8. The method of claim 6, further comprising a body segment-to-whole-body impedance scaling model that maps wrist-local tetrapolar BIS measurements to whole-body hydration estimates using anthropometric inputs including height, weight, age, and sex.

9. The system of claim 1, further comprising a tiered alert system that generates warnings at configurable Hydration Index thresholds including overhydration alerts for sustained high HI values combined with ECW/ICW ratio shifts indicating potential hyponatremia risk.

10. The system of claim 1, further comprising a drinking gesture recognition module that detects fluid intake events from accelerometer wrist rotation patterns, providing implicit recalibration signals to the personalized hydration model without requiring manual user logging.

## Implementation Notes

The system is designed for integration into smartwatch form factors with existing PPG sensor arrays (Apple Watch Series 9+, Samsung Galaxy Watch 5+, and Garmin Venu 3 already include multi-wavelength PPG for SpO2 measurement). The BIS subsystem requires adding four electrodes to the caseback and an impedance measurement analog front-end IC (AD5940, ~$4 BOM). Total additional BOM cost for the BIS subsystem: approximately $8-12. Power consumption for continuous monitoring: ~15 mW average (PPG: 8 mW, BIS sweep every 5 minutes: 2 mW average, MCU inference: 5 mW). For a typical smartwatch battery of 300-500 mAh at 3.8V (1,140-1,900 mWh), continuous 24-hour operation consumes approximately 360 mWh, representing 19-32% of total battery capacity. This consumption can be reduced to ~8-12% by duty-cycling PPG sampling (10 seconds every 2 minutes rather than continuous) and performing BIS sweeps every 15 minutes rather than every 5 minutes, at the cost of reduced temporal resolution. In practice, a hybrid strategy is recommended: high-cadence sampling during exercise or when the Hydration Index approaches alert thresholds, and duty-cycled sampling during sedentary and sleep states.

## Prior Art References

1. [El-Sharkawy et al., Age and Ageing 2015](https://pubmed.ncbi.nlm.nih.gov/30109950/) — Prevalence of dehydration in older adults
2. [Ganio et al., British Journal of Nutrition 2011](https://pubmed.ncbi.nlm.nih.gov/21736786/) — Cognitive and physical performance effects of mild dehydration
3. [Cheuvront & Kenefick, Nutrition Reviews 2014](https://pubmed.ncbi.nlm.nih.gov/26200017/) — Limitations of urinary hydration biomarkers
4. [Earthman, Nutrition in Clinical Practice 2015](https://pubmed.ncbi.nlm.nih.gov/27465459/) — Bioimpedance analysis accuracy review
5. [US20200170525A1](https://patents.google.com/patent/US20200170525A1) (Valencell) — PPG-based hydration sensing (single wavelength pair)
6. [Gao et al., Nature 2016](https://pubmed.ncbi.nlm.nih.gov/26783950/) — Flexible sweat sensor arrays
7. [Sawka et al., Medicine & Science in Sports & Exercise 1998](https://pubmed.ncbi.nlm.nih.gov/9694412/) — Plasma volume changes during dehydration
8. [Lima & Bakker, Revista Brasileira de Terapia Intensiva 2005](https://pubmed.ncbi.nlm.nih.gov/19399024/) — Peripheral perfusion index clinical applications
9. [Thompson et al., Journal of Applied Physiology 1991](https://pubmed.ncbi.nlm.nih.gov/3572864/) — Postural plasma volume redistribution
10. [IEC 60601-1 Ed. 3.2](https://www.iec.ch/dml/info_iec60601-1%7Bed3.2%7Den.pdf) — Medical electrical equipment safety standard (perception current threshold)
11. [Analog Devices AD5940](https://www.analog.com/en/products/ad5940.html) — High-precision impedance and electrochemical front-end IC
12. [TensorFlow Lite for Microcontrollers](https://www.tensorflow.org/lite/microcontrollers) — On-device ML runtime for edge inference
13. [Ambiq Apollo4 Blue Plus](https://ambiq.com/apollo4-blue-plus/) — Ultra-low-power MCU for wearable applications
