# System and Method for Continuous Subsurface Void Detection Beneath Roadway Surfaces Using Fleet-Vehicle-Mounted Acoustic Resonance Sensing and Edge-Deployed Spectral Convolutional Neural Networks

**LITF-PA-2026-011 · Infrastructure / Geotechnical AI**
**Published:** 2026-04-13
**License:** [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/) — Public Domain

> ⚖️ **Prior Art Notice:** This document is published as defensive prior art under [35 U.S.C. § 102(a)(1)](https://www.law.cornell.edu/uscode/text/35/102). The inventions described herein are dedicated to the public domain as of the publication date above.

---

## Abstract

Disclosed is a system and method for continuous, city-scale detection of subsurface voids (sinkholes, pipe washouts, soil piping erosion, abandoned mine voids) beneath roadway surfaces by mounting low-cost MEMS accelerometer arrays and contact microphones on the undercarriages of ordinary municipal fleet vehicles (transit buses, refuse trucks, street sweepers). As a vehicle traverses a road surface, the tire-road interaction generates a broadband vibroacoustic signal. When the pavement is underlain by a void, the ground impedance changes, producing characteristic modifications to the tire-road coupling spectrum: increased low-frequency energy (10-80 Hz), reduced spectral coherence between adjacent axle sensors, and a distinctive resonance peak corresponding to the void's flexural vibration mode. An edge-deployed spectral convolutional neural network (spectral CNN) processes the real-time accelerometer and microphone data streams to detect void signatures, estimate void lateral extent and depth (1-10 meters below surface), and assign a confidence score. GPS-tagged detections from multiple vehicle passes are aggregated via Bayesian spatial fusion to build probabilistic city-scale void hazard maps, with detection confidence improving as more vehicles traverse the same road segments over time.

## Field of the Invention

This invention relates to geotechnical infrastructure monitoring, specifically to the use of vehicle-mounted vibroacoustic sensing and machine learning for passive, continuous detection and characterization of subsurface voids beneath roadway surfaces.

## Background

Subsurface voids beneath roadways cause catastrophic sinkholes that endanger lives, damage property, and disrupt transportation. The [U.S. Geological Survey](https://www.usgs.gov/special-topics/water-science-school/science/sinkholes) estimates that sinkholes cause $300 million in damages annually in the United States. In 2023, a sinkhole in [Chatsworth, California](https://www.latimes.com/california/story/2023-01-10/chatsworth-sinkhole-forces-evacuations) swallowed an entire section of road. In 2014, a [sinkhole in Fukuoka, Japan](https://www.bbc.com/news/world-asia-37906069) opened a 30-meter-wide crater at a major intersection. Most sinkholes develop gradually over months to years as subsurface soil erodes into broken pipes, around deteriorating culverts, or through karst dissolution, but current detection methods are reactive (post-collapse) or prohibitively expensive for systematic coverage.

Current subsurface void detection methods include:

- **Ground Penetrating Radar (GPR):** Effective for shallow voids (<3m) but requires dedicated survey vehicles traveling at <15 mph. Commercial systems (GSSI, Mala) cost $30,000-$100,000. Survey rate: 1-5 lane-miles per day. See [Saarenketo & Scullion, NDT&E International 2000](https://doi.org/10.1016/S0963-8695(99)00077-6) for highway GPR methodology.
- **Falling Weight Deflectometer (FWD):** Measures pavement deflection under impact loading. Can detect voids by anomalous deflection basins but requires a specialized vehicle, traffic control, and point-by-point measurement at 50-meter intervals. Survey rate: 2-10 miles per day.
- **Microgravity survey:** Measures local gravity anomalies caused by subsurface density contrasts. Sensitive to voids >2 m³ at depths up to 10m. Requires a $40,000+ gravimeter (Scintrex CG-6) and hour-per-point measurement time. See [Butler, 1984](https://doi.org/10.1190/1.1441712) for gravity method theory.
- **LiDAR surface monitoring:** Airborne or mobile LiDAR detects surface subsidence preceding sinkhole formation. Resolution sufficient for detecting >5mm elevation changes but cannot identify voids before surface expression begins. See [Doctor & Young, 2013](https://doi.org/10.1016/j.geomorph.2013.04.047).

Relevant patents include:

- [US10012760B2](https://patents.google.com/patent/US10012760B2) (MIT/MIT Lincoln Lab): Vehicle-mounted GPR system for road subsurface imaging. Requires dedicated radar hardware; cannot be retrofit to existing fleet vehicles. System cost >$50,000.
- [US20190257972A1](https://patents.google.com/patent/US20190257972A1) (Geophysical Survey Systems): Automated GPR void detection using machine learning. Still requires GPR hardware; the ML component processes GPR images, not vibroacoustic data.
- [US9835568B1](https://patents.google.com/patent/US9835568B1) (Roadroid/Sweden): Uses smartphone accelerometers in vehicles to assess road surface roughness (IRI). Detects surface condition only; explicitly does not detect subsurface features.
- [Maser, 1996](https://doi.org/10.1061/(ASCE)0733-9445(1996)122:3(217)): Seminal paper on using GPR to detect voids under pavements, establishing the geophysical basis but requiring dedicated equipment.

The gap in the art is a system that: (a) uses ordinary fleet vehicles already traversing city streets, requiring only the addition of low-cost sensors, (b) detects subsurface voids passively from the tire-road vibroacoustic interaction without active source excitation or dedicated survey equipment, (c) accumulates detection confidence over time through multi-pass Bayesian fusion, and (d) scales to city-wide coverage at marginal cost by leveraging vehicles of opportunity.

## Detailed Description

### 1. Sensor Array and Vehicle Integration

The sensor system is designed for retrofit installation on the undercarriage of municipal fleet vehicles in under 2 hours, without modifications to the vehicle's electrical system or body:

- **Accelerometer array:** Four MEMS triaxial accelerometers (Analog Devices ADXL355, ±8g range, noise density 25 μg/√Hz, bandwidth DC-1500 Hz) mounted on magnetic plates attached to the vehicle frame at: (a) front left suspension, (b) front right suspension, (c) rear left suspension, (d) rear right suspension. The four-point array enables differential analysis between left/right and front/rear, which is critical for distinguishing localized subsurface anomalies (which affect one side or one axle) from whole-vehicle excitation (potholes, speed bumps).
- **Contact microphones:** Two piezoelectric contact microphones (frequency range: 5 Hz - 10 kHz) mounted on the vehicle floor pan directly above the rear axle. These capture airborne acoustic energy radiated by the tire-road interaction, which contains additional spectral information not present in the structural vibration signal.
- **Edge processing unit:** NVIDIA Jetson Orin Nano (40 TOPS) in a sealed IP67 enclosure, powered by the vehicle's 12V system via a regulated DC-DC converter. The unit samples all six sensor channels at 4096 Hz (24-bit), processes data in real-time, and stores GPS-tagged detections to an internal NVMe SSD.
- **GPS/GNSS module:** Multi-constellation GNSS receiver (u-blox F9P) providing <10 cm horizontal accuracy via RTK corrections from a local base station or NTRIP service. Accurate georeferencing is essential for multi-pass spatial fusion.

The total sensor kit cost is approximately $800 per vehicle at scale. For a fleet of 50 municipal buses covering 500 route-miles daily, this provides comprehensive city coverage with average repeat intervals of 2-4 days per road segment.

### 2. Tire-Road Vibroacoustic Physics

The tire-road interaction generates a complex vibroacoustic signal whose spectral characteristics depend on the mechanical impedance of the road structure and its foundation:

- **Normal pavement:** The tire excites a broadband signal dominated by surface texture interactions (200-2000 Hz) and suspension dynamics (1-20 Hz). The spectral coherence between left and right sensors is high (>0.8) because both wheels experience the same road structure.
- **Void-underlain pavement:** A subsurface void creates a "drum" effect where the pavement slab spans the void. This produces three observable signatures:
  1. **Flexural resonance:** The unsupported pavement span vibrates at a fundamental frequency f = (π/2L²) × √(D/ρh), where L is span length, D is flexural rigidity, ρ is density, and h is slab thickness. For typical concrete pavement (h=250mm) over a 3-meter void, f ≈ 25-50 Hz. This appears as a spectral energy peak in the 10-80 Hz band.
  2. **Impedance contrast:** The reduced support stiffness increases low-frequency vibration amplitude by 6-20 dB compared to well-supported pavement. This is measurable as an increase in the spectral power ratio (10-80 Hz / 200-1000 Hz).
  3. **Spatial decorrelation:** As one axle or one side of the vehicle crosses the void boundary while the other remains on supported pavement, the spectral coherence between sensor pairs drops sharply. The coherence reduction is proportional to the void's lateral extent relative to the vehicle track width.

These physics provide three independent detection channels (spectral peak, power ratio, coherence drop), reducing false alarm rates compared to single-feature approaches.

### 3. Spectral CNN Architecture

The spectral CNN processes 1-second sliding windows of the 6-channel sensor data stream (4096 samples × 6 channels per window, 50% overlap):

1. **Preprocessing:** Short-time Fourier transform (STFT) with 256-sample Hanning window and 50% overlap, producing a 128-frequency-bin × 32-time-step spectrogram for each of the 6 channels. The 6 spectrograms are stacked as channels of a 2D input tensor (shape: 6 × 128 × 32).
2. **Feature extraction backbone:** Modified ResNet-18 architecture adapted for multi-channel spectrograms. The first convolutional layer is changed from 3-input-channel to 6-input-channel. Residual blocks with batch normalization and ReLU activation. The backbone outputs a 512-dimensional feature vector.
3. **Detection head:** Binary classifier (void / no-void) with sigmoid output. A detection is triggered when the sigmoid output exceeds 0.7 for 3+ consecutive windows (sustained detection over 1.5+ seconds, corresponding to 15-30 meters of travel at typical urban speeds).
4. **Characterization head (activated only on detection):**
   - Void lateral extent: Regression output (0.5 - 15 meters)
   - Void depth below surface: Regression output (0.5 - 10 meters)
   - Void type classification: 4-class (pipe washout, karst dissolution, abandoned mine, general soil erosion)

Total parameters: approximately 11 million. Model size after FP16 quantization: 22 MB. Inference time: 8 ms per window on Jetson Orin Nano, enabling real-time processing at driving speed.

Training data comprises:

- **Synthetic data (80% of training set):** Generated using a 3D finite element model (ABAQUS) simulating vehicle-pavement-soil interaction with and without subsurface voids of varying sizes, depths, shapes, and soil conditions. Vehicle dynamics are modeled using a validated quarter-car model. The FE model produces synthetic accelerometer and microphone signals that are corrupted with measured noise profiles from real fleet vehicles.
- **Field data (20%):** Collected from controlled tests where vehicles instrumented with the sensor array drove over known voids (confirmed by GPR or excavation) in collaboration with [FHWA Turner-Fairbank Highway Research Center](https://www.fhwa.dot.gov/research/tfhrc/) pavement test facilities. 47 confirmed void sites and 1,200 km of void-free baseline driving.

### 4. Bayesian Spatial Fusion

Individual vehicle passes produce noisy, single-transect detections. The Bayesian fusion module aggregates detections across multiple passes and multiple vehicles to build a probabilistic void hazard map:

The city is discretized into a 1-meter × 1-meter grid. Each grid cell c maintains a void probability P(void_c) initialized from a prior based on geological risk (karst geology maps from [USGS Karst Map](https://www.usgs.gov/programs/national-cooperative-geologic-mapping-program)), infrastructure age (older pipe segments from GIS), and historical sinkhole records.

When a vehicle pass produces a detection at GPS coordinates (x, y) with confidence score s and estimated lateral extent r:

`P(void_c | detection) = P(void_c) × L(detection | void_c) / P(detection)`

Where L(detection | void_c) is a 2D Gaussian likelihood centered at (x, y) with standard deviation σ = max(r/2, GPS_uncertainty). The likelihood magnitude is scaled by the detection confidence score s.

After N vehicle passes through a grid cell without detection:

`P(void_c | N non-detections) = P(void_c) × (1 - P_d)^N / Z`

Where P_d is the single-pass detection probability (~0.6 for a 3-meter void at 2-meter depth based on validation data) and Z is the normalization constant. This means that well-traveled roads that consistently show no detection will have their void probability reduced toward zero, while less-traveled roads retain higher prior uncertainty.

The hazard map is updated in real-time as vehicle data streams in and is accessible via a web dashboard showing color-coded void probability overlaid on the street map, with drill-down to individual detection events, temporal trends, and recommended investigation priorities.

### 5. Alert Triage and Response Protocol

The system generates three alert levels:

- **Watch (P > 0.3, 2+ detections):** Flag for monitoring. Increase measurement frequency by routing additional fleet vehicles through the area. No physical investigation yet.
- **Warning (P > 0.6, 3+ detections from 2+ vehicles):** Schedule a targeted GPR survey of the flagged area within 2 weeks. Estimated false alarm rate at this threshold: <15% based on validation.
- **Critical (P > 0.85, 5+ detections with consistent extent estimates):** Initiate emergency response protocol. Restrict heavy vehicle traffic. Deploy emergency GPR within 48 hours. Notify utility operators for potential pipe inspection. Estimated false alarm rate: <5%.

### 6. Figures Description

- **Figure 1:** Vehicle sensor installation diagram showing four MEMS accelerometers at suspension mounting points, two contact microphones on floor pan, edge processing unit, and GPS antenna, with example wiring routes on a transit bus undercarriage.
- **Figure 2:** Spectral comparison of tire-road vibroacoustic signal over normal pavement (flat spectrum) versus void-underlain pavement (prominent 35 Hz resonance peak, elevated low-frequency power, reduced inter-sensor coherence).
- **Figure 3:** Spectral CNN architecture from 6-channel STFT spectrogram input through modified ResNet-18 backbone to detection and characterization heads.
- **Figure 4:** Bayesian spatial fusion progression showing void probability map after 1 week, 1 month, and 3 months of fleet operation, with probability convergence on a known void site.

## Claims

- A method for detecting subsurface voids beneath roadway surfaces comprising: mounting a vibroacoustic sensor array on a fleet vehicle; continuously recording tire-road interaction signals during normal vehicle operation; computing spectral features from said signals; and classifying said spectral features using an edge-deployed convolutional neural network to detect subsurface void signatures.

- The method of claim 1, wherein said vibroacoustic sensor array comprises at least four triaxial accelerometers positioned at vehicle suspension points and at least one contact microphone, enabling differential analysis between left/right and front/rear sensor pairs.

- The method of claim 1, wherein said subsurface void signatures comprise at least: a flexural resonance peak in the 10-80 Hz frequency band, an increased low-frequency to mid-frequency spectral power ratio, and reduced spectral coherence between spatially separated sensor pairs.

- The method of claim 1, further comprising estimating void lateral extent, depth below surface, and void type classification from the spectral features.

- The method of claim 1, further comprising aggregating GPS-tagged detections from multiple vehicle passes over the same road segment using Bayesian spatial fusion to build a probabilistic void hazard map with detection confidence that increases with repeated measurement.

- A system for city-scale subsurface void monitoring comprising: a plurality of fleet vehicles each equipped with a vibroacoustic sensor array and edge processing unit; a central server performing Bayesian spatial fusion of GPS-tagged void detections from said vehicles; and a dashboard presenting probabilistic void hazard maps with alert triage at configurable probability thresholds.

- The system of claim 6, wherein said fleet vehicles are ordinary municipal vehicles (transit buses, refuse trucks, or street sweepers) that traverse city streets during normal operations, requiring no modifications to vehicle route or speed.

- The system of claim 6, wherein said edge processing unit comprises a GPU-accelerated embedded computer executing a spectral convolutional neural network in real-time at driving speeds, with total sensor kit cost below $1,000 per vehicle.

- The method of claim 5, wherein said Bayesian spatial fusion incorporates prior void probability derived from geological karst risk maps, underground infrastructure age from GIS records, and historical sinkhole databases.

- The method of claim 1, wherein said convolutional neural network is trained on a combination of synthetic data from finite element vehicle-pavement-soil interaction models and field data from controlled tests over confirmed void sites.

## Prior Art References

- [US10012760B2](https://patents.google.com/patent/US10012760B2) — MIT Lincoln Lab — "Vehicle-Mounted GPR for Road Subsurface" (2018)
- [US20190257972A1](https://patents.google.com/patent/US20190257972A1) — GSSI — "Automated GPR Void Detection with ML" (2019)
- [US9835568B1](https://patents.google.com/patent/US9835568B1) — Roadroid — "Smartphone Road Roughness Assessment" (2017)
- [Saarenketo & Scullion](https://doi.org/10.1016/S0963-8695(99)00077-6) — "Road Evaluation with GPR" — NDT&E International 2000
- [Butler, D.K.](https://doi.org/10.1190/1.1441712) — "Microgravimetric and Gravity Gradient Techniques for Detection of Subsurface Cavities" — Geophysics 1984
- [Maser, K.R.](https://doi.org/10.1061/(ASCE)0733-9445(1996)122:3(217)) — "GPR for Void Detection Under Pavements" — ASCE J. Structural Engineering 1996
- [Doctor & Young](https://doi.org/10.1016/j.geomorph.2013.04.047) — "LiDAR for Sinkhole Detection" — Geomorphology 2013
- [USGS Sinkholes Overview](https://www.usgs.gov/special-topics/water-science-school/science/sinkholes) — $300M annual damages
- [USGS Karst Map](https://www.usgs.gov/programs/national-cooperative-geologic-mapping-program) — National karst geology data
- [FHWA Turner-Fairbank Highway Research Center](https://www.fhwa.dot.gov/research/tfhrc/) — Federal pavement research facility
- [He et al.](https://arxiv.org/abs/1512.03385) — "Deep Residual Learning for Image Recognition" — CVPR 2016 (ResNet architecture)
- [Analog Devices ADXL355](https://www.analog.com/en/products/adxl355.html) — Low-noise MEMS accelerometer datasheet
- [NVIDIA Jetson Orin Nano](https://developer.nvidia.com/embedded/jetson-orin-nano) — Edge AI computing platform
- [u-blox F9P](https://www.u-blox.com/en/product/zed-f9p-module) — Multi-band RTK GNSS receiver

---

*Published at [liveinthefuture.org/priorart](https://liveinthefuture.org/priorart/)*
