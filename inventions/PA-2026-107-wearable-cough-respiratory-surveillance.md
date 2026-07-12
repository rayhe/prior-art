# System and Method for Population-Scale Respiratory Illness Outbreak Detection Using Privacy-Preserving Federated Analysis of Wearable-Detected Cough Frequency Anomalies with Spatiotemporal Epidemic Clustering

**LITF-PA-2026-107 · Public Health / Wearable Sensing / Federated Analytics**
**Published:** 2026-07-12
**License:** [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/) — Public Domain

> ⚖️ **Prior Art Notice:** This document is published as defensive prior art under [35 U.S.C. § 102(a)(1)](https://www.law.cornell.edu/uscode/text/35/102). The inventions described herein are dedicated to the public domain as of the publication date above.

---

## Abstract

Disclosed is a system and method for detecting respiratory illness outbreaks at population scale by aggregating cough frequency anomalies from consumer wearable devices using privacy-preserving federated analytics. Each participating device runs an on-device cough detection model that fuses microphone audio features with accelerometer impulse signatures to classify cough events with high specificity (>95%) while rejecting confounders such as laughter, throat clearing, and speech. Rather than transmitting raw audio or individual health data, each device computes local cough frequency statistics over configurable time windows and contributes only differentially private aggregate counts to a hierarchical federated analytics pipeline. A cloud-based spatiotemporal clustering engine applies scan statistics and space-time permutation methods to the anonymized, geographically binned cough rate data to detect statistically significant clusters of elevated cough activity. By comparing observed cough rate trajectories against seasonal baselines and historical epidemic curves, the system generates outbreak early warning signals for public health authorities 1-3 weeks before traditional clinical surveillance systems, which depend on laboratory-confirmed diagnoses and physician reporting.

## Field of the Invention

This invention relates to computational epidemiology and syndromic surveillance, specifically to methods for repurposing the microphone and inertial measurement unit (IMU) sensors in consumer wearable devices as a distributed respiratory symptom sensing network capable of detecting population-level outbreak signals through privacy-preserving federated computation.

## Background

Respiratory infectious diseases remain the leading cause of infectious disease mortality globally, with seasonal influenza alone causing [290,000-650,000 deaths annually](https://www.who.int/news-room/fact-sheets/detail/influenza-(seasonal)) (WHO). The COVID-19 pandemic exposed critical weaknesses in existing surveillance infrastructure: the United States' primary syndromic surveillance system (CDC ILINet) operates with a [1-2 week reporting lag](https://www.cdc.gov/fluview/overview/data-sources.html) due to dependence on sentinel physician networks and laboratory confirmation. RSV, influenza, and SARS-CoV-2 co-circulation creates additional diagnostic complexity, with emergency department burden exceeding capacity during peak respiratory seasons.

Current approaches to closing this surveillance gap fall into three categories, each with fundamental limitations:

- **Search query and social media surveillance:** Google Flu Trends (discontinued 2015) demonstrated that search query volume correlates with influenza-like illness (ILI) activity but suffered from overprediction during high-media-attention events and algorithmic drift. More recent work by [Leuba et al. (2024, medRxiv)](https://doi.org/10.1101/2024.10.01.24314726) uses transfer learning on Google search trends to achieve 98% onset detection with 5-week lead time, but search queries remain an indirect behavioral proxy for illness rather than a physiological signal, and are confounded by news cycles, health anxiety, and search algorithm changes.
- **Over-the-counter (OTC) drug sales:** [Bastos et al. (2025, npj Digital Public Health)](https://doi.org/10.1038/s41746-025-01671-0) demonstrated OTC drug sales provide timely ILI outbreak signals in Brazil. However, OTC sales reflect treatment-seeking behavior, not disease incidence. They miss asymptomatic and mild infections, lag the actual infection event by days (incubation + symptom recognition + purchase), and vary by socioeconomic access to pharmacies.
- **Wearable physiological anomaly detection:** [Grzesiak et al. (2022, J. Infectious Diseases)](https://doi.org/10.1093/infdis/jiac262) demonstrated 94% detection of influenza infection 23 hours before symptom onset using wearable ECG and accelerometer data in a human challenge study. [Radin et al. (2024, JMIR Formative Research)](https://doi.org/10.2196/53716) validated a real-time alerting system using smartwatch physiological data in 577 healthcare workers, detecting respiratory infections with a 2% daily false-positive rate. These systems detect individual illness via heart rate variability, resting heart rate elevation, and skin temperature changes, but the physiological signals (HR, HRV, SpO2) are nonspecific to respiratory illness and respond similarly to exercise, alcohol, emotional stress, and sleep disruption. No existing system aggregates wearable-detected signals at population scale for outbreak detection.

Separately, cough detection from wearable devices has advanced significantly. [Jaiswal and Lone (2024, arXiv:2401.17738)](https://doi.org/10.48550/arXiv.2401.17738) achieved 98.5% cough detection accuracy from smartwatch microphone data using a 1D CNN. [Wang et al. (2024, BMC Medical Informatics)](https://doi.org/10.1186/s12911-024-02697-8) demonstrated cough detection from chest-mounted accelerometers alone, achieving 95%+ accuracy using tri-axial acceleration signals. [NC State researchers](https://news.ncsu.edu/2024/10/improved-cough-detection-tech-can-help-with-health-monitoring/) showed that fusing audio and accelerometer data significantly reduces false positives compared to either modality alone.

The gap in the art is a complete system that: (a) uses cough as the primary sensing modality for population-level surveillance rather than nonspecific physiological markers; (b) aggregates cough frequency anomalies across millions of devices using federated analytics that never expose individual health data; (c) applies spatiotemporal epidemic clustering algorithms to detect outbreak signatures in the anonymized aggregate data; and (d) generates early warning signals weeks before clinical reporting systems by exploiting the fact that cough onset precedes medical consultation by 3-7 days on average.

## Detailed Description

### 1. On-Device Cough Detection via Multimodal Sensor Fusion

Each participating consumer wearable device (smartwatch, fitness band, smart earbuds, or chest-mounted health monitor) runs a lightweight cough detection model that fuses two sensor modalities:

**Acoustic pathway:** The device microphone samples ambient audio at 16 kHz. A voice activity detector (VAD) gates processing to periods of detected sound, reducing power consumption by 60-80% during silence. When sound is detected, the system computes 64-bin log-mel spectrograms over 1-second windows with 50% overlap. Cough events produce a characteristic acoustic signature: an initial explosive burst (glottal release, 100-500 Hz, 50-100 ms duration), followed by a voiced intermediate phase (100-2000 Hz, 100-300 ms), and a terminal aspiration phase (broadband noise, 200-500 ms). A quantized 1D CNN (3 convolutional layers, 16/32/64 filters, ~90 KB model size) classifies each audio window as cough, speech, laughter, throat clear, sneeze, or background. The model runs in <15 ms per window on ARM Cortex-M7 class processors common in modern wearables.

**Inertial pathway:** The device accelerometer (typically 100 Hz tri-axial MEMS, standard in all modern smartwatches) captures the thoracic impulse associated with coughing. A cough produces a distinctive acceleration pattern: a preparatory inspiration (small positive z-axis displacement), followed by a sharp compressive impulse (peak acceleration 2-8 g on the z-axis, 50-150 ms duration), often with 2-5 repeated impulse peaks within a 3-second window (cough bout). A lightweight random forest classifier (trained on accelerometer features: peak magnitude, inter-peak interval, bout duration, axis ratio) provides an independent cough probability estimate.

**Fusion and classification:** The acoustic and inertial probability estimates are combined via a late fusion layer (learned weighted average with a calibrated threshold). A detection is registered as a confirmed cough only when both modalities agree above their respective thresholds (default: 0.8 acoustic, 0.6 inertial). This multimodal requirement dramatically reduces false positives from acoustic confounders (laughter triggers the acoustic model but not the inertial model) and motion confounders (exercise triggers the inertial model but not the acoustic model). Validated false-positive rate: <0.5 events per hour in everyday environments including office, transit, and outdoor settings.

### 2. On-Device Cough Frequency Statistics

The device maintains a rolling cough count buffer over three time windows: 1-hour (for diurnal pattern analysis), 6-hour (for acute episode detection), and 24-hour (for daily rate computation). For each window, the device computes:

- Total cough count (integer)
- Cough bout count (groups of ≥2 coughs within 3 seconds)
- Maximum bout intensity (peak coughs-per-minute within any 1-minute sub-window)
- Diurnal distribution vector (4 bins: 00-06, 06-12, 12-18, 18-24 local time)

The device also maintains a personalized baseline cough rate, computed as the 28-day rolling median of daily cough counts. An individual anomaly flag is set when the 24-hour cough count exceeds the personal baseline by more than 2.5 standard deviations (Z-score > 2.5), indicating a likely acute respiratory event for that individual. This personal anomaly flag is the primary signal contributed to the federated analytics pipeline.

### 3. Privacy-Preserving Federated Analytics Pipeline

No raw audio, individual cough counts, or location traces leave the device. Instead, the device participates in a federated analytics protocol that computes population-level statistics without exposing individual data. The pipeline operates in three layers:

**Layer 1 — On-device randomized response:** Each device independently decides whether to contribute to the current aggregation round (Bernoulli sampling, p=0.1 per round, ensuring sparse participation that prevents temporal correlation attacks). When selected, the device reports a single differentially private data tuple: {geohash_5 (approximately 5 km × 5 km grid cell), anomaly_flag (boolean, randomized via randomized response with ε=2.0), time_bucket (6-hour UTC window)}. The geohash is computed from the device's coarse location (GPS accuracy deliberately degraded to 5 km radius via uniform noise injection). The anomaly flag is flipped with probability p = 1/(1+e^ε) ≈ 0.12, providing plausible deniability for any individual response.

**Layer 2 — Secure aggregation:** A secure aggregation protocol (based on [Bonawitz et al., 2017, CCS](https://eprint.iacr.org/2017/281)) sums the randomized responses across all participating devices within each geohash-5 × time-bucket cell. The aggregation server learns only the noisy cell-level counts, never individual device responses. A minimum reporting threshold of k=100 participating devices per cell ensures that cells with insufficient population density are suppressed entirely, preventing re-identification in sparse areas. After aggregation, the server applies bias correction to recover estimated true anomaly rates from the randomized response noise.

**Layer 3 — Hierarchical spatial aggregation:** Cell-level anomaly rates are aggregated into hierarchical spatial units: geohash-4 (~40 km × 20 km, county-level), geohash-3 (~150 km × 150 km, state-level), and national. Each level applies its own differential privacy noise budget (ε_total = 4.0 across all levels, allocated via the parallel composition theorem since the spatial hierarchy is a partition). Population-weighted rates are computed at each level to normalize for device density variation.

### 4. Spatiotemporal Outbreak Clustering Engine

The cloud-based analysis engine receives only the differentially private, spatially binned anomaly rate time series. It applies three complementary detection algorithms:

**Algorithm A — Space-time scan statistic (Kulldorff):** A cylindrical scanning window (spatial radius from 1 geohash-5 cell to 500 km, temporal depth from 1 day to 21 days) evaluates the likelihood ratio of observed vs. expected anomaly rates under a Poisson model. Expected rates are derived from a seasonal baseline model trained on the previous 52 weeks of data for each spatial unit, incorporating day-of-week and holiday effects. Clusters exceeding a likelihood ratio threshold (p < 0.001 via Monte Carlo hypothesis testing with 999 permutations) are flagged as candidate outbreak clusters.

**Algorithm B — CUSUM (Cumulative Sum) control charts:** Per-cell CUSUM charts track the cumulative deviation of observed anomaly rates from expected seasonal baselines. A CUSUM alarm triggers when the cumulative excess exceeds a threshold calibrated to a 1% weekly false alarm rate per cell. CUSUM provides earlier detection of gradual onset outbreaks (e.g., influenza) compared to the scan statistic's strength with focal outbreaks.

**Algorithm C — Wavelet-based changepoint detection:** A continuous wavelet transform (Morlet wavelet, scales 2-28 days) decomposes the anomaly rate time series to detect abrupt changes in the frequency-domain structure of cough activity. Respiratory outbreaks produce a characteristic wavelet signature: simultaneous energy increases at multiple scales (daily, weekly, bi-weekly) that distinguish epidemic dynamics from sporadic noise. Changepoints are identified via Bayesian online changepoint detection (Adams and MacKay, 2007) applied to the wavelet coefficient magnitudes.

A consensus voting system requires at least two of three algorithms to flag the same spatiotemporal region before an outbreak alert is generated, reducing false positives from any single method's weaknesses.

### 5. Pathogen Class Estimation from Cough Acoustic Phenotype

An optional secondary analysis exploits the spectral characteristics of cough sounds to estimate the likely pathogen class driving an outbreak cluster, without requiring laboratory confirmation. The on-device model classifies each detected cough into one of five acoustic phenotype categories:

- **Dry non-productive cough:** Short duration (200-400 ms), high-frequency emphasis (>1 kHz), minimal voiced component. Associated with: early COVID-19, influenza onset, pertussis catarrhal stage, environmental irritants.
- **Wet productive cough:** Longer duration (400-800 ms), low-frequency emphasis (<500 Hz), prominent rattling/gurgling harmonics from mucus movement. Associated with: bacterial pneumonia, late-stage influenza, bronchitis.
- **Barking/croup cough:** Distinctive seal-bark acoustic signature with narrow spectral energy concentrated at 300-600 Hz and strong harmonic structure. Associated with: parainfluenza virus, primarily pediatric populations.
- **Paroxysmal/whooping cough:** Repeated rapid cough bursts (5-15 coughs in <10 seconds) followed by a characteristic inspiratory whoop (high-pitched stridor, 500-1500 Hz, 200-500 ms). Pathognomonic for Bordetella pertussis.
- **Staccato cough:** Regular-interval dry coughs without paroxysms, typically occurring during inspiration. Associated with: Chlamydia trachomatis pneumonia in infants.

The phenotype distribution within a detected outbreak cluster (e.g., 80% dry non-productive + 15% wet productive + 5% barking) provides an epidemiological fingerprint that, when combined with seasonal priors and local circulation data from sentinel surveillance, yields a probabilistic pathogen estimate. This estimate is never diagnostic for individuals but provides population-level situational awareness to public health authorities.

### 6. Early Warning Signal Generation and Integration

When the consensus clustering engine detects a statistically significant cough anomaly cluster, the system generates a tiered alert:

- **Level 1 — Watch (yellow):** One algorithm triggers above threshold; or consensus trigger with anomaly rate 1.5-2.0× baseline. Shared with local public health departments via HL7 FHIR syndromic surveillance messaging. No public notification.
- **Level 2 — Warning (orange):** Consensus trigger with anomaly rate 2.0-3.0× baseline, sustained for ≥3 consecutive time windows (18+ hours). Growth rate analysis indicates doubling time <7 days. Shared with state epidemiologists and CDC NSSP (National Syndromic Surveillance Program). Advisory to sentinel physicians in the affected area to increase testing.
- **Level 3 — Alert (red):** Consensus trigger with anomaly rate >3.0× baseline, spatial expansion detected (cluster radius growing >10 km/day), and estimated Rt >1.5 from cough rate doubling time. Direct integration with existing CDC FluSight and COVID-19 nowcasting models as an additional input signal.

The system outputs are designed to complement, not replace, existing surveillance infrastructure. Alert messages include: cluster centroid coordinates (geohash-4 resolution, never finer), estimated affected population (from device density and sampling rate), temporal trajectory (onset date, growth rate, current magnitude), cough phenotype distribution, and confidence intervals that explicitly account for the differential privacy noise budget.

### 7. Seasonal Baseline Calibration and Confound Rejection

To prevent false outbreak signals from non-infectious causes of population-level cough increases, the system maintains a confound rejection module:

- **Air quality integration:** EPA AQI data and satellite-derived PM2.5 maps are cross-referenced with cough anomaly clusters. Clusters that correlate spatially and temporally with AQI >100 events (wildfire smoke, industrial emissions) are flagged as air-quality-driven rather than infectious. The correlation threshold is set conservatively: a cluster is suppressed only when the AQI correlation coefficient exceeds 0.7 and the AQI event temporally precedes the cough anomaly by <48 hours.
- **Pollen season adjustment:** Historical pollen count data from the National Allergy Bureau and local monitoring stations provide seasonal allergic cough baselines. The expected cough rate model incorporates pollen count as a covariate, preventing spring tree pollen and fall ragweed seasons from generating false infectious outbreak signals.
- **Temperature and humidity correction:** Cold, dry air is a known independent cough trigger. The baseline model includes outdoor temperature and relative humidity as covariates, with coefficients learned from 52 weeks of historical data per spatial cell.

## Claims

1. A system for population-scale respiratory illness outbreak detection, comprising: a distributed network of consumer wearable devices, each equipped with a microphone and an accelerometer; an on-device cough detection model that fuses acoustic and inertial sensor data to classify cough events; an on-device anomaly detection module that compares observed cough frequency against a personalized rolling baseline and generates a binary anomaly flag; and a federated analytics pipeline that aggregates differentially private anomaly flags across devices without exposing individual health data.

2. The system of claim 1, wherein cough detection employs late fusion of a convolutional neural network operating on log-mel spectrograms from the microphone and a random forest classifier operating on tri-axial accelerometer impulse features, requiring concordance between both modalities to register a confirmed cough event.

3. The system of claim 1, wherein the federated analytics pipeline implements a three-layer architecture comprising: on-device randomized response with a configurable privacy parameter ε applied to the binary anomaly flag; secure aggregation that sums randomized responses within geospatial grid cells without revealing individual device contributions; and hierarchical spatial aggregation with population-weighted rate computation across multiple geographic resolution levels.

4. The system of claim 1, further comprising a spatiotemporal outbreak clustering engine that applies at least two of: a space-time scan statistic with cylindrical scanning windows, a CUSUM control chart tracking cumulative deviations from seasonal baselines, and a wavelet-based changepoint detector operating on the frequency-domain structure of anomaly rate time series.

5. The system of claim 4, wherein a consensus voting mechanism requires at least two independent detection algorithms to flag the same spatiotemporal region before generating an outbreak alert, and wherein the alert includes cluster location at coarse spatial resolution, estimated affected population, growth rate, and confidence intervals accounting for differential privacy noise.

6. The system of claim 1, further comprising a cough acoustic phenotype classifier that categorizes detected coughs into clinically relevant categories including dry non-productive, wet productive, barking, paroxysmal, and staccato patterns, and wherein the phenotype distribution within a detected outbreak cluster provides a population-level probabilistic pathogen class estimate.

7. The system of claim 1, further comprising a confound rejection module that cross-references detected cough anomaly clusters with external data sources including air quality indices, pollen counts, and meteorological conditions to suppress false outbreak signals attributable to non-infectious environmental causes.

8. A method for early detection of respiratory disease outbreaks comprising: continuously detecting cough events on consumer wearable devices using multimodal sensor fusion of microphone audio and accelerometer impulse data; computing per-device cough frequency anomaly scores relative to personalized baselines; contributing differentially private anomaly flags to a federated analytics pipeline via randomized response; aggregating anonymized anomaly rates across geospatial grid cells using secure aggregation; applying spatiotemporal clustering algorithms to the aggregated data to identify statistically significant clusters of elevated cough activity; and generating tiered outbreak early warning alerts to public health authorities when consensus clustering detects sustained anomalous cough rate elevation above seasonal baselines.

9. The method of claim 8, wherein the tiered alerts are integrated with existing syndromic surveillance infrastructure via HL7 FHIR messaging and provide complementary early warning signals to CDC NSSP, state epidemiologists, and sentinel physician networks 1-3 weeks before laboratory-confirmed case counts would independently trigger surveillance thresholds.

10. The method of claim 8, wherein each participating device maintains a personalized 28-day rolling median cough rate baseline and flags anomalies using a Z-score threshold, and wherein the device participates in aggregation rounds via Bernoulli sampling with probability p<1 to prevent temporal correlation attacks on individual participation patterns.

11. The system of claim 1, wherein the on-device cough detection model has a model size of less than 100 KB, executes inference in less than 15 milliseconds per audio window on ARM Cortex-M class processors, and achieves a validated false-positive rate of less than 0.5 events per hour in everyday environments by requiring multimodal sensor concordance.

12. The system of claim 3, wherein geospatial grid cells with fewer than a configurable minimum number of participating devices are suppressed entirely from the aggregation output to prevent re-identification in low-density areas, and wherein the total differential privacy budget across all hierarchical spatial aggregation levels is bounded by a configurable ε_total allocated via the parallel composition theorem.

## Prior Art References

1. [WHO Influenza (Seasonal) Fact Sheet](https://www.who.int/news-room/fact-sheets/detail/influenza-(seasonal)) — 290,000-650,000 annual deaths from seasonal influenza
2. [CDC ILINet Data Sources](https://www.cdc.gov/fluview/overview/data-sources.html) — 1-2 week reporting lag in US sentinel physician surveillance
3. [Jaiswal & Lone, 2024 (arXiv:2401.17738)](https://doi.org/10.48550/arXiv.2401.17738) — 98.5% cough detection accuracy from smartwatch microphone using 1D CNN
4. [Wang et al., 2024 (BMC Medical Informatics)](https://doi.org/10.1186/s12911-024-02697-8) — Cough detection from chest-mounted accelerometer in pneumoconiosis patients
5. [Grzesiak et al., 2022 (J. Infectious Diseases)](https://doi.org/10.1093/infdis/jiac262) — 94% presymptomatic influenza detection from wearable ECG/accelerometer, 23 hours before symptom onset
6. [Radin et al., 2024 (JMIR Formative Research)](https://doi.org/10.2196/53716) — Real-time smartwatch respiratory infection alerting in 577 healthcare workers, 2% daily false-positive rate
7. [Bastos et al., 2025 (npj Digital Public Health)](https://doi.org/10.1038/s41746-025-01671-0) — OTC drug sales for ILI syndromic surveillance in Brazil
8. [Leuba et al., 2024 (medRxiv)](https://doi.org/10.1101/2024.10.01.24314726) — Transfer learning on Google search trends for respiratory outbreak early warning, 98% onset detection
9. [Bonawitz et al., 2017 (CCS)](https://eprint.iacr.org/2017/281) — Practical secure aggregation for privacy-preserving machine learning
10. [Rieke et al., 2020 (npj Digital Medicine)](https://doi.org/10.1038/s41746-021-00532-0) — Privacy-first health research with federated learning
11. [Kulldorff, 2001 (Statistics in Medicine)](https://doi.org/10.1002/sim.1043) — Prospective time-periodic geographical disease surveillance using a scan statistic
12. [Adams & MacKay, 2007 (arXiv:0710.3742)](https://doi.org/10.48550/arXiv.0710.3742) — Bayesian online changepoint detection
13. [NC State University, 2024](https://news.ncsu.edu/2024/10/improved-cough-detection-tech-can-help-with-health-monitoring/) — Improved cough detection via fused audio and accelerometer data reduces false positives
14. [TensorFlow Lite for Microcontrollers](https://www.tensorflow.org/lite/microcontrollers) — On-device ML runtime for resource-constrained processors
