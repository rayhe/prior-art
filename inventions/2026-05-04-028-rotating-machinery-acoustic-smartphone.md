# System and Method for Non-Contact Rotating Machinery Health Assessment Using Consumer Mobile Device Acoustic Emission Analysis and Cross-Domain Transfer Learning

**LITF-PA-2026-028 · Industrial IoT / Predictive Maintenance**
**Published:** 2026-05-04
**License:** [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/) — Public Domain

> ⚖️ **Prior Art Notice:** This document is published as defensive prior art under [35 U.S.C. § 102(a)(1)](https://www.law.cornell.edu/uscode/text/35/102). The inventions described herein are dedicated to the public domain as of the publication date above.

---

## Abstract

Disclosed is a system and method for assessing the mechanical health of rotating machinery using acoustic emissions captured by consumer mobile device microphones (smartphones and tablets) without physical contact or dedicated vibration sensors. The system performs on-device spectral decomposition of audio recorded at 44.1-48 kHz within a 5-30 second capture window, extracting envelope spectra, cepstral coefficients, and bearing characteristic frequency components from the acoustic signal. A cross-domain transfer learning framework pre-trains a convolutional neural network on the [Case Western Reserve University (CWRU) Bearing Data Center](https://engineering.case.edu/bearingdatacenter/download-data-file) vibration dataset (approximately 480 recordings across 4 fault conditions at 4 loads), then fine-tunes on a smaller acoustically-captured dataset using domain adaptation via maximum mean discrepancy (MMD) loss. The system classifies machinery health into five categories: healthy, inner race fault, outer race fault, rolling element fault, and mechanical imbalance, with estimated remaining useful life (RUL) computed via a survival analysis head. A federated learning protocol enables model improvement across deployments without transmitting raw audio off-device.

## Field of the Invention

This invention relates to predictive maintenance of industrial rotating machinery, specifically to non-contact acoustic health assessment using consumer mobile devices and machine learning techniques including transfer learning and federated on-device training.

## Background

Bearing failures account for [approximately 40-50% of all electric motor failures](https://ieeexplore.ieee.org/document/539550) (IEEE Motor Reliability Working Group, Thorsen & Dalva 1995). The global predictive maintenance market reached [$8.3 billion in 2024](https://www.marketsandmarkets.com/Market-Reports/predictive-maintenance-market-256636879.html) and is projected to exceed $47 billion by 2032 (MarketsandMarkets). Unplanned downtime in manufacturing costs an estimated [$50 billion per year in the United States alone](https://www.siemens.com/us/en/company/press/true-cost-downtime-2022.html) (Siemens/Senseye, 2022), with a single hour of downtime in automotive manufacturing costing $1.3-2.0 million.

Current rotating machinery monitoring approaches include:

- **Piezoelectric accelerometers:** The gold standard for vibration analysis. Industrial-grade sensors (e.g., [PCB Piezotronics 603C01](https://www.pcb.com/sensors-for-test-measurement/accelerometers/industrial-accelerometers)) cost $200-800 per channel, require wired or wireless mounting, and demand trained vibration analysts for interpretation. Deployment across a typical plant with 200+ motors requires $50,000-150,000 in sensors alone.
- **Wireless vibration sensors:** Battery-powered MEMS sensors (e.g., [SKF Enlight](https://www.skf.com/us/products/condition-monitoring-systems), [Emerson AMS](https://www.emerson.com/en-us/catalog/ams-asset-monitor)) reduce wiring cost but still require per-machine installation ($100-400/sensor), periodic battery replacement, and proprietary analytics platforms with annual licensing fees of $5,000-50,000.
- **Handheld vibration analyzers:** Portable instruments (e.g., [Fluke 810](https://www.fluke.com/en-us/product/vibration/vibration-testers/810) at ~$8,000, Brüel & Kjær VIBER X5) used during walkabout routes. Require capital investment, trained operators, and provide only periodic snapshots during scheduled rounds.
- **SCADA/PLC integration:** Large continuous process plants integrate motor current signature analysis (MCSA) via programmable logic controllers. Limited to permanently installed motors and requires electrical access to motor supply cables.

Acoustic-based machinery diagnostics have been explored in research. [Glowacz, Mechanical Systems and Signal Processing 2019](https://doi.org/10.1016/j.ymssp.2019.106587) demonstrated acoustic fault detection for electric motors using sound recorded at 1 meter distance, achieving 95.8% classification accuracy for three fault types. [Verma et al., Measurement 2020](https://doi.org/10.1016/j.measurement.2020.107919) compared accelerometer and microphone signals for bearing fault detection, confirming that acoustic signals contain diagnosable fault information but with lower signal-to-noise ratio than contact vibration measurements.

Transfer learning for machinery diagnostics has gained traction. [Li et al., ISA Transactions 2019](https://doi.org/10.1016/j.isatra.2019.01.025) applied domain adaptation between vibration datasets collected under different operating conditions, achieving cross-domain bearing fault diagnosis. [Wen et al., IEEE Trans. Industrial Electronics 2020](https://doi.org/10.1109/TIE.2019.2917413) demonstrated transfer from laboratory vibration data to field conditions using deep convolutional networks. Neither addressed the vibration-to-acoustic domain gap inherent in using consumer mobile microphones.

Existing patents in this space include:

- [US10823686B2](https://patents.google.com/patent/US10823686B2) (Augury Systems): Analyzes vibration and magnetic signals from dedicated IoT sensors for machinery diagnostics. Requires proprietary hardware sensors, not consumer devices.
- [US20200132571A1](https://patents.google.com/patent/US20200132571A1) (3M): Uses acoustic signals from dedicated microphone arrays permanently mounted near equipment. Does not address handheld consumer devices or transfer learning from vibration datasets.
- [US11521105B2](https://patents.google.com/patent/US11521105B2) (Uptake Technologies): Cloud-based predictive maintenance using operational data from SCADA systems. Does not use acoustic signals or consumer devices.

The gap in the art is a complete system that: (a) uses unmodified consumer mobile device microphones as the sole sensing modality, (b) bridges the vibration-to-acoustic domain gap via transfer learning to leverage existing large vibration fault datasets, (c) runs inference entirely on-device for latency and privacy, (d) improves continuously via federated learning without centralized data collection, and (e) provides actionable RUL estimates alongside fault classification.

## Detailed Description

### 1. Audio Acquisition Protocol

The user positions a consumer smartphone or tablet at a distance of 10-50 cm from the target machine's bearing housing or casing. The device microphone (typically MEMS, e.g., Knowles SPH0645LM4H or similar) records audio at the device's native sample rate (44.1 kHz or 48 kHz) for a configurable capture window of 5-30 seconds (default: 10 seconds). During capture, the system simultaneously records accelerometer data from the device's built-in IMU to reject hand-motion artifacts and to estimate the user's hand stability for signal quality scoring.

The acquisition module implements a real-time signal quality estimator that evaluates: (a) background noise floor via A-weighted sound pressure level in the 20-200 Hz band, rejecting captures above 75 dBA ambient; (b) signal stationarity via the Augmented Dickey-Fuller test on 1-second sub-windows, requiring p < 0.05 across all windows; (c) hand stability score via IMU RMS acceleration, flagging captures exceeding 0.5 m/s² RMS as potentially motion-contaminated. If quality criteria are not met, the user is prompted to re-record with guidance (move closer, reduce ambient noise, stabilize hand).

### 2. On-Device Feature Extraction Pipeline

Raw audio undergoes the following preprocessing chain, executed entirely on-device:

**Step 1: Bandpass filtering.** A 6th-order Butterworth filter isolates the 200 Hz to 20 kHz band, rejecting low-frequency structural rumble and power supply hum (50/60 Hz and harmonics) while preserving bearing fault characteristic frequencies.

**Step 2: Envelope analysis via Hilbert transform.** The analytic signal is computed using the Hilbert transform. The envelope (magnitude of the analytic signal) is extracted and downsampled to 4 kHz. This demodulates amplitude-modulated fault impulses from the carrier frequencies generated by shaft rotation and gear meshing, making bearing defect frequencies visible in the envelope spectrum.

**Step 3: Bearing characteristic frequency estimation.** Given user-provided or automatically estimated shaft rotational speed (via fundamental frequency detection from the audio), the system computes theoretical bearing defect frequencies using standard kinematic equations: Ball Pass Frequency Outer race (BPFO) = (n/2) × f_r × (1 − d/D × cos α); Ball Pass Frequency Inner race (BPFI) = (n/2) × f_r × (1 + d/D × cos α); Ball Spin Frequency (BSF) = (D/2d) × f_r × (1 − (d/D × cos α)²); and Fundamental Train Frequency (FTF) = (1/2) × f_r × (1 − d/D × cos α). Here n = number of rolling elements, d = ball diameter, D = pitch diameter, α = contact angle, and f_r = shaft rotational frequency. When bearing geometry is unknown, the system searches for spectral peaks at rational multiples of shaft frequency consistent with common bearing geometries (n ∈ {7,8,9,10,11,12,13}, d/D ∈ {0.2, 0.25, 0.3, 0.35, 0.4}).

**Step 4: Multi-representation feature tensor.** Three spectral representations are computed and stacked as a 3-channel image tensor (128×128 pixels) suitable for CNN input: (a) log-mel spectrogram (128 mel bins, 512-sample FFT, 256-sample hop); (b) envelope spectrum (linear frequency scale, 0-2 kHz, 128 bins); (c) cepstrum (real cepstrum from inverse FFT of log power spectrum, quefrency range 2-50 ms, 128 bins). This multi-representation tensor captures complementary diagnostic information: the mel spectrogram provides broadband acoustic context, the envelope spectrum isolates repetitive fault impulses, and the cepstrum highlights periodic patterns including gear mesh harmonics and bearing defect families.

### 3. Cross-Domain Transfer Learning Architecture

The core challenge is that the vast majority of publicly available machinery fault datasets contain vibration signals (accelerometer data), not acoustic signals (microphone data). The system addresses this domain gap via a two-stage transfer learning architecture:

**Stage 1: Pre-training on vibration data.** A ResNet-18 backbone (modified for 3-channel 128×128 input) is pre-trained on the [CWRU Bearing Data Center](https://engineering.case.edu/bearingdatacenter/download-data-file) dataset (~480 recordings, 4 fault types at 4 motor loads: 0-3 HP, sampling at 12 kHz and 48 kHz) and the [Paderborn University Bearing Dataset](https://mb.uni-paderborn.de/kat/forschung/datacenter/bearing-datacenter) (~32 experiments, artificial and real bearing damage, sampling at 64 kHz). Vibration signals are converted to the same 3-channel tensor representation (mel spectrogram + envelope spectrum + cepstrum) used for acoustic data. Pre-training achieves baseline fault classification on vibration data (expected accuracy: >98% on CWRU, >95% on Paderborn).

**Stage 2: Domain adaptation fine-tuning.** The pre-trained network is fine-tuned on a smaller acoustically-captured dataset (target: 2,000+ recordings across 50+ machine types) using a domain adaptation loss. The total loss function is: L_total = L_classification + λ × L_MMD, where L_classification is standard cross-entropy loss on labeled acoustic data, L_MMD is the [Maximum Mean Discrepancy](https://jmlr.org/papers/v13/gretton12a.html) (Gretton et al., JMLR 2012) between the feature representations of vibration and acoustic samples in the penultimate layer, computed using a Gaussian RBF kernel with bandwidth selected by the median heuristic, and λ is a domain adaptation weight annealed from 0 to 1.0 over the first 50 epochs using a sigmoid schedule. The MMD term encourages the network to learn domain-invariant features that capture fault physics regardless of whether the signal originated from a contact accelerometer or a non-contact microphone. An alternative implementation uses [Domain-Adversarial Neural Networks](https://arxiv.org/abs/1505.07818) (DANN, Ganin et al. 2016) with a gradient reversal layer.

**Model quantization:** The trained model is quantized to INT8 via post-training quantization (TensorFlow Lite) or dynamic quantization (PyTorch Mobile). Final model size: approximately 11 MB (ResNet-18 INT8). Inference time on a mid-range smartphone (Snapdragon 778G): <150 ms per capture.

### 4. Fault Classification and Severity Estimation

The network produces two outputs: (a) a 5-class softmax probability vector over {healthy, inner_race_fault, outer_race_fault, rolling_element_fault, imbalance}, and (b) a scalar severity score (0.0-1.0) from a regression head, trained on fault diameter labels from the CWRU dataset (0.007", 0.014", 0.021", 0.028" fault diameters mapped to 0.25, 0.5, 0.75, 1.0 severity). Classification confidence below 0.6 triggers an "inconclusive" result with guidance to re-record under better conditions.

Severity scores are mapped to maintenance urgency levels: 0.0-0.3 = "Monitor" (continue normal operation, re-assess in 30 days); 0.3-0.6 = "Plan" (schedule maintenance within 2 weeks); 0.6-0.8 = "Act" (maintenance required within 72 hours); 0.8-1.0 = "Critical" (stop machine, immediate inspection required).

### 5. Remaining Useful Life Estimation

For machines with longitudinal monitoring history (3+ captures over 30+ days), the system fits a Weibull survival model to the severity score trajectory. The Weibull hazard function h(t) = (β/η) × (t/η)^(β−1) is parameterized by shape β and scale η, estimated via maximum likelihood from the severity score time series. RUL is computed as the expected time until severity exceeds 0.8 (the "Critical" threshold), with 80% confidence intervals derived from the Fisher information matrix.

When fewer than 3 longitudinal data points exist, the system estimates RUL using a lookup table derived from published bearing life data ([SKF bearing life equations](https://www.skf.com/us/products/rolling-bearings/bearing-selection/bearing-selection-process/bearing-life), ISO 281:2007) indexed by fault type, severity, and estimated operating speed.

### 6. Rotational Speed Estimation from Audio

When shaft RPM is not provided by the user, the system estimates it from the acoustic signal via autocorrelation of the envelope signal. The autocorrelation function of the amplitude envelope exhibits peaks at the fundamental rotation period for machines with any asymmetry (residual imbalance, slight misalignment, keyway). The system searches for the highest autocorrelation peak in the 0.5-200 Hz range (corresponding to 30-12,000 RPM), with a confidence metric based on the peak-to-sidelobe ratio. If confidence is below 0.7, the user is prompted to enter RPM manually or hold a stroboscopic tachometer app (using the phone's flashlight) against the shaft.

### 7. Federated Learning Protocol

Model improvement occurs via federated learning without transmitting raw audio off-device. Each device computes local gradient updates from confirmed fault diagnoses (user feedback after maintenance confirms or denies the prediction). Gradient updates are aggregated using [Federated Averaging](https://arxiv.org/abs/1602.05629) (McMahan et al., 2017): each participating device trains for E=5 local epochs on its confirmed cases, compresses the gradient update via top-k sparsification (k=1% of parameters), and uploads the sparse update to a central aggregation server. The server averages updates from N participating devices and pushes the updated model quarterly.

Privacy mechanisms include: (a) gradient sparsification (99% of gradients zeroed) makes reconstruction of training data computationally infeasible; (b) differential privacy noise (Gaussian mechanism, ε=8.0, δ=10⁻⁵) is added to each gradient update before upload; (c) raw audio never leaves the device.

### 8. Machine Identity and History Tracking

Each monitored machine is assigned a unique identifier via one of: (a) QR code or NFC tag affixed to the machine, scanned before audio capture; (b) acoustic fingerprinting of the machine's steady-state spectral signature (a "voiceprint" based on the power spectral density contour), matched against a local database using cosine similarity with a threshold of 0.85; or (c) GPS + compass-derived spatial location within a plant floor plan. Acoustic fingerprinting enables automatic machine identification without any physical tagging, exploiting the fact that each machine produces a unique spectral signature determined by its bearing geometry, shaft speed, load, mounting, and wear state.

### 9. Figures Description

- **Figure 1:** System architecture showing smartphone audio capture, on-device feature extraction pipeline (bandpass → Hilbert envelope → multi-representation tensor), CNN inference, and federated learning update path.
- **Figure 2:** Three-channel feature tensor visualization for (a) healthy bearing, (b) outer race fault, (c) inner race fault, showing mel spectrogram, envelope spectrum, and cepstrum channels with diagnostic features highlighted.
- **Figure 3:** Cross-domain transfer learning architecture showing Stage 1 (vibration pre-training on CWRU/Paderborn) and Stage 2 (acoustic fine-tuning with MMD domain adaptation loss).
- **Figure 4:** Remaining useful life estimation showing severity score trajectory with Weibull survival fit and 80% confidence intervals for three example machines.
- **Figure 5:** Federated learning protocol showing local training on confirmed diagnoses, gradient sparsification, differential privacy noise injection, server aggregation, and quarterly model push.

## Claims

1. A system for non-contact health assessment of rotating machinery, comprising: a consumer mobile device with a built-in microphone and processor; a software application executing on the device that captures acoustic emissions from rotating machinery at a distance of 10-50 cm without physical contact; an on-device signal processing pipeline that computes a multi-representation feature tensor from the captured audio, the tensor comprising a mel spectrogram channel, an envelope spectrum channel derived from Hilbert transform demodulation, and a cepstrum channel; and a convolutional neural network classifier executing on the device that classifies machinery health status from the feature tensor.

2. The system of claim 1, wherein the convolutional neural network is trained via a two-stage transfer learning process: pre-training on vibration-domain bearing fault datasets using the same multi-representation feature tensor format, followed by fine-tuning on acoustically-captured data using a domain adaptation loss that minimizes the distributional distance between vibration-domain and acoustic-domain feature representations.

3. The system of claim 2, wherein the domain adaptation loss comprises Maximum Mean Discrepancy computed in the penultimate layer feature space using a Gaussian radial basis function kernel, with an adaptation weight annealed from zero to a target value over training epochs.

4. The system of claim 1, further comprising a bearing characteristic frequency estimation module that computes theoretical defect frequencies from shaft rotational speed and bearing geometry parameters, and searches the envelope spectrum for energy concentration at those frequencies and their harmonics to identify specific fault types.

5. The system of claim 4, wherein shaft rotational speed is estimated automatically from the audio signal via autocorrelation of the amplitude envelope, without requiring external tachometer input.

6. The system of claim 1, further comprising a remaining useful life estimation module that fits a Weibull survival model to longitudinal severity scores from multiple captures of the same machine over time, computing expected time to critical severity with confidence intervals.

7. The system of claim 1, further comprising a federated learning protocol wherein the device computes local gradient updates from user-confirmed fault diagnoses, applies gradient sparsification and differential privacy noise, and transmits compressed updates to a central aggregation server for model improvement without transmitting raw audio data off-device.

8. The system of claim 1, further comprising an acoustic fingerprinting module that identifies individual machines by comparing the steady-state power spectral density contour of the captured audio against a local database of previously recorded machine signatures, enabling automatic machine identification without physical tagging.

9. A method for cross-domain machinery fault diagnosis comprising: capturing an acoustic signal from rotating machinery using a consumer mobile device microphone; computing a multi-channel feature tensor comprising mel spectrogram, Hilbert envelope spectrum, and cepstral representations of the acoustic signal; classifying the feature tensor using a neural network that was pre-trained on vibration-domain machinery fault data and fine-tuned on acoustic-domain data using a domain adaptation loss function; and outputting a fault classification with severity score and maintenance urgency recommendation.

10. The method of claim 9, further comprising: rejecting low-quality captures based on ambient noise level, signal stationarity, and device motion stability assessed from built-in inertial measurement unit data; and prompting the user for recapture with specific guidance when quality criteria are not met.

## Implementation Notes

A reference implementation can be constructed using: TensorFlow Lite or PyTorch Mobile for on-device inference; the [CWRU Bearing Data Center](https://engineering.case.edu/bearingdatacenter/download-data-file) and [Paderborn University Bearing Dataset](https://mb.uni-paderborn.de/kat/forschung/datacenter/bearing-datacenter) for Stage 1 pre-training; [librosa](https://librosa.org/) (Python) or [aubio](https://aubio.org/) (C) for signal processing primitives; [Flower](https://flower.ai/) (federated learning framework) for the gradient aggregation protocol; and standard Android/iOS audio capture APIs (AudioRecord/AVAudioEngine) for microphone access.

The system is implementable on any smartphone manufactured after 2018 with a MEMS microphone capable of 44.1+ kHz sampling and a processor supporting INT8 neural network inference (ARM Cortex-A76 or later, Apple A12 or later, Qualcomm Snapdragon 670 or later).

## Prior Art References

1. [Thorsen & Dalva, IEEE 1995](https://ieeexplore.ieee.org/document/539550) — Motor failure analysis: 40-50% of failures are bearing-related
2. [MarketsandMarkets, 2024](https://www.marketsandmarkets.com/Market-Reports/predictive-maintenance-market-256636879.html) — Global predictive maintenance market: $8.3B (2024), projected $47B+ (2032)
3. [Siemens/Senseye, 2022](https://www.siemens.com/us/en/company/press/true-cost-downtime-2022.html) — Unplanned downtime costs U.S. manufacturers $50B/year
4. [Glowacz, MSSP 2019](https://doi.org/10.1016/j.ymssp.2019.106587) — Acoustic fault detection for electric motors (95.8% accuracy)
5. [Verma et al., Measurement 2020](https://doi.org/10.1016/j.measurement.2020.107919) — Microphone vs. accelerometer comparison for bearing fault detection
6. [Li et al., ISA Transactions 2019](https://doi.org/10.1016/j.isatra.2019.01.025) — Cross-domain bearing fault diagnosis via transfer learning
7. [Wen et al., IEEE Trans. Ind. Electronics 2020](https://doi.org/10.1109/TIE.2019.2917413) — Deep transfer learning for machinery fault diagnosis
8. [Case Western Reserve University Bearing Data Center](https://engineering.case.edu/bearingdatacenter/download-data-file) — Benchmark vibration fault dataset
9. [Paderborn University Bearing Dataset](https://mb.uni-paderborn.de/kat/forschung/datacenter/bearing-datacenter) — Real damage bearing vibration dataset
10. [Gretton et al., JMLR 2012](https://jmlr.org/papers/v13/gretton12a.html) — Maximum Mean Discrepancy kernel two-sample test
11. [Ganin et al., 2016](https://arxiv.org/abs/1505.07818) — Domain-Adversarial Training of Neural Networks
12. [McMahan et al., 2017](https://arxiv.org/abs/1602.05629) — Federated Averaging (communication-efficient learning)
13. [US10823686B2](https://patents.google.com/patent/US10823686B2) (Augury) — Dedicated sensor machinery diagnostics
14. [US20200132571A1](https://patents.google.com/patent/US20200132571A1) (3M) — Mounted microphone array equipment monitoring
15. [US11521105B2](https://patents.google.com/patent/US11521105B2) (Uptake) — Cloud SCADA predictive maintenance
16. [SKF Bearing Life Equations](https://www.skf.com/us/products/rolling-bearings/bearing-selection/bearing-selection-process/bearing-life) — ISO 281:2007 bearing life calculations
