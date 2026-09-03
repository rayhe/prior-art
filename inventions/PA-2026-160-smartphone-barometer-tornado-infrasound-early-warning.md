# PA-2026-160: Tornado Early Warning Using Distributed Smartphone Barometric Infrasound Sensing with Edge-Deployed Vortex Signature Classification and Multi-Device Triangulation

**Title:** System and Method for Tornado Early Warning Using Distributed Smartphone Barometric Infrasound Sensing with Edge-Deployed Vortex Signature Classification and Multi-Device Triangulation

**Filing:** LITF-PA-2026-160  
**Published:** September 3, 2026  
**Domain:** Severe Weather / Distributed Sensing / Edge AI  
**Full Disclosure:** [liveinthefuture.org/priorart/smartphone-barometer-tornado-infrasound-early-warning.html](https://liveinthefuture.org/priorart/smartphone-barometer-tornado-infrasound-early-warning.html)

---

## Abstract

Disclosed is a system and method for tornado early warning that repurposes the barometric pressure sensors in participating consumer smartphones as a distributed infrasound array. Tornadoes radiate infrasound with a fundamental frequency in the 0.5 to 10 Hz band, detectable tens of kilometers from the vortex, in some documented cases minutes before touchdown. During National Weather Service severe weather watches, participating phones switch their barometers into a high-rate sampling mode (25 to 100 Hz), run an on-device spectral classifier that distinguishes tornadic vortex signatures (a fundamental tone with linearly spaced overtones consistent with vortex-core resonance) from non-tornadic convection, wind buffeting, and indoor pressure artifacts, and transmit compact event reports containing spectral features, timestamps, and coarse location to a fusion service. The fusion service cross-correlates detections across multiple phones to estimate source bearing and range via time-difference-of-arrival, gates alerts against radar-derived rotation tracks to suppress false alarms, and issues targeted warnings to phones in the projected path. The system converts the existing installed base of barometer-equipped smartphones into a dense, zero-hardware-cost tornado sensor network.

## Technical Field

This invention relates to severe weather detection and public warning systems, specifically to the use of distributed consumer smartphone barometric sensors for infrasound-based tornado detection, on-device acoustic signature classification, multi-sensor source localization, and targeted alert dissemination.

## Background

Tornadoes kill an average of 70 to 80 people per year in the United States and cause approximately $3 billion in annual damage (NOAA Storm Prediction Center long-term averages). Warning performance remains limited: for the five-year period ending 2018, the national average tornado warning lead time was 8.6 minutes with a probability of detection of 59% and a false alarm rate of 70% (Elbing et al., NOAA repository). Radar-based detection struggles with tornadoes that form quickly, occur beyond effective radar range or below the radar horizon, or develop in regions with sparse low-level radar coverage such as the southeastern United States.

Tornado infrasound is a well-documented physical phenomenon. Tornadoes emit infrasound with a fundamental frequency consistently reported in the 0.5 to 10 Hz range (Bedard, 2005; reviewed in Allen et al., AMT 2022), where smaller vortex core diameters produce higher fundamental frequencies. Elbing et al. (2019) recorded a small Oklahoma tornado at 18.7 km range and observed a 75 dB spectral peak near 8.3 Hz, 18 dB above pre-tornado levels, with linearly spaced overtones at approximately 18, 29, 36, and 44 Hz; elevated infrasound began approximately 7 to 8 minutes before the verified tornado report. Infrasound array deployments during the spring 2018 severe weather season in the southeastern United States obtained accurate bearings on tornadoes at ranges exceeding 100 km, with the dominant band of coherent infrasound between 2 and 6 Hz (JASA, 2024). The empirical frequency-to-core-diameter relationship fn = (4n+5)c/4d (Abdullah, 1966) provides a physical basis for estimating vortex size from the measured fundamental.

Prior work demonstrates the sensing principle without building a warning system. The RedVox Infrasound Recorder app records infrasonic pressure via phone microphone and barometer and streams recordings to a cloud server for geophysical research; Sandia National Laboratories evaluated a Samsung S10 running RedVox as a low-cost infrasound sensor (Slad and Merchant, 2021). RedVox performs no automated tornado classification, no multi-device triangulation, and no alerting. PressureNet (University of Washington, Mass et al., 2013) crowdsources smartphone pressure at synoptic rates for numerical weather forecasting; it does not sample at infrasound rates and does not address tornadoes. Dedicated research infrasound arrays use purpose-built microbarometers costing thousands of dollars per station and cannot scale to neighborhood density.

The gap in the art is a complete automated warning system combining: (a) opportunistic high-rate phone barometer sampling during severe weather threats; (b) on-device real-time tornadic vortex classification rejecting wind noise and indoor artifacts; (c) multi-phone fusion localizing the vortex by time-difference-of-arrival; (d) radar rotation cross-checks controlling the false alarm rate; and (e) targeted alerts to people in the projected damage path.

## Detailed Description

### 1. Threat-Gated High-Rate Barometric Sampling

Two operating modes. In standby mode, the phone samples pressure at 1 Hz or less. When the phone's location falls inside an NWS tornado watch polygon, severe thunderstorm watch with tornado-possible tagging, or elevated-tornado-probability convective outlook, the system switches to surveillance mode: 25 to 100 Hz barometer output data rate for the watch duration plus a 30-minute buffer. A low-cost trigger computes short-term RMS pressure fluctuation energy in the 0.5 to 15 Hz band at 1 Hz cadence with integer-arithmetic IIR filters (under 1 mW); full spectral analysis engages only when band energy exceeds 6 dB above the trailing 10-minute noise floor. Building attenuation of 5 to 15 dB at 2 to 8 Hz is absorbed by the per-device adaptive threshold.

### 2. On-Device Vortex Signature Classifier

On trigger, the phone captures a 60-second rolling window and computes a 0.25 Hz-resolution power spectral density via Welch's method (8-second segments, 50% overlap, Hann window). Stage 1 (tonal comb detector) searches 0.5 to 12 Hz for a fundamental peak 10 dB above local background, then tests for overtones at linearly spaced multiples consistent with tornadic vortex resonance, scoring via the geometric mean of signal-to-background ratios at the fundamental and first three predicted overtones. Stage 2 (neural classifier): a compact 1D CNN (3 convolutional layers, 32/64/64 channels, kernel 7, global average pooling, 2-layer MLP; ~45,000 parameters, 180 KB INT8) takes the 0.5 to 50 Hz log-magnitude spectrum plus comb score and outputs P(tornadic vortex) vs P(clutter), with clutter classes covering gust-front rumble, HVAC cycling (rejected via 60 Hz mains harmonics), traffic, aircraft, indoor transients, and wind buffeting. Training combines published tornado infrasound recordings, the RedVox community storm archive, and synthetic vortex signatures through the Abdullah resonance model with randomized core diameters (30 to 1500 m) and ranges (5 to 150 km). Target: 90% recall at under 0.5 false triggers per phone per watch.

### 3. Accelerometer-Coherent Wind Noise Cancellation

A 32-tap adaptive LMS filter at 25 Hz models the transfer function from 3-axis acceleration magnitude to the pressure signal and subtracts the coherent component. Distant vortex infrasound produces negligible phone acceleration while local wind produces strongly correlated acceleration and pressure, so the canceller is expected to suppress wind-induced pressure variance by 8 to 14 dB while attenuating genuine distant infrasound by under 1 dB. Phones above a sustained motion threshold are down-weighted, not discarded, in fusion.

### 4. Multi-Device Coherence and TDOA Triangulation

Phones with classifier output above 0.8 transmit event reports: 64-bin spectral magnitudes (0.5 to 16 Hz, 8-bit log-compressed), classifier score, GPS truncated to ~100 m, cellular-disciplined timestamp (better than 100 ms), and device capability descriptor. The fusion service clusters reports in 5-minute sliding windows; clusters of 3+ phones within 60 km undergo pairwise spectral cross-correlation, and coherent clusters (mean correlation above 0.6, fundamental consistent within 0.5 Hz) proceed to localization. TDOA from sub-sample-interpolated cross-correlation peaks of short uploaded waveform snippets (2-second, 25 Hz, only within clusters) feeds hyperbolic multilateration for a maximum-likelihood source location with uncertainty ellipse; bearing-only fallback via spatial covariance eigendecomposition covers degenerate geometries.

### 5. Radar Fusion Gating

Infrasound source locations are cross-checked against MRMS rotation track / azimuthal shear data before public alerting. Tier 1 (confirmed): coherent cluster plus radar-indicated rotation (shear above 0.01 s^-1) within 5 km and 10 minutes triggers immediate targeted alert. Tier 2 (probable): persistent coherent tonal comb without radar rotation triggers advisory notification to emergency managers and low-urgency public heads-up within 15 km. Tier 3 (single-sensor): logged for forecaster review, no public alert. Requiring two independent physical observations addresses the documented 70% false alarm rate.

### 6. Projected-Path Targeted Alerting

On Tier 1 confirmation, a damage-path polygon is projected from the localized source using radar storm motion vectors (15-minute extrapolation, lateral uncertainty growing at 20% of forward distance, minimum half-width 1.5 km). Push alerts go to participating phones inside the polygon and to the Wireless Emergency Alert interface for affected counties, carrying estimated vortex location, Abdullah-relation core diameter class (small under 100 m, medium 100 to 500 m, large over 500 m), confidence, and protective-action guidance. Alerts refresh or cancel as the cluster evolves; 5 minutes of lost coherence triggers an all-clear.

### 7. Privacy-Preserving Architecture

No audio-band data leaves the phone. Only spectral feature vectors and sub-16 Hz waveform snippets (no intelligible content) are transmitted, and only after local detection. Locations truncated to 100 m; device identifiers rotate daily. Raw pressure series never uploaded except 2-second TDOA snippets within confirmed clusters, discarded after correlation.

## Claims

1. A system for tornado early warning, comprising: a plurality of consumer smartphones each containing a barometric pressure sensor; a surveillance mode switching the sensor into 25 to 100 Hz sampling when the phone's location falls within a severe weather watch polygon; an on-device spectral classifier evaluating 0.5 to 50 Hz pressure spectra for tornadic vortex signatures; and a fusion service aggregating multi-phone event reports and issuing targeted alerts.
2. The system of claim 1, wherein surveillance mode is gated by a low-power integer-arithmetic band-energy trigger engaging full spectral analysis only when 0.5 to 15 Hz energy exceeds an adaptive per-device threshold.
3. The system of claim 1, wherein the classifier comprises a tonal comb detector identifying a 0.5 to 12 Hz fundamental with linearly spaced overtones, and a neural classifier distinguishing tornadic vortex spectra from clutter including HVAC cycling identified by mains-frequency harmonics.
4. The system of claim 1, further comprising an accelerometer-coherent wind noise canceller adaptively subtracting the acceleration-correlated pressure component.
5. The system of claim 1, wherein the fusion service clusters event reports in sliding windows and admits only spectrally coherent clusters with consistent fundamental frequency to localization.
6. The system of claim 5, further comprising TDOA source localization via sub-sample-interpolated cross-correlation of short infrasound waveform snippets, yielding a maximum-likelihood vortex location with uncertainty ellipse.
7. The system of claim 1, further comprising radar fusion gating cross-checking infrasound locations against radar azimuthal shear and requiring coincident rotation before highest-urgency public alerts.
8. The system of claim 1, further comprising projected-path alerting extrapolating a damage-path polygon from the localized source using radar storm motion vectors.
9. The system of claim 1, wherein the classifier estimates vortex core diameter class from the measured fundamental via a vortex resonance frequency-to-diameter relationship included in the alert payload.
10. The system of claim 1, wherein no audio-band data leaves the phone, reports contain only spectral features with ~100 m location precision, and identifiers rotate at least daily.
11. A method for tornado early warning using consumer smartphones, comprising: threat-gated 25 to 100 Hz barometer sampling; adaptive energy triggering; on-device tonal-comb plus neural vortex classification; accelerometer-coherent wind cancellation; compact event reporting; cross-device coherence clustering; TDOA localization; radar gating; and projected-path warning dissemination.
12. The method of claim 11, further comprising tiered escalation: uncorroborated single-sensor detections logged without public alert; coherent clusters without radar corroboration escalated to advisory tier.

## Implementation Notes

Deployable as an OS-level emergency feature or background service in a weather app. Hardware: MEMS barometer with 25 Hz+ ODR and sub-1 Pa RMS noise in the 0.5 to 15 Hz band (Bosch BMP380/BMP390, STMicro LPS22HH, most flagship phones since ~2018); an estimated 60 to 70% of smartphones in the US tornado belt qualify. Power: surveillance-mode incremental drain under 2% per watch day. Density: 3+ reporting phones within ~60 km suffices; 1% participation in a typical 50,000-person county (~250 phones) meets the geometric requirement. Limitations: 5 to 15 dB indoor attenuation; simultaneous extreme wind noise at all nearby phones can mask signals; downwind-favoring asymmetric range. Radar gating ensures these degrade to missed advisories, not false alarms.

## Prior Art References

1. Allen et al., Atmospheric Measurement Techniques 15, 2022: Infrasound measurement system for real-time tornado measurements; 0.5 to 10 Hz fundamental band (Bedard, 2005). https://amt.copernicus.org/articles/15/2923/2022/amt-15-2923-2022.html
2. Elbing et al., JASA 146, 2019: 8.3 Hz peak 18 dB above background, linear overtones, elevated signal 7-8 min before verification. https://arxiv.org/abs/1809.00038v2
3. Elbing et al., NOAA repository: warning stats, 8.6 min lead, 59% POD, 70% FAR. http://repository.library.noaa.gov/view/noaa/27461/noaa_27461_DS1.pdf
4. JASA 2024: SE US arrays, bearings beyond 100 km, dominant coherent band 2-6 Hz. https://pubmed.ncbi.nlm.nih.gov/39302133/
5. AMS 2020: infrasound propagation, downwind ducting. https://ams.confex.com/ams/2020Annual/webprogram/Paper366123.html
6. RedVox Infrasound Recorder: passive smartphone infrasound recording. https://play.google.com/store/apps/details?id=io.redvox.InfraSoundRecorder&hl=en-US
7. Slad and Merchant, Sandia 2021: S10 + RedVox low-cost sensor evaluation. https://www.sandia.gov/research/publications/details/evaluation-of-low-cost-infrasound-sensor-packages-2021-10-01/
8. Mass et al., UW, 2013: PressureNet crowdsourced phone pressure. https://sciencedaily.com/releases/2013/02/130206141533.htm
9. Abdullah, A.J., 1966: vortex resonance relation fn = (4n+5)c/4d.
10. Frazier et al., 2014: 0.2-500 Hz recordings of three Oklahoma tornadoes; infrasound beamforming.

---

**License:** [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/) — Public Domain

> ⚖️ **Prior Art Notice:** This document is published as defensive prior art under [35 U.S.C. § 102(a)(1)](https://www.law.cornell.edu/uscode/text/35/102). The inventions described herein are dedicated to the public domain as of the publication date above.
