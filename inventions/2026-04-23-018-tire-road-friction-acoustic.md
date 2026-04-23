# LITF-PA-2026-018: System and Method for Continuous Road Surface Friction Coefficient Estimation Using Crowdsourced Vehicle Tire-Road Acoustic Emission, Wheel Speed Sensor Micro-Slip Analysis, and Self-Supervised Contrastive Learning for Real-Time Friction Mapping

**Filing:** LITF-PA-2026-018  
**Domain:** Automotive / Road Safety  
**Published:** April 23, 2026  
**License:** [CC0 1.0 — Public Domain](https://creativecommons.org/publicdomain/zero/1.0/)  
**HTML Version:** [liveinthefuture.org/priorart/tire-road-friction-acoustic.html](https://liveinthefuture.org/priorart/tire-road-friction-acoustic.html)

---

## Abstract

Disclosed is a system and method for continuously estimating road surface friction coefficients in real time by fusing two complementary signals already available in modern production vehicles: tire-road acoustic emission captured by cabin microphones (installed for active noise cancellation and voice assistants), and wheel speed sensor micro-slip ratios measured during normal straight-line and cornering maneuvers below the tire's saturation region. A self-supervised contrastive learning framework eliminates the need for labeled friction ground truth by exploiting a consistency constraint: multiple vehicles traversing the same road segment within a short temporal window should produce concordant friction estimates regardless of vehicle mass, tire compound, or speed. The system aggregates per-vehicle friction estimates into a spatiotemporal map tiled at 10-meter longitudinal resolution, updated at sub-minute latency, and served to navigation systems and autonomous driving stacks via a standardized API. Field-deployable without any new hardware on vehicles manufactured after 2020, the system converts the existing fleet into a distributed friction sensing network covering every road that vehicles drive on.

## Field of the Invention

This invention relates to automotive safety and road condition monitoring, specifically to methods for estimating tire-road friction coefficients using passive acoustic sensing and vehicle dynamics telemetry, combined with machine learning techniques that require no labeled training data, to generate real-time crowdsourced friction maps at scale.

## Background

Road surface friction is the single most important environmental variable governing vehicle stopping distance, cornering grip, and crash risk. The [Federal Highway Administration](https://www.fhwa.dot.gov/publications/research/safety/14078/index.cfm) estimates that wet pavement contributes to approximately 860,000 crashes annually in the United States, causing 4,700 fatalities and 338,000 injuries. Black ice alone accounts for roughly 1,300 deaths per year according to [FHWA road weather data](https://ops.fhwa.dot.gov/weather/q1_roadimpact.htm). Yet no production vehicle sold today can directly measure the friction coefficient of the road ahead.

Existing approaches to friction estimation fall into three categories, each with fundamental limitations:

- **Slip-based estimation during braking or hard cornering:** ABS and ESC systems estimate friction by measuring wheel slip during emergency maneuvers. [US8626454B2](https://patents.google.com/patent/US8626454B2/en) (Continental, 2014) discloses using ABS activation events. The problem is latency. You only learn the road is slippery after the driver has already lost traction. [Acosta et al. (2020)](https://doi.org/10.1016/j.ymssp.2019.106437) surveyed these methods and confirmed that slip-based estimators perform well above 40% tire utilization but are unreliable under normal driving conditions.
- **Dedicated friction measurement vehicles:** The [ASTM E274](https://www.astm.org/e0274-23.html) locked-wheel skid trailer and the [Grip Tester](https://doi.org/10.1061/JTEPBS.0000100) produce laboratory-grade friction measurements. Cost: $200,000-$500,000 per vehicle. Coverage: state DOTs typically test each lane-mile once every 2-3 years.
- **Camera and LiDAR-based surface classification:** [Roychowdhury et al. (IEEE T-ITS, 2021)](https://doi.org/10.1109/TITS.2020.3025674) demonstrated CNN-based road surface classification (dry, wet, snowy, icy) from forward-facing cameras. But classification is not quantification. Knowing a road is "wet" does not distinguish a friction coefficient of 0.5 (light rain on good asphalt) from 0.25 (standing water on polished concrete).

Two physical phenomena make passive friction estimation possible without triggering a slip event:

**Tire-road acoustic emission.** The contact patch between a tire and the road surface generates broadband noise through three mechanisms: air pumping in tread grooves (dominant at 800-2000 Hz), tread block snap-out (600-1200 Hz), and surface texture-induced vibration (200-1000 Hz). [Sandberg and Ejsmont (Applied Acoustics, 2018)](https://doi.org/10.1016/j.apacoust.2017.11.011) showed that the 800-1600 Hz band correlates with macrotexture depth (R² = 0.82), and macrotexture depth correlates with friction at highway speeds (R² = 0.71-0.88) according to [Rado (TRR, 1996)](https://doi.org/10.1177/0361198196155300107).

**Wheel speed sensor micro-slip.** Even during straight-line constant-speed driving, individual wheels exhibit small slip ratio fluctuations (on the order of 0.001-0.01). Modern ABS wheel speed sensors resolve these fluctuations at 100 Hz with 0.1 km/h precision. [Cai et al. (2023)](https://arxiv.org/abs/2302.03560) demonstrated cooperative friction estimation from wheel speed data, achieving R² = 0.76 on labeled test data.

The gap in the prior art is a complete system that: (a) fuses acoustic and micro-slip signals for friction estimation during normal driving without requiring excitation events, (b) eliminates the need for labeled friction ground truth through self-supervised learning, (c) aggregates estimates across a vehicle fleet to produce continuous spatiotemporal friction maps, and (d) operates on hardware already present in modern production vehicles.

## Detailed Description

### 1. On-Vehicle Sensor Configuration

The system requires no hardware modifications to vehicles manufactured after approximately 2020. It uses three existing sensor subsystems: cabin microphones (for hands-free calling, voice assistants, and ANC — vehicles with road noise cancellation typically have 4-8 microphones sampling at 44.1 kHz or higher); four ABS wheel speed sensors sampling at 100 Hz; and the vehicle's IMU providing lateral acceleration, longitudinal acceleration, and yaw rate at 100-200 Hz via the ESC module.

### 2. Acoustic Feature Extraction

Audio from cabin microphones is processed in 250 ms frames with 50% overlap: bandpass filtering from 200 Hz to 4000 Hz; computation of a 128-bin log-mel spectrogram using a 2048-point FFT with Hann windowing; extraction of spectral centroid, spread, rolloff (95th percentile), and flux; and computation of a 20-coefficient MFCC vector. Speed normalization uses the [Sandberg speed exponent](https://doi.org/10.3141/2058-10) (~25-35 for passenger tires) to normalize to 80 km/h reference speed.

### 3. Micro-Slip Feature Extraction

For each 250 ms window: longitudinal slip ratio κ = (v_wheel − v_vehicle) / max(v_wheel, v_vehicle); standard deviation of κ; power spectral density of κ in the 1-50 Hz band; cross-correlation between left and right wheel slip ratios. Lateral dynamics features include understeer gradient proxy and rear-to-front slip angle ratio. Combined feature vector: ~180 features (128 acoustic + 52 dynamics).

### 4. Self-Supervised Contrastive Learning Framework

The key innovation: a self-supervised learning approach requiring no labeled friction ground truth. The self-supervision signal comes from spatiotemporal consistency — two vehicles traversing the same road segment within 15 minutes should produce concordant friction estimates despite different masses, tires, and speeds.

Architecture: 1D temporal convolutional encoder (6 layers, 64/128/256 channels, kernel size 7) with 2-layer MLP projection head mapping to 32-dimensional friction embedding z. Contrastive loss with temperature τ = 0.07, hard negative mining from 65,536-embedding memory bank. Linear regression head maps embedding to scalar friction μ ∈ [0.0, 1.2], calibrated using sparse ground truth from ASTM E274 surveys, weather-correlated friction curves per [Hippi and Kangas (TRR, 2015)](https://doi.org/10.3141/2482-06), and ABS/ESC activation events.

### 5. Fleet Aggregation and Map Generation

Per-vehicle estimates (~50 bytes per 10m segment, ~14 KB/min at highway speed) are aggregated using a Bayesian hierarchical model over [H3 hexagonal grid](https://h3geo.org/) cells at resolution 15. The model accounts for per-vehicle calibration offsets, tire degradation drift, weather-dependent temporal dynamics, and spatial autocorrelation. Posterior μ and 95% credible interval served via tile-based REST API. Target latency: 30-90 seconds.

### 6. Privacy Architecture

No vehicle trajectories transmitted. On-vehicle processing produces only segment-level friction estimates with one-way vehicle class hash. Differential privacy (ε = 1.0) applied at aggregation layer.

### 7. Applications

- **Autonomous vehicle motion planning:** Real-time friction maps enable speed profiles matched to actual conditions, reducing unnecessary deceleration by 15-30%.
- **ADAS forward collision warning:** Adjust time-to-collision thresholds based on friction of road ahead.
- **DOT pavement maintenance:** Identify declining friction trends months before biennial surveys.
- **Insurance telematics:** Risk scores reflecting actual road conditions encountered.
- **Navigation rerouting:** Avoid low-friction segments (bridge ice, oil spills).

## Claims

1. A system for estimating road surface friction coefficients, comprising: vehicle cabin microphones capturing tire-road acoustic emission; wheel speed sensors; an on-vehicle processor extracting acoustic spectral features and micro-slip features during normal driving; and a neural network mapping said features to friction coefficient estimates.

2. The system of claim 1, wherein acoustic features comprise log-mel spectrograms, spectral centroid, spread, rolloff, flux, and MFCCs from the 200-4000 Hz band.

3. The system of claim 1, wherein micro-slip features comprise per-wheel longitudinal slip ratio statistics, PSD of slip fluctuations in 1-50 Hz, and left-right wheel slip cross-correlation.

4. The system of claim 1, wherein the neural network is trained via self-supervised contrastive learning using cross-vehicle consistency on shared road segments as positive pairs.

5. The system of claim 4, with configurable temporal coherence window (default 15 min) and precipitation state change filtering.

6. A method for generating real-time crowdsourced friction maps by collecting fleet friction estimates, aggregating via Bayesian hierarchical model accounting for per-vehicle offsets, tire drift, weather dynamics, and spatial autocorrelation, and serving posteriors via tile-based API.

7. The method of claim 6, with differential privacy (configurable ε) on per-segment vehicle counts.

8. The method of claim 6, incorporating weather station data as covariates for temporal interpolation.

9. The system of claim 1, using ANC reference microphones in wheel wells when available for improved SNR.

10. A method for calibrating the self-supervised model using sparse ground truth from ASTM E274 surveys, weather-correlated friction tables, and ABS/ESC activation events, with periodic linear head updates without full encoder retraining.

## Prior Art References

1. [FHWA](https://www.fhwa.dot.gov/publications/research/safety/14078/index.cfm) — Wet pavement crash statistics
2. [FHWA Road Weather](https://ops.fhwa.dot.gov/weather/q1_roadimpact.htm) — Weather-related crash data
3. [US8626454B2](https://patents.google.com/patent/US8626454B2/en) (Continental, 2014) — ABS-based friction estimation
4. [Acosta et al., MSSP 2020](https://doi.org/10.1016/j.ymssp.2019.106437) — Tire-road friction estimation survey
5. [ASTM E274](https://www.astm.org/e0274-23.html) — Locked-wheel skid trailer standard
6. [Roychowdhury et al., IEEE T-ITS 2021](https://doi.org/10.1109/TITS.2020.3025674) — CNN road surface classification
7. [Sandberg & Ejsmont, Applied Acoustics 2018](https://doi.org/10.1016/j.apacoust.2017.11.011) — Tire-road noise-macrotexture correlation
8. [Rado, TRR 1996](https://doi.org/10.1177/0361198196155300107) — Macrotexture-friction correlation
9. [Sandberg speed exponent](https://doi.org/10.3141/2058-10) — Tire-road noise speed dependence
10. [Cai et al., 2023](https://arxiv.org/abs/2302.03560) — Cooperative friction estimation from wheel speed
11. [Hippi & Kangas, TRR 2015](https://doi.org/10.3141/2482-06) — Friction-temperature-precipitation curves
12. [H3 Hexagonal Grid](https://h3geo.org/) — Uber spatial indexing
13. [Bose QuietComfort RNC](https://www.bose.com/automotive) — Road noise cancellation
14. [Gustafsson et al., Applied Sciences 2017](https://www.mdpi.com/2076-3417/7/12/1230) — Road friction virtual sensing review
15. [Grip Tester](https://doi.org/10.1061/JTEPBS.0000100) — Continuous friction measurement device
