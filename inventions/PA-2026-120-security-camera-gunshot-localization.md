# System and Method for Distributed Acoustic Gunshot Localization and Ballistic Trajectory Reconstruction Using Heterogeneous Consumer Security Camera Microphone Networks with Edge-Deployed Deep Learning

**LITF-PA-2026-120 · Public Safety / Acoustic Sensing**
**Published:** 2026-07-25
**License:** [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/) — Public Domain

> ⚖️ **Prior Art Notice:** This document is published as defensive prior art under [35 U.S.C. § 102(a)(1)](https://www.law.cornell.edu/uscode/text/35/102). The inventions described herein are dedicated to the public domain as of the publication date above.

---

## Abstract

Disclosed is a system and method for detecting, localizing, and classifying gunshot events in residential and urban environments by repurposing the embedded microphones of existing consumer security cameras (Ring, Nest/Google, UniFi Protect, Arlo, Wyze, Reolink, and similar devices) as a distributed acoustic sensor network. The system runs a lightweight gunshot detection convolutional neural network (CNN) on each camera's existing application processor, extracting acoustic feature embeddings from raw audio without transmitting or storing the audio itself. When two or more cameras in a neighborhood detect a candidate gunshot event within a time window consistent with acoustic propagation at local sound speed, a coordination layer performs time-difference-of-arrival (TDOA) multilateration using GPS-synchronized timestamps to estimate the shooter's position to within 3–8 meters. For supersonic projectiles, the system separately detects the ballistic shockwave and the muzzle blast, exploiting the temporal separation between these two wavefronts to reconstruct the bullet's trajectory azimuth and elevation. A multi-path acoustic propagation model, informed by 3D building geometry derived from publicly available LiDAR datasets and camera installation coordinates, corrects for reflection-induced TDOA bias that degrades localization accuracy in dense built environments. The system further classifies firearm type (handgun, rifle, shotgun) and estimates caliber range from the muzzle blast spectral envelope and shockwave N-wave duration. A federated self-calibration protocol uses GPS-timestamped ambient acoustic events of opportunity (vehicle pass-bys, construction equipment, emergency vehicle sirens with known dispatch times) to continuously estimate and correct inter-camera clock skew without manual calibration. All processing occurs on-device or within the local network; only event metadata (timestamp, location estimate, classification, confidence) leaves the premises, preserving audio privacy.

## Field of the Invention

This invention relates to public safety and acoustic sensing, specifically to the detection and geolocation of gunfire using opportunistic networks of consumer security camera microphones, edge-deployed neural networks, and physics-informed acoustic propagation modeling.

## Background

Gunshot detection and localization in civilian environments is dominated by a single commercial system: ShotSpotter (now SoundThinking). ShotSpotter deploys purpose-built acoustic sensor arrays, typically 15–25 sensors per square mile, mounted on rooftops and utility poles. SoundThinking estimates the average annual cost at $65,000–$90,000 per square mile, with a $10,000 per square mile initiation fee. More than 170 U.S. cities and towns have adopted the technology.

Despite widespread deployment, ShotSpotter faces significant criticism on both cost and accuracy grounds:

- **False positive rates:** The MacArthur Justice Center analyzed over 40,000 ShotSpotter dispatches in Chicago over a 21-month period and found that 89% of alerts resulted in no evidence of gun-related crime, and 86% resulted in no crime of any kind. The Chicago Inspector General independently confirmed these findings.
- **Cancellations:** Chicago, Portland, San Antonio, and other cities have canceled or declined to renew ShotSpotter contracts. A National Institute of Justice-funded study using 15 years of data from Chicago and Kansas City found that the technology did not significantly reduce shootings, gun-related crime, or clearance rates.
- **Privacy concerns:** ShotSpotter sensors record continuous audio. The ACLU has documented that recorded audio has been admitted as evidence in criminal trials, raising Fourth Amendment questions about persistent acoustic surveillance.
- **Human override:** The Associated Press reviewed a confidential operations document indicating that 10% of the algorithm's classification decisions are overridden by human analysts at SoundThinking's incident review center, raising questions about the system's autonomy and reproducibility.

Military gunshot detection systems provide higher accuracy but are not applicable to civilian environments. BBN Technologies' counter-sniper system (US5930202A) uses shockwave time-of-arrival across tightly calibrated sensor arrays to estimate bullet trajectory, Mach number, and caliber. Sallai et al. at Vanderbilt University demonstrated muzzle blast and shockwave fusion for shooter localization using wireless sensor networks. These systems require purpose-built, precisely calibrated hardware that costs thousands of dollars per node.

Meanwhile, the installed base of consumer security cameras has grown enormously. Industry estimates suggest over 100 million internet-connected security cameras are deployed in U.S. residential settings as of 2025, with an average of 2.4 cameras per equipped household. Nearly all of these cameras contain MEMS microphones (typically Knowles SPH0645LM4H or InvenSense INMP441, sensitivity -26 to -42 dBFS, SNR 58–65 dB) and application processors (Ambarella CV25, Ingenic T31, or similar) with sufficient compute headroom for lightweight neural network inference.

The gap in the art is a system that: (a) repurposes existing consumer camera microphones as an ad-hoc distributed acoustic array, eliminating per-square-mile sensor deployment costs; (b) performs all gunshot classification on-device to preserve audio privacy; (c) handles the heterogeneity of consumer microphone hardware through federated self-calibration; (d) accounts for multi-path acoustic propagation in built environments using available 3D geometry data; and (e) provides ballistic trajectory reconstruction from the differential timing of muzzle blast and supersonic shockwave arrivals across the network.

## Detailed Description

### 1. System Architecture

The system comprises three tiers: edge nodes (individual cameras), a neighborhood coordinator (running on a local hub, NAS, or cloud endpoint), and a notification/dispatch interface.

Each participating camera runs a firmware extension or sideloaded application that continuously monitors its microphone input. The camera's existing application processor (e.g., Ambarella CV25 with 1 TOPS INT8 inference, Ingenic T31 with 500 MOPS) runs a lightweight gunshot detection model alongside its normal video encoding pipeline, consuming less than 5% of available compute and under 15 mW additional power draw.

The neighborhood coordinator aggregates event reports from participating cameras over the local network (mDNS/Bonjour discovery, encrypted WebSocket connections). It performs TDOA multilateration, multi-path correction, trajectory reconstruction, and classification aggregation.

### 2. On-Device Gunshot Detection

Each camera's microphone samples audio at its native rate (typically 8 kHz for doorbell cameras, 16 kHz for outdoor cameras, 48 kHz for some UniFi Protect models). Audio is processed in 250 ms frames with 50% overlap. Each frame undergoes:

1. **Impulsive event detection:** A zero-crossing rate (ZCR) and short-time energy (STE) gate identifies frames containing impulsive transients. Frames with STE below -35 dBFS or ZCR below 50/s are immediately discarded. This gate rejects >99% of frames at negligible computational cost.
2. **Feature extraction:** Candidate frames are transformed into 64-bin log-mel spectrograms (FFT size 512, Hann window, 50% overlap). A parallel time-domain feature vector is computed: peak amplitude, rise time to 90% peak (gunshots: 0.1–0.5 ms; fireworks: 2–10 ms; backfires: 5–20 ms), total impulse duration, and decay envelope time constant.
3. **Neural classification:** A MobileNetV3-Small backbone (width multiplier 0.5, ~150K parameters, INT8 quantized, ~45 KB model size) processes the mel spectrogram and time-domain features through a dual-input architecture. Outputs: handgun muzzle blast, rifle muzzle blast, shotgun muzzle blast, ballistic shockwave, firework, vehicle backfire, construction impulsive, background. Inference: <8 ms on Ambarella CV25.
4. **Temporal sequence validation:** Post-classification check validates acoustic source signature against known gunshot physics (primer blast, muzzle blast, optional shockwave, optional mechanical sounds).

When a gunshot candidate passes all gates, the camera transmits an event report containing: camera ID, GPS coordinates, NTP-synchronized timestamp (microsecond precision), classification vector, peak SPL estimate, rise time, decay constant, and a 128-dimensional acoustic embedding vector from the CNN's penultimate layer. Raw audio is never transmitted.

### 3. TDOA Multilateration and Shooter Localization

The coordinator receives event reports from multiple cameras and performs spatial-temporal clustering. For each cluster with N ≥ 3 cameras, hyperbolic multilateration using the TDOA matrix estimates the source position. Given cameras at known positions (x_i, y_i, z_i) with arrival times t_i, the TDOA between pairs defines hyperboloids whose intersection yields the estimated position.

The system solves the nonlinear TDOA system using iterative least-squares (Levenberg-Marquardt) initialized from a grid search. For N ≥ 4 cameras, 3D localization is performed. Expected accuracy: 3–8 m with cameras spaced 20–50 m apart and NTP clock synchronization accurate to ±1 ms.

### 4. Multi-Path Acoustic Propagation Correction

In built environments, sound reflects off building facades, pavement, and terrain, creating multi-path arrivals that bias TDOA estimates by 5–50 ms (1.7–17 m). The system constructs a 3D acoustic environment model from USGS 3DEP airborne LiDAR point clouds and OpenStreetMap building footprints. A ray-tracing model precomputes expected reflection paths for candidate source positions. A matched-filter approach selects direct-path arrival times, rejecting reflection-induced bias. Storage: 2–10 MB per square mile; lookup: <50 ms per event.

### 5. Ballistic Trajectory Reconstruction

For supersonic projectiles (muzzle velocity >343 m/s), the system separately detects the muzzle blast (spherical propagation) and ballistic shockwave (conical Mach cone). The temporal separation between these wavefronts at each camera, combined with the estimated shooter position, constrains the bullet's trajectory. With N ≥ 4 cameras detecting both wavefronts, the system estimates trajectory azimuth (±5°), elevation (±8°), approximate bullet velocity (±15%), and caliber range from shockwave N-wave duration and muzzle blast spectral envelope.

### 6. Firearm Classification

The system classifies firearm type using:
- **Muzzle blast spectral envelope:** Handguns: 500–2000 Hz. Rifles: 100–800 Hz. Shotguns: 80–500 Hz.
- **Temporal structure:** Semi-automatic slide/bolt sounds, revolver cylinder gap gas jet.
- **Shot cadence:** Semi-auto handgun: 150–400 ms. Semi-auto rifle: 100–300 ms. Full auto: 60–120 ms. Pump shotgun: 800–2000 ms.
- **Shockwave presence/absence:** Confirms supersonic vs. subsonic loads.

### 7. Federated Self-Calibration

The system handles heterogeneous microphone hardware through continuous calibration:
- **Vehicle pass-bys:** Broadband tire noise tracked across cameras estimates pairwise clock offsets to <100 μs.
- **Emergency sirens:** Known waveforms with dispatch timestamps provide absolute time references.
- **Microphone normalization:** Per-camera equalization filters normalize frequency response using ambient sound spectral correlation.
- Calibration parameters: 24-hour exponentially weighted moving average.

### 8. Privacy Architecture

- Raw audio never leaves the camera.
- Acoustic embeddings are non-invertible (128 dimensions from 16,000+ samples; reconstruction adversary achieves <0.05 correlation).
- "Wake on impulsive event" mode: no audio buffered beyond current 250 ms frame.
- Per-camera opt-in with immediate withdrawal.

## Claims

1. A system for detecting and localizing gunshot events in civilian environments, comprising: a plurality of consumer security cameras, each containing an embedded microphone and an application processor; wherein each camera runs an on-device neural network that classifies impulsive acoustic events as gunshot or non-gunshot without transmitting raw audio; and a neighborhood coordinator that performs time-difference-of-arrival multilateration across event reports from multiple cameras to estimate the geographic position of the gunshot source.

2. The system of claim 1, wherein the on-device classifier is a dual-input convolutional neural network that processes both a log-mel spectrogram and a time-domain feature vector comprising peak amplitude, rise time, impulse duration, and decay time constant to distinguish gunshots from fireworks, vehicle backfires, and construction impulsive sounds.

3. The system of claim 1, wherein each camera transmits only an acoustic feature embedding vector and event metadata to the coordinator, and wherein the embedding is extracted from a bottleneck layer of the neural network designed to be non-invertible such that speech and conversation cannot be reconstructed.

4. The system of claim 1, further comprising a multi-path acoustic propagation correction module that uses 3D building geometry derived from airborne LiDAR data and building footprint datasets to identify and reject reflection-induced TDOA bias, selecting direct-path arrival times via matched-filter comparison against precomputed multi-path arrival patterns.

5. The system of claim 1, further comprising a ballistic trajectory reconstruction module that separately detects the muzzle blast and ballistic shockwave of a supersonic projectile at each camera, computes the temporal separation between the two wavefronts, and estimates the bullet trajectory azimuth and elevation from the differential timing across multiple cameras.

6. The system of claim 5, wherein the ballistic trajectory reconstruction module further estimates caliber range from the shockwave N-wave duration and muzzle blast spectral envelope, classifying the projectile into categories including .22 LR, 9mm-class handgun, 5.56mm-class rifle, 7.62mm-class rifle, and 12-gauge shotgun slug.

7. The system of claim 1, further comprising a federated self-calibration protocol that estimates and corrects inter-camera clock skew using ambient acoustic events of opportunity, including vehicle pass-by tire noise tracked across multiple cameras, emergency vehicle sirens with known waveform signatures, and classified non-gunshot impulsive events.

8. A method for gunshot localization comprising: continuously monitoring audio at a plurality of consumer security cameras using an on-device impulsive event gate; classifying candidate impulsive events using an edge-deployed neural network; transmitting only non-invertible acoustic embeddings and event metadata to a coordinator; performing TDOA multilateration across cameras detecting the same event; and correcting TDOA estimates using a precomputed multi-path acoustic propagation model derived from 3D environmental geometry.

9. The method of claim 8, further comprising classifying the firearm type as handgun, rifle, or shotgun based on muzzle blast spectral envelope frequency distribution, temporal sub-event sequence analysis, and inter-shot interval for multiple-round events, aggregating features across all detecting cameras weighted by proximity and microphone quality.

10. The system of claim 1, wherein participating cameras are heterogeneous consumer devices with varying microphone hardware, sample rates, and frequency responses, and wherein the federated self-calibration protocol normalizes each camera's effective frequency response to a common reference curve using correlation of ambient sound spectra against reference spectra.

11. The system of claim 1, wherein the neighborhood coordinator computes a geometric dilution of precision (GDOP) coverage map from the spatial distribution of participating cameras and identifies coverage gaps where additional camera enrollment would most improve localization accuracy.

## Prior Art References

1. [SoundThinking (ShotSpotter)](https://en.wikipedia.org/wiki/SoundThinking). Commercial gunshot detection, $65K–$90K/sq mi/year, 170+ U.S. cities
2. [Beacon Journal, March 2025](https://www.beaconjournal.com/story/news/nation/2025/03/28/united-states-cities-shotspotter-gun-violence/82697958007/). U.S. cities canceling ShotSpotter due to cost and efficacy concerns
3. [Piza et al., NIJ-funded study](https://policinginsight.com/feature/analysis/i-studied-shotspotter-in-chicago-and-kansas-city-heres-what-people-in-other-cities-and-towns-using-this-technology-should-know/). 15-year analysis: ShotSpotter did not reduce shootings or improve clearance rates
4. [US5930202A (Duckworth et al., BBN Technologies, 1999)](https://patents.google.com/patent/US5930202A/en). Counter-sniper system using shockwave TDOA for trajectory estimation
5. [Sallai et al., Vanderbilt University](https://raw.githubusercontent.com/Vegas-Oct-1-Sounds/Gunshot-Acoustics/1f2ccefc66ee656c7aaf0e7a43cc393b4f9494d7/Library/Sallai%20et%20al.%20-%20Fusing%20Distributed%20Muzzle%20Blast%20and%20Shockwave%20Dete.pdf). Muzzle blast and shockwave fusion for shooter localization with wireless sensor networks
6. [Acoustical Society of America](https://acoustics.org/novel-audio-analysis-helps-identify-multiple-sounds-in-forensic-gunshot-recordings/). Forensic audio analysis of gunshot sub-event sequences
7. [WO2016032918A1](https://patents.google.com/patent/WO2016032918A1/en). Near-field gunshot and explosion detection using distributed acoustic sensors
8. [USGS 3D Elevation Program (3DEP)](https://www.usgs.gov/3d-elevation-program). Nationwide airborne LiDAR coverage for 3D building geometry
9. [OpenStreetMap](https://www.openstreetmap.org/). Building footprints with height estimates
10. [Gunshot Audio Forensics Dataset (Zenodo)](https://zenodo.org/record/3997472). Labeled gunshot audio recordings
11. [TensorFlow Lite for Microcontrollers](https://www.tensorflow.org/lite/microcontrollers). On-device ML runtime for embedded inference
12. [MacArthur Justice Center](https://www.macarthurjustice.org/case/shotspotter/). 89% of Chicago ShotSpotter alerts found no gun-related crime
