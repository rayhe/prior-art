# PA-2026-152: WiFi Arc Fault RF Detection

**Title:** System and Method for Residential Electrical Arc Fault Detection and Localization Using Distributed WiFi Access Point Radio Frequency Spectral Anomaly Monitoring with On-Device Machine Learning Classification

**Filing:** LITF-PA-2026-152  
**Published:** August 26, 2026  
**Domain:** Electrical Safety / WiFi Sensing / Edge AI  
**Full Disclosure:** [liveinthefuture.org/priorart/wifi-arc-fault-rf-detection.html](https://liveinthefuture.org/priorart/wifi-arc-fault-rf-detection.html)

---

## Abstract

A system and method for detecting, classifying, and localizing electrical arc faults in residential and commercial structures using the radio frequency spectral monitoring capabilities already present in deployed consumer WiFi access points. Electrical arcing between conductors generates broadband electromagnetic emissions spanning approximately 1 MHz to 3 GHz, with characteristic spectral signatures that overlap the 2.4 GHz and 5 GHz WiFi operating bands. Modern WiFi 6/6E/7 chipsets (e.g., Qualcomm QCA9880, QCA6490, MediaTek MT7915, Broadcom BCM4389) incorporate spectral scan engines that compute FFT-based power spectral density measurements across their operating bands as part of standard interference management.

This disclosure describes repurposing these existing spectral scan capabilities, combined with WiFi Channel State Information (CSI) subcarrier-level amplitude and phase data, to detect the broadband non-Gaussian noise floor elevation and characteristic temporal burst patterns produced by series and parallel arc faults. A lightweight convolutional neural network classifier running on the access point's application processor distinguishes arc fault emissions from common interference sources (microwave ovens, Bluetooth, ZigBee, baby monitors, cordless phones) based on spectral shape, temporal envelope, and 60 Hz periodicity features. When three or more access points in a mesh network detect the same arc event, received signal strength differential and time-difference-of-arrival analysis localizes the fault source to within approximately 2-3 meters. The system provides continuous whole-structure arc fault monitoring using infrastructure already deployed in over 90 million U.S. households with WiFi mesh systems, requiring only a firmware update and no additional hardware.

## Key Technical Claims

1. Using existing WiFi AP spectral scan engines (hardware FFT modules for DFS radar detection) to detect broadband noise floor elevation characteristic of electrical arcing, without additional RF hardware.

2. Detecting 60 Hz and 120 Hz amplitude modulation in the temporal envelope of broadband noise events, arising from periodic re-ignition of AC arc faults synchronized to the power line cycle — a feature absent from all common wireless interference sources.

3. A CNN classifier (120 KB INT8, <5 ms inference on Cortex-A53) trained to distinguish parallel arc faults, series arc faults, microwave oven interference, Bluetooth, ZigBee, and background noise using spectral flatness, 120 Hz modulation index, cross-band correlation, and amplitude kurtosis.

4. Multi-AP localization via received signal strength differentials across 3+ mesh access points, using pre-calibrated radio propagation models for room-level (~2-3 m) fault source estimation.

5. Cross-band correlation analysis verifying broadband events appear simultaneously in both 2.4 GHz and 5 GHz spectral scan outputs — a discriminator exploiting the fact that arc emissions span both bands while single-band interferers do not.

6. Time-difference-of-arrival localization using IEEE 802.11mc Fine Timing Measurement synchronization for sub-meter accuracy in WiFi 6E/7 systems.

7. Three-tier alert escalation from logged watch events through push notification warnings to critical alerts with optional automated circuit de-energization via Matter/SmartThings/HomeKit smart home integration.

8. Self-test mechanism using broadband noise-like test signals with known pseudo-random modulation transmitted between APs to verify detection pipeline operational readiness.

9. WiFi CSI-based complementary detection using per-subcarrier amplitude perturbation patterns across 242+ OFDM subcarriers to identify arc fault signatures versus frequency-selective fading.

10. Firmware-only deployment to existing consumer WiFi mesh hardware (quantized model <200 KB, inference <10 ms) requiring no hardware modification.

## Prior Art Distinguished

- **AFCI breakers (NEC 210.12):** Monitor current waveforms on individual circuits. Cannot detect faults on unprotected circuits or in service entrance wiring. Require per-circuit hardware installation.
- **GB2225185A (1988):** Dedicated RF receiver coupled to mains wiring at ~170 kHz. Single narrowband frequency, no spatial localization, requires purpose-built hardware.
- **Wang et al. (MDPI):** Characterized arc EM radiation at ~14 MHz but proposed dedicated antenna measurement, not repurposing existing WiFi infrastructure.

## Why This Matters

Arc faults remain the leading cause of residential electrical fires despite decades of AFCI deployment. The retrofit cost barrier ($800-2,400 per home) has left 80+ million pre-2014 U.S. homes without protection. WiFi mesh systems are already deployed in 90+ million households. This disclosure makes it impossible for any party to patent the concept of using those existing WiFi access points as distributed arc fault sensors, preserving the ability of any WiFi manufacturer to add this safety feature via firmware update without licensing fees.

---

*Defensive prior art — dedicated to the public domain under [35 U.S.C. § 102(a)(1)](https://www.law.cornell.edu/uscode/text/35/102)*
