# PA-2026-153: Dual-Site PPG Pulse Transit Time Blood Pressure Estimation

**Title:** System and Method for Continuous Non-Invasive Blood Pressure Estimation Using Dual-Site Photoplethysmography Pulse Transit Time Measurement Across Spatially Separated Smart Glasses and Smartwatch Wearables with Personalized Arterial Model Calibration

**Filing:** LITF-PA-2026-153  
**Published:** August 27, 2026  
**Domain:** Wearables / Biomedical Sensing / Cardiovascular Health / Edge AI  
**Full Disclosure:** [liveinthefuture.org/priorart/wearable-dual-site-ptt-blood-pressure.html](https://liveinthefuture.org/priorart/wearable-dual-site-ptt-blood-pressure.html)

---

## Abstract

A system and method for continuous, non-invasive estimation of arterial blood pressure using pulse transit time (PTT) measured between photoplethysmography (PPG) sensors embedded in two wearable devices worn simultaneously at anatomically distinct arterial sites: smart glasses with a PPG sensor positioned over the superficial temporal artery at the temple, and a smartwatch with a PPG sensor positioned over the radial artery at the wrist. The arterial path between the superficial temporal artery and the radial artery traverses approximately 70-90 cm of elastic arterial conduit, including segments of the external carotid, common carotid, subclavian, axillary, brachial, and radial arteries. Pulse transit time across this path is inversely related to pulse wave velocity (PWV), which is governed by the Moens-Korteweg equation linking PWV to arterial wall elastic modulus — a quantity that increases monotonically with transmural blood pressure due to the non-linear stress-strain relationship of the arterial wall.

The system achieves sub-millisecond cross-device time synchronization using Bluetooth Low Energy (BLE) connection event timestamps with linear clock drift correction, enabling PTT resolution of approximately 0.3 ms at 1 kHz PPG sampling. A personalized calibration model, trained on periodic oscillometric cuff reference measurements taken by the user every 2-4 weeks, maps measured PTT to systolic and diastolic blood pressure with an initial calibration accuracy meeting the AAMI/ANSI/ISO 81060-2:2018 standard (mean error ≤ ±5 mmHg, standard deviation ≤ 8 mmHg). On-device recalibration compensates for slow arterial compliance changes from aging, medication, and hydration state. 14 claims.

## Key Technical Claims

1. A system using a PPG sensor in smart glasses (superficial temporal artery) and a PPG sensor in a smartwatch (radial artery) to measure true pulse transit time between two naturally worn wearable devices, with a general-purpose processor estimating blood pressure from the measured PTT using a personalized calibration model.

2. Smart glasses PPG sensor integrated into the temple arm with dual-wavelength illumination (green 525 nm + infrared 940 nm) and transimpedance amplifier sampling at ≥500 Hz, positioned to measure pulsatile blood volume in the superficial temporal artery.

3. Cross-device time synchronization using BLE connection event timestamps with least-squares linear clock drift correction, achieving <200 μs synchronization accuracy — sufficient for sub-millisecond PTT resolution.

4. Multi-fiducial PTT extraction using pulse foot, maximum first-derivative, and systolic peak timing, with weighted combination based on inter-beat variability for robust beat-to-beat measurement.

5. Personalized calibration via ridge regression on [ln(PTT_slope), ln(PTT_foot), heart_rate, PTT_ratio] feature vector, calibrated against ≥3 paired oscillometric cuff measurements, with recursive least squares incremental updates (forgetting factor 0.98) for long-term drift compensation.

6. Postural hydrostatic correction using dual-device 3-axis accelerometers to estimate height difference between sensor sites relative to heart, adjusting BP estimate for the 35-55 mmHg hydrostatic pressure gradient between temple and wrist in standing posture.

7. Activity-aware measurement strategy: continuous 1 kHz in stationary/sleep, gait-phase-gated in walking, burst-mode in running, sparse-sampled in sleep — with activity-specific motion artifact filters and SQI thresholds.

8. Blood pressure estimation model using the Moens-Korteweg equation with Hayashi exponential arterial compliance (E = E₀·exp(αP)), yielding a linear ln(PTT) vs. BP relationship with subject-specific coefficients.

9. Arrhythmia detection and beat exclusion: AF detected via RMSSD > 50 ms / Shannon entropy > 2.0 bits / absent RSA; PVCs detected via short-long interval patterns; affected beats excluded from PTT computation.

10. Calibration drift detection monitoring 24-hour BP standard deviation and nocturnal dipping ratio over multi-day windows, prompting recalibration when distributions deviate from calibration-period baselines.

11. Dual-wavelength adaptive noise cancellation exploiting wavelength-dependent motion-to-pulsatile signal ratio for motion artifact separation at the glasses PPG site.

12. Clinical reporting module computing daytime/nighttime averages, nocturnal dipping ratio, morning surge, BP variability CV, and structured exportable PDF per AHA 2019 consensus format.

13. Fully on-device processing: all PPG analysis, PTT computation, and BP estimation run on wearable processors with no raw PPG waveforms transmitted to external servers; only PTT timestamps traverse the BLE link.

14. Alternative high-precision synchronization via IEEE 802.11mc Fine Timing Measurement on WiFi 6E chipsets for sub-nanosecond accuracy enabling advanced hemodynamic waveform analysis.

## Prior Art Distinguished

- **Samsung Galaxy Watch BP (US10667706B2):** Uses ECG + single-site PPG to measure pulse arrival time (PAT), which includes pre-ejection period (PEP) — a 30-80 ms confound unrelated to arterial blood pressure. This disclosure eliminates PEP by using two peripheral PPG sites, both downstream of the aortic valve.
- **Apple Blood Pressure (US20210030283A1):** Proposes ultrasonic tonometric measurement at a single wrist site. Requires dedicated ultrasonic transducer hardware, not dual-site PTT from existing PPG sensors.
- **Biobeat BB-613WP / Aktiia:** Single-site PPG morphology analysis. Achieves SD of 9.4-14.2 mmHg in independent validation, consistently failing the AAMI 8 mmHg threshold. This disclosure uses true PTT across a 70-90 cm arterial path for fundamentally better physical grounding.
- **Finapres/ClearSight volume-clamp devices:** Clinical-grade continuous BP but require dedicated finger cuff hardware ($5,000-15,000). Not suitable for daily wear.

## Why This Matters

Hypertension is the leading modifiable risk factor for cardiovascular disease worldwide (1.28 billion affected), yet blood pressure monitoring remains episodic and clinic-bound. The 2017 ACC/AHA guideline threshold reduction to 130/80 mmHg reclassified 31 million Americans as hypertensive, most of whom do not monitor their BP regularly. Nocturnal dipping patterns, morning surges, and exercise-related transients — which carry independent prognostic value — are invisible to sporadic office cuff measurements and require continuous monitoring to detect.

Smart glasses and smartwatches are converging toward simultaneous daily wear. This disclosure prevents any party from patenting the concept of using these two naturally worn devices as spatially separated PPG measurement sites for continuous cuff-less blood pressure monitoring via pulse transit time. The fundamental physics (Moens-Korteweg + Hayashi compliance model), the cross-device synchronization method (BLE connection events), the multi-fiducial PTT extraction, and the personalized calibration framework are all placed into the public domain, ensuring that any wearable manufacturer can implement this life-saving capability without licensing fees.

---

*Defensive prior art — dedicated to the public domain under [35 U.S.C. § 102(a)(1)](https://www.law.cornell.edu/uscode/text/35/102)*
