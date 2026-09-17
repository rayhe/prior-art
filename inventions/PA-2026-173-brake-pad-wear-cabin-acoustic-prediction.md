# PA-2026-173: Brake Pad Wear Estimation from In-Cabin Acoustic Analysis

**Title:** System and Method for Estimating Automotive Brake Pad Wear and Predicting Brake Service Timing Using In-Cabin Acoustic Analysis and Driving Pattern Fusion

**Filing:** LITF-PA-2026-173
**Published:** September 16, 2026
**Domain:** Automotive / Predictive Maintenance
**Full Disclosure:** [liveinthefuture.org/priorart/brake-pad-wear-cabin-acoustic-prediction.html](https://liveinthefuture.org/priorart/brake-pad-wear-cabin-acoustic-prediction.html)
**License:** [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/) — Public Domain

> Prior Art Notice: This document is published as defensive prior art under
> [35 U.S.C. Sec. 102(a)(1)](https://www.law.cornell.edu/uscode/text/35/102).
> The inventions described herein are dedicated to the public domain as of the
> publication date above.

---

## Abstract

Disclosed is a system and method for estimating automotive brake pad wear and predicting brake service timing using only a consumer smartphone mounted in the vehicle cabin. The system detects braking events from GPS speed and inertial measurement unit (IMU) deceleration, extracts acoustic features from cabin audio recorded during each braking event, and tracks three wear-related signal channels over weeks to months: (1) wear-indicator squealer onset, a new stable narrowband tone in the 1 to 7 kHz band that first appears at light, low-speed stops and spreads to higher speeds and pressures as pads thin toward the 2 to 3 mm squealer engagement point; (2) pre-squeal broadband braking noise trending, where 200 Hz to 2 kHz band energy per unit braking work rises as the friction material thins and loses damping; and (3) grinding signature detection for the metal-on-metal state indicating rotor damage in progress. A benign-squeal discriminator separates wear-induced squeal from morning rust oxidation squeal, cold-damp squeal, and glazed-pad noise using temporal persistence, first-stop-of-day gating, and weather data. Per-axle localization uses dual-microphone amplitude and phase differences. A wear-rate model fuses the acoustic wear index with cumulative braking work derived from GPS trip history to estimate remaining friction material thickness per axle and predict miles and days to the 3 mm service threshold, issuing escalating advisories from maintenance planning through rotor-damage emergency.

## Technical Field

This invention relates to automotive predictive maintenance, specifically to estimating disc brake pad wear and forecasting brake service timing through passive in-cabin acoustic sensing, brake-event segmentation, spectral wear-signature trending, and fusion with driving-pattern data from a consumer smartphone.

## Background

Disc brake pads are consumable friction components. New pads carry 10 to 12 millimeters of friction material, and replacement is generally recommended at 3 millimeters or less ([Engineer Fix](https://engineerfix.com/how-much-brake-pad-thickness-is-safe-2/)). Most pads carry an acoustic wear indicator, a metal tab (squealer) protruding 2 to 3 millimeters past the backing plate that contacts the rotor when the friction material wears down to 2 to 3 millimeters, producing a high-pitched screech designed as an unmistakable warning ([Raybestos Bulletin 18-13](https://www.brakepartsinc.com/dam/jcr:24db1e41-6fd5-4661-9508-0ec1604bb88d/Brake%20Wear%20Sensor%20Tech%20Bulletin.pdf); [HELLA](https://www.hella.com/techworld/za/ti/brake-pad-wear-indicator/)). Electronic wear sensors exist on some vehicles but cover only one or two pads per axle and trigger only at the wear limit.

Brake degradation is a material safety problem. In NHTSA's National Motor Vehicle Crash Causation Survey, among crashes where the critical pre-crash reason was attributed to the vehicle, brakes failed or degraded in 25.0 percent (weighted estimate 11,144 crashes), second only to tires ([NMVCCS Report to Congress, 2008](https://mail.thenewspaper.com/rlc/docs/2008/us-crashcause.pdf)). Worn pads also carry a steep cost gradient: pad-only replacement runs roughly $200 to $500 per axle, while ignoring the squealer until grinding begins raises the repair to $500 to $1,200 per axle for pads plus rotors, with caliper damage adding $300 to $700 more.

The dominant detection method today is the driver's ear: waiting for the squealer to become audible, then scheduling service before grinding starts. The squealer only engages at 2 to 3 mm, leaving little margin, and many drivers cannot distinguish wear squeal from benign noise. Morning rust oxidation on rotors after overnight parking produces a brief squeal that disappears within the first few stops ([Gulf Euro Clinic](https://medium.com/@gulfeuroclinic.bradenton/brake-squealing-decoded-when-its-wear-indicators-vs-when-it-s-something-worse-db5d463eb33a)), and cold or damp conditions can provoke transient squeal in healthy pads. Human hearing cannot track the slow growth of braking noise over months.

Brake squeal itself is well studied in NVH research. Low-frequency squeal spans 1 to 7 kHz and high-frequency squeal 8 to 16 kHz, with the human ear most sensitive between 1 and 4 kHz; the mechanism is geometric instability from mode coupling between rotor and pad ([J. Braz. Soc. Mech. Sci.](https://www.scielo.br/j/jbsmse/a/6kdr4CknLmB3Dyv4SxkCZjP/?format=html&lang=en)). Squeal is generated by friction-induced self-excited vibration and is most prominent at low speeds under 20 km/h in disc-brake test stands ([JSME, tested on railroad-car disc brakes](https://www.jstage.jst.go.jp/article/transjsme/advpub/0/advpub_16-00337/_article/-char/en)).

Smartphone-based vehicle diagnostics are an established research direction. Siegel, Sarma, and colleagues at MIT demonstrated that a phone's microphone, accelerometer, and GPS can diagnose engine misfire, clogged air filters, wheel imbalance, and tire pressure with accuracy above 90 percent, published in *Engineering Applications of Artificial Intelligence* ([Siegel et al., EAAI](https://linkinghub.elsevier.com/retrieve/pii/S0952197617302294); [Smithsonian](https://www.smithsonianmag.com/innovation/app-can-diagnose-your-car-trouble-180967412/)). Separately, Terwilliger and Siegel at Michigan State developed cascading deep architectures for acoustic vehicle characterization and misfire fault diagnosis ([arXiv:2205.09667](http://arXIV.org/pdf/2205.09667)). Those works addressed engine, wheel, and filter faults. Neither addressed brake pad wear, which presents a distinct signal problem: the signature of interest is a slow temporal trend across hundreds of braking events, not a single-event fault classification, and it must be separated from the rich benign noise background of normal braking.

The gap in the art is a system that: (a) passively tracks brake wear acoustics longitudinally across braking events using only consumer hardware already in the cabin; (b) detects pre-squealer wear through broadband noise trending before the designed warning engages; (c) discriminates wear-induced squeal from benign transient squeal via temporal persistence and environmental gating; (d) localizes wear to the front or rear axle; and (e) converts the acoustic trend into a remaining-thickness estimate and a dated service forecast fused with the driver's actual braking workload.

## Detailed Description

### 1. Sensor Configuration and Brake Event Segmentation

The sensing device is a consumer smartphone in a dashboard or windshield mount, a placement the MIT work found sufficient for vehicle acoustic diagnostics. Three onboard sensors are used: the microphone (44.1 kHz or 48 kHz sampling), the IMU accelerometer, and GPS. No vehicle modification, OBD connection, or external hardware is required.

Braking events are segmented from GPS speed and IMU longitudinal deceleration. A candidate event requires: speed above 15 km/h at event start (excluding parking maneuvers), peak deceleration below minus 0.08 g sustained for at least 1 second, and an approximately monotonic speed decrease (tolerant to GPS jitter). Audio is recorded into a rolling ring buffer; only the 10 seconds surrounding each validated braking event are retained for feature extraction, and raw audio is discarded after on-device feature computation to limit privacy exposure.

### 2. Acoustic Feature Extraction

Each braking event's audio is processed with a 4096-point short-time Fourier transform (93 ms frames, 50 percent overlap, Hann window), producing per-event spectrograms from which three feature families are derived:

- **Narrowband tonal track (1 to 7 kHz):** Peak detection identifies stable tonal components present during the braking phase but absent in the pre-braking baseline. For each tone the system records center frequency, bandwidth, signal-to-noise ratio relative to the event's broadband floor, and the speed and deceleration at which it appears. The wear-indicator signature is a tone whose center frequency is stable across events (squealer geometry fixes the contact, so the tone does not drift) while its SNR grows over weeks and its onset conditions expand from low-speed light stops to higher speeds and pressures.
- **Broadband braking noise (200 Hz to 2 kHz):** Total band energy is normalized by the event's braking work (integral of deceleration times speed) to produce a noise-per-unit-work metric. As friction material thins, the pad's damping decreases and this metric rises. This channel provides the pre-squealer early indicator: a statistically significant upward trend over 30 or more braking events before any narrowband tone emerges.
- **Grinding signature (below 2 kHz):** Metal-on-metal contact produces high-kurtosis broadband energy with strong low-frequency growl components and amplitude modulation at wheel rotation frequency. A single confident detection triggers the emergency tier.

### 3. Benign-Squeal Discriminator

Not all braking squeal indicates wear. The discriminator separates four classes using a gradient-boosted decision tree trained on labeled braking events:

- **Wear squeal:** tone persists across consecutive days, SNR trend positive over a 14-day window, onset speed threshold rising, unaffected by weather.
- **Morning rust squeal:** appears only in the first 1 to 5 stops after the vehicle was parked overnight (first-stop-of-day flag), decays within the drive cycle, correlates with overnight humidity above 80 percent or rain.
- **Cold-damp squeal:** transient, correlates with ambient temperature below 5 C and high humidity, no positive multi-day trend.
- **Glazed or contaminated pad noise:** broadband elevation without a stable narrowband tone, often following a hard-braking episode or car wash.

Ambient temperature and humidity are obtained from a weather API keyed to GPS position at trip start. The discriminator's output is a per-event probability that the observed signature is wear-induced; only wear-probability-weighted features feed the wear index.

### 4. Per-Axle Localization

Modern smartphones carry two or more microphones (typically bottom and top). The system computes inter-microphone level difference and phase difference in the wear-signature bands during braking events. Because the phone's orientation in a dash mount is roughly fixed (bottom mic toward the driver, top mic toward the windshield), the relative level difference maps to front-versus-rear source position after a one-time per-vehicle calibration performed during the first 20 braking events. The output is a front-axle and rear-axle wear index, since front and rear pads wear at different rates (front pads typically wear 2 to 3 times faster, an industry rule of thumb). Localization accuracy is limited by the cabin transfer function and by phase ambiguity at kilohertz frequencies, so the module reports a confidence-weighted axle attribution rather than a precise source position.

### 5. Wear-Rate Model and Remaining-Thickness Estimation

The acoustic wear index is mapped to estimated remaining friction material thickness per axle through a wear-rate model with two inputs:

- **Acoustic channel:** the wear-probability-weighted tonal SNR trend and broadband noise-per-work trend, each normalized to the vehicle's own 30-day baseline.
- **Braking workload channel:** cumulative braking work per mile, computed from GPS trip logs as the integral of deceleration times speed over all braking events, divided by odometer-equivalent distance. This captures the difference between a highway commuter (low braking work per mile, pads last 60,000+ miles) and a city rideshare driver (high braking work per mile, pads last 20,000 to 30,000 miles).

Calibration anchors: on first run the user enters the date and mileage of the last brake service (or confirms new pads), establishing the 10 to 12 mm starting point. The model then tracks thickness loss as a function of cumulative braking work, with the acoustic channels providing independent confirmation and correction: when the squealer tone is first confidently detected, the model snaps the estimate to 2 to 3 mm remaining for that axle regardless of the workload integration, since squealer engagement is a physical ground truth.

### 6. Alert Tiers and Service Forecasting

The system issues escalating advisories based on the minimum per-axle thickness estimate:

- **Plan (estimated 4 to 5 mm):** informational notice with projected miles and date to the 3 mm threshold, computed from the driver's recent braking workload rate. Includes the cost framing: pad-only service now versus pad-plus-rotor service if deferred.
- **Schedule (squealer confirmed, 2 to 3 mm):** service recommendation within 2 to 4 weeks, naming the affected axle. The advisory explains that the designed warning has engaged and grinding has not yet begun.
- **Urgent (estimated below 1.5 mm or grinding detected):** immediate service warning with rotor-damage risk, since backing-plate contact scores the rotor and converts a pad job into a pad-plus-rotor job.

### 7. Fleet Learning and Privacy

Brake NVH signatures are vehicle-model-specific: rotor diameter, pad compound, and caliper geometry shift the squeal frequencies. The system uploads only anonymized per-event feature vectors (tonal frequencies, SNRs, broadband metrics, wear probabilities) with a vehicle model identifier, never raw audio. A server-side model learns per-model squealer frequency priors and wear-rate baselines, which are pushed back to devices as calibration updates. No location history or audio leaves the device.

### 8. Implementation Notes

A reference implementation runs as a mobile application with a foreground driving-detection service. Audio capture uses the platform's voice-recognition audio source preset (which applies mild noise suppression tuned for speech but preserves the 1 to 7 kHz band of interest) or the raw unprocessed source where available. The ring buffer holds 60 seconds of audio; the segmentation module runs on 1 Hz GPS and 50 Hz IMU. Feature extraction runs on-device using a fixed-point FFT library, and the wear model updates once per trip. Battery impact is bounded by duty-cycling audio processing to validated braking events only, typically under 5 percent of driving time. Electronic pad-wear sensors, where the vehicle has them, are treated as an additional ground-truth input: a dashboard wear-light event snaps the estimate identically to acoustic squealer detection.

## Claims

1. A system for estimating automotive brake pad wear, comprising: a consumer smartphone mounted in a vehicle cabin, the smartphone comprising a microphone, an inertial measurement unit, and a GPS receiver; a brake-event segmentation module that identifies braking events from GPS speed and IMU deceleration; an acoustic feature extraction module that computes per-braking-event spectral features; and a wear estimation module that tracks the spectral features across a plurality of braking events to estimate remaining brake pad friction material thickness.
2. The system of claim 1, wherein the acoustic feature extraction module tracks a narrowband tonal component in the 1 to 7 kHz band across braking events, and wherein the wear estimation module detects wear-indicator squealer onset from a stable-center-frequency tone whose signal-to-noise ratio increases over a multi-week window and whose onset conditions expand from low-speed light braking to higher speeds and pressures.
3. The system of claim 1, further comprising a pre-squealer wear channel that computes broadband braking noise energy in the 200 Hz to 2 kHz band normalized by per-event braking work, and detects pad thinning from a statistically significant upward trend in noise per unit braking work before any narrowband squealer tone emerges.
4. The system of claim 1, further comprising a benign-squeal discriminator that classifies braking-event acoustic signatures into wear squeal, morning rust oxidation squeal, cold-damp squeal, and glazed-pad noise using temporal persistence across days, first-stop-of-day gating, and ambient temperature and humidity, and weights the wear estimation by the wear-squeal probability.
5. The system of claim 1, further comprising a per-axle localization module that computes inter-microphone level and phase differences between two microphones of the smartphone during braking events to attribute wear signatures to a front axle or a rear axle.
6. The system of claim 1, further comprising a braking-workload model that integrates deceleration and speed from GPS trip logs into cumulative braking work per mile, and fuses the cumulative braking work with the tracked acoustic features to estimate remaining friction material thickness per axle in millimeters.
7. The system of claim 6, wherein detection of the wear-indicator squealer tone causes the wear estimation module to set the remaining thickness estimate for the attributed axle to the squealer engagement range of 2 to 3 millimeters as a physical calibration anchor.
8. The system of claim 1, further comprising an alert module issuing escalating advisories: a planning advisory at an estimated 4 to 5 mm remaining with a projected date to the 3 mm threshold; a scheduling advisory upon confirmed squealer onset; and an urgent advisory upon grinding-signature detection indicating rotor damage in progress.
9. The system of claim 1, wherein raw cabin audio is processed on-device and discarded after feature extraction, and only anonymized per-event feature vectors with a vehicle model identifier are transmitted for fleet learning of vehicle-model-specific squealer frequency priors.
10. A method for predicting automotive brake service timing, comprising: passively recording cabin audio with a dashboard-mounted smartphone during braking events identified from GPS and IMU data; extracting per-event narrowband tonal and broadband noise features; discriminating wear-induced signatures from benign transient squeal via multi-day temporal persistence and environmental gating; localizing signatures to a vehicle axle via dual-microphone differences; estimating remaining pad thickness per axle by fusing acoustic wear trends with cumulative braking workload; and issuing a dated service forecast when the estimate crosses a service threshold.

## Prior Art References

1. Engineer Fix: New pads 10-12 mm; replacement recommended at 3 mm or less. [https://engineerfix.com/how-much-brake-pad-thickness-is-safe-2/](https://engineerfix.com/how-much-brake-pad-thickness-is-safe-2/)
2. Raybestos Bulletin 18-13: Acoustic squealer tab protrudes 2-3 mm past backing plate; contact produces high-pitched squeal. [https://www.brakepartsinc.com/dam/jcr:24db1e41-6fd5-4661-9508-0ec1604bb88d/Brake%20Wear%20Sensor%20Tech%20Bulletin.pdf](https://www.brakepartsinc.com/dam/jcr:24db1e41-6fd5-4661-9508-0ec1604bb88d/Brake%20Wear%20Sensor%20Tech%20Bulletin.pdf)
3. HELLA Tech World: Disc brake pad wear limit usually 2 mm; acoustic and electronic indicator types. [https://www.hella.com/techworld/za/ti/brake-pad-wear-indicator/](https://www.hella.com/techworld/za/ti/brake-pad-wear-indicator/)
4. NHTSA NMVCCS Report to Congress, 2008: Brakes failed/degraded in 25.0% of vehicle-attributed crashes. [https://mail.thenewspaper.com/rlc/docs/2008/us-crashcause.pdf](https://mail.thenewspaper.com/rlc/docs/2008/us-crashcause.pdf)
5. J. Braz. Soc. Mech. Sci.: Low-frequency brake squeal 1-7 kHz, high-frequency 8-16 kHz; mode-coupling mechanism. [https://www.scielo.br/j/jbsmse/a/6kdr4CknLmB3Dyv4SxkCZjP/?format=html&lang=en](https://www.scielo.br/j/jbsmse/a/6kdr4CknLmB3Dyv4SxkCZjP/?format=html&lang=en)
6. JSME: Brake squeal of disc brake apparatus at low speed (20 km/h or less); coupled rotor-pad vibration (railroad-car disc brake test stand). [https://www.jstage.jst.go.jp/article/transjsme/advpub/0/advpub_16-00337/_article/-char/en](https://www.jstage.jst.go.jp/article/transjsme/advpub/0/advpub_16-00337/_article/-char/en)
7. Siegel, Sarma et al., Engineering Applications of Artificial Intelligence: Smartphone-based diagnosis of engine misfire, clogged air filters, wheel imbalance, and tire pressure at above 90% accuracy (MIT); and Terwilliger and Siegel, arXiv:2205.09667: cascading deep architectures for acoustic vehicle characterization and misfire diagnosis (Michigan State, 2022). [https://linkinghub.elsevier.com/retrieve/pii/S0952197617302294](https://linkinghub.elsevier.com/retrieve/pii/S0952197617302294) [http://arXIV.org/pdf/2205.09667](http://arXIV.org/pdf/2205.09667)
8. Smithsonian: MIT smartphone car-diagnostics app; above 90% accuracy on misfire, air filter, wheel imbalance, tire pressure. [https://www.smithsonianmag.com/innovation/app-can-diagnose-your-car-trouble-180967412/](https://www.smithsonianmag.com/innovation/app-can-diagnose-your-car-trouble-180967412/)
9. Gulf Euro Clinic: Morning rust squeal versus wear-indicator squeal; pad-only vs pad-plus-rotor cost gradient. [https://medium.com/@gulfeuroclinic.bradenton/brake-squealing-decoded-when-its-wear-indicators-vs-when-it-s-something-worse-db5d463eb33a](https://medium.com/@gulfeuroclinic.bradenton/brake-squealing-decoded-when-its-wear-indicators-vs-when-it-s-something-worse-db5d463eb33a)
