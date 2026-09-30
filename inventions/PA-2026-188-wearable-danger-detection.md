# PA-2026-188: Wearable-Based Danger Detection with Staged Escalation and Multi-Sensor Confirmation

**Title:** System and Method for Wearable-Based Danger Detection with Staged Escalation and Multi-Sensor Confirmation

**Filing:** LITF-PA-2026-188
**Published:** September 29, 2026
**Domain:** Wearables / Emergency Response / Sensor Fusion
**Full Disclosure:** [liveinthefuture.org/priorart/wearable-danger-detection-escalation.html](https://liveinthefuture.org/priorart/wearable-danger-detection-escalation.html)
**License:** [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/) — Public Domain

> Prior Art Notice: This document is published openly as a technical disclosure
> under [35 U.S.C. Sec. 102(a)(1)](https://www.law.cornell.edu/uscode/text/35/102).
> Whether it constitutes prior art, and what it discloses, depends on the facts,
> including public accessibility, timing, and the disclosure's content.
> Publication does not establish novelty, patentability, freedom to operate, or
> public-domain status. This disclosure is offered as evidence of the state of
> the art for examiners and challengers to consider. This is general
> information, not legal advice.

---

## Abstract

Disclosed is a wrist-worn system for danger detection that fuses accelerometer, gyroscope, barometer, microphone, photoplethysmography (PPG), and GPS signals on-device and escalates in stages. Stage one is on-device classification of a danger event (overdose, assault/struggle, or gunshot). Stage two is a discreet user-confirmation window with a cancel option. Stage three, reached only if the user does not cancel, is an automatic call to emergency services relaying location and available medical identification, plus notification of enrolled emergency contacts. Overdose detection uses a respiration/apnea signature from wrist sensors with opt-in at-risk enrollment. Assault/struggle classification is disclosed together with its base-rate precision problem, and automated dispatch for assault-type events is gated on the confirmation window. Gunshot acoustic detection is gated on multi-device corroboration before any automated dispatch. A duress PIN is disclosed as a deterministic user-interface input only; passive detection of coercion is explicitly not claimed.

## Technical Field

This disclosure relates to wearable health and safety systems, specifically to multi-sensor fusion on a wrist-worn device for detecting medical and interpersonal-danger events, and to staged escalation architectures that separate on-device classification from automated emergency response by means of a user confirmation gate and multi-device corroboration.

## Background

Consumer wearables already automate emergency response for two narrow event classes. Fall Detection on the Apple Watch taps the wrist, sounds an alarm, and shows an alert after a hard fall; if the wearer is immobile for about one minute, a 30-second countdown begins before the watch automatically calls emergency services and emergency contacts with the device location (Apple Support). Crash Detection fuses a dual-core accelerometer measuring up to 256 G, a high-dynamic-range gyroscope, a barometer detecting cabin-pressure changes, GPS speed-change detection, and a microphone listening for crash-like noise, with motion algorithms trained on more than one million hours of real-world driving and crash-record data (Apple Newsroom). These systems work because their signals are separable and their confirmation logic is simple: a fall victim is immobile; a crash victim may not move at all.

Danger events beyond falls and crashes are harder. In the United States in 2023, about 22.5 violent victimizations occurred per 1,000 persons age 12 or older (BJS NCVS 2023). University research demonstrated that a smartphone using its speaker and microphone as an active sonar breathing monitor detected overdose-related breathing problems about 90% of the time in 94 supervised-injection-site participants (UW Second Chance), and that a closed-loop wearable injector tracked respiration with two accelerometers to reverse overdose automatically (UW newsroom). Controlled studies have classified aggressive versus non-aggressive movements from smartwatch accelerometer and gyroscope data at above 98.4% accuracy in a scripted setting (PMC), and convolutional-network gunshot classification has been shown running on a consumer phone microphone (Patsi thesis). But controlled accuracy does not survive the base-rate problem of real-world assault, and a deep-learning gunshot prototype documented false positives from close-range screaming, which co-occurs with attacks (MDPI Acoustics). This disclosure records an architecture that takes those limitations into account: staged escalation, a mandatory confirmation gate for assault-type events, and multi-device corroboration for gunshot events before any automated dispatch.

## Detailed Description

### 1. Sensor platform

The system runs on a wrist-worn device carrying at least an accelerometer, a gyroscope, a barometer, a microphone, a PPG optical heart sensor, and GPS/GNSS. Classification is performed on-device. The sensor stack mirrors the crash-detection precedent (accelerometer, gyroscope, barometer, GPS, microphone) with the addition of wrist-centric physiological sensing (PPG heart rate and heart-rate variability, respiration-linked motion from the accelerometer) that a fall detector does not need.

### 2. Staged escalation pipeline

Every danger event passes through the same stages. Stage one, classification: on-device models produce a danger-event hypothesis from fused sensor data. Stage two, user confirmation: the device presents a discreet prompt with a cancel path and a countdown, modeled on the fall-detection alert but silent or vibration-only when silence is safer. Stage three, automated response: only if the user does not cancel, the device calls emergency services relaying latitude and longitude plus available Medical ID information, and notifies enrolled emergency contacts. Stage four, contact escalation: for enrolled at-risk populations (overdose), non-response can additionally escalate to a trusted contact before or instead of emergency services. No stage may be skipped: the classifier alone never dispatches for assault-type or gunshot events.

### 3. Overdose detection

Overdose detection uses a respiration/apnea signature derived from wrist-worn sensors: prolonged apnea or breathing at or below seven breaths per minute, consistent with the threshold demonstrated in the UW Second Chance sonar system, detected here from wrist PPG respiration modulation and dual-axis accelerometer respiration tracking. Because overdose is a low-motion, non-adversarial event, the confirmation gate is simple: the device asks the user to respond (tap or voice), and non-response escalates to a trusted contact or emergency services. Enrollment is opt-in and framed for the at-risk population, which raises the prior probability of a true event and directly addresses the base-rate problem. False positives cost a contact check-in on someone who is fine; false negatives equal the status quo.

### 4. Assault/struggle detection

Assault/struggle classification fuses three signal families: inertial struggle patterns (repeated jolts, grappling oscillations from accelerometer and gyroscope), PPG-derived heart-rate and heart-rate-variability deviation against a personalized baseline (never a static heart-rate threshold, which produces exercise false positives), and microphone distress acoustics (screams, shouting, impact sounds). Context features include barometric altitude change during an attack and GPS context.

This disclosure explicitly states the precision problem. Using NCVS figures (22.5 violent victimizations per 1,000 persons age 12+ per year), a detector with 99% sensitivity and 99% specificity evaluating once per user-day produces roughly 0.0223 true alarms and 3.65 false alarms per user-year, a precision of about 0.6%, or roughly one real alarm in 164. Even at 99.9% specificity, precision is about 5.8%. A controlled 98.4%-accuracy study on scripted aggressive movements says nothing about real-world generalization, and no ethically collectable labeled dataset of real assaults exists at scale.

The honest survival path is therefore the confirmation gate: a discreet, cancellable prompt whose false alarms cost a wrist tap, as with fall detection's cancel window, not a 911 call. Automated emergency dispatch on assault-classifier output alone, without the confirmation window elapsing, is explicitly excluded from this disclosure. The confirmation prompt for assault-type events defaults to vibration and screen, not an audible alarm, because an audible challenge may tip off an attacker.

### 5. Gunshot acoustic detection

Gunshot detection classifies impulse noise from the device microphone using a convolutional-network impulse classifier running on-device. Documented failure modes are disclosed: close-range screaming false-alarms the classifier and co-occurs with attacks; urban impulse noise (backfires, construction, fireworks) false-alarms at municipal scale; and a microphone occluded in a pocket or bag degrades the impulse signature.

Accordingly, a single-device gunshot classification never triggers automated dispatch. Escalation requires multi-device corroboration: compatible acoustic-impulse reports from two or more nearby devices, exchanged over Bluetooth or equivalent peer discovery, arriving within a short time window and localized by GNSS-synchronized time-difference-of-arrival. Only a corroborated cluster proceeds to the confirmation window and then to staged escalation. This is the same multi-device architecture as the phone-network gunshot-localization literature, reused here as a false-positive gate rather than only as a localization method.

### 6. Duress PIN

The system includes a deterministic duress input: a second PIN, gesture, or biometric variant that outwardly performs the requested action (unlock, payment, compliance) while silently transmitting an alarm and location to emergency contacts or services. Precedent exists in patents for second duress passcodes (US9805586B1). This is a user-interface input, not a sensing claim. Passive detection of coercion, duress, or abduction from biometric stress or situational context is explicitly not claimed here: silence or a normal-looking dismissal cannot reliably prove coercion, and a probabilistic auto-escalation in a coercion event can tip off the attacker.

### 7. What is not claimed

For clarity of scope, this disclosure does not claim: passive duress or coercion sensing; abduction or kidnapping detection from route deviation, speed, or tilt signatures; choking detection; or real-time domestic-violence escalation prediction from wearable signals. Each of these was evaluated and cut as an invention claim: abduction signals fire on ordinary errands and the phone is typically absent in real abductions; choking kills faster than any confirmation gate allows and acoustic separators are undemonstrated; coercion classification has no demonstrated signal separator and the highest harm-of-being-wrong. A duress PIN is claimed only as the deterministic input described above.

## Implementation Notes

The classifier stack is deliberately layered: cheap on-device anomaly features (energy envelopes, jerk statistics, apnea timers) gate a heavier fused model so the always-on cost stays within a watch power budget. Personalized baselines for heart rate and heart-rate variability are learned over days of wear so that exercise and caffeine baselines do not trigger distress hypotheses. The confirmation window duration and modality (audible vs vibration-only) are configurable per event class at enrollment time, with assault-type events defaulting to silent. The multi-device corroboration exchange uses short-lived anonymous beacons over Bluetooth; no identity is transmitted until the user has confirmed escalation or a trusted-contact escalation fires for an enrolled at-risk population.

## Claims

1. A danger-detection system comprising: a wrist-worn device carrying an accelerometer, a gyroscope, a barometer, a microphone, a photoplethysmography heart sensor, and a GPS receiver; an on-device classifier producing danger-event hypotheses from fused signals of the sensors; and a staged escalation pipeline of on-device classification, a user confirmation window with a cancel path, an automatic emergency-services call relaying device location and available medical identification, and notification of enrolled emergency contacts; wherein the pipeline advances to the automatic call only after the confirmation window elapses without cancellation.
2. The system of claim 1, wherein the escalation pipeline comprises: stage one, the on-device classifier issuing a danger-event hypothesis; stage two, a discreet prompt with a countdown and a cancel input; stage three, the automatic emergency-services call with latitude, longitude, and medical identification relay; and stage four, emergency-contact notification; and wherein no stage may be skipped for assault-type or gunshot events.
3. The system of claim 1, further comprising an overdose detector configured to detect a respiration/apnea signature from the photoplethysmography sensor and the accelerometer, the signature comprising prolonged apnea or breathing at or below seven breaths per minute; wherein overdose detection is enabled by opt-in at-risk enrollment; and wherein the confirmation window for a detected overdose asks the user to respond, and non-response escalates to a trusted contact or emergency services.
4. The system of claim 1, further comprising an assault/struggle classifier fusing inertial struggle patterns from the accelerometer and gyroscope, heart-rate and heart-rate-variability deviation against a personalized baseline from the photoplethysmography sensor, and distress acoustics from the microphone; wherein automated emergency dispatch on the assault/struggle classifier output alone, without the confirmation window elapsing, is excluded.
5. The system of claim 4, wherein the assault/struggle classifier is base-rate gated: the disclosure records that at 99% sensitivity and 99% specificity with one evaluation per user-day against a 22.5-per-1,000 annual violent-victimization base rate, precision is approximately 0.6%; and wherein the system is therefore configured so that classifier false alarms cost a user cancellation rather than an automated emergency call.
6. The system of claim 1, further comprising a gunshot acoustic detector comprising a convolutional-network impulse classifier running on the microphone signal; wherein a single-device gunshot classification alone never triggers automated dispatch.
7. The system of claim 6, further comprising a multi-device corroboration gate: the device exchanges compatible acoustic-impulse reports with nearby devices over peer discovery, and escalation proceeds only when two or more devices report compatible impulses within a time window, localized by GNSS-synchronized time-difference-of-arrival; wherein the corroborated cluster then enters the confirmation window of claim 2.
8. The system of claim 1, wherein the confirmation prompt for assault-type events is vibration and screen only, with no audible alarm, to avoid alerting a nearby attacker.
9. The system of claim 1, further comprising a duress input comprising a second PIN, gesture, or biometric variant that outwardly performs a requested action while silently transmitting an alarm and location; wherein the duress input is a deterministic user-interface input and not a passive coercion detector.
10. The system of claim 1, wherein the system does not claim passive detection of coercion, abduction, or choking, and these event classes are explicitly excluded from the scope of the claimed detection.
11. A method for wearable-based danger response, comprising: fusing accelerometer, gyroscope, barometer, microphone, photoplethysmography, and GPS signals on a wrist-worn device to classify a danger event; presenting a discreet confirmation prompt with a cancel path; upon the prompt elapsing without cancellation, automatically calling emergency services with device location and medical identification; and notifying enrolled emergency contacts.
12. The method of claim 11, wherein the danger event is a gunshot classification, and wherein the method further comprises withholding automated dispatch until a multi-device corroboration gate is satisfied by compatible acoustic-impulse reports from at least two nearby devices within a time window.

## Prior Art References

1. Use Fall Detection with Apple Watch, Apple Support: wrist fall detection with immobility gate, 30-second countdown, automatic emergency call with location and Medical ID relay
2. Apple debuts iPhone 14 Pro and iPhone 14 Pro Max, Apple Newsroom, September 2022: Crash Detection sensor stack (dual-core accelerometer to 256 G, high-dynamic-range gyroscope, barometer, GPS, microphone) and 1M+ hours of training data
3. US11276290B2, Apple: fall-detection patent describing wrist trajectory, impact, and posture classification with automated escalation
4. Smartphone app detects opioid overdose, University of Washington News, January 2019: Second Chance sonar breathing monitor detected overdose-related breathing problems about 90% of the time in 94 supervised-injection participants
5. Wearable injector can detect and reverse opioid overdose, UW Newsroom: closed-loop wearable naloxone injector with dual-accelerometer respiration tracking
6. Smartwatch-based aggressive-behavior classification study, PMC: above 98.4% accuracy classifying scripted aggressive vs non-aggressive movements; controlled setting limits real-world generalization
7. Personalized context-aware distress recognition system, Scientific Reports, 2025: heart rate, temperature, GPS, and accelerometer fusion for autonomous distress recognition; warns static heart-rate thresholds produce exercise false positives
8. Gunshot detection and localization thesis, Teemu Patsi: CNN impulse classification on a consumer phone microphone with GNSS-synchronized time-difference-of-arrival localization across phone nodes
9. Deep-learning gunshot detection prototype, MDPI Acoustics: documents false positives from close-range screaming
10. US9805586B1: second duress passcode or gesture that outwardly performs the requested action while silently transmitting an alarm
11. Criminal Victimization, 2023, Bureau of Justice Statistics: 22.5 violent victimizations per 1,000 persons age 12 or older, the base-rate figure used in the precision analysis
