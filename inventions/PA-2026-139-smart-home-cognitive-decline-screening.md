# PA-2026-139: System and Method for Non-Intrusive Screening of Cognitive Decline Using Longitudinal Analysis of Smart Home Appliance Interaction Patterns with Sequential Activity Modeling and Behavioral Drift Detection via Edge-Deployed Temporal Transformers

**Filing:** LITF-PA-2026-139  
**Domain:** Digital Health / Edge AI / Ambient Computing  
**Published:** August 13, 2026  
**Type:** Defensive Prior Art Disclosure  

---

## Abstract

Disclosed is a system and method for continuous, non-intrusive screening of early-stage cognitive decline in residential occupants by analyzing longitudinal patterns of interaction with existing smart home devices. The system ingests timestamped event logs from smart speakers, smart thermostats, smart lighting controls, smart locks, kitchen appliances, and entertainment systems already present in the home, requiring no additional hardware installation and no wearable compliance. A local hub (e.g., Matter/Thread border router, Home Assistant instance, or purpose-built edge appliance) runs an edge-deployed temporal transformer model that learns the household's baseline interaction grammar: the characteristic sequences, timing distributions, repetition rates, and contextual associations of device interactions for each identified occupant. The model continuously computes a composite Behavioral Regularity Index (BRI) across seven cognitive domains mapped from device interaction features: executive function (multi-step task completion patterns), episodic memory (redundant interaction frequency), temporal orientation (circadian adherence of routine activities), attention/concentration (task-switching fragmentation rate), semantic memory (voice command vocabulary diversity), visuospatial navigation (lighting and room-transition patterns), and processing speed (interaction-to-completion latency trends). Statistical process control charts (CUSUM and EWMA) applied to each domain's BRI component detect sustained directional drift exceeding learned within-subject variability. When drift in two or more domains exceeds a configurable threshold for 14+ consecutive days, the system generates a structured screening report delivered to a designated caregiver or primary care physician via a secure channel, containing the drift trajectories, anonymized interaction exemplars, and a recommendation for formal neurocognitive evaluation. All processing occurs on-device. No raw interaction data, voice recordings, or personally identifiable information leaves the home network.

## Field of the Invention

This invention relates to digital health screening, ambient assisted living, and edge computing for behavioral analytics, specifically to the use of existing consumer smart home device interaction logs as a passive, longitudinal behavioral biomarker for early detection of neurodegenerative cognitive decline including mild cognitive impairment (MCI) and early-stage dementia.

## Background

Alzheimer's disease and related dementias affect an estimated 6.9 million Americans aged 65 and older (Alzheimer's Association, 2024 Facts and Figures), with total care costs exceeding $360 billion annually. Mild cognitive impairment, the prodromal stage that precedes dementia in many patients, affects an additional 15.6% of adults over 60 (Petersen et al., Mayo Clinic Proceedings 2018). The median time from MCI onset to clinical diagnosis is 2.8 years (Rasmussen & Bhatt, 2020), a gap during which disease-modifying interventions like lecanemab (van Dyck et al., NEJM 2023) and donanemab (Sims et al., JAMA 2023) are most effective but cannot be administered because the patient has not yet been identified.

Current screening approaches share a common problem: they require the person to know something is wrong.

- **Clinical cognitive tests:** The Montreal Cognitive Assessment (MoCA) and Mini-Mental State Examination (MMSE) require a clinic visit, take 10-15 minutes, produce a single-timepoint snapshot, and suffer from practice effects on repeated administration. Sensitivity for MCI detection is 0.90 at specificity 0.87 (Nasreddine et al., JAGS 2005) in clinical populations, but real-world screening uptake is poor: fewer than 16% of adults over 65 receive any cognitive assessment at annual wellness visits (Alzheimer's Association, 2019).
- **Digital cognitive tests:** Tablet and smartphone-based assessments (Cogstate, Linus Health, Neurotrack) improve accessibility but still require active participation, suffer from digital literacy confounds, and produce episodic rather than continuous data.
- **Wearable-based monitoring:** Actigraphy, gait analysis, and sleep tracking via wrist-worn devices can detect behavioral changes correlating with cognitive decline (Mc Ardle et al., Gait & Posture 2019), but compliance among elderly populations is approximately 50-60% after 6 months (Keogh et al., JMIR 2019). Charging requirements, skin irritation, and forgetting to wear the device undermine longitudinal data continuity.

The gap in the art is a screening system that: (a) requires zero active participation from the person being monitored, (b) uses hardware already installed in the home for other purposes, (c) builds a continuous longitudinal behavioral profile rather than episodic snapshots, (d) maps observable device interactions to established cognitive domains, (e) preserves privacy by processing all data locally, and (f) produces clinically actionable screening output that a physician can act on.

## Detailed Description

### 1. Data Ingestion Layer

The system operates as a software agent running on a local compute hub within the home network. It ingests event streams from smart home devices via standard protocols:

- **Matter/Thread devices:** The agent subscribes to cluster attribute change events (On/Off, Level Control, Thermostat, Door Lock clusters) via the Matter controller's event stream API.
- **Home Assistant:** The agent subscribes to `state_changed` events via the WebSocket API on the local instance.
- **Proprietary ecosystems:** For devices reporting only to cloud platforms (Amazon Alexa, Google Home, Apple HomeKit), the agent optionally pulls activity logs via authorized API access with user consent. Where local-only operation is preferred, the agent captures LAN traffic metadata (mDNS queries, CoAP messages, BLE advertisements) without decrypting payload content, using only timing and message-type patterns.

Each ingested event is normalized to a canonical tuple: `(timestamp, device_id, device_type, event_type, value, room_id, occupant_id)`. Occupant identification in multi-person households uses a probabilistic model combining: voice speaker identification from smart speaker events (where available as a metadata field, never raw audio), room-presence inference from motion sensor activation sequences, smartphone BLE proximity beacons, and habitual device-use timing profiles. The system explicitly handles shared devices by attributing interactions to the occupant most likely present, with a confidence score; interactions below 0.7 confidence are excluded from individual profiles and contributed only to household-level statistics.

### 2. Behavioral Feature Extraction

The system extracts 47 behavioral features grouped into seven cognitive domain proxies. The mapping from device interaction observables to cognitive domains draws on the DSM-5 criteria for Major and Mild Neurocognitive Disorder, which define six cognitive domains. We add a seventh (circadian temporal orientation) based on the established relationship between circadian rhythm disruption and neurodegeneration (Musiek & Holtzman, Science 2016).

| Cognitive Domain | Device Interaction Features | Feature Count |
|---|---|---|
| Executive Function | Multi-step task completion rate (e.g., unlock door → disarm alarm → turn on lights), task ordering consistency, abandoned multi-device sequences, novel-vs-routine task ratio | 8 |
| Episodic Memory | Redundant interaction frequency (re-locking already-locked doors, re-setting already-set thermostats, repeated identical voice queries within 30 min), check-back rate for completed actions | 7 |
| Temporal Orientation | Circadian drift of routine activities (meal preparation timing via kitchen appliance activation, sleep onset via bedroom light-off, wake time via first interaction), weekend/weekday pattern consistency, seasonal adjustment tracking | 8 |
| Attention / Concentration | Task-switching fragmentation rate (starting an activity in one room, moving to another room mid-task, returning), entertainment session duration stability, thermostat adjustment frequency within single comfort-seeking episodes | 6 |
| Semantic Memory | Voice command vocabulary diversity (unique command stems per week), command reformulation rate (rephrasing failed commands), proper noun usage in queries (names of contacts, places, media titles) | 6 |
| Visuospatial Navigation | Room transition sequence regularity, light-on-before-entry anticipation rate, novel path frequency through the home (inferred from sequential room sensor activations), nighttime navigation pattern changes | 6 |
| Processing Speed | Interaction-to-completion latency for goal-directed device sequences, voice command response selection time (for devices reporting confirmation delay), time from alarm trigger to dismissal | 6 |

Each feature is computed over rolling 7-day windows with 1-day stride. Raw feature values are z-score normalized against the occupant's own 90-day trailing baseline, producing a deviation score that captures within-subject change independent of absolute values. This design is critical: the system never compares one person to another. A person who has always adjusted their thermostat six times a day is normal at six; the signal is when six becomes twelve.

### 3. Temporal Transformer Architecture

The core model is a lightweight temporal transformer operating on the 47-dimensional feature vector time series. The architecture comprises:

- An input embedding layer that projects each 7-day feature vector (47 dimensions) into a 64-dimensional latent space with learned positional encoding.
- 4 transformer encoder layers with 4 attention heads, 128-dimensional feedforward layers, and GELU activation, with a total parameter count of approximately 180K (small enough for real-time inference on ESP32-S3 or Raspberry Pi Zero 2W).
- A domain-specific output head that produces 7 BRI component scores (one per cognitive domain), each normalized to [0, 1] where 1.0 represents the occupant's learned baseline and values below 1.0 indicate deviation.
- A composite BRI score computed as the attention-weighted mean of domain scores, where attention weights are learned during pre-training to reflect the relative diagnostic weight of each domain for MCI/dementia classification.

Pre-training uses a self-supervised masked feature prediction objective on a large corpus of synthetic smart home interaction sequences generated by an activity simulator calibrated against the CASAS Smart Home Dataset (Washington State University, 400+ participants, 30+ smart home testbeds) and the UCI HAR Dataset. Fine-tuning on real household data occurs entirely on-device during the 90-day baseline learning phase, using the occupant's own interaction patterns. No training data leaves the home. The model adapts to household-specific device configurations automatically: if a home has no smart speaker, the 6 semantic memory features are masked and the remaining 41 features are reweighted via attention redistribution.

### 4. Drift Detection Engine

Raw BRI scores fluctuate day to day due to illness, visitors, travel, and routine disruptions. The system distinguishes transient perturbations from sustained cognitive decline using two complementary statistical process control methods applied to each domain's BRI component independently:

- **CUSUM (Cumulative Sum Control Chart):** Detects sustained small shifts in BRI mean level. The CUSUM accumulates daily BRI deviations from the expected baseline, resetting when the deviation is in the favorable direction. A downward shift exceeding 0.5 standard deviations sustained over 14+ days triggers a drift flag.
- **EWMA (Exponentially Weighted Moving Average):** Smooths daily BRI scores with a decay factor λ = 0.15, producing a trend line that reveals directional movement while damping noise. The EWMA control limit is set at ±2.7σ of the baseline period's EWMA distribution. Exceeding the lower control limit for 7+ consecutive days triggers a drift flag.

A domain is flagged as "drifting" when both CUSUM and EWMA agree on sustained downward movement. The system generates a screening alert when two or more domains are simultaneously flagged for 14+ consecutive days. This dual-method, multi-domain requirement dramatically reduces false positives from single-domain transient causes.

### 5. Confound Handling

The system incorporates explicit confound detection to prevent false alerts:

- **Household composition changes:** Arrival/departure of visitors or co-residents is detected via occupant count changes (smart lock events, BLE device count, voice identification new-speaker flags). The system suspends drift scoring during detected guest periods and for 7 days following departure.
- **Device topology changes:** Addition, removal, or relocation of smart devices triggers a 30-day re-baselining window for affected features.
- **Acute illness:** Patterns consistent with acute illness (significantly increased time in bedroom, decreased kitchen appliance use, increased thermostat temperature setpoint) trigger a temporary "illness probable" flag that suspends alerting for up to 21 days.
- **Seasonal and daylight effects:** Circadian features are adjusted for local sunrise/sunset times (computed from latitude/longitude, no internet required) and DST transitions.
- **Medication effects:** If the occupant or caregiver reports a medication change via the companion app, the system opens a 45-day observation window with relaxed drift thresholds.

### 6. Output and Clinical Integration

The screening report generated upon alert includes: a 90-day time-series plot of composite BRI and per-domain BRI scores; identification of which domains triggered the alert, with magnitude and duration of drift; anonymized exemplar interaction patterns showing the behavioral change; comparison to published MCI behavioral signature profiles from the research literature; and an explicit disclaimer that this is a screening tool, not a diagnostic instrument, and that formal neurocognitive evaluation is recommended.

The report is delivered via: encrypted email to a designated caregiver email address; a secure API endpoint for integration with EHR systems supporting HL7 FHIR R4 DiagnosticReport resources; or a companion smartphone application with caregiver authentication. The occupant controls all data sharing permissions.

### 7. Privacy Architecture

All computation occurs on the local hub. The system never transmits raw event logs, voice recordings, or personally identifiable interaction data to any cloud service. The model runs on-device and updates on-device. The screening report contains only aggregate statistical measures and anonymized behavioral summaries. If the occupant or their legal representative revokes consent, all stored data and learned model weights are cryptographically erased. The system is designed to comply with HIPAA as a non-covered entity and with GDPR Article 9 requirements for processing health-related data with explicit consent.

## Claims

1. A system for non-intrusive cognitive decline screening comprising: a software agent running on a local compute hub within a residential network that ingests timestamped interaction event logs from two or more smart home devices of different categories; a behavioral feature extraction module that computes a plurality of behavioral features from said event logs and maps them to cognitive domains defined by recognized neurocognitive diagnostic criteria; a temporal transformer model running on-device that produces per-domain Behavioral Regularity Index scores by comparing current interaction patterns to the occupant's learned baseline; a drift detection engine applying statistical process control methods to said BRI scores to identify sustained directional deviation in two or more cognitive domains; and a report generation module that produces a structured screening report for delivery to a designated caregiver or healthcare provider.

2. The system of claim 1, wherein the behavioral features include: redundant interaction frequency as a proxy for episodic memory function, multi-step task completion rate as a proxy for executive function, circadian drift of routine activities as a proxy for temporal orientation, task-switching fragmentation rate as a proxy for attention, voice command vocabulary diversity as a proxy for semantic memory, room transition sequence regularity as a proxy for visuospatial navigation, and interaction-to-completion latency as a proxy for processing speed.

3. The system of claim 1, wherein all computation including model inference, feature extraction, drift detection, and report generation occurs entirely on a local device within the home network, with no raw interaction data transmitted to any external server.

4. The system of claim 1, wherein the temporal transformer model is pre-trained using self-supervised masked feature prediction on synthetic smart home interaction sequences and fine-tuned on-device using the occupant's own interaction data during a baseline learning period.

5. The system of claim 1, wherein the drift detection engine applies both CUSUM and EWMA statistical process control methods independently to each cognitive domain's BRI score, requiring agreement between both methods before flagging a domain as drifting.

6. The system of claim 1, further comprising a confound detection module that identifies and compensates for: household composition changes, device topology changes, acute illness patterns, seasonal and daylight effects, and caregiver-reported medication changes.

7. The system of claim 1, wherein occupant identification in multi-person households uses a probabilistic model combining voice speaker identification metadata, room-presence inference, smartphone BLE proximity, and habitual interaction timing profiles, with interactions below a configurable confidence threshold excluded from individual cognitive profiles.

8. A method for screening cognitive decline comprising: continuously ingesting interaction events from a plurality of smart home devices over a period of at least 90 days to establish a behavioral baseline; extracting behavioral features mapped to recognized neurocognitive domains from said interaction events using rolling temporal windows; computing a Behavioral Regularity Index for each cognitive domain using an on-device temporal transformer model; applying statistical process control analysis to detect sustained directional drift in BRI scores; and generating a screening alert when two or more cognitive domains exhibit simultaneous sustained drift exceeding a configurable threshold for a minimum consecutive duration.

9. The method of claim 8, wherein the behavioral features are z-score normalized against the occupant's own trailing baseline, such that the system detects within-subject change independent of absolute interaction rates and never compares the occupant's behavior to population norms.

10. The method of claim 8, wherein the screening alert includes per-domain drift trajectories, anonymized behavioral exemplars, comparison to published MCI behavioral signatures, and a recommendation for formal neurocognitive evaluation.

11. The method of claim 8, further comprising a privacy enforcement layer that processes all data on a local device, stores no raw voice recordings, encrypts all stored interaction logs, supports complete cryptographic erasure upon consent revocation, and delivers screening reports only through encrypted channels to authorized recipients.

12. The method of claim 8, wherein the temporal transformer model automatically adapts to the specific device configuration of each household by masking features derived from absent device categories and redistributing attention weights to available features, requiring no manual configuration.

## Implementation Notes

A minimal viable implementation requires: a Raspberry Pi 4 or equivalent local compute device ($35-75), running Home Assistant or a custom MQTT broker; 5+ smart home devices spanning at least 3 device categories, which are already present in an estimated 63% of US households as of 2025 (Statista); and Python 3.10+ with PyTorch or TensorFlow Lite for model inference. The temporal transformer model at 180K parameters requires approximately 720 KB of storage and performs inference in under 50 ms on a Cortex-A72 processor.

The 90-day baseline learning period is a practical requirement, not a theoretical minimum. Shorter baselines (30-60 days) are feasible for occupants with highly regular routines but produce higher false positive rates. Longer baselines (120-180 days) improve specificity at the cost of delayed screening onset.

Known limitations: the system cannot screen for cognitive decline in occupants who do not interact with smart home devices; multi-person households with similar interaction patterns may produce noisy occupant attribution; the system is not a diagnostic tool and cannot distinguish between MCI etiologies; and the efficacy of the screening approach has not yet been validated in a prospective clinical trial.

---

*Published at [liveinthefuture.org/priorart/smart-home-cognitive-decline-screening.html](https://liveinthefuture.org/priorart/smart-home-cognitive-decline-screening.html)*
