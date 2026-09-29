# PA-2026-187: Acoustic Escapement Authentication of Mechanical Timepieces

**Title:** System and Method for Authenticating Mechanical Timepieces Using Smartphone Acoustic Escapement Analysis with Machine-Learned Counterfeit Detection

**Filing:** LITF-PA-2026-187
**Published:** September 29, 2026
**Domain:** Horology / Acoustic Sensing / Authentication
**Full Disclosure:** [liveinthefuture.org/priorart/watch-acoustic-escapement-authentication.html](https://liveinthefuture.org/priorart/watch-acoustic-escapement-authentication.html)
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

Disclosed is a system and method for authenticating mechanical timepieces by recording the acoustic signature of the escapement with a consumer smartphone microphone and analyzing it with a combination of classical timing-parameter extraction and machine-learned spectral fingerprinting. The system extracts beat rate, beat error, and amplitude from the escapement's per-beat transient triplet, computes a convolutional neural network embedding of each beat's spectral structure, and compares both against a crowdsourced, watchmaker-verified reference database keyed by movement caliber. A positional-variation liveness challenge defeats replay spoofing: the user records the watch in two or more orientations, and the system verifies that the orientation-dependent rate shifts match the reference caliber's known positional signature. The output is a calibrated authenticity score distinguishing genuine movements from counterfeit and "superclone" replicas whose escapement geometry differs from the genuine caliber.

## Technical Field

This disclosure relates to horological authentication, specifically to non-invasive acoustic analysis of mechanical watch escapements using consumer mobile devices and machine learning for counterfeit detection.

## Background

The Swiss Customs Service estimates that 30 to 40 million counterfeit watches are put into circulation each year, and the Federation of the Swiss Watch Industry estimated in 2012 that counterfeit Swiss watch sales generated approximately $1 billion per year. Fake watches account for roughly 9 percent of customs seizures, second only to textiles among counterfeited product categories. The problem has intensified in the "superclone" era: replica factories now produce clone movements that match the genuine caliber's beat rate, decoration, and even component dimensions, so visual inspection, the traditional first line of defense for collectors and dealers, increasingly fails.

Existing acoustic analysis is confined to the watchmaker's bench. Timegrapher instruments such as the Weishi No. 1000 and No. 1900 measure rate deviation (±999 s/d), amplitude (100°–360°), and beat error (0–9.9 ms) through a contact microphone, with pre-programmed beat trains from 12,000 to 43,200 beats per hour and a default lift angle of 52°. These are diagnostic tools: they require the operator to preset the caliber's lift angle, they test in six fixed positions, and they perform no authentication against any reference. Laboratory alternatives exist, including elemental chemical profiling of watchcases by inductively coupled plasma mass spectrometry (University of Lausanne and the Federation of the Swiss Watch Industry, published in *Forensic Science International*, 2019), but these are destructive or near-destructive, expensive, and unavailable to consumers. No consumer-accessible, non-invasive method combines timing-parameter analysis with learned acoustic fingerprinting and anti-spoofing to authenticate a mechanical movement.

## Detailed Description

### 1. Escapement acoustic physics

Each beat of a Swiss lever escapement produces not a single click but a triplet of acoustic transients: the unlock (pallet stone releasing the escape-wheel tooth), the impulse (the tooth striking the pallet impulse face), and the drop (the tooth landing on the locking face). These three transients are spaced tens of microseconds apart, and their relative timing and spectral content encode the escapement's geometry: pallet stone angles, lift angle, escape-wheel tooth profile, and balance assembly mass distribution. A counterfeit movement that copies the beat rate but not the exact escapement geometry produces a triplet with measurably different inter-transient timing and spectral balance. Co-axial and other non-lever escapements produce structurally different transient patterns and are handled by separate trained models.

### 2. Signal acquisition protocol

The user holds the watch caseback 1–3 cm from the smartphone microphone in a quiet environment. The application measures ambient noise before capture and requires it below 40 dBA, displaying a live noise meter during a 3-second pre-roll. Audio is captured at 48 kHz, 16-bit mono, for 30 seconds per orientation. A guided interface prompts the user through the orientations required for the liveness challenge (Section 7), showing a diagram of each position. Captures with fewer than 20 cleanly detected beats, or with clipping on more than 1% of samples, are rejected with a request to re-record.

### 3. Timing-parameter extraction

The signal is band-pass filtered to 2–8 kHz, the band carrying escapement transient energy, and the analytic envelope is computed via the Hilbert transform. An adaptive threshold peak detector locates each beat's triplet; inter-beat intervals yield the beat rate in vibrations per hour (18,000, 21,600, 28,800, and 36,000 vph cover the great majority of calibers). Tick-tock asymmetry, the time difference between alternating beats, yields beat error in milliseconds; healthy, well-adjusted movements typically show beat error under 0.8 ms. Amplitude is computed from the time between the first and third transients of the triplet using the caliber's lift angle (default 52° for Swiss lever when the caliber is unknown, overridden by the database value when the caliber is identified). Rate deviation in seconds per day is computed against the nominal beat rate. All parameters are reported as medians with interquartile ranges over the capture window.

### 4. Spectral fingerprint embedding

Each detected beat triplet is converted to a log-mel spectrogram (64 mel bins, 25 ms analysis window, 10 ms hop, 2–8 kHz band) and passed through a compact convolutional neural network producing a 128-dimensional embedding. The network is trained with triplet loss on labeled recordings: anchor and positive beats from the same genuine caliber, negative beats from different calibers and from known counterfeit movements. The embedding captures escapement-geometry information that scalar timing parameters miss, such as the spectral balance between the unlock, impulse, and drop transients. The model is quantized to under 5 MB for on-device inference via TensorFlow Lite or Core ML, with inference latency under 200 ms per beat on 2020-era smartphone hardware.

### 5. Reference database construction

The reference database is keyed by movement caliber. Each entry stores: the caliber's nominal beat rate and lift angle; the median and covariance of the timing-parameter vector (rate deviation, beat error, amplitude) measured across verified genuine specimens; and the centroid and covariance of the embedding distribution. Entries are built from crowdsourced recordings contributed through the application and verified by participating watchmakers who confirm the specimen's provenance (service records, authorized-dealer purchase, or factory service). Each entry requires a minimum of 25 verified specimens spanning at least 5 contributing watchmakers before it is marked production-grade. The database is versioned, and every authenticity report cites the database version used.

### 6. Authenticity scoring

For a test recording, the system computes two distances against the claimed caliber's reference entry: the Mahalanobis distance of the timing-parameter vector, and the cosine distance of the mean beat embedding to the reference centroid. The two distances are fused by a logistic regression calibrated on a held-out set of genuine and counterfeit recordings, producing an authenticity score from 0 to 100. Scores above 80 are reported as "consistent with genuine," scores below 30 as "likely counterfeit," and scores between as "inconclusive," with the report always showing the underlying timing parameters so a watchmaker can interpret borderline cases. Thresholds are set to hold the false-accusation rate (genuine scored likely-counterfeit) under 1% on the calibration set.

### 7. Liveness: positional challenge and replay detection

A recording played back through a speaker could otherwise spoof the analysis. The system defeats replay with a positional challenge: the user records in at least two orientations (dial up and crown down at minimum). Genuine mechanical movements exhibit caliber-specific positional rate variation, typically 2–15 s/d between positions, caused by gravity acting on the balance assembly. The system checks that the measured inter-position rate delta falls within the reference caliber's recorded positional signature; a replayed single recording cannot produce correct orientation-dependent deltas. As a second layer, the system scans the 20–24 kHz band for the ultrasonic compression and transducer artifacts characteristic of loudspeaker playback, which are absent in direct acoustic recordings of a live escapement.

### 8. On-device implementation and privacy

Signal acquisition, filtering, peak detection, timing-parameter extraction, and embedding inference all run on the device. Only the timing-parameter vector, the mean embedding, and the claimed caliber are transmitted to the reference-database service; raw audio never leaves the phone. Per-device microphone calibration uses a one-time 1 kHz reference tone played by the application to normalize frequency-response differences across phone models. The reference database is also published as a downloadable snapshot so the full pipeline, including scoring, can run offline.

## Claims

1. A system for authenticating mechanical timepieces, comprising: a microphone-equipped mobile computing device configured to acquire an acoustic signal of the timepiece's escapement; a timing-parameter extractor configured to derive beat rate, beat error, and amplitude from per-beat transient triplets in the acoustic signal; a neural embedding module configured to compute a spectral fingerprint embedding of each beat; a reference database keyed by movement caliber storing genuine-specimen timing statistics and embedding centroids; and a scoring module configured to compare the extracted parameters and embeddings against the reference entry for a claimed caliber and output a calibrated authenticity score.
2. The system of claim 1, wherein the timing-parameter extractor identifies, within each beat, three acoustic transients corresponding to escapement unlock, impulse, and drop, and derives amplitude from the inter-transient timing using the caliber's lift angle.
3. The system of claim 1, wherein the neural embedding module converts each beat triplet to a log-mel spectrogram and processes it with a convolutional neural network trained with triplet loss on genuine-caliber and counterfeit recordings.
4. The system of claim 1, wherein the reference database is constructed from crowdsourced recordings verified by participating watchmakers, each caliber entry requiring a minimum number of verified specimens from multiple independent contributors before production use.
5. The system of claim 1, wherein the scoring module fuses a Mahalanobis distance of the timing-parameter vector with a cosine distance of the embedding, calibrated to hold the false-accusation rate under 1%.
6. The system of claim 1, further comprising a liveness module that requires acoustic captures in at least two physical orientations and verifies that the measured inter-orientation rate delta matches the claimed caliber's recorded positional signature, thereby defeating replayed recordings.
7. The system of claim 6, wherein the liveness module further scans the ultrasonic band above 20 kHz for loudspeaker playback artifacts.
8. The system of claim 1, wherein all signal processing and embedding inference execute on the mobile device and only derived parameters and embeddings are transmitted, with raw audio retained on-device.
9. The system of claim 1, further comprising per-device microphone calibration using an application-generated reference tone to normalize frequency response across phone models.
10. A method of building a horological acoustic reference database, comprising: receiving escapement recordings from contributors; verifying specimen provenance through participating watchmakers; extracting timing parameters and spectral embeddings from each verified recording; and aggregating per-caliber timing statistics and embedding centroids once a minimum specimen count from multiple independent contributors is reached.
11. A method for pre-purchase authentication of a mechanical timepiece in a marketplace transaction, comprising: guiding a seller through multi-orientation acoustic capture with a smartphone; computing an authenticity score per claim 1; and presenting the score with underlying timing parameters to the buyer before funds are released.
12. The system of claim 1, wherein a beat error distribution characteristic of clone movements, specifically elevated or bimodal beat error inconsistent with the claimed caliber's genuine distribution, contributes a negative weight to the authenticity score.

## Implementation Notes

The guided capture interface is the difference between a lab demo and a usable product: users must be told exactly how close to hold the phone, shown the live noise meter, and walked through orientations with diagrams. In testing the concept, the most common failure mode is ambient noise, not algorithm weakness; the 40 dBA gate and re-record prompting handle the majority of bad captures. Per-phone microphone calibration matters more than model size: an uncalibrated frequency response skews the spectral balance between transients and degrades the embedding. The downloadable offline database snapshot should be offered from the start, since authentication is most needed at in-person transactions where connectivity is unreliable. Database governance needs a dispute process: when a specimen's provenance is challenged, its recordings are quarantined and the caliber entry is recomputed without them.

## Prior Art References

1. [Counterfeit watch](https://en.wikipedia.org/wiki/Counterfeit_watch), Wikipedia: Swiss Customs estimate of 30–40 million counterfeit watches per year; Federation of the Swiss Watch Industry 2012 estimate of ~$1B/year in counterfeit Swiss watch sales
2. [Weishi Timegrapher 1900](https://welwynwatchparts.co.uk/products/weishi-timegrapher-1900-testing-machine-used), Welwyn Watch Parts: rate deviation ±999 s/d, amplitude 100°–360°, beat error 0–9.9 ms, pre-programmed beats 12,000–43,200, lift angle default 52°, six testing positions
3. [Weishi MTG-1000 Multifunction Timegrapher](https://dynagem.co.uk/collections/watch-rotation-timing-machines/products/mtg-1000-multifunction-timegrapher-watch-timing-machine-calibration-tools-tester), Dynagem: microphone-based mechanical movement timing for service centres
4. [Chemical profiling uncovers fake luxury watches](https://www.securingindustry.com/clothing-and-accessories/tuesday-chemical-profiling-uncovers-fake-luxury-watches/s107/a9730/), SecuringIndustry: University of Lausanne / FHS elemental analysis of counterfeit watchcases, *Forensic Science International* 2019; fake watches ~9% of customs seizures per FHS
5. [Weishi multifunction timegrapher documentation](https://manuals.plus/m/8dd0f57b5db0b10b275397d8c47ddbaad2b9d73c1283ccc92b788c5988afbab1.pdf), via Manuals.Plus: automatic beat detection, sampling periods 2–60 s, signal level auto-adjustment
