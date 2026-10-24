# PA-2026-213: Sleep Bruxism Detection from In-Ear Bone-Conducted Acoustic Sensing

**Title:** System and Method for Detecting Sleep Bruxism Using In-Ear Bone-Conducted Acoustic Sensing, Bilateral Jaw Motion Correlation, and Autonomic Arousal Fusion in Wireless Earbuds

**Filing:** LITF-PA-2026-213
**Published:** October 24, 2026
**Domain:** Sleep Health / Wearables / Sensor Fusion
**Full Disclosure:** [liveinthefuture.org/priorart/earbud-sleep-bruxism-detection.html](https://liveinthefuture.org/priorart/earbud-sleep-bruxism-detection.html)
**License:** [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/) — Public Domain

> Prior Art Notice: This document is a defensive technical disclosure published
> to constitute prior art under [35 U.S.C. Sec. 102(a)(1)](https://www.law.cornell.edu/uscode/text/35/102),
> dedicated to the public domain as of the publication date above.

---

## Abstract

Disclosed is a system and method for detecting sleep bruxism with ordinary wireless earbuds worn during sleep. Each earbud contributes three signals that no single channel can carry alone: an inward-facing microphone that picks up tooth contact sounds conducted through bone and the ear canal wall, an inertial sensor that records jaw and head motion, and a photoplethysmography sensor that tracks heart rate. An on-device fusion module scores candidate grind and clench events only when the acoustic, motion, and arousal evidence agree, then groups bursts into episodes using the rhythmic masticatory muscle activity rules used in sleep medicine. The result is a nightly bruxism episode index, computed without a sleep laboratory, without electrodes glued to the face, and without sending raw audio off the device. Most people who grind their teeth at night never learn it from a clinician. They learn it from a cracked filling, a sore jaw at breakfast, or a partner who finally complains. This disclosure puts that detection into hardware millions of people already own.

## Technical Field

This disclosure relates to sleep health monitoring with consumer wearables, specifically to detecting masticatory muscle activity during sleep from sensors already present in wireless earbuds. It sits at the intersection of in-ear acoustics, inertial sensing, and cardiovascular signal processing: repurposing an earbud's inward microphone, motion sensor, and heart rate sensor as a combined bruxism instrument, where the diagnostic value comes from the agreement between channels rather than from any one channel in isolation.

## Background

Sleep bruxism is masticatory muscle activity during sleep that is rhythmic (phasic) or non-rhythmic (tonic), under the international consensus definition updated by Lobbezoo and colleagues in 2018. The consensus treats bruxism in otherwise healthy people as a behavior and a risk factor, not a movement disorder or a sleep disorder in itself, while recognizing that sustained or forceful activity can damage teeth and restorations, strain the jaw joints and muscles, and produce morning headache and facial pain. Reported prevalence in adults runs roughly 8 to 31 percent depending on the measurement method, which already hints at the core problem: the condition is common, and the measurement tools are scarce.

The reference standard is overnight polysomnography with masseter electromyography, audio, and video. A trained scorer counts rhythmic masticatory muscle activity episodes, defined by bursts of jaw muscle activity, and reports them per hour of sleep as the bruxism episode index. Research criteria commonly treat fewer than 2 episodes per hour as no bruxism, 2 to 4 as mild to moderate, and more than 4 as severe. That standard is precise and almost nobody gets it. A sleep laboratory night costs hundreds to thousands of dollars, requires a referral, and covers one night, while bruxism varies substantially from night to night. A single laboratory night can miss a habitual grinder or catch an unusually quiet night and understate the problem.

What people actually use today is weaker:

- **Self-report and partner report:** Free, but unreliable. Many bruxers sleep alone, and a partner's report depends on how loudly the person grinds, how lightly the partner sleeps, and whether either of them remembers in the morning.
- **Clinical signs:** Tooth wear, fractured restorations, and jaw muscle tenderness on examination. These signs show cumulative damage, not current activity, and wear has many causes besides grinding. By the time wear is obvious, the damage is done.
- **Single-channel home devices:** Bite splints with embedded force sensors, and headbands or patches that record one jaw muscle. These measure one signal at one site, cost extra hardware, and are uncomfortable enough that adherence drops after the novelty wears off.
- **Bedside audio recorders:** A phone or microphone on the nightstand can hear loud grinding, but it also hears snoring, a partner, a fan, and traffic. It cannot tell whether a sound came from the sleeper's teeth, the sleeper's throat, or the other side of the bed.

Earbuds change the geometry of the problem. An earbud sits millimeters from the temporomandibular joint, coupled to the skull through the ear canal. Sound generated by teeth striking each other reaches that microphone largely by bone conduction, through tissue and bone, while room sounds arrive mostly by air conduction through the ear tip seal. That difference in path is measurable. Jaw motion also deforms the ear canal and moves the earbud measurably, and bruxism episodes are frequently accompanied by brief autonomic arousals that raise heart rate for seconds. No existing consumer product combines these three facts into one detector. The gap in the art is a detector that: (a) uses hardware the sleeper already wears, (b) separates tooth sounds from the many other sounds of a bedroom at night using conduction path and bilateral timing, (c) separates grinding from chewing, talking, coughing, and swallowing using motion and sleep context, and (d) reports a sleep-medicine-compatible episode index over many nights instead of one.

## Detailed Description

### 1. Sensor Hardware

Each earbud in the preferred embodiment contains hardware already standard in premium wireless earbuds: an inward-facing microphone used for active noise cancellation feedback, a 6-axis inertial measurement unit (accelerometer and gyroscope), a photoplethysmography sensor for heart rate, and a low-power processor with a few hundred kilobytes of memory for on-device inference. No new sensor is required. An optional embodiment adds an ear-canal pressure sensor that measures slow deformation of the canal wall as the jaw opens, closes, and shifts laterally; the pressure channel is helpful for sustained clenching, which makes little sound, but the system is specified to work without it.

The two earbuds exchange timestamped feature summaries over their existing low-energy link, or each bud processes its own channels and the phone reconciles them in the morning. Either arrangement works. What matters is that left and right channels share a common clock accurate to a few milliseconds, because bilateral timing is one of the discriminators in section 4.

### 2. Signal Acquisition and Sleep Gating

Recording runs only inside a sleep window. The window opens when the earbuds detect a sustained period of low head motion, a horizontal head orientation held for minutes, and a low ambient light or user-confirmed bedtime, and it closes on sustained upright posture, large head motion, or removal of either bud. This gating is the first defense against the most embarrassing false positive in the field: scoring daytime chewing, talking, or eating as bruxism. A candidate event outside the sleep window is discarded before classification, not merely down-weighted.

Within the window, the inward microphone samples at 16 kHz, the inertial sensors at 100 Hz, and the photoplethysmography sensor at 50 Hz or higher. Raw audio never leaves the earbud. The microphone stream is reduced on device to short frames of spectral features and burst envelopes, and those features are all that any later stage sees. A user can delete a night's features from the phone app, and deletion removes the derived episode list with it.

### 3. Tooth Contact Burst Detection

Tooth contact produces short, impulsive acoustic bursts. A grind is a train of such bursts as opposing tooth surfaces slide across each other; a clench is a sustained contraction that may produce one onset burst, a low-level creak, or nothing audible at all, which is why the acoustic channel alone is never sufficient. The detector works on 25-millisecond frames with 10-millisecond steps and computes, per frame, band energy in the 300 Hz to 4 kHz range, spectral flatness, zero-crossing rate, and a bone-conduction score defined below.

The bone-conduction score is the ratio of energy in the inward microphone to energy in the outward-facing microphone over the same frame, after both are normalized by their long-term nighttime baselines. A sound generated inside the head, at the teeth, reaches the inward microphone strongly and the outward microphone weakly, giving a high ratio. A sound generated in the room reaches the outward microphone first and at least as strongly, giving a low ratio. Frames with high bone-conduction score and impulsive spectral shape become candidate burst onsets. A burst is retained only if its energy exceeds the sleeper's own nighttime noise floor by a margin set during calibration, not by a fixed factory threshold, because ear canal coupling varies widely between people and between nights as the bud settles.

### 4. Bilateral Correlation and Jaw Motion Confirmation

Grinding involves both sides of the jaw. The detector therefore requires, for a burst to count as a grind burst, that a matching burst appear in the other earbud within 30 milliseconds with a similar envelope shape, or that a single-sided burst coincide with a jaw motion signature described below. This bilateral rule rejects most room sounds, which arrive at the two ears with interaural level and time differences set by the direction of the source and by the head shadow, and which rarely match in envelope shape at both ears at once. It also rejects single-sided artifacts such as a bud shifting against the pillow, which produces a loud transient in one ear only and a large inertial spike in the same bud.

The jaw motion signature comes from the inertial sensors. Rhythmic grinding drives the mandible in a roughly 1 Hz chewing-like pattern of small lateral and vertical excursions, superimposed on an otherwise still head. The detector extracts the component of earbud acceleration in the 0.5 to 2 Hz band, after removing gravity and slow head drift, and measures its rhythmicity over a 2-second window. A clench shows a different signature: a sustained shift in earbud orientation and canal pressure lasting more than 2 seconds, with little rhythmic motion, accompanied by a slow rise in the low-frequency acceleration baseline as the jaw muscles tense and hold. Head turns and pillow adjustments produce large, brief, broadband inertial transients that do not resemble either pattern and are rejected by shape, not merely by size.

### 5. Autonomic Arousal Fusion

Bruxism episodes frequently coincide with brief arousals: a small heart rate rise, a change in heart rate variability, and sometimes a body micro-movement in the seconds around the episode. The fusion module computes a heart rate trace from the photoplethysmography signal, smooths it over 5 seconds, and flags an arousal when heart rate rises at least 5 beats per minute above the surrounding 60-second baseline within 10 seconds before or after a candidate burst train. Arousal is supportive evidence, never required evidence, because many genuine episodes show no measurable heart rate change and because snoring-related arousals also raise heart rate. A candidate with acoustic, bilateral, and motion evidence scores as an episode with or without arousal; a candidate with acoustic evidence only, and no motion or arousal support, is held as unconfirmed and excluded from the nightly index.

### 6. Episode Grouping and the Nightly Index

Confirmed bursts are grouped into episodes using the rhythmic masticatory muscle activity rules adapted from sleep medicine. Bursts separated by less than 3 seconds belong to the same episode. An episode is phasic when it contains at least 3 bursts, each lasting 0.25 to 2 seconds; tonic when it contains a single sustained burst longer than 2 seconds; and mixed when it contains both. Episodes separated by 3 seconds or more are distinct. The system then reports the bruxism episode index: confirmed episodes divided by hours of sleep in the window, using the same thresholds as the research criteria (under 2, 2 to 4, over 4) as interpretive bands, not as a diagnosis. Alongside the index it reports total episode count, phasic versus tonic versus mixed counts, median episode duration, a force proxy computed from burst energy normalized to the sleeper's own calibration distribution, and the clock times of the longest episodes.

Two trend views matter more than any single night. A 14-night rolling median smooths the notorious night-to-night variability of bruxism, and a week-over-week comparison flags sustained rises that might follow a new medication, a stressful period, or increased evening alcohol or caffeine, all of which the user can annotate in the app. The morning report is deliberately boring: a number, a band, a small chart, and one line saying whether the night was typical for this sleeper. It does not wake the sleeper, does not sound an alarm during an episode, and does not claim to diagnose a disorder.

### 7. Differential Classification Against Nighttime Confounders

The classifier assigns each candidate to one of the following classes, and only the first two count toward the index:

- **Grinding:** bilateral impulsive bursts with rhythmic jaw motion, high bone-conduction score, sleep-window gated.
- **Clenching:** sustained jaw orientation shift over 2 seconds with canal pressure or low-frequency motion change, with or without an onset burst, plus arousal support or bilateral onset.
- **Snoring:** periodic low-frequency bursts tied to the breathing cycle, strong outward microphone energy, low bone-conduction score, no rhythmic jaw motion.
- **Coughing:** single explosive broadband transient with a large chest and head motion spike, no burst train, no sustained jaw shift.
- **Sleep talking:** voiced speech with formant structure and harmonic pitch, strong outward microphone energy, jaw motion that tracks speech articulation rather than rhythmic grinding.
- **Swallowing:** single low-frequency thump with a brief throat motion signature, isolated, no burst train.
- **Partner or room sound:** low bone-conduction score, asymmetric bilateral timing and level, no jaw motion.
- **Pillow and bud artifact:** single-sided transient with a large inertial spike in the same bud, no bilateral match, no jaw pattern.

Classification uses a compact on-device model, under 200 kilobytes, combining a small convolutional network over burst spectrograms with gradient-boosted trees over the engineered features (bilateral delay, bone-conduction ratio, rhythmicity, arousal delta, sleep-window context). Inference runs on the earbud processor or the phone, in either case before morning, and the model is trained on labeled nights where earbud features are time-aligned with laboratory masseter electromyography and audio-video scoring as ground truth.

### 8. Calibration, Reports, and Clinician Handoff

Calibration runs over the first 14 nights. The system learns the sleeper's nighttime noise floor, typical earbud coupling, resting heart rate during sleep, and baseline jaw motion, then sets personal thresholds as percentiles of that baseline rather than fixed values. A user who grinds quietly with excellent bud coupling and a user who grinds loudly with a loose fit get different absolute thresholds and comparable episode counts, which is the point. After calibration, thresholds update slowly with an exponentially weighted average so that a gradual change in fit or physiology does not masquerade as a change in bruxism.

The clinician report packages the 14-night median index, the distribution of nightly indices, the phasic and tonic breakdown, the force proxy distribution, and the three longest episodes with their times. It states plainly that earbud detection is a screening measure, that polysomnography with masseter electromyography remains the reference standard, and that treatment decisions, including an occlusal splint, belong to a dentist who has examined the teeth. The report also flags a pattern worth a dentist's attention on its own: a rising force proxy with a stable episode count, which can indicate harder grinding rather than more frequent grinding, and which a count-only index would miss entirely.

### 9. Figures Description

- **Figure 1:** System overview showing a sleeper wearing two earbuds, with arrows tracing bone-conducted sound from the teeth to each inward microphone and air-conducted room sound reaching the outward microphones.
- **Figure 2:** Signal traces for one grinding episode: bilateral burst envelopes aligned within milliseconds, the 1 Hz jaw motion trace, and the heart rate rise around the episode.
- **Figure 3:** Decision flowchart from sleep-window gating through burst detection, bilateral and motion confirmation, arousal fusion, and episode grouping to the nightly index.
- **Figure 4:** Confounder comparison panel showing burst envelope shapes for grinding, snoring, coughing, sleep talking, and pillow artifact side by side.
- **Figure 5:** Morning report mockup with the nightly index, the 14-night median, the phasic and tonic breakdown, and the force proxy trend.

## Claims

1. A system for detecting sleep bruxism, comprising: a pair of wireless earbuds, each containing an inward-facing microphone, an inertial sensor, and a photoplethysmography sensor; and an on-device fusion module that detects tooth contact bursts from the inward microphones, confirms the bursts with jaw motion from the inertial sensors, and computes a nightly bruxism episode index from confirmed bursts grouped into episodes.
2. The system of claim 1, wherein the fusion module computes a bone-conduction score for each acoustic frame as the ratio of inward microphone energy to outward microphone energy, and treats a high bone-conduction score as evidence that a sound originated at the teeth rather than in the room.
3. The system of claim 1, wherein the fusion module requires bilateral confirmation for a grind burst, comprising a matching burst envelope in the other earbud within 30 milliseconds, or single-sided acoustic evidence combined with a jaw motion signature.
4. The system of claim 1, wherein the fusion module extracts a rhythmic jaw motion signature in the 0.5 to 2 Hz band from earbud acceleration after removing gravity and slow head drift, and distinguishes rhythmic grinding motion from sustained clenching orientation shifts lasting more than 2 seconds.
5. The system of claim 1, wherein the fusion module computes an autonomic arousal signal from the photoplethysmography heart rate trace, flags a heart rate rise of at least 5 beats per minute above a surrounding baseline within 10 seconds of a candidate burst train as supportive evidence, and excludes acoustic-only candidates lacking motion or arousal support from the nightly index.
6. The system of claim 1, further comprising a sleep-window gate that enables scoring only during sustained low head motion with horizontal head orientation, and discards candidate events outside the sleep window before classification.
7. The system of claim 1, wherein the fusion module groups confirmed bursts into phasic, tonic, and mixed episodes using burst separation under 3 seconds, phasic bursts of 0.25 to 2 seconds, and tonic bursts over 2 seconds, and reports episodes per hour of sleep as the nightly index.
8. The system of claim 1, wherein the fusion module calibrates detection thresholds over a 14-night baseline period as percentiles of the individual sleeper's nighttime noise floor, earbud coupling, resting sleep heart rate, and jaw motion, and updates those thresholds with an exponentially weighted average thereafter.
9. The system of claim 1, further comprising a differential classifier that assigns candidates among grinding, clenching, snoring, coughing, sleep talking, swallowing, partner or room sound, and pillow or bud artifact, and counts only grinding and clenching toward the nightly index.
10. A method for detecting sleep bruxism with wireless earbuds, comprising: recording inward microphone, inertial, and photoplethysmography signals during a sleep window; reducing the microphone signal on device to spectral features without transmitting raw audio; detecting impulsive tooth contact bursts with a bone-conduction score; confirming bursts by bilateral timing and jaw motion; fusing a heart rate arousal signal as supportive evidence; grouping confirmed bursts into episodes; and reporting a nightly bruxism episode index with a 14-night rolling median and a force proxy normalized to the sleeper's calibration distribution.

## Prior Art References

1. Updating the Bruxism Definitions: Report of an International Consensus Meeting: the 2018 consensus definition of sleep bruxism as rhythmic (phasic) or non-rhythmic (tonic) masticatory muscle activity during sleep, and the grading of assessment from possible to definite diagnosis with instrumental measurement. https://pmc.ncbi.nlm.nih.gov/articles/PMC12408978/
2. Moving beyond bruxism episode index: the bruxism episode index as episodes of rhythmic masticatory muscle activity per hour of sleep, with research criteria bands under 2, 2 to 4, and over 4 episodes per hour, and the limits of a count-only index as a severity measure. https://pmc.ncbi.nlm.nih.gov/articles/PMC11744249/
3. Systematic Review on the Link between Sleep Bruxism and Systemic Chronic Inflammation: background on the 2013 and 2018 consensus definitions, the shift from viewing bruxism as a parafunctional act to viewing it as masticatory muscle activity with possible protective as well as harmful roles, and the 8 to 31 percent prevalence range. https://www.mdpi.com/2076-3425/13/7/1104
4. Frequency of Sleep Bruxism Behaviors in Healthy Young Adults over a Four-Night Recording Span in the Home Environment: home measurement of sleep-time masticatory muscle activity over four nights with a portable electromyography device, showing tonic and phasic events and night-to-night variability that motivates multi-night measurement. https://www.mdpi.com/2076-3417/11/1/195?type=check_update&version=1
5. LITF-PA-2026-083, Earbud Otoacoustic Emission Hearing Dosimetry: prior LITF disclosure using earbud sensors for in-ear health measurement; this disclosure differs by targeting masticatory muscle activity during sleep with multi-channel fusion rather than hearing assessment. https://liveinthefuture.org/priorart/earbud-oae-hearing-dosimetry.html
6. LITF-PA-2026-108, Wearable PPG Carbon Monoxide Detection: prior LITF disclosure using wearable photoplethysmography for physiological inference; this disclosure differs by using heart rate only as arousal support for an acoustic and motion primary detector. https://liveinthefuture.org/priorart/wearable-ppg-carbon-monoxide-detection.html
7. LITF-PA-2026-001, Wearable Gait Neurodegeneration Screening: prior LITF disclosure using wearable inertial sensing for health screening; this disclosure differs by using earbud inertial signals for jaw motion during sleep rather than gait during waking activity. https://liveinthefuture.org/priorart/wearable-gait-neurodegeneration-screening.html
8. 35 U.S.C. Sec. 102: conditions for patentability, including the printed publication and public availability provisions under which this disclosure is published. https://www.law.cornell.edu/uscode/text/35/102

## Implementation Notes

This disclosure describes an untested design proposal; quantitative parameters are design choices awaiting field calibration, and no accuracy claim is made against laboratory polysomnography until that validation exists. A minimal prototype needs no custom hardware: a pair of earbuds that exposes inward microphone audio, inertial data, and heart rate over a developer interface, plus a phone app that implements sleep gating, burst detection, and episode grouping. Where earbud firmware does not expose raw inward audio, the same design runs on the phone from the earbuds' audio stream during a sleep session, with the bilateral timing requirement relaxed to envelope matching within 50 milliseconds to absorb Bluetooth jitter.

Bring-up procedure: wear both buds for 14 calibration nights, confirm in the morning report that the sleep window opened and closed at plausible times, and check that the bone-conduction score separates a deliberate tooth tap performed while awake (high ratio, bilateral, brief) from a hand clap in the room (low ratio, asymmetric). Then verify the confounder classes one at a time: a recorded snore sample played in the room should classify as room sound or snoring, never as grinding, and a pillow adjustment should classify as artifact. If either test fails, the personal noise floor calibration is the first thing to recheck, since a mis-set floor corrupts every later stage.

Expected performance envelope (design targets for a validation program, not measured results): episode-level sensitivity target at least 80 percent against laboratory scoring on the same nights, false episode rate under 1 per hour on non-bruxer control nights, and nightly index agreement within 1 episode per hour on at least 70 percent of nights after calibration. Single-sided wear, meaning only one bud worn, disables bilateral confirmation and falls back to requiring both jaw motion and arousal support for every burst; the report marks single-sided nights as reduced confidence rather than silently mixing them with bilateral nights.

Limitations: silent clenching without tooth contact and without measurable jaw shift will be missed, since the system detects the mechanical consequences of muscle activity rather than the muscle activity itself. People who wear an occlusal splint will show a different acoustic signature, because the contacting materials change, and the calibration period must be repeated after starting or stopping splint use. Earbuds that fall out mid-night truncate the sleep window, and the index divides by recorded sleep hours only, which the report states. This is a screening tool for discussion with a dentist, not a diagnosis, and a high index alone does not establish a need for treatment, since bruxism in otherwise healthy people is a risk factor whose clinical significance depends on tooth condition, pain, and other findings a dentist assesses in person.

This disclosure describes an untested design proposal; quantitative parameters are design choices awaiting field calibration (see Implementation Notes).

---

*Published at [liveinthefuture.org/priorart](https://liveinthefuture.org/priorart/)*
