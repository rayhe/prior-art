# System and Method for Detecting Hidden Damage in Carbon Fiber Bicycle Frames Using Smartphone-Guided Tap-Test Acoustic Analysis with Per-Frame Baseline Learning

**LITF-PA-2026-236 · Cycling / Composite Inspection**
**Published:** 2026-11-16
**License:** [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/) - Public Domain

> ⚖️ **Prior Art Notice:** This document is published as a printed publication and defensive prior art under [35 U.S.C. § 102(a)(1)](https://www.law.cornell.edu/uscode/text/35/102). It is intended to serve as prior art against later patent claims covering the inventions described herein. This disclosure is dedicated to the public domain under CC0 1.0.

---

Tap your frame with a coin, the way riders always have. Now the phone listens, remembers how your frame sounded when it was healthy, and tells you whether to ride or not. Carbon fiber does not dent like steel or bend like aluminum. It hides its damage between layers, under intact paint, and fails suddenly once a crack forms. The only test most riders have is tapping the frame with a coin and listening, a test that depends on the rider's ear and catches only large damage. This disclosure turns the phone in the rider's pocket into a quantitative tap-test instrument: it guides a grid of taps, records each one, compares it to the same frame's own baseline, and scores every point for delamination and cracking before the next ride.

*A frame can look flawless while having already lost significant strength. The tap test is the only check most riders have, and it depends entirely on the rider's ear.*

## Abstract

Disclosed is a system and method for detecting hidden structural damage in carbon fiber bicycle frames using a smartphone-guided acoustic tap test. A phone application guides the user through a grid of tap points covering the frame's critical zones, records the acoustic response of each tap with the phone microphone, and extracts quantitative features including dominant resonant frequency, decay time constant, spectral centroid, and band energy. Each point is scored against two references: the same frame's own baseline, recorded when the frame was new or known to be undamaged, and the symmetric point on the opposite side of the frame. An on-device classifier trained on labeled tap recordings from frames with ultrasonically verified damage converts the features into a per-point probability of delamination or cracking and renders a damage map with a ride, inspect, or do-not-ride recommendation. A crash-detection trigger prompts a focused re-test of the impact zones after any detected crash, and an optional clip-on ultrasonic accessory confirms flagged points. The system replaces a subjective listening test with a repeatable measurement, using hardware the rider already owns. It is a screen, not a certification: a green result means no damage was found by this screen, not that the frame is proven sound.

**By the numbers:**

- **$0** is what the base system costs: the phone in your pocket and a coin, against a $7,000 professional rig.
- **$7,000** is the Cycle Inspect ultrasonic rig with training and software, the professional route to a confirmed answer (Bicycle Retailer, 2022).
- **$30 to $60** is the target price for the clip-on tapper, and **$150 to $300** for the ultrasonic accessory, both estimates.
- **60 to 120** tap points cover a full frame in about ten minutes. A focused test after a crash takes about three.
- **10% vs 25%:** the Boeing RD³ digital tap hammer detects a 10% pulse-width change, against the roughly 25% threshold of human hearing.
- **$3.48 billion** was the carbon fiber bike frames market in 2025 by one estimate, projected to reach $6.79 billion by 2034 (Dataintelo). Other reports put the frame market far lower, so treat market figures as scope-dependent.

**Send this to someone who rides carbon and has ever crashed it.**

## Field of the Invention

This invention relates to nondestructive inspection of composite structures, specifically to consumer detection of hidden delamination and cracking in carbon fiber bicycle frames using smartphone acoustic analysis, per-frame baseline learning, and crash-triggered re-testing.

## Background

Carbon fiber frames were a $3.48 billion market in 2025 by one estimate, projected to reach $6.79 billion by 2034 (Dataintelo), with road bikes taking the largest share. Other reports put the frame market far lower, a second Dataintelo report puts frames at $0.808 billion inside a $1.91 billion components market, so treat market figures as scope-dependent. Average selling prices for carbon fiber frames run $1,200 to $4,500, against $400 to $800 for aluminum frames (Dataintelo). The material earned that share by being light and stiff, and it keeps a dark secret. Steel dents. Aluminum bends. Carbon fiber hides its damage between layers of cloth and resin, under paint that can look flawless, and then fails suddenly once a crack forms. Delamination, a separation between layers of carbon cloth, sits beneath intact paint and is frequently invisible from outside, while the frame has already lost significant strength. A frame can look perfect and still be broken inside.

Five methods exist today for finding that damage, and each one leaves a hole:

- **Visual inspection:** Raking light across the tubes catches surface cracks, paint whitening, spiderweb crazing, and creases. It cannot see delamination that has not reached the surface, and that is the damage that matters most. A clean visual check is reassuring. It is not proof the frame is undamaged.
- **Coin tap test:** The standard home method. Hold a coin, tap gently along the frame, and compare the sound to a known-good area of the same tube. Intact laminate gives a crisp note. A delaminated area sounds dull and muffled. The method is real and it works, within limits. Cycle Inspect, which sells professional inspection equipment to bike shops, states the limits plainly: accuracy depends on the listener's auditory skill and on tapping technique, audible sound has limited penetration into the laminate, and only large damage reliably changes the tone.
- **Professional ultrasonic inspection:** Ruckus Composites in Portland maps internal delamination with an Evident 45MG ultrasonic thickness gauge, and Cycle Inspect sells an ultrasonic rig with training and software for about $7,000. Both work. Neither is in the rider's garage after a crash on a Sunday ride, and neither is priced for a pre-purchase check on a used bike.
- **Instrumented tap hammers:** Boeing's RD³, patented as US 6,748,791 (2004) and sold by WichiTech, is a handheld tap hammer with an accelerometer in its head that measures impact pulse width and gives a quantitative, recordable numeric readout, detecting a 10% pulse-width change against the roughly 25% threshold of human hearing. It already quantifies the tap test and already compares each reading to a known-good baseline. It is an aircraft tool, priced for airlines and repair depots, with no frame map, no learned per-point baseline, no classifier, and no connection to a rider's phone.
- **Automated tap testing:** US11,624,690 (2023) discloses an automated tap tester, a linear stage that moves a tapping member across a composite surface and models subsurface damage from the acoustic signals. Its method starts from surface damage that has already been identified, then models the subsurface damage beneath it. It is an industrial rig for composite structures in general. It does not address bicycle frames, per-frame baselines, consumer use, or crash-triggered testing.

Crash detection from phone inertial sensors is established technology, in Apple Crash Detection and Garmin incident detection, and this disclosure claims no novelty in detecting a crash. The novelty is what the crash triggers: a focused frame re-test scored against the frame's own baseline. Mirror-symmetry comparison is likewise standard manual tap-test practice, and the claim here is its automated, quantified form, scored against the frame's own data rather than the rider's ear.

What does not exist is a consumer system that: (a) quantifies the tap test on hardware the rider already owns, with learned per-point baselines instead of a single known-good reference; (b) learns the individual frame, so each point is judged against its own healthy sound rather than a generic threshold; (c) re-tests the right zones after a crash without waiting for a shop visit; and (d) turns the result into a specific recommendation the rider can act on. That is the gap this disclosure addresses.

## Detailed Description

### 1. System Overview

Three parts make up the system. A phone application runs the test, records audio, and scores the results on-device. A tapping implement, a coin, a plastic screwdriver handle, or an optional clip-on solenoid tapper described in the Implementation Notes, excites the frame at each grid point. An optional clip-on ultrasonic accessory, described in section 8, confirms flagged points. No part of the frame is modified, cut, or drilled. The phone needs only its built-in microphone, sampling at 44.1 or 48 kHz, and its inertial sensors. Tap energy in carbon laminate is estimated to concentrate between roughly 1 and 8 kHz, a proposed range pending measurement, well inside the phone microphone passband, so no external microphone is required for the base system. All inference runs on the phone. No cloud connection is needed to complete a test.

On first use the app asks for the frame: make, model, and year where known, or a generic tube map for an unknown frame. Frame geometry matters because tap tone changes with tube diameter, wall thickness, and layup, so a down tube and a chainstay never sound alike even on a healthy frame. Where the manufacturer publishes tube dimensions, the app uses them to set expected frequency bands per tube. Where it does not, the baseline session described in section 4 establishes them empirically.

### 2. Guided Grid Tap Test

Testing starts with a grid, not a freeform wander. The app overlays a tap grid on a diagram of the frame, with points spaced 3 to 5 cm apart along each tube, denser in the critical zones: the head tube junctions, the down tube behind the head tube, the bottom bracket shell, the seat tube cluster, the chainstays near the dropouts, and the fork blades. These zones carry the highest load and the highest crash exposure. A full-frame test covers roughly 60 to 120 points and takes about ten minutes. A focused test after a crash covers only the zones near the impact, typically 15 to 30 points, and takes about three minutes.

At each point the app prompts the tap, records a 100 ms window starting 5 ms before the detected tap onset, and then validates the tap before accepting it. Rejection rules are explicit: a tap whose peak amplitude falls below a floor is too soft and is retried; a window containing two onsets is a double tap and is retried; a window whose background noise floor exceeds a threshold, from wind, traffic, or a running fan, is retried. Three consecutive rejections at a point mark the point as untestable rather than guessing. The app also records a 500 ms ambient sample before each session and subtracts its spectrum from the tap spectra, so a test done in a noisy garage is not silently corrupted.

Consistency is the whole game, and the app enforces it. It shows a target tap force on screen, and an absolute floor applies so a bad start cannot corrupt the calibration: the first accepted taps must reach a peak amplitude of at least -20 dBFS, proposed, and any session whose first three accepted taps fall below that floor prompts the rider to tap harder and restarts calibration. If tap amplitudes drift by more than 6 dB from the calibration band mid-session, the app re-prompts the force target. It paces the taps, one per second at most, so the frame fully rings down between taps and one tap never contaminates the next. A rider tapping by hand will still vary, and the feature extraction in section 3 is designed to tolerate that variation by leaning on ratios and decay shapes rather than absolute amplitude.

### 3. Acoustic Feature Extraction

Each accepted tap window yields a feature vector. Delamination changes the local stiffness of the laminate, and stiffness changes the sound in three measurable ways, which is why a single number is never enough:

- **Dominant resonant frequency:** The peak of the tap spectrum, found by FFT with zero padding for interpolation. A delaminated region is locally less stiff, so its resonance drops. Healthy carbon tubes are estimated to ring between 2 and 6 kHz depending on tube and wall thickness, a proposed range pending measurement; a drop of more than 15% from the point baseline is flagged.
- **Decay time constant:** The exponential fit to the Hilbert envelope of the bandpassed tap signal. Delamination adds internal friction, so the ring dies faster. Healthy taps are estimated to decay with time constants of roughly 8 to 25 ms, proposed; a delaminated spot is estimated to cut that in half. Decay is the most reliable feature because it depends on damping, not on how hard the rider tapped.
- **Spectral centroid and band energy:** The energy-weighted mean frequency and the share of energy in the 1 to 3 kHz, 3 to 5 kHz, and 5 to 8 kHz bands. A crack is expected to add nonlinear harmonic content and shift energy toward lower bands. A dull thud is, quantitatively, a low centroid with fast decay.
- **Spectral flatness:** The ratio of geometric to arithmetic mean of the spectrum. A healthy tap is tonal, with a sharp peak. A cracked region produces a noisier, flatter spectrum as energy scatters across frequencies.

All features are computed on-device in under 50 ms per tap, a design target. Nothing leaves the phone during a test. The feature vector per point, four numbers plus a confidence flag, is stored with the point location, the tap timestamp, and the ambient noise estimate.

### 4. Per-Frame Baseline Learning and Symmetry Reference

Generic thresholds would fail here. Two healthy frames of the same model can ring differently because layup, paint thickness, and resin content vary, and a down tube never sounds like a chainstay. The system therefore judges each point against that point's own history, not against a population average.

A baseline session runs when the frame is new, or immediately after a professional inspection clears it. The app records the full grid twice, on different days, and stores the per-point feature means and variances. Every later test compares each point to its own baseline with per-feature z-scores, combined into a Mahalanobis distance that accounts for how the features co-vary at that point. A point whose distance exceeds a threshold, proposed at 3.0 standard deviations, is flagged for review. Because the baseline is per-point, a frame with an unusual but healthy tone never triggers false alarms, and a frame whose tone changes after a crash triggers them immediately.

A second reference covers frames with no baseline, including every used bike bought before the app existed. Most frames are left-right symmetric: the left and right chainstays, seatstays, and fork blades are mirror images with the same layup. The app compares each point to its mirror twin and flags asymmetries beyond a threshold, proposed at a 20% difference in resonant frequency or a 30% difference in decay constant. Healthy asymmetry is expected, since asymmetric layups, internal cable routing, and bottle bosses on one side all shift tone, so the thresholds are set loose on purpose and symmetry flags stay advisory. Symmetry is weaker than a baseline, because manufacturing is never perfectly symmetric, so a symmetry flag is advisory where a baseline flag is a warning. Both references are reported separately, so the rider sees which evidence produced the flag.

Baselines drift legitimately. Temperature changes resin stiffness, so the app records ambient temperature at each session and applies a correction, proposed at 0.3% frequency shift per degree Celsius, fitted from the frame's own sessions across seasons. A baseline older than 12 months prompts a re-baseline, because slow aging, paint wear, and accumulated micro-damage shift the frame's tone even without a crash.

### 5. On-Device Damage Classifier

Feature drift alone cannot distinguish delamination from a crack, and the distinction matters: a crack can propagate under load while delamination grows slowly, so a crack earns a do-not-ride recommendation where delamination earns an inspect recommendation. A classifier makes the call.

A gradient-boosted tree ensemble, under 200 KB, serves as the classifier, trained on labeled tap recordings. Training data comes from three sources: frames with damage verified by ultrasonic scanning or CT, with each tap point labeled healthy, delaminated, or cracked; frames tapped before and after controlled impact testing, with the impact energy and location recorded; and healthy frames of many models, tapped across seasons to teach the classifier what normal variation looks like. Data augmentation covers the rest: recorded taps are re-rendered at different amplitudes, with added garage noise, and with simulated temperature shifts, so the classifier learns features that survive real-world conditions.

Per point, the classifier outputs three probabilities: healthy, delaminated, cracked. Calibration matters more than raw accuracy, so the model is calibrated with isotonic regression on a held-out set, and the app displays the probability alongside the recommendation, never a bare verdict. A point at 55% delamination reads as uncertain and earns an inspect recommendation, not a false certainty. The classifier is updated over the air as the labeled dataset grows, with each update versioned so a frame's test history records which model version scored it.

### 6. Damage Map and Recommendation

Results land as a map, not a spreadsheet. The app renders the frame diagram with each tested point colored green, amber, or red: green means within baseline and classifier-healthy, amber means a symmetry flag or a borderline classifier score, red means a baseline breach or a crack probability above a threshold, proposed at 0.6. Untested points show as gray. Tapping a point shows its features, its baseline comparison, and its classifier probabilities.

Above the map sits one recommendation, in plain language. Green across the board reads differently depending on the evidence behind it. A baseline-backed green reads "No damage found by this screen. This is not a certification. Ride with normal caution." A symmetry-only green reads "No damage found by comparison only. No baseline exists for this frame. Ride with normal caution and arrange a professional inspection." Any red point reads "Do not ride. Get a professional ultrasonic inspection." Amber without red reads "Inspect before your next hard ride: a visual self-check now, and a professional ultrasonic inspection before any hard riding. Easy riding only until then, and re-test in one week. If any point carries a crack probability above 0.4, do not ride until the professional inspection clears it." The app states the basis for the recommendation, naming the flagged points and the evidence behind each, so the rider can show the map to a shop instead of describing a sound.

Every test exports as a report: the frame diagram with the map, the per-point table, the classifier version, and the session conditions. The report serves three uses the disclosure names explicitly: a buyer checking a used carbon bike before paying, a seller proving a frame was inspected, and an insurance claim documenting a crash. Reports are signed with a hash of the session data so a later reader can verify the report matches the recorded taps.

### 7. Crash-Triggered Re-Test

Most frame damage happens in crashes, and most riders skip inspection after the crashes that feel minor. The system closes that gap with a trigger. A companion app on the rider's phone, or a paired bike computer, watches the inertial sensors during rides. A deceleration spike above a threshold, proposed at 6 g sustained for more than 20 ms, or a free-fall signature followed by an impact, registers as a crash event with an estimated impact direction from the accelerometer axes.

When a crash is detected, the app prompts a focused re-test as soon as the rider stops, naming the zones nearest the estimated impact. If the rider dismisses the prompt, it returns at the next ride start and stays pending until completed or explicitly waived, with the waiver logged. A manual "I crashed" button covers crashes the sensors miss, including low-speed tip-overs that still crack a top tube against a curb. The re-test compares against the pre-crash baseline, so even a small change lands as a flag rather than disappearing into noise.

Crash data stays on the phone unless the rider opts into sharing. Sharing is anonymized and feeds the fleet learning in section 9, because crash energy and damage patterns across many riders are the only way to learn which impacts actually break frames.

### 8. Optional Ultrasonic Confirmation Accessory

A tap test is a screen, not a verdict, and a red flag deserves confirmation before a rider writes off a $1,200 to $4,500 frameset. The accessory is a clip-on single-element ultrasonic thickness gauge, in the class of the Evident 45MG used by professional inspectors, simplified to one frequency and one measurement mode, with a consumer target price of $150 to $300. It clamps to the tube with a coupling gel pad, fires a pulse, and measures the echo time from the far wall of the laminate.

Guidance comes from the tap map: the app directs the accessory to flagged points only, rather than asking the rider to scan the whole frame. At each flagged point the gauge takes five readings and reports the median wall thickness and the echo clarity. Delamination shows as a thickness reading that jumps between measurements, or as an early echo from the separated layer. The app fuses the ultrasonic result with the tap score: a red tap point confirmed by an anomalous ultrasonic reading becomes a confirmed-damage verdict, while a red tap point with a clean ultrasonic reading drops to amber with a re-test recommendation. The accessory never overrides a red tap score to green on its own, because a single-point ultrasonic reading can miss damage sitting millimeters away.

### 9. Fleet Learning and Batch Defect Detection

When riders opt in, anonymized per-point feature vectors and classifier outcomes upload to a shared dataset, with no identity, no location, and no frame serial number, only a rotating random device identifier and the frame make and model. Two things come back. First, the classifier improves: taps from frames whose damage was later verified by a shop become labeled training data, and the model updates ship to all devices. Second, batch defects surface. When a statistically significant share of one frame model shows correlated drift in the same zone, for example premature delamination at the seat tube cluster across dozens of frames of one model within months of each other, the system issues a batch alert naming the model and zone, suitable for forwarding to the manufacturer and to safety regulators. This converts isolated rider anecdotes into evidence of a defective production run.

### 10. Figures Description

- **Figure 1:** System overview showing the phone, the frame on a stand, the tap grid overlaid on the frame diagram, and the optional clip-on tapper and ultrasonic accessory.
- **Figure 2:** Tap waveforms and spectra for a healthy point and a delaminated point on the same tube, showing the frequency drop, faster decay, and lower spectral centroid of the damaged response.
- **Figure 3:** Damage map rendering of a frame with green, amber, and red points, with the recommendation banner and per-point detail panel.
- **Figure 4:** Baseline learning diagram showing per-point feature distributions at baseline and after a crash, with the Mahalanobis distance threshold marked.
- **Figure 5:** Crash-triggered re-test flow, from inertial crash detection through impact-zone selection to the focused grid test and updated recommendation.

## Claims

1. A system for detecting hidden damage in a carbon fiber bicycle frame, comprising: a smartphone application that guides a user through a grid of tap points covering critical zones of the frame; a microphone of the smartphone that records the acoustic response of each tap; a feature extraction unit that computes, for each tap, dominant resonant frequency, decay time constant, spectral centroid, and band energy; a baseline unit that stores per-point feature values recorded when the frame was known to be undamaged; and a scoring unit that compares each tap against the stored baseline for the same point and renders a damage map with a ride, inspect, or do-not-ride recommendation.
2. The system of claim 1, wherein the scoring unit further compares each tap point to its mirror-symmetric point on the opposite side of the frame, flagging asymmetries beyond a threshold as advisory evidence of damage for frames lacking a recorded baseline.
3. The system of claim 1, further comprising an on-device classifier trained on labeled tap recordings from frames with ultrasonically verified damage, wherein the classifier outputs per-point probabilities of healthy, delaminated, and cracked states, and wherein the recommendation distinguishes delamination from cracking.
4. The system of claim 1, wherein the application validates each tap before acceptance, rejecting taps below an amplitude floor, double taps, and taps recorded above a background noise threshold, and marking points with repeated rejections as untestable rather than scored.
5. The system of claim 1, further comprising a crash-detection trigger that monitors inertial sensors during rides, registers a crash event from a deceleration spike or free-fall-plus-impact signature, and prompts a focused re-test of the frame zones nearest the estimated impact direction.
6. The system of claim 1, further comprising a clip-on solenoid tapper that delivers repeatable excitation force at each grid point, wherein the application controls tap timing and force to reduce user technique variation.
7. The system of claim 1, further comprising a clip-on ultrasonic thickness gauge accessory that measures wall thickness and echo clarity at points flagged by the tap test, wherein the application fuses the ultrasonic result with the tap score to confirm or downgrade flagged points.
8. The system of claim 1, further comprising a fleet learning unit that receives anonymized per-point feature vectors from a plurality of devices, refines the classifier from shop-verified damage outcomes, and detects correlated feature drift across frames of a common model as a production batch defect.
9. The system of claim 1, wherein the application exports a signed report comprising the damage map, per-point feature values, classifier version, and session conditions, for use in used-bike purchase inspection, sale documentation, and insurance claims.
10. A method for detecting hidden damage in a carbon fiber bicycle frame, comprising: recording a per-point acoustic baseline of the frame when it is known to be undamaged; guiding a user through a grid tap test of the frame's critical zones; recording the acoustic response of each tap with a smartphone microphone; extracting resonant frequency, decay time constant, spectral centroid, and band energy from each tap; scoring each point against its own baseline and its mirror-symmetric point; classifying each point as healthy, delaminated, or cracked using an on-device classifier; and rendering a damage map with a ride, inspect, or do-not-ride recommendation.

## Implementation Notes

Bill of materials for the base system is zero beyond the phone: the app is the product, and a coin is the tapper. The optional clip-on solenoid tapper targets $30 to $60 at volume: a small solenoid, a spring return, a Bluetooth Low Energy radio, and a coin-sized striking face, powered by a coin cell for roughly 500 taps, a target. The app controls the tapper over Bluetooth, firing one tap per second at a fixed force, which is what makes the excitation repeatable. The optional ultrasonic accessory targets $150 to $300: a single-element 5 MHz transducer, a pulser-receiver, and a coupling gel pad in a clamp housing sized for tube diameters from 25 to 60 mm. The classifier fits in under 200 KB and runs in milliseconds per point on a modern phone, both targets. The app stores baselines and test history locally, encrypted at rest, and syncs only on opt-in. Frame geometry data for common models ships in the app; unknown frames fall back to a generic tube map with wider grid spacing. The system is positioned as a screening tool, not a certification: every red result directs the rider to professional ultrasonic inspection, and the app says so on the result screen.

Phone placement matters. The microphone should sit 20 to 40 cm from the tap point, unobstructed, which the app enforces by asking the rider to prop the phone on a stand or a nearby surface rather than holding it in the tapping hand. Tapping hand and recording hand must be different hands, or the phone picks up handling noise. The app checks the first three taps for handling noise signatures and re-prompts placement if it finds them.

## Limitations

This disclosure describes an unbuilt system. No prototype has been constructed, no accuracy figure has been measured, and every threshold in it is a proposed value. The 15% resonant frequency drop, the 3.0 Mahalanobis distance threshold, the 0.6 crack probability cutoff, and the 6 g crash trigger are design targets, not measured results. The labeled training dataset the classifier needs does not exist yet at scale; collecting it requires shops to tap frames before ultrasonic verification, and that collection effort is the largest barrier to building this system. The tap test's depth problem is physical, not just human: audible sound has limited penetration into laminate, so quantifying the tap improves repeatability without extending its reach, and deep delamination between inner plies, damage under thick paint, and damage near bonded joints will still evade it. A green result is reassurance, not proof. Paint thickness, aftermarket protective film, and frame bags clamped to a tube all change the tap tone and can produce false flags. Temperature correction is proposed, not validated. Finally, the system inspects only the frame. Forks, handlebars, seatposts, and wheels are also carbon, also fail, and are not covered here except as future work.

## Strongest Case Against

The honest objection is that the tap test's limits are physical, and quantifying a limited test does not remove its limits. Audible sound cannot reach deep delamination, thick paint defeats it, and bonded joints already sound irregular on a healthy frame, so a system built on tap acoustics will miss exactly the damage that matters most while occasionally flagging a healthy frame and sending its rider to a shop for a $150 inspection of nothing. A green score could give false confidence, and false confidence on a cracked frame is worse than no test at all, because the rider who never tested would at least ride cautiously after a crash. A second objection is behavioral: most riders will never run a ten-minute tap test, and the riders who crash hard enough to damage carbon usually know it and go to a shop anyway, so the system serves a narrow slice of crashes, the ones that feel minor but are not. Both objections are fair. The answer to the first is scope discipline: the system is a screen that catches some hidden damage, and it says so on every result screen rather than claiming certification. The answer to the second is that the marginal cases are the point: the crash that felt minor, the used bike bought from a stranger, the slow crack growth between seasons. Even a screen that caught only a share of hidden damage would still beat an ear that catches only the largest damage, provided nobody mistakes the screen for a verdict.

*The crashes that feel minor are the ones that hide damage.*

## What a Cyclist Can Do Now

None of this needs to be built before it helps. Inspect a carbon frame under raking light after every crash, looking for spiderweb crazing, whitening, and creases in the clear coat. Tap the tubes with a coin and compare each spot to the symmetric spot on the other side; a dull, hollow note where its mirror twin rings crisp is a reason to stop riding. Learn the critical zones: head tube junctions, down tube, bottom bracket, chainstays. After any crash, do not ride until the frame is checked, even when the crash felt minor, because the crashes that feel minor are the ones that hide damage. Buy a used carbon bike only with a professional ultrasonic inspection, or price the inspection into the purchase. Use a torque wrench on every bolt, because over-torquing cracks carbon, and never clamp a carbon frame by its tubes in a workstand; clamp the seatpost. If a frame is ever in doubt, the cost of an inspection is a fraction of the cost of a frame failing at speed.

**Know someone who just bought a used carbon bike? Send them this.**

## Prior Art References

1. US11,624,690: Device and method for examining a composite structure; automated tap tester with linear stage (2023): https://patents.justia.com/patent/11624690
2. US5505090A: Method and apparatus for non-destructive inspection of composite materials; acoustic pulse with laser Doppler vibrometry (1996, expired): https://patents.google.com/patent/US5505090A/en
3. US 6,748,791: Georgeson et al., Boeing, Damage detection device and method; RD³ electronic digital tap hammer with accelerometer head measuring impact pulse width against a known-good baseline (2004): https://patentimages.storage.googleapis.com/c2/48/45/b1126f6b9b204f/US6748791.pdf
4. Cycle Inspect: Tap testing, why it is not a reliable choice for the cycling industry; human-ear limits, technique dependence, limited penetration: https://www.cycleinspect.com/insights/tap-testing/
5. Bicycle Retailer: Cycle Inspect ultrasonic inspection program for retailers; about $7,000 for equipment, training, and software: https://www.bicycleretailer.com/industry-news/2022/10/21/australian-start-markets-carbon-fiber-inspection-program-retailers-and
6. Ruckus Composites / Evident: Ultrasonic thickness gauge inspection of carbon fiber bike frames; Evident 45MG: https://www.wabtecims.com.cn/en/insight/ruckus-composites-inspects-thousands-of-bikes-with-one-ultrasonic-thickness-gage/
7. Domex Bikes: Carbon frame damage signs; sudden fracture mode, delamination under intact paint, coin tap test method: https://domexbikes.co.uk/mechanics-blog/bike-care/carbon-frame-damage-signs/
8. Yoeleo: Post-crash carbon inspection; tap test grid, comparison to known-good areas, limits at depth and near joints: https://www.yoeleobike.com/en-de/blogs/frames/how-to-check-carbon-frame-after-crash
9. BikeLabStudio: Carbon frame damage detection; visual inspection, coin tap test, critical zones, repair vs discard criteria: https://www.bikelabstudio.com/articles/danos-cuadro-carbono.html
10. Dataintelo: Carbon fibre bike frames market; $3.48 billion in 2025, projected $6.79 billion by 2034: https://dataintelo.com/report/carbon-fibre-bike-frames-market
11. Dataintelo: Carbon fiber gear market; average selling prices for carbon fiber frames $1,200 to $4,500, against $400 to $800 for aluminum frames: https://dataintelo.com/report/carbon-fiber-gear-market
12. Resonance, Vibration Analysis: Smartphone vibration analysis app; generic FFT and damping estimation, not a frame inspection system: https://play.google.com/store/apps/details?id=com.morganandwife.vibration
13. LITF-PA-2026-208: Bicycle drivetrain chain wear detection via acoustic and cadence analysis; related acoustic bicycle sensing, distinct target (chain, not frame): https://liveinthefuture.org/priorart/bicycle-drivetrain-chain-wear-acoustic-cadence.html
14. LITF-PA-2026-215: FDM 3D printer nozzle partial obstruction detection; related acoustic sensing of manufactured parts, distinct domain: https://liveinthefuture.org/priorart/fff-nozzle-partial-obstruction-detection.html

---

*Published at [liveinthefuture.org/priorart](https://liveinthefuture.org/priorart/)*
