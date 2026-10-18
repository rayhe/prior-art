# PA-2026-207: Crowdsourced Escalator and Moving Walkway Mechanical Health Assessment from Passenger Phone IMU

**Title:** System and Method for Crowdsourced Escalator and Moving Walkway Mechanical Health Assessment Using Passenger Device Inertial Measurement Telemetry with Step-Band Periodicity Analysis and Fleet Anomaly Detection

**Filing:** LITF-PA-2026-207
**Published:** October 18, 2026
**Domain:** Urban / Predictive Maintenance / Sensor Fusion
**Full Disclosure:** [liveinthefuture.org/priorart/escalator-crowdsourced-passenger-imu-health.html](https://liveinthefuture.org/priorart/escalator-crowdsourced-passenger-imu-health.html)
**License:** [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/) — Public Domain

> Prior Art Notice: This document is a defensive technical disclosure published
> to constitute prior art under [35 U.S.C. Sec. 102(a)(1)](https://www.law.cornell.edu/uscode/text/35/102),
> effective as of the publication date above. It is intended to be cited as
> prior art against later-filed patent applications covering the subject matter
> described. The inventors, through this publication, reserve their rights
> under 35 U.S.C. Sec. 102(b)(1). The disclosure is intended to enable a
> person of ordinary skill in the art to make and use the described systems.
> This is general information, not legal advice.
>
> **Design proposal:** no prototype has been built and no field validation has
> been performed. All quantitative parameters are untested design choices.

---

## Abstract

A standing escalator rider has no gait, so their phone becomes a free accelerometer
bolted to the machine by gravity and friction. Dedicated escalator condition monitors
cost thousands of dollars per unit and reach a tiny fraction of the installed base;
passengers' phones are already riding every machine in the city and already carry
100 Hz 6-axis IMUs.

Disclosed is a system and method for assessing the mechanical health of escalators
and moving walkways using inertial measurement telemetry from smartphones carried by
ordinary passengers, with no installed sensors, no machine-side hardware, and no
cooperation from the equipment manufacturer or maintenance contractor. On-device
software detects ride segments (constant-velocity inclined motion with gait absent),
verifies the ride with barometric altitude slope, rotates the accelerometer stream
into the incline plane, and extracts a step-band periodicity signature: the comb-plate
impact line at step-pass frequency (rated speed divided by step pitch, typically
~1.23 Hz at 0.5 m/s over 0.405 m steps), the drive-train harmonic series, and
low-frequency modulation from step-band revolution (loop length divided by speed,
typically one revolution per 60 to 120 seconds). Because riders board at random times,
no single ride sees a full revolution; a cloud aggregation service phase-aligns ride
fragments by step-band phase and reconstructs the per-revolution vibration signature
from dozens of independent passenger observations per day. A per-machine statistical
baseline tracks the signature over weeks, and a change-point detector flags
fault-class deviations: step roller and track wear (impulses phase-locked to a
step-band sector), comb-plate misalignment (broadband transients at the step-pass
line), step-chain elongation (drift and broadening of the drive-train harmonic
series relative to the step-pass line), and lateral guide wear (growth of
cross-incline vibration). The system transmits only compact spectral feature vectors
(band powers at machine-synchronous frequencies, typically under 1 kB per ride),
never raw audio, video, or fine-grained location, and consumes well under one tenth
of one percent of a typical day's phone battery.

> *A single rider can never ride a full revolution, but strangers boarding at random
> phase tile the entire step band: the crowd's randomness substitutes for a
> slip-ring.*

---

## Technical Field

This disclosure relates to predictive maintenance of passenger conveyor systems,
specifically to crowdsourced mechanical health assessment of escalators and moving
walkways (cf. IPC B66B 25/00, escalators; IPC G01H 1/00, measuring vibrations of
solid bodies) using inertial measurement telemetry from passenger-carried mobile
devices, step-band periodicity analysis, and phase-aligned fleet anomaly detection.

---

## Background

Escalators fail in ways that hurt people. An estimated ten thousand
escalator-related injuries per year present to United States emergency departments
(McGeehan et al., American Journal of Emergency Medicine, 2008), and the failure
modes that matter most are mechanical and progressive: step-chain elongation that
lets steps collapse into the comb at the landing, worn step rollers that strike the
track, and misaligned comb plates that catch footwear. These faults announce
themselves in vibration long before they become visible, but today's escalators are
inspected on calendars, not on signals. ASME A17.1 requires periodic maintenance by
qualified personnel, and dedicated condition-monitoring hardware exists (for example,
the step-mounted SmartStep system from Transmission Dynamics, which reports
drive-chain imbalance and uneven chain wear from sensors bolted to the machine), yet
the installed base of escalators in North America numbers in the tens of thousands,
and only a small fraction will ever be retrofitted with dedicated instrumentation at
thousands of dollars per unit. The result is that most machines run open-loop
between inspections: a maintenance technician sees each escalator perhaps monthly,
while the machine performs tens of thousands of revolutions between visits.

Meanwhile, the sensing hardware that could close this loop is already riding the
machine. Every modern smartphone carries a 3-axis accelerometer and 3-axis gyroscope
sampling at 100 Hz or more (Bosch BMI260-class and ST LSM6DSO-class parts, 16-bit
resolution, noise floors in the hundreds of micro-g per root Hz), plus a barometer
with roughly 0.01 hPa (about 0.08 m) altitude resolution. The research literature has
already shown that a phone IMU can separate conveyor motion from arbitrary
pedestrian behavior: ELESON (IEEE Transactions on Mobile Computing, 2025) classifies
elevator, escalator, or neither from smartphone inertial and magnetic readings under
36,420 instances of walking, turning, phone-in-pocket, and phone-in-bag behaviors,
exceeding 0.9 F1 score, and Chen et al. (IEEE Sensors Journal, 2024) estimate
pedestrian height from escalator rides using phone IMU alone with relative errors
under 8%. What neither the literature nor industry products do is run the analysis
the other direction: from conveyor-ride detection to machine-health assessment.
Conveyor classification treats the machine as a disturbance to be subtracted. This
disclosure treats the machine as the signal.

The physics are unusually cooperative. A walking pedestrian's gait dominates their
IMU at 1.5 to 2.5 Hz, which buries machine vibration. A standing escalator rider has
no gait. With the passenger stationary relative to the step band, the phone becomes a
free-running accelerometer bolted to the machine itself, and the machine's own
periodicity stands out: steps cross the comb plate at step-pass frequency (0.5 m/s
rated speed divided by 0.405 m standard step pitch gives about 1.23 Hz), each
crossing producing a small impact; the step band completes one revolution every
loop-length divided by speed (60 to 120 seconds on transit machines); and the drive
train contributes its own harmonic series from motor, gearbox, and sprocket. The gaps
in this observation are systematic: a single ride of 15 to 40 seconds covers only a
fraction of one band revolution, and riders board at random phase. That randomness is
the key to filling the gap. With dozens of rides per day at a busy station, the
ensemble of ride fragments, each tagged by its step-band phase at boarding, tiles
the full revolution. This disclosure describes the ride detector, the phase
alignment, the baseline model, and the fault taxonomy that turn passenger phones
into a zero-marginal-hardware condition-monitoring network for every escalator in a
city.

---

## Detailed Description

### 1. Ride Segment Detection

The on-device pipeline begins with a low-power activity gate. The device's
significant-motion detector (hardware step detector on Android; CoreMotion
device-motion activity classification on iOS) wakes the IMU sampler only when
sustained motion is present, so the system never samples continuously. Once awake,
the ride classifier looks for the escalator signature: (a) a constant-velocity
inclined trajectory, i.e., barometric altitude changing at a near-constant rate
between 0.15 and 0.40 m/s vertically (0.5 to 0.75 m/s along a 30-degree incline) for
at least 10 seconds; (b) gait absent, defined as band power in the 1.5 to 2.5 Hz
vertical channel below a device-calibrated threshold, which rejects walking,
stair-climbing, and elevator-with-baggage-drag cases; and (c) lateral sway below
walking levels, since a standing rider sways at under 0.3 Hz with small amplitude
while a walker exhibits stride-synchronized lateral oscillation. These heuristics
require no machine learning model and no training data, though an implementation may
add a lightweight logistic classifier trained on labeled ride segments as a second
stage; ELESON's published result (F1 above 0.9 for conveyor classification under
adversarial pedestrian behavior) demonstrates that the classification problem is well
within reach of on-device models.

Ride verification uses barometer-IMU fusion. The barometric altitude slope gives the
vertical component of ride velocity; dividing by sin(30 degrees) recovers the
along-incline speed, which the system compares against the machine's rated speed
(0.5 m/s typical, 0.75 m/s high-speed). A ride whose inferred speed matches rated
speed within 10% is labeled a confirmed ride; a ride whose speed deviates
persistently is itself an anomaly input (speed drift is a baseline-tracked parameter
in its own right, and sudden speed deviations trigger the emergency-event detector
of section 7).

### 2. Incline-Plane Signal Conditioning

Phones sit in pockets at arbitrary orientations, so the raw accelerometer axes carry
no anatomical meaning. The pipeline estimates gravity direction with a low-pass
filter (cutoff 0.5 Hz) over the ride, constructs a world-frame quaternion, and
rotates the 3-axis acceleration into a machine frame: one axis along the incline
(uphill/downhill), one axis across the incline (lateral), and one axis normal to the
step surface. Gravity is subtracted from the along-incline and lateral axes; the
step-normal axis retains gravity plus the normal vibration component. All subsequent
spectral analysis operates on this machine frame. The transform is recomputed
continuously to absorb phone shifting in the pocket, with quaternion slew-rate
limits that reject instantaneous reorientations (phone picked up mid-ride) by ending
the current segment.

### 3. Step-Band Periodicity Signature

The machine frame streams are band-passed into three analysis bands. The step-pass
band (0.5 to 3 Hz) isolates the comb-plate impact line at step-pass frequency f_sp =
v_rated / p_step, approximately 1.23 Hz for a 0.5 m/s machine with 0.405 m steps,
and its first several harmonics. The drive-train band (3 to 50 Hz at 100 Hz sampling;
3 to 100 Hz when the device permits higher output data rates) captures gearbox and
sprocket harmonics. The revolution band (below 0.05 Hz, computed from multi-ride
aggregates) captures once-per-revolution modulation.

For each confirmed ride, the device computes short-time Fourier spectra over
4-second windows with 50% overlap, then integrates spectral power into a feature
vector: power at f_sp and its harmonics 2 through 6, total band power in the
drive-train band in octave sub-bands, peak-to-average ratio in the time domain (a
spike detector for impacts), and cross-axis coherence between the along-incline and
lateral channels. The vector is typically 48 to 96 single-precision floats, under
400 bytes, and it is the only telemetry that leaves the device. No waveform data, no
audio, no camera frames, and no position finer than the venue geofence identifier (a
transit station, a mall) accompany it. A daily-rotating pseudonymous device token
prevents longitudinal tracking of any individual across days while still permitting
within-day phase alignment.

### 4. Phase-Aligned Revolution Reconstruction

The signature that a single phone cannot see is the once-per-revolution profile:
which part of the step band is rough. A ride of 30 seconds on a machine whose band
revolves every 90 seconds covers one third of the band, and a single rider can never
be asked to ride for a full revolution. The aggregation service solves this with
phase alignment. Each ride carries a boarding timestamp and a measured along-incline
speed. Knowing the rated speed and a machine-specific loop-length estimate (learned
from the ride-duration distribution, since ride duration equals visible length
divided by speed, and visible length is half the loop), the service assigns every
4-second window of every ride a band-phase coordinate: the position, modulo one
revolution, of the step beneath the rider at that instant.

Windows from many rides, many phones, and many days accumulate into phase bins
(typically 64 to 128 bins per revolution). A busy transit escalator with 40 confirmed
rides per day fills every bin within days. The median of each bin's impact-line
power (peak-to-average ratio at f_sp) yields the reconstructed revolution signature:
a 64-point trace of how rough each sector of the step band is, averaged over the
crowd. The crowd is doing the sensing job that a single step-mounted accelerometer
would do, with the random boarding phase of strangers substituting for a slip-ring.

### 5. Per-Machine Baseline and Change-Point Detection

The service maintains, per machine (identified by venue geofence plus inferred
direction and floor-pair from ride altitude range), a slowly adapting baseline:
per-bin median and interquartile range of the revolution signature, plus the global
parameters (f_sp, drive-train band powers, lateral sway level, inferred ride speed).
Baselines adapt with a weekly exponential decay so that gradual seasonal effects
(temperature-dependent lubricant viscosity changes the broadband floor) do not
trigger alerts, while a Bayesian online change-point detector with a 4-week lookback
flags abrupt shifts. A shift is alerted when the posterior probability of a change
point within the last 7 days exceeds 0.95 on any tracked parameter.

### 6. Fault Taxonomy

Alert classification maps signature changes to the mechanical faults they precede,
in order of evidentiary strength:

1. **Step roller and track wear (strongest).** A damaged roller or a worn track
   section strikes once per revolution at a fixed band phase, producing impulses at
   f_sp whose amplitude is phase-locked to one sector of the revolution signature.
   As wear progresses, the impulse amplitude grows and its harmonic content spreads.
   A sector-localized, growing f_sp impulse is the earliest warning in the taxonomy,
   and it directly precedes step collapse into the comb, the most dangerous
   progressive failure mode.
2. **Comb-plate misalignment.** Misaligned combs produce broadband transients at
   every step crossing, raising f_sp power across all phase bins uniformly rather
   than in one sector. Uniform f_sp growth without sector localization distinguishes
   comb faults from roller faults.
3. **Step-chain elongation.** As a roller chain elongates from pin and bushing wear,
   the pitch between links grows; the chain's tooth-pass signature at the drive
   sprocket shifts in frequency relative to the step-pass line, and the harmonic
   series broadens as engagement becomes uneven. The system tracks the ratio of the
   drive-train harmonic centroid to f_sp and the bandwidth of the sprocket-pass
   line; sustained drift in either over weeks indicates elongation approaching the
   manufacturer's stretch limit.
4. **Lateral guide and skirt wear.** Worn guides allow the step band to wander
   sideways, raising power in the lateral (cross-incline) channel and reducing
   cross-axis coherence. Growth of the lateral-to-along-incline power ratio beyond
   the per-machine baseline is the skirt-entrapment precursor signal.
5. **Drive-train wear.** Gearbox tooth wear and bearing degradation raise narrowband
   power in the drive-train band at frequencies that are not harmonics of f_sp.
   These lines are weaker than the step-band signals and are reported as a
   low-confidence watch condition rather than an alert, because phone IMU bandwidth
   (Nyquist 50 to 100 Hz depending on device) excludes the high-frequency gear-mesh
   lines that dedicated sensors catch.
6. **Speed and leveling drift.** Inferred ride speed drifting more than 5% below
   rated speed indicates drive slip or VFD fault; the parameter is already computed
   in section 1 and is tracked as a first-class baseline.

### 7. Emergency-Event Detection

A separate always-on path watches for sudden-stop events: along-incline
deceleration exceeding 1 m/s^2 sustained for more than 0.3 seconds during a
confirmed ride, the signature of an emergency stop or a step-band jam. A single such
event from any rider triggers an immediate safety-review notification to the
building operator, independent of the baseline machinery. False positives (a rider
abruptly stopping their own phone, a dropped phone) are filtered by requiring the
deceleration signature on at least two independent devices within the same 10-second
window, or a corroborating barometric deceleration profile.

### 8. Energy and Privacy Design

The on-device pipeline draws power only during confirmed rides: the
significant-motion gate keeps the IMU off otherwise, the spectrum computation runs
in 4-second batches on the application processor, and transmission is a single
sub-kilobyte upload per ride batched with the venue's next network wakeup. A
commuter riding four escalators per day (about 2 minutes of total ride time)
consumes well under one tenth of one percent of a typical day's battery on a 6-axis
100 Hz capture. Privacy rests on four properties: no microphone, camera, or fine
location is ever captured; only spectral feature vectors leave the device; venue
attribution uses geofence identifiers rather than coordinates; and device tokens
rotate daily so no rider is traceable across days. Fleet aggregates may additionally
be noised with differential privacy before publication to transit authorities.

---

## Claims

1. A method for assessing the mechanical health of an inclined passenger conveyor
   (escalator or moving walkway) comprising: detecting, on a mobile device carried
   by a passenger, a ride segment characterized by constant-velocity inclined motion
   with gait absent; verifying the ride segment by comparing barometric altitude
   slope against the conveyor's rated speed; transforming inertial measurements
   into a machine frame aligned with the conveyor's incline plane; extracting a
   step-band periodicity signature comprising spectral power at a step-pass
   frequency derived from rated speed and step pitch; and transmitting only a
   compact spectral feature vector to an aggregation service.
2. The method of claim 1, wherein detecting the ride segment comprises computing
   band power in a gait frequency band of 1.5 to 2.5 Hz and rejecting segments above
   a device-calibrated threshold, and computing lateral sway amplitude and rejecting
   segments above a walking threshold.
3. The method of claim 1, wherein verifying the ride segment comprises dividing the
   barometric altitude change rate by the sine of the incline angle to recover
   along-incline speed and confirming agreement with rated speed within a tolerance
   band.
4. The method of claim 1, further comprising estimating a band-phase coordinate for
   each time window of the ride from boarding time, measured along-incline speed,
   and an estimated step-band loop length, and accumulating spectral features from
   multiple rides of multiple passengers into phase bins to reconstruct a
   per-revolution vibration signature of the step band.
5. The method of claim 4, wherein the loop length is estimated from the
   distribution of confirmed ride durations at a venue, the visible length equaling
   mean ride duration multiplied by rated speed and the loop equaling twice the
   visible length.
6. The method of claim 4, further comprising localizing a step roller or track
   defect to a step-band sector by identifying phase bins in which
   step-pass-frequency impulse amplitude exceeds the per-machine baseline.
7. The method of claim 1, further comprising maintaining a per-conveyor statistical
   baseline of the periodicity signature with weekly exponential decay and applying
   an online change-point detector to flag abrupt deviations.
8. The method of claim 7, further comprising classifying deviations into a fault
   taxonomy comprising sector-localized step-pass impulse growth (step roller or
   track wear), uniform step-pass power growth (comb-plate misalignment), drift of
   the drive-train harmonic centroid relative to the step-pass line (step-chain
   elongation), growth of the lateral-to-along-incline power ratio (lateral guide or
   skirt wear), and inferred ride-speed drift below rated speed (drive slip).
9. The method of claim 1, further comprising detecting a sudden-stop event by
   identifying along-incline deceleration exceeding a threshold during a confirmed
   ride and triggering a safety-review notification when corroborated by at least
   one independent device or a barometric deceleration profile.
10. The method of claim 1, wherein the aggregation service identifies individual
    conveyors by venue geofence identifier combined with inferred travel direction
    and floor-pair from ride altitude range, without recording passenger
    coordinates.
11. The method of claim 1, further comprising comparing per-conveyor baselines
    across a fleet of conveyors of the same model to identify population-level
    deviations indicative of a systemic defect or maintenance-contract performance
    gap, with fleet aggregates protected by differential privacy.

---

## Implementation Notes

This is a design disclosure: no prototype has been built and no field validation has
been performed, so every quantitative parameter above (sampling rates, band edges,
thresholds, bin counts, alert posteriors) is an untested design choice, not a
measured result. The disclosure is intended to enable a person of ordinary skill in
the art to build the system, and to place the core idea (phase-aligned,
crowdsourced reconstruction of a conveyor's per-revolution vibration signature from
passenger device IMUs) in the public record as prior art.

Several deliberate choices distinguish this disclosure from the adjacent
literature. The LITF elevator-barometric disclosure (LITF-PA-2026-171) assesses
vertical elevator cars from barometric profiles; this disclosure assesses inclined
conveyors from IMU vibration, a different machine class, a different sensor, and
different fault physics. ELESON classifies conveyor state to subtract the machine
from pedestrian tracking; this disclosure inverts the relationship and makes the
machine the measurement target. Dedicated step-mounted sensors (SmartStep and
equivalents) measure the machine directly but require installation, downtime, and
per-unit cost; this system measures the machine through its passengers and costs
nothing beyond software. None of these sources combines ride detection,
incline-frame transformation, step-pass harmonic extraction, multi-rider phase
alignment, and change-point-based fault classification into a single system.

The strongest case against this system is data quality. Passenger phones are
uncalibrated, their placement varies from hand to pocket to bag, sampling rates
differ by device, and coverage is dense at transit stations but thin at suburban
malls. A dedicated accelerometer bolted to a step is simply a better instrument. The
answer is not that phones replace calibrated sensors but that the vast majority of
the installed base will never get calibrated sensors: the choice is between noisy
crowdsourced data and no data at all, and the phase-alignment machinery exists
precisely to convert quantity of observations into quality of inference. The second
objection is regulatory: ASME A17.1 already requires qualified maintenance, so a
monitoring layer adds little. But code-mandated maintenance is interval-based, and
the faults that injure people are progressive; a signal that turns a monthly visit
into a visit with a specific step-band sector to inspect is a complement to the
code, not a substitute. The third objection is privacy, addressed in section 8 by
never capturing audio, video, or fine location and by rotating device tokens daily;
the system learns about machines, not about riders.

A builder implementing this disclosure should start with two empirical questions
before anything else: (1) what is the actual signal-to-noise ratio of the step-pass
line in phone accelerometer data across a range of devices and carriage positions,
measured by standing on a known machine with a reference accelerometer; and (2) how
many confirmed rides per day a typical transit escalator produces under an opt-in
deployment, which determines how fast phase bins fill and therefore how early the
system can speak. Everything downstream of those two measurements is engineering.
Everything upstream of them is the idea, and the idea is now public.

---

## Prior Art References

1. [ASME A17.1/CSA B44, Safety Code for Elevators and Escalators](https://www.asme.org/codes-standards/find-codes-standards/a17-1-safety-code-elevators-escalators): maintenance requirements for escalators and moving walkways (Section 8.6); interval-based inspection regime this disclosure complements
2. [ELESON (IEEE Transactions on Mobile Computing, 2025)](https://pdfs.assets.alphaxiv.org/2405.03218v3.pdf): classifies elevator, escalator, or neither from smartphone INS readings under arbitrary pedestrian behavior; F1 above 0.9 across 36,420 instances; establishes that ride detection from phone IMU is feasible, treats the conveyor as a signal to classify rather than a machine to assess
3. [Chen, Kuang, Xia, Niu, IEEE Sensors Journal, 2024](https://doi.org/10.1109/JSEN.2024.3444918): height estimation in escalator scenarios from smartphone IMU; relative height errors of 3.50% (up) and 7.97% (down); demonstrates escalator ride dynamics are recoverable from phone inertial data without barometers
4. [Transmission Dynamics SmartStep escalator condition monitoring](https://www.transmissiondynamics.com/products/smartstep-escalator-condition-monitoring/): dedicated step-mounted sensor system reporting drive-chain imbalance and uneven chain wear; commercial existence of machine-side monitoring; requires per-unit installation this disclosure avoids
5. McGeehan, et al., "Escalator-Related Injuries Among Children in the United States, 1990-2002," American Journal of Emergency Medicine, 2008: NEISS-based estimate on the order of ten thousand annual escalator-related emergency department visits; establishes the injury burden behind the fault taxonomy
6. [Android Sensors overview](https://developer.android.com/guide/topics/sensors/sensors_overview): hardware step detection, significant-motion triggers, and sensor batching; the low-power gating primitives used in section 1
7. [Apple CoreMotion](https://developer.apple.com/documentation/coremotion): device-motion activity classification and accelerometer access on iOS; the low-power gating primitives used in section 1
8. EN 115-1, Safety of escalators and moving walks, construction and installation: step depth, rated speeds, and incline geometry parameters underlying the step-pass frequency calculation
9. [LITF-PA-2026-171](https://liveinthefuture.org/priorart/elevator-barometric-health-monitoring.html): crowdsourced elevator health from smartphone barometric profiles; vertical-car machine class, barometer sensor, distinct from the inclined-conveyor IMU assessment disclosed here
10. [EP 3915926 A1 (Otis Elevator Company)](https://data.epo.org/publication-server/rest/v1.2/publication-dates/20211201/patents/EP3915926NWA1/document.pdf): escalator with distributed state sensors secured to individual steps and a master sensor reporting fault conditions to a call center; representative of machine-side instrumentation patents this disclosure places crowd-sensed alternatives against
