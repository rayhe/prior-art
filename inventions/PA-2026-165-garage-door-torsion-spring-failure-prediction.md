# PA-2026-165: Predictive Torsion Spring Failure Detection in Overhead Garage Doors via Motor Current Signature Analysis

**Title:** System and Method for Predicting Torsion Spring Failure in Overhead Garage Doors Using Opener Motor Current Signature Analysis and Stroke Dynamics

**Filing:** LITF-PA-2026-165
**Published:** September 8, 2026
**Domain:** Home Maintenance / Predictive Maintenance
**Full Disclosure:** [liveinthefuture.org/priorart/garage-door-torsion-spring-failure-prediction.html](https://liveinthefuture.org/priorart/garage-door-torsion-spring-failure-prediction.html)

---

## Abstract

      Disclosed is a non-intrusive system and method for predicting torsion spring failure in overhead residential garage doors before it occurs. A current sensor on the garage door opener's AC power supply (a current-sensing plug, a clamp-on current transformer, or an integrated metering circuit) samples opener motor current at 1 kHz or faster and segments each open/close cycle into strokes using an RMS-envelope state machine. An optional door-mounted accelerometer reports door angle and stroke direction over a wireless link. As a torsion spring fatigues toward its rated cycle life, it contributes progressively less counterbalance torque, and the opener motor must supply the deficit: upstroke RMS current rises monotonically, up/down stroke current asymmetry increases, stroke duration drifts, and the inrush transient grows. The system extracts per-stroke features, temperature-compensates them, fuses them into a Spring Health Index, and fits a Weibull fatigue model to project remaining useful cycles, issuing escalating alerts at calibrated thresholds. A sudden full or partial counterbalance imbalance, the signature of a broken spring, triggers an immediate lockout of automatic operation and a homeowner alert.

## Field of the Invention

      This invention relates to residential predictive maintenance, specifically to non-intrusive condition monitoring of overhead garage door counterbalance systems through electrical signature analysis of the opener motor and kinematic measurements of door travel, enabling failure prediction before spring fracture.

## Background

      Torsion springs are the most failure-prone mechanical component in a residential garage door system. A standard torsion spring is rated for approximately 10,000 open-close cycles (about 7-10 years of typical use) ([Overhead Door of Puget Sound](https://ohdpugetsound.com/services/garage-door-torsion-spring-repair/)), with high-cycle variants rated 20,000 to 50,000 cycles ([Engineer Fix](https://engineerfix.com/what-is-a-torsion-spring-on-a-garage-door/)). At four cycles per day a standard spring lasts roughly 7 years; at eight or more cycles per day, a busy household burns through the same rating in 3-4 years ([Edge Garage Doors](https://edgegaragedoorstx.com/how-long-do-garage-door-springs-last-in-austin-tx-complete-lifespan-guide/)). Remote work, package deliveries, and multi-driver households have increased daily cycle counts, and repair technicians report [spring failures rising across North America](https://lifestyle.kynt1450.com/story/59902/garage-door-spring-failures-are-rising-across-north-america/).

      The failure mode is dangerous and expensive to experience unplanned. A broken torsion spring is announced by a sudden loud bang, after which the door becomes dead weight and may hang crooked with dangling cables ([Engineer Fix](https://engineerfix.com/what-is-a-torsion-spring-on-a-garage-door/)). Replacement is dangerous DIY territory and requires a trained technician; professional replacement runs [$150-$450+ per pair](https://ohdpugetsound.com/services/garage-door-torsion-spring-repair/). Cold weather can temporarily cut spring output by 10-20%, so late-life springs often fail during the first cold snap of winter.

      The physics is simple and measurable. A healthy torsion spring system counterbalances roughly 90-95% of the door's weight; the opener motor supplies only the residual plus friction. The required counterbalance shaft torque is approximately the door weight multiplied by the cable-drum radius ([Garage Door Champ](https://garagedoorchamp.com/garage-door-torsion-springs-everything-need-know/)). When a spring fatigues, its effective spring rate falls, the motor draws more current on the upstroke to lift the unassisted portion of the door, and the downstroke dynamics change because the motor's braking/regeneration balance shifts. Typical residential openers are fractional-horsepower AC induction motors (a common 1/2 HP unit draws roughly 4-5 A RMS running current at 120 VAC, with an inrush transient 3-5 times running current).

      Motor current signature analysis (MCSA) is a mature condition-monitoring discipline for industrial induction motors. EPRI conducted independent MCSA and electrical signature analysis studies through 2020-2021, the technique is codified in ISO 20958 (2013), and the foundational method was disclosed in US Patent 4,965,513 (ORNL) ([NACleanEnergy/AMC summary](https://www.nacleanenergy.com/wind/development-of-the-first-certification-program-for-electrical-and-current-signature-analysis-1)). The academic literature applies MCSA to rotor bars, eccentricity, and bearings ([IntechOpen MCSA survey](https://www.Intechopen.com/chapters/1171032)). In garage doors specifically, prior patents use current only for immediate fault handling: [CN111665008B](https://patents.google.com/patent/CN111665008B/en) compares barrier-gate motor operation data to standard data to warn of spring fault after it occurs, in parking barrier gates with a different mechanism; US 2015/0059989 A1 discloses a sensor that detects the energy suddenly released when a counterbalance spring breaks, that is, it detects the break, not its approach; [US 10,643,408 B2](https://patents.google.com/patent/US10643408B2/en) and US 10,968,676 B2 use motor current and shaft position to detect jams and obstructions for safety reversal, not spring fatigue; and [US 2018/0247475 A1](https://patents.google.com/patent/US20180247475A1/en) uses current-draw detection to determine whether a door moved, for remote-drive-off scenarios.

      The gap in the art is a residential, non-intrusive system that: (a) predicts torsion spring fatigue before fracture rather than detecting the break afterward, (b) uses up/down stroke asymmetry and temperature-compensated per-stroke features rather than single-event thresholds, (c) models fatigue with a cycle-life projection calibrated to the spring's rated life, and (d) requires no modification of the opener, door, or wiring.

      **Non-obviousness.** No reference teaches applying current-signature analysis to garage door counterbalance fatigue, and several teach away from the claimed approach. The LiftMaster-family patents teach monitoring motor current for acute jam detection and reversal, with fault thresholds set to react to sudden blockage; nothing there suggests tracking slow, monotonic drift in the up/down asymmetry ratio over thousands of cycles as a fatigue predictor. US 2015/0059989 A1 teaches placing sensors on or near the spring to catch the break event itself, directly away from non-intrusive electrical monitoring at the outlet. CN111665008B operates on parking barrier gates, a rotating-arm mechanism with no counterbalance-door kinematics, and warns after spring fault rather than predicting it. The insight that the ratio of upstroke to downstroke motor current is the robust fatigue signature, because it self-normalizes for line-voltage drift, door lubrication, and opener gear wear, is not disclosed or suggested anywhere in the cited art.

## Detailed Description

### 1. Non-Intrusive Current Sensing Hardware

      The preferred embodiment is a current-sensing pass-through plug installed at the opener's ceiling outlet, requiring no wiring changes. It contains: a Hall-effect current sensor or split-core current transformer with 0.1-15 A range and better than 1% accuracy; a 32-bit microcontroller (e.g., ESP32 class) with ADC sampling the current waveform at 2 kHz; an energy metering front end reporting true RMS current, active power, and power factor; an ambient temperature and humidity sensor for compensation; and a WiFi radio that transmits per-stroke feature vectors (not raw waveforms) to a home hub or cloud service. Power consumption is under 1 W. Target bill-of-materials cost: $25-40.

      An alternative embodiment uses a clamp-on current transformer on the opener's branch circuit conductor inside the service panel, powered by a USB supply, for installations where the opener is hardwired. A further embodiment integrates the metering circuit into the opener control board at manufacture. An optional door-mounted accessory is a small accelerometer and BLE radio adhered to the door's top panel, reporting panel angle and vibration signature. Battery life exceeds 24 months at typical cycle rates.

### 2. Stroke Segmentation and Cycle Extraction

      The microcontroller runs an RMS-envelope state machine on the sampled current waveform. Idle is declared below 0.3 A RMS. A stroke begins when RMS current exceeds 1.5 A for more than 200 ms, rejecting relay clicks and lamp loads. The inrush transient is the first 400 ms of elevated current, during which the motor breaks the static friction of the door at rest. Constant-velocity travel is the plateau between inrush and the limit-switch stop, detected as RMS current falling below 0.5 A for more than 1 second. A stroke is thus bounded by its start and stop timestamps; one full open followed by one full close constitutes a cycle and increments the cycle counter.

      Stroke direction is resolved by one of three mechanisms: (a) the door-mounted accelerometer accessory, reporting panel angle increasing for opening and decreasing for closing; (b) a learned heuristic, since on a fatiguing system the upward stroke is longer in duration and higher in current than the downward stroke; or (c) an acoustic envelope feature, since the cable drum and limit switches produce direction-distinctive click patterns. The preferred embodiment uses (a); (b) is the fallback when no accessory is installed. Partial strokes (interrupted by safety reversal or user intervention) are flagged and excluded from fatigue trending but counted for the cycle counter.

### 3. Per-Stroke Feature Extraction

      For each completed stroke the system computes: upstroke RMS current (I_up) and downstroke RMS current (I_down); the asymmetry ratio A = I_up / I_down; upstroke and downstroke durations (T_up, T_down); inrush charge Q_inrush, the integral of current over the first 400 ms; plateau current ripple, the RMS of the envelope's deviation from its median during constant-velocity travel; and limit-switch transition sharpness. The preferred feature set fits in under 64 bytes per stroke and is the only data transmitted off-device.

      The critical insight is that A is self-normalizing: line-voltage sag, opener gear wear, roller friction, and temperature affect both strokes, while spring fatigue acts almost entirely on the upstroke. In baseline data a healthy balanced door shows A near 1.0-1.2. As the spring loses counterbalance torque, A rises monotonically. A fatigued single-spring system typically shows A above 1.6 before fracture; a dual-spring system in which one spring has broken shows a step increase in A of roughly 40-60% relative to the two-spring baseline.

### 4. Temperature Compensation

      Spring output and motor characteristics both shift with garage temperature. Steel spring stiffness falls as temperature drops, and industry practice notes roughly 10-20% reduced spring output in cold weather ([Overhead Door of Puget Sound](https://ohdpugetsound.com/services/garage-door-torsion-spring-repair/)). The system records ambient temperature per stroke from the sensor's temperature element and applies a correction model: features are normalized to a 20 C reference using a piecewise-linear compensation coefficient learned during the first 60 days of operation. Strokes recorded within 2 hours of a cold start (opener body temperature more than 10 C below the garage ambient) are weighted down in the trend estimator. Without this compensation, the first winter cold snap would produce false health alarms on healthy springs.

### 5. Spring Health Index and Remaining-Life Projection

      The Spring Health Index H is a dimensionless quantity initialized to 1.0 during a 30-day calibration period after installation, during which the system learns baseline values A0, I_up0, T_up0, and Q0 at each temperature band. The index is computed as:

          H = 1 - [w1*(A - A0)/A0 + w2*(I_up - I_up0)/I_up0 + w3*(T_up - T_up0)/T_up0 + w4*(Q_inrush - Q0)/Q0]

      with default weights w1 = 0.45, w2 = 0.25, w3 = 0.15, w4 = 0.15, and each feature passed through an exponentially weighted moving average with a time constant of 200 cycles to suppress cycle-to-cycle noise. H falls monotonically as the spring fatigues.

      Remaining useful cycles are estimated by fitting a two-parameter Weibull distribution to the H trend, with the shape parameter beta constrained to 2.5-4.0, the accepted range for steel fatigue failure modes. The user supplies (or the installer scans from the spring's color code) the spring's rated cycle life, defaulting to 10,000 cycles for an unmarked standard spring. The system reports: a healthy/balanced confirmation while H remains above 0.85; an inspection notice when H crosses 0.85, roughly corresponding to 70-75% of rated life consumed; a replacement-scheduling alert when H crosses 0.70 or the Weibull projection shows fewer than 500 cycles remaining; and a critical alert when H crosses 0.55 or fewer than 150 cycles remain. Fleet learning across deployed units refines the Weibull priors and the temperature compensation coefficients by spring type, door size, and opener model, without transmitting any identifiable household data.

### 6. Broken-Spring Detection and Safety Interlock

      A fractured torsion spring produces a distinctive one-cycle signature: A steps upward by 40-100% between consecutive cycles with no temperature change, the inrush charge Q_inrush jumps as the motor breaks away a fully unbalanced door, and in dual-spring systems the plateau ripple gains a once-per-revolution component from the now-unloaded spring shaft oscillating. When this signature is detected, the system: (a) sends an immediate alert identifying which failure is likely (single spring fracture vs. dual-spring progressive fatigue), (b) instructs the hub to issue a lockout command to any smart opener controller or relay, refusing remote open commands while permitting only supervised local operation, and (c) logs the event with the cycle count at failure, feeding the fleet survival model. The lockout is advisory and removable by the homeowner, since emergency egress requirements in some jurisdictions prohibit hard-locking a garage door; the disclosure's safety interlock is a software-level refusal that the homeowner can override at the wall control.

### 7. Implementation Notes

      Calibration quality determines prediction accuracy. The 30-day calibration period should capture at least 60 full cycles; doors cycled less than twice daily should extend calibration to 60 days. The learned baseline absorbs installer-specific factors such as door lubrication state, track alignment, and opener model, so a baseline must be re-learned after any professional service event, detected automatically by a step change in the ripple feature followed by a stable new plateau.

      Extension-spring doors are explicitly supported: the same current-signature approach applies, since fatigued extension springs also leave the motor lifting unassisted weight, though the asymmetry-ratio signature is weaker and the default alert thresholds should be shifted to 0.88/0.75/0.60. The system distinguishes torsion from extension installations during calibration by the characteristic plateau ripple spectrum of the respective counterbalance geometry. No claim is made on the specific spring replacement procedure; this disclosure covers detection and prediction only.

### 8. Figures Description

- **Figure 1:** RMS current envelope traces for a single upstroke at three health states: newly balanced (A near 1.1), late fatigue (A near 1.6, elevated inrush), and post-fracture (A above 2.0, extended stroke duration).
- **Figure 2:** System block diagram showing the current-sensing plug on the opener outlet, the optional door accelerometer, the feature extraction pipeline, the temperature compensation module, the Weibull projection engine, and the alert hub.
- **Figure 3:** Example Spring Health Index trend over 4,000 cycles showing EWMA smoothing, temperature-compensated bands, the 0.85/0.70/0.55 alert thresholds, and the Weibull projection to end of life.
- **Figure 4:** State machine diagram for stroke segmentation from the RMS current envelope, including idle, inrush, constant-velocity travel, limit-switch stop, and partial-stroke rejection paths.

## Claims

1. A system for predicting torsion spring failure in an overhead garage door, comprising: a non-intrusive current sensor electrically coupled to the power supply of the door's opener without modification of the opener, door, or wiring; wherein the sensor samples opener motor current waveform at 1 kHz or faster, segments each door travel into strokes using an RMS-envelope state machine, extracts per-stroke current and timing features, and computes a spring health index from the temperature-compensated up/down stroke asymmetry of motor current.

2. The system of claim 1, further comprising a door-mounted wireless accelerometer that reports door panel angle to resolve stroke direction independently of the current envelope.

3. The system of claim 1, wherein the per-stroke features include upstroke RMS current, downstroke RMS current, the asymmetry ratio A = I_up / I_down, upstroke and downstroke durations, and inrush charge integrated over the first 400 ms of the stroke, with plateau current ripple and limit-switch transition sharpness as secondary features.

4. The system of claim 1, further comprising a temperature compensation module that normalizes per-stroke features to a reference temperature using a compensation coefficient learned during a calibration period, and that de-weights strokes recorded during cold-start conditions.

5. The system of claim 1, further comprising a remaining-life projection engine that fits a Weibull fatigue distribution with shape parameter between 2.5 and 4.0 to the spring health index trend, calibrated against a rated cycle life supplied for the installed spring or defaulted to 10,000 cycles, and that issues escalating alerts at index thresholds of 0.85, 0.70, and 0.55.

6. The system of claim 1, further comprising a broken-spring detector that identifies a 40% or greater step increase in the asymmetry ratio between consecutive cycles with no corresponding temperature change, and in response issues an immediate alert and a software-level lockout of remote automatic operation.

7. The system of claim 1, wherein only per-stroke feature vectors are transmitted off the sensing device, with raw current waveforms processed locally and discarded.

8. A method for predicting torsion spring failure in overhead garage doors comprising: non-intrusively measuring opener motor current at 1 kHz or faster from the opener's power supply; segmenting door travel into upstrokes and downstrokes using an RMS-envelope state machine; computing the temperature-compensated ratio of upstroke to downstroke motor current; fusing the ratio with upstroke current drift, stroke duration drift, and inrush charge growth into a monotonic spring health index; and projecting remaining useful cycles via a Weibull fatigue model calibrated to the spring's rated cycle life.

9. The method of claim 8, further comprising differentially detecting a single-spring fracture in a dual-spring installation from the magnitude of the asymmetry-ratio step change, and issuing a failure identification alert distinguishing single-spring fracture from progressive dual-spring fatigue.

## Prior Art References

1. [Overhead Door of Puget Sound](https://ohdpugetsound.com/services/garage-door-torsion-spring-repair/): Torsion spring rated 10,000 cycles, 7-10 years; $150-$450+ per pair; cold weather reduces output 10-20%
2. [Engineer Fix](https://engineerfix.com/what-is-a-torsion-spring-on-a-garage-door/): Torsion springs rated 10,000-20,000 cycles; failure symptoms including loud bang and heavy door
3. [Edge Garage Doors](https://edgegaragedoorstx.com/how-long-do-garage-door-springs-last-in-austin-tx-complete-lifespan-guide/): Cycle-life vs. daily-use lifespan tables; 8+ cycles/day exhausts standard springs in 3-4 years
4. [Garage Door Champ](https://garagedoorchamp.com/garage-door-torsion-springs-everything-need-know/): Required shaft torque approximately door weight times cable-drum radius; cycle-rating lifespan tables
5. [KYNT-AM / Townsquare Media](https://lifestyle.kynt1450.com/story/59902/garage-door-spring-failures-are-rising-across-north-america/): Rising spring failure rates across North America linked to increased daily usage
6. [IntechOpen](https://www.Intechopen.com/chapters/1171032): Motor current signature analysis for induction motor fault detection
7. [NACleanEnergy](https://www.nacleanenergy.com/wind/development-of-the-first-certification-program-for-electrical-and-current-signature-analysis-1): EPRI MCSA studies, IEEE 1415-2006, ISO 20958, US Patent 4,965,513 (ORNL)
8. [CN111665008B](https://patents.google.com/patent/CN111665008B/en): Barrier gate spring fault early warning via motor operation data comparison (post-failure, parking gate mechanism)
9. [US 2015/0059989 A1](https://patentimages.storage.googleapis.com/52/9d/9d/237de515a473c5/US20150059989A1.pdf): Sensor detecting energy released when counterbalance spring breaks (post-break detection)
10. [US 10,643,408 B2](https://patents.google.com/patent/US10643408B2/en): Automatic garage door control with current detection for door movement and obstruction handling
11. [US 10,968,676 B2](https://patentimages.storage.googleapis.com/62/8a/ef/2485df8b7c82f0/US10968676B2.pdf): Motor current and shaft position profiling for movable barrier fault (jam) detection
12. [US 2018/0247475 A1](https://patents.google.com/patent/US2018/0247475): Garage door control using current-draw and vibration detection to confirm door movement
