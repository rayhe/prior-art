# PA-2026-210: Concrete Curing Strength from Smartphone Vibration Actuation and Resonance Tracking

**Title:** System and Method for Estimating Concrete Curing Strength Using Smartphone Vibration Actuation and Resonant Frequency Tracking

**Filing:** LITF-PA-2026-210
**Published:** October 21, 2026
**Domain:** Construction Technology / Structural Health Monitoring / Sensor Fusion
**Full Disclosure:** [liveinthefuture.org/priorart/concrete-curing-strength-smartphone-resonance.html](https://liveinthefuture.org/priorart/concrete-curing-strength-smartphone-resonance.html)
**License:** [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/) — Public Domain

> Prior Art Notice: This document is a defensive technical disclosure published
> to constitute prior art under [35 U.S.C. Sec. 102(a)(1)](https://www.law.cornell.edu/uscode/text/35/102),
> dedicated to the public domain as of the publication date above.

---

## Abstract

Disclosed is a system and method for estimating the in-place compressive strength of curing concrete using an unmodified consumer smartphone as both excitation source and sensor. An operator places the smartphone in contact with the concrete surface. Its haptic vibration actuator emits a known excitation waveform (a swept-sine chirp, a tone that sweeps through a frequency range); its inertial measurement unit (IMU) accelerometer records the structural response. From the drive and response signals, the system computes their transfer function, essentially a frequency-by-frequency map of how strongly the concrete answers the phone's vibration, and extracts the resonant frequency of the coupled phone-concrete system. That frequency rises as cement hydration stiffens the concrete. An on-device calibrated model fuses the resonance track with a temperature-history maturity index, a running tally of heat-driven curing computed from the pour's temperature history per ASTM C1074, plus mix-design metadata, and outputs estimated compressive strength.

Field decisions the design is meant to enable include formwork removal, post-tensioning, saw cutting, and opening to traffic, all without embedded sensors, sacrificial hardware, or waiting on laboratory cylinder breaks for operational timing. No microphone is required; the sensing channel is inertial only. This disclosure describes an untested design proposal; quantitative parameters are design choices awaiting field calibration (see Implementation Notes).

## Technical Field

This disclosure relates to nondestructive evaluation of concrete, specifically to in-situ estimation of compressive strength development during curing using active vibration excitation and resonant frequency tracking. It sits at the intersection of structural health monitoring and mobile sensing: repurposing the vibration actuator and inertial sensors already present in consumer phones as an active nondestructive evaluation instrument.

## Background

Every concrete placement faces the same question: is it strong enough yet? Formwork removal, post-tensioning, saw cutting, and opening pavement to traffic are all strength-gated decisions. Guess wrong in one direction and the schedule slips; guess wrong in the other and the structure is damaged.

The acceptance basis remains the cylinder break test (ASTM C39): companion cylinders cast from the same batch, cured under standard conditions, crushed at specified ages. Cylinders are cheap and legally defensible. They are also slow, with results arriving hours to days after the decision was needed, and standard-cured cylinders do not experience the same thermal history as the in-place concrete. Field-cured cylinders narrow the gap but add logistics: each cylinder must be cast, protected, transported, and tested at lab expense.

Instrumenting the pour itself, the maturity method (ASTM C1074) improved on this. Embedded temperature sensors record the thermal history; a maturity function, the Nurse-Saul temperature-time factor or the Arrhenius equivalent age, converts that history into a maturity index; a pre-established, mix-specific strength-maturity calibration curve converts maturity into estimated strength. Products built on this principle include Giatec SmartRock, COMMAND Center, and Maturix. Vendors report strong agreement with cylinder breaks for properly calibrated mixes. These limitations are structural. Each sensor is a consumable embedded in the concrete, with vendor-rated battery life around four months, after which it is entombed. Every mix change demands a new laboratory calibration program (commonly 1, 3, 7, 14, and 28-day breaks). Because the method infers strength from temperature rather than measuring stiffness directly, estimates degrade when mixes change mid-project or when they are extrapolated beyond the ages covered by the calibration breaks.

Resonance methods attack the problem from the other side: measure stiffness directly. ASTM C215 standardizes resonant-frequency testing of laboratory prisms. Wavelogix's REBEL Concrete Strength Sensing System implements acoustical resonance in the field with sensors embedded in the fresh concrete before the pour; the method was approved as AASHTO T 412-24 in August 2024. REBEL's sensors vibrate the concrete and track frequency change over time, correlating it to strength gain, with no mix-specific calibration curve. Wavelogix reports field variability around 63 psi for acoustical resonance versus roughly 130 psi for maturity-based estimates (the vendor's own published comparison; not independently verified). Its limitation is the hardware: each sensor is a consumable embedded in the pour.

Smartphones have meanwhile become credible vibration sensors. Kang et al. (2023) demonstrated that phone accelerometers reproduce structural natural periods in agreement with reference piezoelectric accelerometers, with 90%+ accuracy on peak acceleration and displacement. MyShake showed phone networks estimating building frequencies within 5% error. A 2025 review of smartphone-based structural health monitoring surveys bridge-monitoring apps performing automated modal analysis with cloud dashboards, and citizen-sensor networks reproducing mode shapes from spatially distributed phones. All of this work is passive: phones listen to ambient or earthquake excitation of existing structures.

No published system of which the authors are aware uses a consumer phone's own vibration actuator as a calibrated excitation source to probe fresh concrete, nor fuses the resulting resonance track with maturity data for curing-strength estimation. The reason no one has tried is instructive: phone haptic actuators are metrologically poor exciters (weak, narrowband, temperature-drifting, and dependent on the original equipment manufacturer (OEM) in their drive behavior), which is precisely why combining them with reference-mass characterization, drift correction, and maturity cross-validation is a non-obvious step rather than a routine substitution.

The novel combination is this: a phone placed on fresh concrete uses its own vibration motor as a calibrated excitation source and its own accelerometer as the sensor, tracking how the coupled resonance rises as the concrete stiffens, then cross-checks that direct stiffness signal against a temperature-based maturity estimate to catch the mix changes that silently break maturity-only systems. No embedded sensors, no dedicated instrument, no lab wait.

## Detailed Description

### 1. System Overview and Operating Modes

The system comprises a software application running on a consumer smartphone and, optionally, a passive coupling puck. Three operating modes are disclosed.

**Spot-check mode.** An operator places the phone flat on the concrete surface at a marked test location, initiates a measurement (60–90 seconds), and receives an estimated strength with a confidence interval. This mode serves formwork-removal and traffic-opening decisions at discrete locations.

**Continuous monitoring mode.** A low-cost or retired smartphone is left in contact with the concrete (screen-up, in a sealed enclosure protecting against alkaline pore water, with theft deterrence appropriate to the site), taking scheduled measurements every 15–60 minutes through the curing period. This could convert a used phone into a semi-permanent strength node at a hardware cost comparable to a consumable embedded maturity sensor, with the difference that the phone is retrievable afterward.

**Fleet mapping mode.** Multiple phones distributed across a large placement (slab, deck, pavement section) take synchronized measurements via NTP (Network Time Protocol); results are aggregated into a spatial strength map identifying cold spots, slow-curing zones, and areas ready for loading. Spatial registration uses operator-entered grid coordinates or phone GNSS (Global Navigation Satellite System) fixes.

### 2. Excitation Subsystem

The excitation source is the phone's built-in haptic vibration actuator: a linear resonant actuator (LRA, typical resonant frequency 150–235 Hz depending on model) or an eccentric rotating mass (ERM) motor on older devices. The arbitrary-waveform chirp described below is the Android path, driven through `VibrationEffect.createWaveform` (API 26+) with amplitude control (API 29+). On iOS, Core Haptics does not expose arbitrary waveform streaming; the iOS path approximates the sweep as a scheduled sequence of transient and continuous haptic events stepping through the band.

A logarithmic swept-sine chirp spanning 40–400 Hz over 3 seconds, repeated 8–16 times per measurement with 500 ms quiet gaps, is the preferred excitation. The logarithmic sweep concentrates energy at low frequencies where slab resonances occur while still covering the LRA's efficient band. A stepped-sine alternative (discrete tones, 5 Hz steps, 200 ms dwell) is disclosed for actuators with poor low-frequency authority, trading measurement time for low-band energy.

Before field use, the application runs a one-time actuator characterization: the phone is placed on a rigid reference mass (a steel plate or a fully cured concrete block of known properties) and the drive-to-acceleration response is recorded. This reference transfer function captures the actuator's own frequency response, including its resonant peak, so that field measurements can be normalized. Because LRA resonant frequency drifts with temperature (magnet and spring stiffness are temperature-dependent), the application re-measures a brief free-air reference chirp before each field session. The measured IMU response to that free-air chirp, rather than the commanded electrical waveform, serves as the effective input reference for the transfer-function computation, which partially closes the loop on the actuator's nonlinear, temperature-drifting, OEM-dependent dynamics.

Actuator force authority is small, estimated at 1–2 N for a phone LRA (roughly the weight of an apple; a rough estimate, not a measured specification). The design hypothesis is that this is sufficient because the measurement is of the near-field coupled response, not far-field wave propagation: the phone and the concrete beneath it form a single vibrating system, and even small forces produce measurable acceleration at resonance. A frank tension is disclosed here: the expected fresh-concrete resonance (40–80 Hz) sits below the LRA's own resonance (150–235 Hz), in the actuator's weakest band, with output rolling off steeply below resonance. The design closes this gap with coherent averaging (√N noise suppression over 8–16 repetitions), resonant amplification of the coupled system at its own lower resonance, and the stepped-sine alternative that dwells longer per tone in the low band. Whether the resulting signal-to-noise ratio suffices is an open experimental question and the central risk of the design.

### 3. Response Acquisition

On the sensing side, the phone's IMU accelerometer does the listening: typically a Bosch BMI160/BMI270-class or ST LSM6DSx-class MEMS device offering 16-bit resolution at ±2 g full scale, with noise density in the 75–180 µg/√Hz range (the range documented for production phones in Kang et al.: 75 for the ST LSM6DSM in a Huawei P20 Pro, 180 for the Bosch BMI160 in a Xiaomi Redmi Note 4). Sampling starts with the fastest available rate request (`SENSOR_DELAY_FASTEST` on Android; 100 Hz on iOS, capped by Core Motion). As a design target, 200 Hz minimum sustained sampling is required, enough to resolve vibrations up to 100 Hz. Capturing the full 40–400 Hz chirp band requires 800 Hz or better; devices sustaining 400–500 Hz are usable to roughly 200 Hz. Devices capped near 100 Hz operate in a degraded mode that tracks only the lowest structural mode below about 45 Hz, with widened uncertainty bounds.

For each measurement the phone lies flat on the concrete, screen up, coupled by gravity, with its case removed. On rough or textured surfaces (broom-finished slabs, exposed aggregate), a 50 mm diameter, 10 mm thick disc of compliant elastomer (Shore A 40–60) with a phone-sized recess sits between phone and concrete as an optional coupling puck, ensuring repeatable contact stiffness. The puck's own resonance is characterized once and deconvolved (mathematically removed from the measurement).

Each measurement records all three accelerometer axes. The vertical axis carries the primary signal, while the horizontal axes feed a coupling-quality metric: excessive horizontal energy indicates rocking or sliding, and the measurement is flagged.

Detectability is an open question, but the orders of magnitude are these: at 75–180 µg/√Hz, integrated over a 100 Hz analysis bandwidth, the sensor noise floor is roughly 1–2 mg, improving by a factor of √N (about 2.8–4×) with coherent averaging over 8–16 repetitions. The resonant response must clear that floor. With only an estimated 1–2 N of actuator authority driving a coupled system whose effective mass includes the concrete beneath the phone, whether it does is the central experimental risk noted above.

### 4. Signal Processing and Resonance Extraction

In plain terms: for each frequency in the chirp, the system divides the measured vibration by the known drive strength, producing a map of how efficiently the phone-concrete system converts drive into motion at each frequency. Peaks in that map are resonances.

Formally, for each chirp repetition the system computes the H1 transfer function estimate H(f) = S_yx(f) / S_xx(f), where x is the effective drive reference (the measured free-air IMU response from §2) and y is the measured acceleration, S_yx the cross-spectral density, S_xx the drive auto-spectral density. Spectra use 1024-point FFTs with Hann windowing and 50% overlap (about 5 seconds per window at 200 Hz sampling, roughly one chirp period). Averaging H(f) across the 8–16 repetitions suppresses uncorrelated noise by roughly √N.

Resonant peaks are extracted by peak-picking on |H(f)| followed by Lorentzian curve fitting, fitting a symmetric bell-shaped curve to each peak to pinpoint its center frequency f₀ and half-power bandwidth Δf. The damping ratio ζ = Δf / (2f₀) is retained as an experimental secondary feature; contact damping is expected to dominate over any pore-structure signal, so the hydration information content of ζ is unvalidated and it is not a primary estimator input.

Two quality gates apply. First, coherence gating: the magnitude-squared coherence between drive reference and response must exceed 0.9 across the fitted peak; lower coherence indicates poor coupling or external vibration contamination (nearby equipment, traffic), and the repetition is discarded. Second, repeatability gating: the fitted f₀ across accepted repetitions must agree within ±2%; wider spread indicates an unstable setup (phone sliding on a wet surface, operator handling the phone).

Note that the extracted f₀ is the resonant frequency of the coupled phone-concrete system, not of the concrete alone. Within this coupled system, the phone's mass and the contact stiffness form a subsystem whose resonance shifts as the concrete's dynamic modulus, its stiffness under vibration (denoted E_d), rises during hydration. For a given geometry, f₀ scales with √(k/m): as hydration products stiffen the matrix, contact stiffness k increases and f₀ moves upward. Tracking proceeds differentially against the first post-initial-set measurement on the placement (the baseline), which removes phone-specific and geometry-specific offsets. Typical expected behavior, stated here as an untested design hypothesis, is a baseline near 40–80 Hz on fresh concrete rising toward 120–250 Hz as the concrete approaches its 28-day stiffness, with the exact band depending on slab thickness, subgrade, and phone mass. A disclosed risk: the actuator's own resonance (150–235 Hz) lies inside the chirp band and near the upper end of the hypothesized cured-concrete track, so the extracted f₀ may be dominated by the actuator's resonance rather than the concrete's stiffness; the reference-mass normalization is an unvalidated design hypothesis, and a field build must demonstrate that the tracked peak actually moves with hydration.

### 5. Maturity Fusion

In parallel with resonance tracking, the application maintains a temperature-history maturity index per ASTM C1074. Temperature sources, in order of preference: a Bluetooth LE temperature probe placed adjacent to the test location; the phone's own internal thermistor (corrected for self-heating using a pre-characterized offset model); or ambient temperature from a weather API as a fallback. Both the Nurse-Saul temperature-time factor and the Arrhenius equivalent age are computed: M = Σ(T − T₀)·Δt (datum temperature T₀ typically −10 °C or 0 °C per project specification) and t_e = Σ exp[−(E/R)·(1/T − 1/T_r)]·Δt, with activation energy E in the 33–45 kJ/mol range typical of Type I/II cements (user-adjustable for mixes heavy in supplementary cementitious materials (SCMs)).

This cross-validation is the principal contribution of the disclosure. Resonance measures stiffness directly but is geometry-sensitive; maturity is geometry-independent but infers strength indirectly and needs calibration. Cross-validation works as follows: when the resonance-derived stiffness trend and the maturity-derived strength estimate agree within tolerance, confidence is high; when they diverge beyond tolerance (for example, resonance stalls while maturity advances, suggesting a mix change or sensor fault), the system flags the measurement for operator review rather than silently reporting a number. This disagreement detection is designed to catch the mid-project mix changes that silently invalidate maturity-only systems.

### 6. Strength Estimation Model

A small supervised model serves as the estimator, deployed on-device via TensorFlow Lite (target size under 100 KB): either gradient-boosted decision trees or a two-layer multilayer perceptron. Inputs: current f₀, shift of f₀ from baseline, Nurse-Saul maturity, Arrhenius equivalent age, and mix metadata (water-cementitious ratio, cement type, supplementary cementitious materials and dosages, admixture flags, specified strength f'c). Output: estimated compressive strength in MPa or psi, with a prediction interval.

Training proceeds in two stages. Population pre-training: the model would ship with weights trained on a fleet dataset of paired measurements (phone resonance tracks plus maturity histories plus companion cylinder breaks at 1, 3, 7, 14, and 28 days) contributed across projects; no such dataset exists yet, and assembling it is future work. Project calibration: the operator optionally enters cylinder-break results for the specific mix; the model fine-tunes its final layer on these points, reproducing the ASTM C1074 calibration workflow with resonance features added. A calibration-free mode is disclosed for non-critical decisions: the model reports strength as a fraction of the mix's specified f'c using population curves, with widened prediction intervals and a clear on-screen disclaimer.

### 7. Decision Outputs

The application issues advisories only; decisions to strip formwork, stress tendons, saw-cut, or open pavement to traffic remain with the engineer of record and the project specifications, so the thresholds below are configuration starting points, not engineering recommendations. Defaults reflect common specification practice consistent with ACI 347 (formwork) and ACI 318 (post-tensioning, commonly 75% of f'c or greater):

- **Formwork / shoring removal advisory:** per-location readiness against the configured threshold (default 70% of f'c; for a 4000-psi mix that is 2800 psi, roughly the point where the concrete carries its own weight), with the margin above or below shown.
- **Post-tensioning readiness:** flags when the lowest-reading location in the stressed zone clears the drawing-specified threshold.
- **Saw-cut window:** advisory issued when estimated strength enters the project-configured early-entry window, with elapsed time shown alongside; cutting too early ravels the joint, cutting too late lets shrinkage cracks beat the saw.
- **Cold-spot alerts:** in fleet mode, locations lagging the placement mean by more than a configured margin trigger inspection prompts (possible cold joint, poor consolidation, or water ponding).

Every measurement is written to a timestamped, append-only audit log (UTC timestamps, device ID, grid coordinates or GNSS fix, threshold configuration, estimate and interval), exportable as CSV or PDF for the project record.

### 8. Figures Description

- **Figure 1:** System block diagram: smartphone (actuator drive through haptic actuator; IMU accelerometer) coupled to concrete surface; signal processing chain (transfer function, peak fit, f₀ extraction); maturity module (temperature through Nurse-Saul/Arrhenius); fusion estimator with disagreement detection; decision outputs and audit log.
- **Figure 2:** Example transfer-function magnitude plots at 6 hours (post-initial-set), 24 hours, 3 days, and 7 days after placement, showing the resonant peak migrating upward in frequency as hydration proceeds.
- **Figure 3:** Fusion architecture: resonance track and maturity index as independent inputs to the on-device strength estimator, with a disagreement-detection comparator gating the reported estimate.
- **Figure 4:** Fleet mapping mode: plan view of a slab with phone node locations, interpolated estimated-strength contours, and flagged cold spots.

## Claims

1. A method for estimating the compressive strength of curing concrete, comprising: placing a mobile computing device comprising a vibration actuator and an accelerometer in contact with a concrete surface; driving the vibration actuator with a known excitation waveform; recording an acceleration response with the accelerometer; computing a transfer function between the excitation waveform and the acceleration response; extracting a resonant frequency from the transfer function; and estimating concrete compressive strength based at least in part on the resonant frequency.

2. The method of claim 1, wherein the known excitation waveform is a swept-sine chirp or a stepped approximation thereof spanning 40–400 Hz, and the resonant frequency is extracted by Lorentzian curve fitting of transfer-function magnitude peaks.

3. The method of claim 1, further comprising tracking the shift of the resonant frequency relative to a baseline measurement on the same placement, wherein increasing resonant frequency indicates increasing concrete stiffness from cement hydration.

4. The method of claim 1, further comprising computing a damping ratio from the half-power bandwidth of a resonant peak of the transfer function and recording the damping ratio as an experimental secondary curing-progress indicator.

5. The method of claim 1, further comprising computing a temperature-history maturity index according to ASTM C1074 from a temperature record associated with the concrete, and fusing the maturity index with the resonant frequency in a calibrated estimation model.

6. The method of claim 5, wherein the maturity index comprises at least one of a Nurse-Saul temperature-time factor and an Arrhenius equivalent age, and the method further comprises a disagreement-detection step that flags the estimate for review when the resonance-derived stiffness trend and the maturity-derived estimate diverge beyond a tolerance.

7. The method of claim 5, wherein the calibrated estimation model is an on-device machine learning model receiving as inputs the resonant frequency, the maturity index, and concrete mix-design metadata, and outputting an estimated compressive strength with a prediction interval.

8. The method of claim 1, further comprising quality-gating the measurement by requiring magnitude-squared coherence between excitation and response above 0.9 and repeatability of the extracted resonant frequency within ±2% across repeated excitations.

9. The method of claim 1, further comprising characterizing the vibration actuator on a rigid reference mass and applying a temperature-drift correction derived from a free-air reference chirp before each measurement session.

10. A system for in-place concrete strength estimation, comprising: a smartphone with a linear resonant actuator, a MEMS accelerometer, and a processor; and software instructions that, when executed, cause the smartphone to emit a swept-sine excitation (or a stepped approximation thereof on platforms without arbitrary waveform streaming) through the actuator, record the acceleration response, extract a coupled-system resonant frequency via transfer-function analysis, fuse the resonant frequency with an ASTM C1074 maturity index, and report an estimated compressive strength with a field-decision advisory selected from formwork removal, post-tensioning readiness, saw-cut timing, and traffic-opening readiness.

11. The method of claim 1, further comprising deploying a plurality of mobile computing devices across a concrete placement, synchronizing their measurements, and aggregating per-location strength estimates into a spatial strength map identifying slow-curing zones.

## Implementation Notes

This disclosure describes a design proposal. No prototype has been built and no field validation has been performed; all quantitative parameters (including but not limited to chirp bands, repetition counts, coherence thresholds, repeatability tolerances, FFT window lengths, puck dimensions, monitoring intervals, and expected frequency ranges) are untested design choices that would need calibration against real placements. Several practical points follow.

The actuator is the principal constraint. Phone LRAs are designed for fingertip haptics, not structural excitation: force authority is small, low-frequency output rolls off steeply below the actuator's own resonance, and drive APIs are constrained (iOS Core Haptics exposes continuous and transient events but not arbitrary waveform streaming; Android amplitude control requires API 29+ and varies by OEM). The swept chirp must be designed around these limits, and the actuator characterization step is not optional.

Sampling rate is the second constraint. Many Android devices do not sustain the sensor's maximum output data rate through the framework; 200 Hz sustained is a realistic floor, which caps the usable chirp at about 90 Hz on those devices and motivates the degraded low-band mode. iOS Core Motion delivers accelerometer data at 100 Hz, similarly constraining.

Coupling dominates repeatability. A phone on a steel-troweled slab couples differently than on a broom finish; a phone in a thick rubber case couples differently than a bare phone. Operators must remove cases, clean the contact area, and use the coupling puck on textured surfaces. The horizontal-axis rocking metric exists to catch bad setups, but it cannot fix them.

Two physics risks are stated explicitly because they could each invalidate the method. First, the extracted resonance may be dominated by the actuator's own resonance (150–235 Hz) rather than the concrete's stiffness; the reference-mass normalization is an unvalidated design hypothesis, and a field build must demonstrate that the tracked peak actually moves with hydration rather than with actuator temperature. Second, the method interrogates the near-surface zone beneath the phone, which in fresh concrete is the laitance and bleed-water layer: the weakest, least representative stratum. Contact stiffness may therefore track surface finish quality more than bulk compressive strength, and the correlation must be validated per placement type.

Measurement becomes valid at or after initial set. Fresh concrete is a viscous fluid in the first hours; the coupled-oscillator contact model assumes a solid, and a phone placed on fluid concrete will settle or partially embed. Pre-set readings are not claimed, and the 6-hour baseline in Figure 2 should be read as post-initial-set.

What this system does not do: it does not replace acceptance testing. ASTM C39 cylinders remain the legal basis for accepting concrete against its specified strength, and nothing here changes that. It does not measure rebar corrosion, alkali-silica reaction, freeze-thaw damage, or any durability property; it tracks stiffness gain during curing, nothing more. It cannot see deep inside the placement; it interrogates the near-surface zone beneath the phone. It does not work on concrete that has already fully cured (the resonance track saturates; the method is for the curing window).

Privacy is structural. The sensing channel is inertial: no microphone, no camera, no audio is ever captured. The audit log holds primarily construction data (timestamps, estimates, coordinates); device identifiers should be pseudonymous when operator-owned phones are used, since a device ID combined with coordinates and timestamps can identify an individual.

## Prior Art References

1. [ASTM C215](https://www.astm.org/): Standard Test Method for Fundamental Transverse, Longitudinal, and Torsional Resonant Frequencies of Concrete Specimens (laboratory resonance method)
2. [ASTM C1074](https://www.astm.org/): Standard Practice for Estimating Concrete Strength by the Maturity Method (Nurse-Saul and Arrhenius maturity functions; mix-specific calibration)
3. [ASTM C39](https://www.astm.org/): Standard Test Method for Compressive Strength of Cylindrical Concrete Specimens (acceptance basis)
4. [AASHTO T 412-24](https://wavelogix.tech/wavelogix-rebel-becomes-new-national-standard-for-estimating-concrete-strength/): Standard Method of Test for Estimating Concrete Strength by the Acoustical Resonance Method (in-place concrete; approved August 2024; basis for the Wavelogix REBEL system)
5. [Wavelogix: Concrete Maturity Testing, How It Works, Limitations and Alternatives](https://blog.wavelogix.tech/blog/concrete-maturity-testing-how-it-works-limitations-alternatives): vendor-published comparison of the maturity method (ASTM C1074) against acoustical resonance (AASHTO T-412), including the vendor's reported variability figures
6. [Giatec SmartRock Concrete Sensors](https://certifiedmtp.com/giatec-smartrock-wireless-maturity-sensor-12in/): embedded wireless maturity sensors; temperature measurement every 15 minutes; ASTM C1074 strength estimation; consumable single-pour sensors
7. [Giatec: Concrete Maturity and In-Situ Compressive Strength, ASTM C1074 FAQ](https://www.giatecscientific.com/education/concrete-maturity-in-situ-compressive-strength/?pdf=5653): maturity method principles, calibration requirements, recording intervals
8. [Sensytec: Concrete Sensors, How They Work, Accuracy, ASTM C1074 Guide](https://www.sensytec.com/concrete-sensors.html): wireless concrete sensor principles; vendor-reported agreement figures against cylinder breaks
9. [Wake Inc: How to Validate Maturity Curves for Concrete](https://www.wakeinc.com/blog/how-to-validate-maturity-curves.html): mix-specific calibration procedure; 1/3/7/14/28-day break programs; datum temperature and activation energy selection
10. [Kang et al., Buildings 2023](https://www.mdpi.com/2075-5309/13/4/1069/xml): evaluation of vibration detection using smartphones in a reinforced-concrete frame building; phone-derived natural periods agree with reference accelerometers; production-phone IMU noise figures
11. [Smartphone-based structural health monitoring review, Preprints 2025](https://www.Preprints.org/manuscript/202507.0719): survey covering MyShake (building frequencies within 5% error), bridge modal-analysis apps, and citizen-sensor networks; passive sensing only

---

*Published at [liveinthefuture.org/priorart](https://liveinthefuture.org/priorart/)*
