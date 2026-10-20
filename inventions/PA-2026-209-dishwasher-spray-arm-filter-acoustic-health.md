# PA-2026-209: Dishwasher Spray Arm and Filtration Health from Passive Acoustic Rotation Signatures

**Title:** System and Method for Dishwasher Spray Arm and Filtration Health Assessment Using Passive Acoustic Rotation Signatures Fused with Drain Hydraulics Telemetry

**Filing:** LITF-PA-2026-209
**Published:** October 20, 2026
**Domain:** Kitchen Appliances / Predictive Maintenance / Sensor Fusion
**Full Disclosure:** [liveinthefuture.org/priorart/dishwasher-spray-arm-filter-acoustic-health.html](https://liveinthefuture.org/priorart/dishwasher-spray-arm-filter-acoustic-health.html)
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

Disclosed is a system and method for assessing the mechanical health of a
residential dishwasher's spray arms and filtration system using passive acoustic
sensing from outside the machine, fused with drain-phase hydraulics telemetry.

A dishwasher's cleaning performance degrades silently. Spray arm jets clog with
food particles, mineral scale, and detergent residue; the fine-mesh filter
assembly at the sump loads with debris. Owners notice only when dishes come out
dirty, by which point the machine has been underperforming for many cycles.
Every major manufacturer instructs owners to clean the filter regularly,
typically monthly. Almost nobody does.

The disclosed system places a microphone near the dishwasher: a smartphone on
the countertop (zero-installation), an adhesive-mounted piezoelectric contact
sensor on the cabinet panel, or a built-in microphone in smart models. Rotating
spray arms imprint a periodic amplitude modulation on the broadband spray noise
at each arm's rotation frequency (typically 0.15-1.0 Hz, i.e., 10-60 RPM). The
system segments the wash cycle into acoustic phases (fill, wash, drain, dry)
with a hidden Markov model, computes the Hilbert envelope spectrum of
wash-phase audio in the 1-8 kHz spray band, and identifies per-arm rotation-rate
peaks. From these it derives a per-arm jet clogging index (rotation rate
depression, modulation depth increase, high-frequency spectral balance shift)
and, from drain-phase audio, a filter loading index (drain duration, drain pump
current, cavitation band energy). A fusion layer separates filter-caused
multi-arm slowdown from single-arm jet clogs. An on-device classifier produces
tiered advisories naming the exact part and action, with trend projection.
Raw audio is processed on-device and never transmitted.

> *The filter hides its loading behind a baffle and the manual lives in a drawer; the microphone hears what neither shows.*

---

## Technical Field

This disclosure relates to kitchen appliance maintenance, specifically to
passive acoustic condition monitoring of dishwasher spray arm rotation and
filtration loading, fused with hydraulic telemetry from the drain and fill
phases of the wash cycle.

---

## Background

A residential dishwasher cleans by pumping heated water through rotating spray
arms. The arms spin because their jets are angled: water pressure both sprays
the dishes and drives rotation, typically 10 to 60 revolutions per minute
depending on the arm, pump pressure, and load. Cleaning performance depends on
every jet firing at full pressure. When jets clog with food particles, mineral
scale, or detergent residue, the affected arm slows or stalls, and the dishes
in its spray shadow come out dirty.

Filtration is the other half of the system. Modern dishwashers use a removable
fine-mesh filter assembly at the sump to trap food debris before it reaches the
pump. Every major manufacturer instructs owners to clean this filter regularly,
typically monthly. In practice, almost nobody does. A loaded filter restricts
flow to the wash pump, starves the spray arms of pressure, extends cycle times,
and forces the drain pump to work against higher resistance. The failure mode
is silent: cleaning degrades gradually over weeks, and the owner notices only
when dishes are visibly dirty, by which point the machine has been
underperforming for many cycles.

Existing in-machine sensing is limited. Higher-end dishwashers carry turbidity
sensors (for example, the AquaSensor in Bosch models) that measure soil in the
wash water and adjust cycle length, but these sensors report water clarity, not
mechanical health: they cannot tell a clogged spray arm from a lightly soiled
load. No consumer system continuously monitors spray arm rotation or filter
loading, and there is no standard diagnostic for either condition short of
opening the machine mid-cycle or removing and inspecting parts by hand.

Acoustic and vibration condition monitoring is established practice for
rotating machinery: acoustic emission analysis detects bearing faults from
characteristic frequencies (Hidle et al., Sensors 2022;
https://www.mdpi.com/1424-8220/22/14/5187/xml), and audio-based inspection
with machine learning reaches high fault detection accuracy on industrial
equipment (MDPI Sensors review, 2023;
https://www.mdpi.com/1424-8220/23/4/1902/htm). A dishwasher is, mechanically,
a small pump driving rotating arms through a water circuit: exactly the kind
of machine whose acoustic signature carries its health state.

The gap is a passive, zero-installation health assessment that: (a) captures
dishwasher acoustics with hardware the owner already has, (b) separates the
spray arms' rotation signatures from pump, fill, drain, and ambient kitchen
noise, (c) quantifies jet clogging and filter loading as trackable indices
rather than binary alarms, and (d) converts the result into the specific
maintenance actions owners already understand: clean the filter, clear the
jets, call for service.

---

## Detailed Description

### 1. Sensing Configuration

The system captures audio with a microphone positioned outside the dishwasher
tub. Three configurations are disclosed. In the preferred zero-installation
configuration, the owner's smartphone rests on the countertop within 2 meters
of the dishwasher during the cycle; the phone's microphone samples at 16 kHz.
In the semi-permanent configuration, an adhesive-mounted piezoelectric contact
sensor attaches to the dishwasher's outer cabinet panel or an adjacent
cabinet, capturing structure-borne vibration with better signal-to-noise ratio
and no airborne kitchen noise. In the integrated configuration, a microphone
built into a smart dishwasher's control panel or door captures the same
signals with fixed geometry. All processing runs on-device: on the phone's
processor, on a microcontroller in the sensor module, or on the appliance's
control board. Raw audio is discarded after feature extraction; only feature
vectors and indices leave the device, and then only with the owner's consent
for fleet-level model improvement.

### 2. Cycle Phase Segmentation

A dishwasher cycle is a sequence of acoustic phases, and each health feature
is meaningful only within its phase. The system segments the cycle
automatically. The fill phase is identified by the sustained narrowband hiss
of the inlet valve (typically 2-6 kHz) lasting 30-120 seconds. The wash phase
is identified by the broadband roar of the circulation pump plus spray noise,
lasting 20-90 minutes, with the spray arms' modulation riding on top. The
drain phase is identified by the drain pump's distinctive lower-frequency tone
plus gurgling as the sump empties, lasting 1-4 minutes. The dry phase is
comparatively quiet with periodic fan or heating-element relay clicks.
Segmentation uses a hidden Markov model over short-time spectral features,
trained once per dishwasher model family and adapted to the specific machine
during a two-cycle calibration period. Phase boundaries gate all downstream
feature extraction: rotation features come only from wash-phase audio, filter
features only from drain-phase audio.

### 3. Spray Arm Rotation Rate Estimation

Each rotating spray arm modulates the broadband spray noise at its rotation
frequency. As an arm sweeps, the distance and angle between its jets and the
tub walls change periodically, producing an amplitude modulation of the spray
noise envelope at the arm's rotation rate (typically 0.15-1.0 Hz, i.e., 10-60
RPM) and its harmonics. The system computes the Hilbert envelope of the
wash-phase audio in the 1-8 kHz spray band, then takes the power spectrum of
that envelope over 60-second windows. Peaks in the 0.1-1.2 Hz band correspond
to arm rotation rates. Most residential dishwashers have two or three arms
(lower, middle, upper); the arms rotate at different rates because they differ
in diameter, jet count, and the pressure available at their height, so their
modulation peaks are separable. The system tracks each detected peak across
the wash phase and across cycles, producing a per-arm rotation rate time
series. A healthy arm holds a stable rate for a given cycle program and load;
a clogged or obstructed arm shows a depressed rate, erratic rate wandering,
or a missing peak (stalled arm).

### 4. Jet Clogging Index

Clogged jets change the spray noise in two measurable ways. First, fewer open
jets means less total high-frequency spray energy: the spectral balance of the
wash-phase audio shifts, with energy in the 4-8 kHz band falling relative to
the 1-4 kHz band as jets clog. Second, clogging is rarely uniform: a partially
clogged arm produces an asymmetric spray pattern, which deepens the amplitude
modulation (the loud-quiet contrast per revolution grows as the working jets
cluster on one side). The jet clogging index combines three normalized
features: (a) rotation rate depression relative to the machine's calibrated
baseline for the same cycle program, (b) modulation depth increase at the
arm's rotation frequency, and (c) high-frequency spectral balance shift. Each
feature is z-scored against the calibration baseline; the index is the
weighted sum, scaled 0-100. An index above 60 sustained across three
consecutive cycles triggers a "clear the spray arm jets" advisory naming the
affected arm (lower, middle, or upper) based on which modulation peak
degraded. The advisory includes the standard remediation: remove the arm,
rinse under running water, and clear jet openings with a soft brush or
toothpick, per manufacturer guidance.

### 5. Filter Loading Index from Drain Hydraulics

The filter assembly sits between the sump and the pumps, so its loading
affects both wash and drain phases, but the drain phase gives the cleanest
measurement. A loaded filter restricts flow: the drain pump runs longer to
empty the sump, and the pump's acoustic signature shifts as it works against
higher resistance, with increased cavitation noise (broadband energy in the
2-5 kHz band with a rough, crackling character distinct from normal pump
tone). The filter loading index combines: (a) drain phase duration relative
to the calibrated baseline for the same cycle program, (b) drain pump current
draw where available (integrated configuration, or a smart plug measuring the
dishwasher's power in the phone configuration), and (c) cavitation band energy
during the drain phase. The index is scaled 0-100 against the calibration
baseline. An index above 60 sustained across three consecutive cycles triggers
a "clean the filter" advisory with the manufacturer's recommended procedure:
twist out the filter assembly, rinse under running water, and remove trapped
debris. Because the index is tracked per cycle, the system also projects the
date the filter will reach the advisory threshold, so the owner gets a "clean
the filter within the next week" heads-up rather than a surprise.

### 6. Wash Performance Degradation Fusion

Spray arm and filter health interact: a loaded filter starves all arms of
pressure, depressing every arm's rotation rate simultaneously, while a clogged
arm affects only its own modulation peak. The fusion layer uses this
distinction to separate root causes. If all arms slow together and the filter
index is elevated, the advisory leads with filter cleaning. If one arm's peak
degrades while the others hold, the advisory names that arm's jets. Secondary
telemetry sharpens the picture: fill volume inferred from fill-phase duration
(a lengthening fill suggests inlet valve or pressure issues rather than spray
problems), heater duty cycle inferred from the power signature (extended
heating suggests the machine compensating for poor spray coverage), and total
cycle duration drift (the machine's own turbidity-driven extensions, which
lengthen as cleaning efficiency falls). These signals feed a wash performance
score, also 0-100, that the owner can watch trend downward weeks before dishes
start coming out dirty.

### 7. On-Device Classifier and Advisories

A lightweight gradient-boosted decision tree (under 200 KB, runnable on a
phone processor or appliance microcontroller) takes the per-cycle feature
vector: per-arm rotation rates and modulation depths, jet clogging indices,
filter loading index, drain duration, fill duration, heater duty estimate, and
cycle duration drift. It outputs the advisory tier: none, filter cleaning due,
spray arm jet clearing due, or service recommended (for patterns matching pump
bearing wear or inlet valve failure, which the acoustic features also reveal
but which are outside owner-serviceable scope). Advisories are delivered as
phone notifications or appliance display messages, each naming the specific
part and the specific action. Trend projection uses linear extrapolation of
the indices over the trailing 30 cycles to estimate cycles-until-threshold,
presented as "about N more washes." A fleet mode (opt-in) aggregates
anonymized indices by dishwasher model to establish model-specific baselines
and to detect design-level issues, such as a model family whose upper arm
clogs at twice the fleet rate.

### 8. Figures Description

- **Figure 1:** System overview showing the three sensing configurations
  (countertop smartphone, adhesive cabinet sensor, integrated appliance
  microphone) capturing wash-cycle audio, with on-device processing producing
  maintenance advisories.
- **Figure 2:** Cycle phase segmentation example: spectrogram of a full wash
  cycle with fill, wash, drain, and dry phases labeled, showing where
  rotation and filter features are extracted.
- **Figure 3:** Spray arm modulation analysis: Hilbert envelope spectrum of
  wash-phase audio showing separable rotation peaks for lower, middle, and
  upper arms, with a degraded peak indicating a clogged arm.
- **Figure 4:** Advisory flow: feature extraction to per-arm clogging and
  filter loading indices, fusion logic distinguishing filter-caused from
  jet-caused degradation, and the tiered advisory output with trend
  projection.

---

## Claims

1. A system for assessing dishwasher spray arm health, comprising: a
   microphone positioned outside the dishwasher tub; a cycle phase
   segmentation module that identifies wash-phase audio; and a rotation
   estimation module that computes the amplitude modulation spectrum of the
   wash-phase audio envelope and identifies rotation-rate peaks corresponding
   to individual spray arms.
2. The system of claim 1, wherein the microphone is a smartphone microphone,
   an adhesive-mounted piezoelectric contact sensor on the dishwasher cabinet,
   or a microphone integrated into the dishwasher control panel or door.
3. The system of claim 1, further comprising a jet clogging index module that
   combines spray arm rotation rate depression, amplitude modulation depth
   increase, and high-frequency spectral balance shift into a per-arm clogging
   score normalized against a calibration baseline.
4. The system of claim 3, wherein the jet clogging index identifies the
   affected spray arm as lower, middle, or upper based on which rotation-rate
   peak degraded.
5. A system for assessing dishwasher filter loading, comprising: a microphone
   positioned outside the dishwasher tub; a cycle phase segmentation module
   that identifies drain-phase audio; and a filter loading index module that
   combines drain phase duration, drain pump current draw, and drain-phase
   cavitation band energy into a loading score normalized against a
   calibration baseline.
6. The system of claim 5, further comprising a trend projection module that
   extrapolates the filter loading index over trailing wash cycles to estimate
   the number of cycles remaining before a filter cleaning threshold is
   reached.
7. A method for diagnosing dishwasher wash performance degradation,
   comprising: estimating per-arm spray arm rotation rates from passive
   acoustic amplitude modulation; computing a per-arm jet clogging index;
   computing a filter loading index from drain-phase hydraulics; and applying
   a fusion rule that attributes simultaneous multi-arm rotation depression
   with elevated filter loading to filter restriction, and single-arm rotation
   degradation to jet clogging of the affected arm.
8. The method of claim 7, further comprising inferring fill volume from
   fill-phase acoustic duration, heater duty cycle from the power signature,
   and cycle duration drift, and combining these into a wash performance score
   tracked across cycles.
9. The system of claim 1, wherein all acoustic processing runs on-device, raw
   audio is discarded after feature extraction, and only feature vectors and
   health indices are retained or transmitted.
10. The system of claim 1, further comprising an advisory module that issues
    tiered maintenance advisories (filter cleaning due, spray arm jet clearing
    due naming the affected arm, or service recommended) when a health index
    exceeds a threshold across a configurable number of consecutive cycles.
11. The system of claim 10, further comprising an opt-in fleet mode that
    aggregates anonymized health indices by dishwasher model to establish
    model-specific baselines and detect model-family-level spray arm or
    filtration issues.

---

## Implementation Notes

This is a design disclosure: no prototype has been built and no field
validation has been performed, so every quantitative parameter above
(rotation rate bands, index thresholds, cycle counts, model sizes) is an
untested design choice, not a measured result. The disclosure is intended to
enable a person of ordinary skill in the art to build the system, and to
place the core idea (passive acoustic spray arm and filter health assessment
for dishwashers, fused with drain hydraulics) in the public record as prior
art.

The strongest case against this system is acoustic variability. Kitchen
ambient noise (range hoods, conversation, other appliances) contaminates
airborne captures; phone placement changes the transfer function; different
dishwasher models differ in insulation, pump power, and arm geometry; and
loading variation (a tall pot blocking a jet) mimics clogging. The design
answers each with a specific mechanism: phase gating discards non-wash audio,
the contact-sensor configuration bypasses airborne noise, per-machine
calibration absorbs model and placement differences, and the
three-consecutive-cycle rule averages out loading variation. The second
objection is that turbidity sensors already adapt the cycle. They do, but
they measure water clarity, not mechanical health: a turbidity sensor
lengthens the cycle when water stays dirty without knowing whether the cause
is a clogged arm, a loaded filter, or a heavily soiled load, and it never
tells the owner which part to service. The third objection is accuracy: can
envelope modulation really resolve individual arm rotation rates through a
cabinet panel? The claim here is that the rotation frequencies are low
enough (under 1.2 Hz) and the modulation coherent enough over 60-second
windows that the peaks separate, provided the calibration baseline is
per-machine; a builder should verify this first by recording a dishwasher
with a known stalled arm (one arm zip-tied) and checking that the
corresponding peak disappears while the others persist.

Calibration matters more than the model. Dishwasher models differ in pump
power, arm geometry, insulation, and cycle programs, so the two-cycle
calibration period is not optional: baselines must be learned per machine and
per cycle program. A baseline learned on "Normal" does not transfer to
"Heavy" or "Quick."

What this system does not do: it does not measure actual soil removal or dish
cleanliness, only the mechanical conditions that enable cleaning. It cannot
distinguish a clogged jet from a jet blocked by a tall dish loaded in front
of it; the three-consecutive-cycle rule exists partly to average out loading
variation, but owners should still check loading before assuming a clog. It
does not detect leaks, heating element failure beyond duty-cycle inference,
or detergent dosing problems.

Privacy is structural, not promised. Kitchen audio can contain speech. The
on-device processing requirement means speech never needs to leave the
device; feature extraction should run on short frames that are discarded
immediately, and the fleet mode must transmit indices only, never audio or
spectrograms.

---

## Prior Art References

- [10 CFR Part 430, Subpart B, Appendix C2](https://www.ecfr.gov/current/title-10/chapter-II/subchapter-D/part-430) — U.S. DOE uniform test method for measuring the energy consumption of dishwashers
- [Bosch Home Appliances](https://www.bosch-home.com/us/) — AquaSensor turbidity-based wash water soil measurement in dishwasher auto cycles
- [ENERGY STAR Dishwashers](https://www.energystar.gov/products/dishwashers) — U.S. EPA dishwasher efficiency program and maintenance guidance
- [Hidle et al., Sensors 2022: acoustic emission analysis for bearing condition monitoring](https://www.mdpi.com/1424-8220/22/14/5187/xml) — Characteristic-frequency fault detection from acoustic emission in rotating machinery
- [MDPI Sensors review, 2023: audio-based fault inspection with machine learning](https://www.mdpi.com/1424-8220/23/4/1902/htm) — Audio-based fault detection for industrial equipment; the method family this disclosure adapts to dishwashers
- [TensorFlow Lite for Microcontrollers](https://www.tensorflow.org/lite/microcontrollers) — On-device ML runtime for microcontroller-class processors

---

*Published at [liveinthefuture.org/priorart](https://liveinthefuture.org/priorart/)*
