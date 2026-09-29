# PA-2026-185: Deadbolt Alignment Diagnostics and Predictive Lockout Prevention via Actuator Current Signature Analysis

**Title:** System and Method for Deadbolt Alignment Diagnostics and Predictive Lockout Prevention via Actuator Current Signature Analysis with Seasonal Frame-Movement Compensation

**Filing:** LITF-PA-2026-185
**Published:** September 28, 2026
**Domain:** Smart Home / Access Control / Predictive Maintenance
**Full Disclosure:** [liveinthefuture.org/priorart/deadbolt-alignment-current-signature.html](https://liveinthefuture.org/priorart/deadbolt-alignment-current-signature.html)
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

Smart deadbolts report "jammed" only after they fail to throw the bolt, usually at the worst possible moment. This disclosure adds a diagnostic layer that records the actuator motor's current waveform on every lock cycle, builds a per-door mechanical baseline, and splits drift into reversible seasonal frame movement (tracked against outdoor humidity) and progressive misalignment from hinge sag or settling. A position-resolved friction map pinpoints whether the drag is at the strike lip, the bore, or the mechanism, and the companion app tells the homeowner exactly which way to move the strike plate and by how much, weeks before a lockout. Secondary functions include battery end-of-life prediction separated from mechanical load by voltage-normalized current analysis, and forced-entry attempt classification from uncommanded torque transients.

## Technical Field

This disclosure relates to electromechanical door locks, specifically to diagnostic systems that use actuator motor current signature analysis for alignment monitoring, failure prediction, and maintenance guidance in motorized deadbolt locks, and to methods for separating reversible environmental effects from progressive mechanical degradation in door hardware.

## Background

A motorized deadbolt throws a bolt of about 25 mm (1 inch) into a strike bore in the door frame, with clearances on the order of 1 to 2 mm per side. Anything that shifts the door relative to the frame by more than that clearance makes the bolt drag; if drag exceeds the motor's torque capability, the lock jams. Wood doors and frames are hygroscopic: they swell in humid air and shrink in dry air, shifting by millimeters across the seasons, which is exactly the scale that closes a bolt clearance. This is why doors stick in August and swing free in January. Separately, doors drift monotonically: hinges sag, screws loosen, foundations settle. Structural engineers distinguish the two by reversibility: predictable seasonal sticking points to moisture, while sudden or persistent sticking points to movement below.

Existing smart locks are reactive. August's Smart Lock Pro reports open/closed/jammed and its DoorSense sensor detects a door left ajar. Schlage connected deadbolts expose Locked/Unlocked/Jammed. Yale locks raise jam alarms. All are binary, after-the-fact reports. The patent literature has the building blocks but not the system: WO2013168114A1 discloses a motor-current threshold with reverse-and-retry; US10233672B2 discloses stall detection via motor current; US10140828B2 and US20160343188A1 disclose estimating door friction from motor current and reporting it to the user; US20200372738A1 discloses stall detection via accelerometer/knob position. None discloses longitudinal current-signature recording, per-door baselines, seasonal versus progressive decomposition, position-resolved friction mapping, predictive lockout warnings, or directional strike-plate adjustment guidance.

## Detailed Description

### 1. The failure mode

The bolt must travel its full stroke into the strike bore. Vertical shift of the door makes the bolt tip strike the strike-plate lip; lateral shift binds the bolt against the bore wall; depth shift drags the bolt face across the plate. Each geometry produces a different drag signature along the travel, which is why position-resolved sensing matters: where the friction sits in the stroke identifies the direction of misalignment. The drive motor is a small brushed DC gearmotor whose current is proportional to delivered torque (I = (V - k_e*w)/R). A healthy cycle shows an inrush spike, a steady travel current, and a small seat rise; drag raises current in the affected region of the stroke; a stall drives current to the driver limit with no position advance.

### 2. Sensing hardware

One added sensing channel: motor current via a low-side shunt resistor (illustrative 50 mOhm) and a current-sense amplifier of the INA180 class, sampled by the lock MCU's ADC at 1 to 5 kHz during the 1 to 2 second actuation window only, so sensing adds negligible energy. Illustrative BOM cost is under one US dollar at volume. Bolt position comes from the Hall/optical encoder or thumbturn potentiometer many locks already carry, with at least ~20 counts across the stroke (~1 mm resolution); fallback is back-EMF rotation counting or time-normalized position. Outdoor temperature and humidity come from a weather service via the companion app.

### 3. Per-cycle features

Each cycle yields: peak inrush current, travel RMS current, cycle energy (integral of V*i), travel time, seat current, throw/retract asymmetry, and voltage-normalized current (features divided by contemporaneous battery voltage, separating mechanical load from battery sag). Only these features leave the lock; raw waveforms are discarded, keeping payloads to tens of bytes per cycle.

### 4. Position-resolved friction map

Travel current versus bolt position, averaged over a rolling window of recent cycles (illustrative: 50), localizes the drag source. Four canonical signatures: a late-travel spike means the bolt tip strikes the strike-plate lip (vertical misalignment, the classic seasonal signature); uniform elevation means bore friction or mechanism wear; an early-travel hump means mechanism-side binding before the bolt reaches the frame; a mid-travel notch means a burr, paint, or debris at a specific bore depth. Throw and retract maps are computed separately; throw-only elevation indicates angular entry from hinge-side sag.

### 5. Baseline and decomposition

A per-door, per-direction baseline is learned during commissioning (illustrative: first 200 cycles or 30 days). Drift is then decomposed: F(t) = S(H(t), T(t)) + P(t) + noise, where S is the seasonal component modeled as a function of trailing humidity and temperature, and P(t) is the progressive component constrained to be monotonic via isotonic regression. Seasonal drift reverses annually and concentrates in the late-travel region; progressive drift persists through dry months. This operationalizes the structural engineer's reversibility test in the lock's own telemetry.

### 6. Prediction, guidance, and adaptation

The system tracks the margin between peak travel current and the stall threshold, extrapolates the progressive component, and issues tiered warnings: advisory below 50% margin, action below 30% margin or predicted lockout within 60 days, urgent below 15%. Action-level guidance converts the friction map into a concrete instruction ("move the strike plate 2 mm toward the hinge side") with a diagram, bounded by what a screwdriver can do. An installer mode gives live friction readouts during adjustment, detects the step-change improvement, closes the work order, and re-baselines; if nothing improves it escalates (check hinge screws, inspect for settling). Meanwhile, the controller adapts drive within thermal limits, raising torque budget selectively through high-drag stroke regions (informed adaptation, not blind retry), slowing the bolt there since lower speed at fixed PWM yields more torque.

### 7. Battery and forced entry

Cycle energy per lock, tracked against rated capacity, yields a remaining-cycle estimate; voltage-normalized features attribute rising energy to mechanical load versus battery aging, so the app can report "battery at 30%, load normal" versus "battery at 60% but friction is doubling its drain." A secondary classifier watches for uncommanded torque transients: impact-like rapid transients with no motor command are flagged as suspected forced entry (reported conservatively as an "unusual force event"); slow uncommanded drift is logged; commanded-cycle anomalies feed the alignment diagnostics.

### 8. Fleet learning

With consent, anonymized feature histories aggregated by climate zone and door material produce priors for seasonal friction behavior, shortening commissioning for new installs. No raw waveforms or household identifiers participate.

## Claims

1. A diagnostic system for a motorized deadbolt lock, comprising: a current sensor arranged to measure current drawn by the lock's bolt actuator motor during lock and unlock cycles; a position sensor arranged to indicate bolt position along its travel; a processor that records, for each cycle, a current-versus-position trace; a baseline store holding per-door baseline statistics of the trace learned during a commissioning period; and an analyzer that decomposes drift of the trace away from the baseline into a reversible seasonal component correlated with outdoor temperature and humidity and a monotonic progressive component, and that issues a predictive lockout warning when the progressive component extrapolated toward the motor's stall margin crosses a warning threshold.
2. The system of claim 1, wherein the analyzer computes a position-resolved friction map of average current versus bolt position over a rolling window of cycles, and localizes a drag source along the bolt stroke from the map's shape, distinguishing at least a late-travel spike indicative of strike-plate lip interference, a uniform elevation indicative of bore friction or mechanism wear, and an early-travel hump indicative of mechanism-side binding.
3. The system of claim 1, wherein the seasonal component is modeled as a function of trailing outdoor relative humidity and temperature, the progressive component is constrained to be monotonic via isotonic regression, and warnings at the action level are driven by the progressive residual after removal of the seasonal component, suppressing false alarms from annual humidity-driven friction cycles.
4. The system of claim 2, further comprising a guidance generator that converts the friction map into directional strike-plate adjustment guidance specifying an adjustment direction and magnitude, derived from the position of a late-travel current spike within the stroke and from throw/retract current asymmetry.
5. The system of claim 1, further comprising an adaptive drive controller that raises the actuator torque budget within motor thermal limits in proportion to measured friction, applying increased torque selectively through high-drag regions of the stroke identified in the friction map, rather than repeating identical failed attempts.
6. The system of claim 1, further comprising a battery estimator that tracks per-cycle electrical energy and battery voltage sag under load, and that attributes rising energy consumption to mechanical load versus battery aging using voltage-normalized current features, producing a remaining-cycle estimate with an attributed cause.
7. The system of claim 1, further comprising a forced-entry classifier that detects uncommanded torque transients on the bolt, distinguishes impact-like rapid transients from gradual misalignment drift by timescale and by absence of a motor command, and reports suspected forced-entry events separately from alignment diagnostics.
8. A method for diagnosing deadbolt lock alignment, comprising: measuring actuator motor current and bolt position during each lock and unlock cycle of a motorized deadbolt; extracting per-cycle features including travel RMS current, cycle energy, travel time, and throw/retract asymmetry; learning a per-door baseline of the features during a commissioning period; fitting a two-component model to feature drift comprising a reversible seasonal component as a function of outdoor humidity and temperature and a monotonic progressive component; and issuing a predictive lockout warning with an estimated lead time when the progressive component approaches the actuator's stall margin.
9. The method of claim 8, further comprising an installer mode that presents a real-time friction readout during manual strike-plate adjustment, detects a step-change improvement in the friction map attributable to the adjustment, and re-baselines the per-door baseline from post-adjustment cycles.
10. The system of claim 1, further comprising a fleet-learning module that aggregates anonymized per-door friction histories by climate zone and door material to produce priors for seasonal friction behavior, the priors being applied to shorten the commissioning period of newly installed locks.
11. The system of claim 1, wherein all current features are normalized by contemporaneous battery voltage before baseline comparison, separating mechanical-load drift from battery-aging drift.
12. The system of claim 4, wherein the guidance generator renders a diagram of the strike plate with an arrow indicating the adjustment direction and a numeric adjustment magnitude, the magnitude being bounded by the mechanical adjustment range of a standard strike plate.

## Implementation Notes

This disclosure describes a proposed design; no prototype has been built and no performance figures have been measured. The primary target is retrofit battery-powered smart deadbolts, which already contain a microcontroller, a radio, and usually a position sensor; the incremental hardware is a shunt plus current-sense amplifier at well under one US dollar in volume. Limitations: locks without a true position sensor get coarser time-normalized maps (the scalar seasonal/progressive decomposition still works); manual key operation contributes no data; mortise and multi-point locks need a locksmith regardless and the system should say so; renters get a documented diagnosis for their landlord plus adaptive drive as a stopgap. The counterargument is that binary jam detection plus "call a locksmith" is cheaper; it covers the same event but not the same cost, since a predicted misalignment fixed with a screwdriver in daylight differs from a 2 a.m. lockout, a $150 to $300 locksmith bill, or a door left effectively unlocked by a silently failed auto-lock. Privacy: raw waveforms never leave the lock; only per-cycle features are transmitted, and fleet aggregation uses climate zone and door material with no household identifiers.

## Prior Art References

1. August Smart Lock Pro with DoorSense: reports open/closed/jammed lock states; frame sensor detects door-ajar condition. SlashGear <https://www.slashgear.com/august-smart-lock-pro-doorbell-cam-pro-doorsense-18495354/>
2. Schlage connected deadbolts expose lock status Locked / Unlocked / Jammed (documented in the Unfolded Circle integration). GitHub <https://github.com/mase1981/uc-intg-schlage>
3. WO2013168114A1, "A lock": motor current detection with fixed threshold; reverse-and-retry on clash before registering a fault. Google Patents <https://patents.google.com/patent/WO2013168114A1/en>
4. US10233672B2, "Lock devices, systems and methods": monitoring motor current to determine a stall condition. Google Patents <https://patents.google.com/patent/US10233672B2/en>
5. US10140828B2, "Intelligent door lock system with camera and motion detector": current sensor estimating friction experienced by the door/lock, friction information reported to the user, provision for current adjustment. Google Patents <https://patents.google.com/patent/US10140828B2/en>
6. US20200372738A1, "Smart lock system": stall detection via accelerometer/knob position monitor; motor disengaged to prevent overcurrent damage. Google Patents <https://patents.google.com/patent/US20200372738A1/en>
7. Edens Structural Solutions, "Why Is My Door Sticking, and What Does the Crack Beside It Mean?": seasonal versus foundation-driven sticking and the reversibility test. <https://edensstructural.com/why-is-my-door-sticking-and-what-does-the-crack-beside-it-mean/>
8. USDA Forest Products Laboratory, *Wood Handbook* (FPL-GTR-118): dimensional change of wood with moisture content; tangential shrinkage approximately twice radial. <https://www.fpl.fs.usda.gov/documnts/fplgtr/fplgtr118.pdf>
9. Purdue Extension FNR-163, "EMC's by Region": equilibrium moisture content by region and season; small MC changes produce significant dimensional change. <https://extension.purdue.edu/extmedia/fnr/fnr-163.pdf>
10. Angi, "How To Fix A Door That Sticks": seasonal sticking pattern and homeowner remediation. <https://www.angi.com/articles/how-to-fix-stuck-door.htm?entry_point_id=32949645>
11. Homes & Gardens, "How to fix a door that sticks": strike plate adjustment as the standard remedy. <https://www.homesandgardens.com/life-design/how-to-fix-a-door-that-sticks>
12. Yale Assure Lever Troubleshooting Guide: jam alarms on failed electronic operation. Manuals.Plus <https://manuals.plus/m/15e22ee769dac9ee7578c16f0ad405d5aad2f6bd323289f12d74ea6ef7691b3a.pdf>
