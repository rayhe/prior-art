# PA-2026-170: Oven Thermal Envelope Degradation Detection via Preheat-Time and Duty-Cycle Trending

**Title:** System and Method for Detecting Thermal Envelope Degradation in Residential Ovens Using Preheat-Time and Duty-Cycle Trending

**Filing:** LITF-PA-2026-170
**Published:** September 13, 2026
**Domain:** Appliances / Home Maintenance
**Full Disclosure:** [liveinthefuture.org/priorart/oven-thermal-envelope-degradation.html](https://liveinthefuture.org/priorart/oven-thermal-envelope-degradation.html)
**License:** [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/) — Public Domain

> Prior Art Notice: This document is published as defensive prior art under
> [35 U.S.C. Sec. 102(a)(1)](https://www.law.cornell.edu/uscode/text/35/102).
> The inventions described herein are dedicated to the public domain as of the
> publication date above.

---

## Abstract

Disclosed is a system and method for detecting degradation of the thermal envelope of a residential oven, comprising the door gasket, cavity insulation, and door closure geometry, by trending two signals the oven already produces across cooking cycles: (1) normalized preheat duration to a fixed temperature setpoint, and (2) heating-element or burner duty cycle during temperature maintenance. A sound thermal envelope loses heat slowly; a degraded envelope loses heat quickly. Increased heat loss therefore appears directly as a rising maintain-phase duty cycle at fixed setpoint and as lengthening preheat times. The system records per-cycle preheat duration and maintain-phase duty cycle, normalizes for setpoint, cooking mode, and starting cavity temperature, excludes cycles containing door openings, and establishes a per-oven baseline during an initial learning period. A thermal loss index computed from the drift of these metrics is compared against advisory thresholds, and crossing a threshold produces a maintenance advisory identifying the probable cause, such as door gasket replacement, insulation inspection, or hinge adjustment, before the degradation causes chronic energy waste and uneven cooking. The drift signature is further classified to distinguish envelope loss from heating-element degradation and temperature-sensor drift. Embodiments include integrated control-board logging, cloud analytics for connected ovens, and a non-invasive retrofit using a circuit-level energy monitor that infers preheat and maintain phases from the power-draw envelope alone, requiring no modification to the oven.

## Technical Field

This invention relates to residential appliance diagnostics, specifically to detection of thermal envelope degradation in conventional ovens through longitudinal analysis of preheat-time and duty-cycle signals, classification of degradation signatures, and predictive maintenance advisories for oven door gaskets, insulation, and closure hardware.

## Background

Residential conventional ovens are major household cooking appliances regulated as "cooking products" under 10 CFR 430.2, defined as compartments intended for cooking or heating food by gas flame or electric resistance heating ([DOE test procedure final rule for conventional ovens](https://www.energy.gov/sites/prod/files/2015/06/f23/conventional_ovens_tp_finalrule.pdf)). Every cooking cycle begins with a preheat phase in which the heating element or burner runs continuously until the cavity temperature sensor reaches the setpoint, followed by a maintain phase in which the heat source cycles on and off to hold temperature.

Manufacturers publish normal preheat expectations: Whirlpool states an oven will normally preheat to 350°F in 12 to 15 minutes, 20 minutes (±5) for hidden-bake-element models, and lists the factors that lengthen preheat, including unused racks left inside, door openings during preheat, 208V versus 240V supply, and cold room temperature ([Whirlpool product help: long pre-heat times](https://producthelp.whirlpool.com/Cooking/Wall_Ovens_and_Ranges/Product_Info/Oven_Product_Assistance/Long_Pre-Heat_Times_for_Oven)). No manufacturer suggests recording preheat times across cycles.

The oven door gasket is the most failure-prone element of the thermal envelope. It is typically a braided fiberglass rope, friction-fit or clip-mounted in a channel around the cavity opening. When the braid becomes stiff, flattened, cracked, or torn, heat escapes past the door; repair guidance describes the consequences as increased energy cost and improperly cooked food ([Weekand: replacing a braided oven door seal](https://www.weekand.com/home-garden/article/replace-braided-oven-door-seal-18014531.php)). A worn gasket forces the oven to work harder to reach setpoint and lengthens cooking times ([All Things Home: replacing your oven door gasket](https://stage.allthingshome.ca/home-improvement/articles/article/homeowner-helpers-replacing-your-oven-door-gasket-is-an-easy-diy-home-improvement-home-maintenance-reminders/)). The standard field test is qualitative: pass a hand near the closed door while the oven runs and feel for escaping heat ([HowStuffWorks: replacing a door gasket](https://home.howstuffworks.com/how-to-repair-an-oven2.htm)). Gasket replacement is a low-cost DIY repair, typically a $15 to $30 part installed in minutes, yet most households never inspect the gasket until baking results visibly deteriorate.

Repeated pyrolytic self-clean cycles impose severe thermal cycling on the gasket and accelerate its aging, as do the thousands of door open-close cycles and the gradual settling of fiberglass cavity insulation and hinge-spring relaxation over a 10 to 15 year appliance life. The degradation is slow, monotonic, and invisible: the oven still reaches setpoint, the preheat chime still sounds, and the user compensates unconsciously by adding minutes to recipes.

The closest art addresses adjacent problems without teaching the disclosed method. [EP3155491B1](https://patents.google.com/patent/EP3155491B1/en) (BSH) describes preventive maintenance of kitchen appliances through maintenance data recorded by a service operator into the appliance memory and transmitted to a server; it logs service events, not thermal signals, and performs no trending. [US20160040892A1](https://patents.google.com/patent/US20160040892A1/en) (Haier) discloses operating methods to minimize preheat time in hidden-bake ovens, an operating strategy rather than a diagnostic. Manufacturer troubleshooting pages list causes of long preheat as discrete faults to check once, never as a signal to trend.

Non-intrusive load monitoring literature teaches that appliance faults produce altered consumption signatures, giving the example of a faulty refrigerator seal drawing attention through increased energy use ([GlobalSpec: smart meter breaks down energy use by appliance](https://insights.globalspec.com/article/3735/smart-meter-breaks-down-energy-use-by-appliance), summarizing Fraunhofer IMS NILM work), and HVAC fault-detection practice teaches trending operational parameters such as equipment runtime. Neither body of art teaches oven-specific preheat-duration trending, the dual-signal combination of normalized preheat duration with maintain-phase duty cycle, or signature classification via early-preheat heating rate that distinguishes envelope loss from heating-source degradation and sensor drift.

The gap in the art is a system that records preheat duration and maintain-phase duty cycle across oven cycles, normalizes for the known confounds, establishes each oven's individual baseline, detects the slow drift characteristic of envelope degradation, classifies that drift by signature, and converts the finding into a timely, specific maintenance advisory. No reference teaches this combination for oven thermal envelope diagnostics.

## Detailed Description

### 1. Thermal Model

During the maintain phase at fixed setpoint, steady-state energy balance requires that average heat input equal average heat loss. For an electric oven with element power P (e.g., 2.5 kW bake element) and duty cycle D (fraction of time the element is energized), Q_loss = P × D. Duty cycle is therefore a direct, thermal-mass-independent measure of envelope heat loss at a given temperature difference. During preheat, the time to reach setpoint is approximately t_preheat = C × ΔT / (P − Q_loss_avg), where C is the cavity thermal mass and Q_loss_avg is the average loss rate during the ramp. Envelope degradation increases Q_loss, which lengthens preheat and raises maintain-phase duty cycle simultaneously. This dual-signature response is the diagnostic foundation.

### 2. Signal Acquisition Embodiments

**Integrated embodiment.** The oven control board logs, for each cooking cycle: the commanded setpoint, cooking mode (bake, convection bake, roast), cavity temperature sensor readings at 1 Hz or faster, heating-element relay or gas valve state, and door-open events from the existing door switch. Preheat duration is measured from cycle start (door confirmed closed, heat commanded) to first thermostat cutout at setpoint. Maintain-phase duty cycle is computed over the maintain phase, defined as the interval from preheat completion to cycle end capped at 60 minutes, as energized time divided by interval duration.

**Connected embodiment.** A WiFi-connected oven transmits the logged cycle records to a cloud service that performs normalization, baselining, and trending across the installed fleet, enabling model-specific priors for expected preheat and duty values.

**Retrofit embodiment.** A non-invasive circuit-level energy monitor (current-transformer clamps in the electrical panel, sampling at 1 Hz or faster) records the oven circuit's power draw. The preheat phase is identified as the initial continuous-draw interval; the maintain phase is identified by the onset of regular on/off cycling in the power envelope. Preheat duration is the length of the continuous-draw interval; duty cycle is the energized fraction during the subsequent cycling interval. This embodiment requires no modification to the oven, no knowledge of the setpoint, and no network connection to the appliance. It applies to electric ovens and ranges on dedicated circuits. For gas ovens, the retrofit embodiment uses a surface-mounted thermocouple on the oven vent or flue to detect burner cycling by exhaust temperature oscillation.

### 3. Preheat-Time Metric and Normalization

Raw preheat duration is confounded by setpoint, starting temperature, cooking mode, and load. The system normalizes as follows: cycles are bucketed by (setpoint band, cooking mode); within each bucket, preheat duration is corrected for starting cavity temperature using a linear correction fitted during the learning period (typical coefficient: 60 to 90 seconds per 25°F of starting-temperature deficit, learned per oven rather than assumed); cycles in which the door opens during preheat are excluded; cycles following a self-clean cycle within 24 hours are excluded (residual heat distorts the ramp); and rapid-preheat or Sabbath modes are bucketed separately. The normalized preheat metric for a cycle is the corrected duration in its bucket. In the retrofit embodiment, where setpoint is unknown, normalization uses the distribution of observed continuous-draw intervals: the system tracks the median interval for the household's dominant usage pattern and flags drift in that median, which is robust because households repeat similar setpoints. Door-open exclusion during preheat applies to embodiments with a door switch; in the retrofit embodiment a door opened during preheat is indistinguishable from a long preheat in the power envelope, so such cycles are instead caught by the outlier rejection described in section 5 (continuous-draw intervals beyond 3 median absolute deviations, or above the 95th percentile of the household distribution, are excluded from trending).

### 4. Maintain-Phase Duty-Cycle Metric

Maintain-phase duty cycle is computed over the maintain phase defined in section 2, with door-open intervals excised (door openings appear as extended energized periods of recovery firing and would bias the metric upward if included; in the retrofit embodiment they appear as anomalously long on-gaps in the power envelope and are excised by the same rule). Food load does not bias the metric in steady state: added thermal mass slows the cycle period but steady-state duty still equals Q_loss / P. Cold-load warmup transients elevate duty in the early maintain window and are absorbed by per-oven baselining and mode bucketing. The metric is bucketed by setpoint band and mode identically to preheat. Duty cycle is the more robust of the two metrics because it is independent of cavity thermal mass and starting temperature.

### 5. Baseline Establishment and Thermal Loss Index

During an initial learning period (default: the first 60 days of operation or the first 40 qualifying cycles, whichever comes first), the system computes per-bucket medians of normalized preheat duration and duty cycle, with outlier rejection (cycles beyond 3 median absolute deviations are excluded from the baseline but retained for diagnostics). These medians constitute the oven's individual baseline, which absorbs installation-specific factors including 208V versus 240V supply, ambient kitchen temperature, and cavity size.

Ongoing cycles update exponentially weighted moving averages of both metrics (time constant approximately 30 days). The thermal loss index (TLI) is defined as a weighted combination of the fractional increases: TLI = 0.5 × (D − D0)/D0 + 0.5 × (t − t0)/t0, where D0 and t0 are baseline duty cycle and preheat duration. Advisory thresholds: TLI exceeding 0.20 (a 20% combined degradation) sustained over 14 days generates an informational advisory; TLI exceeding 0.35 generates a maintenance recommendation. Hysteresis (advisory clears only when TLI falls below 0.10) prevents flapping after gasket replacement.

### 6. Degradation Signature Classification

Not all drift indicates envelope loss, so the system classifies the observed signature using the early-preheat heating rate in addition to the two primary metrics. The early-preheat heating rate r = dT/dt measured over the first 3 minutes of preheat (before losses dominate) reflects delivered heating power:

- **Envelope-loss signature:** normalized preheat duration increased, duty cycle increased, early heating rate normal. Heat delivery is intact but heat retention has degraded. Probable causes, in order: door gasket wear, hinge sag reducing door compression, insulation settling. Recommended action: inspect and replace the door gasket; check door closure force.
- **Heating-source signature:** normalized preheat duration increased, duty cycle increased, early heating rate decreased. Delivered power has fallen. Probable causes: element resistance increase with age, partial element failure (in dual-element preheat), low supply voltage, weak gas burner. Recommended action: element resistance check or burner inspection, not gasket replacement.
- **Sensor-drift signature:** normalized preheat duration decreased, duty cycle decreased, optionally corroborated by a connected-app prompt asking the user whether recent bakes required extended times or came out undercooked. The temperature sensor reads high, so the control terminates preheat early and maintains a lower true temperature. The objective signature alone (shortened preheat plus reduced duty at unchanged usage) is sufficient to raise a provisional sensor-drift flag. Recommended action: temperature calibration check with an independent oven thermometer; sensor replacement if offset exceeds 25°F.
- **Step-change signature:** abrupt increase in both metrics within a single week, correlated with a self-clean cycle or a physical impact to the door. Indicates gasket displacement, torn gasket section, or door misalignment. Recommended action: immediate visual inspection of the gasket channel.

An optional array of door-frame temperature sensors (adhesive thermocouples or IR sensors mounted at intervals on the cabinet face around the door, in the integrated or retrofit-thermocouple embodiment) localizes gasket leaks: the sensor reporting the greatest maintain-phase elevation relative to baseline indicates the leak position along the door perimeter.

### 7. Advisory Content and Energy Quantification

Each advisory reports: the current TLI and its trend, the classified signature, the estimated excess energy per cycle (computed as P × (D − D0) × maintain duration, e.g., a 15-percentage-point duty increase on a 2.5 kW element over a 1-hour maintain phase wastes approximately 0.38 kWh per cycle), the projected annual waste at the household's observed usage rate, and the specific recommended part or service. For the envelope-loss signature, the advisory identifies the door gasket as a user-replaceable part, describes the qualitative field test (feeling for escaping heat around the closed door), and notes that replacement typically takes minutes with a friction-fit or clip-mounted gasket.

Fleet aggregation (connected embodiment) learns model-specific degradation curves, enabling the advisory to state, for example, that the observed drift matches the 75th percentile of gasket wear for the oven's model at its age, and allowing property managers to prioritize gasket replacement across rental units by measured loss rather than by schedule.

### 8. Figures Description

- **Figure 1:** System block diagram showing the three acquisition embodiments (integrated control-board logging, connected cloud analytics, retrofit circuit-level energy monitor) feeding the normalization, baseline, trending, and classification modules and producing the maintenance advisory.
- **Figure 2:** Representative power-draw envelope from the retrofit embodiment: continuous-draw preheat interval, transition to cycled maintain phase, excised door-open gap, and computed duty cycle.
- **Figure 3:** Example longitudinal trend over 24 months showing baseline period, monotonic rise in normalized preheat duration and duty cycle, advisory threshold crossings, and metric recovery following gasket replacement.
- **Figure 4:** Signature classification decision tree mapping combinations of preheat drift, duty drift, and early-preheat heating rate to envelope-loss, heating-source, sensor-drift, and step-change diagnoses.

## Claims

1. A system for detecting thermal envelope degradation in a residential oven, comprising: a signal acquisition module that records, for each of a plurality of cooking cycles, a preheat duration from cycle start to first thermostat cutout at setpoint and a maintain-phase duty cycle of the heating element or burner; a normalization module that corrects the preheat duration for setpoint, cooking mode, and starting cavity temperature and excludes cycles containing door openings; a baseline module that establishes per-oven baseline values of normalized preheat duration and duty cycle during an initial learning period; and a trending module that computes a thermal loss index from the drift of the metrics relative to baseline and generates a maintenance advisory when the index crosses a threshold.

2. The system of claim 1, wherein the signal acquisition module is a non-invasive circuit-level energy monitor that infers the preheat phase as an initial continuous power-draw interval and the maintain phase as a subsequent cycled power-draw interval from the power envelope alone, without modification to the oven and without knowledge of the temperature setpoint.

3. The system of claim 1, wherein the trending module classifies a drift signature as envelope loss when normalized preheat duration and duty cycle increase while early-preheat heating rate remains normal, and issues a door gasket inspection or replacement advisory in response.

4. The system of claim 1, wherein the trending module classifies a drift signature as heating-source degradation when normalized preheat duration and duty cycle increase while early-preheat heating rate decreases, and issues a heating-element or burner inspection advisory rather than a gasket advisory.

5. The system of claim 1, wherein the trending module classifies a drift signature as temperature-sensor drift when normalized preheat duration and duty cycle decrease relative to baseline, and issues a temperature calibration advisory.

6. The system of claim 1, further comprising an array of door-frame temperature sensors mounted at intervals around the oven door, wherein the sensor reporting the greatest maintain-phase temperature elevation relative to baseline localizes a gasket leak position along the door perimeter.

7. The system of claim 1, wherein the maintenance advisory quantifies excess energy per cycle as heating power multiplied by the duty-cycle increase multiplied by maintain duration, and projects annual energy waste at the observed usage rate.

8. A method for detecting thermal envelope degradation in a residential oven, comprising: recording preheat duration and maintain-phase heating duty cycle across a plurality of cooking cycles; normalizing the preheat duration for setpoint, cooking mode, and starting cavity temperature and excluding cycles with door openings during preheat; establishing per-oven baseline values during an initial learning period; computing a thermal loss index from metric drift relative to baseline; classifying the drift signature using early-preheat heating rate to distinguish envelope loss from heating-source degradation and sensor drift; and generating a maintenance advisory identifying the probable cause when the thermal loss index crosses a threshold.

9. The method of claim 8, further comprising excluding cycles occurring within 24 hours after a pyrolytic self-clean cycle from the trending computation.

10. The system of claim 1, further comprising fleet aggregation across a plurality of connected ovens that learns model-specific thermal degradation curves and prioritizes maintenance advisories by measured thermal loss.

## Prior Art References

1. [DOE: Test Procedures for Conventional Ovens, Final Rule](https://www.energy.gov/sites/prod/files/2015/06/f23/conventional_ovens_tp_finalrule.pdf): conventional ovens defined under 10 CFR 430.2

2. [Whirlpool Product Help: Long Pre-Heat Times for Oven](https://producthelp.whirlpool.com/Cooking/Wall_Ovens_and_Ranges/Product_Info/Oven_Product_Assistance/Long_Pre-Heat_Times_for_Oven): normal preheat 12-15 min to 350°F; 20 min (±5) hidden bake; listed confounds

3. [Weekand: How to Replace a Braided Oven Door Seal](https://www.weekand.com/home-garden/article/replace-braided-oven-door-seal-18014531.php): stiff or worn braid lets heat escape; higher energy cost, improper cooking

4. [All Things Home: Replacing Your Oven Door Gasket](https://allthingshome.ca/home-improvement/articles/article/homeowner-helpers-replacing-your-oven-door-gasket-is-an-easy-diy-home-improvement-home-maintenance-reminders/): worn gasket forces harder heating, longer cook times; cracked, torn, or frayed gaskets need replacement

5. [HowStuffWorks: Replacing a Door Gasket](https://home.howstuffworks.com/how-to-repair-an-oven2.htm): hand test for escaping heat; friction-fit channel replacement procedure

6. [GlobalSpec: Smart meter breaks down energy use by appliance](https://insights.globalspec.com/article/3735/smart-meter-breaks-down-energy-use-by-appliance): NILM literature; appliance faults produce altered consumption signatures, e.g., a faulty refrigerator seal identified through increased energy use (Fraunhofer IMS)

6. [EP3155491B1](https://patents.google.com/patent/EP3155491B1/en): BSH: preventive maintenance of kitchen appliances via recorded service data (no signal trending)

7. [US20160040892A1](https://patents.google.com/patent/US20160040892A1/en): Haier: oven operating method to minimize preheat time (operating strategy, not diagnostics)

8. [35 U.S.C. § 102](https://www.law.cornell.edu/uscode/text/35/102): conditions for patentability; novelty and prior art

## Implementation Notes

**Per-oven baselining absorbs installation confounds.** Supply voltage (208V versus 240V), ambient kitchen temperature, cavity size, and hidden versus exposed bake elements all shift absolute preheat times substantially; because every comparison is against the oven's own learned baseline, these factors require no explicit modeling.

**Door-open exclusion is essential.** A single mid-cook door opening for basting can add several minutes of recovery firing; the door switch (integrated) or long off-gap detection (retrofit) must gate the metrics.

**Mode bucketing.** Convection modes preheat and maintain differently from conventional bake due to fan-forced heat transfer; never mix modes in a bucket. Rapid-preheat modes that fire broil and bake elements together form their own bucket.

**Gas ovens.** The integrated embodiment applies directly via burner valve state; the retrofit embodiment substitutes a vent thermocouple for the CT clamps, detecting burner cycling from exhaust temperature oscillation. Duty cycle semantics are identical.

**Privacy.** All trending computation can run on-device or on the local energy monitor; only anonymized metric aggregates need leave the home for fleet learning, and fleet participation is opt-in.

**Limitations.** The method detects gradual degradation, not sudden faults; a failed element relay or blown thermal fuse presents as a discrete fault outside this method's scope. Households with highly irregular usage (e.g., frequent self-clean cycles, commercial-style high-temperature roasting) will have fewer qualifying cycles and wider confidence intervals, which the system reports alongside the index.

**Complementary to existing art.** This method does not replace manufacturer troubleshooting for discrete faults; it addresses the slow degradation regime that one-time troubleshooting cannot see.
