# PA-2026-214: Espresso Machine Scale and Pump Diagnostics from Pressure Ripple, Flow Decline, and Heating Duty Fusion

**Title:** System and Method for Estimating Espresso Machine Limescale Accumulation and Distinguishing Scale from Pump Wear Using Pump Pressure Ripple, Shot Flow Decline, and Heating Duty Fusion

**Filing:** LITF-PA-2026-214
**Published:** October 25, 2026
**Domain:** Appliance Health / Sensor Fusion / Predictive Maintenance
**Full Disclosure:** [liveinthefuture.org/priorart/espresso-scale-pump-diagnostics.html](https://liveinthefuture.org/priorart/espresso-scale-pump-diagnostics.html)
**License:** [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/) — Public Domain

> Prior Art Notice: This document is a defensive technical disclosure published
> to constitute prior art under [35 U.S.C. Sec. 102(a)(1)](https://www.law.cornell.edu/uscode/text/35/102),
> dedicated to the public domain as of the publication date above.

---

## Abstract

Disclosed is a system and method for estimating limescale buildup inside a home espresso machine without opening the boiler, and for telling scale apart from the other faults that make shots run slow. Most machines today descale on a calendar or a shot counter, which knows nothing about the water actually used. This system instead watches three signals the machine already produces or that a smart plug and a cup scale can measure: the ripple in pump pressure and current as the pump works against a narrowing water path, the shape of the shot flow curve from first drip to target weight, and the duty cycle and recovery time of the boiler heater as scale insulates it. Each signal is compared with a personal baseline recorded just after a descale, using the same dose, grind setting, basket, and coffee where possible. A fusion score combines flow decline, heating duty rise, recovery time rise, and ripple change into one scale index, while a differential classifier separates scale from pump wear, basket or screen clogging, grinder drift, channeling, and stale coffee. When the index crosses a staged threshold, the system says descale soon or descale now, and after a descale it checks whether the signals returned toward baseline, which retrospectively confirms or refutes the diagnosis. Nobody needs to taste a bad shot to learn the machine needed care weeks earlier.

## Technical Field

This disclosure relates to appliance health monitoring and predictive maintenance for espresso machines, specifically to non-invasive estimation of internal limescale from electrical, hydraulic, and thermal signals. It sits at the intersection of pump diagnostics, flow metrology, and heat transfer: a narrowing water path raises hydraulic resistance, a scaled heating surface raises thermal resistance, and a wearing pump loses pressure head. Because those three effects overlap in the single symptom of a slow shot, the diagnostic value here comes from the pattern across channels rather than from any one channel alone.

## Background

Water hardness is dissolved calcium and magnesium, reported in milligrams per liter as calcium carbonate. The United States Geological Survey classifies water below 60 mg/L as soft, 61 to 120 mg/L as moderately hard, 121 to 180 mg/L as hard, and above 180 mg/L as very hard. When hard water is heated, calcium carbonate comes out of solution and deposits as scale on heating elements, boiler walls, and narrow passages. Scale is a poor heat conductor, with thermal conductivity on the order of 1 to 2 watts per meter-kelvin, against roughly 16 for stainless steel and roughly 400 for copper. Even a sub-millimeter layer therefore adds meaningful thermal resistance at the exact surface where heat must cross into the water, and the same deposit narrows tubes, jets, and valve passages where flow resistance rises steeply as diameter falls.

Espresso extraction makes the problem acute. A conventional target is about 9 bars of pressure at the group head, with brew water near 90 to 96 degrees Celsius. Small changes in flow resistance change shot time and extraction, and small changes in heater performance change temperature stability. Industry guidance summarized for home users puts total dissolved solids near 75 to 250 parts per million, calcium hardness near 50 to 175 parts per million as calcium carbonate, alkalinity near 40 to 75 parts per million, and pH near 6.5 to 7.5. Those ranges balance flavor against scale risk, but they describe the water going in, not the scale already inside a given machine. A household on hard water and a household on filtered soft water can own identical machines and need descaling on entirely different schedules, which is why a calendar interval is a weak proxy.

What owners actually use today is weaker:

- **Calendar or shot-count reminders:** Simple and blind. They fire too early for soft-water users, who descale unnecessarily, and too late for hard-water users, who brew through a scaled boiler for weeks.
- **Taste and shot time:** A sour or thin shot, or a shot that suddenly runs long, is a late signal. By the time taste changes clearly, flow and temperature have already drifted, and taste confounds scale with bean age, grind setting, dose, and tamping.
- **Visible scale:** White crust on a kettle or a steam wand tip is easy to see, but the boiler interior, thermoblock channels, and solenoid passages that matter most are hidden. External crust proves hard water, not the state of the internal path.
- **Single-sensor add-ons:** A flow meter alone cannot separate a scaled boiler from a clogged basket. A temperature probe alone cannot separate a scaled heater from a failing thermostat. A pressure gauge alone cannot separate a scaled path from a tired pump. Each single channel answers one question while the owner has several.

The gap in the art is a detector that: (a) uses signals already available from the pump, heater, and shot, with at most a smart plug and a cup scale added, (b) compares each shot with the same machine's own post-descale baseline instead of a factory threshold, (c) separates scale from pump wear and from the daily confounders of coffee preparation, and (d) closes the loop by verifying recovery after descaling, so a wrong call is visible instead of silently repeated.

## Detailed Description

### 1. System Overview and Sensors

The preferred embodiment needs no modification to the machine. A plug-level power meter samples current at 1 Hz or faster and records heater on and off transitions, pump current draw, and total energy per shot. A cup scale with 0.1 gram resolution, or an internal flow meter where the machine already has one, records beverage weight against time. The user enters dose in grams, grind setting, basket identity, coffee identity, and water source once, then updates only what changes. An optional embodiment adds a clamp-on current sensor at the pump, a pressure transducer at the group head for machines with a gauge port, and a boiler temperature probe for machines that expose one. All embodiments share the same fusion logic; extra sensors tighten the estimates but do not change the claims.

Vibratory pumps, the common home-machine type, produce a characteristic pressure and current ripple at the mains-related pumping frequency and its harmonics. Rotary pumps produce a smoother trace with a different ripple spectrum. The system records the pump type at setup and learns its ripple spectrum during calibration, rather than assuming one pump behaves like the other. That distinction matters, because a ripple change that signals wear in one pump type may be normal operating variation in the other.

### 2. Baseline Calibration After Descaling

Calibration runs over the first 20 reference shots after a descale, or after first setup if the machine is new. Reference shots use a fixed recipe: a stated dose, grind setting, basket, and target beverage weight, for example 18 grams in and 36 grams out. From those shots the system stores reference values for time to first drip, flow rate in grams per second at 5-second intervals, total time to target weight, pump current ripple amplitude in the pump band, heater duty cycle during the shot and during a 10-minute recovery window, and time for the boiler to return to its ready temperature after a shot. Medians, not means, define the baseline, so one channeled shot does not move the reference.

Calibration also records the water story. The user enters measured hardness in mg/L as calcium carbonate from a test strip, a municipal report, or a filter vendor's specification, plus the filter type and install date if a filter is used. Where hardness is unknown, the system starts with a regional default and widens its uncertainty bands until a measurement arrives. Hardness does not enter the scale index directly; it sets the expected deposition rate and the prior probability that a given drift is scale rather than another fault.

### 3. The Scale Index

For each new shot that matches the reference recipe within tolerance, the system computes four normalized drift terms. Flow decline F is the fractional fall in mean flow rate over the middle of the shot, where the coffee puck dominates less than at first drip and less than at the tail. Heating duty rise H is the fractional rise in heater duty cycle needed to hold brew temperature during the shot. Recovery rise R is the fractional rise in time to return to ready temperature after the shot. Ripple change P is the fractional change in pump-band ripple amplitude relative to baseline, signed so that a rise in ripple counts toward the index.

The scale index S combines them as S = 0.35F + 0.30H + 0.20R + 0.15P, with each term clipped at 1.0 so no single channel can trigger a descale call alone. A value near 0 means the machine behaves as it did just after descaling. A value near 0.25 is the watch level, a value near 0.45 is the descale-soon level, and a value near 0.65 is the descale-now level. Those thresholds are design choices, stated here so a builder can implement and then calibrate them, not measured performance claims. A 7-shot rolling median smooths the index, because single shots vary with tamping and bean age, and a shot is excluded from the index when its dose, grind, or basket differs from the reference recipe, rather than being corrected with a guessed adjustment.

Two physical arguments support the weighting. First, flow resistance in a tube rises with the inverse fourth power of diameter in laminar flow, so a thin scale layer in a narrow jet or passage produces a disproportionate flow effect, while heater duty responds more gradually as thermal resistance accumulates. Second, scale insulates the heater while also narrowing the path, so genuine scale moves flow and heating together. A fault that moves only one channel is, on that evidence, probably not scale, which is the core of the differential diagnosis in section 4.

### 4. Differential Diagnosis

The classifier assigns each drift pattern to one of the following classes, using the channel pattern plus the response to simple interventions:

- **Scale:** Flow down, heater duty up, recovery up, ripple up modestly. Pattern persists across baskets, coffees, and grind settings, and improves after descaling. This is the only class that moves heating and flow together and then reverses with descaling.
- **Pump wear:** Ripple up strongly, maximum pressure down, flow down at high resistance but near normal at low resistance. Heating duty and recovery unchanged. Pattern worsens with puck resistance and does not improve after descaling or basket cleaning.
- **Basket or shower screen clog:** Flow down, first drip late, heating unchanged, ripple up slightly. Pattern resolves when the basket or screen is swapped or soaked, without descaling.
- **Grinder drift or dose change:** Flow changes in the direction of finer or coarser grind, but heating duty and recovery stay at baseline, and the pattern tracks the grinder setting rather than calendar time. The system excludes non-reference shots from the index precisely to avoid scoring a new grind setting as scale.
- **Channeling:** Flow fast and erratic, often with a normal or short time to first drip, and a jagged flow curve. Channeling is a single-shot event, not a trend, and the rolling median rejects it.
- **Bean age or roast change:** Flow and taste shift after a new bag, but heating channels stay flat, and the pattern resets when the previous coffee returns or when the recipe is recalibrated for the new coffee.
- **Filter exhaustion:** Hardness reaching the machine rises as a filter saturates, which raises the expected deposition rate without immediately changing any measured channel. The system raises the scale prior and shortens the watch interval when the filter passes its rated volume, but it does not raise the index on filter age alone.
- **Thermostat or sensor fault:** Temperature swings large and abrupt, heater duty erratic rather than steadily rising, recovery times scattered. A steady trend points to scale; a step change points to a sensor or control fault.

Where the pattern is ambiguous between scale and pump wear, the system prescribes a discriminating test before recommending a repair: pull a water-only shot through an empty pressurized basket or a blind filter at a known resistance. Heating channels are irrelevant to that test, while flow and ripple at fixed resistance isolate the hydraulic path. If flow at fixed resistance has fallen and ripple has risen while heating drift persists on coffee shots, scale leads. If flow at fixed resistance is normal but coffee shots run slow, the coffee preparation side leads, and descaling is not the first action.

### 5. Deposition Model and Worked Example

Alongside the measured index, the system maintains a simple deposition estimate that connects water chemistry to expected scale mass. For throughput volume V in liters, hardness C in mg/L as calcium carbonate, and a precipitation fraction f for the fraction of hardness that actually deposits at boiler temperature, estimated scale mass is M = V times C times f, in milligrams. The precipitation fraction is the least certain input, so the system carries it as a range, 0.05 to 0.30, rather than a point value, and it learns the effective fraction for a given machine and water supply by comparing predicted mass with the observed index change between descales.

A worked example shows the scale of the numbers. Suppose water hardness is 150 mg/L, total throughput including brewing, flushing, and steaming is 100 liters between descales, and the precipitation fraction is 0.20. Estimated scale mass is 100 times 150 times 0.20, which is 3,000 milligrams, or 3 grams. Three grams spread over boiler walls, a thermoblock channel, and valve passages is a small mass with a large hydraulic effect, because the passages are narrow. The same arithmetic at 50 mg/L and the same throughput and fraction gives 1 gram, which is why soft-water machines can run far longer between descales, and why a shot counter alone misleads both households in opposite directions.

### 6. Staged Response and Descale Verification

At the watch level, the system reports the index trend and suggests checking the filter date and hardness input. At descale-soon, it recommends scheduling a descale within the next two weeks or the next 30 shots, whichever comes first, and it asks the user to run one reference-recipe shot before descaling so the pre-descale state is recorded. At descale-now, it recommends descaling before further regular use, while noting that a machine can still brew, so the recommendation is about taste, temperature stability, and long-term wear, not safety. After descaling, the system requires a short re-baseline of 5 reference shots and compares them with the old baseline. If flow, heating duty, and recovery return to within 10 percent of the old baseline, the diagnosis is marked confirmed. If they do not, the system re-runs the differential diagnosis and suggests the next test, basket and screen cleaning, pump inspection, or thermostat check, instead of repeating the descale call.

Privacy and data handling follow the same restraint as other on-device appliance diagnostics. Pump current, heater duty, and shot weight curves are small feature summaries. No audio is recorded, no video is needed, and raw high-rate waveforms can stay on the plug or phone. A user can delete the shot history, and deletion removes the index history with it, since the index is derived from that history.

### 7. Figures Description

- **Figure 1:** System overview showing an espresso machine, a plug power meter, and a cup scale, with arrows tracing pump current, heater duty, and beverage weight into a fusion module that outputs a scale index.
- **Figure 2:** Reference and drifted shot curves side by side: flow rate against time, heater duty against time, and pump ripple spectrum, each showing baseline and drifted traces.
- **Figure 3:** Decision flowchart from reference-recipe gating through drift computation, the scale index, the differential classifier, and the staged descale response.
- **Figure 4:** Differential diagnosis panel showing the channel pattern for scale, pump wear, basket clog, grinder drift, channeling, and thermostat fault, with one row per fault and one column per channel.
- **Figure 5:** Owner report mockup with the scale index trend, the deposition estimate range, the filter date, and the post-descale verification result.

## Claims

1. A system for estimating limescale accumulation in an espresso machine, comprising: a power meter measuring pump current and heater duty; a shot flow sensor measuring beverage weight against time; and a fusion module that compares flow, heating duty, recovery time, and pump ripple against a post-descale baseline and computes a scale index from the combined drift.
2. The system of claim 1, wherein the fusion module computes the scale index as a weighted sum of flow decline, heating duty rise, recovery time rise, and pump ripple change, with each term clipped so no single channel alone triggers a descale recommendation.
3. The system of claim 1, wherein the fusion module establishes the baseline from median values over reference shots using a fixed dose, grind setting, basket, and target beverage weight after a descale, and excludes non-reference shots from the index.
4. The system of claim 1, wherein the fusion module applies a rolling median to the scale index over multiple shots, so single-shot channeling and tamping variation do not trigger a descale recommendation.
5. The system of claim 1, further comprising a differential classifier that distinguishes scale from pump wear, basket or screen clogging, grinder drift, channeling, bean change, filter exhaustion, and thermostat fault by the pattern of drift across flow, heating, recovery, and ripple channels.
6. The system of claim 1, wherein the differential classifier prescribes a water-only shot at fixed resistance through an empty pressurized basket or blind filter to isolate hydraulic drift from coffee preparation drift before recommending descaling or repair.
7. The system of claim 1, further comprising a deposition estimator that computes estimated scale mass from throughput volume, water hardness in mg/L as calcium carbonate, and a precipitation fraction carried as a range, and that updates the effective precipitation fraction by comparing predictions with observed index change between descales.
8. The system of claim 1, wherein the fusion module adjusts the expected deposition rate and the prior probability of scale from an entered water hardness value and filter install date, without raising the scale index on filter age alone.
9. The system of claim 1, further comprising a staged response module that reports watch, descale-soon, and descale-now levels at increasing index thresholds, and that verifies a descale by comparing post-descale reference shots with the pre-descale baseline.
10. A method for estimating espresso machine limescale, comprising: recording pump current ripple, heater duty, recovery time, and shot flow curves for reference-recipe shots; computing normalized drift in each channel against a post-descale baseline; combining the drifts into a scale index; classifying the drift pattern against pump wear, clogging, grinder drift, and thermostat fault patterns; and recommending and then verifying a descale based on the index and the post-descale recovery of the measured channels.

## Prior Art References

1. United States Geological Survey, Hardness of Water: hardness as dissolved calcium and magnesium, measured as calcium carbonate, with the soft, moderately hard, hard, and very hard bands used in this disclosure. https://pubs.usgs.gov/wsp/0496/report.pdf
2. Water Hardness and Minerals Impact on Espresso Machines and Coffee: summary of industry water guidance, including total dissolved solids near 75 to 250 parts per million, calcium hardness near 50 to 175 parts per million as calcium carbonate, alkalinity near 40 to 75 parts per million, and pH near 6.5 to 7.5. https://bartovation.com/hardness-minerals-impact-espresso-machines/
3. Descale Espresso Machine: Step-by-Step Guide: background on scale insulating heating elements, reducing water temperature and flow, raising pump noise, and requiring regular descaling. https://outin.com/blogs/news/how-to-descale-espresso-machine
4. Descaling, 9Barista: manufacturer descaling guidance, including light descaling when visible buildup appears and deep descaling on a longer interval for hard-water areas. https://9barista.com/a/knowledge-base/instructions/descaling
5. LITF-PA-2026-162, Refrigerator Compressor Health Prognostics Using Ambient Smart-Speaker Microphone Sensing: prior LITF disclosure using acoustic and electrical signatures for appliance prognostics; this disclosure differs by targeting espresso scale through hydraulic, thermal, and ripple fusion rather than compressor acoustics alone. https://liveinthefuture.org/priorart/refrigerator-compressor-acoustic-prognostics.html
6. LITF-PA-2026-212, Water Heater Dip Tube Failure Drawdown Diagnostics: prior LITF disclosure using drawdown and thermal signals for water heater diagnosis; this disclosure differs by using shot flow curves and heater duty in an espresso machine, with a differential classifier against pump wear and coffee preparation confounders. https://liveinthefuture.org/priorart/water-heater-dip-tube-failure-drawdown-diagnostics.html
7. LITF-PA-2026-159, Gas Oven Hot-Surface Igniter Failure Prediction: prior LITF disclosure using electrical signatures for appliance wear prediction; this disclosure differs by fusing hydraulic flow decline with heating duty rise to estimate a deposit, not a component failure alone. https://liveinthefuture.org/priorart/gas-oven-igniter-wear-prediction.html
8. 35 U.S.C. Sec. 102: conditions for patentability, including the printed publication and public availability provisions under which this disclosure is published. https://www.law.cornell.edu/uscode/text/35/102

## Implementation Notes

This disclosure describes an untested design proposal; quantitative parameters are design choices awaiting field calibration, and no accuracy claim is made against opened boilers or weighed scale until that validation exists. A minimal prototype needs no machine modification: a plug power meter, a 0.1 gram cup scale with a phone timer, and a spreadsheet or phone app that logs reference shots, computes the four drift terms, and plots the rolling median index. Where a machine already exposes flow and temperature data, the same logic runs on those native channels with tighter sampling.

Bring-up procedure: descale the machine, run 20 reference shots with one recipe, and confirm that the baseline medians are stable across the last 10 shots. Then run the discriminating tests one at a time: a basket swap should move flow without moving heating duty, a grind change should move flow without moving heating duty, and a water-only shot at fixed resistance should give a clean hydraulic reference. If heating duty drifts while flow stays flat, suspect a thermostat or sensor issue before suspecting scale, since scale moves both channels together in this model.

Expected performance envelope (design targets for a validation program, not measured results): a descale-now call should correspond to a post-descale recovery of flow and heating duty toward baseline on at least 70 percent of calls, a false descale-now rate below 1 in 5 calls on machines verified clean by inspection, and a scale index that rises monotonically across at least 80 percent of monitored descale intervals. Thermoblock machines will need separate ripple and recovery calibrations from boiler machines, because their thermal mass and water path geometry differ, and rotary-pump machines will need a separate ripple baseline from vibratory-pump machines.

Limitations: coffee preparation dominates single-shot flow, so the system only trusts reference-recipe shots, and owners who change coffee, dose, or grind frequently will generate fewer index points. Scale in a steam circuit that never feeds the brew path may not appear in brew signals, and scale that deposits mainly in a steam wand will need a separate steam-flow check. Descaling chemistry, water temperature, and contact time affect how completely a descale resets the baseline, so a partial recovery does not always mean the diagnosis was wrong. This is a maintenance aid, not a safety device, and it does not replace manufacturer descaling guidance or professional service for electrical, pressure, or boiler faults.

The strongest counterargument is that flow and heating drift may simply be too confounded by coffee to support a trustworthy index in ordinary home use, even with reference recipes, because dose, grind, tamping, bean age, and ambient temperature all move shot curves. That objection is fair, and it is why the design leans on medians over many shots, on heating channels that coffee preparation cannot easily move, on the fixed-resistance water test, and on post-descale verification as a check that can falsify the call. If field data show that verified-clean machines routinely cross the descale-now threshold, the thresholds and weights in section 3 are wrong and should be recalibrated, not defended.

This disclosure describes an untested design proposal; quantitative parameters are design choices awaiting field calibration (see Implementation Notes).

---

*Published at [liveinthefuture.org/priorart](https://liveinthefuture.org/priorart/)*
