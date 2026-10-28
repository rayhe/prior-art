# PA-2026-217: Detecting Windshield Wiper Blade Degradation and Washer Nozzle Obstruction Using Wiper Motor Current Signatures, Rain Sensor Data, and Forward Camera Streak Analysis

**Title:** System and Method for Detecting Windshield Wiper Blade Degradation and Washer Nozzle Obstruction Using Wiper Motor Current Signatures, Rain Sensor Data, and Forward Camera Streak Analysis

**Filing:** LITF-PA-2026-217
**Published:** October 28, 2026
**Domain:** Automotive / Sensor Fusion / Predictive Maintenance
**Full Disclosure:** [liveinthefuture.org/priorart/windshield-wiper-blade-wear-motor-current-vision.html](https://liveinthefuture.org/priorart/windshield-wiper-blade-wear-motor-current-vision.html)
**License:** [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/) - Public Domain

> Prior Art Notice: This document is a defensive technical disclosure published
> to constitute prior art under [35 U.S.C. Sec. 102(a)(1)](https://www.law.cornell.edu/uscode/text/35/102),
> dedicated to the public domain as of the publication date above.

---

## Abstract

Disclosed is a system and method for detecting windshield wiper blade degradation and washer nozzle obstruction without putting any sensor on the blade, the arm, or the nozzle. The system reads three signals that are already present on many modern vehicles, or cheaply added: the wiper motor current from the body controller or a current sensor, the rain rate from the rain and light sensor, and the forward camera view from an advanced driver assistance camera or a dashcam. Each wiper sweep is a labeled event, because the controller knows the wiper speed setting, the sweep direction, and the vehicle speed. The system learns a per-vehicle baseline for a fresh blade, then watches later sweeps for deviations. A fusion classifier separates a worn or chipped blade edge (streak lines that stay at the same wiper angle across sweeps, plus a rise in motor current ripple) from hardened rubber (higher mean current and slower sweeps with uniform haze), a bent arm or uneven pressure (streaks on one side only, with outward and return strokes differing), a clogged washer nozzle (a missing spray sector and a changed washer pump signature, with wiper current near normal), and dry or icy conditions that must not be scored as wear. A streak persistence test separates blade damage from glass contamination: contamination washes away or moves after a washer cycle, while blade streaks return at the same angles sweep after sweep.

## Technical Field

This disclosure relates to vehicle maintenance diagnostics, specifically to wiper system health monitoring using motor current analysis, rain sensing, and camera vision. It sits at the intersection of predictive maintenance, non-intrusive sensing applied to a moving mechanism, and camera perception: a worn blade matters twice now, once for the driver and once for the forward camera, whose lane and obstacle perception degrades through the same streaked glass.

## Background

A wiper blade is a consumable that fails gradually, then suddenly matters. Rubber hardens under sun and heat, chips on grit and ice, and takes a set from sitting parked in one position. The driver notices the failure in the worst conditions, in heavy rain at night, when streaks scatter oncoming headlights across the field of view. Fleet operators face the same failure at scale, with no practical way to inspect every blade between services.

What drivers and fleets actually use today is weaker than the problem:

- **Visual inspection:** Looking at the blade edge or waiting for streaks. It works, but it is infrequent, it depends on the inspector, and a blade that looks intact can still chatter and streak under load.
- **Calendar replacement:** Replacing blades every 6 or 12 months. This wastes good blades in mild climates and still misses damage that happens early in the interval, especially ice damage in one winter storm or grit damage on one dusty road.
- **Rain sensor alone:** A rain sensor reports how wet the glass is. It cannot say whether the wiper cleared it, because it measures water arriving, not water remaining after a sweep.
- **Driver complaint:** Waiting for a driver to report streaking. In a fleet, that report arrives late, if at all, and it never distinguishes a worn blade from a clogged nozzle, a bent arm, or a dirty windshield, so the wrong part gets replaced.
- **Camera-only streak detection:** A camera can see streaks, but a camera alone cannot tell why they are there. Washer fluid starvation, glass film, ice, and blade wear all leave water on glass, and they need different fixes.

Motor current analysis is established for rotating machinery diagnostics: a failing bearing, a binding linkage, or a changing friction load shows up in current mean, ripple, and transient shape. Camera streak analysis is established for windshield inspection in manufacturing. Neither technique has been combined with rain sensor gating, sweep-direction asymmetry, and a streak persistence test to diagnose individual wiper faults by type, using only signals already on the vehicle, with no sensor on the blade itself. That combination is the gap this disclosure fills.

## Detailed Description

### 1. System Components

The system comprises: a wiper motor and linkage with a current measurement path, either controller telemetry from a smart wiper module or a shunt or Hall sensor on the motor supply, sampled at 100 Hz or faster during sweeps; a rain and light sensor reporting rain rate in bands, or a camera-derived rain estimate where no dedicated sensor exists; a forward-facing camera, either the vehicle advanced driver assistance camera or a dashcam, with a view through the wiped zone; a washer pump with a current measurement path and, where fitted, a reservoir level input; vehicle speed, ambient temperature, and wiper setting inputs from the vehicle network; and a diagnostic unit, which may be the body controller, an assistance system controller, or a small edge computer, that stores per-vehicle baselines and runs the classifier. No sensor is installed on any wiper blade, arm, or nozzle.

### 2. Fresh Blade Baseline and Normalization

When a new blade is fitted, or after a baseline reset, the system records a fresh blade fingerprint during the first 30 wet sweeps: mean and root mean square motor current per sweep, current at sweep reversal, sweep duration, and current ripple in the 5 to 50 Hz band, each recorded separately for outward and return strokes, for each wiper speed setting, and for rain rate bands. The baseline is normalized for the factors that move current without any wear: vehicle speed, because aerodynamic load on the blade rises with speed; temperature, because cold rubber is stiffer and winter current runs higher; and wiper speed setting, because intermittent, low, and high speeds load the motor differently. A later sweep is scored against the baseline for its own combination of speed setting, rain band, temperature band, and vehicle speed band, not against a single factory threshold, so a large SUV blade and a small hatchback blade are judged by their own histories.

### 3. Fault Signatures and the Differential Classifier

The classifier combines motor current, washer pump current, rain rate, and camera streak features into a fault score per fault type. The signatures are:

- **Worn or chipped blade edge:** motor current ripple up 20 percent or more from baseline, with streak lines in the camera view that persist at the same wiper angles across three or more consecutive sweeps. The fixed angle is the key evidence: a chip in the edge leaves the same unwiped line on every sweep, while a random droplet does not.
- **Hardened or aged rubber:** mean motor current up 15 to 30 percent from baseline, sweep duration longer by 5 percent or more, and uniform haze or fine streaking across the whole wiped zone rather than lines at fixed angles. The classifier requires the shift to persist across days and temperature bands, so one cold morning does not trigger a replacement flag.
- **Bent arm or uneven blade pressure:** streaks confined to one end or one side of the wiped zone, combined with a stroke asymmetry index, the difference between outward and return stroke current, outside its baseline band. A bent arm loads the linkage differently in the two directions, while a worn edge loads both directions alike.
- **Clogged washer nozzle:** washer pump current or run time shifted from its baseline, a spray sector missing or weak in the camera view during a washer cycle, and wiper motor current near normal. The streaks sit in the unsprayed sector and clear after a manual wash, which separates a nozzle fault from blade wear, where streaks return after washing.
- **Low washer fluid or failing washer pump:** pump current below baseline or no spray at all in the camera view, with a reservoir level input, where fitted, confirming low fluid. Wiper current stays near normal, so the system does not blame the blade for a fluid problem.
- **Glass contamination or film:** streaks that move, fade, or clear after a washer cycle, or that cover the glass outside the wiped zone as well as inside it. The persistence test in section 5 rejects these events, so a bug-splattered windshield does not generate a blade replacement alert.
- **Ice, snow, or dry wipe (suppression conditions):** ambient temperature at or below freezing with a large current spike or a stalled sweep, or rain rate near zero with a dry glass signature. These events are logged but excluded from wear scoring, because ice load and dry friction mimic wear and would otherwise flood the driver with false alerts.

### 4. Motor Current Signature Analysis

Each sweep yields a current trace from park to reversal to park. The diagnostic unit reduces it to a feature vector: mean current, root mean square current, reversal spike amplitude and width, ripple energy in the 5 to 50 Hz band, sweep duration, and the outward versus return asymmetry index. A healthy blade on wet glass produces a smooth trace with a modest reversal spike as the linkage changes direction. A chipped edge produces ripple as the damaged section catches and releases across the glass, with the ripple repeating at the same sweep phase. Hardened rubber raises the whole trace, because friction is higher through the full stroke. A binding linkage, which is a different fault from blade wear, produces a localized current hump at the same linkage angle on every sweep, with camera streaks absent, and the classifier reports it as a linkage inspection item rather than a blade item.

### 5. Forward Camera Streak Analysis and the Persistence Test

After each sweep, the camera compares frames before and after the sweep inside the wiped zone, mapped into wiper angle coordinates so that a pixel position corresponds to a blade angle. Residual water is detected as thin lines and haze that remain after the blade has passed. Each residual is scored by angle, length, and width, and the same angle map is compared across sweeps. The persistence test is the core of the vision step: a residual that appears at the same angle on three or more consecutive wet sweeps is scored as blade damage, while a residual that appears once, moves, or disappears after a washer cycle is scored as contamination or a transient droplet. Scoring is gated by rain rate: sweeps in very light rain, where the glass is only partly wet, and sweeps in very heavy rain, where new water arrives faster than the blade can clear it, are excluded from streak scoring, because both conditions make even a new blade look streaked. Night scoring uses headlight and streetlight glare masking, so scattered light through a streak is measured only where the glare source is stable across frames.

### 6. Rain Sensor Gating and Fusion Decision

The rain sensor sets the scoring window. A sweep is scored for wear only when the rain rate sits in the calibrated middle band, the glass is wet but not flooded, and the vehicle speed sits in a band where aerodynamic load is predictable. The fusion rule requires agreement before a replacement alert: either vision persistence plus a current deviation, or a strong single-channel signal, such as a fixed-angle streak persisting across ten sweeps, confirmed on a later drive. A current-only deviation produces a watch flag and a request for a washer cycle test, which often separates contamination from wear without a shop visit. A vision-only flag produces the same washer cycle test before any alert, because the cheapest explanation for a streak is dirt, not a dead blade.

### 7. Staged Alerts, Fleet Reporting, and Replacement Verification

Each confirmed fault produces a report containing: the most likely fault type with a confidence score, the affected blade (driver or passenger side, where the camera and current paths can separate them, or the pair where they cannot), the streak angle map, the estimated remaining useful life in wet sweeps, and the recommended action, such as replace blade, clear nozzle, refill fluid, or inspect linkage. For fleets, reports aggregate into a dashboard ranked by streak persistence and by camera perception impact, so vehicles whose assistance camera looks through the worst glass are serviced first. After a blade replacement, the system watches the next 10 wet sweeps: if streak persistence falls below threshold and current returns to within 10 percent of the fresh blade baseline, the fault is marked verified-repaired and a new baseline is stored; if streaks persist at the same angles, the report stays open and the second-ranked fault type, typically a bent arm or contaminated glass, is promoted, so a wrong first call is visible instead of silently repeated.

### 8. Figures Description

- **Figure 1:** System diagram showing the wiper motor, linkage, rain sensor, forward camera, washer pump, and diagnostic unit, with vehicle network inputs for speed, temperature, and wiper setting.
- **Figure 2:** Motor current traces for a fresh blade, a chipped blade, and a hardened blade, showing ripple, mean shift, and reversal spike differences.
- **Figure 3:** Camera streak angle maps for three consecutive sweeps, showing fixed-angle lines for blade damage versus moving residuals for contamination.
- **Figure 4:** Fault signature table mapping current, washer pump, rain, and vision deviations to worn edge, hardened rubber, bent arm, clogged nozzle, low fluid, contamination, and ice or dry wipe.
- **Figure 5:** Example replacement verification plot showing streak persistence and current deviation returning to baseline over ten post-replacement wet sweeps.

## Claims

1. A system for diagnosing windshield wiper faults, comprising: a wiper motor current measurement path; a rain sensor or camera-derived rain estimate; a forward-facing camera with a view through a wiped zone; and a diagnostic unit; wherein the diagnostic unit attributes motor current and camera measurements to individual wiper sweeps using the wiper setting and vehicle speed, without any sensor on any wiper blade, arm, or nozzle.
2. The system of claim 1, wherein the diagnostic unit builds a fresh blade baseline of motor current mean, ripple, reversal spike, and sweep duration, recorded separately for outward and return strokes, wiper speed settings, rain rate bands, temperature bands, and vehicle speed bands.
3. The system of claim 1, wherein the diagnostic unit scores a later sweep against the baseline for its own combination of wiper speed, rain band, temperature band, and vehicle speed band, rather than against a single fixed threshold.
4. The system of claim 1, further comprising a differential classifier that separates a worn or chipped blade edge from hardened rubber, a bent arm, a clogged washer nozzle, low washer fluid, glass contamination, and ice or dry wipe conditions, using motor current, washer pump current, rain rate, and camera streak features.
5. The system of claim 4, wherein a worn or chipped edge is identified by motor current ripple above baseline combined with camera streak lines persisting at the same wiper angles across multiple consecutive sweeps.
6. The system of claim 4, wherein a clogged washer nozzle is identified by a shifted washer pump signature and a missing spray sector in the camera view during a washer cycle, with wiper motor current near normal.
7. The system of claim 1, further comprising a streak persistence test that maps camera residuals into wiper angle coordinates and rejects residuals that move, fade, or clear after a washer cycle, as contamination rather than blade damage.
8. The system of claim 1, further comprising a rain gating module that excludes from wear scoring sweeps in very light rain, very heavy rain, freezing conditions, or dry wipe conditions.
9. The system of claim 1, further comprising a stroke asymmetry index comparing outward and return stroke current, wherein a bent arm is identified by streaks confined to one side of the wiped zone combined with an asymmetry index outside its baseline band.
10. A method for diagnosing windshield wiper faults without a blade sensor, comprising: recording wiper motor current and forward camera views for sweeps labeled by wiper setting and vehicle speed; building a fresh blade baseline normalized for speed, temperature, and rain rate; classifying deviations into fault types using current, washer pump, and streak persistence features; and verifying a repair by checking streak persistence and current against the fresh blade baseline over subsequent wet sweeps.

## Prior Art References

1. [National Highway Traffic Safety Administration](https://www.nhtsa.gov/): Federal Motor Vehicle Safety Standard 104, windshield wiping and washing systems.
2. [SAE International](https://www.sae.org/): Wiper system and windshield washer standards.
3. [Bosch Mobility](https://www.bosch-mobility.com/): Wiper blades, wiper motors, and rain sensors.
4. [Insurance Institute for Highway Safety](https://www.iihs.org/): Headlight and visibility research in rain and at night.
5. [Continental](https://www.continental.com/): Rain and light sensors and camera-based driver assistance.
6. [Valeo](https://www.valeo.com/): Wiper systems and windshield washer technology.
7. [AAA](https://www.aaa.com/): Vehicle maintenance guidance, including wiper blade replacement.
8. [U.S. Environmental Protection Agency](https://www.epa.gov/): Water use and washer fluid considerations for vehicle washing.

## Implementation Notes

This disclosure describes an untested design proposal; quantitative parameters are design choices awaiting field calibration, and no accuracy claim is made against inspected blades or measured currents until that validation exists. The cheapest viable build uses signals already on the vehicle: body controller wiper motor telemetry where it exists, a rain sensor already fitted for automatic wipers, and an assistance camera already fitted for lane keeping. Where motor telemetry is unavailable, a clamp-on Hall sensor on the motor supply and a dashcam provide the same inputs for under $100 at retail quantities, versus a dedicated blade sensor that would need to survive flexing, ice, and washer fluid on every blade. The washer cycle test in section 6 should run only when the reservoir is not empty, and the system must not run the wipers dry to generate a test sweep, because dry wiping damages the blade it is trying to assess. Vehicles with a single wiper, or with opposed wipers that cannot be separated by current alone, should report the pair as one unit and let the camera angle map, not the current path, separate the sides.

Limitations: the forward camera view covers only part of the wiped zone on many vehicles, so streaks outside the camera field are missed unless a dashcam with a wider view is used. Night glare, oncoming headlights, and sun low on the horizon all raise the vision noise floor, and runs in those conditions should be down-weighted, not trusted equally. Motor current varies across vehicle models and across wiper linkage designs, so thresholds must be learned per vehicle, not copied from another model. Aftermarket blades with different rubber compounds and beam designs produce different baselines, which is why the baseline resets at every blade change.

The strongest counterargument is that drivers already see streaks, so an automated blade monitor solves a problem nobody has. That objection is fair for a careful owner who replaces blades on sight. The case for this design is the driver who does not notice, the fleet that cannot inspect every vehicle, and the assistance camera that loses lane lines through streaked glass long before the driver complains, because the camera stares through one fixed patch of glass on every drive. If field data show that drivers replace blades promptly once streaks appear, and that assistance cameras are robust to the streak levels a worn blade produces, the vision claims should be dropped and the design judged on fleet maintenance scheduling alone, where current and pump signatures still separate blade wear from nozzle clogs without a shop visit.
