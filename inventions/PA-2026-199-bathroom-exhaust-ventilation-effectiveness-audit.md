# PA-2026-199: Auditing Residential Bathroom Exhaust Ventilation Effectiveness via Humidity Decay Analysis and Acoustic Airflow Verification

**Title:** System and Method for Auditing Residential Bathroom Exhaust Ventilation Effectiveness via Humidity Decay Analysis and Acoustic Airflow Verification

**Filing:** LITF-PA-2026-199
**Published:** October 10, 2026
**Domain:** Indoor Air Quality / Moisture / Sensor Fusion
**Full Disclosure:** [liveinthefuture.org/priorart/bathroom-exhaust-ventilation-effectiveness-audit.html](https://liveinthefuture.org/priorart/bathroom-exhaust-ventilation-effectiveness-audit.html)
**License:** [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/) — Public Domain

> Prior Art Notice: This document is a defensive technical disclosure published
> to constitute prior art under [35 U.S.C. Sec. 102(a)(1)](https://www.law.cornell.edu/uscode/text/35/102),
> effective as of the publication date above. It is intended to be cited as
> prior art against later-filed patent applications covering the subject matter
> described. The inventors, through this publication, reserve their rights
> under 35 U.S.C. Sec. 102(b)(1). The disclosure is intended to enable a
> person of ordinary skill in the art to make and use the described systems.
> This is general information, not legal advice.

---

## Abstract

Disclosed is a system and method for auditing whether residential bathroom exhaust fans actually ventilate, by measuring what the fan accomplishes rather than whether it is switched on. A temperature and relative humidity sensor in each bathroom records the moisture decay curve after every moisture-generating event (shower, bath). An edge hub converts the readings to absolute humidity, fits an exponential decay model to the post-event curve, and derives the effective air changes per hour (ACH_eff) achieved during the decay. The hub compares ACH_eff against the room-volume-normalized expectation implied by the installed fan rating and ASHRAE 62.2 local exhaust requirements. A MEMS microphone verifies the fan's acoustic run signature during the decay window, so the system distinguishes a fan that never ran from a fan that ran but moved no air. A fault-decoupling module separates seized motors, disconnected or crushed ducts, stuck backdraft dampers, clogged grilles, undersized fans, door-open dilution, and window-open confounds by directional signature tests across the humidity, acoustic, and optional current channels. The system produces a per-bathroom ventilation effectiveness score, a mold-risk index from cumulative time at elevated humidity, and graduated maintenance guidance. A commissioning mode verifies new or serviced installations against expected decay behavior.

## Technical Field

This disclosure relates to residential indoor air quality and moisture management, specifically to diagnostic auditing of intermittent local exhaust ventilation effectiveness through post-event humidity decay analysis fused with acoustic fan-operation verification.

## Background

Bathrooms carry the highest intermittent moisture load in a home. A single shower can drive room relative humidity from 45% to near saturation in minutes, and that moisture must leave the building envelope or it condenses on cool surfaces, feeds mold growth in wall cavities and on grout, and degrades paint, drywall, and framing over years. Sustained relative humidity above roughly 60% supports mold growth, which is why ventilation standards treat bathroom exhaust as a moisture-control requirement, not a comfort amenity.

ASHRAE Standard 62.2, the residential ventilation standard adopted into most US residential codes, requires local exhaust in each bathroom of at least 50 cubic feet per minute (CFM) intermittent (demand-controlled) or 20 CFM continuous, with local exhaust fans rated at no more than 3 sones at the required airflow and rated to deliver that airflow at a minimum static pressure of 0.25 inches water column. Model mechanical codes require bathroom exhaust ducts to terminate outdoors, never in attics, crawlspaces, or wall cavities.

The installed reality falls well short of the rated one. Fans are noisy, so occupants do not switch them on. Ducts installed by the lowest bidder are crushed, kinked, excessively long, or simply disconnected in the attic, discharging shower steam into the insulation. Backdraft dampers stick shut. Grilles clog with dust and lint. Cheap fans never delivered their nameplate CFM against real installed static pressure. And the occupant has no feedback channel at all: the fan hums, the mirror stays fogged, and nobody measures anything.

The existing art is entirely about *control*, not *audit*. Humidistat-equipped fans (Broan InVent Series with 50-80% RH setpoints, Panasonic WhisperSense and similar) switch the fan on when humidity crosses a threshold. [US7984859B2](https://patents.google.com/patent/US7984859B2/en) discloses automatic exhaust fan control by humidity level and rate of rise, citing [US6935570B2](https://patents.google.com/patent/US6935570B2/en) (Acker, humidity-sensor ventilation controller) and a hot-water-pipe temperature sensor that triggers the fan when hot water flows. [EP0707180A2](https://patents.google.com/patent/EP0707180A2/en) discloses a ventilator with the humidity sensor placed in the airflow path. [EP2450640](https://data.epo.org/publication-server/rest/v1.2/patents/EP2450640NWA2/document.html) improves the trigger by counting consecutive humidity rises, so slow ambient weather drift does not cause nuisance running. [US11920813](https://patents.justia.com/patent/11920813) (2024) discloses a drop-in humidity exhaust controller with analytics that distinguish air-conditioner operation from shower events. Every one of these asks "should the fan be on?" None asks "did the fan actually move the moisture out?" A humidistat fan with a disconnected duct runs obediently for an hour and achieves nothing, and no product or publication in the art measures that failure.

The strongest objection to this system is that humidity decay is confounded: doors open, windows open, the HVAC runs, outdoor humidity swings with weather. The answer is that the confounds are measurable and separable, which is what the fault-decoupling module is for. A decay curve that is fast because the window was open is not a passing grade, and a decay curve that is slow because the door was open and the moisture migrated is not a fan failure. The system earns its keep by telling these apart, not by fitting curves blindly.

The gap in the art is a complete deployable system that: (a) measures achieved ventilation effectiveness as effective air changes per hour derived from post-event moisture decay, rather than assuming rated airflow; (b) verifies fan operation acoustically so run state is known independently of the humidity signal; (c) decouples the distinct failure modes (dead motor, disconnected duct, stuck damper, crushed duct, clogged grille, undersized fan, occupant behavior) by directional signature tests; (d) detects code-noncompliant duct termination into attics via a correlated second sensor; and (e) converts the measurement into a per-bathroom effectiveness score and mold-risk index with graduated maintenance guidance.

## Detailed Description

### 1. Sensing architecture

Each audited bathroom contains one temperature and relative humidity sensing element: a digital sensor of SHT4x class (±1.5% RH typical accuracy, ±0.2°C), sampling at 30-second intervals, wall-mounted at 1.2 to 1.5 meters height on the wall opposite the shower, away from direct spray and away from the exhaust grille intake airstream. Placement matters: a sensor directly under the fan grille measures diluted air and understates the room load; a sensor inside the shower stall saturates and adds nothing after the event ends.

Fan-run verification uses a MEMS microphone node: in one embodiment a smart speaker already present in or near the bathroom, in another a dedicated low-cost microphone module on the hub, in another the microphone of an opted-in smartphone. No audio is stored and no audio leaves the device; the node computes band-energy features on-device and reports only run/no-run intervals with timestamps.

Optional channels strengthen the decoupling: a smart switch or plug on the fan circuit reporting current draw (distinguishes commanded-on from actually-on); a second temperature/RH sensor in the attic near the duct run (detects duct discharge into the attic); and an outdoor temperature/RH feed from a local weather station or the home's existing weather sensor (normalizes weather confounds). No sensor penetrates ductwork. Target bill-of-materials for the base kit (one room sensor, hub): under $30.

### 2. Moisture event detection

A moisture event is declared when relative humidity rises faster than 1.5% RH per minute sustained over 3 minutes, a threshold that shower and bath events exceed by a wide margin while cooking, breathing, and weather drift do not. A consecutive-rises discriminator (in the spirit of EP2450640, here used for audit gating rather than fan control) requires at least 4 of 5 consecutive 30-second samples to rise, rejecting sensor noise spikes. The event window opens at detection and closes when humidity returns to within 10% of the pre-event baseline or 90 minutes elapse, whichever comes first. Events shorter than 4 minutes of rise phase are discarded as non-shower transients (hand washing, toilet flush aerosol).

Each event is tagged with context the hub already knows: time of day, day of week, outdoor absolute humidity at event start, and whether the HVAC system ran during the window (from thermostat telemetry where available). This context feeds the confound handling in Section 6.

### 3. From relative humidity to absolute humidity

Relative humidity is temperature-coupled: as the bathroom cools after the shower stops, RH stays elevated even as actual moisture leaves, which would bias a decay fit. The hub therefore converts every sample to absolute humidity (grams of water per cubic meter of air) using the August-Roche-Magnus approximation for saturation vapor pressure:

```
e_s = 0.61094 × exp(17.625 × T / (T + 243.04))   [kPa, T in °C]
AH = 216.7 × (RH/100 × e_s) / (273.15 + T)       [g/m³]
```

All decay analysis operates on excess absolute humidity: AH_excess(t) = AH_room(t) − AH_outdoor(t), where outdoor absolute humidity comes from the local feed. This removes both the temperature coupling and the weather baseline in one step. A shower that raises room AH by 8 g/m³ against a dry winter outdoor baseline and the same shower against a humid summer baseline now produce comparable decay curves, which is the property that makes cross-season scoring possible.

### 4. Decay-curve fitting and effective air changes per hour

For a well-mixed room with ventilation rate Q and volume V, excess moisture decays exponentially once the source stops: AH_excess(t) = AH_0 × e^(−t/τ), with τ = V/Q. The hub fits the decay phase by least squares on ln(AH_excess) versus time, excluding the first 3 minutes after the humidity peak (the mixing transient, when the room is not yet well-mixed) and excluding the tail below 10% of peak excess (where sensor noise dominates). Fits with R² below 0.85 are rejected and the event is marked unusable rather than scored badly; a bad fit is a measurement failure, not a ventilation failure.

Effective air changes per hour follow directly: ACH_eff = 3600 / τ_seconds. The 62.2-implied requirement for the room is ACH_req = (50 CFM × 60) / V_ft³, with room volume from user input at commissioning (length × width × height) or a default 640 ft³ for a standard full bath.

*Illustrative worked example (not a measured result):* an 8×10×8 ft bathroom (640 ft³) with a nominal 50 CFM fan should achieve ACH_req = 4.7, i.e. τ ≈ 13 minutes. If the fitted decay shows τ = 50 minutes, ACH_eff = 1.2, equivalent to roughly 13 CFM of effective ventilation: the fan is delivering about a quarter of its nameplate, and the system has a number to say so. These figures illustrate the computation, not measurements from any prototype.

The well-mixed assumption is the method's known weakness, and the system treats it as such: bathrooms with the door closed and the fan running approach well-mixed within a few minutes (which is why the first 3 minutes are excluded), while bathrooms with the door open during the decay are flagged as dilution events (Section 6) rather than fitted. Stratification in tall bathrooms biases τ long by an estimated 10-20%; the scoring bands (Section 7) are wide enough to absorb this, and the trend across events matters more than any single fit.

### 5. Acoustic fan-run verification

The microphone node continuously computes band energy in the 120-500 Hz region (bathroom fan blade-pass and motor tones) plus broadband flow noise energy, compared against a rolling 24-hour ambient baseline for the same time of day. A fan-run interval is declared when band energy exceeds the baseline by 6 dB sustained over 60 seconds. The detector is deliberately coarse: it answers "did the fan run during the decay window?" not "how healthy is the fan?", because the humidity channel carries the effectiveness measurement.

Fusing the two channels is what makes the audit diagnostic rather than descriptive. Four combinations cover the field:

- **Fan ran, decay fast (τ near expectation):** ventilation verified. This is the passing grade, and the system says so explicitly so the homeowner learns what good looks like.
- **Fan ran, decay slow:** the fan is moving air somewhere other than out of the building, or barely moving it. Duct, damper, grille, or sizing fault. Proceed to decoupling.
- **Fan did not run, decay slow:** occupant behavior or control failure. The fan cannot be blamed for a job it was never given. The guidance is behavioral (run the fan; consider a humidistat or timer switch) rather than mechanical.
- **Fan did not run, decay fast:** window open, door open with strong stack effect, or HVAC return pulling air through the room. Not a ventilation failure, but the system notes that the moisture left via an uncontrolled path and does not credit the fan.

### 6. Fault decoupling by directional signature tests

When the fan ran but the decay was slow, the decoupling module runs directional tests to name the most likely cause:

- **Seized or dead motor:** acoustic signature absent or reduced to a 60 Hz electrical hum with no blade-pass tone, while the smart-switch channel (if present) shows current draw. Fan is commanded on, motor is not turning. Guidance: replace the fan unit.
- **Disconnected duct discharging into attic:** fan runs normally by acoustic signature, decay slow, and the optional attic sensor shows a humidity spike correlated within minutes of the bathroom event. This is both a ventilation failure and a code violation (exhaust must terminate outdoors). Guidance names both, with the code point stated plainly.
- **Stuck backdraft damper or crushed duct:** fan runs, but the blade-pass tone sits higher in frequency than the per-installation baseline (the motor unloads against higher static pressure; small shaded-pole and DC fan motors speed up as load drops), and decay is slow. Guidance: inspect the damper at the fan housing and the duct run for crushing or excessive length.
- **Clogged grille:** ACH_eff drifts downward gradually over months across events, then snaps back after the homeowner cleans the grille. The hub detects the snap-back (feature step exceeding 3 standard deviations of recent noise within two events) and prompts for confirmation, mirroring the filter-change anchoring used in HVAC filter loading estimation. Guidance: clean the grille; it is the highest-ROI first step and the system says so.
- **Undersized fan:** fan runs, decay is cleanly exponential (good R²) but τ is consistently longer than the 62.2-implied expectation for the measured room volume, with no drift over time and no damper signature. The fan works; there is not enough of it. Guidance: upgrade to a fan rated for the room volume at real installed static pressure.
- **Door-open dilution:** decay is fast but a second sensor in the adjacent hallway or bedroom shows a correlated humidity rise during the event: the moisture left the bathroom without leaving the house. The bathroom scores well on τ but the system flags moisture migration and does not credit the event toward the ventilation score.
- **Window-open confound:** decay fast, outdoor absolute humidity near indoor baseline, event tagged with plausible window weather (mild temperatures). The system marks the event window-ventilated and excludes it from the fan score: open-window drying is effective moisture control, but it is not the fan working.

When the evidence does not separate two causes (a crushed duct and a stuck damper look similar without the attic sensor), the guidance lists both in likelihood order and names the cheapest discriminating check first: look at the grille, then the damper, then the duct.

### 7. Ventilation effectiveness score and mold-risk index

Each bathroom receives a ventilation effectiveness score from 0 to 100: score = min(100, 100 × ACH_eff,median / ACH_req), where ACH_eff,median is the median across the last 10 usable events. The median rejects single-event outliers (a shower with the door wide open, a houseguest's 40-minute steam session). Score bands: 80-100 verified effective, 50-79 degraded (maintenance indicated), below 50 failing (diagnose promptly). The score is an audit of achieved ventilation, not a code certification, and the product copy says so.

Separately, the hub computes a mold-risk index: cumulative hours per week with room RH above 70%, the threshold band where mold growth accelerates on common bathroom materials. Tiers: under 5 hours/week low, 5-15 moderate, above 15 high. A bathroom can score well on ventilation effectiveness yet carry mold risk if the fan is never switched on; the two numbers together tell the occupant whether the problem is the equipment or the behavior, which is the entire point of auditing rather than controlling.

### 8. Commissioning mode and baseline learning

After installation, or after any fan service, the hub runs a commissioning test: the occupant runs a hot shower for 5 minutes with the bathroom door closed and the fan switched on, then leaves. The hub fits the decay, computes ACH_eff, and reports pass/fail against the 62.2-implied expectation for the room volume. A new fan installation that fails commissioning has a duct or damper problem on day one, which is exactly when it is cheapest to fix.

During the first 10 usable events the system reports provisional scores and learns the per-bathroom baseline: typical peak excess AH, typical decay τ with the fan running, and the acoustic fan signature. All subsequent scoring is relative to this baseline, so differences in room geometry, sensor placement, and fan model cancel out.

### 9. Graduated response and fleet embodiments

For a homeowner, the system presents a per-bathroom report card: the effectiveness score, the mold-risk tier, the trend across events, and the single most likely fault with its cheapest check first. Alerts are graduated: a new degradation trend notifies once ("hall bathroom ventilation has drifted down 30% over 6 weeks; check the grille"), a failing score with a likely disconnected duct escalates ("fan runs but moisture is not leaving; possible duct disconnect discharging into the attic").

In a property-management embodiment, scores across dozens or hundreds of bathrooms feed a fleet dashboard sorted by mold-risk index, so maintenance staff clean grilles and inspect ducts in the worst units before the moisture becomes a remediation invoice. In a real-estate embodiment, a 2-week pre-listing audit produces a ventilation report for the disclosure packet: cheap, factual, and more informative than the inspector's "fan operates" checkbox.

### 10. Edge processing and privacy

All humidity time series and all acoustic feature extraction run on the hub. Raw audio is buffered in a rolling 60-second window for the fan detector and discarded; no audio is stored and no audio leaves the home in any embodiment. Only per-event scalars leave the device: event timestamps, fitted τ, ACH_eff, score, and alert state. The microphone hears the bathroom, which is precisely why the acoustic channel is architecturally constrained to on-device band-energy features with no recording path.

### 11. Description of Figures

- **Figure 1:** System layout: room temp/RH sensor placement, microphone node, hub, optional smart switch, optional attic sensor, and outdoor humidity feed, with data flows.
- **Figure 2:** Example moisture event: RH and temperature traces, conversion to excess absolute humidity, the excluded mixing transient, and the fitted exponential decay with τ marked.
- **Figure 3:** The four fan-run/decay-rate combinations and their diagnostic meanings.
- **Figure 4:** Fault-decoupling decision table: directional signatures of seized motor, disconnected duct, stuck damper/crushed duct, clogged grille, undersized fan, door-open dilution, and window-open confound across humidity, acoustic, current, and attic-sensor channels.
- **Figure 5:** Commissioning test procedure and pass/fail bands against 62.2-implied ACH for typical bathroom volumes; grille-cleaning snap-back and baseline re-anchoring illustration.

## Claims

1. A system for auditing bathroom exhaust ventilation effectiveness, comprising: a temperature and relative humidity sensor mounted in a bathroom; a microphone node that verifies exhaust fan operation by acoustic signature; and an edge hub that detects moisture-generating events from the humidity signal, fits an exponential decay model to post-event humidity, derives an effective air changes per hour (ACH_eff) from the fitted decay time constant, and compares ACH_eff against a room-volume-normalized ventilation expectation, wherein the system measures achieved ventilation rather than fan switch state.
2. The system of claim 1, wherein the hub converts relative humidity and temperature samples to absolute humidity via a saturation-vapor-pressure approximation and performs the decay fit on excess absolute humidity above the outdoor baseline, removing temperature coupling and weather-baseline effects from the measurement.
3. The system of claim 1, wherein moisture events are detected by a rate-of-rise threshold on relative humidity combined with a consecutive-rises discriminator that rejects sensor noise and slow ambient drift, and wherein decay fits below a goodness-of-fit threshold are rejected as measurement failures rather than scored as ventilation failures.
4. The system of claim 1, wherein the microphone node computes on-device band-energy features in the fan blade-pass frequency region against a rolling ambient baseline, reports only fan-run intervals with no audio stored and no audio transmitted, and wherein the hub fuses fan-run state with decay rate into four diagnostic combinations: fan-ran/fast-decay (verified), fan-ran/slow-decay (duct or fan fault), fan-not-run/slow-decay (behavioral), and fan-not-run/fast-decay (uncontrolled ventilation path).
5. The system of claim 1, further comprising a fault-decoupling module that distinguishes a seized motor (current draw without blade-pass tone), a disconnected duct (normal acoustic signature with slow decay), a stuck backdraft damper or crushed duct (elevated blade-pass frequency against higher static pressure with slow decay), a clogged grille (gradual ACH_eff decline with snap-back on cleaning), and an undersized fan (clean exponential decay with consistently long time constant relative to room volume) by directional signature tests.
6. The system of claim 5, further comprising a second temperature and humidity sensor positioned in the attic near the exhaust duct run, wherein a humidity spike in the attic sensor correlated with a bathroom moisture event identifies duct discharge into the attic as a code-noncompliant termination.
7. The system of claim 1, further comprising a ventilation effectiveness score computed as the median ACH_eff across recent usable events normalized to the ASHRAE 62.2-implied air changes per hour for the measured room volume, reported in graduated bands distinguishing verified, degraded, and failing ventilation.
8. The system of claim 1, further comprising a mold-risk index computed from cumulative weekly hours with room relative humidity above an elevated threshold, reported independently of the ventilation effectiveness score so that equipment faults are distinguished from occupant behavior.
9. The system of claim 1, further comprising a commissioning mode that guides an occupant through a standardized moisture event with the door closed and the fan on, fits the resulting decay, and reports pass or fail against the expected decay time constant for the room volume, verifying new or serviced installations on day one.
10. The system of claim 1, wherein the hub learns a per-bathroom baseline across an initial set of usable events and expresses all subsequent effectiveness estimates as drift relative to that baseline, and wherein a snap-back in ACH_eff exceeding a statistical threshold after grille cleaning re-anchors the baseline on occupant confirmation.
11. A method for auditing bathroom exhaust ventilation effectiveness, comprising: mounting a temperature and humidity sensor in a bathroom away from the exhaust grille airstream; recording humidity during and after moisture-generating events; converting samples to excess absolute humidity above the outdoor baseline; fitting an exponential decay to the post-event curve excluding the mixing transient; deriving effective air changes per hour from the decay time constant; verifying exhaust fan operation during the decay window by acoustic signature; comparing the effective air changes per hour against the room-volume-normalized expectation; and decoupling fan, duct, damper, grille, sizing, and behavioral causes of slow decay by directional signature tests.
12. A fleet embodiment of the system of claim 1, comprising a dashboard aggregating per-bathroom ventilation effectiveness scores and mold-risk indices across multiple dwelling units, sorted by mold risk, for prioritized maintenance dispatch, and a pre-listing audit embodiment producing a dated ventilation report from a multi-week measurement period.

## Implementation Notes

Mount the room sensor on the wall opposite the shower at 1.2 to 1.5 meters height, never in the exhaust grille's intake airstream and never where direct spray hits it. A sensor that measures the diluted air at the grille will understate the room load and flatter the fan; a sensor in the spray zone saturates and contributes nothing after the event.

Do not score half-baths. Without a bathing fixture there are no moisture events, and the system will sit idle. Powder rooms with only a toilet and sink do not need this audit; the disclosure is for full bathrooms.

Commission with the door closed. An open door during the commissioning shower turns the test into a dilution measurement and the resulting baseline will be wrong for every closed-door event after. The guided test says this explicitly and the hub rejects commissioning events where the decay fit is suspiciously fast relative to room volume.

Clean the grille before you replace the fan. Field experience says the grille is the most common cause of gradual degradation and the cheapest fix, which is why the guidance orders checks by cost: grille, damper, duct, then fan. A system that recommends a $200 fan replacement before a 30-second grille wipe has its priorities backwards.

Outdoor humidity normalization is not optional. Without it, a humid August week makes every bathroom look broken and a dry January week makes every bathroom look heroic. The excess-absolute-humidity formulation exists so that scores mean the same thing in both months.

Say plainly what the score is not. The ventilation effectiveness score is an audit of achieved moisture removal, not a code compliance certification and not an airflow measurement. It does not replace a flow-hood traverse at commissioning. What it replaces is the current state of knowledge, which is nothing: today nobody measures whether the fan works after the installer leaves.

Renters and landlords split the benefit and the cost here, so design the product accordingly. The renter gets the mold-risk number; the landlord gets the fleet dashboard and the maintenance ticket. The sensor kit must be installable without tools beyond a wall anchor, removable without damage, and clearly not a camera: say "humidity and temperature only" on the box, because a sensor in a bathroom that anyone mistakes for a camera is a product-killing misunderstanding.

## Prior Art References

1. [HVI / ASHRAE 62.2 Ventilation Best Practices Guide](https://www.hvi.org/hviorg/document-server/?cfp=HVIORG/assets/File/public/CEC-400-2010-006.pdf): 50 CFM intermittent / 20 CFM continuous bathroom exhaust, 3 sone limit, 0.25 in. w.c. rating static pressure
2. [ASHRAE Addenda k and m to Standard 62.2-2010](https://www.ashrae.org/file library/technical resources/standards and guidelines/standards addenda/62_2_2010_k_m_final.pdf): Local ventilation exhaust airflow rates (bathroom 50 CFM / 25 L/s)
3. [US7984859B2](https://patents.google.com/patent/US7984859B2/en): Automatic exhaust fan control apparatus and method (humidity level and rate-of-rise control)
4. [US6935570B2](https://patents.google.com/patent/US6935570B2/en) (Acker): Ventilation controller with humidity sensor (cited in US7984859B2)
5. [EP0707180A2](https://patents.google.com/patent/EP0707180A2/en): Ventilator with humidity sensor in the air flow
6. [EP2450640](https://data.epo.org/publication-server/rest/v1.2/patents/EP2450640NWA2/document.html): Humidity control system (consecutive-rises method to reject ambient drift)
7. [US11920813](https://patents.justia.com/patent/11920813): Drop-in exhaust fan humidity controller (shower vs. air-conditioner analytics)
8. [Broan InVent Series AE80SL](https://acdistributors.com/product/broan-invent-series-bathroom-exhaust-fan-with-led-light-80-cfm-0-8-sones-humidity-sensing-energy-star-certified-ae80sl/): Humidity-sensing bathroom exhaust fan, 80 CFM, 0.8 sones, 50-80% RH setpoints
9. EPA, "A Brief Guide to Mold, Moisture and Your Home": Mold growth on materials dampened by sustained elevated humidity; moisture control as the key to mold control
10. ASHRAE Standard 160, Criteria for Moisture-Control Design Analysis in Buildings: Mold index methodology for assessing mold risk from temperature/humidity time series
11. IRC Section M1501 / IMC Section 501: Mechanical exhaust air required to discharge to the outdoors; termination in attics or crawlspaces prohibited
12. [35 U.S.C. Sec. 102](https://www.law.cornell.edu/uscode/text/35/102): Conditions for patentability; novelty and prior art
