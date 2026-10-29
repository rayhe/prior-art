# PA-2026-218: Estimating Residential Propane Tank Fill Level Using Acoustic Resonance Spectroscopy and Thermal Time-Constant Fusion

**Title:** System and Method for Estimating Residential Propane Tank Fill Level Using Acoustic Resonance Spectroscopy and Thermal Time-Constant Fusion

**Filing:** LITF-PA-2026-218
**Published:** October 29, 2026
**Domain:** Energy / Sensor Fusion / Acoustic Diagnostics
**Full Disclosure:** [liveinthefuture.org/priorart/propane-tank-fill-acoustic-resonance.html](https://liveinthefuture.org/priorart/propane-tank-fill-acoustic-resonance.html)
**License:** [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/) - Public Domain

> Prior Art Notice: This document is a defensive technical disclosure published
> to constitute prior art under [35 U.S.C. Sec. 102(a)(1)](https://www.law.cornell.edu/uscode/text/35/102),
> dedicated to the public domain as of the publication date above.

---

## Abstract

Disclosed is a system and method for estimating the fill level of a residential propane tank without penetrating the tank, replacing the gauge, or relying on the mechanical float gauge. A clamp-on module strikes or excites the tank's steel wall at a marked point and records the structural resonance response with a contact acoustic pickup. Because liquid propane mass-loads the shell, the tank's resonant modes ring lower as fill rises and higher as the tank empties. A second, independent modality measures the tank wall's thermal time constant from the overnight cooling curve: more liquid means more thermal mass and a slower response.

The two modalities are fused with a per-tank self-calibration that uses delivery tickets as labeled ground truth and a degree-day-normalized consumption model between deliveries. The system predicts run-out date, flags a stuck float gauge when the acoustic estimate disagrees with the dial, and reports over a low-power radio link.

## Technical Field

This invention relates to fuel inventory monitoring, specifically to non-invasive estimation of liquefied petroleum gas fill level in stationary residential tanks using structural acoustic resonance and thermal response analysis. It sits at the intersection of acoustic diagnostics, thermal system identification, and edge-deployed sensor fusion: a steel pressure vessel becomes its own sensor, and two weak physical signals combine into a level estimate that neither could produce alone.

## Background

Millions of rural and suburban homes heat with propane stored in on-site ASME tanks, typically 120 to 1,000 gallons, filled to a maximum of 80% of water capacity per [NFPA 58](https://www.nfpa.org/codes-and-standards/nfpa-58-standard-development/58) to leave vapor space for thermal expansion. One gallon of liquid propane holds roughly 91,500 BTU ([U.S. Energy Information Administration](https://www.eia.gov/energyexplained/units-and-calculators/energy-conversion-calculators.php)), so a 500-gallon tank at 80% carries about 400 gallons, or roughly 36.6 million BTU: about 366 hours of a 100,000-BTU/hr furnace, or some 45 days of cold-weather heating at 8 hours a day. A cold snap can burn through that inventory in weeks.

Running a tank dry is expensive and dangerous. The furnace shuts down, pipes can freeze, and restarting requires a pressure test and appliance relight by the propane company, often with an emergency delivery surcharge. Delivery itself takes days to schedule, so the homeowner must order well before the tank is empty. The entire logistics chain depends on knowing the fill level.

What homeowners actually rely on falls short of the problem:

- **Mechanical float gauges:** The standard dial gauge (e.g., [Rochester](https://www.rochester-gauges.com/) R3D remote-ready dial) couples magnetically to a float arm inside the tank. Floats stick on corroded pivot points, read high after a partial fill sloshes the arm, and drift with age. A stuck gauge reads 40% while the tank sits at 8%.
- **The hot-water trick:** Pour hot water down the tank side and feel where the steel turns cool. The liquid-filled section conducts heat away faster. It works, but it is manual, weather-dependent, and gives a rough level, not a number.
- **Float-dial remote readers:** Products such as [Otodata](https://otodatagps.com/) monitors magnetically couple to the existing remote-ready dial. They inherit every failure mode of the float they read: if the float sticks, the remote reading sticks with it.
- **Under-tank ultrasonic sensors:** Products such as [Mopeka](https://mopeka.com/) mount an ultrasonic transducer under the tank and measure echo time to the liquid surface. This requires a clean acoustic coupling path through the tank bottom, fails on tanks with heavy scale or saddles that block the beam, and cannot be used on buried or mounded tanks.

Structural vibration of cylindrical shells is well understood: [Leissa, "Vibration of Shells," NASA SP-288 (1973)](https://ntrs.nasa.gov/citations/19730018197) gives the modal structure, and internal liquid mass-loading of shell modes is textbook fluid-structure interaction. [ASTM E2001](https://www.astm.org/e2001-23.html) standardizes resonant ultrasound spectroscopy for material characterization. Closer to the present application, [García et al., "Non-Intrusive Tank-Filling Sensor Based on Sound Resonance," Electronics 7(12):378 (2018)](https://www.mdpi.com/2079-9292/7/12/378) demonstrated non-intrusive fill estimation of tanks from the exterior using knock excitation, a piezo pickup, FFT analysis, a resonance-to-fill transfer function with per-tank calibration, and wireless reporting, achieving good resolution especially above half-tank. Remote propane tank monitoring with predicted-versus-actual level comparison is also established (see [US7937216B2, Humphrey, 2011](https://patents.google.com/patent/US20100241277A1/en)), and temperature-curve-based indirect level inference for pressurized liquids has been disclosed (see [US20250020499A1](https://patents.google.com/patent/US20250020499A1/en)).

What the art still lacks is the combination this disclosure teaches: fusing the acoustic resonance modality with an independent overnight thermal time-constant modality, so each covers the other's failure modes; self-calibration from delivery tickets with a fleet prior and degree-day-normalized consumption interpolation between deliveries, converging without a technician visit; a stuck-float cross-check that distrusts the mechanical dial instead of inheriting its failures; run-out prediction tied to delivery scheduling with staged alerts; quiet-window measurement scheduling with contamination rejection; and explicit unsupported-installation detection. No single reference teaches that combination as a deployable residential system.

## Detailed Description

### 1. System Components

The module is a weatherproof (IP65) puck, roughly 90 mm in diameter, that clamps to the tank's cylindrical shell with a magnetic base or a stainless strap. It carries no penetration into the pressure boundary and mounts at least 300 mm from any valve, fitting, or weld, on the upper half of the shell where liquid contact varies with fill. Components: a solenoid tapper or piezoelectric actuator that strikes the wall with a repeatable impulse (approximately 2 N peak, 1 ms contact); a contact piezoelectric pickup (accelerometer or piezo disc) with a 20 Hz to 20 kHz bandwidth, coupled to the steel with acoustic gel under the magnetic base; a tank-wall temperature sensor (RTD or thermistor, ±0.2 °C); an ambient temperature sensor; a microcontroller with DSP capability (e.g., ARM Cortex-M4F); a LoRa or WiFi radio; and a battery with a small solar trickle cell sized for one measurement per night. Target bill of materials: under $45 at 10,000-unit volumes.

### 2. Resonance Measurement Protocol

Measurements run once per night, defaulting to 3:00 AM local time, when no propane appliance is drawing (no flow-induced vibration), wind is typically lowest, and the tank is thermally settled. The actuator delivers five impulses spaced 2 seconds apart at the marked strike point. The pickup records 250 ms per strike at 48 kHz sampling. The five spectra are averaged in the frequency domain after rejecting any strike whose broadband RMS exceeds the median by more than 6 dB (wind gust or passing vehicle rejection).

From the averaged spectrum the processor extracts the center frequencies of the first three axisymmetric shell breathing modes, typically found between 200 Hz and 2 kHz for a 500-gallon tank, using parabolic interpolation around each spectral peak for sub-bin resolution. Liquid propane mass-loads these modes: a full tank (80%) rings lower than an empty tank by a design target of roughly 8 to 15% in the fundamental breathing mode, depending on shell thickness and diameter. The per-tank calibration in section 4 absorbs the actual value for each installation, so the system never depends on the nominal figure being exact. The processor stores the three peak frequencies plus temperature at measurement time.

### 3. Thermal Time-Constant Modality

Independently, the module records the tank wall temperature every 5 minutes overnight. After the evening's last appliance draw and before sunrise heating, the wall follows a first-order cooling curve toward ambient. The processor fits an exponential to the 10:00 PM to 4:00 AM segment and extracts the thermal time constant τ. Liquid propane's specific heat is near 2.5 kJ/(kg·K), so the roughly 770 kg (about 1,680 lb) of liquid in a full 500-gallon tank represents on the order of 1,900 kJ/K of thermal mass, against the steel shell's roughly 200 kJ/K. The filled tank's τ is therefore several times longer than the empty tank's. τ is normalized by the overnight ambient temperature swing so that a cold night and a mild night remain comparable. This modality needs no excitation hardware beyond the temperature sensor already on the board, and it is insensitive to the acoustic noise that can corrupt a resonance reading.

### 4. Per-Tank Self-Calibration

Resonant frequencies vary between tanks with shell thickness, diameter, saddle placement, and paint buildup, so no universal frequency-to-level table is used. Instead each tank calibrates itself. Labeled ground-truth points come from delivery tickets: when the propane company fills the tank, the ticket states gallons delivered, and the pre-delivery level is known from the previous estimate, giving a post-delivery fill fraction. The homeowner enters the delivered gallons in the companion app, or photographs the ticket for OCR. Two or three deliveries establish the mapping.

Between deliveries, a consumption model interpolates: each heating appliance's input rating (BTU/hr, from its nameplate, entered once) multiplied by its runtime (from the smart thermostat or a current sensor on the furnace) converts to gallons at 91,500 BTU per gallon, adjusted for appliance efficiency. The model is normalized by heating degree days so a cold week and a mild week do not distort the interpolation. The per-tank estimator is a ridge regression mapping the feature vector (three resonant peak frequencies, thermal τ, wall temperature, ambient temperature) to fill fraction, fit on the labeled delivery points and the consumption-interpolated trajectory. A new tank starts with a fleet prior (population-average coefficients for its nominal size) and converges to its own fit within two delivery cycles.

### 5. Fusion, Run-Out Prediction, and Stuck-Gauge Detection

The nightly estimate fuses the resonance-based level and the thermal-based level with inverse-variance weighting, where each modality's variance is learned from its residual history. The fused level feeds a run-out predictor: current level minus a 10% reserve, divided by the degree-day-normalized daily burn rate, yields days remaining. The app alerts at 30, 14, and 7 days, and can share the estimate with the delivery company for automatic scheduling.

A stuck float gauge is detected by cross-check: the homeowner periodically enters the dial reading (or a camera reads the dial in a variant embodiment). If the resonance-based level and the dial disagree by more than 12 percentage points for three consecutive nights, the system flags the float as suspect and trusts the acoustic estimate, telling the homeowner the dial needs service. The flag is a maintenance prompt, not a substitute: it does not replace the mechanical gauge for the delivery driver's visual check or any gauge requirement under the dealer's service terms or NFPA 58. Below 30% indicated fill the comparison threshold widens, because resonance sensitivity degrades at low fill (see Limitations).

### 6. Multi-Tank and Edge Cases

For manifolded tank pairs, one module per tank measures independently and a hub sums the per-tank estimates into a site inventory total. Underground or mounded tanks are excluded: soil coupling damps the shell modes unpredictably, and the module reports "unsupported installation" rather than a false reading. Vertical 120-gallon cylinders (common for standby generators) are supported with a separate fleet prior for the vertical geometry.

### 7. Figures Description

- **Figure 1:** Clamp-on module mounted on a horizontal residential propane tank, showing the actuator strike point, contact pickup, temperature sensor, and radio link to the home hub.
- **Figure 2:** Averaged resonance spectra for one tank at 75%, 40%, and 12% fill, showing the upward frequency shift of the first three shell breathing modes as liquid mass decreases.
- **Figure 3:** Overnight wall-temperature cooling curves for high and low fill, with fitted exponentials and extracted thermal time constants.
- **Figure 4:** Self-calibration timeline: delivery-ticket labeled points, consumption-model interpolation between deliveries, and convergence of the per-tank regression from fleet prior to individualized fit.

### 8. Limitations and Failure Modes

No sensing modality is unconditional. The following limitations are material to deployment and are stated here so that no embodiment claims more than the physics supports.

- **Coldest-night draw:** The 3:00 AM quiet-window assumption fails on the coldest nights, when the furnace may run continuously and flow-induced vibration contaminates every strike. This is exactly when run-out risk peaks. The module vets each window against thermostat runtime data when available and against strike-to-strike spectral consistency; a contaminated window is discarded and retried at the next quiet hour. If no clean window occurs for 48 hours, the system reports the last good estimate with widened uncertainty and a "stale" flag rather than a corrupted reading.
- **Weather effects:** Rain, wind, and frost change the external heat-transfer coefficient that τ depends on and add mass damping to the shell modes. τ normalization covers ambient temperature swing but not precipitation. The home hub ingests a precipitation flag from a local weather station; nights with rain, freezing rain, or frost carry widened uncertainty or are skipped, at the cost of slower estimate updates during storms.
- **Coating and corrosion drift:** A fresh coat of paint or advancing rust changes the shell's mass and damping mid-life, silently invalidating the per-tank signature learned at install. The re-calibration trigger is the delivery ticket itself: when labeled delivery points start disagreeing with the per-tank regression beyond a set residual threshold across two consecutive deliveries, the estimator resets to the fleet prior and re-learns over the next two delivery cycles, flagging the event in the app.
- **Manifolded-pair cross-talk:** Tapping one tank of a manifolded pair excites the other through the shared piping, so the second module's pickup hears the first module's strike. Modules on a shared manifold measure sequentially with a settle gap, mount at the far ends of their tanks, and discard any strike whose cross-correlation with the sibling module's pickup exceeds threshold, retrying until clean.
- **Low-fill accuracy degradation:** Resonance sensitivity is weakest at low fill, consistent with García et al. 2018, who report the best resolution above half-tank. Below roughly 30% indicated fill the fused estimate's confidence interval widens substantially; the stuck-gauge comparison threshold widens with it, and run-out alerts below 15% carry an explicit low-confidence flag. The system must never present a single-digit fill estimate as precise.

## Claims

1. A system for estimating fill level of a stationary liquefied petroleum gas tank, comprising: a non-invasive module clamped to the tank's steel shell without penetrating the pressure boundary; an actuator that mechanically excites the shell at a marked point; a contact acoustic pickup that records the shell's resonance response; and a processor that extracts resonant mode frequencies and maps them to a fill level estimate, wherein liquid mass-loading shifts the resonant frequencies downward as fill level rises.
2. The system of claim 1, wherein the processor tracks the center frequencies of at least two axisymmetric shell breathing modes in the 200 Hz to 2 kHz band using sub-bin peak interpolation on averaged impulse-response spectra.
3. The system of claim 1, further comprising a tank-wall temperature sensor, wherein the processor fits a thermal time constant to the overnight wall cooling curve, derives a thermal-based level from the thermal time constant, and fuses the resonance-based level and the thermal-based level using inverse-variance weighting.
4. The system of claim 1, wherein measurements are scheduled during a nightly quiet window when no appliance draws fuel, and individual strikes are rejected when broadband RMS energy exceeds a median-based threshold indicating wind or traffic contamination.
5. A method for self-calibrating a tank level estimator, comprising: recording labeled fill fractions from delivery tickets entered by a user or extracted by optical character recognition; interpolating consumption between deliveries using appliance input ratings, runtimes, and heating degree day normalization; fitting a per-tank regression mapping resonant frequencies and thermal time constant to fill fraction, initialized from a fleet prior for the tank's nominal size; and fusing the resonance-based level and the thermal-based level into a fused level estimate.
6. The method of claim 5, further comprising predicting a run-out date from the fused level estimate minus a reserve fraction divided by a degree-day-normalized burn rate, and issuing staged alerts at configurable day thresholds.
7. The method of claim 5, further comprising detecting a stuck mechanical float gauge by comparing the resonance-based level against a user-entered or camera-read dial value, flagging the gauge as suspect when disagreement exceeds a threshold for a plurality of consecutive nights.
8. The system of claim 1, wherein the module is battery powered with solar trickle charging, performs all spectral processing on-device, and reports the level estimate over a low-power radio link.
9. The system of claim 1, further comprising independent modules on each tank of a manifolded pair, wherein a hub sums the per-tank estimates into a site inventory total.
10. The system of claim 1, wherein the processor identifies an unsupported installation, including underground or mounded tanks, from anomalous modal damping and reports an unsupported-installation status instead of a level estimate.

## Prior Art References

1. [NFPA 58, Liquefied Petroleum Gas Code](https://www.nfpa.org/codes-and-standards/nfpa-58-standard-development/58): 80% maximum fill requirement for ASME containers
2. [U.S. Energy Information Administration](https://www.eia.gov/energyexplained/units-and-calculators/energy-conversion-calculators.php): Propane energy content, approximately 91,500 BTU per gallon
3. [Rochester Gauges](https://www.rochester-gauges.com/): R3D remote-ready dial float gauges for LP-gas tanks
4. [Mopeka Products](https://mopeka.com/): Ultrasonic under-tank propane level sensors
5. [Otodata](https://otodatagps.com/): Remote tank monitors coupling to existing float-gauge dials
6. [US7937216B2 (Humphrey, 2011)](https://patents.google.com/patent/US20100241277A1/en): System for monitoring propane or other consumable liquid in remotely located storage tanks, including predicted versus actual level comparison
7. [Leissa, A.W., "Vibration of Shells," NASA SP-288, 1973](https://ntrs.nasa.gov/citations/19730018197): Modal structure of cylindrical shells
8. [ASTM E2001-23](https://www.astm.org/e2001-23.html): Standard Guide for Resonant Ultrasound Spectroscopy
9. [García, A. et al., "Non-Intrusive Tank-Filling Sensor Based on Sound Resonance," Electronics 7(12):378, 2018](https://www.mdpi.com/2079-9292/7/12/378): Knock excitation with piezo pickup and FFT analysis for non-intrusive tank fill estimation; best resolution above half-tank
10. [US20250020499A1](https://patents.google.com/patent/US20250020499A1/en): Device for the indirect measurement of the fill level of a pressurized liquid in a fluid container, via the temperature curve of extracted gas
11. [ASME BPVC Section VIII](https://www.asme.org/codes-standards/find-codes-standards/bpvc-viii-1-bpvc-section-viii-rules-construction-pressure-vessels): Construction rules for pressure vessels including LP-gas tanks

## Implementation Notes

The strike point matters more than the actuator choice. Mark the spot with the supplied adhesive ring and always strike within 10 mm of it; moving the excitation point between nights injects more variance than any algorithm can remove. Couple the pickup with acoustic gel and check it yearly, since dried gel is the most common cause of degraded signal in field units.

Start every new tank on the fleet prior and do not trust the absolute number until the second delivery ticket lands. The first delivery gives the regression its slope; the second confirms it. Between those points, report the estimate with a "calibrating" badge and wider uncertainty bounds rather than a false precision.

Keep the module's radio transmissions away from the measurement window. A LoRa transmission's current draw can couple into the analog front end through a shared ground; the firmware measures first, transmits second, with a 500 ms settle gap. The solenoid tapper draws a short current spike, so it fires from a dedicated capacitor bank rather than the main rail.

Privacy deserves its own paragraph because heating-fuel consumption reveals occupancy patterns, including vacations and vacant homes, and radio traffic can be sniffed by anyone in range. All spectral processing happens on-device, so raw audio never leaves the home; only the level estimate and diagnostics go out, over encrypted transport (AES-128 on LoRaWAN, TLS on WiFi). Sharing the estimate with the delivery company is opt-in with informed consent, revocable in the app. Delivery ticket photos contain account numbers and service addresses, so the app extracts gallons and date and then redacts or deletes the image.

Safety review: the module is low-voltage DC, potted, with no sparking contacts, and mounts on the shell away from valves and fittings. It does not replace the mechanical gauge for the delivery driver's visual check, and it must never be presented as a safety device. It is an inventory instrument. The run-out estimate is advisory: it carries the uncertainty of the acoustic and thermal estimates plus the burn-rate model, and it must not be the sole basis for deferring a scheduled delivery or overriding a keep-fill agreement with the dealer. Underground tanks, tanks with heavy external scale at the measurement point, and tanks under 120 gallons are out of scope for the first hardware revision.

