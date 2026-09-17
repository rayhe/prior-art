# PA-2026-174: HVAC Run Capacitor Failure Prediction from Start Transient Analysis

**Title:** System and Method for Predicting HVAC Run Capacitor Failure Using Compressor Start Transient Electrical Signature Analysis

**Filing:** LITF-PA-2026-174
**Published:** September 17, 2026
**Domain:** HVAC / Predictive Maintenance
**Full Disclosure:** [liveinthefuture.org/priorart/hvac-run-capacitor-prognostics.html](https://liveinthefuture.org/priorart/hvac-run-capacitor-prognostics.html)
**License:** [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/) — Public Domain

> Prior Art Notice: This document is published as defensive prior art under
> [35 U.S.C. Sec. 102(a)(1)](https://www.law.cornell.edu/uscode/text/35/102).
> The inventions described herein are dedicated to the public domain as of the
> publication date above.

---

## Abstract

Disclosed is a non-invasive system and method for predicting run capacitor failure in residential single-phase HVAC equipment by analyzing the electrical signature of every compressor and condenser fan start transient. A split-core current transformer and voltage sensor capture current and voltage waveforms at kilohertz sampling rates inside a several-second window triggered by contactor closure. For each start event, the system extracts: start duration (time from contactor closure until the current envelope settles within 15 percent of steady-state running current), inrush current peak and envelope decay constant, supply voltage sag depth and duration, start-cycle harmonic distortion, failed-start re-attempt counts, and running current drift relative to the motor nameplate rating. A physics-informed model maps the longitudinal drift of these features against a per-unit healthy baseline to estimate remaining capacitance as a percentage of rated microfarads, tracking the industry replacement threshold of 10 percent below nominal. A confounder discriminator separates capacitor degradation from low supply voltage, contactor pitting, locked rotor conditions, and high head pressure restarts using idle voltage, waveform dropout, and outdoor temperature gating. An exponential degradation fit on the estimated capacitance series forecasts the date the capacitor will cross the replacement threshold, issuing tiered alerts from maintenance planning through urgent compressor protection, including a hard-start kit versus immediate replacement recommendation.

## Technical Field

This invention relates to heating, ventilation, and air conditioning (HVAC) predictive maintenance, specifically to non-invasive condition monitoring of single-phase induction motor run capacitors through per-start electrical transient signature analysis, longitudinal degradation trending, and remaining-useful-life forecasting.

## Background

Nearly every residential air conditioner and heat pump built in the last fifty years starts its compressor and condenser fan with the help of a run capacitor: a metallized-film can, typically rated 5 to 80 microfarads at 370 or 440 VAC, that phase-shifts current into the motor auxiliary winding to produce starting torque. Most units use a dual-run capacitor with three terminals (C, Herm, Fan) serving the compressor and fan from one can (capacitor replacement guide). The component is a consumable with a typical service life of about 10 to 11 years, and heat is its enemy: capacitor reliability follows the standard rule that life roughly halves for each 10 C of sustained temperature rise, which is why capacitors mounted inside sun-baked condenser cabinets fail first.

Industry practice treats a capacitor reading more than 10 percent below its rated microfarads as end of life and replaces it preemptively (HVAC technician practice, Bogleheads). The economics explain why: the part costs roughly $12 to $15 at retail and about $99 to $180 installed, while a capacitor allowed to fail completely produces hard starting, humming stall, repeated thermal overload trips, and elevated running current that cooks the compressor windings. One widely viewed technician walkthrough documents the common and costly misdiagnosis: homeowners told they need a $1,800 to $2,500 compressor replacement when the actual fault is a $160 to $180 capacitor (Casey Services). Cheap imported capacitors compound the problem, with one contractor documenting a 35 percent failure rate before 18 months of service (Swinson, justcapacitors.com). Contractors list failing capacitors among the most frequent causes of summer no-cool calls (Pioneers Heating and Air, September 2026), and preventative maintenance visits exist largely to catch weak capacitors before they damage motors (Shirley Air).

The detection method used today is episodic and invasive: a technician powers down the unit, discharges the capacitor, disconnects it, and measures microfarads with a multimeter, typically once a year at most. Between visits, degradation is invisible. Homeowners get no warning between the last good reading and the first failed start on the hottest day of the year.

Non-intrusive load monitoring (NILM), proposed by Hart in 1992, disaggregates whole-home power data into per-appliance consumption from a single meter (NILM survey, arXiv). NILM identifies which appliance is running; it does not assess the health of components inside the appliance. Motor current signature analysis (MCSA) is an established non-invasive condition monitoring technique for industrial motors, where anomalies in the supply current reveal mechanical and electrical faults without interrupting operation (PMC review of fault detection techniques; Miljković, MCSA review). That literature targets three-phase industrial machines and steady-state faults such as broken rotor bars and eccentricity. It does not address the single-phase residential case, the start transient as the diagnostic window, or the capacitor as the component under test.

The gap in the art is a system that: (a) captures the start transient of every compressor and fan start non-invasively at the disconnect or panel; (b) extracts start-duration, inrush envelope, voltage sag, and harmonic features per start event; (c) trends those features longitudinally against a per-unit healthy baseline to estimate remaining capacitance; (d) discriminates capacitor degradation from confounders (low line voltage, contactor pitting, locked rotor, high head pressure); (e) assesses the compressor and fan sections of a dual-run capacitor independently; and (f) forecasts the replacement date with tiered homeowner alerts before the first failed start.

## Detailed Description

### 1. Sensor Configuration and Start Event Capture

The sensing unit installs without breaking any conductors: a split-core current transformer (100 A rated, 1 percent accuracy class) clamps around the compressor conductor inside the condenser disconnect box or at the branch breaker, and a voltage sense lead measures line voltage at the same point. A microcontroller with a 12-bit ADC samples current at 4 kHz and voltage at 1 kHz. No disconnection of the capacitor, no power-down, and no refrigerant circuit access are required.

Start events are captured with a triggered recording scheme. The firmware watches the current channel for a step edge exceeding 3 A within 50 ms, the signature of contactor closure. On trigger, it records a 5-second window: 500 ms of pre-trigger baseline and 4.5 seconds of start and run. Idle sampling between events drops to 1 Hz RMS logging to limit storage; full waveform capture occurs only on starts. A typical residential compressor starts 3 to 8 times per day in cooling season, producing a dense longitudinal dataset within weeks.

### 2. Start Transient Feature Extraction

Each captured start transient is processed to extract a feature vector:

- **Start duration Δt:** time from contactor closure until the current envelope decays to within 15 percent of the steady-state running current. A healthy 3-ton compressor typically settles in 300 to 600 ms; this is the primary degradation indicator.
- **Inrush peak and envelope decay τ:** the locked-rotor current peak (typically 5 to 7 times running current) and the exponential decay constant of the envelope fit. Weakening capacitance lengthens τ before it lengthens Δt, giving early warning.
- **Voltage sag depth and duration:** maximum RMS voltage dip during the start and the time for voltage to recover to within 2 percent of pre-start level, distinguishing motor-side from supply-side problems.
- **Start-cycle harmonic distortion:** total harmonic distortion and 3rd/5th harmonic magnitudes computed over the start window. As capacitance falls, the auxiliary winding phase shift degrades, torque pulsation grows, and harmonic content rises.
- **Failed-start re-attempts:** sequences where current rises to locked-rotor levels, collapses within 2 seconds (thermal overload trip), and retries within 5 minutes. Counts per 24 hours are tracked; 3 or more triggers the urgent tier.
- **Running current drift:** steady-state RMS current averaged over the 60 seconds after settling, compared against the compressor nameplate rated load amps (RLA). A weak capacitor raises running current 5 to 15 percent before starting becomes audibly labored.

### 3. Capacitance Estimation Model

The physical basis: in a permanent-split-capacitor motor, the run capacitor sets the phase lead of the auxiliary winding current, which sets starting torque. As capacitance drifts down, starting torque falls, and the rotor takes longer to accelerate through the high-slip region, stretching Δt and τ. The model inverts this relationship.

During a 30-day commissioning period, the system establishes a per-unit healthy baseline: median Δt, τ, sag depth, and harmonic levels across starts, each regressed against outdoor temperature and pre-start off duration to remove weather and short-cycle effects. Thereafter, each start event produces feature residuals relative to baseline. A calibrated mapping, fit from laboratory measurements of start transients across capacitors artificially aged to 100, 90, 80, and 70 percent of rating, converts the residual vector into an estimated remaining capacitance percentage. The mapping is monotonic in Δt residual by construction: longer starts mean less capacitance. The estimate is reported as a 7-day rolling median to suppress single-event noise from voltage sags and short cycles.

### 4. Confounder Discrimination

Several conditions mimic capacitor degradation and must be excluded before alerting:

- **Low supply voltage:** if pre-start idle voltage is below 108 V on a nominal 120 V leg (or the equivalent per-unit on 240 V), the start lengthening is attributed to the utility or panel, not the capacitor. The system reports a supply voltage advisory instead.
- **Contactor pitting:** pitted contacts produce current waveform dropouts (sub-cycle interruptions) during the start, visible as notches in the envelope. Envelope notch count above threshold classifies the fault as contactor, not capacitor.
- **Locked rotor / mechanical seizure:** inrush current that holds at locked-rotor levels with no decay, followed by overload trip within seconds, with no gradual Δt drift in preceding weeks, indicates mechanical failure. This bypasses the degradation model and triggers an immediate service alert.
- **High head pressure restarts:** starts within 3 minutes of a previous run on days above 32 C ambient lengthen naturally due to unequalized refrigerant pressures. These events are excluded from the degradation series via outdoor temperature and off-duration gating.
- **Electric backup heat:** on heat pump systems, auxiliary heat strips (5 to 15 kW) energize during defrost or below the balance point, producing step edges far larger than any compressor start. The trigger logic ignores edges above 40 A and, where a thermostat W/AUX signal is available, suspends capture while backup heat is active.

A rules engine applies these gates in order before any capacitance estimate updates the trend or any alert fires.

### 5. Dual-Run Capacitor Independent Section Assessment

Because most residential units use one dual-run can for both compressor (Herm terminal, typically 35 to 55 µF) and condenser fan (Fan terminal, typically 5 to 7.5 µF), the two sections age independently and often fail months apart. The system separates them: the condenser fan starts 200 to 500 ms before the compressor contactor pulls in on most units, producing a distinct smaller start transient in the same current record. Fan start duration, fan inrush, and fan running current are extracted from this pre-compressor window and trended independently, yielding separate capacitance estimates for the Herm and Fan sections. A fan-section failure (the more common first failure, since the smaller capacitance section degrades faster) triggers its own alert tier with a lower parts cost and a note that compressor operation may continue safely with the fan section addressed promptly.

### 6. Remaining Useful Life Forecasting and Alert Tiers

The 7-day median capacitance estimates form a degradation series. An exponential decay fit, the standard form for capacitor aging under thermal stress, extrapolates the series to the 90 percent replacement threshold and the 80 percent urgent threshold. The fit is seasonally derated: degradation rates measured in summer are scaled by the Arrhenius relationship for the cooler months, so a forecast made in October does not optimistically assume winter aging rates will persist into the next July.

Alerts escalate through three tiers:

- **Planning (estimated 88 to 90 percent):** notify the homeowner that the capacitor has entered the replacement window; recommend ordering the exact rated part ($12 to $15) and scheduling replacement at convenience, ideally before cooling season.
- **Scheduling (estimated 82 to 88 percent, or forecast crossing 80 percent within 60 days):** recommend booking a technician visit within 30 days; note that continued operation is safe but the failure risk is rising.
- **Urgent (estimated below 82 percent, or 3+ failed-start re-attempts in 24 hours):** recommend immediate replacement; offer the hard-start kit option explicitly as a bridge measure that reduces start stress but does not restore capacitance, with a warning that a hard-start kit masks the underlying degradation and should not be treated as a permanent fix.

Every alert includes the estimated capacitance percentage, the trend chart, the forecast date, and the confounder checks that were passed, so a technician can verify the diagnosis with a single multimeter reading.

### 7. Implementation Notes

Bill of materials target is under $40 at modest volume: split-core CT ($8), voltage sense transformer ($6), ESP32-class microcontroller with 12-bit ADC ($5), enclosure and disconnect-box mounting hardware ($8), power supply ($5), miscellaneous ($8). All feature extraction and the degradation model run on-device; only the daily capacitance estimate and alert state are transmitted, so no waveform data leaves the home and there are no audio privacy concerns. Smart thermostat integration is optional: where a communicating thermostat exposes equipment runtime, its cooling-call log validates start segmentation and flags short-cycling the electrical sensor might miss.

Scope is single-phase permanent-split-capacitor and capacitor-start motors: the residential installed base. Variable-speed inverter-driven systems modulate the compressor electronically and do not present the same start transient; they are explicitly out of scope and the system disables itself when it detects a variable-frequency drive signature (soft ramp with no inrush step).

Known limitations: the 30-day commissioning baseline assumes a healthy capacitor at install. If the unit is commissioned on an already degraded capacitor, the baseline is polluted and absolute percentage estimates read high, though the relative trend and the forecast slope remain valid; a one-time technician multimeter reading at install resolves the absolute offset. Two-stage compressors produce distinct start signatures per stage and require per-stage baselines, handled by clustering start events on inrush peak before trending. The laboratory calibration mapping covers standard PSC compressor and fan motors from 1.5 to 5 tons; commercial three-phase equipment is out of scope.

Safety: run capacitors retain charge after power-down and can deliver a dangerous shock; the sensing installation requires no contact with capacitor terminals, but any replacement must be performed with the breaker off, the capacitor discharged through a resistor, and, for most homeowners, by a qualified technician. The alert text carries this warning verbatim.

## Claims

1. A system for predicting run capacitor failure in single-phase HVAC equipment, comprising: a split-core current transformer configured to clamp around a compressor conductor without breaking the circuit; a voltage sensor; a microcontroller configured to detect contactor closure from a current step edge and capture a multi-second current and voltage waveform window per start event; wherein the microcontroller extracts a start transient feature vector comprising start duration, inrush envelope decay constant, voltage sag depth and duration, start-cycle harmonic distortion, and running current drift.
2. The system of claim 1, wherein a per-unit healthy baseline is established over a commissioning period with temperature and off-duration regression, and per-start feature residuals relative to the baseline are mapped through a calibrated model to an estimated remaining capacitance expressed as a percentage of rated microfarads.
3. The system of claim 2, wherein the estimated capacitance is reported as a rolling median and an exponential decay fit forecasts the date the capacitance will cross a replacement threshold set at 10 percent below rated capacitance.
4. The system of claim 1, further comprising a confounder discriminator that attributes start lengthening to low supply voltage when pre-start idle voltage is below a threshold, to contactor pitting when the current envelope contains sub-cycle dropouts, to locked rotor when inrush shows no decay followed by overload trip without prior gradual drift, and to high head pressure when the start follows a short off-cycle on a hot day, excluding such events from the degradation series.
5. The system of claim 1, wherein the condenser fan start transient occurring in a pre-compressor window of the same current record is extracted separately, yielding independent capacitance estimates for the Herm and Fan sections of a dual-run capacitor.
6. The system of claim 3, wherein the exponential forecast is seasonally derated using the Arrhenius temperature relationship so that degradation rates measured in one season are adjusted for expected operating temperatures in future seasons.
7. The system of claim 1, wherein tiered alerts are issued at a planning tier, a scheduling tier, and an urgent tier based on estimated capacitance percentage and failed-start re-attempt counts, each alert including the estimate, trend, forecast date, and confounder checks passed.
8. The system of claim 7, wherein the urgent tier presents a hard-start kit explicitly as a bridge measure with a warning that it masks rather than restores capacitance degradation.
9. A method for non-invasive HVAC run capacitor prognostics comprising: capturing current and voltage waveforms for every compressor start via triggered recording on contactor closure; extracting start duration, inrush envelope decay, voltage sag, harmonic distortion, and running current features per start; trending feature residuals against a per-unit baseline; estimating remaining capacitance percentage; forecasting threshold crossing with a seasonally derated exponential fit; and issuing tiered replacement alerts before the first failed start.
10. The method of claim 9, further comprising fusing smart thermostat equipment runtime data to correlate starts with cooling calls and to validate start event segmentation.

## Prior Art References

1. NILM survey (arXiv): Non-intrusive load monitoring proposed by Hart, 1992; disaggregates appliances but does not assess component health. [http://arxiv.org/pdf/2010.16050v1](http://arxiv.org/pdf/2010.16050v1)
2. Miljković, "Brief Review of Motor Current Signature Analysis": MCSA fundamentals and fault signatures for induction motors. [https://hrcak.srce.hr/148715](https://hrcak.srce.hr/148715)
3. Signal Injection as a Fault Detection Technique (PMC): MCSA as a recognized non-invasive industrial standard. [https://pmc.ncbi.nlm.nih.gov/articles/PMC3231603/](https://pmc.ncbi.nlm.nih.gov/articles/PMC3231603/)
4. Bogleheads HVAC thread: Industry practice: replace capacitors reading more than 10% below rated microfarads. [https://www.bogleheads.org/forum/viewtopic.php?p=4742168](https://www.bogleheads.org/forum/viewtopic.php?p=4742168)
5. AC Fan or Compressor Not Working? DIY Capacitor Replacement Guide: Run capacitor typical life 10-11 years; dual-run C/Herm/Fan terminals; ~$12 part cost. [https://www.youtube.com/watch?v=19A9lvQ6lIA](https://www.youtube.com/watch?v=19A9lvQ6lIA)
6. Bad AC/Heat Pump Compressor or Capacitor? (Casey Services): $1,800-$2,500 compressor misdiagnosis vs $160-$180 capacitor repair. [https://www.youtube.com/watch?v=F51cbT9ZFkM](https://www.youtube.com/watch?v=F51cbT9ZFkM)
7. Swinson, "Cheap capacitors": 35% failure rate before 18 months for low-quality import capacitors. [https://www.justcapacitors.com/wp-content/uploads/CheapCapacitors.pdf](https://www.justcapacitors.com/wp-content/uploads/CheapCapacitors.pdf)
8. Pioneers Heating and Air, September 2026: Failing capacitors among the most frequent summer breakdown causes. [https://www.economyjack.com/2026/09/17/pasadena-hvac-contractor-pioneers-heating-and-air-outlines-common-causes-of-ac-breakdowns/](https://www.economyjack.com/2026/09/17/pasadena-hvac-contractor-pioneers-heating-and-air-outlines-common-causes-of-ac-breakdowns/)
9. Shirley Air via Z106.3: Preventative maintenance catching weak capacitors before compressor damage. [https://lifestyle.all80sz1063.com/story/285156/air-conditioner-repair-warning-signs-costs-hurst-tx-hvac-experts-explain/](https://lifestyle.all80sz1063.com/story/285156/air-conditioner-repair-warning-signs-costs-hurst-tx-hvac-experts-explain/)
