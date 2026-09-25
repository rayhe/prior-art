# PA-2026-182: Predicting EV Auxiliary Battery Failure from Contactor Wake Transients and DC-DC Charge Acceptance

**Title:** System and Method for Predicting Auxiliary Battery Failure in Electric Vehicles Using DC-DC Converter Telemetry and Quiescent Load Analysis

**Filing:** LITF-PA-2026-182
**Published:** September 25, 2026
**Domain:** Electric Vehicles / Battery Diagnostics / Edge AI
**Full Disclosure:** [liveinthefuture.org/priorart/ev-12v-auxiliary-battery-failure-prediction.html](https://liveinthefuture.org/priorart/ev-12v-auxiliary-battery-failure-prediction.html)
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

Electric vehicles retain a low-voltage auxiliary battery (nominally 12 V, increasingly 16 V lithium-ion in newer models) that energizes the high-voltage contactors, body controllers, and safety systems whenever the traction pack is isolated. When this battery dies, the vehicle is immobilized even with a fully charged traction pack, a failure mode behind a large share of electric-vehicle roadside breakdowns. Conventional field diagnostics for low-voltage batteries rely on starter-motor crank voltage, an observable that does not exist in electric vehicles, and the DC-DC converter masks degradation by holding the bus at absorption voltage while driving. Disclosed is an edge diagnostics system that predicts auxiliary battery failure weeks in advance by fusing electric-vehicle-specific observables: voltage sag during high-voltage contactor-closure inrush, DC-DC converter charge-acceptance profiling, temperature-compensated quiescent voltage decay during sleep, and the interval between automatic DC-DC top-up events during extended parking. Per-vehicle baselines, changepoint detection, and remaining-useful-life estimation produce tiered service alerts, and the system distinguishes genuine battery degradation from abnormal parasitic drain.

## Technical Field

This disclosure relates to battery prognostics for electric vehicles, low-voltage power distribution in battery-electric and plug-in hybrid vehicles, DC-DC converter diagnostics, and edge-computed state-of-health estimation for lead-acid and lithium-ion auxiliary batteries.

## Background

Every production electric vehicle carries two electrical systems: a high-voltage traction pack (typically 400 V or 800 V) and a low-voltage auxiliary battery (historically 12 V lead-acid; 16 V lithium-ion in newer Tesla models and some others). The auxiliary battery powers the contactor coils that connect the traction pack, the body control modules, door locks, hazard lights, and emergency-call systems during the interval between vehicle wake and DC-DC converter startup. A dead auxiliary battery therefore bricks the vehicle completely, regardless of traction-pack state of charge. Owner reports document this repeatedly: stranded drivers with full traction batteries, some suffering three or more auxiliary failures in a few months.

The scale is large. Germany's ADAC reported 3.7 million roadside assistance calls in 2025, with 45.4 percent caused by weak or dead 12-volt batteries, and noted that the low-voltage system fails more often in electric cars than in combustion-engine cars. Industry reporting indicates lead-acid auxiliary batteries last roughly two to four years in electric vehicles, shorter than in combustion vehicles, because electric vehicles never subject the battery to the high-current starter discharges that exercise lead-acid plates, while holding the battery at partial state of charge for long periods accelerates sulfation and grid corrosion. Manufacturers have responded in part by switching chemistries (Tesla moved the Model S and X to lithium-ion auxiliary batteries in 2021, then to 16 V lithium-ion across newer models) and some next-generation architectures propose deleting the separate low-voltage battery entirely, but the installed fleet of hundreds of millions of vehicles will carry discrete auxiliary batteries for a decade or more.

Existing diagnostics transfer poorly to electric vehicles. In combustion vehicles, the standard field test is crank-voltage sag: the starter draws 150 to 300 A and the depth of the voltage dip reveals internal resistance. Electric vehicles have no starter, so the canonical observable is absent. Dashboard voltage readings are equally uninformative, because the DC-DC converter holds the auxiliary bus at 13.8 to 14.8 V whenever the vehicle is awake or charging, masking degradation until the battery can no longer survive a sleep-to-wake transition. Some vehicles issue a low-voltage warning, but these are threshold alerts that fire hours before failure, not trend-based predictions that give weeks of notice. Aftermarket Bluetooth battery monitors report voltage and estimated state of charge but apply combustion-era heuristics to electric-vehicle duty cycles.

What is needed is a prognostics method built around the electrical events that actually exist in an electric vehicle: contactor-closure inrush, DC-DC converter charge behavior, and sleep-state quiescent decay.

## Detailed Description

### System Architecture

The system comprises a voltage monitor sampling the auxiliary bus, a current estimator, a temperature input, and a processor performing event detection, feature extraction, baseline learning, and alerting. Three embodiments are described: (a) an OBD-II dongle measuring unswitched battery voltage directly at pin 16 of the diagnostic connector with a local analog-to-digital converter sampling at 100 Hz or higher during wake transients (1 kHz preferred for resolving sub-50 ms inrush events) and at 1 Hz during sleep; (b) a battery-terminal-mounted wireless monitor (Bluetooth Low Energy or similar) paired with a phone application that performs the analytics; and (c) an OEM or fleet-telematics integration consuming auxiliary-rail voltage already present on the vehicle CAN bus or in cloud telemetry. In all embodiments the analytics may run on the edge device, the phone, or a server, and all thresholds are learned per vehicle rather than fixed globally.

### Observable 1: Contactor-Closure Inrush Sag

When an electric vehicle wakes from sleep, the auxiliary battery alone must energize the high-voltage contactor coils before the DC-DC converter starts. Coil inrush draws tens of amperes for tens of milliseconds, producing a brief voltage sag whose depth is governed by the battery's internal resistance (sag voltage equals inrush current times internal resistance, to first order). This is the electric-vehicle analog of starter crank voltage: a repeatable, high-current, battery-only transient that occurs on nearly every drive cycle. The system detects wake events, isolates the contactor-closure window (typically the first 10 to 100 ms of bus activity, identifiable by its characteristic current step before DC-DC voltage regulation begins), and computes an internal-resistance estimate from the sag depth. Because inrush current varies with coil temperature and model, the absolute resistance value matters less than its trend: a doubling of estimated internal resistance relative to the per-vehicle baseline indicates end of life approaching. Sulfation, the dominant lead-acid aging mechanism, manifests directly as rising internal resistance and collapsing charge acceptance, so this observable tracks the primary failure mode.

### Observable 2: DC-DC Charge Acceptance Profiling

Once the vehicle is awake or charging, the DC-DC converter regulates the auxiliary bus toward an absorption setpoint (about 14.4 V for lead-acid, chemistry-dependent for lithium-ion). A healthy battery accepts substantial current at this voltage early in the drive and tapers gradually. A sulfated battery reaches the absorption voltage abnormally quickly while accepting little current, because its elevated internal resistance converts the voltage limit into a current limit prematurely; published sulfation research shows charge acceptance falling cycle by cycle as sulfate crystals accumulate, with the charger cutting off at its current threshold long before the battery holds meaningful energy. The system records, per drive or charge cycle, the time from DC-DC startup to absorption voltage, the integrated ampere-hours accepted, and the current taper slope. Declining acceptance at fixed absorption voltage, trended over weeks, provides a second independent degradation signal that requires no additional sensors.

### Observable 3: Quiescent Decay During Sleep

With the DC-DC converter off during sleep, parasitic loads (body controllers, telematics, keyless-entry receivers) drain the auxiliary battery, typically 20 to 50 mA in a healthy vehicle. The system measures open-circuit voltage decay over sleep periods of several hours or more, compensated for temperature using the battery's known voltage-temperature coefficient. A healthy battery of 40 to 50 Ah loses only a few hundredths of a volt over a 12-hour sleep at room temperature; a degraded battery with reduced capacity shows several times that decay for the same parasitic load. Because decay rate conflates capacity loss with parasitic current, the system maps the voltage trajectory through the chemistry's open-circuit-voltage versus state-of-charge curve to estimate the effective quiescent current, enabling the discrimination described below.

### Observable 4: Automatic Top-Up Interval

Many electric vehicles periodically wake the DC-DC converter during extended parking to replenish the auxiliary battery. Each top-up appears in the voltage log as a brief excursion to absorption voltage while the vehicle is otherwise asleep. As capacity fades, the battery reaches the top-up trigger voltage sooner, so the interval between top-ups shortens. This observable is valuable because it is coarse (no high-rate sampling needed), tolerant of sensor noise, and directly tied to usable capacity: it answers how long the battery sustains the vehicle's own sleep loads. The system logs top-up timestamps and trends the median interval over a rolling window.

### Fusion and Remaining-Useful-Life Estimation

During an initial calibration window (about 30 days of normal use, or about 50 wake cycles, whichever comes first), the system learns per-vehicle baselines for all four observables along with their temperature and state-of-charge dependencies. Thereafter a Bayesian changepoint detector (or equivalently a cumulative-sum control chart) flags statistically significant degradation in each channel. The channels are fused in a weighted health index, with weights adapted to which observables are available in each embodiment (for example, an OBD-II dongle parked on pin 16 sees all four; a CAN-based integration may lack high-rate inrush sampling and weight charge acceptance more heavily). The fused index is mapped to a remaining-useful-life estimate in weeks via a per-model degradation-rate prior that sharpens as the individual vehicle's trajectory accumulates. Three alert tiers result: advisory (schedule replacement at next service), service-soon (replace within weeks), and replace-now (risk of no-start on the next sleep-to-wake transition). Alerts are delivered through the phone application, the dongle's indicator, or the vehicle/telematics interface depending on embodiment.

### Parasitic Drain Discrimination

Not every dead auxiliary battery is a bad battery; faulty control modules can draw hundreds of milliamperes (documented cases show 200 mA drains killing batteries in one to two weeks of parking). Replacing the battery without addressing the drain guarantees recurrence. The system discriminates the two cases: if estimated quiescent current substantially exceeds the model-typical range while inrush-sag resistance and charge acceptance remain near baseline, it reports a probable parasitic drain fault and recommends electrical diagnosis rather than battery replacement. If all channels degrade together, it reports battery end of life. This distinction is a direct consequence of fusing the four observables rather than relying on voltage thresholds alone.

### Embodiments

**Aftermarket OBD-II dongle.** A self-contained dongle draws power from pin 16, samples auxiliary voltage with a dedicated ADC at 100 Hz or higher during detected wake transients (1 Hz otherwise to conserve power), and streams features over Bluetooth to a phone application performing baseline learning and alerting. No CAN decoding is required, making it vehicle-agnostic.

**Terminal-mounted monitor.** A ring-terminal or clamp-mounted module with BLE connectivity performs the same measurements at the battery posts, capturing inrush sag with minimal wiring impedance. The phone application is identical to the dongle embodiment.

**OEM and fleet integration.** The analytics run in the vehicle's body controller or in cloud telematics using auxiliary-rail voltage already reported on CAN or to the fleet backend. Aggregated anonymized per-model trajectories calibrate the degradation-rate priors and failure thresholds for each vehicle model, improving remaining-useful-life accuracy across the fleet without exposing individual vehicle data.

### Illustrative Worked Example

The following parameters are illustrative, not measured from any prototype. Consider a 45 Ah AGM auxiliary battery with a contactor-coil inrush of 30 A for 20 ms. When new, internal resistance of 6 milliohms yields a sag of about 0.18 V. After sulfation raises internal resistance to 25 milliohms, the same inrush produces a sag of about 0.75 V, more than four times the baseline and well above a 2.5-times-baseline replacement threshold. During driving, the healthy battery accepts over 10 A at the 14.4 V absorption setpoint early in the cycle; the degraded battery reaches 14.4 V within minutes at under 4 A, accepting a small fraction of the ampere-hours. During a 12-hour sleep at 20 C with 30 mA parasitic load, the healthy battery's terminal voltage falls less than 0.05 V while the degraded battery falls more than 0.15 V, and automatic top-up intervals shorten from roughly 72 hours to under 24 hours. Any single channel might be noisy; the fused trend across all four gives weeks of advance warning.

## Claims

1. A battery prognostics system for an electric vehicle having a high-voltage traction pack, a low-voltage auxiliary battery, at least one high-voltage contactor energized from the auxiliary battery, and a DC-DC converter, the system comprising: a voltage monitor configured to sample the auxiliary bus at 100 Hz or higher; a processor configured to detect contactor-closure events, compute an internal-resistance estimate from voltage sag during contactor-coil inrush, and trend the estimate against a per-vehicle baseline; and an alert output activated when the trended internal resistance exceeds a replacement threshold.
2. The system of claim 1, wherein the processor further profiles DC-DC converter charge acceptance, including time from converter startup to absorption voltage and ampere-hours accepted per drive cycle, as a sulfation indicator.
3. The system of claim 1, wherein the processor further measures temperature-compensated quiescent voltage decay during DC-DC-off sleep periods.
4. The system of claim 1, wherein the processor further tracks intervals between automatic DC-DC top-up events during extended parking, and wherein shortening top-up intervals indicate capacity fade.
5. The system of claim 3, wherein the processor estimates effective quiescent current from the voltage decay trajectory and reports a probable parasitic drain fault, distinct from battery end of life, when the estimated quiescent current exceeds a model-typical range while internal resistance and charge acceptance remain near baseline.
6. The system of claim 1, wherein the per-vehicle baseline is learned over an initial calibration window and a changepoint detector flags statistically significant deviation thereafter.
7. The system of claim 1, wherein the processor computes a remaining-useful-life estimate from the fused degradation trend and issues tiered alerts comprising advisory, service-soon, and replace-now levels.
8. The system of claim 1 embodied as an OBD-II dongle measuring unswitched battery voltage at pin 16 of the diagnostic connector.
9. The system of claim 1 embodied as a battery-terminal-mounted wireless monitor communicating with a phone application.
10. The system of claim 1 integrated with vehicle or fleet telematics using auxiliary-rail voltage reported on the vehicle network.
11. A method of predicting auxiliary battery failure in an electric vehicle without a starter motor, comprising: sampling auxiliary bus voltage at 100 Hz or higher; identifying contactor-closure inrush windows; computing internal resistance from voltage sag during the windows; trending the internal resistance against a per-vehicle baseline; and generating an alert before the auxiliary battery can no longer energize the high-voltage contactors.
12. The method of claim 11, further comprising aggregating anonymized per-model battery trajectories across a vehicle fleet to calibrate degradation-rate priors and failure thresholds.

## Implementation Notes

This disclosure describes a proposed design; no prototype has been built and no performance figures have been measured. Several practical considerations apply. Contactor-coil inrush current varies by vehicle model and coil temperature, so per-model calibration of the inrush window and per-vehicle baseline learning are required rather than fixed global thresholds. Reliable capture of a 10 to 100 ms inrush event requires a dedicated analog-to-digital converter sampling at 100 Hz or higher; voltage values polled over the OBD-II data link at typical scan rates are too slow and too aliased for this observable, which is why the dongle embodiment measures pin 16 directly. Lithium-ion auxiliary batteries (12 V or 16 V) exhibit flatter open-circuit-voltage curves and different failure modes (including abrupt battery-management-system cutoff) than lead-acid, so chemistry-specific voltage models and thresholds are needed; the inrush-sag resistance observable remains applicable because internal resistance growth precedes failure in both chemistries. The method cannot detect sudden internal faults such as a shorted cell that develop between wake cycles, and it is not a substitute for periodic conductance testing at service. Temperature compensation is essential because both internal resistance and open-circuit voltage shift substantially with temperature. Estimated quiescent current is indirect, derived from the voltage trajectory through the open-circuit-voltage curve, and should be treated as a screening signal for parasitic drain rather than a calibrated ammeter reading.

## Prior Art References

1. AutoNext, "Leapmotor CTC 3.0 Deletes 12V Lead Battery; ADAC: 12V Causes 45% of Breakdowns," September 2026, reporting ADAC's 2025 figures of 3.7 million breakdown calls with 45.4 percent caused by the 12-volt battery and higher failure rates in electric vehicles. <https://www.autonext.co/news/leapmotor-ctc-3-0-deletes-12v-lead-battery-45-percent-breakdowns-2027>
2. InsideEVs, "Tesla Switches To 12V Li-Ion Auxiliary Battery," covering Tesla's move from lead-acid to lithium-ion auxiliary batteries and reported 2-to-4-year lead-acid service life. <https://insideevs.com/news/546087/tesla-liion-12v-auxiliary-battery/>
3. Electrek, "Tesla Model S/X get rid of lead-acid 12V battery, moves to Li-ion," February 2021. <https://electrek.co/2021/02/02/tesla-model-s-x-get-rid-lead-acid-12v-battery-moves-li-ion/>
4. InsideEVs, "EV 12-Volt Battery Problems Are Leaving Owners Stranded," covering Hyundai/Kia ICCU failures, Chevrolet Bolt and Toyota bZ4X incidents. <https://insideevs.com/news/752720/ev-12-volt-battery-problems/>
5. MakeUseOf, "Why Do Full EVs Still Have a 12V Battery?" explaining the auxiliary battery's role in energizing high-voltage contactors and the DC-DC converter's charging function. <https://www.makeuseof.com/why-do-full-evs-still-have-12v-battery/>
6. TorqueNews, ID.Buzz owner stranded by dead 12-volt battery despite charged traction pack. <https://WWW.TORQUENEWS.COM/17998/i-was-stranded-my-idbuzz-dead-12v-battery-vws-hidden-battery-location-had-me-calling-help>
7. TorqueNews, Kia EV9 12-volt battery failure repeated three times in six months. <https://www.TorqueNews.com/17998/my-kia-ev9s-12-volt-battery-died-third-time-six-months-while-we-were-200-miles-home-despite>
8. KnowledgeMentors, Nissan Ariya 12V battery drain analysis, documenting approximately 200 mA parasitic draw versus acceptable thresholds. <https://knowledgementors.com/m-auto/afs/default/rsoc/784d5184-c759-449b-a996-6cd9eec72aa7/nissan-ariya-years-to-avoid-for-12v-battery-drain?channel=Ch_57506>
9. Batteries (MDPI), "Novel Test Procedure for Assessing Lead-Acid Batteries for Partial-State-of-Charge Duty Using Internal Resistance Charge Acceptance Technique," documenting progressive sulfation reducing charge acceptance cycle by cycle. <https://www.mdpi.com/2313-0105/11/4/131>
10. Journal of the Electrochemical Society (IOP), "Modeling of Sulfation in a Flooded Lead-Acid Battery and Prediction of its Cycle Life," modeling sulfation-driven resistance increase and capacity fade. <https://iopscience.iop.org/article/10.1149/1945-7111/ab679b>
11. EDN, "Dead Lead-acid Batteries: Desulfation-resurrection opportunities?" describing how sulfated batteries reach charger cutoffs rapidly while holding little energy. <https://www.edn.com/dead-lead-acid-batteries-desulfation-resurrection-opportunities/>
12. Not a Tesla App, "Tesla Starts Using Lithium-Ion 12 Volt Batteries," on the 16 V lithium-ion auxiliary transition. <https://www.notateslaapp.com/news/513/tesla-starts-using-lithium-ion-12-volt-batteries>
