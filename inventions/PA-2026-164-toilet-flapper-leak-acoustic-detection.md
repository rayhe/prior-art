# PA-2026-164: Silent Toilet Flapper Leak Detection via Non-Intrusive Acoustic Fill-Event Sensing

**Title:** System and Method for Detecting and Quantifying Silent Toilet Flapper Leaks Using Non-Intrusive Acoustic Fill-Event Sensing

**Filing:** LITF-PA-2026-164
**Published:** September 7, 2026
**Domain:** Water Conservation / Acoustic Sensing
**Full Disclosure:** [liveinthefuture.org/priorart/toilet-flapper-leak-acoustic-detection.html](https://liveinthefuture.org/priorart/toilet-flapper-leak-acoustic-detection.html)

---

## Abstract

      Disclosed is a system and method for detecting, quantifying, and alerting on silent flapper leaks in gravity-flush toilets without any intrusion into the plumbing. A low-cost acoustic sensor strapped to the toilet supply line (or angle-stop valve body) detects fill-valve acoustic events, segments them into full-flush refills, short top-up fills, and continuous-fill fault states, and applies periodicity analysis to the top-up event train. A flapper leak produces a metronomic signature: short top-ups recurring at near-constant intervals, including overnight hours when no flushes occur, because the tank level falls at the leak rate until the fill valve's hysteresis band trips a refill. The system estimates leak rate from the inter-arrival period and a calibrated top-up volume, converts waste to gallons per day and estimated cost, differentiates flapper leaks from fill-valve seal failures and supply-line leaks, and issues escalating alerts. An optional motorized shutoff valve closes the supply when accumulated waste crosses a threshold. Fleet learning across deployed units builds valve acoustic signature clusters and flapper survival models for replacement-interval guidance.

## Field of the Invention

      This invention relates to residential water conservation and plumbing diagnostics, specifically to non-intrusive detection of silent toilet leaks through acoustic sensing of fill-valve events on the fixture supply line, with statistical periodicity analysis to distinguish leak-driven refills from normal use.

## Background

      Toilets are the largest indoor water user in the typical home, and they are also the most common source of silent, long-duration leaks. The failure mechanism is mundane: the rubber flapper that seals the tank to the bowl warps, mineralizes, or chains-tangles over 3 to 7 years of service, and tank water seeps continuously into the bowl. The leak is silent because it never reaches the bowl with enough force to make noise, and it is invisible because the water vanishes down the drain it was destined for anyway. The only observable symptom is the fill valve periodically topping up the tank, a sound most occupants either do not hear or do not recognize.

      The scale of the waste is large. The U.S. Environmental Protection Agency's WaterSense program notes that a continuously running toilet can waste 200 gallons or more per day, and household leaks nationally waste nearly one trillion gallons of water annually ([EPA WaterSense, Fix a Leak Week](https://www.epa.gov/watersense/fix-leak-week)). A moderate flapper leak of 0.5 liters per minute, 720 liters (190 gallons) per day, can persist for months between water bills, adding hundreds of dollars in water and sewer charges per billing cycle while contributing nothing to any human use.

      Existing detection methods are manual, intrusive, or whole-home:

- **Dye-tablet test:** EPA WaterSense recommends dropping food coloring into the tank and watching for color in the bowl without flushing. It works but requires a human to remember to perform it, catches only the leak present at test time, and provides no quantification or monitoring.
- **In-tank electronic detectors:** Products such as the LeakAlertor hang inside the tank and sense water-level drops, flashing or chirping an alert. They require opening the tank, placing hardware in chlorinated water (which degrades electronics and conflicts with in-tank cleaning tablets), run on batteries in a hostile environment, and report a binary leak/no-leak state with no leak-rate estimate.
- **Whole-home flow disaggregation:** Services such as Flume ([flumewater.com](https://www.flumewater.com/)) and Phyn ([phyn.com](https://www.phyn.com/)) monitor flow at the meter or main and disaggregate fixture-level use. They require meter access or a plumber-installed sensor, solve the general disaggregation problem rather than the toilet-specific one, and cannot always separate a periodic toilet top-up train from other low-flow signatures.
- **Municipal acoustic leak detection:** Water utilities use acoustic correlators on distribution mains ([AWWA](https://www.awwa.org/) water-loss control practice) to find pressurized pipe leaks. These are correlator-based, crew-deployed instruments for buried infrastructure, not consumer devices for in-home fixtures.

      The lineage of non-intrusive sensing, founded by Hart (Proc. IEEE 1992) for electrical load disaggregation, has never been applied to the specific, high-value problem of the silent toilet flapper leak using the one signal that is always available without touching the water: the sound of the fill valve. The gap in the art is a non-intrusive, fixture-level system that: (a) senses fill events acoustically from outside the plumbing, (b) distinguishes leak-driven top-ups from normal flushes by their statistical periodicity rather than by any single event's appearance, (c) quantifies the leak rate and its cost, and (d) differentially diagnoses the three common failure modes (flapper leak, fill-valve seal failure, supply-line leak) from the same event stream.

      **Non-obviousness.** The combination is not suggested by the references, and several teach away from it. Municipal acoustic correlators are two-sensor, crew-deployed instruments for buried pressurized mains; nothing in that art suggests a single strap-on consumer sensor listening to one fixture's fill valve. In-tank electronic detectors teach placing hardware inside chlorinated tank water, directly away from exterior acoustic sensing. Whole-home flow disaggregation (Flume, Phyn) teaches solving the general multi-fixture disaggregation problem at the meter, not exploiting a single fixture's fill-valve hysteresis periodicity. Hart's NILM operates on electrical current waveforms, a different physical domain. No reference discloses the deterministic relation T = V_h / Q between the fill valve's hysteresis volume and the top-up inter-arrival period, nor the insight that a metronomic top-up train persisting through nocturnal low-use hours is a unique signature of a flapper leak that no human behavior produces. The result is a specific, fixture-level, quantitative diagnostic, not a mere automation of the dye-tablet test.

## Detailed Description

### 1. Sensing Hardware

      The preferred embodiment is a strap-on sensor module clamped to the toilet's flexible supply line or the angle-stop valve body, upstream of the fill valve. The sensing element is a piezoelectric disc (27 mm, unit cost under $1) or a contact-coupled MEMS microphone pressed against the pipe wall with a silicone couplant pad. Pipe-borne structure-borne sound from the fill valve conducts efficiently through copper, PEX, and braided stainless supply lines, so the sensor never contacts water and installation requires no tools, no plumber, and no tank access.

      The module contains: the acoustic pickup; an analog front end with a bandpass filter (800 Hz to 6 kHz, the band where turbulent fill-valve hiss concentrates); a microcontroller with ADC (e.g., ESP32-C3 class, unit cost under $3) sampling at 16 kHz with 12-bit resolution (satisfying Nyquist for the 6 kHz band edge); and a BLE 5 radio reporting event metadata (not audio) to a phone or home hub. Power is supplied by two AA cells; with the duty-cycled architecture of Section 2, expected battery life exceeds 12 months at typical household event rates. Target bill-of-materials cost: $8 to $12 per unit.

      An alternative embodiment uses the microphone of a nearby always-on smart speaker or a phone placed in the bathroom for a spot-check survey, applying the same signal chain to air-conducted sound. A further embodiment integrates the pickup into a smart angle-stop replacement valve installed at fixture rough-in.

### 2. Signal Acquisition and Fill-Event Detection

      The microcontroller computes the RMS envelope of the bandpassed signal in 100 ms windows. A fill event is declared when the envelope exceeds an adaptive threshold (median background plus 12 dB) for a minimum continuous duration of 2 seconds, and the event ends when the envelope falls below the threshold for more than 3 seconds (bridging brief valve chatter). This yields a timestamped event list with per-event duration and mean envelope energy.

      To conserve power, the ADC and processor sleep between envelope checks: a low-power analog comparator wakes the MCU when bandpassed energy crosses a coarse threshold, and the MCU then validates the event with the full envelope detector. False wakes from speech, shower noise, or HVAC are rejected by the 2-second minimum duration and by the spectral flatness test of Section 3, since human speech and fan noise do not sustain the broadband turbulence spectrum of a fill valve.

### 3. Event Classification

      Each detected event is classified into one of four categories using duration, envelope shape, and spectral features:

- **Full-flush refill (40 to 120 s):** long fill following a complete tank dump. Envelope shows a fast attack, a sustained plateau, and a characteristic two-stage decay as the tank nears full and the valve throttles, followed by a short bowl-refill trickle. These events are human-driven and aperiodic.
- **Top-up fill (3 to 25 s):** short fill restoring only the hysteresis band of the fill valve (typically 0.2 to 0.5 L depending on valve model and supply pressure). Envelope is a clean plateau with symmetric attack and decay. When these recur periodically, they are the flapper-leak signature.
- **Continuous fill (tens of minutes to indefinite):** the valve never closes. Indicates fill-valve seal failure, a stuck float, or a severely displaced flapper holding the tank perpetually below the shutoff level.
- **Partial/dual-flush refill (15 to 40 s):** intermediate duration from reduced-volume flushes. Classified as normal use and excluded from leak analysis.

      A spectral flatness check confirms the event is valve turbulence (flat broadband spectrum, crest factor under 6 dB) rather than impulsive plumbing noise such as water hammer (high crest factor, decaying resonant tones), which is logged separately as a pipe-stress indicator.

### 4. Periodicity Analysis and Leak Confirmation

      This is the core of the invention. A single top-up event is ambiguous: it could follow a partial flush, a cat drinking from the bowl, or evaporation. A *train* of top-ups with near-constant inter-arrival time is not ambiguous, because no human behavior produces it.

      The physics is deterministic. With the flapper leaking at rate Q (volume per time) and the fill valve's hysteresis band holding volume V_h between its open and close levels, the tank level falls at rate Q until the valve trips, refills V_h, and the cycle repeats with period T = V_h / Q. For a typical V_h of 0.3 L and a leak of 0.1 L/min, T is 3 minutes; the toilet performs roughly 480 identical top-ups per day, each about 10 seconds long, totaling 144 liters (38 gallons) per day. At 0.5 L/min the period collapses to 36 seconds and daily waste reaches 720 liters (190 gallons), matching the EPA's worst-case figure.

      The leak detector therefore maintains a rolling 72-hour buffer of top-up inter-arrival times and declares a confirmed flapper leak when all of the following hold:

- **Run length:** at least 6 consecutive top-up events with no intervening full-flush refill.
- **Period stability:** coefficient of variation (standard deviation divided by mean) of the inter-arrival times at or below 0.25, and of the event durations at or below 0.35.
- **Nocturnal persistence:** at least 2 qualifying top-ups in the 00:00 to 05:00 local window, when human flushes are rare. A daytime-only periodic train is flagged as suspected but not confirmed, since a dripping faucet into the tank or a houseguest pattern could mimic it.
- **Exclusion of continuous fill:** no event in the buffer exceeds 10 minutes, which would instead indicate the fill-valve failure mode of Section 6.

      The confirmation logic is deliberately conservative: it trades one to three days of detection latency for near-zero false positives, because an alert that cries leak during normal guest-heavy use will be ignored or uninstalled.

### 5. Leak-Rate Quantification and Cost Estimation

      Once confirmed, the leak rate is estimated as Q = V_h / T, where T is the median inter-arrival period and V_h is the calibrated top-up volume. V_h is obtained at commissioning: after installation, the system observes the next full-flush refill, measures its duration, and divides the fixture's rated tank volume (1.6 gal / 6.1 L for post-1992 fixtures per the Energy Policy Act of 1992 and [ASME A112.19.2](https://www.asme.org/); user-adjustable for older 3.5 gal fixtures) by the measured fill time to obtain the supply flow rate in liters per second. Multiplying by the median top-up duration yields V_h. The system re-calibrates on every observed full flush, tracking supply-pressure drift over time.

      Daily waste is Q multiplied by 1,440 minutes, reported in liters and gallons. The companion app converts waste to estimated cost using the local combined water-plus-sewer rate (user-entered or looked up by ZIP code from a utility rate table), and projects the cost to the next billing cycle. Alert tiers: **advisory** above 40 L/day (~10 gal/day, roughly a dollar a week), **action** above 150 L/day, and **urgent** above 400 L/day or any continuous-fill state.

### 6. Differential Diagnosis

      The same event stream distinguishes the three common failure modes without additional sensors:

- **Flapper leak:** periodic short top-ups meeting the Section 4 criteria. Remedy: replace the $5 flapper.
- **Fill-valve seal failure or stuck float:** continuous fill event exceeding 10 minutes, often with the tank visibly discharging through the overflow tube. The envelope never decays to background. Remedy: replace or clean the fill valve (fill-valve performance is standardized under ASSE 1002).
- **Supply-line leak upstream of the valve:** continuous low-level broadband hiss with *no* fill cycling, because the leak is downstream of the sensor but the tank level never falls enough to trip the valve, or the leak is upstream of the sensor entirely. Cross-check: if the home has a meter-level monitor, flow continues with zero fill events. The app advises checking the supply line and angle stop for moisture. Remedy: tighten or replace the supply line.

      Water-hammer transients captured by the spectral check of Section 3 are reported as a separate pipe-stress indicator, since repeated hammer accelerates both flapper and valve-seat wear, closing a diagnostic loop the disclosure is designed to catch early.

### 7. Commissioning and Baseline Learning

      On installation the unit enters a 72-hour learning mode: it records all events, clusters them by duration into the Section 3 categories, estimates the supply flow rate from the first observed full flush, and measures the household's baseline flush cadence. The periodicity detector is armed after learning completes, using the learned flush cadence to set the nocturnal-persistence window adaptively (e.g., shifting the quiet window for night-shift households).

      A manual "dye-test mode" in the app guides the user through the classic food-coloring test while the sensor records, providing ground-truth labels that validate the acoustic classifier on that specific fixture and valve model.

### 8. Alerting and Automatic Shutoff

      Alerts are delivered via the companion app and, where integrated, the home hub: push notification at the advisory tier with the estimated daily waste and projected bill impact, escalating in tone and frequency at the action and urgent tiers. Each alert names the diagnosed failure mode and the specific $5 to $20 part that fixes it, because a leak alert without a remedy is just anxiety.

      An optional embodiment pairs the sensor with a motorized ball valve on the supply line. When accumulated waste since confirmation crosses a user-set threshold (default 2,000 L), the valve closes the supply and notifies the user, with a manual override at the valve and in the app. A "vacation mode" arms automatically when the sensor observes 48 hours with zero flush events, closing the valve until the first manual reopen, eliminating the catastrophic-supply-line-failure risk during travel as a side benefit.

### 9. Fleet Learning

      Deployed units report anonymized per-event feature vectors (duration, envelope statistics, spectral flatness; never raw audio, per the privacy design of Section 10) to a cloud service. Clustering these vectors by valve acoustic signature identifies fill-valve models in the field, allowing the classifier to load model-specific hysteresis volumes and expected refill envelopes instead of learning them from scratch. Survival analysis over the fleet fits flapper replacement-interval curves as a function of water hardness (from user ZIP code) and valve model, producing per-household "replace your flapper in N months" guidance before the leak starts. This converts the system from a leak detector into a leak prevention program.

### 10. Implementation Notes

- **Privacy by design:** raw audio never leaves the sensor module and is not stored; only event timestamps, durations, and aggregate spectral statistics are transmitted. The 800 Hz high-pass cutoff excludes most speech energy, and the envelope detector cannot reconstruct intelligible audio.
- **Multi-toilet homes:** one sensor per toilet; the hub correlates across fixtures and reports per-bathroom waste.
- **Pressure-assisted and dual-flush fixtures:** pressure-assisted tanks produce a distinct impulsive flush signature and are classified separately; dual-flush partial flushes fall in the intermediate duration band and are excluded from leak analysis, though a leaking dual-flush seal produces the same periodic top-up train and is detected identically.
- **Tankless and wall-hung systems:** out of scope; the invention targets the gravity-flush tank fixtures that constitute the large majority of installed residential toilets.
- **Power budget:** the comparator-gated wake architecture keeps average current under 50 microamps; two AA alkaline cells provide over a year of operation at typical household flush cadences.
- **Limitations:** the method cannot detect leaks downstream of the flapper seal that bypass the tank entirely (e.g., a cracked bowl leaking to the floor); those present as water damage, not fill events. Extremely slow leaks (under ~5 L/day) produce top-up intervals longer than the 72-hour analysis buffer can confirm and are reported as unconfirmed suspicion only.

### 11. Figures Description

- **Figure 1:** Strap-on sensor module clamped to a toilet supply line at the angle stop, with callouts for the piezo pickup, MCU, BLE radio, and battery compartment.
- **Figure 2:** Envelope waveforms for the four event classes: full-flush refill with two-stage decay, short top-up plateau, continuous-fill fault, and partial-flush intermediate event.
- **Figure 3:** 24-hour event timeline contrasting a healthy toilet (irregular daytime flushes, silent night) with a flapper-leak toilet (metronomic top-up train persisting through the night), with the inter-arrival histogram showing the low-CV peak.
- **Figure 4:** System block diagram: acoustic pickup, analog front end, comparator-gated MCU, event classifier, periodicity analyzer, leak quantifier, alerting path, and optional shutoff actuator.
- **Figure 5:** Fleet survival curves for flapper replacement interval versus water hardness, derived from aggregated anonymized deployments.

## Claims

1. A system for detecting silent leaks in a gravity-flush toilet, comprising: a non-intrusive acoustic sensor configured to couple to an exterior surface of the toilet's water supply line or angle-stop valve body without contacting the water; a processor configured to detect fill-valve acoustic events from a signal of the sensor, classify each event by duration and envelope shape into at least full-flush refill events and top-up fill events, accumulate a sequence of the top-up fill events, and apply periodicity analysis to the sequence, the periodicity analysis comprising determining inter-arrival times of the sequence; wherein the processor declares a flapper leak when the inter-arrival times of a run of the top-up fill events exhibit a coefficient of variation at or below a threshold and the run persists into a low-use time window.
2. The system of claim 1, wherein the acoustic sensor comprises a piezoelectric disc or contact microphone coupled to the supply line through a compliant couplant pad, and wherein the processor computes an RMS envelope of a bandpass-filtered signal in a range of approximately 800 Hz to 6 kHz to detect turbulent fill-valve flow.
3. The system of claim 1, wherein the periodicity analysis requires a minimum run length of consecutive top-up fill events, a coefficient of variation of the inter-arrival times at or below 0.25, and at least two qualifying events within a nocturnal low-use window, and wherein a daytime-only periodic train is reported as suspected rather than confirmed.
4. The system of claim 1, further comprising a leak-rate quantifier configured to estimate a leak rate as a calibrated top-up volume divided by a median inter-arrival period of the sequence, and to convert the leak rate to a daily waste volume and a projected water cost using a local utility rate.
5. The system of claim 4, wherein the calibrated top-up volume is derived by measuring a duration of an observed full-flush refill event, dividing a rated tank volume by the duration to obtain a supply flow rate, and multiplying the supply flow rate by a median top-up duration, with re-calibration on each subsequently observed full-flush refill event.
6. The system of claim 1, further comprising a differential diagnosis module configured to distinguish a flapper leak, indicated by a periodic top-up train, from a fill-valve seal failure or stuck float, indicated by a continuous fill event exceeding a duration threshold with no envelope decay, and from a supply-line leak, indicated by a continuous broadband hiss with no fill cycling, and to report a corresponding remedy for each diagnosed condition.
7. The system of claim 1, wherein the processor employs a comparator-gated wake architecture in which an analog comparator wakes a sleeping microcontroller only when bandpassed acoustic energy crosses a coarse threshold, and the microcontroller then validates each event with an envelope-duration test and a spectral flatness test that rejects speech, HVAC, and impulsive water-hammer noise.
8. The system of claim 1, further comprising a motorized shutoff valve on the supply line configured to close when an accumulated estimated waste since leak confirmation crosses a user-set threshold, and a vacation mode that closes the motorized shutoff valve automatically after a period with no observed flush events.
9. The system of claim 1, further comprising a fleet learning service that receives anonymized per-event feature vectors from a plurality of deployed units, clusters fill-valve acoustic signatures to identify fill-valve models, and fits flapper survival models as a function of water hardness to generate per-household preventive replacement guidance.
10. A method for detecting silent toilet flapper leaks without plumbing intrusion, comprising: acoustically sensing an exterior surface of a toilet supply line; detecting fill-valve events via a bandpassed RMS envelope detector; classifying the events into full-flush refill events, top-up fill events, and continuous-fill fault events by duration and envelope shape; accumulating a multi-day buffer of inter-arrival times of the top-up fill events; confirming a flapper leak when a run of the top-up fill events exhibits a low coefficient of variation of the inter-arrival times and nocturnal persistence; estimating a leak rate from a median inter-arrival period and a calibrated top-up volume; and issuing a tiered alert identifying a failure mode, an estimated daily waste, and a replacement part.

## Prior Art References

1. [EPA WaterSense, Fix a Leak Week](https://www.epa.gov/watersense/fix-leak-week): Household leak waste statistics; running toilets wasting 200+ gallons per day; dye-tablet test method
2. LeakAlertor: In-tank electronic water-level leak detector (intrusive, binary alert, no quantification)
3. [Flume](https://www.flumewater.com/): Whole-home flow monitoring via meter-mounted sensor with fixture disaggregation
4. [Phyn](https://www.phyn.com/): Whole-home pressure-wave-based leak detection and flow disaggregation
5. Hart, G.W., "Nonintrusive appliance load monitoring," Proc. IEEE 1992: Foundational non-intrusive sensing via aggregate signal disaggregation
6. [AWWA](https://www.awwa.org/) water-loss control practice: Municipal acoustic correlator leak detection on distribution mains (buried infrastructure, crew-deployed)
7. ASSE 1002: Performance standard for anti-siphon fill valves (ballcocks)
8. ASME A112.19.2: Vitreous china plumbing fixtures; 1.6 gpf water-consumption basis
9. Energy Policy Act of 1992: 1.6 gallons-per-flush maximum for residential toilets
