# PA-2026-192: Non-Intrusive Appliance Load Disaggregation Using Optical Flicker Signatures Captured by Consumer Camera Devices

**Title:** System and Method for Non-Intrusive Appliance Load Disaggregation Using Optical Flicker Signatures Captured by Consumer Camera Devices

**Filing:** LITF-PA-2026-192
**Published:** October 3, 2026
**Domain:** Smart Home / Energy Monitoring / Computer Vision
**Full Disclosure:** [liveinthefuture.org/priorart/optical-flicker-appliance-disaggregation.html](https://liveinthefuture.org/priorart/optical-flicker-appliance-disaggregation.html)
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

Disclosed is a system and method for non-intrusive appliance load monitoring (NALM/NILM) that uses no electrical metering hardware whatsoever. A consumer camera with a rolling shutter is aimed at an existing electric lamp in the home. The lamp's luminous output flickers at twice the mains frequency (100 or 120 Hz), and the depth of that flicker tracks the instantaneous branch-circuit voltage: every appliance switching event, inrush transient, and steady-state load change on the branch imprints a step, dip, or drift on the flicker envelope. By extracting a per-sensor-row luminance time series from the rolling shutter at effective sampling rates of tens of kilohertz, demodulating the mains-frequency flicker component, and applying change-point detection to the resulting envelope, the system recovers the appliance event stream that a conventional power meter would report. Event features (envelope step magnitude, inrush dip depth and duration, overshoot) are matched against a per-household appliance signature library to identify the appliance, and an envelope-to-watt calibration maps signature strength to power and energy. Because no panel access, electrician, or plug-level sensor is required, the system brings NILM to rental units and retrofit scenarios where conventional approaches are blocked by installation friction.

## Technical Field

This disclosure relates to non-intrusive load monitoring for residential energy management, specifically to extracting appliance-level electrical load information from the optical flicker of existing lamps using consumer camera devices, without direct electrical measurement.

## Background

Non-intrusive appliance load monitoring (NALM), introduced by Hart (MIT, 1992), promised appliance-level energy breakdowns from a single meter. Three decades later the promise is still mostly unfulfilled in homes. The classical approach requires metering hardware at the electrical panel or a smart meter with second-level reporting, both of which face adoption friction: panel access requires an electrician or the homeowner's electrical confidence, rental tenants cannot touch shared panels, and utility smart meters in most deployments report 15-minute intervals that smear away the fast transients on which disaggregation depends. Plug-level monitors solve the per-appliance problem at the cost of one device per appliance, one pairing step each, and parasitic standby draw.

Meanwhile, an unmetered sensor channel already exists in every lit room. Incandescent, halogen, and many inexpensive LED lamps modulate their light output with the mains waveform: luminous flux follows instantaneous power, so the light carries a 100 or 120 Hz flicker component whose amplitude scales with the RMS branch voltage. When a 1.5 kW appliance switches on, the branch voltage sags by a few percent under the inrush current; that sag is visible as a transient dip in the lamp's flicker envelope, and the appliance's steady-state draw leaves the envelope sitting at a new level once the event settles. The optical channel thus encodes the same event stream as the electrical channel, with no contact to conductors and no panel access.

Related techniques extract information from lamp flicker but do not disaggregate loads. Rolling-shutter extraction of the electric network frequency (ENF) from lamp flicker is used for video forensics and grid-frequency estimation: it measures the phase and frequency of the flicker, treating its amplitude as a nuisance. Flicker-fusion and visible-light-communication research treats mains flicker as interference to be rejected. No existing method uses the flicker amplitude envelope, captured optically, as the signal for appliance disaggregation, and none exploits split-phase leg selectivity or per-appliance inrush characterization through this channel.

## Detailed Description

### 1. Optical coupling physics

For a resistive lamp, luminous flux follows approximately the 1.6 power of instantaneous voltage, producing full-wave flicker at twice the mains frequency with a modulation depth of 30 to 100 percent for incandescent and halogen lamps. Inexpensive LED lamps with capacitive-dropper or simple buck drivers also exhibit strong mains-locked modulation (often 40 to 90 percent), while LED lamps with active power-factor-correction stages are heavily filtered and modulate weakly (under 10 percent). A pre-screening step (Section 7) measures each visible lamp's modulation depth and selects lamps with the strongest coupling as sensing targets. The disclosure is agnostic to lamp technology as long as the measured modulation depth exceeds the camera's noise floor.

### 2. Rolling-shutter row-rate luminance acquisition

A rolling-shutter CMOS sensor exposes rows sequentially, so consecutive rows sample the scene at staggered times. For a 1080-row sensor running at 30 frames per second, the effective row rate is approximately 32,400 rows per second, giving a Nyquist limit far above the 120 Hz flicker fundamental and resolving its harmonics. The system fixes a region of interest (ROI) on the selected lamp, locks camera exposure and white balance (Section 8), and records the mean luminance of each sensor row within the ROI. Concatenating rows across frames yields a continuous luminance time series with sub-millisecond effective sampling from an ordinary camera.

### 3. Flicker demodulation

The row-rate luminance series is band-pass filtered around twice the nominal mains frequency (100 or 120 Hz, with the local grid frequency refined by a phase-locked loop on the flicker fundamental itself). The analytic signal via the Hilbert transform yields the flicker envelope, which is normalized by the slow-varying DC luminance to reject ambient light changes such as daylight drift or someone walking past the lamp. The output is an envelope time series at the camera frame rate (30 samples per second), fast enough to capture appliance events lasting longer than roughly 100 ms, while the raw row-rate series is retained in a short ring buffer for transient characterization at full resolution.

### 4. Event detection by change-point analysis

Appliance switching produces step changes in the envelope. The system applies a cumulative-sum (CUSUM) change-point detector tuned to the noise floor of the envelope, supplemented by a matched-filter bank whose templates are the household's previously observed event shapes. On detection, the system replays the ring-buffered row-rate series around the event timestamp to extract the inrush transient: dip depth (percent), dip duration (milliseconds), and post-inrush overshoot or ringing. These transient features discriminate motor-driven appliances (deep, long inrush dips, 50 to 500 ms) from resistive heaters (sharp steps, minimal overshoot) and electronic loads (shallow steps with high-frequency hash).

### 5. Appliance signature library and classification

Each event is reduced to a feature vector: envelope step magnitude, inrush dip depth, inrush duration, overshoot amplitude, envelope noise-floor shift during the on-period, and time-of-day context. The per-household signature library is bootstrapped in two ways: supervised calibration, in which the user confirms a small number of events through a guided "turn each appliance on and off" wizard; and unsupervised clustering, in which recurring event shapes are grouped and presented to the user for labeling. A classifier (k-nearest neighbors or random forest over the feature space) assigns subsequent events to appliances. The library is per-household because lamp coupling, branch impedance, and appliance models all vary by home.

### 6. Envelope-to-watt calibration and energy estimation

Relative event strengths are converted to watts by a one-time calibration against the utility meter or a clamp meter: the user (or an installer) records total-load steps during the guided wizard and regresses envelope step magnitude against measured watts, yielding a per-branch coupling coefficient in watts per percent of envelope change. Alternatively, nameplate ratings of identified appliances anchor the scale. Labeled events are then integrated over time to produce per-appliance energy estimates in kilowatt-hours, with daily, weekly, and monthly breakdowns and cost estimates using the local tariff.

### 7. Split-phase leg identification

Residential service in North America is split-phase: two 120 V legs (L1, L2) with 240 V appliances across both. An appliance on one leg sags that leg's voltage far more than the other leg's, so a lamp on the same leg as the appliance shows a larger envelope step than a lamp on the opposite leg. With lamps visible on both legs (e.g., two lamps in different rooms, or a stereo camera pair), the system compares envelope steps across ROIs: a symmetric step on both legs identifies a 240 V appliance (dryer, oven, water heater), while an asymmetric step identifies a 120 V appliance and its leg. This leg attribution is a disaggregation feature unavailable to single-point electrical meters and substantially reduces the ambiguity of simultaneous events on different legs.

### 8. Robustness: exposure lock, dimmer rejection, daylight

Camera auto-exposure would fight the measurement, so exposure, gain, and white balance are locked after an initial metering pass. Daylight is a slow DC term removed by the DC normalization in Section 3. Phase-cut dimmers inject their own chopped-waveform flicker at variable phase; the system detects dimmer-modulated lamps by their non-sinusoidal flicker spectrum and excludes them as sensing targets, optionally flagging them for the user. Multiple lamp ROIs vote on each event, suppressing false positives from someone briefly occluding one lamp.

### 9. Privacy-preserving design

The camera needs to see the lamp, not the room. Only the per-row mean luminance of each ROI is ever retained; full frames are processed in memory and discarded, and no image content leaves the device unless the user opts into cloud backup of aggregated statistics. The ROI is chosen to cover the lamp shade or bulb and exclude faces, screens, and documents. All disaggregation runs on-device; the utility-meter calibration in Section 6 is the only step that ingests external data, and it ingests a scalar wattage, not a waveform.

### 10. Fault and degradation detection

Because the system continuously records inrush transients, it detects drift in an appliance's signature over months: a refrigerator compressor drawing a progressively deeper inrush dip signals winding degradation; a furnace blower whose step magnitude grows signals bearing wear or a clogged filter. When a tracked appliance's features drift beyond a per-appliance tolerance band, the system issues a maintenance alert with the measured trend, turning the optical NILM channel into a predictive-maintenance sensor at zero additional hardware cost.

## Claims

1. A system for non-intrusive appliance load disaggregation, comprising: a rolling-shutter camera device aimed at an electric lamp; a region-of-interest selector configured to isolate the lamp's luminous area; a row-rate luminance extractor configured to produce a luminance time series at the sensor's row rate; a flicker demodulator configured to isolate the mains-frequency flicker component and compute its amplitude envelope; an event detector configured to detect appliance switching events from change points in the envelope; and a classifier configured to assign detected events to appliances by matching event features against a per-household signature library, wherein no electrical metering hardware is used.

2. The system of claim 1, wherein the row-rate luminance extractor concatenates per-row mean luminance values within the region of interest across frames, achieving an effective sampling rate substantially above the camera's frame rate and resolving the mains-frequency flicker harmonics.

3. The system of claim 1, wherein the event detector applies cumulative-sum change-point detection to the flicker envelope, supplemented by a matched-filter bank of previously observed household event shapes.

4. The system of claim 1, further comprising an inrush-characterization module configured to replay a ring buffer of raw row-rate luminance around each detected event and extract inrush dip depth, dip duration, and post-inrush overshoot as classification features.

5. The system of claim 1, further comprising an envelope-to-power calibration module configured to regress envelope step magnitudes against measured total-load steps from a utility or clamp meter, yielding per-branch coupling coefficients that convert envelope changes to watts.

6. The system of claim 1, further comprising a split-phase leg discriminator configured to compare envelope steps across regions of interest on lamps served by different supply legs, identifying 240 V appliances by symmetric leg response and 120 V appliances by asymmetric leg response.

7. The system of claim 1, further comprising a lamp pre-screening module configured to measure the mains-locked modulation depth of each visible lamp and select sensing targets whose modulation depth exceeds the camera's noise floor, excluding lamps with phase-cut dimmer modulation.

8. The system of claim 1, wherein camera exposure, gain, and white balance are locked after metering, and the envelope is normalized by slow-varying DC luminance to reject daylight drift and ambient occlusion.

9. The system of claim 1, wherein only region-of-interest luminance statistics are retained, full frames are discarded in memory, and all disaggregation executes on-device, preserving the privacy of the monitored space.

10. A method of building a per-household optical appliance signature library, comprising: guiding a user through a wizard that switches individual appliances on and off; recording envelope step and inrush features for each confirmed event; clustering unlabeled recurring event shapes; and presenting clusters to the user for labeling.

11. A method of appliance fault detection, comprising: tracking an appliance's inrush and envelope features over time; detecting drift beyond a per-appliance tolerance band; and issuing a maintenance alert with the measured trend, the drift indicating winding degradation, bearing wear, or airflow restriction.

12. The system of claim 1, wherein the camera device is an already-installed security camera, video doorbell, or baby monitor repurposed for load monitoring, the lamp region of interest being selected from its existing field of view.

## Implementation Notes

Exposure locking is the single most important implementation detail: any camera that is allowed to auto-expose will cancel the very signal being measured, so the metering pass must run before locking and the system should verify lock by confirming that the flicker fundamental is visible in a test capture. Lamp selection matters more than camera quality: an incandescent lamp or a cheap capacitive-dropper LED gives a far stronger signal than a high-CRI PFC-corrected LED, and the pre-screening step should rank candidates by measured modulation depth and let the user pick. The guided calibration wizard is the difference between a research demo and a product: ten minutes of turning appliances on and off yields a labeled library that unsupervised clustering alone cannot reach in a house with twenty loads. For split-phase discrimination, the user only needs to tell the wizard which lamp sits on which breaker; an incorrect assignment shows up immediately as 120 V appliances that look like 240 V ones, and the wizard should detect and correct this. Inrush replay from the ring buffer is what separates motor from resistive loads, so size the buffer for at least one second of row-rate data.

## Limitations

This method measures branch-circuit voltage through a lamp, so it inherits the lamp's limitations: heavily filtered LED drivers with active power-factor correction produce modulation depths below the noise floor and cannot serve as sensing targets. Phase-cut dimmers distort the flicker spectrum and are excluded. Two appliances switching within the same detection window produce a superimposed step that the classifier may mislabel; the split-phase discriminator in Section 7 resolves coincidences across legs but not within one leg. Absolute watt accuracy depends on the one-time calibration against a meter and drifts if the lamp is replaced with a different driver technology, so the wizard should re-run after lamp changes. The method estimates electrical behavior from an optical proxy; it cannot distinguish two identical appliance models on the same branch without user labeling.

## Prior Art References

1. Hart, G. W., "Nonintrusive appliance load monitoring," Proceedings of the IEEE, vol. 80, no. 12, 1992: the original NILM formulation using electrical metering at the service entry; all subsequent work in the field assumes an electrical measurement channel.
2. Rolling-shutter electric network frequency (ENF) extraction from lamp flicker for video forensics: measures the phase and frequency of mains-coupled flicker, treating flicker amplitude as a nuisance term rather than the signal.
3. LITF-PA-2026-190, "System and Method for Grid Frequency Estimation Using Rolling-Shutter Flicker Analysis": optical ENF for grid frequency; the present disclosure targets load disaggregation from flicker amplitude, a different measurand and application.
4. LED flicker modulation studies (IEEE 1789-2015, recommended practices for modulating current in high-brightness LEDs): document the relationship between driver topology and mains-locked luminous modulation depth used by the lamp pre-screening step.
5. Page, E. S., "Continuous inspection schemes," Biometrika, 1954 (CUSUM change-point detection): the statistical basis for the envelope event detector.
