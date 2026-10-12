# PA-2026-201: Quantifying Sediment Accumulation in Residential Storage Water Heaters Using Passive Smartphone Acoustic Analysis

**Title:** System and Method for Quantifying Sediment Accumulation in Residential Storage Water Heaters Using Passive Smartphone Acoustic Analysis

**Filing:** LITF-PA-2026-201
**Published:** October 12, 2026
**Domain:** Home Maintenance / Appliance Diagnostics / Acoustics
**Full Disclosure:** [liveinthefuture.org/priorart/water-heater-sediment-passive-acoustic-quantification.html](https://liveinthefuture.org/priorart/water-heater-sediment-passive-acoustic-quantification.html)
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

Disclosed is a system and method for quantifying mineral sediment accumulation inside residential storage-type water heaters using only the microphone of a commodity smartphone, with no installed hardware, no contact sensors, and no tank draining. The application records audio in the vicinity of the tank during heating cycles and extracts three sediment-correlated acoustic signals. First, steam-bubble collapse transients: water trapped beneath the sediment layer flash-boils when the burner or element fires, and the escaping steam bubbles burst through the mineral crust as impulsive broadband bursts, the familiar popping and kettling. A transient detector counts these events and measures their cumulative energy per heating cycle. Second, water-column resonance shift: the tank's water column supports longitudinal acoustic modes whose frequencies and Q-factors change as the growing sediment layer alters the effective acoustic depth and the bottom boundary impedance, tracked against a per-tank baseline established at installation or after a flush. Third, heating-cycle duration: burner or element on-time per recovery lengthens as sediment insulates the heat source, and this on-time is recovered directly from the audio envelope. The three signals are fused into a sediment index mapped to estimated sediment depth and lost tank capacity, and threshold crossings produce a flush-now alert with an estimated efficiency penalty.

## Technical Field

This disclosure relates to appliance condition monitoring, specifically to passive acoustic diagnostics of residential storage water heaters using commodity smartphone microphones, spectral analysis, and longitudinal trending to quantify internal mineral sediment accumulation without installed sensors.

## Background

Residential storage water heaters are among the largest energy consumers in the home, and mineral sediment is their most common silent efficiency thief. Hard water is widespread across the United States. When heated, dissolved calcium and magnesium precipitate out and settle at the tank bottom, forming a mineral crust. In gas units the burner fires directly beneath this layer; in electric units the lower element becomes coated. The U.S. Department of Energy recommends flushing storage tanks annually. Almost nobody does.

The failure physics are audible. Water trapped beneath the sediment layer superheats and flash-boils, and steam bubbles bursting through the crust produce the popping, rumbling, and kettling sounds plumbers have diagnosed by ear for decades (The Spruce; EngineerFix; South Surrey Plumbing). Meanwhile the sediment insulates the heat source, forcing longer heating cycles, higher energy bills, overheating of components, and eventually premature tank failure (Better Homes & Gardens; Norcan Heating and Air).

Today's diagnostics are qualitative or after-the-fact. A plumber listens and judges. A homeowner who finally drains the tank sees cloudy water and infers the past. No method gives a homeowner a number: how many inches of sediment, how many gallons of capacity lost, how much efficiency sacrificed, and whether a flush is due now or can wait a year. Instrumented alternatives exist for adjacent problems. LITF-PA-2026-157 discloses anode-rod depletion monitoring via a bonded piezoelectric exciter, Hall-effect galvanic current sensing, and steel-shell resonance tracking. That system targets corrosion and wall thinning with installed hardware. Sediment accumulation is a different failure mode with different acoustics, and nothing in the art measures it passively with a phone alone.

The gap in the art is a complete quantitative method that: (a) measures sediment from sound alone with no hardware installation; (b) separates sediment signatures from other tank noises; (c) calibrates per tank via a post-flush baseline; and (d) converts the measurement into a flush decision with an efficiency cost attached.

## Detailed Description

### 1. Capture protocol

The user places the phone within roughly one meter of the tank, microphone facing the tank, on a stable surface, and starts a diagnostic listen; alternatively the app schedules captures automatically. Recording covers at least one full heating cycle, typically 3 to 15 minutes at 44.1 or 48 kHz. In gas units the app detects burner ignition from the combustion roar onset. In electric units it detects the contactor click and the 120 Hz element hum. Sessions are tagged by fuel type, tank capacity as entered by the user, and ambient noise level. Captures with sustained background noise above a threshold, from a nearby HVAC system or garage door operation, are flagged low-confidence rather than silently trusted.

### 2. Steam-bubble transient detection

Each bubble collapse is an impulsive event: 5 to 50 milliseconds long, broadband with peak energy between roughly 500 Hz and 4 kHz, fast attack and exponential decay. The detector runs a spectral-flux onset function on 20 ms frames with 50 percent overlap, gated by high-frequency band energy, against an adaptive threshold derived from the session's own noise floor. Per heating cycle the module reports transient count, cumulative transient energy, median peak amplitude, and inter-arrival time statistics. Clean tanks produce near-zero counts. The count rate grows monotonically with sediment depth in the regime where bubbles can still escape.

A fully cemented sediment bed that no longer passes bubbles reads quiet. The fusion stage handles this edge case by weighting the resonance and cycle-duration signals more heavily at high index values. This behavior is stated explicitly because a naive transient-only detector would call the worst tanks clean.

### 3. Rumble envelope analysis

Dense bubble trains merge into the low-frequency rumbling characteristic of gas units: amplitude modulation of the 40 to 200 Hz band at a few hertz. The module computes the envelope of the low band via Hilbert transform and examines its modulation spectrum. Strong modulation energy between 1 and 8 Hz, coherent with burner-on periods, is scored as the rumble component. Electric units show a different texture. Scale on the element produces sharper crackling, which the transient detector of Section 2 captures rather than the rumble path.

### 4. Water-column resonance tracking

The water inside the tank forms an acoustic cavity. Its longitudinal modes sit near f_n = n·c / 2L for mode number n, sound speed c (about 1500 to 1550 m/s in hot water), and water-column height L. A 50-gallon tank with roughly 1.4 m of water puts the fundamental near 540 Hz, inside the phone microphone's clean range. As sediment accumulates it shortens the effective water column and replaces the rigid steel bottom with a lossy granular boundary. Both change the observed modes: frequencies shift and Q-factors drop as the boundary absorbs more energy per cycle.

This disclosure claims no universal direction for the frequency shift. The lossy-boundary loading and the shortened column push in opposite directions, and their balance depends on tank geometry. Instead the system records the fundamental frequency and Q-factor at a baseline session, at installation or after a flush, and tracks deviations session over session. A drift exceeding the session-to-session noise floor, sustained across three or more sessions, is scored as the resonance component.

### 5. Heating-cycle duration from the audio envelope

Sediment insulates the heat source, so each recovery takes longer. The module detects burner or element on/off transitions from the audio envelope, combustion roar for gas and 120 Hz hum plus contactor clicks for electric, and records on-time per cycle. Absolute on-time depends on how much hot water was drawn, so the system compares only usage-matched cycles: overnight recoveries with no draws, or cycles preceded by the same draw signature. Long-baseline detrending separates the slow sediment-driven lengthening from seasonal inlet-temperature variation. A 15 percent lengthening of matched recovery cycles is the default advisory threshold, and thirty percent triggers the flush-now tier.

### 6. Sediment index fusion and thresholds

The three components normalize to z-scores against the tank's own baseline and combine into a single sediment index from 0 to 100. Laboratory calibration against tanks of known sediment depth maps index ranges to estimated sediment depth in inches and lost capacity in gallons. A one-inch layer in a 50-gallon tank displaces roughly 2 to 3 gallons, and the index reports that conversion with its uncertainty band.

Action follows two thresholds, starting with advisory (index 40): sediment measurable, efficiency penalty estimated at 5 to 10 percent, flush recommended within the year. Flush-now (index 70): popping rate above 30 transients per minute during heating, or matched recovery cycles lengthened beyond 30 percent, or resonance drift beyond the noise floor on all three components; estimated penalty above 15 percent. The alert names the evidence. Example: "87 popping events in a 6-minute burn; recovery 34 percent longer than your March baseline."

### 7. Differential diagnosis

Not every tank noise is sediment. Aluminum anode rods in high-alkalinity water produce a sizzling present from installation day, constant rather than growing; the trending module rejects signals with no upward slope. Water hammer is a single sharp transient tied to valve closure, not a heating-cycle phenomenon. A hissing temperature-and-pressure relief valve is a safety issue the app flags separately and refuses to fold into the sediment score. Loose electric elements produce a 120 Hz buzz modulated by thermal expansion that tracks element temperature, not sediment depth. Each confound has a distinct temporal and spectral signature, and the fusion stage down-weights components when a confound's signature dominates.

### 8. Figures description

- **Figure 1:** System block diagram showing the smartphone, capture module, the three analysis paths (transient detection, resonance tracking, cycle-duration trending), the fusion module, and the alert output.
- **Figure 2:** Example transient waveform and spectrogram of steam-bubble collapse pops recorded during a gas burner cycle, with the spectral-flux onset function overlaid.
- **Figure 3:** Water-column fundamental frequency and Q-factor drift over 18 months spanning a flush event, showing reset to baseline.
- **Figure 4:** Matched recovery-cycle duration trend with seasonal detrending, showing the sediment-driven lengthening component.
- **Figure 5:** Sediment index dial with advisory and flush-now thresholds, annotated with estimated depth, lost capacity, and efficiency penalty.

## Claims

1. A system for quantifying mineral sediment accumulation in a residential storage water heater, comprising: a smartphone microphone configured to record audio in the vicinity of the tank during at least one heating cycle, with no sensor installed on or inside the tank; a transient detection module that identifies steam-bubble collapse transients as impulsive broadband bursts and reports a per-cycle transient count and cumulative transient energy; a resonance tracking module that measures longitudinal acoustic modes of the tank's water column and tracks their frequency and Q-factor deviation from a per-tank baseline; a cycle-duration module that detects heating-element on/off transitions from the audio envelope and trends on-time per recovery across sessions; and a fusion module that combines the transient, resonance, and cycle-duration signals into a sediment index and issues a maintenance alert when the index crosses a threshold.

2. The system of claim 1, wherein the transient detection module applies a spectral-flux onset function to 20 ms frames with 50 percent overlap, gated by band energy between 500 Hz and 4 kHz, against an adaptive threshold derived from the session noise floor, identifying events of 5 to 50 ms duration.

3. The system of claim 1, further comprising a rumble analysis module that computes the amplitude-modulation envelope of the 40 to 200 Hz band and scores modulation energy between 1 and 8 Hz coherent with heating-element-on periods.

4. The system of claim 1, wherein the resonance tracking module estimates longitudinal mode frequencies near f_n = n·c/2L and records the fundamental frequency and Q-factor at a baseline session, scoring subsequent sessions by sustained deviation beyond the session-to-session noise floor without asserting a universal direction of frequency shift.

5. The system of claim 1, wherein the cycle-duration module detects burner ignition and extinction from combustion-roar envelope transitions in gas units and from contactor clicks plus 120 Hz hum transitions in electric units.

6. The system of claim 5, wherein the cycle-duration module compares only usage-matched recovery cycles and applies long-baseline detrending to separate sediment-driven lengthening from seasonal inlet-temperature variation.

7. The system of claim 1, wherein the fusion module maps the sediment index to an estimated sediment depth and lost tank capacity using a calibration against tanks of known sediment depth, reporting the estimate with an uncertainty band.

8. The system of claim 1, further comprising a two-tier alert module issuing an advisory alert at a first index threshold with an estimated efficiency penalty and a flush-now alert at a second, higher threshold, each alert naming the measured evidence supporting it.

9. The system of claim 1, further comprising a differential diagnosis module that rejects non-sediment acoustic confounds including constant-from-installation anode sizzle, valve-closure water hammer transients, temperature-and-pressure relief valve hiss, and thermally modulated loose-element buzz, by their distinct temporal and spectral signatures.

10. The system of claim 1, wherein all audio processing occurs on the smartphone and no audio leaves the device, with only the sediment index, component scores, and timestamps retained for trending.

11. The system of claim 1, further comprising a longitudinal trending module that stores per-session component scores with capture metadata and distinguishes progressive sediment accumulation from static installation conditions by the slope of the component trajectories.

12. A method for quantifying mineral sediment accumulation in a residential storage water heater, comprising: recording audio near the tank during at least one heating cycle using a smartphone microphone with no installed sensors; detecting steam-bubble collapse transients and computing a per-cycle transient count and cumulative energy; measuring longitudinal water-column resonance modes and their deviation from a per-tank baseline; detecting heating-cycle on-time from the audio envelope and trending it across usage-matched cycles; fusing the transient, resonance, and duration signals into a sediment index; and issuing a flush alert when the index crosses a threshold.

## Implementation Notes

Placement matters. The phone sits within about one meter of the tank, microphone facing the tank, on a stable surface such as a shelf or tripod. Handheld recording adds handling noise that the adaptive threshold only partly rejects. Garages and utility closets are noisy environments; the low-confidence flagging in Section 1 exists so a bad session degrades gracefully instead of corrupting the trend.

The single most valuable session is the post-flush baseline. Without it the index still functions, because the transient count is meaningful in absolute terms, but the depth mapping and the resonance component lose precision. The app therefore prompts for a baseline capture immediately after the user confirms a flush.

The method estimates total accumulation, not mineral composition. Calcium carbonate, magnesium deposits, and sand behave alike for the prescription, which is to flush. Tankless, heat-pump, and solar-thermal units are out of scope; storage gas and electric units only.

The system is implementable as a smartphone app using commodity hardware and open-source libraries, with no custom sensors. Spectral-flux onset detection and FFT analysis use the platform's native DSP libraries (Accelerate on iOS, FFTW or the NDK on Android). A three-minute 48 kHz capture with the described frame processing runs in well under real time on modern phone SoCs when analysis is restricted to the three bands of interest.

Privacy is by design: audio is processed on-device and never uploaded; only the sediment index, component scores, calibration metadata, and timestamps are stored for trending.

This document describes a proposed design, not a validated engineering method: thresholds, band assignments, and calibration curves here are design parameters for implementers to verify against measured data, not certified inspection criteria.

## Prior Art References

1. [The Spruce: water heater popping noises](https://www.thespruce.com/water-heater-problem-making-noises-4103881): Steam bubbles forming under sediment burst as the water heats, producing popping sounds
2. [EngineerFix: water heater rumbling](https://engineerfix.com/why-is-my-water-heater-making-a-rumbling-noise/): Trapped water flash-boils under the sediment layer; steam bubbles burst through, creating rumbling and popping ("kettling")
3. [South Surrey Plumbing: water heater noises](https://www.southsurreyplumbing.com/blog/water-heater-noises-popping-rumbling-hissing/): Popping and rumbling point to sediment at the tank bottom; steam bubbles pushing through the layer produce the sound
4. [Better Homes & Gardens: why to flush a water heater](https://www.bhg.com/why-to-flush-water-heater-11894331): Sediment insulates heating elements, reduces efficiency, raises energy bills, and shortens lifespan
5. [Norcan Heating and Air: sediment buildup science](https://www.norcanheatingandair.com/water-heater-maintenance-sediment-buildup/): Sediment acts as an insulating barrier hindering heat transfer; regular flushing mitigates efficiency loss
6. [LITF-PA-2026-157](https://liveinthefuture.org/priorart/water-heater-anode-depletion-acoustic-galvanic.html): Anode-rod depletion monitoring via bonded piezoelectric exciter, Hall-effect galvanic current sensing, and steel-shell resonance tracking; targets corrosion and wall thinning with installed hardware, distinct from passive sediment acoustics
7. Leighton, T. G., "The Acoustic Bubble," Academic Press, 1994: Acoustic emissions of collapsing vapor bubbles, broadband transient character
8. Kinsler, L. E., Frey, A. R., Coppens, A. B., and Sanders, J. V., "Fundamentals of Acoustics," Wiley: Longitudinal resonant modes of fluid-filled cavities
9. U.S. Department of Energy, Energy Saver program: Annual flushing recommended for storage water heaters to remove sediment and maintain efficiency
