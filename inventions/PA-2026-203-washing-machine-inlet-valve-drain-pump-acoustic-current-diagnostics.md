# PA-2026-203: Diagnosing Washing Machine Inlet Valve Degradation, Drain Pump Cavitation, and Unbalanced Spin Loads Using Passive Acoustic and Electrical Current Signature Fusion

**Title:** System and Method for Diagnosing Washing Machine Inlet Valve Degradation, Drain Pump Cavitation, and Unbalanced Spin Loads Using Passive Acoustic and Electrical Current Signature Fusion

**Filing:** LITF-PA-2026-203
**Published:** October 14, 2026
**Domain:** Appliances / Acoustic Diagnostics / Predictive Maintenance
**Full Disclosure:** [liveinthefuture.org/priorart/washing-machine-inlet-valve-drain-pump-acoustic-current-diagnostics.html](https://liveinthefuture.org/priorart/washing-machine-inlet-valve-drain-pump-acoustic-current-diagnostics.html)
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

Disclosed is a system and method for passive, longitudinal health monitoring of residential washing machines using a commodity smartphone placed in the laundry area, optionally fused with plug-level electrical current from a smart plug. The phone's microphone and accelerometer capture each wash cycle. An on-device segmenter divides the cycle into phases (fill, agitation, drain, spin, rinse) from acoustic envelope and spectral features. Three diagnostic cores operate on the segmented phases. First, inlet valve health is measured as fill-phase duration normalized by load mass, where load mass is estimated from agitation-phase motor current or acoustic slosh energy; drift of the normalized fill duration against a per-machine rolling baseline indicates inlet screen clogging or solenoid plunger wear. A water-hammer transient analysis at valve closure provides a second, independent valve signal: a worn solenoid closes more slowly and softens the pipe transient. Hot and cold inlet lines are scored separately by comparing temperature-select cycles. Second, drain pump health is measured from drain-phase duration drift, impeller blade-pass harmonic energy, broadband cavitation noise, and optional current ripple at pump rotation frequency; the onset pattern distinguishes a clogged pump filter from impeller wear from a kinked drain hose. Third, spin imbalance and suspension wear are measured from structure-borne vibration at spin rotation frequency via the phone accelerometer, with per-load severity scoring and chronic-imbalance counting over a rolling window; longitudinal tracking of spin-phase acoustic sidebands serves as a drum bearing wear indicator. Tiered advisories name their evidence. All processing runs on the device; no audio leaves the phone.

## Technical Field

This disclosure relates to appliance prognostics, specifically to passive health monitoring of residential washing machines using fused acoustic and electrical current signatures, unsupervised wash-cycle phase segmentation, and per-machine baseline trending to diagnose inlet valve degradation, drain pump cavitation, and unbalanced spin loads before failure.

## Background

Washing machines fail in expensive ways. On the supply side, an inlet valve solenoid that sticks or a supply hose that bursts floods the laundry room; water damage from appliance failures ranks among the most frequent and costly homeowners insurance claims. On the drain side, a clogged pump filter or a failed drain pump strands water in the drum and soaks the floor. Mechanically, chronic unbalanced spins hammer suspension rods and drum bearings until the machine walks across the floor or the bearing seizes. In every case, the homeowner's first notice is the failure itself: an error code, a puddle, or a dead machine.

Existing approaches detect faults, not degradation. Manufacturer diagnostic features (for example, LG's Smart Diagnosis) transmit the machine's own onboard fault codes as DTMF tones to a phone; that is a readout of failures the machine has already detected, not passive trending of wear before failure. Non-intrusive load monitoring research, originating with Hart's 1992 work on current-signature disaggregation, identifies which appliance is running from electrical signatures but does not fuse acoustic phase structure to diagnose mechanical health. Acoustic appliance monitoring has been disclosed for adjacent machines, including passive acoustic sediment quantification in water heaters (LITF-PA-2026-201), refrigerator compressor prognostics, and pump current-signature diagnostics for well and sump pumps. Nothing in the art passively trends washing machine inlet valve, drain pump, and imbalance health from fused acoustic and current signatures with per-machine baselines, per-phase normalization, and temperature-line-separated valve scoring.

Filling the gap is a complete sensorless method that: (a) segments each wash cycle into acoustic phases without user input; (b) normalizes fill duration by estimated load mass so valve health is comparable across loads; (c) scores hot and cold inlet valves separately from temperature-select cycles; (d) extracts an independent solenoid wear signal from the water-hammer transient at valve closure; (e) diagnoses drain pump condition from blade-pass harmonics, cavitation noise, and onset pattern; (f) quantifies spin imbalance per load and tracks chronic imbalance and bearing wear longitudinally; and (g) reports tiered advisories that name the measured evidence.

## Detailed Description

### 1. Sensing Configuration

The primary sensor is a commodity smartphone placed in the laundry area, on a shelf or wall-mounted holder within approximately 3 meters of the machine with the microphone unobstructed. The phone records monaural audio at 16 kHz and triaxial acceleration at 100 Hz. An optional smart plug on the washer's outlet reports RMS current at 1 Hz or higher and fuses with the acoustic record. Cycle capture is triggered by acoustic onset (sustained broadband energy above the room baseline for a design-parameter duration) or, in plug-fused mode, by current exceeding the standby draw. Enrollment takes the first approximately 10 cycles as a provisional baseline, flagged as such in the user interface until the baseline stabilizes.

All processing runs on the device. Audio is analyzed in short frames and discarded after feature extraction; no raw audio is retained beyond the current cycle buffer, and no audio leaves the phone. Laundry-room cycle audio carries no speech content of diagnostic interest and none is stored or transmitted. This on-device constraint is structural, not cosmetic: the method's value is longitudinal trending of one household's machine, which requires no cloud.

### 2. Wash-Cycle Phase Segmentation

Each captured cycle is divided into phases by an on-device segmenter using acoustic envelope, spectral centroid, and periodicity features, implemented as a hidden Markov model or rule-based state machine over frame-level features (design choice, unvalidated). The phase inventory is: fill (broadband hiss, elevated 2-6 kHz energy, no periodicity), agitation or tumble (periodic slosh bursts at the stroke or reversal rate), drain (pump whine plus water gurgle, broadband with a tonal component), spin (rising tonal whine with strong structure-borne vibration), and rinse (a second fill-agitate-drain subsequence). A cycle-type classifier labels the cycle as normal, heavy-duty, quick, or delicates from the phase sequence and relative phase durations; health indices are compared only within cycle type.

Top-load and front-load machines present different acoustic signatures: top-load agitators or impellers produce impulsive slosh bursts, while front-load drums produce tumble reversals with a different periodicity. At enrollment the user selects the machine type, or the app selects it from the agitation-phase periodicity signature; the segmenter then uses the matching phase model. Misclassified machine type degrades segmentation, so the enrollment selection is confirmed against the first three cycles and corrected if the observed periodicity disagrees.

### 3. Inlet Valve Diagnostics

Fill-phase duration is measured from fill onset to the valve-closure transient. Raw fill duration is meaningless across loads: a large load legitimately takes longer to fill. The method therefore normalizes by load mass. In plug-fused mode, load mass is estimated from agitation-phase motor current RMS, which rises with the mass being moved. In phone-only mode, load mass is estimated from agitation-phase acoustic slosh energy, which similarly scales with load. The Fill Performance Index is fill duration divided by the load estimate, further conditioned on the temperature-select setting. A rolling 20-cycle median of the index per cycle type forms the per-machine baseline; sustained drift beyond a design-parameter threshold indicates flow restriction at the inlet.

Hot and cold inlet lines are scored separately. Washers fill through independent hot and cold solenoid valves with separate inlet screens. Cycles run with hot selected fill predominantly through the hot valve; cold-selected cycles fill through the cold valve. Comparing the Fill Performance Index drift between temperature selections isolates which valve is degrading. This differential measurement is the method's primary defense against confounds like falling household water pressure, which would degrade both lines together and is reported as a pressure advisory rather than a valve advisory.

A second, independent valve signal comes from the water-hammer transient. When a solenoid valve snaps shut, the sudden flow stoppage launches a pressure transient that rings the supply pipes as an audible thunk. A healthy solenoid closes crisply: the transient has a fast rise, a fast decay, and a spectral centroid set by the pipe run. A worn solenoid plunger closes more slowly and less completely: the transient softens, its decay lengthens, and its centroid shifts downward. The app detects the closure transient in the fill-phase tail, measures its decay time constant and spectral centroid, and trends them per temperature line. Screen clogging and solenoid wear produce distinguishable joint signatures: clogging shows gradual flow reduction on one temperature line with an unchanged transient (the valve still snaps shut, it just passes less water), while solenoid wear shows transient softening with comparatively less flow reduction. The advisory names which mechanism the evidence supports, because the remedies differ: clean the inlet screens for clogging, replace the valve for solenoid wear.

Hard water regions clog inlet screens faster than valves wear out; the method treats screen clogging as the default first hypothesis and the advisory instructs screen cleaning before valve replacement. Screens are inexpensive, user-serviceable parts behind the inlet hoses, and cleaning them resolves the majority of fill-performance drift in hard-water households.

### 4. Drain Pump Diagnostics

Drain-phase duration is measured from drain onset (pump whine onset) to the gurgle tail-off, normalized by load mass as in Section 3, and trended against the per-machine baseline. The pump's acoustic spectrum carries the mechanical signal. The impeller blade-pass frequency equals pump RPM multiplied by impeller blade count; healthy pumps concentrate energy at the blade-pass fundamental and its harmonics. Three measurements are extracted per drain phase: the harmonic energy ratio (energy at blade-pass harmonics relative to total drain-phase energy), the broadband cavitation noise level in a design-parameter band above the blade-pass harmonics, and, in plug-fused mode, current ripple amplitude at the pump rotation frequency.

Onset pattern classifies the failure mode. A clogged pump filter (lint, coins, debris at the pump inlet) produces gradual broadband noise rise and gradual drain lengthening over many cycles, because the restriction accumulates. A kinked or restricted drain hose produces sudden onset, often correlated with the machine having been moved, with preserved harmonic structure but elevated broadband noise. Impeller wear or damage produces decaying harmonic energy with relatively stable broadband levels, because the impeller moves water less effectively per revolution without adding restriction noise. The classifier reports the mode whose onset pattern the evidence matches, with the corresponding action: clean the pump filter, inspect the drain hose routing, or schedule pump service.

Detergent over-sudsing confounds drain acoustics: excess suds produce prolonged high-frequency hiss and extended drain phases that mimic restriction. A suds-lock detector flags drain phases where hiss energy is elevated without the corresponding harmonic decay of impeller wear, and reports over-sudsing as a detergent dosing advisory rather than a pump advisory.

### 5. Spin Imbalance, Suspension Wear, and Drum Bearing Diagnostics

During the spin ramp and steady spin, the phone accelerometer measures structure-borne vibration transmitted through the floor or wall. Imbalance severity for a load is computed as narrowband vibration energy at the drum rotation frequency, normalized by the load mass estimate from Section 3 (a heavy balanced load vibrates more than a light balanced load, and normalization prevents false imbalance flags). Each spin phase receives a per-load severity score. A single severe event is reported as informational with the likely cause class (single heavy item, such as a rug or comforter, versus distributed imbalance), because one bad load is user behavior, not machine wear.

Chronic imbalance is what damages machines. The method counts severe-imbalance events over a rolling 30-cycle window; sustained elevation of the count indicates the suspension system (rods, springs, or dampers, depending on machine type) is no longer centering the drum, and the advisory recommends suspension inspection. Separately, longitudinal tracking of spin-phase acoustic sidebands serves as a drum bearing wear indicator: as bearings degrade, modulation sidebands grow around the rotation tone at bearing defect frequencies. Sideband energy is trended over months against the enrollment baseline; slow upward drift with stable imbalance counts points to bearing wear rather than suspension wear, because worn bearings generate tonal modulation even in balanced loads.

### 6. Baselines, Normalization, and Confounds

Every health index is referenced to a per-machine rolling baseline, not a population norm, because plumbing pressure, pipe material, floor construction, phone placement, and machine model all shift absolute values. The baseline absorbs the water-hammer transient's dependence on pipe length and material, the vibration transmission of the particular floor, and the acoustic character of the particular laundry room. What the method detects is change, not absolute level.

Shared-wall apartments present cross-talk: a neighbor's machine may be audible through the wall. In plug-fused mode, cross-talk is rejected by requiring acoustic phase onsets to correlate with the local plug's current steps; a neighbor's fill hiss produces no local current step and is ignored. In phone-only mode, the enrollment procedure captures a spectral fingerprint of the local machine's phases, and cycles whose phase spectra do not match the fingerprint are excluded from trending with a notice to the user.

### 7. Outputs and Advisories

- **Per-failure-mode health index:** a 0-100 score with trend arrow for inlet valve (hot and cold separately), drain pump, suspension, and drum bearing, each computed from its section's measurements against the per-machine baseline.
- **Tiered advisories:** Monitor (early drift, watch the trend), Maintenance due (actionable now: clean inlet screens, clean pump filter, redistribute loads, reduce detergent), and Service recommended (sustained degradation indicating part replacement). Every advisory names the measured evidence, for example: "Cold fill duration up 18% over 30 normal cycles with unchanged closure transient: inlet screen clogging likely. Clean the cold inlet screen."
- **Cycle log export:** timestamped per-cycle phase durations, health indices, and flagged events in a portable format for a service technician, so the homeowner arrives with data instead of a vague complaint.
- **Pressure advisory:** when both temperature lines degrade together, the app reports suspected household water pressure change rather than valve failure, and suggests checking the pressure regulator.

### 8. Figures Description

- **Figure 1:** System configuration showing smartphone placement in the laundry area relative to the washer, optional smart plug on the washer outlet, and the on-device processing pipeline from audio and current capture through phase segmentation to the three diagnostic cores.
- **Figure 2:** Example segmented cycle: acoustic envelope and spectral centroid over time with phase boundaries marked (fill, agitation, drain, spin, rinse), plus the plug current trace aligned beneath.
- **Figure 3:** Example (illustrative): Fill Performance Index trend over 60 cycles showing gradual drift on the cold line only, with the valve-closure transient waveforms (healthy crisp transient versus softened worn transient) inset.
- **Figure 4:** Example (illustrative): drain-phase spectra for healthy, clogged-filter, and worn-impeller conditions showing blade-pass harmonic energy versus broadband cavitation noise, and a spin vibration severity timeline distinguishing single-event imbalance from chronic imbalance.

## Claims

1. A system for passively monitoring washing machine health, comprising: a smartphone with a microphone and an accelerometer placed in the laundry area; optionally a smart plug measuring the washer's electrical current; and on-device software that captures wash cycles, segments each cycle into acoustic phases, computes per-phase health indices for inlet valve, drain pump, and spin mechanical condition, trends the indices against a per-machine rolling baseline, and outputs tiered advisories naming the measured evidence, wherein no audio leaves the phone.

2. The system of claim 1, wherein the on-device software segments each cycle into fill, agitation, drain, spin, and rinse phases from acoustic envelope, spectral centroid, and periodicity features, and classifies the cycle type from the phase sequence so that health indices are compared only within cycle type.

3. The system of claim 1, further comprising an inlet valve diagnostic module that measures fill-phase duration, estimates load mass from agitation-phase motor current or acoustic slosh energy, computes a load-normalized Fill Performance Index, and detects flow restriction from drift of the index against the per-machine baseline.

4. The system of claim 3, wherein the inlet valve diagnostic module scores hot and cold inlet lines separately by comparing Fill Performance Index drift across temperature-select cycles, thereby isolating which solenoid valve is degrading and distinguishing valve degradation from household water pressure change.

5. The system of claim 3, further comprising a water-hammer transient analysis module that detects the valve-closure acoustic transient in the fill-phase tail, measures its decay time constant and spectral centroid, and infers solenoid plunger wear from transient softening, distinguishing screen clogging (flow reduction with unchanged transient) from solenoid wear (transient softening with less flow reduction).

6. The system of claim 1, further comprising a drain pump diagnostic module that measures drain-phase duration drift normalized by load mass, extracts impeller blade-pass harmonic energy and broadband cavitation noise from the drain-phase acoustic spectrum, and, in plug-fused mode, measures current ripple at pump rotation frequency.

7. The system of claim 6, further comprising an onset-pattern classifier that distinguishes a clogged pump filter (gradual broadband rise and gradual drain lengthening), a kinked or restricted drain hose (sudden onset with preserved harmonic structure), and impeller wear (decaying harmonic energy with stable broadband levels).

8. The system of claim 6, further comprising a suds-lock detector that flags drain phases with elevated high-frequency hiss without harmonic decay as detergent over-sudsing and reports a dosing advisory rather than a pump advisory.

9. The system of claim 1, further comprising a spin imbalance module that computes per-load imbalance severity from narrowband accelerometer vibration energy at drum rotation frequency normalized by load mass, and counts severe-imbalance events over a rolling window to distinguish single-event user-caused imbalance from chronic suspension wear.

10. The system of claim 9, further comprising a drum bearing wear module that longitudinally trends spin-phase acoustic sideband energy around the rotation tone against the enrollment baseline, wherein slow upward drift with stable imbalance counts indicates bearing degradation.

11. The system of claim 1, further comprising a cross-talk rejection module that, in plug-fused mode, requires acoustic phase onsets to correlate with local plug current steps, and in phone-only mode, excludes cycles whose phase spectra do not match the enrollment spectral fingerprint.

12. A method for passively monitoring washing machine health, comprising: placing a smartphone with a microphone and accelerometer in the laundry area; capturing wash-cycle audio and vibration; segmenting each cycle into fill, agitation, drain, spin, and rinse phases from acoustic features; computing load-normalized fill duration, drain-phase pump spectral features, and spin vibration severity; trending the measurements against a per-machine rolling baseline; and outputting tiered advisories naming the measured evidence.

13. The method of claim 12, further comprising scoring hot and cold inlet valves separately via temperature-select cycle comparison and analyzing the valve-closure water-hammer transient to distinguish inlet screen clogging from solenoid plunger wear.

## Implementation Notes

**Scope of this disclosure:** This document is a design proposal. No prototype has been built and no field validation has been performed; all quantitative parameters in this disclosure (baseline window lengths, drift thresholds, frequency bands, sampling rates, and classifier architecture) are untested design choices.

Phone placement dominates signal quality. The microphone must have a clear path to the machine; a phone buried in a laundry basket hears the basket. The app should verify placement at enrollment by checking fill-hiss signal-to-noise against the room baseline and prompt the user to move the phone if the margin is poor. Accelerometer coupling matters equally: a phone on a shelf bolted to studs couples differently than one on a hollow-core door. The per-machine baseline absorbs placement specifics, but the phone should stay in the same place across cycles, and the app should detect placement changes (sudden baseline shifts across all indices simultaneously) and re-baseline rather than report phantom degradation.

The water-hammer signal is the most placement- and plumbing-sensitive measurement in the disclosure. Transient character depends on pipe material (copper rings brighter than PEX), pipe run length, and arrestor presence. None of that matters for trending, because the baseline is per-installation, but it means transient features are not comparable across households and should never be presented as absolute scores.

Top-load and front-load machines differ enough to need separate phase models, as noted in Section 2. High-efficiency top-loaders without center agitators (impeller plates) sit between the two acoustically; the enrollment periodicity check should place them correctly, and implementers should validate the segmenter on all three configurations before claiming coverage.

Bearing sideband trending is a months-scale signal, not a diagnostic code. The advisory copy must say "bearing wear suspected, trending upward over N cycles" and not "bearing failure in 30 days." The same honesty applies to every index: these are drift detectors with named evidence, not failure predictors with dates.

Misuse is foreseeable in one direction: a landlord could in principle use cycle logs to monitor tenant laundry behavior. The cycle log export exists for the machine owner's technician; the app should not offer remote multi-machine fleet dashboards without the explicit consent framework of whoever owns the machines. Nothing in this disclosure authorizes monitoring of machines the user does not own or operate.

## Prior Art References

1. Hart, G. W., "Nonintrusive appliance load monitoring," Proceedings of the IEEE, 1992: Current-signature disaggregation identifying which appliance is running; does not fuse acoustic phase structure for mechanical health diagnosis
2. LG Electronics Smart Diagnosis: DTMF-tone transmission of onboard fault codes to a phone; a readout of machine-detected faults, not passive degradation trending
3. LITF-PA-2026-201: Passive acoustic sediment quantification in water heaters; adjacent appliance, different machine and failure modes, no cycle-phase segmentation
4. Refrigerator compressor acoustic prognostics (LITF prior art): Acoustic health trending for a different appliance; no inlet valve, drain pump, or imbalance diagnostics
5. Well pump short-cycling diagnostics (LITF prior art): Pump current-signature diagnostics for a different pump application; no acoustic fusion
6. Sump pump current drainage health (LITF prior art): Plug-level current health monitoring for sump pumps; single-phase operation, no wash-cycle phase structure
7. Association of Home Appliance Manufacturers (AHAM): Clothes washer test procedures and performance standards; laboratory measurement, not in-home passive monitoring
