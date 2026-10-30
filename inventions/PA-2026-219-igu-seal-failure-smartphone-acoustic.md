# PA-2026-219: Detecting Insulated Glass Unit Seal Failure Using Smartphone-Based Acoustic Tap Resonance Analysis

**Title:** System and Method for Detecting Insulated Glass Unit Seal Failure Using Smartphone-Based Acoustic Tap Resonance Analysis

**Filing:** LITF-PA-2026-219
**Published:** October 30, 2026
**Domain:** Windows / Acoustic Diagnostics / Building Envelope
**Full Disclosure:** [liveinthefuture.org/priorart/igu-seal-failure-smartphone-acoustic.html](https://liveinthefuture.org/priorart/igu-seal-failure-smartphone-acoustic.html)
**License:** [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/) - Public Domain

> Prior Art Notice: This document is a defensive technical disclosure published
> to constitute prior art under [35 U.S.C. Sec. 102(a)(1)](https://www.law.cornell.edu/uscode/text/35/102),
> dedicated to the public domain as of the publication date above.

---

## Abstract

Disclosed is a system and method for detecting perimeter seal failure in insulated glass units (IGUs) using nothing more than a consumer smartphone. The user taps the glass pane with a fingertip at a marked point and the phone's microphone records the resulting ring. A sealed IGU is two glass plates coupled through a hermetic gas cavity: the trapped argon acts as a pneumatic spring that binds the panes into characteristic in-phase and anti-phase vibration modes with high decay quality factors. When the seal fails, argon escapes, humid air diffuses in, and the cavity equalizes with ambient pressure; the panes decouple toward independent-plate behavior, shifting the modal frequencies, narrowing the mode-pair splitting, and shortening the ring-down. An on-device classifier, trained on laboratory units with gas concentrations verified by tunable diode laser absorption spectroscopy (TDLAS), converts the tap spectrum into a seal-health score. By storing a per-window baseline and trending repeated measurements over seasons, the system distinguishes genuine seal degradation from a single noisy reading, detecting gas loss months or years before the visible fogging that homeowners currently wait for.

## Technical Field

This invention relates to fenestration diagnostics, specifically to non-destructive detection of hermetic seal failure in multi-pane insulating glass units using acoustic resonance excited by manual tapping and analyzed on a consumer smartphone. It sits at the intersection of structural acoustics, building envelope diagnostics, and edge-deployed machine learning: the window becomes its own diagnostic instrument, and a sensor every homeowner already owns performs a test that currently requires either waiting for visible damage or hiring a professional with a dedicated gas analyzer.

## Background

A modern insulating glass unit is a glass sandwich: two or three panes separated by a spacer frame, bonded with a primary sealant (typically polyisobutylene or butyl rubber) and a secondary sealant (polysulfide, silicone, or polyurethane), with desiccant in the spacer and the cavity filled with argon or krypton at 90% or higher concentration for certified units ([Infinita Lab, ASTM testing guide](https://infinitalab.com/blog/sealed-insulating-glass-testing-guide/)). The gas fill is what earns the window its rating: argon, being denser and less conductive than air, raises a sealed double-pane unit to roughly R-2.5 to R-3.5, against R-1.0 for single pane, cutting conductive heat loss by up to half ([The Healthy Attitude](https://thehealthyattitude.com/m-auto/afs/default/rsoc/7f465bd1-1d30-407f-aad4-f367c4fa555b/window-seal-failure-what-to-know?channel=Ch_64796)).

Seals fail. The dominant mechanism is solar pumping: daytime sun heats the glass, the cavity gas expands and pressurizes the seal; overnight cooling contracts it, and this daily breathing cycle works the perimeter sealant until micro-cracks open and gas exchanges with outside air ([InsuredAndMore](https://insuredandmore.com/how-often-do-window-seals-fail)). When properly installed, permanent visible failure runs about 1 percent of units after 10 years ([Window & Door](https://www.windowanddoor.com/article/insulating-glass-problems-and-prevention)), but industry estimates place background gas leakage around 1 percent of fill per year ([InsuredAndMore](https://insuredandmore.com/how-often-do-window-seals-fail)), and accelerated aging studies show aluminum-spacer units losing an average of 7 percent of argon under harsh cycling ([Asphaug, Jelle, Gullbrekken and Uvsløkk, 2016, via Sparklike](https://sparklike.com/en/improving-the-durability-of-double-glazed-sealed-units-sparklike-technology-preventing-gas-leakage-and-energy-loss/)).

What makes seal failure cruel is its staging. Early-stage failure stays invisible for months or even years: gas leaks out, humid air replaces it, and the desiccant still absorbs enough moisture to keep the cavity clear. Home inspection standards do not require inspectors to report failed seals precisely because there is no visible evidence at this stage ([The Healthy Attitude](https://thehealthyattitude.com/m-auto/afs/default/rsoc/7f465bd1-1d30-407f-aad4-f367c4fa555b/window-seal-failure-what-to-know?channel=Ch_64796)). Visible fogging only appears after the desiccant saturates, which is the final stage, not the first. By then the unit's insulating value is already gone, mineral deposits have etched the inner glass surfaces, and the only remedy is full unit replacement. Each fogged window costs on the order of $30 to $80 per year in extra heating and cooling ([The Healthy Attitude](https://thehealthyattitude.com/m-auto/afs/default/rsoc/7f465bd1-1d30-407f-aad4-f367c4fa555b/window-seal-failure-what-to-know?channel=Ch_64796)).

Current field detection options are inadequate for early-stage failure. The established non-destructive method is tunable diode laser absorption spectroscopy: devices such as the [Sparklike Laser Portable](https://sparklike.com/en/product/sparklike-laser-portable/) measure argon concentration through the glass with ±2 percent accuracy, verified against gas chromatography by ift Rosenheim. But the device is a professional instrument: multi-thousand-dollar hardware operated by a trained technician. Its own documentation makes the key methodological point that this disclosure adopts: a single gas reading cannot prove seal tightness; seal performance becomes visible only when the same unit is measured repeatedly over time and a concentration trend appears ([Sparklike](https://sparklike.com/en/product/sparklike-laser-portable/)). Earlier approaches, such as a 2008 study using DC electrical discharge through the glass to estimate argon content ([Sparklike, citing the 2008 study](https://sparklike.com/en/how-to-test-for-argon-gas-in-windows-ensuring-quality-in-insulated-glass/)), remain laboratory methods.

Acoustic methods exist for other glass and panel diagnostics but have not been applied to IGU seal state. Tap-based resonance is standard practice for detecting delamination in composite panels, voids behind tile, and debonded coatings: the excitation is trivial and the ring-down carries the structure's stiffness and damping signature. What the art lacks is the combination this disclosure teaches: using the coupled-pane acoustic physics of the sealed cavity itself as the observable, measured with the microphone and processor in an ordinary smartphone, classified by a model calibrated against TDLAS-verified gas concentrations, and trended per window over seasons so that slow seal degradation emerges from measurement noise. No single reference teaches that combination as a deployable consumer system.

## Detailed Description

### 1. Coupled-Pane Acoustic Physics

A sealed double-pane IGU is mechanically a coupled two-oscillator system. Each glass pane, typically 3 to 6 mm thick, supports bending modes whose frequencies depend on its dimensions, thickness, and edge constraint. The sealed gas cavity couples the two panes through two paths: the structural stiffness of the perimeter seal and spacer, and the pneumatic stiffness of the trapped gas itself. When one pane is struck, the cavity pressure oscillates and drives the second pane; the system responds as paired modes, one in-phase (both panes moving together, cavity volume roughly constant) and one anti-phase (panes moving oppositely, cavity volume alternately compressed and expanded, the gas acting as a spring).

Pneumatic coupling depends on the gas in the cavity: its density, its adiabatic bulk modulus, and, crucially, whether the cavity is sealed or vented. Argon is about 38 percent denser than air and has a lower speed of sound; a cavity at 90 percent argon presents measurably different acoustic impedance than the same cavity exchanged to humid air. When the perimeter seal fails and the cavity equalizes with atmosphere, the pneumatic spring weakens toward the behavior of an open-backed pair of plates: the anti-phase mode shifts downward in frequency, the mode-pair splitting narrows or the modes merge, and the decay quality factor changes as moist air replaces dry argon and as the loss of seal constraint alters edge damping. These are the features the system measures. No frequency value is absolute across window sizes; the signature is per-geometry, which is why the per-window baseline of section 5 is load-bearing.

### 2. Measurement Protocol

To take a reading, the user opens the companion app, selects the window, and places the phone with its microphone 10 to 30 cm from the glass, held steady or resting on the sill. A small adhesive dot (supplied, or a coin-sized sticker) marks the tap point on the pane; consistent excitation location matters more than excitation force, because moving the tap point injects more variance than any downstream algorithm can remove. The user taps once with a fingertip or knuckle, striking through the phone's countdown so the recording captures a clean pre-tap noise floor.

Each accepted tap captures 500 ms of audio at 44.1 or 48 kHz, 16-bit. The app verifies signal quality before analysis: the tap transient must exceed the pre-tap noise floor by at least 20 dB in the 200 Hz to 4 kHz band, and the 50 ms before the tap must show no speech or impulse contamination above a median-based threshold. A rejected tap is discarded immediately and the user is prompted to tap again. Window coverings and blinds must be open and screens removed or retracted for the test, because a screen against the glass adds damping that the model cannot separate from cavity effects. The measurement takes under a minute per window, indoors or out, in any weather calm enough to hold the noise floor.

### 3. Feature Extraction

From each accepted recording the processor computes a short-time Fourier transform (512-point Hann-windowed frames) and extracts the following features from the first 300 ms after the tap transient:

- **Modal frequencies:** the center frequencies of the first three bending-mode pairs in the 200 Hz to 3 kHz band, located by parabolic interpolation around spectral peaks on the averaged spectrum.
- **Mode-pair splitting:** the frequency separation between the in-phase and anti-phase members of each pair, the most direct acoustic read of pneumatic coupling strength.
- **Decay quality factors:** the Q of each mode, estimated from the exponential decay envelope of the band-passed mode signal; sealed dry cavities ring longer.
- **Spectral centroid and roll-off:** aggregate descriptors that capture the redistribution of energy toward independent-plate behavior.
- **Cavity resonance decay:** standing acoustic waves across the cavity depth (fundamental depth mode near 8.6 kHz for 20 mm cavities, up to roughly 29 kHz for 6 mm cavities) ring with higher quality factors when the cavity is sealed and pressurized; venting to atmosphere degrades that ring-down, so the decay envelope of these modes, not merely their presence, carries seal state.

That pipeline yields a feature vector of approximately 20 dimensions. All computation is on-device; the raw recording is discarded after feature extraction and is never uploaded.

### 4. Classification Against TDLAS-Verified Ground Truth

Training runs on laboratory IGU specimens spanning intact, partially degraded, and fully failed seals, with each specimen's cavity gas concentration measured by TDLAS (Sparklike Laser Portable or equivalent) at the time of recording. Ground truth is therefore continuous gas-fill fraction, not a binary label, and the model outputs a seal-health score mapped to estimated argon concentration: above 85 percent reads healthy, 60 to 85 percent reads degraded, and below 60 percent reads failed, with the band edges configurable.

For inference, a compact gradient-boosted tree ensemble (under 200 KB) or a quantized 1D convolutional network over the log-mel spectrum, chosen so inference runs in under 100 ms on a mid-range phone. Training data covers common residential geometries: pane thicknesses 3, 4, and 5 mm; cavity depths 6, 12, 16, and 20 mm; and single-hung, slider, casement, and fixed configurations. Laminated panes, triple-glazed units, and panes smaller than 0.25 m² are separate configuration flags with their own model variants; see Limitations.

Temperature compensation is applied before classification. The app records ambient temperature from the phone's sensors or a paired weather source, because both glass modulus and cavity gas stiffness shift with temperature enough to move modal frequencies by a few percent across seasons. The compensation model is learned from controlled-temperature specimen data and shifts the feature vector to a 20 °C reference before scoring. Same-season comparisons are preferred when available.

### 5. Per-Window Baseline and Longitudinal Trending

The strongest embodiment of this disclosure is a trend, not a single measurement. The first accepted tap on a window establishes its baseline signature; every later tap is compared against it. A stable signature across seasons means the seal holds. A monotonic drift in mode-pair splitting and Q-factor, exceeding the model's per-window noise estimate over three or more readings spanning at least 90 days, flags degradation even when every individual reading sits inside the population "healthy" band. This directly implements the methodological point from professional gas analysis that seal tightness is a trend, not a snapshot ([Sparklike](https://sparklike.com/en/product/sparklike-laser-portable/)), translated into a modality a homeowner can repeat for free.

Seasonal reminders go out in spring and fall, when indoor-outdoor temperature differentials are moderate, and the app maintains a per-window history with dates, temperatures, and scores. When a window crosses the degraded threshold, the app generates a dated evidence packet: the trend chart, the raw feature history, and the measurement conditions log, suitable for a warranty claim with the window manufacturer or installer. Most IGU warranties run 10 to 20 years and cover seal failure explicitly; the packet exists because homeowners currently discover failure only after the warranty's useful window has closed on the invisible years.

### 6. Population Learning With Privacy Preservation

With user consent, the app contributes anonymized feature vectors (never audio) tagged with window geometry class and region to a population model. The population prior improves first-measurement accuracy on windows with no baseline yet and refines the temperature-compensation model across climates. Contributions are opt-in, carry no device identifier, and the per-window history never leaves the phone.

### 7. Figures Description

- **Figure 1:** Cross-section of a double-pane IGU showing panes, spacer, desiccant, primary and secondary seals, and the sealed argon cavity, with the tap point and phone position marked.
- **Figure 2:** Paired in-phase and anti-phase bending modes of the coupled panes, with the pneumatic spring of the sealed cavity coupling them.
- **Figure 3:** Averaged tap spectra for one window geometry in three states: intact seal (sharp paired modes), degraded seal (merged, shifted, lower Q), and fully failed seal (independent-plate response).
- **Figure 4:** Longitudinal seal-health trend for one window over 24 months, showing the drift that crosses the degraded threshold while individual readings stay within the population healthy band.

### 8. Limitations and Failure Modes

No sensing modality is unconditional. The following limitations are material and are stated here so that no embodiment claims more than the physics supports.

- **Seal failure vs. factory under-fill:** the acoustic method measures gas composition and coupling, not history. A unit that left the factory under-filled at 70 percent argon reads the same as a unit that leaked from 95 to 70 percent. Both are out of specification and both deserve attention, but the method cannot attribute cause. The per-window trend separates the two only when a baseline was captured early.
- **Laminated and small panes:** laminated glass (PVB interlayer) is heavily damped and its signature differs fundamentally from monolithic panes; panes under 0.25 m² have sparse modal structure in the phone's usable band. These are excluded from the first model release and gated by a configuration flag, not silently scored.
- **Triple-glazed units:** two coupled cavities produce a more complex modal structure; an extended model variant covers them, but the base release flags triple-pane windows as unsupported rather than mis-scoring them.
- **Capillary-tube and intentionally vented units:** some IGUs are manufactured with capillary tubes or breather vents for high-altitude shipment and are intentionally not sealed. These read as failed by design; the app's configuration screen excludes them.
- **Temperature and weather:** without compensation, a winter tap and a summer tap on the same healthy window differ enough to suggest drift. The compensation model corrects the first-order effect; residual seasonal bias is absorbed by requiring trend confirmation across at least two same-season readings before a degradation flag.
- **Detection floor:** the method resolves meaningful gas loss (target: below 80 percent argon) rather than trace leaks of 1 to 2 percent per year. A unit leaking at the EN 1279-3 limit of under 1 percent per year will take several years to cross the detection floor, which matches the multi-year timescale on which seal failure actually matters.
- **User technique:** taps on the frame instead of the glass, a phone held against the pane (structure-borne coupling), or heavy exterior noise produce garbage readings. The SNR gate and tap-location dot exist for this reason; the app would rather reject a tap than score it wrong.

## Claims

1. A system for detecting perimeter seal failure in an insulated glass unit, comprising: a consumer smartphone having a microphone and a processor; and an application that instructs a user to tap a glass pane of the unit at a marked point, records the resulting acoustic ring-down with the microphone, extracts coupled-pane modal features from the recording, and determines a seal-health score indicating whether the unit's hermetic seal is intact, degraded, or failed.
2. The system of claim 1, wherein the coupled-pane modal features include the frequency splitting between in-phase and anti-phase bending mode pairs and the decay quality factors of the modes, the splitting and quality factors reflecting the pneumatic coupling stiffness of the sealed gas cavity.
3. The system of claim 1, wherein the seal-health score is produced by a machine-learned classifier trained on insulated glass unit specimens whose cavity gas concentrations were measured by tunable diode laser absorption spectroscopy, mapping acoustic features to estimated inert gas concentration.
4. The system of claim 1, further comprising a per-window baseline store, wherein the first accepted measurement establishes a window-specific reference and subsequent measurements are compared against it, and wherein a monotonic drift in the coupled-pane features over at least three measurements spanning at least 90 days flags seal degradation independently of population thresholds.
5. The system of claim 1, further comprising temperature compensation, wherein the measured ambient temperature shifts the extracted features to a reference temperature before scoring, using a compensation model learned from controlled-temperature specimen data.
6. A method for assessing insulated glass unit seal integrity, comprising: marking a consistent tap point on a glass pane; tapping the pane while a smartphone microphone records the acoustic response; verifying the recording's signal quality against a noise-floor threshold and rejecting contaminated recordings; extracting modal frequencies, mode-pair splitting, and decay quality factors from the recording; and reporting a seal-health score derived from the extracted features.
7. The method of claim 6, further comprising scheduling repeated measurements across seasons, trending the seal-health score per window, and generating a dated evidence packet including the trend history and measurement conditions for use in a window warranty claim.
8. The system of claim 1, further comprising a configuration step that records the unit's geometry class including pane thickness, cavity depth, pane count, and laminated status, and that excludes unsupported configurations including laminated panes below a size threshold, triple-glazed units in the base model, and intentionally vented capillary-tube units from scoring.
9. The system of claim 1, wherein all raw audio is processed on-device and discarded after feature extraction, and only anonymized feature vectors leave the device with explicit user consent to improve a population prior model.
10. A professional-inspector embodiment comprising a handheld tap exciter with a contact acoustic pickup in place of the smartphone microphone, performing the coupled-pane resonance analysis recited in claim 1 with calibrated excitation force for repeatable measurements across a building's window inventory.

## Prior Art References

1. [Window & Door, "Insulating Glass: Problems and Prevention"](https://www.windowanddoor.com/article/insulating-glass-problems-and-prevention): permanent fogging as the late visible symptom of seal failure; seal failure causes inert gas loss and degraded thermal performance; solar pumping mechanism; ~1% permanent failure rate after 10 years when properly installed
2. [Infinita Lab, "Sealed Insulating Glass Testing: ASTM E2190, E2188 & E2189"](https://infinitalab.com/blog/sealed-insulating-glass-testing-guide/): IGU construction (primary/secondary sealants, desiccant, argon fill); ASTM E2190/E2188/E2189 durability and fog-resistance test protocols; typical certified fill ≥90% argon
3. [FGIA, "Insulating Glass Certification"](https://fgiaonline.org/product-certification/insulating-glass/): IG certification requirements; ASTM E2190 required for NFRC thermal performance certification and ENERGY STAR inclusion
4. [ASTM E2190, Standard Specification for Insulating Glass Unit Performance and Evaluation](https://www.astm.org/e2190-08.html): durability qualification based on frost/dew point and absence of fog after specified test durations
5. [Sparklike, citing Asphaug, Jelle, Gullbrekken and Uvsløkk (2016)](https://sparklike.com/en/improving-the-durability-of-double-glazed-sealed-units-sparklike-technology-preventing-gas-leakage-and-energy-loss/): accelerated aging of double-glazed units; aluminum-spacer units lost an average of 7% argon; gas loss directly increases U-values and heating demand
6. [Sparklike Laser Portable](https://sparklike.com/en/product/sparklike-laser-portable/): non-destructive argon/krypton measurement via TDLAS with ±2% accuracy (ift Rosenheim verified); single measurement cannot prove seal tightness, trend over time is required; EN 1279-3 limits gas leakage to under 1% per year
7. [Sparklike, "How to Test for Argon Gas in Windows"](https://sparklike.com/en/how-to-test-for-argon-gas-in-windows-ensuring-quality-in-insulated-glass/): 2008 study on DC electrical discharge method for argon estimation in double-pane windows
8. [EngineerFix, "What IGCC Certification Means for Insulating Glass"](https://engineerfix.com/what-igcc-certification-means-for-insulating-glass/): seal failure displaces argon with humid air, significantly increasing the unit's U-factor and degrading NFRC-verified energy performance
9. [InsuredAndMore, "How often do window seals fail?"](https://insuredandmore.com/how-often-do-window-seals-fail): ~1% per year background gas leakage estimate; solar pumping as the dominant seal-failure mechanism; typical seal lifespan of 10–15 years, longer for quality units
10. [Polyisobutylene and Silicone in Warm Edge Glazing Systems, Polymers 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8269632/): long-term warm-edge seal performance; over 95% of examined units showed no complete sealing failure; warm-edge seal pull-in correlates with degraded thermal performance in infrared imaging

## Implementation Notes

That tap point dot is not decoration. Place it at least 150 mm from any edge and 100 mm from the spacer shadow line; edge proximity couples frame damping into the reading and the spacer shadow region has atypical stiffness. The same dot stays on the window for the life of the monitoring program, so every reading excites the same point.

Calibrate the classifier against TDLAS on real production units, not only on laboratory specimens with drilled seals. Drilled-seal specimens exchange gas too fast and develop different humidity profiles than genuinely failed units; a model trained only on drilled specimens overfits to the exchange rate rather than the steady-state signature. The training set should include field-failed units pulled from window replacement jobs, with their TDLAS readings taken before removal.

Ship the first release with the per-window trend as the primary output and the single-shot score clearly labeled as provisional. A homeowner's first tap establishes a baseline, not a diagnosis; the second same-season tap, six to twelve months later, is the first reading that can actually flag drift. Setting that expectation in the UI avoids both false alarms and false confidence.

Any warranty evidence packet should include the measurement temperature log, because a manufacturer disputing a claim will reach for temperature artifacts first. Export the packet as a signed PDF with a hash of the feature history so the record cannot be casually dismissed as user-entered numbers.
