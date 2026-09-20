# PA-2026-177: Per-Address Hail Damage Severity Assessment of Roofing Assemblies Using Attic-Mounted Acoustic Impact Sensing with Material Fragility Fusion

**Title:** System and Method for Per-Address Hail Damage Severity Assessment of Roofing Assemblies Using Attic-Mounted Acoustic Impact Sensing with Material Fragility Fusion

**Filing:** LITF-PA-2026-177
**Published:** September 20, 2026
**Domain:** Property Insurance / Acoustic Sensing / Roofing
**Full Disclosure:** [liveinthefuture.org/priorart/attic-hail-impact-roof-damage-assessment.html](https://liveinthefuture.org/priorart/attic-hail-impact-roof-damage-assessment.html)
**License:** [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/) — Public Domain

> Prior Art Notice: This document is published as defensive prior art under
> [35 U.S.C. Sec. 102(a)(1)](https://www.law.cornell.edu/uscode/text/35/102).
> The inventions described herein are dedicated to the public domain as of the
> publication date above.

---

## Abstract

Disclosed is a system and method for assessing hail damage severity to a specific roof within hours of a hailstorm, using acoustic and vibration sensors mounted inside the attic on the underside of the roof deck. Unlike hail disdrometers, which measure free-falling hail for meteorological purposes, and unlike radar-derived hail products, which estimate storm severity at kilometer scale, this system measures the hailstorm as experienced by the actual roof assembly: every impact is captured as a structure-borne acoustic transient transmitted through the shingles, underlayment, and deck to the attic-side sensor. An edge machine learning classifier distinguishes hailstone impacts from rain, wind-driven debris, foot traffic, and aircraft overflight. The system inverts the per-impact kinetic energy distribution into a hailstone size distribution, then fuses that distribution with a roof-specific fragility model that accounts for roofing material, UL 2218 impact rating, age-based weathering derating, deck construction, roof slope, and the temperature of the shingles at the time of impact (asphalt shingles embrittle as temperature falls, so identical hail fractures cold shingles that warm shingles would survive). The output is a per-storm Roof Damage Severity Score quantifying expected granule loss, shingle fracture probability, and leak risk for that specific address, delivered before an adjuster could be scheduled. Deployed across many homes, the sensor network produces address-resolution hail swath maps that verify which properties were actually struck, providing fraud-resistant evidence for parametric insurance triggers and claim triage.

## Technical Field

This invention relates to property damage assessment, specifically to acoustic sensing of hailstone impacts transmitted through residential roofing assemblies, machine-learned impact classification and size inversion, and fusion with material fragility models for per-address hail damage severity scoring in property insurance workflows.

## Background

Hail is the costliest severe-convective-storm peril for U.S. property insurers. Verisk estimates the insurable gross average annual loss from severe thunderstorms in the contiguous United States at approximately $25 billion, with hail contributing roughly half, on the order of $12 billion per year (Verisk). Roofing claims dominate hail losses: the roof is the largest hail-exposed surface on a home, and a single storm can generate tens of thousands of claims across a metro area in one afternoon.

How those claims are assessed today is slow, subjective, and adversarial. The industry-standard method is a manual roof inspection: an adjuster or contractor climbs the roof, chalks a 10-foot by 10-foot test square, and counts hail bruises within it, extrapolating to the full roof. After a major hail event, adjuster capacity collapses. Homeowners wait weeks or months for inspections while damaged roofs leak. The assessment itself is disputed territory: the industry has litigated for decades over whether observed shingle damage is functional (fracture through the mat, warranting replacement) or merely cosmetic (granule displacement, denting without fracture). Storm-chasing contractors descend on hail-struck neighborhoods offering free inspections and contingency-fee claim filing, creating a claims environment where neither the homeowner nor the insurer has an objective, contemporaneous measurement of what actually struck the roof.

The laboratory standard for shingle impact resistance, UL 2218, rates prepared roof coverings by dropping steel balls onto new shingles: Class 1 resists a 1.25-inch ball dropped from 12 feet, Class 4 resists a 2-inch ball dropped twice from 20 feet with no fracture (Malarkey Roofing / UL 2218). But UL 2218 tests factory-fresh shingles at room temperature. It does not evaluate weathering, temperature, or aging. Per IBHS aging studies, the flexibility of impact-rated shingles decreases significantly over time, and a Class 4 shingle can perform closer to Class 2 after 10 to 15 years of field exposure (CCR Magazine). A roof's real hail vulnerability is a function of what it is, how old it is, and how cold it was when the hail hit. No existing measurement system captures all three at the address level.

Existing hail measurement technology measures the storm, not the roof. Hail disdrometers, including acoustic impact transducers developed for NASA Kennedy Space Center launch operations (Lane et al.) and the HS-01 acoustic hailstone disdrometer deployed in Xinjiang (MDPI), measure free-falling hailstone size distributions for meteorological research. They are sparse, expensive, and deliberately decoupled from any building. Weather radar products such as MESH (Maximum Expected Size of Hail) estimate hail size at kilometer-scale resolution from reflectivity, with known biases under certain atmospheric conditions. Recent work applies deep neural networks to radar, environmental, and insurance-claims data to estimate hail damage (Atmospheric Measurement Techniques, 2024, critical success index 0.88 against observed damage), but these are area-level estimates trained on claims data that is itself the product of the slow, subjective inspection process. Drone photogrammetry can map hailstones on the ground (AMT, 2024), but only where someone flies a drone promptly after the storm.

The gap in the art is a system that: (a) senses hail impacts from inside the building, through the actual roof assembly being assessed, with no exterior mounting and no weather exposure of the sensor; (b) classifies and sizes individual impacts at the edge; (c) converts the measured impact distribution into damage severity using a roof-specific fragility model that includes material, age derating, and impact-time temperature; and (d) networks many such sensors into address-resolution hail swath maps that verify per-property storm exposure for insurance triage.

## Detailed Description

### 1. Attic-Mounted Sensor Hardware

Each sensing node is installed inside the attic, mechanically coupled to the underside of the roof deck (the interior face of the OSB or plywood sheathing, or to a rafter within 30 cm of the deck). Mounting is by screw, adhesive pad, or magnetic base; no roof penetration and no exterior work is required, so installation needs no ladder work on the roof and does not void roofing warranties. The interior mounting is a deliberate design choice: the sensor is protected from the storm it measures, and it captures the impact energy that actually propagated through the roofing assembly, which is the energy available to cause damage.

Each node comprises: a MEMS microphone (flat response 100 Hz to 20 kHz, for airborne attic reverberation of impacts), a MEMS accelerometer (plus or minus 16 g range, for structure-borne transients conducted through the deck), a temperature sensor (for shingle temperature estimation), a microcontroller with edge inference capability (ARM Cortex-M class), and a WiFi or sub-GHz radio for backhaul. Target bill of materials is under $60 per node. A typical single-family home uses two to three nodes distributed across roof planes, which additionally enables coarse localization of impact zones to roof facets via arrival-time differences.

### 2. Impact Detection and Classification

The node continuously monitors the accelerometer and microphone streams with a low-power onset detector. When the short-time energy in the 2 kHz to 12 kHz band exceeds an adaptive background threshold (hail impacts on shingles produce broadband transients with fast attack; rain produces a sustained lower-level wash), the node captures a 200 ms window around the onset and extracts features for classification.

An edge classifier (quantized gradient-boosted tree or small convolutional network, under 300 KB) assigns each transient to one of: hailstone impact, rain impact, wind-driven debris (branches, which produce longer-duration lower-frequency thuds), human footstep on the roof (rhythmic, low-frequency, sustained sequences), aircraft overflight (slow amplitude envelope, narrowband), or attic-internal noise (HVAC, stored-object shifts). The hail-versus-rain discrimination builds on the established finding that hail and rain impacts on a plate are spectrally distinct: hail produces a sharper attack and higher spectral centroid than rain at comparable intensity (Lane et al.). The classifier is trained on labeled data from instrumented roofs exposed to natural hailstorms, supplemented by controlled drop tests with ice spheres of known diameter onto representative roof assemblies.

Only transients classified as hailstone impacts with confidence above 0.8 are retained for sizing. All other classes are counted for context (rain rate, debris load) but excluded from the damage calculation.

### 3. Per-Impact Energy Estimation and Size Distribution Inversion

For each classified hail impact, the system estimates the impact kinetic energy from the peak deck acceleration and the spectral energy in the impact transient, using a per-installation calibration transfer function. Calibration is performed at installation with a reference tap test: a solenoid-driven impactor of known energy strikes the roof exterior at marked points while the attic nodes record, establishing the assembly-specific mapping from transmitted signal to impact energy for that roof's combination of shingle, underlayment, and deck.

Impact kinetic energy is converted to hailstone diameter using terminal-velocity physics. A hailstone of diameter d and density near 0.9 g/cm3 falls at a terminal velocity that scales approximately with the square root of diameter; kinetic energy therefore scales approximately with d cubed times velocity squared. A 1-inch (2.5 cm) hailstone at terminal velocity carries on the order of 1 joule; a 2-inch (golf-ball-size) stone carries tens of joules. The system accumulates the per-impact energy histogram over the storm and inverts it into a hailstone size distribution (number of stones per square meter per size bin), normalizing by the effective sensing area of each node derived from the calibration tap tests. The output is the storm's measured size distribution at that address, not an area-averaged radar estimate.

### 4. Roof-Specific Fragility Model

The measured size distribution is converted to damage through a fragility model parameterized for the specific roof, populated at installation from a short homeowner questionnaire plus optional photo verification:

- **Material and UL 2218 class:** Asphalt shingle (3-tab, architectural, or Class 3/4 impact-resistant), metal (standing seam, stone-coated steel), concrete or clay tile, wood shake, or synthetic. Each material maps to a baseline fracture-energy threshold derived from UL 2218 test energies and published IBHS impact research.
- **Age-based weathering derate:** Asphalt shingle impact resistance degrades with UV exposure, thermal cycling, and granule loss. The model applies a derating curve calibrated to IBHS aging research, under which a Class 4 shingle's effective fracture threshold declines toward Class 2 levels over 10 to 15 years of field exposure. Roof age is taken from installation records or permit data where available, homeowner report otherwise, with an uncertainty band that widens the score's confidence interval when the age is unverified.
- **Deck and assembly:** Sheathing type and thickness (7/16-inch OSB versus 5/8-inch plywood versus plank decking), underlayment (synthetic versus felt), and number of shingle layers change how impact energy couples into the shingle mat. The calibration tap test captures the assembly's net transfer function directly, so the model does not need to simulate the stackup from first principles.
- **Roof slope:** Impact energy normal to the shingle surface scales with the cosine of the incidence angle; steep-slope facets see lower normal impact energy from vertically falling hail than low-slope facets. Multi-node installations attribute impacts to facets and apply per-facet slope correction.

### 5. Impact-Time Temperature Embrittlement

Asphalt is a viscoelastic material: its fracture toughness falls as temperature drops. A hailstone striking a shingle at near-freezing temperature can fracture the mat at an impact energy that the same shingle would absorb without damage at 25 C. UL 2218 testing at room temperature therefore systematically overstates cold-weather impact resistance, and hailstorms frequently occur with surface temperatures far below room temperature (spring and fall hail events, high-plains storms, nighttime events).

The node records attic air temperature and roof-deck temperature continuously. A thermal model of the roof assembly (shingle thermal mass, solar loading history, attic ventilation rate) estimates the shingle surface temperature at the time of each impact. The fragility model applies a temperature derating factor to the fracture-energy threshold: colder shingles fracture at lower impact energies. This temperature fusion is, to the inventor's knowledge, absent from all existing hail damage estimation methods, which treat a 2-inch hail report identically whether the roof was at 5 C or 35 C.

### 6. Roof Damage Severity Score

For each detected hailstorm (defined as a cluster of classified hail impacts separated from other clusters by at least 30 minutes of quiet), the system computes a Roof Damage Severity Score (RDSS) from 0 to 100:

- **Sub-score A, fracture risk (0 to 60 points):** For each size bin of the measured hailstone distribution, the fraction of impacts whose estimated energy exceeds the temperature- and age-adjusted fracture threshold of the roof material. Weighted by bin population and normalized by roof area.
- **Sub-score B, granule-loss and surface degradation (0 to 25 points):** Cumulative sub-fracture impact energy, which dislodges granules and accelerates aging even without mat fracture. Calibrated against granule-loss measurements from controlled ice-sphere testing.
- **Sub-score C, leak-risk concentration (0 to 15 points):** Spatial concentration of high-energy impacts from multi-node facet attribution. A storm that concentrates large hail on one roof plane (wind-driven) poses higher leak risk than the same total energy spread uniformly, because overlapping fracture zones create water paths.

Score bands: 0 to 20, no actionable damage expected; 21 to 45, cosmetic and granule-loss damage likely, monitor; 46 to 70, functional shingle damage probable, professional inspection recommended; 71 to 100, widespread functional damage likely, prioritize for claim filing and emergency dry-in. Each score ships with a confidence interval reflecting roof-age uncertainty, calibration age, and classification confidence. The full report, including the measured size distribution, temperature log, and per-facet impact map, is delivered to the homeowner and, with consent, to the insurer within hours of storm end.

### 7. Network Hail Swath Mapping and Fraud Resistance

A single sensor scores one roof. A network of sensors across a metro area produces an address-resolution hail swath map: which streets were struck, by what size hail, at what time, at what shingle temperature. This resolves the central information asymmetry of hail claims. Today, an insurer deciding whether a claim filed six months after a storm is legitimate has only radar archives and the contractor's photos. With the network, the insurer queries the swath map: did a damaging hail event actually occur at this address on the claimed date, and did the measured size distribution exceed this roof's fragility threshold?

The measurement is fraud-resistant in both directions. A homeowner cannot fabricate a hailstorm the sensors did not record; a contractor cannot attribute pre-existing wear to a storm whose measured impacts were below the fracture threshold. Conversely, a homeowner whose roof genuinely took 2-inch hail at 8 C has timestamped, calibrated evidence that does not depend on winning an argument with an adjuster about whether a bruise is functional. The network also serves parametric insurance directly: a policy can pay out automatically when the measured RDSS at the insured address exceeds an agreed threshold, with no inspection and no claims adjuster.

### 8. Privacy Architecture

The system processes all audio on the device. The microphone stream is analyzed in overlapping short windows for transient features only; no continuous audio is stored, and no audio leaves the home. Only impact event records (timestamp, classified type, estimated energy, confidence) and aggregate storm reports leave the device, transmitted over encrypted channels. The sensor cannot reconstruct speech: the analysis band and windowing are tuned to impulsive transients, and raw waveform buffers are overwritten within seconds. Attic mounting means the sensor has no line of sight to living spaces and no view of occupants.

## Claims

1. A system for assessing hail damage severity to a roofing assembly, comprising: at least one sensor node mounted inside the building on the interior side of the roof deck, the node comprising an accelerometer and a microphone configured to capture structure-borne and airborne transients of hailstone impacts transmitted through the roofing assembly; an edge classifier that distinguishes hailstone impacts from non-hail transients including rain, wind-driven debris, human footsteps, and aircraft; an energy estimation module that converts classified hail impacts into per-impact kinetic energies using a per-installation calibration transfer function; and a damage scoring module that fuses the per-impact energies with a roof-specific fragility model to produce a per-storm damage severity score for the address.

2. The system of claim 1, wherein the roof-specific fragility model comprises a baseline fracture-energy threshold derived from the roofing material's UL 2218 impact resistance class, adjusted by an age-based weathering derating curve that reduces the effective fracture threshold over the service life of the roof.

3. The system of claim 1, further comprising a temperature estimation module that determines the shingle surface temperature at the time of each hail impact and applies a temperature derating factor to the fracture-energy threshold, wherein colder shingle temperatures reduce the impact energy required to fracture the roofing material.

4. The system of claim 1, wherein the energy estimation module inverts the per-impact kinetic energy distribution into a hailstone size distribution using terminal-velocity scaling of hailstone mass with diameter, normalized by an effective sensing area derived from a calibration impact test performed at installation.

5. The system of claim 1, wherein the damage severity score comprises a fracture-risk sub-score based on the fraction of impacts exceeding the temperature- and age-adjusted fracture threshold, a granule-loss sub-score based on cumulative sub-fracture impact energy, and a leak-risk concentration sub-score based on the spatial concentration of high-energy impacts across roof facets.

6. The system of claim 1, comprising a plurality of sensor nodes distributed across roof planes of the building, wherein arrival-time differences of impact transients across nodes attribute impacts to specific roof facets and per-facet slope corrections adjust the normal impact energy for roof pitch.

7. A method for per-address hail damage assessment, comprising: sensing hailstone impacts from the interior side of a roof deck through the roofing assembly; classifying impact transients at the edge to isolate hailstone impacts from rain, debris, footstep, and aircraft transients; estimating per-impact kinetic energy via a per-installation calibration transfer function; estimating shingle temperature at impact time; adjusting a material- and age-specific fracture-energy threshold by the estimated temperature; and computing a damage severity score from the fraction of impacts exceeding the adjusted threshold.

8. The method of claim 7, further comprising determining the roofing material, UL 2218 impact class, installation age, sheathing type, and roof slope at enrollment, and deriving the baseline fracture-energy threshold and weathering derating curve therefrom.

9. The method of claim 7, further comprising delivering the damage severity score, the measured hailstone size distribution, and the impact-time temperature log to the property owner and, with consent, to an insurer within 24 hours of storm end.

10. A network hail verification system, comprising: a plurality of attic-mounted hail impact sensing systems according to claim 1 deployed across a geographic region; a swath mapping module that aggregates per-address hailstone size distributions and damage severity scores into an address-resolution hail swath map; and a claim verification interface that, given a property address and a claimed storm date, returns whether a damaging hail event was measured at that address and whether the measured impacts exceeded the address's roof fragility threshold.

11. The network hail verification system of claim 10, further configured to trigger a parametric insurance payout automatically when the measured damage severity score at an insured address exceeds a policy-defined threshold, without requiring a physical roof inspection.

12. The system of claim 1, wherein all acoustic analysis occurs on the sensor node, no continuous audio is stored, raw waveform buffers are overwritten within seconds, and only impact event records and aggregate storm reports leave the premises over encrypted channels.

## Implementation Notes

Target hardware cost is under $60 per node in volume, using commodity MEMS microphones, MEMS accelerometers, and Cortex-M microcontrollers. Installation is a 30-minute attic visit with no exterior work. The calibration tap test uses a handheld solenoid impactor of known energy striking marked exterior points; a two-person crew (one on the roof with the impactor, one in the attic confirming capture) completes calibration in under 15 minutes. Recalibration is recommended after any roof replacement or major repair, which changes the assembly transfer function.

Known limitations, stated plainly. The system estimates damage from transmitted impact energy and fragility models; it does not image the shingles. Cosmetic-only damage (granule displacement without mat fracture) and functional damage (mat fracture) produce overlapping transmitted signatures at the margin, so scores near the 45-point inspection threshold carry wide confidence intervals and the system recommends human inspection there rather than asserting a claim outcome. Metal roofs dent rather than fracture, so the fracture sub-score is replaced by a dent-depth estimator for metal assemblies, calibrated separately. Tile roofs crack discretely; a single cracked tile dominates the score, and the system flags discrete high-energy events on tile rather than accumulating energy. Snow, ice, and sleet produce impact transients that the classifier must reject; winter-storm confusion is the largest classification error source and the reason the hail confidence threshold is set at 0.8. Attics with spray-foam insulation applied directly to the deck underside attenuate structure-borne transients significantly; such installations require accelerometer coupling to rafters or purlins rather than the deck, with reduced sensitivity disclosed in the calibration report.

The temperature embrittlement model is the least validated component: while the direction of the effect (colder asphalt fractures more easily) is well established in roofing science, the quantitative derating curve used here is fit to limited laboratory data and carries the widest uncertainty band in the scoring model. Field validation against post-storm inspection outcomes across temperature ranges is the highest-priority research step before scores are used for automated claim decisions.

## Prior Art References

1. Verisk — U.S. severe thunderstorm insurable gross average annual loss approximately $25 billion; hail contributes roughly half (https://www.verisk.com/blog/managing-severe-thunderstorm-risk-iii/)
2. Malarkey Roofing / UL 2218 — Class 4 impact rating: 2-inch steel ball dropped from 20 feet, twice, with no fracture; standard does not evaluate weathering, temperature, or aging effects (https://www.malarkeyroofing.com/class-4-impact-resistance-roofing-shingles/)
3. CCR Magazine / IBHS aging studies — Impact-rated shingle flexibility decreases significantly with age; Class 4 performance degrades toward Class 2 after 10 to 15 years (https://ccr-mag.com/do-hail-resistant-shingles-perform-as-their-ratings-claim/)
4. Lane et al., "A Hail Size Distribution Impact Transducer" — Acoustic impact transducer for hail monitoring at NASA Kennedy Space Center; frequency analysis discriminates hail impacts from rain impacts more robustly than amplitude alone (https://arxiv.org/pdf/1408.4702)
5. MDPI — HS-01 acoustic hailstone disdrometer deployed in Aksu, Xinjiang; measured hailstone sizes predominantly under 20 mm with 5 to 10 mm mode (https://www.mdpi.com/2863636)
6. AMS 2016 — Low-cost rapidly deployable network of hail impact disdrometers; impact-energy to hail-size relationships and their precision limits (http://ams.confex.com/ams/96Annual/webprogram/Manuscript/Paper283560/AMS%20National%202016%20Manuscript.pdf)
7. Atmospheric Measurement Techniques, 2024 — Deep neural network hail damage estimates from radar, environmental, and insurance-claims data; critical success index 0.88 against observed damage (http://amt.copernicus.org/articles/17/407/2024/amt-17-407-2024.pdf)
8. Atmospheric Measurement Techniques, 2024 — Drone-based photogrammetry with deep learning to estimate hail size distributions on the ground (https://amt.copernicus.org/articles/17/2539/2024/amt-17-2539-2024.pdf)
9. DECRA Roofing — UL 2218 class definitions (1.25-inch through 2-inch steel balls); dents and granule loss classified as aesthetic rather than functional damage under manufacturer warranties (https://www.decra.com/pro/hail-understanding-impact-resistance)
