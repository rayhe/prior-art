# PA-2026-172: Pothole Formation Prediction from Fleet Suspension Telemetry

**Title:** System and Method for Predicting Pothole Formation Using Fleet Vehicle Suspension Telemetry and Freeze-Thaw Cycle Modeling

**Filing:** LITF-PA-2026-172
**Published:** September 15, 2026
**Domain:** Transport / Road Infrastructure
**Full Disclosure:** [liveinthefuture.org/priorart/pothole-formation-prediction-fleet-telemetry.html](https://liveinthefuture.org/priorart/pothole-formation-prediction-fleet-telemetry.html)
**License:** [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/) — Public Domain

> Prior Art Notice: This document is published as defensive prior art under
> [35 U.S.C. Sec. 102(a)(1)](https://www.law.cornell.edu/uscode/text/35/102).
> The inventions described herein are dedicated to the public domain as of the
> publication date above.

---

## Abstract

Disclosed is a system and method for predicting where potholes will form on a road network 7 to 30 days before the pavement surface fails, by fusing suspension vibration telemetry from fleet vehicles with freeze-thaw cycle modeling, traffic loading, pavement age, and municipal repair records. Each vehicle's accelerometer stream is band-split at 50 to 100 Hz sampling: high-frequency energy in the 30 to 80 Hz band reveals subsurface raveling and binder stripping weeks before spalling, while a per-vehicle transfer function, learned against a known-smooth calibration stretch, converts measured chassis motion into true pavement stiffness. Freeze-thaw exposure is gated on surface moisture, because dry freezing does no damage: a nominal cycle count, a moisture-gated count, and a moisture-and-load-gated count. Municipal repair logs provide labeled training data with a label-lag correction that only rewards predictions that beat the crew's own patrols. Segments are scored to a 30-day failure probability, risk-weighted by traffic volume, and issued as seal-and-repair work orders while a crack is still a roughly $50 cold-patch visit.

## Technical Field

This invention relates to road infrastructure monitoring and predictive maintenance, specifically to predicting pothole formation by fusing vehicle suspension vibration telemetry with environmental freeze-thaw modeling, traffic loading history, and pavement maintenance records to generate actionable repair forecasts.

## Background

Potholes are one of the most expensive and least predicted failures in civil infrastructure. They arrive suddenly, damage vehicles, injure cyclists, and cost municipalities millions in emergency repair. A driver survey by AAA found that pothole damage costs American drivers about $3 billion annually, and in a single bad winter the city of Boston can spend roughly $2 million filling them. Yet the dominant approach to pothole management is still reactive: wait for a citizen to call 311, or wait for a maintenance crew to spot the crater, then dispatch a patch truck.

Research on pothole sensing has focused almost entirely on detection. Eriksson et al.'s Pothole Patrol (ACM MobiSys 2008) mounted accelerometers and GPS on Boston taxis and detected existing potholes from vibration signatures, citing over 500,000 annual US insurance claims from pothole damage. Mednis et al. (IEEE DCOSS 2011) demonstrated 90% detection of marked potholes using Android smartphone accelerometers with simple threshold algorithms (Z-THRESH, Z-DIFF, STDEV(Z), G-ZERO). Carrera, Guerin, and Thorp (ISPRS 2013) described StreetBump, Boston's crowdsourced pothole mapping app, deployed operationally from 2012 with an Innocentive algorithm competition to improve detection. Mohan, Padmanabhan, and Ramjee (ACM SenSys 2008) described Nericell, which used smartphone sensors for road and traffic monitoring including bump detection. These systems all answer the question "where is the pothole?" None answers "where will the pothole be?"

The physics of pothole formation is well understood in materials science but disconnected from live sensing. Mohi ud Din, Mir, and Farooq (Transportation Research Procedia, 2020) reviewed how freeze-thaw cycling destroys asphalt cohesion and adhesion: water infiltrates microcracks, freezes and expands, and each cycle widens the damage until the binder strips from the aggregate and the surface spalls. The KTH Royal Institute of Technology developed a thermomechanical framework for frost heave and thaw settlement in asphalt pavements, showing that subgrade stiffness drops sharply after thaw. The ASCE Journal of Cold Regions Engineering published freeze-thaw cycle zoning using equivalent freeze-thaw cycles and k-means performance classification. These models predict long-term pavement life at the design stage. They are not connected to real-time telemetry from vehicles driving the roads.

Pavement condition surveys (PCI, IRI) measure the current state of the road, and smartphone-based roughness estimation correlated to IRI has been demonstrated (Douangphachanh and Oneyama, 2013; Aleadelat and Ksaibati, 2017, with R-squared above 0.9 for predicting Pavement Serviceability Index). But surveys are sparse snapshots, months apart, and they measure the present, not the future. A roughness survey tells a city the road is getting worse; it does not say which 50-meter segment will crater in the next three weeks.

The gap in the art is a prediction system that converts the continuous, free vibration signal of vehicles already driving the network into a forward-looking failure forecast, fused with the environmental driver (freeze-thaw cycling gated on moisture) and the load driver (traffic volume) that actually cause potholes. Detection tells the crew where to patch. Prediction tells the crew where to seal, weeks earlier, at a fraction of the cost.

## Detailed Description

### 1. System Architecture

The system comprises five functional modules: a telemetry ingestion module, a per-vehicle calibration module, an environmental modeling module, a failure prediction module, and a work-order issuance module. In the primary embodiment, ingestion and calibration run as a lightweight edge process on each fleet vehicle's telematics unit (or on a paired smartphone), the environmental and prediction modules run in a municipal or fleet-operator cloud, and work orders are issued through an API to the existing road-maintenance dispatch system.

Data flow: raw 3-axis accelerometer and GPS traces stream from the edge to the cloud in compressed summaries, never as full raw traces. The calibration module converts each vehicle's measured chassis motion into pavement-true excitation using that vehicle's learned transfer function. The environmental module maintains, per road segment, a freeze-thaw exposure state updated from weather feeds. The prediction module fuses calibrated vibration features, environmental exposure, traffic load, pavement age, and repair history into a 30-day pothole formation probability per 50-meter segment. The work-order module ranks segments by risk-weighted expected cost and emits seal-and-repair work orders above a configurable probability threshold.

### 2. Telemetry Ingestion and Edge Summarization

Each participating vehicle contributes 3-axis accelerometer samples at 50 to 100 Hz and GPS position at 1 Hz during driving. Raw transmission of this data is prohibitive: 1,000 vehicles driving 6 hours per day at 50 Hz on 3 axes at 2 bytes per sample generate roughly 6.5 gigabytes per day of raw vibration data. The edge module therefore computes per-second summaries and transmits only those: RMS and peak acceleration in each of three frequency bands (0.5 to 8 Hz body motion, 8 to 30 Hz wheel hop, 30 to 80 Hz high-frequency raveling energy), GPS-snapped position, speed, and a segment identifier from an offline road-network map.

The edge module applies a 4th-order Butterworth high-pass filter at 0.5 Hz before band energy computation, removing gravity and slow road-grade components so the transmitted features reflect pavement excitation rather than topography. Segments are matched to a 50-meter tiled road network using GPS plus map matching; traversals slower than 15 km/h or faster than 110 km/h are discarded, as are traversals during hard braking or cornering (lateral acceleration above 0.3 g), which contaminate the vertical vibration signal. Per traversal, the transmitted payload is approximately 2 kilobytes: band energies at 1 Hz, segment id, timestamp, speed profile, and vehicle id.

### 3. Per-Vehicle Suspension Calibration

The central technical problem of fleet vibration sensing is that every vehicle is a different instrument. A delivery van's stiff suspension, a sedan's soft springs, worn shock absorbers, tire pressure, and cargo load all reshape the measured vibration. Uncalibrated, a stiff truck driving smooth pavement can look worse than a soft sedan driving rough pavement. The disclosed solution is a per-vehicle transfer function learned against a known-smooth calibration stretch.

Each vehicle is driven (or its historical traces are mined) over a designated smooth reference segment: a recently resurfaced road section whose true profile is measured once with a reference profilometer or taken from the municipal IRI survey. The calibration module estimates, per vehicle, a linear transfer function H(f) mapping measured chassis acceleration to pavement vertical excitation, parameterized as a second-order low-pass (the suspension) in series with a first-order high-pass (the tire and unsprung mass), fitted in the 0.5 to 80 Hz band by least squares against the reference profile. The transfer function is re-estimated weekly from opportunistic passes over the calibration stretch, tracking shock-absorber wear, tire changes, and load drift.

Calibration quality is monitored: if a vehicle's fitted transfer function deviates by more than 3 dB from its 30-day median across the 8 to 30 Hz band, the vehicle is flagged for maintenance (worn shocks) and its data is down-weighted in the aggregation until recalibration. This turns a sensing liability into a fleet-maintenance feature.

### 4. Vibration Feature Extraction

From calibrated traversals, the prediction module extracts, per 50-meter segment per day, the following features:

**High-frequency raveling energy (30 to 80 Hz):** The earliest mechanical precursor of pothole formation. As binder strips from aggregate and microcracks open below the surface, the tire-pavement contact generates broadband high-frequency excitation that a healthy surface does not produce. The disclosed design targets this signature: the 30 to 80 Hz band rising by 2 to 6 dB in the weeks before spalling, while the low-frequency roughness bands (0.5 to 8 Hz) remain flat. This inversion, high-frequency energy rising without low-frequency roughness rising, is the signature that distinguishes a segment about to fail from a segment that is merely rough.

**Stiffness proxy:** The ratio of calibrated vertical excitation energy in the 8 to 30 Hz wheel-hop band to vehicle speed, normalized by the segment's historical baseline. A dropping stiffness proxy indicates subgrade softening, the thaw-weakened foundation that precedes surface failure. Frost-heave research shows subgrade stiffness can drop by an order of magnitude after thaw; the stiffness proxy is the live observable of that process.

**Roughness trend:** The 0.5 to 8 Hz band energy slope over a 90-day window, capturing gradual deterioration. Alone it predicts poorly (rough roads persist for years without cratering), but interacted with freeze-thaw exposure it becomes informative.

**Traversal count and vehicle diversity:** The number of traversals and distinct vehicles contributing to the segment-day aggregate, used as a confidence weight. Segments with fewer than 5 vehicle-traversals per week are flagged low-confidence and excluded from work-order generation.

### 5. Moisture-Gated Freeze-Thaw Modeling

Not all freeze-thaw cycles damage pavement. Dry freezing does nothing; damage requires water in the pore structure when the freeze arrives. The environmental module maintains three cycle counters per segment, updated daily from a weather feed (temperature, precipitation, humidity) plus a surface-moisture estimate:

1. **Nominal freeze-thaw cycles:** days on which the temperature crosses 0 C in either direction. This is the naive count used in design manuals.
2. **Moisture-gated cycles:** nominal cycles on days when estimated surface moisture exceeds a threshold (from recent precipitation, snowmelt, or humidity persistence). Only wet cycles count.
3. **Moisture-and-load-gated cycles:** moisture-gated cycles weighted by that day's heavy-vehicle traffic volume on the segment, because freeze damage under load propagates cracks far faster than unloaded freezing.

Surface moisture is estimated from a simple bucket model: precipitation adds, evaporation (from temperature, wind, and solar estimates) subtracts, with a 3-day memory. Snow cover is treated as a moisture reservoir released at melt. The module also tracks the current frost depth estimate from cumulative freezing degree-days, since deep frost followed by rapid thaw is the highest-risk pattern (maximum subgrade saturation at minimum stiffness).

### 6. Label Construction with Lag Correction

Training labels come from municipal repair logs: 311 pothole reports, crew work orders, and resurfacing records, geocoded to the 50-meter segment grid. Raw labels are noisy: citizens mislocate pins, duplicate reports are common, and a reported pothole may have existed for weeks before anyone called. The label pipeline applies:

- **Deduplication:** reports within 100 meters and 14 days of each other are merged to a single event.
- **Date correction:** the event date is set to the earliest of the report date, the work-order date, or the first date on which the segment's stiffness proxy dropped below its failure threshold, preventing the model from learning to predict the reporting delay instead of the failure.
- **Negative mining:** segments with high traversal counts and no reports for 180 days are hard negatives.
- **Lag-corrected evaluation:** a prediction counts as a true positive only if it precedes the crew's own discovery (the earliest report or patrol observation) by at least 7 days. Predicting a pothole the crew already knew about scores zero. This is the acceptance criterion that makes the system a prediction system rather than a detection system with a forecast label.

### 7. Prediction Model

The prediction module trains a gradient-boosted decision tree (LightGBM in the reference implementation; any calibrated classifier is acceptable) on segment-day feature vectors, predicting the probability of pothole formation within the next 30 days. Features: the vibration features of section 4 (current values, 7-day and 30-day slopes, and the high-frequency/low-frequency inversion ratio), the three freeze-thaw counters (cumulative since last resurfacing, and 14-day and 60-day windows), traffic loading (daily heavy-vehicle count, cumulative since resurfacing), pavement age and surface type, and repair history (days since last seal, patch, or resurfacing).

The model is trained per climate zone (freeze-thaw regions vs. non-freeze regions use different feature weights; in warm climates the moisture-gated counters carry near-zero weight and the model relies on vibration and load features). Probability outputs are calibrated by isotonic regression against held-out label data so that a predicted 0.3 means a 30% empirical formation rate. Acceptance criteria for deployment: area under the ROC curve of at least 0.85 on lag-corrected labels, and precision of at least 0.60 at the top 5% of ranked segments (the work-order cutoff). Retraining is monthly; features are recomputed daily.

### 8. Work-Order Issuance and Risk Weighting

Predicted probabilities are converted to expected cost: probability times the cost of a full pothole repair (crew dispatch, traffic control, material, plus a statistical allowance for vehicle-damage claims on high-speed segments), minus the cost of a preventive seal treatment. Segments are ranked by net expected savings, and work orders are issued for the top-ranked segments down to the crew's weekly capacity. Each work order includes the segment location, the predicted failure window, the driving features (e.g., "high-frequency raveling energy up 4 dB over 3 weeks; 12 moisture-gated freeze-thaw cycles since January"), and a recommended treatment (crack seal vs. patch vs. mill-and-fill, selected by the stiffness proxy).

The work-order module also feeds a public dashboard and a navigation-feed API: predicted high-risk segments are published as a data feed that navigation applications can use for routing advisories, and the dashboard shows the city's predicted-vs-actual pothole counts as a running accountability metric.

### 9. Implementation Notes

**Fleet recruitment:** The reference deployment uses municipal fleets (buses, garbage trucks, maintenance vehicles) whose routes cover the network repeatedly, supplemented by opted-in commercial delivery fleets. A city with 200 instrumented vehicles achieves full network coverage (every 50-meter segment traversed at least weekly) for a road network of approximately 10,000 lane-kilometers; the coverage calculation is 200 vehicles times 150 km per day divided by 2 (both directions), yielding 15,000 lane-km per day against a 10,000 lane-km network.

**Privacy:** Raw traces never leave the device in full; only per-second band energies and segment ids are transmitted. Segment ids are coarse (50 meters), traces are retained 7 days for debugging then deleted, and no driver identity is attached to vibration data. Aggregation requires a minimum of 5 distinct vehicles per segment-week before features are published, preventing single-vehicle tracking.

**Cold start:** A new city with no repair-log history bootstraps from a model trained on a climatically similar city, with probabilities discounted by a transfer factor (0.5) until 90 days of local labels accumulate. Cold-start predictions are flagged as low-confidence and excluded from automated work orders, shown on the dashboard only.

**Concept drift:** Resurfacing resets all counters and baselines for the segment. New vehicle models entering the fleet get a 14-day calibration burn-in. Climate shifts are absorbed by the moisture-gated counters, which are computed from observed weather, not historical normals.

## Claims

1. A method for predicting pothole formation on a road network, the method comprising: receiving, from a plurality of vehicles, accelerometer vibration data sampled at 50 Hz or higher together with position data; calibrating each vehicle's vibration data with a per-vehicle transfer function learned against a known-smooth reference segment, producing pavement-true excitation estimates; extracting, per road segment, a high-frequency raveling energy in the 30 to 80 Hz band and a stiffness proxy from the calibrated data; maintaining, per road segment, a moisture-gated freeze-thaw cycle count that increments only on freeze-thaw days when estimated surface moisture exceeds a threshold; and applying a trained machine learning model to the vibration features, the moisture-gated cycle count, traffic loading, and pavement age to produce a 30-day pothole formation probability per segment.
2. The method of claim 1, wherein the per-vehicle transfer function is re-estimated at least weekly from opportunistic passes over the reference segment, and wherein a deviation exceeding 3 dB from the vehicle's 30-day median transfer function triggers a maintenance flag and down-weighting of that vehicle's data.
3. The method of claim 1, wherein the high-frequency raveling energy is computed as band energy in the 30 to 80 Hz band after 4th-order Butterworth high-pass filtering at 0.5 Hz, and wherein a rising high-frequency energy with flat 0.5 to 8 Hz roughness energy is treated as a pre-spalling signature.
4. The method of claim 1, wherein the moisture-gated freeze-thaw count comprises three counters: a nominal freeze-thaw cycle count, a moisture-gated count incremented only when surface moisture exceeds a threshold, and a moisture-and-load-gated count weighted by heavy-vehicle traffic volume on freeze-thaw days.
5. The method of claim 1, further comprising constructing training labels from municipal repair logs with a label-lag correction that sets the event date to the earliest of the report date, the work-order date, or the first date the segment's stiffness proxy crossed its failure threshold, and evaluating predictions with a lag-corrected criterion that counts a true positive only when the prediction precedes the crew's own discovery by at least 7 days.
6. The method of claim 1, further comprising converting the 30-day formation probability into a net expected savings by subtracting preventive seal-treatment cost from probability-weighted full-repair cost including a vehicle-damage claim allowance, ranking segments by net expected savings, and issuing seal-and-repair work orders down to crew capacity.
7. The method of claim 1, further comprising publishing predicted high-risk segments as a navigation data feed for routing advisories in third-party navigation applications.
8. The method of claim 1, wherein raw accelerometer traces are summarized on the vehicle to per-second band energies before transmission, traces are retained no longer than 7 days, no driver identity is attached to vibration data, and segment features are published only when at least 5 distinct vehicles traversed the segment in the preceding week.
9. A system for predicting pothole formation, the system comprising: a fleet of vehicles each carrying an edge telemetry module sampling 3-axis acceleration at 50 Hz or higher and computing per-second frequency-band energies; a calibration module maintaining a per-vehicle suspension transfer function learned against a known-smooth reference segment; an environmental module maintaining per-segment moisture-gated freeze-thaw cycle counts from weather feeds; a prediction module fusing calibrated vibration features, environmental exposure, traffic loading, and pavement age into 30-day formation probabilities; and a work-order module ranking segments by risk-weighted expected cost and issuing preventive repair orders.
10. The method of claim 1, further comprising a cold-start fallback wherein, for a road network lacking local repair-log labels, predictions are generated by a model trained on a climatically similar network with probabilities discounted by a transfer factor, flagged as low-confidence, and excluded from automated work-order generation until 90 days of local labels accumulate.

## Prior Art References

1. Eriksson, Girod, Hull, Newton, Madden, and Balakrishnan, 2008: "The Pothole Patrol: Using a Mobile Sensor Network for Road Surface Monitoring," ACM MobiSys 2008, pp. 29-39 (pothole detection from taxi-mounted sensors; 500,000+ annual US insurance claims cited). [https://people.csail.mit.edu/hari/papers/p102.pdf](https://people.csail.mit.edu/hari/papers/p102.pdf)
2. Mednis, Strazdins, Zviedris, Kanonirs, and Selavo, 2011: "Real Time Pothole Detection Using Android Smartphones with Accelerometers," IEEE DCOSS 2011 (90% detection of marked potholes; Z-THRESH, Z-DIFF, STDEV(Z), G-ZERO algorithms). [https://doi.org/10.1109/DCOSS.2011.5982206](https://doi.org/10.1109/DCOSS.2011.5982206)
3. Mohan, Padmanabhan, and Ramjee, 2008: "Nericell: Rich Monitoring of Road and Traffic Conditions Using Mobile Smartphones," ACM SenSys 2008 (virtual sensor reorientation; bump and braking detection).
4. Carrera, Guerin, and Thorp, 2013: "By the People, for the People: The Crowdsourcing of StreetBump, an Automatic Pothole Mapping App," ISPRS Archives XL-4/W1 (Boston operational deployment from 2012; Innocentive algorithm competition). [https://doi.org/10.5194/isprsarchives-XL-4-W1-19-2013](https://doi.org/10.5194/isprsarchives-XL-4-W1-19-2013)
5. Boston StreetBump coverage: Discover Magazine (Boston's ~$2M annual pothole repair expenditure; one in six filled potholes publicly reported). [https://www.discovermagazine.com/want-the-city-to-fix-a-crater-of-doom-pothole-theres-an-app-for-that-29371](https://www.discovermagazine.com/want-the-city-to-fix-a-crater-of-doom-pothole-theres-an-app-for-that-29371)
6. Mohi ud Din, Mir, and Farooq, 2020: "Effect of Freeze-Thaw Cycles on the Properties of Asphalt Pavements in Cold Regions: A Review," Transportation Research Procedia (cohesion and adhesion failure mechanisms under F-T cycling). [https://doi.org/10.1016/j.trpro.2020.08.087](https://doi.org/10.1016/j.trpro.2020.08.087)
7. KTH Royal Institute of Technology: Thermomechanical framework for asphalt pavement performance under frost heave and thaw settlement (subgrade stiffness reduction after thaw). [https://www.kth.se/en/om/upptack/kalender/disputationer/a-mechanistic-framework-for-evaluating-the-performance-of-asphalt-pavements-subjected-to-frost-heave-and-thaw-settlement-1.1448369](https://www.kth.se/en/om/upptack/kalender/disputationer/a-mechanistic-framework-for-evaluating-the-performance-of-asphalt-pavements-subjected-to-frost-heave-and-thaw-settlement-1.1448369)
8. ASCE Journal of Cold Regions Engineering: Freeze-thaw cycle zoning with equivalent freeze-thaw cycles and k-means performance classification (precipitation-paired freeze risk for pothole distress). [https://ascelibrary.org/doi/10.1061/JCRGEI.CRENG-865](https://ascelibrary.org/doi/10.1061/JCRGEI.CRENG-865)
9. Douangphachanh and Oneyama, 2013: Smartphone-based road roughness estimation correlated to IRI (Eastern Asia Society for Transportation Studies).
10. Aleadelat and Ksaibati, 2017: Smartphone sensor variables predicting Pavement Serviceability Index for Wyoming county roads (R-squared above 0.9).
11. AAA pothole damage survey: ~$3B annual US driver costs from pothole damage.
12. Boston municipal reporting: ~$2M annual pothole repair expenditure in bad winters.
13. Michigan DOT historical record: 7,500 potholes patched in a single day during the February 2005 freeze-thaw crisis.
