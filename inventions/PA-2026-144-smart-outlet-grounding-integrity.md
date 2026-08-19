# PA-2026-144: System and Method for Continuous Residential Electrical Grounding Integrity Monitoring Using Distributed Smart Outlet Neutral-to-Ground Voltage Spectral Analysis with Fault Localization and Progressive Corrosion Detection

**Filing:** LITF-PA-2026-144  
**Domain:** Electrical Safety / Smart Home / Predictive Maintenance  
**Published:** August 19, 2026  
**Type:** Defensive Prior Art Disclosure  

---

## Abstract

Disclosed is a system and method for continuous monitoring of residential electrical grounding system integrity using distributed smart outlets equipped with voltage measurement capability. Each participating smart outlet periodically measures the root-mean-square (RMS) voltage between the neutral and equipment grounding conductors (N-G voltage) and performs spectral decomposition of this waveform to extract the fundamental 60 Hz component and its harmonics through the 15th order (900 Hz). Under normal conditions in a properly grounded NEC-compliant residential system, N-G voltage is below 2 V RMS and exhibits a clean 60 Hz waveform dominated by the fundamental. Degraded grounding connections introduce nonlinear contact resistance at corroded junctions, generating characteristic harmonic signatures that differ by fault type: oxide-film corrosion at screw terminals produces elevated odd harmonics (3rd, 5th, 7th) due to the voltage-dependent tunneling resistance of the oxide layer; soil-moisture-driven ground rod corrosion elevates even harmonics (2nd, 4th) from the asymmetric electrochemical half-cell; and loose backstab connections produce intermittent broadband spectral energy from micro-arcing. By collecting synchronized N-G voltage spectra from multiple smart outlets across the home and applying a graph-based electrical network model, the system localizes grounding faults to specific circuit branches, tracks corrosion progression rates over weeks to months, and generates maintenance alerts before the grounding system degrades to the point of creating shock or fire hazards.

## Technical Field

This invention relates to residential electrical safety monitoring, specifically to automated, continuous assessment of equipment grounding conductor integrity using voltage measurements from distributed smart outlet devices with spectral analysis for fault classification and localization.

## Background

The equipment grounding conductor (EGC) is the last line of defense against electrical shock in residential buildings. When a line-to-ground fault occurs in an appliance or fixture, the EGC provides a low-impedance return path to the service panel, enabling the overcurrent protection device (breaker or fuse) to trip and clear the fault. A compromised EGC leaves metal appliance enclosures, receptacle faceplates, and plumbing fixtures energized at lethal voltage during a ground fault. The National Electrical Code (NEC) Article 250 mandates grounding electrode systems, bonding, and EGC sizing precisely because this path must have low enough impedance to trip the breaker within the fault-clearing time specified in UL 489.

Grounding system degradation is widespread and largely invisible:

- **Ground rod corrosion:** Copper-clad steel ground rods corrode in acidic soils (pH < 5.5) and high-chloride environments. Marungsri et al. (IEEE, 2015) measured ground rod resistance increases of 200-400% over 15 years in corrosive soils. The Copper Development Association estimates that 15-25% of residential ground rods in the US exceed the NEC-recommended 25-ohm grounding electrode resistance.
- **Screw terminal corrosion:** Copper-to-steel and copper-to-aluminum junctions at receptacles, switches, and panel lugs develop oxide films over time. CPSC data attributes approximately 46,000 residential electrical fires annually to faulty wiring and connections, with corroded connections being a major contributing factor.
- **Backstab connection failures:** Spring-tension "backstab" connections used in inexpensive receptacles are prone to loosening. NFPA fire statistics show that electrical distribution equipment (including receptacles) is the third leading cause of home structure fires.
- **Bootleg grounds:** IAEI documents the practice of connecting neutral to ground at individual receptacles to fool three-light testers, creating a shock hazard by energizing the EGC during normal circuit operation.

Current methods for detecting grounding problems are episodic and manual:

- **Three-light receptacle testers:** Cost $5-$15. Can detect open grounds, open neutrals, and reversed polarity. Cannot detect bootleg grounds, high-impedance connections, or degraded ground rods.
- **Ground impedance testing:** Requires a dedicated ground impedance tester ($200-$2,000) and a qualified electrician. Rarely performed outside of new construction inspections.
- **Ground rod resistance testing (fall-of-potential method):** Requires driving auxiliary test stakes. Cost: $150-$500 per test. Performed at initial installation and almost never re-tested.

## Detailed Description

### 1. Neutral-to-Ground Voltage as a Grounding Health Indicator

In a residential 120/240V split-phase electrical system, the neutral conductor is bonded to the grounding electrode system at the main service panel (NEC 250.24(A)(1)). At this bonding point, neutral and ground are at the same potential. At any downstream receptacle, the N-G voltage equals the voltage drop along the neutral conductor from the outlet back to the panel, minus the voltage drop along the EGC over the same path (which is zero under normal conditions because no current flows in the EGC during fault-free operation).

Under normal loading, N-G voltage at a receptacle typically ranges from 0.5 to 2.0 V RMS, proportional to the load current on the circuit multiplied by the neutral conductor resistance (typically 0.1-0.5 ohms for 14 AWG copper over 50-100 foot runs).

When the EGC or grounding electrode system degrades, the N-G voltage changes in three diagnostically useful ways:

1. **Magnitude increase:** A high-impedance EGC allows voltage to develop across the ground conductor during normal operation due to capacitive coupling and stray currents.
2. **Waveform distortion:** Corroded junctions introduce nonlinear resistance (voltage-dependent). The resulting N-G voltage waveform acquires harmonic content absent in the purely resistive case.
3. **Temporal instability:** Loose connections produce intermittent contact, causing rapid fluctuations in N-G voltage magnitude and spectral content.

### 2. Smart Outlet Measurement Hardware

Consumer smart outlets with energy monitoring (e.g., TP-Link Kasa EP25, Shelly Plug S, Meross MSS310) already contain voltage measurement circuitry. The typical measurement chain consists of a resistive voltage divider from the hot conductor to a sigma-delta ADC (e.g., HLW8032, BL0937, or ADE7953) sampling at 3.2 kHz or higher with 16-bit resolution.

The disclosed system requires a firmware modification that adds one measurement channel: the voltage between the neutral and ground pins. A dedicated resistive divider from neutral to ground (100 kΩ / 1 kΩ, dissipating < 150 µW) feeds a GPIO ADC input on the microcontroller (ESP8266 or ESP32) at 12-bit resolution and 1 kHz sample rate. The measurement is non-invasive, draws negligible current through the grounding conductor, and does not compromise the safety function of the EGC.

Each outlet acquires 1-second N-G voltage waveform snapshots at configurable intervals (default: every 5 minutes). Each snapshot comprises 1,000 samples at 1 kHz, sufficient for spectral analysis through 500 Hz.

### 3. Spectral Decomposition and Fault Signature Classification

Each 1-second N-G voltage waveform is processed on-device using a 1024-point FFT with Hann windowing. The system extracts the following feature vector:

- Fundamental magnitude (60 Hz), V RMS
- Total harmonic distortion (THD) of the N-G voltage, percent
- Individual harmonic magnitudes: 2nd (120 Hz) through 15th (900 Hz)
- Odd-to-even harmonic ratio (OER)
- Spectral centroid of the harmonic content
- Inter-measurement variance (computed over the last 12 readings)
- Crest factor of the N-G voltage waveform

Fault type classification uses a random forest model (50 trees, max depth 8) trained on labeled data from controlled laboratory fault simulations. The fault taxonomy comprises five classes:

1. **Normal:** N-G RMS < 2.0 V, THD < 5%, low variance.
2. **Oxide-film corrosion (screw terminal):** N-G RMS 2-8 V, THD 10-25%, OER > 3.0 (strong odd harmonics). The voltage-dependent tunneling resistance of copper oxide (Cu₂O) and tin oxide (SnO₂) films produces predominantly odd-order harmonics.
3. **Electrochemical corrosion (ground rod):** N-G RMS 3-15 V, THD 8-20%, OER < 1.5 (elevated even harmonics). The asymmetric half-cell at the corroded electrode-soil interface generates even-order harmonics.
4. **Loose/intermittent connection (backstab):** High inter-measurement variance (> 30% CV), broadband spectral energy from micro-arcing, elevated crest factor (> 2.0).
5. **Bootleg ground (neutral-to-ground bond at outlet):** Anomalously low N-G voltage (< 0.1 V RMS) at a single outlet when other outlets on the same circuit show normal N-G voltage.

### 4. Distributed Fault Localization via Graph-Based Network Modeling

When multiple smart outlets are deployed across a home (3-8 typical), the system constructs an electrical network graph where nodes represent measurement points and edges represent wiring segments. Graph topology is inferred by correlating N-G voltage waveforms across outlets: outlets on the same circuit exhibit Pearson r > 0.9.

Fault localization analyzes the spatial pattern of N-G voltage anomalies:

- **Panel-side fault:** All outlets show elevated N-G voltage and similar harmonic signatures.
- **Branch circuit fault:** Outlets on one circuit affected; others normal.
- **Outlet-local fault:** Single outlet anomalous; adjacent outlets on same circuit normal.
- **Downstream fault:** Faults between two measurement points isolated by comparing their N-G voltage magnitudes.

The graph model uses a minimum spanning tree algorithm weighted by inter-outlet N-G voltage correlation, then applies Kirchhoff's voltage law constraints to solve for fault location and impedance magnitude.

### 5. Temporal Tracking and Predictive Maintenance

The system stores daily summary statistics for each outlet. A linear regression model on 90-day history detects progressive degradation trends. Alert thresholds:

- **Advisory (yellow):** N-G RMS > 3.0 V or THD > 10%. Recommend inspection within 30 days.
- **Warning (orange):** N-G RMS > 5.0 V, or variance > 40%, or trend projects crossing 5.0 V within 30 days. Recommend inspection within 7 days.
- **Critical (red):** N-G RMS > 10.0 V, or micro-arcing signature detected, or bootleg ground detected. Immediate hazard.

Environmental compensation correlates N-G voltage trends with temperature, humidity, and soil moisture data to distinguish seasonal fluctuations from irreversible corrosion progression.

### 6. Calibration and Self-Test

Daily self-test verifies measurement chain by comparing measured N-G voltage against expected value calculated from hot-neutral voltage, load current, and calibrated neutral conductor impedance. Initial calibration uses a known load step change to measure neutral conductor impedance to each outlet.

### 7. Implementation Notes

Firmware modification: ~8 KB flash for FFT library, feature extraction, and classifier. 4 KB RAM for FFT buffer. Data transmission: ~200 bytes per measurement (58 KB/day at 5-minute intervals). No additional hardware required beyond firmware update to existing three-prong smart outlets.

## Claims

1. A system for continuous monitoring of residential electrical grounding integrity, comprising: a plurality of smart outlet devices, each containing a voltage measurement circuit capable of measuring the voltage between the neutral conductor and the equipment grounding conductor; a spectral analysis module that performs frequency-domain decomposition of the measured neutral-to-ground voltage waveform to extract harmonic magnitudes through at least the 7th harmonic order; and a fault classification module that identifies the type of grounding degradation based on the relative magnitudes of odd-order and even-order harmonics, inter-measurement temporal variance, and waveform crest factor.

2. The system of claim 1, wherein oxide-film corrosion at screw terminals is identified by an odd-to-even harmonic ratio exceeding 3.0 in the neutral-to-ground voltage spectrum, reflecting the symmetric nonlinear resistance characteristic of metal oxide tunnel junctions.

3. The system of claim 1, wherein electrochemical corrosion of a grounding electrode is identified by elevated even-order harmonics in the neutral-to-ground voltage spectrum, reflecting the asymmetric electrochemical impedance of the corroded electrode-soil interface.

4. The system of claim 1, wherein loose or intermittent grounding connections are identified by a coefficient of variation in neutral-to-ground RMS voltage exceeding 30% over a sliding window, combined with broadband spectral energy above the 5th harmonic order indicative of micro-arcing at the contact interface.

5. The system of claim 1, wherein bootleg grounding (unauthorized neutral-to-ground bond at the outlet) is identified by anomalously low neutral-to-ground voltage at a single measurement point relative to other measurement points on the same branch circuit.

6. A method for localizing grounding faults in a residential electrical system, comprising: measuring neutral-to-ground voltage spectra at a plurality of smart outlet locations distributed across the electrical system; inferring branch circuit topology by computing pairwise correlation coefficients of neutral-to-ground voltage waveforms across all outlet pairs; constructing a graph-based electrical network model with measurement points as nodes and wiring segments as edges; and solving for the fault location and impedance magnitude using the spatial pattern of neutral-to-ground voltage anomalies constrained by Kirchhoff's voltage law relationships in the network model.

7. The method of claim 6, further comprising temporal trend analysis of daily neutral-to-ground voltage statistics over a 90-day window to detect progressive grounding degradation, with predictive alerting when linear regression projects that the neutral-to-ground voltage will exceed a safety threshold within a configurable forecast horizon.

8. The method of claim 7, further comprising environmental compensation by correlating neutral-to-ground voltage trends with ambient temperature, humidity, and soil moisture data to distinguish reversible seasonal variations in grounding electrode resistance from irreversible corrosion progression.

9. The system of claim 1, wherein each smart outlet performs a daily self-test by comparing its measured neutral-to-ground voltage against an expected value calculated from the hot-to-neutral voltage, measured load current, and historically calibrated neutral conductor impedance, flagging its own measurement as unreliable if the measured value deviates by more than 10% from the expected value.

10. A system for grounding integrity monitoring implemented as a firmware update to existing consumer smart outlets containing voltage measurement integrated circuits, wherein the firmware modification adds neutral-to-ground voltage measurement, on-device spectral analysis, fault classification, and network-coordinated fault localization without requiring additional hardware components beyond those present in the unmodified smart outlet.

11. The system of claim 1, further comprising a tiered alert system with advisory, warning, and critical thresholds calibrated to NEC-relevant safety limits, wherein advisory alerts are generated when neutral-to-ground RMS voltage exceeds 3.0 V, warning alerts when it exceeds 5.0 V or when intermittent connection signatures are detected, and critical alerts when it exceeds 10.0 V or when micro-arcing spectral signatures are present.

## References

1. NFPA 70 (National Electrical Code) — Article 250: Grounding and Bonding
2. Marungsri et al., IEEE 2015 — Ground rod corrosion and resistance degradation field study
3. Copper Development Association — Practical grounding electrode guidelines
4. CPSC Electrical Wiring Fire Data — ~46,000 annual residential electrical fires
5. NFPA Fire Statistics — Electrical distribution equipment as third leading fire cause
6. IAEI Magazine — Bootleg grounds detection challenges and hazard documentation
7. ECM Magazine — Limitations of three-light testers for bootleg ground detection
8. UL 489 — Molded-case circuit breaker fault-clearing time requirements
9. Braunovic et al., IEEE Holm Conference 2003 — Contact resistance of corroded connections under AC loading
10. Espressif ESP32 SoC — Microcontroller specifications for smart outlet hardware
11. IEEE Std 142-2007 (Green Book) — Grounding of Industrial and Commercial Power Systems
12. NFPA Home Electrical Fire Report — Detailed electrical fire cause breakdown
