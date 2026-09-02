# PA-2026-159: Predicting Gas Oven Hot-Surface Igniter Failure and Delayed-Ignition Hazard Using Smart Plug Current Signature Trend Analysis and Acoustic Ignition Transient Classification

**Title:** System and Method for Predicting Gas Oven Hot-Surface Igniter Failure and Delayed-Ignition Hazard Using Smart Plug Current Signature Trend Analysis and Acoustic Ignition Transient Classification

**Filing:** LITF-PA-2026-159  
**Published:** September 2, 2026  
**Domain:** Appliance Safety / Predictive Maintenance / Edge AI  
**Full Disclosure:** [liveinthefuture.org/priorart/gas-oven-igniter-wear-prediction.html](https://liveinthefuture.org/priorart/gas-oven-igniter-wear-prediction.html)

---

## Abstract

Disclosed is a system that predicts failure of hot-surface (glow bar) igniters in residential gas ovens using only a commodity energy-monitoring smart plug on the range's 120V outlet, optionally fused with acoustic ignition transient classification from a kitchen smart speaker. A healthy silicon carbide igniter draws 3.2 to 3.6 A, and the series-wired gas safety valve requires approximately 3.0 A to open. As the igniter ages its resistance rises and current falls; below roughly 2.9 A the valve opens intermittently, admitting unburned gas that ignites seconds later in a delayed-ignition event audible as a low-frequency whoomp. The system samples RMS current at 1 Hz, extracts per-cycle features including steady-state current, preheat-to-ignition time, valve-open current dip, and re-strike count, fits a degradation trend that forecasts the 3.0 A crossing 2 to 8 weeks in advance, and classifies ignition acoustics to distinguish normal soft ignition from delayed-ignition pressure transients concentrated at 40 to 120 Hz. Differential diagnosis separates igniter wear (declining current trend) from safety-valve or control-board faults (normal current with no ignition), directing replacement of the correct $15 to $40 part instead of a misdiagnosed $150+ service call. Deployment requires no electrician and no gas line work.

## Field of the Invention

This invention relates to residential appliance prognostics, specifically to non-invasive prediction of gas oven hot-surface igniter end of life and detection of the delayed-ignition safety hazard, using external electrical current signature analysis and acoustic event classification without disassembly of the appliance.

## Background

Most residential gas ovens built since the 1980s use a hot-surface (glow bar) igniter wired in series with the oven gas safety valve. The igniter must heat to incandescence and simultaneously pass enough current to warp the bimetal element inside the safety valve open; no adequate igniter current means no gas flow. The igniter is the most frequently replaced part in gas ovens per appliance repair trade sources. Its failure mode is gradual: thermal cycling and oxidation raise element resistance over years of service, so current draw declines from a healthy 3.2 to 3.6 A toward the valve's operating threshold near 3.0 A. Trade references state the test explicitly: a properly working igniter draws 3.0 to 3.4 A, any igniter drawing under 3.0 A is weak and should be replaced, and ignition should occur within 60 seconds.

The dangerous regime is the marginal band around 2.8 to 3.0 A. The valve opens intermittently, admits unburned gas the weakened igniter fails to light promptly, and ignition occurs seconds late as a single explosive event, described by technicians as a loud whoomp that can rattle the oven door, stress door glass and hinges, and expose nearby users to a fireball from the oven cavity.

Current practice detects none of this in advance. Homeowners discover wear only when the oven stops heating, and diagnosis requires a service visit or a clamp meter on live 120V wiring inside the appliance. Smart plugs with energy monitoring already measure the exact signal a technician measures, RMS current at the outlet, but no existing system interprets it as an igniter degradation trend or fuses it with acoustic delayed-ignition detection.

## Detailed Description

### 1. System Architecture Overview

Two sensing paths feed a prognostics engine on a home hub, smartphone, or cloud service. Path A is electrical: a commodity energy-monitoring smart plug (15 A / 1800 W, 1 Hz or faster RMS current reporting, local MQTT or equivalent API) between the wall outlet and the range's 120V cord. Path B is acoustic and optional: an existing kitchen smart speaker or smartphone microphone capturing the 2 to 5 second ignition transient at each cycle start. The engine extracts per-cycle features from Path A, maintains a longitudinal wear trend per igniter (bake and broil tracked separately), classifies each ignition acoustic event via Path B, and issues tiered alerts.

### 2. Igniter Physics and the Current Threshold

A new silicon carbide glow bar draws 3.2 to 3.6 A at 120V in steady state. The safety valve's bimetal actuator opens only when series current exceeds approximately 2.9 to 3.1 A (nominal 3.0 A design point). Aging mechanisms: silicon carbide oxidation at grain boundaries raising bulk resistance, and micro-cracking from thermal cycling reducing effective cross-section. Three regimes: above 3.1 A normal operation with ignition in 20 to 60 seconds; 2.8 to 3.1 A marginal operation with lengthening preheat-to-ignition and the onset of delayed-ignition events; below 2.8 A the valve never opens while the igniter still glows, misleading owners into suspecting the gas supply.

### 3. Current Signature Acquisition and Feature Extraction

A bake cycle presents a stereotyped signature: near-zero baseline (controls under 0.3 A), sharp rise to the igniter plateau within 2 seconds of relay closure, flat preheat plateau at steady-state current for 20 to 90 seconds, a small downward step of 0.1 to 0.3 A when the safety valve opens, then burner-on cycling with 20 to 45 second re-energizations per thermostat cycle. Per-cycle features: steady-state igniter current I_ss (median over plateau excluding 3 s inrush, the primary wear indicator); preheat-to-ignition time T_ign (relay close to valve-open dip); valve-open dip magnitude (absent when the valve fails to open); re-strike count beyond normal thermostat cadence; plateau variance indicating threshold teetering and valve chatter. Cycle detection: current above 1.5 A sustained at least 15 seconds. Self-clean cycles excluded from trend by duration.

### 4. Degradation Trending and Remaining Useful Life

Per-igniter I_ss time series with 7-cycle median filtering; piecewise-linear fit over trailing 60 days estimates wear rate dI/dt. Remaining useful life = (I_current minus 3.05 A) / wear rate, with the alert threshold at 3.05 A for margin above the 2.9 A hazard band. A design embodiment assumes 0.05 to 0.25 A/year, giving roughly 8 weeks of warning at 3.15 A for slow wear and 2 weeks for fast wear. Secondary rule: three consecutive cycles with T_ign over 60 seconds triggers inspection advice regardless of trend. RUL display requires at least 10 bake cycles over 14 days.

### 5. Acoustic Delayed-Ignition Detection

Normal ignition is a soft sub-0.5-second event. Delayed ignition shows a 1 to 3 second gas-flow hiss or whoosh terminated by a sharp low-frequency pressure transient with peak energy at 40 to 120 Hz, sometimes with door rattle. The acoustic path records only a 10-second window opened by electrical relay-close detection, so no continuous kitchen audio is captured. A compact INT8 CNN (under 200 KB) on 64-bin mel spectrograms classifies normal ignition vs delayed ignition vs non-ignition background, running on the smart speaker or phone. A confirmed delayed-ignition event escalates to the hazard tier immediately, independent of the current trend.

### 6. Differential Diagnosis

Three fault classes presenting identically as "oven not heating": igniter wear (declining I_ss trend, lengthening T_ign, progressing acoustics) -> $15 to $40 owner-replaceable igniter, no gas fittings touched; safety valve failure (healthy steady current, no valve-open dip, no ignition acoustics) -> technician valve repair; control board or relay failure (no current rise on heat command, or sub-second relay chatter) -> board diagnosis. The diagnosis arrives with the evidence attached, eliminating the most common misdiagnosis path.

### 7. Alerting Tiers

Healthy (I_ss above 3.2 A): silent with trend visible in app. Monitor (3.05 to 3.2 A, or threshold crossing projected within 8 weeks): in-app notice with estimated weeks remaining and the model-specific part number. Replace soon (2.9 to 3.05 A, or repeated T_ign over 60 s): push notification advising replacement within 2 weeks. Hazard (acoustically confirmed delayed ignition, or I_ss below 2.9 A): urgent alert advising discontinued oven use until replacement, with logged event timestamps for the technician. Setup: plug range through smart plug, run one bake cycle for baseline, optionally enable the acoustic path.

## Claims

1. A system for predicting hot-surface igniter failure in a residential gas oven, comprising: an energy-monitoring smart plug connected between a wall outlet and the gas range's power cord, the plug reporting RMS current at 1 Hz or faster; and a prognostics engine that detects bake and broil cycles from the current waveform, extracts per-cycle steady-state igniter current, and fits a longitudinal degradation trend that forecasts crossing of a valve-opening current threshold.
2. The system of claim 1, wherein the valve-opening current threshold is set between 2.9 and 3.1 A, and wherein the engine issues a replacement advisory when the trended steady-state current is projected to cross the threshold within 8 weeks.
3. The system of claim 1, wherein the engine extracts preheat-to-ignition time from the interval between relay closure and a valve-open current dip of 0.1 to 0.3 A, and triggers an inspection advisory after three consecutive cycles exceeding 60 seconds.
4. The system of claim 1, further comprising an acoustic sensor that captures a time-bounded audio window opened by electrical relay-close detection, and a classifier that distinguishes normal ignition transients from delayed-ignition events characterized by a gas-accumulation hiss followed by a pressure transient with peak energy at 40 to 120 Hz.
5. The system of claim 4, wherein a classified delayed-ignition event escalates the alert to a hazard tier advising discontinuation of oven use until igniter replacement, independent of the current trend value.
6. The system of claim 1, wherein the engine performs differential diagnosis among igniter wear indicated by declining steady-state current, safety-valve failure indicated by healthy current with absent valve-open dip and absent ignition acoustics, and control-board failure indicated by absent current rise on heat command.
7. The system of claim 1, wherein bake and broil igniters are tracked as separate degradation trends, and self-clean cycles are identified by duration and excluded from trend fitting.
8. The system of claim 1, wherein remaining useful life is computed as the difference between current trended steady-state current and the threshold divided by the trailing wear rate, and is displayed only after a minimum of 10 cycles over at least 14 days.
9. The system of claim 1, wherein the engine detects threshold teetering from elevated plateau current variance during preheat, indicating safety-valve chatter near its operating point.
10. A method for non-invasive gas oven igniter prognostics comprising: measuring externally, at the range power cord, the RMS current waveform of bake and broil cycles without disassembly of the appliance; extracting per-cycle steady-state igniter current and preheat-to-ignition time; maintaining a per-igniter wear trend; forecasting the date at which the trend crosses the gas safety valve operating current; and issuing a tiered alert culminating in a hazard advisory upon acoustic confirmation of delayed ignition.
11. The method of claim 10, further comprising capturing a relay-triggered bounded audio window at ignition and classifying it with an on-device neural network under 200 KB to confirm delayed-ignition hazard events without continuous audio recording.

## Prior Art References

1. Fred's Appliance Academy, "How to test a gas range ignitor" (https://academy.fredsappliance.com/cooking/how-to-test-a-gas-range-ignitor/) — 3.0 to 3.4 A good, under 3.0 A weak, ignition within 60 seconds
2. Fred's Appliance Academy, "Hot surface ignition in gas ranges" (https://academy.fredsappliance.com/video/hot-surface-ignition-in-gas-ranges/) — igniter teetering at 2.9 A fails to open the gas valve; gas buildup with audible whooshing ends in delayed ignition
3. Bob Vila, "How to replace an oven igniter" (https://www.bobvila.com/diy/how-to-replace-oven-igniter/) — good igniter pulls 3.0 to 3.4 A; the igniter doubles as the gas safety switch
4. InspectApedia, "Gas cooktop and stove igniter repair diagnostics" (https://inspectapedia.com/Appliances/Gas_Stove_Igniter_Repair.php)
5. EIA Residential Energy Consumption Survey (https://www.eia.gov/consumption/residential/)
