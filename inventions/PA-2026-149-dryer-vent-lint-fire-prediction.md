# System and Method for Predictive Residential Clothes Dryer Exhaust Vent Lint Accumulation and Blockage Detection Using Exhaust Air Temperature, Humidity, and Ultrasonic Airflow Velocity Multi-Sensor Fusion with Predictive Fire Risk Scoring

**LITF-PA-2026-149 · Home Safety / Predictive Maintenance / Fire Prevention**
**Published:** 2026-08-24
**License:** [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/) — Public Domain

> ⚖️ **Prior Art Notice:** This document is published as defensive prior art under [35 U.S.C. § 102(a)(1)](https://www.law.cornell.edu/uscode/text/35/102). The inventions described herein are dedicated to the public domain as of the publication date above.

---

## Abstract

Disclosed is a non-invasive sensor module and accompanying method for continuously monitoring residential clothes dryer exhaust vent systems to detect progressive lint accumulation, predict impending blockage, and compute a real-time fire risk score. The module clips to the exterior vent terminus without modification to the dryer or ductwork. It fuses three independent sensing modalities: (1) exhaust-to-ambient air temperature differential measured via matched NTC thermistors, which decreases as lint insulates the duct and restricts airflow; (2) exhaust relative humidity transport efficiency, computed as the ratio of exhaust humidity elevation to dryer runtime, which degrades as blockage forces moisture back into the living space; and (3) ultrasonic transit-time airflow velocity at the vent terminus, measured by a pair of 40 kHz piezoelectric transducers oriented along the duct axis, which declines proportionally to cross-sectional area reduction from lint deposition. An on-device gradient-boosted decision tree model (22 KB, running on an ESP32-C3) fuses these three sensor streams with dryer cycle metadata (frequency, duration, time-of-day patterns) to estimate cumulative lint mass in the duct, predict days-to-critical-blockage, and output a Fire Risk Index (0-100). The system communicates via Matter-over-Thread to trigger smart home alerts, dryer operation inhibition via smart plug control, and professional cleaning service scheduling. Calibration is self-supervised: the system establishes its clean-duct baseline during the first three drying cycles after installation.

## Field of the Invention

This invention relates to residential fire prevention and predictive maintenance, specifically to automated monitoring of clothes dryer exhaust vent systems for lint accumulation using multi-sensor fusion and machine learning-based fire risk prediction.

## Background

The U.S. Fire Administration (USFA) reports that clothes dryers cause an estimated [2,900 residential structure fires annually](https://www.usfa.fema.gov/statistics/residential-fires/dryer-fires/), resulting in approximately 5 deaths, 100 injuries, and $35 million in property damage per year. The [National Fire Protection Association (NFPA)](https://www.nfpa.org/education-and-research/research/nfpa-research/fire-statistical-reports/home-fires-involving-clothes-dryers-and-washing-machines) identifies failure to clean the dryer vent as the leading contributing factor in 34% of these fires. Lint is combustible (ignition temperature approximately 210-250°C), and accumulated lint in exhaust ducts creates the three conditions for fire simultaneously: fuel (cellulose and synthetic fibers), oxygen (from dryer airflow), and heat (exhaust temperatures of 57-80°C at the heating element, potentially exceeding 135°C at duct restrictions where turbulent recirculation concentrates thermal energy).

Current approaches to dryer vent maintenance are inadequate:

- **Manual inspection:** Homeowners are advised to clean dryer vents annually. Compliance is low; a 2019 NFPA survey found fewer than 30% of households had cleaned their dryer vent in the past 12 months. Visual inspection requires physical access to the full duct run, often routed through walls, attics, or crawlspaces.
- **Lint trap indicators:** Some dryers include a "check lint filter" indicator that detects lint screen blockage via airflow or pressure differential. This monitors only the lint screen, not the exhaust duct, where the majority of fire-causing accumulation occurs downstream. [US7827705B2](https://patents.google.com/patent/US7827705B2) (LG Electronics) discloses a dryer exhaust temperature sensor for duct blockage detection, but relies on a single sensing modality (temperature) with no predictive modeling, no lint accumulation estimation, and no fire risk quantification.
- **Professional cleaning services:** Dryer vent cleaning costs $100-200 per visit and is recommended annually. Without monitoring data, homeowners cannot assess whether cleaning is needed sooner or later than the annual default, resulting in either unnecessary service calls or dangerous overdue maintenance.
- **Booster fan systems:** Powered duct booster fans (e.g., DryerFlex, Tjernlund LB2) compensate for long duct runs but do not monitor lint accumulation. They mask the symptoms of blockage (reduced airflow) while the underlying fire risk continues to increase. [US10753033B2](https://patents.google.com/patent/US10753033B2) (Samsung) describes a dryer with airflow sensing for lint screen blockage but does not extend sensing to the exhaust duct system, does not perform multi-sensor fusion, and does not compute predictive fire risk.

The gap in the art is a complete, non-invasive, consumer-installable system that: (a) monitors the exhaust duct rather than just the lint screen, (b) fuses multiple independent sensing modalities to distinguish lint accumulation from other airflow changes (wind, seasonal temperature variation, duct configuration), (c) estimates cumulative lint mass and predicts time-to-critical-blockage, (d) computes a quantitative fire risk score, and (e) integrates with smart home ecosystems for automated response.

## Detailed Description

### 1. Sensor Module Hardware

The sensor module is designed to clip onto the exterior terminus of a standard 4-inch (102 mm) residential dryer exhaust vent without tools or modification to the existing duct system. The module comprises:

- **Exhaust temperature sensor:** An NTC thermistor (10 kΩ at 25°C, ±0.5°C accuracy, e.g., Murata NCP15XH103F03RC, $0.12) positioned in the exhaust airflow path via a stainless steel probe extending 15 mm into the duct interior through an existing louver gap.
- **Ambient temperature sensor:** A matched NTC thermistor mounted on the exterior housing, shielded from direct sunlight and exhaust heat by a vented baffle. The matched pair enables differential measurement that cancels ambient temperature drift.
- **Exhaust humidity sensor:** A capacitive humidity sensor (e.g., Sensirion SHT40, ±1.8% RH, $1.80) positioned adjacent to the exhaust thermistor in the airflow path.
- **Ambient humidity sensor:** A matched capacitive sensor on the exterior housing for differential humidity computation.
- **Ultrasonic airflow transducer pair:** Two 40 kHz piezoelectric ceramic transducers (e.g., Murata MA40S4S, $1.50 each) mounted on opposite sides of the duct terminus, angled 45° to the airflow axis. Transit-time measurement: the upstream-to-downstream transit time is shorter than downstream-to-upstream by Δt = 2·L·v·cos(θ) / (c² - v²), where L is transducer spacing (100 mm), v is airflow velocity, θ is transducer angle (45°), and c is speed of sound (~343 m/s). At typical dryer exhaust velocity (3-6 m/s), Δt is 1.2-2.4 μs, resolvable with a 12 MHz timer on the ESP32-C3.
- **Microcontroller:** ESP32-C3 (RISC-V core, WiFi/BLE, Thread radio via 802.15.4, $1.80). Runs sensor sampling, ML inference, and Matter-over-Thread communication stack.
- **Power:** Two CR123A lithium batteries (3V, 1,500 mAh each, series for 6V regulated to 3.3V). Expected battery life: 18-24 months at 1 measurement cycle per dryer run plus hourly ambient baseline samples. Deep sleep current: 5 μA. Active measurement cycle: 120 mA for 8 seconds per dryer event detection.
- **Dryer run detection:** The exhaust temperature sensor detects dryer activation when ΔT (exhaust minus ambient) exceeds 8°C for more than 60 seconds. This triggers the full measurement cycle and transitions the module from deep sleep to active sensing. False triggers from solar heating of the vent hood are rejected by requiring simultaneous humidity elevation above ambient.

Target bill-of-materials cost: $18-24. Retail price target: $49-69.

### 2. Temperature Differential Sensing

During each dryer cycle, the module samples exhaust and ambient temperatures at 1 Hz. It computes two derived metrics:

- **Peak ΔT:** The maximum temperature differential during steady-state drying (after the initial 3-minute ramp). In a clean 4-inch duct up to 25 feet long, peak ΔT typically ranges from 25-45°C depending on dryer model, load size, and cycle setting. As lint accumulates, two competing effects occur: (a) reduced airflow velocity decreases convective heat transport, lowering exhaust temperature; (b) lint insulation on duct walls retains heat, potentially raising it locally. The net effect at the terminus is a characteristic decline in peak ΔT of 0.5-1.5°C/month under normal use, accelerating nonlinearly as blockage exceeds 40% cross-sectional area reduction.
- **ΔT decay time constant (τ):** After the dryer stops, the exhaust temperature decays exponentially toward ambient. The time constant τ depends on duct thermal mass and airflow. A clean duct with good natural convection cools with τ ≈ 90-180 seconds. Lint accumulation increases τ by both adding thermal mass (lint acts as insulation with thermal conductivity ~0.04 W/m·K, similar to fiberglass) and reducing convective airflow. Monitoring τ over time provides an independent lint mass estimator that is less sensitive to ambient conditions than peak ΔT.

### 3. Humidity Transport Efficiency

A residential clothes dryer extracts 2-5 kg of water per load, depending on fabric type, load mass, and initial moisture content. This moisture must exit through the exhaust duct. The module computes Humidity Transport Efficiency (HTE) as:

HTE = (∫ΔRH · v dt) / T_cycle

where ΔRH is the exhaust-to-ambient relative humidity differential, v is ultrasonic-measured airflow velocity, and T_cycle is total cycle duration. In a clean duct, HTE is high (efficient moisture removal). As lint accumulates, three effects reduce HTE: (a) reduced airflow carries less moisture per unit time; (b) lint absorbs and retains moisture, creating a slow-release reservoir that extends the humidity tail after the dryer stops; (c) excessive blockage forces humid air to leak back into the living space through the dryer drum seal, reducing the fraction exiting via the duct. The extended humidity tail after dryer shutoff is a particularly sensitive early indicator of lint accumulation, detectable before significant airflow reduction occurs.

### 4. Ultrasonic Airflow Velocity Measurement

The paired 40 kHz transducers perform transit-time airflow measurement. Each measurement cycle fires 8 burst pulses upstream-to-downstream and 8 downstream-to-upstream, computing the transit time difference via zero-crossing detection on the received waveform. The measurement is averaged over 16 bidirectional cycles (total: 32 firings, ~200 ms) to reduce noise.

Clean-duct baseline exhaust velocity for residential dryers ranges from 3.0-6.5 m/s depending on duct length, number of elbows, and dryer blower capacity. The International Residential Code (IRC M1502.6) requires a minimum exhaust velocity of 1.5 m/s. As lint reduces the effective cross-sectional area, airflow velocity at the terminus changes in a characteristic pattern: initially increasing slightly (same volume flow through smaller area) before decreasing as the dryer blower reaches its stall pressure and total volume flow drops. This non-monotonic velocity profile is a diagnostic signature that distinguishes lint accumulation from other causes of reduced airflow (e.g., bird nest obstruction, which causes immediate velocity drop without the initial increase phase).

Wind-induced velocity offsets are compensated by measuring airflow velocity during non-dryer periods (hourly ambient baselines) and subtracting the wind component vector from dryer-active measurements.

### 5. Multi-Sensor Fusion and Fire Risk Model

The on-device gradient-boosted decision tree ensemble (LightGBM, quantized to INT8, 22 KB model size) fuses the following 18-dimensional feature vector computed per dryer cycle:

1. Peak ΔT (current cycle)
2. Peak ΔT normalized to ambient temperature (compensates seasonal variation)
3. ΔT decay time constant τ
4. Humidity Transport Efficiency (HTE)
5. Post-cycle humidity tail duration (time for ΔRH to fall below 5%)
6. Peak exhaust airflow velocity
7. Mean exhaust airflow velocity
8. Velocity profile shape descriptor (monotonic decrease vs. non-monotonic lint signature)
9. Cycle duration
10. Time since last cycle (inter-cycle interval)
11. Cumulative cycles since last cleaning event (or installation)
12. Rolling 7-day cycle count
13. Rolling 30-day mean peak ΔT trend (slope)
14. Rolling 30-day mean HTE trend (slope)
15. Rolling 30-day mean peak velocity trend (slope)
16. Ambient temperature (seasonal context)
17. Ambient humidity (seasonal context)
18. Wind speed estimate (from non-dryer airflow measurements)

The model outputs three quantities:

- **Estimated lint accumulation (0-100% blockage):** Calibrated against laboratory measurements of lint mass vs. cross-sectional area reduction in 4-inch smooth and flex duct. Training data generated by progressively inserting known masses of dryer lint (0-500 g in 25 g increments) into duct sections of varying length (5-35 feet) and configuration (0-4 elbows).
- **Days to critical blockage:** Extrapolation from the current accumulation rate, accounting for seasonal usage patterns. Critical blockage is defined as >70% cross-sectional area reduction, corresponding to the empirical threshold at which exhaust temperature at the restriction point can exceed lint ignition temperature (210°C) during high-heat cycles.
- **Fire Risk Index (FRI, 0-100):** Composite score fusing estimated blockage percentage, days-to-critical, recent cycle frequency (more cycles = more lint + more ignition opportunities), and dryer duty cycle (longer cycles indicate the dryer is working harder against restricted airflow, generating more heat). FRI thresholds: 0-25 (green, normal), 26-50 (yellow, schedule cleaning within 30 days), 51-75 (orange, clean within 7 days), 76-100 (red, stop using dryer immediately, fire risk acute).

### 6. Self-Supervised Calibration

The system requires no manual entry of duct length, configuration, or dryer model. During the first three drying cycles after installation (or after a cleaning event, detected as a sudden improvement in all three sensor channels), the system records baseline values for peak ΔT, HTE, and airflow velocity. All subsequent measurements are normalized to this baseline, enabling the system to function across the wide range of residential duct installations (smooth metal, semi-rigid aluminum, flex duct; 5-35 feet; 0-4 elbows; horizontal, vertical, or mixed runs).

A cleaning event is automatically detected when all three sensor channels simultaneously improve by more than 15% relative to the prior 7-day rolling average. The system resets its baseline and restarts the accumulation model from zero. This eliminates the need for manual "I just cleaned my vent" button presses.

### 7. Smart Home Integration and Automated Response

The module communicates via Matter-over-Thread protocol (IEEE 802.15.4 radio on ESP32-C3), enabling integration with all Matter-compatible smart home ecosystems (Apple Home, Google Home, Amazon Alexa, Samsung SmartThings). Automated response capabilities include:

- **Progressive alerts:** Push notifications at FRI threshold crossings (yellow/orange/red) with specific recommended actions and estimated time remaining before critical risk.
- **Dryer operation inhibition:** At FRI ≥ 76 (red), the system sends a Matter command to a smart plug controlling the dryer's power outlet, preventing dryer operation until the vent is cleaned. This requires the dryer to be plugged into a Matter-compatible smart plug (240V for electric dryers, which represents the majority of US residential dryers).
- **Professional service scheduling:** API integration with dryer vent cleaning service providers. When FRI reaches orange, the system can automatically request a quote or schedule a cleaning appointment via the provider's API, with homeowner approval required via push notification.
- **Insurance and home warranty reporting:** Opt-in continuous monitoring data export for home insurance providers, potentially enabling premium discounts for homes with verified clean dryer vents (analogous to monitored smoke detector discounts).

## Claims

1. A system for predictive detection of lint accumulation in residential clothes dryer exhaust vent systems, comprising: a sensor module attachable to the exterior terminus of a dryer exhaust vent without modification to the dryer or ductwork; said module containing at least a temperature sensor pair measuring exhaust-to-ambient temperature differential, a humidity sensor pair measuring exhaust-to-ambient relative humidity differential, and an ultrasonic transducer pair measuring exhaust airflow velocity via transit-time measurement; and a microcontroller running an on-device machine learning model that fuses data from said sensors to estimate cumulative lint accumulation and compute a fire risk score.

2. The system of claim 1, wherein the ultrasonic airflow velocity measurement uses a pair of 40 kHz piezoelectric transducers mounted at 45° to the duct axis, performing bidirectional transit-time measurement to determine airflow velocity independent of temperature and humidity conditions.

3. The system of claim 1, wherein the machine learning model is a gradient-boosted decision tree ensemble processing an 18-dimensional feature vector that includes peak temperature differential, temperature decay time constant, humidity transport efficiency, post-cycle humidity tail duration, airflow velocity profile shape, cycle metadata, and rolling trend slopes.

4. The system of claim 1, wherein the system detects a non-monotonic airflow velocity profile signature characteristic of progressive lint accumulation, distinguishing it from sudden obstructions that produce monotonic velocity decline.

5. The system of claim 1, wherein the system performs self-supervised calibration by recording baseline sensor values during the first three drying cycles after installation or after automatic detection of a cleaning event, enabling operation across varying duct configurations without manual parameter entry.

6. The system of claim 1, wherein a cleaning event is automatically detected when all three sensor channels simultaneously improve by more than a predetermined threshold relative to a rolling average, triggering baseline reset and accumulation model restart without manual input.

7. The system of claim 1, wherein the system communicates via Matter-over-Thread protocol and generates progressive alerts at Fire Risk Index threshold crossings, with the capability to inhibit dryer operation by sending a control command to a smart plug when fire risk exceeds a critical threshold.

8. A method for predicting fire risk in residential clothes dryer exhaust vent systems, comprising: measuring exhaust-to-ambient temperature differential, humidity transport efficiency, and ultrasonic airflow velocity at the vent terminus during each drying cycle; computing rolling trend slopes for each sensor modality over configurable time windows; fusing said measurements and trends with cycle frequency and duration metadata using an on-device machine learning model; outputting an estimated lint blockage percentage, predicted days to critical blockage, and a composite Fire Risk Index; and generating automated alerts or dryer operation inhibition commands based on Fire Risk Index thresholds.

9. The method of claim 8, further comprising computing a post-cycle humidity tail duration as the time for exhaust-to-ambient humidity differential to fall below a threshold after dryer shutoff, wherein extended humidity tail duration serves as an early indicator of lint accumulation detectable before significant airflow reduction occurs.

10. The method of claim 8, further comprising wind compensation by periodically measuring airflow velocity at the vent terminus during non-dryer periods and subtracting the wind component from dryer-active measurements.

11. The system of claim 1, wherein the sensor module has a bill-of-materials cost below $25, is powered by replaceable batteries with expected life exceeding 18 months, and requires no professional installation, wiring, or duct modification.

## Implementation Notes

The dryer vent fire problem sits in the gap between two well-funded industries that have no incentive to solve it. Appliance manufacturers focus on the lint screen because it is inside their product boundary. Insurance companies pay claims because individual dryer vent fires are small relative to total loss pools. Neither has economic motivation to deploy $50 sensors that prevent $12,000 average-loss fires.

The sensor fusion approach described here specifically avoids the single-sensor fragility of prior art. Temperature alone cannot distinguish lint accumulation from seasonal ambient variation or a change in dryer heat setting. Airflow alone cannot distinguish lint from a crushed duct section or wind conditions. Humidity alone varies with load size and fabric type. But the joint trajectory of all three channels over time produces a degradation signature that is uniquely attributable to lint accumulation, with the non-monotonic velocity profile providing a physics-based fingerprint that no other failure mode produces.

The 40 kHz ultrasonic transit-time approach was chosen over alternatives (hot-wire anemometer, differential pressure, Pitot tube) because it has no moving parts, no elements in the airflow path that could collect lint, and no calibration drift from lint fouling. The transducers are mounted flush with the duct wall, outside the airflow boundary layer.

Matter-over-Thread was chosen as the communication protocol because it operates on 802.15.4 mesh networking (low power, no WiFi credentials needed), is increasingly standard in smart home ecosystems (Apple, Google, Amazon, Samsung all support it as of 2025), and enables the dryer operation inhibition use case through direct device-to-device communication without cloud dependency.

## Prior Art References

1. [USFA — Clothes Dryer Fires in Residential Buildings](https://www.usfa.fema.gov/statistics/residential-fires/dryer-fires/) — 2,900 fires/year, 5 deaths, 100 injuries
2. [NFPA — Home Fires Involving Clothes Dryers and Washing Machines](https://www.nfpa.org/education-and-research/research/nfpa-research/fire-statistical-reports/home-fires-involving-clothes-dryers-and-washing-machines) — failure to clean as leading factor (34%)
3. [US7827705B2](https://patents.google.com/patent/US7827705B2) (LG Electronics) — Dryer exhaust temperature sensor for duct blockage detection
4. [US10753033B2](https://patents.google.com/patent/US10753033B2) (Samsung) — Dryer with airflow sensing for lint screen blockage detection
5. [IRC M1502](https://codes.iccsafe.org/content/IRC2021P7/chapter-15-exhaust-systems) — International Residential Code, Clothes Dryer Exhaust requirements
6. [ESP32-C3 SoC](https://www.espressif.com/en/products/socs/esp32-c3) — Espressif RISC-V microcontroller with 802.15.4 radio
7. [Murata MA40S4S](https://www.murata.com/en-global/products/sensor/ultrasonic/overview/lineup) — 40 kHz piezoelectric ultrasonic transducer
8. [Sensirion SHT40](https://sensirion.com/products/catalog/SHT40) — Digital humidity and temperature sensor
9. [Matter Protocol (CSA)](https://csa-iot.org/all-solutions/matter/) — Connectivity Standards Alliance smart home interoperability standard
10. [LightGBM](https://lightgbm.readthedocs.io/) — Gradient boosting framework for on-device inference
