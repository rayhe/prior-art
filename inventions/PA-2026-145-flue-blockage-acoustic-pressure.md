# PA-2026-145: System and Method for Automated Detection of Residential Chimney and Furnace Flue Obstruction Using Acoustic Resonance Spectroscopy and Differential Pressure Sensing with Predictive Carbon Monoxide Risk Scoring and Smart HVAC Integration

**Filing:** LITF-PA-2026-145  
**Domain:** Home Safety / Acoustic Sensing / HVAC  
**Published:** August 20, 2026  
**Type:** Defensive Prior Art Disclosure  

---

## Abstract

Disclosed is a system and method for continuous automated detection of obstructions in residential chimney flues, furnace exhaust vents, and water heater draft hoods using a combination of acoustic resonance spectroscopy and differential pressure sensing. The system comprises a low-cost sensor module installed at the base of the flue or vent connector, containing a MEMS microphone, a small loudspeaker, and a MEMS differential pressure sensor. The loudspeaker periodically emits frequency-swept chirp signals (50-4,000 Hz over 200 ms) into the flue, which acts as an acoustic waveguide. The microphone captures the reflected signal, and an on-device microcontroller computes the impulse response via matched-filter cross-correlation, extracting the flue's resonant mode frequencies, Q-factors, and reflection coefficients. A partial or complete obstruction shifts the resonant frequencies, increases reflection amplitude, and reduces Q-factors in patterns characteristic of the obstruction type: gradual creosote accumulation produces progressive narrowing of higher-order modes, animal nests create abrupt impedance discontinuities with strong mid-frequency reflections, collapsed liner sections produce broadband attenuation with characteristic double-reflection signatures, and ice caps generate temperature-dependent resonance shifts correlated with outdoor temperature. Simultaneously, the differential pressure sensor measures the static draft pressure during combustion appliance operation, comparing the observed draft (-5 to -25 Pa for normal operation) against a model that accounts for outdoor temperature, wind speed, and flue geometry via the stack-effect equation. An on-device random forest classifier fuses acoustic and pressure features to classify obstruction type, estimate severity (percentage cross-sectional area reduction), localize the obstruction height within the flue, and compute a predictive carbon monoxide risk score. When the CO risk score exceeds a configurable threshold, the system sends a command via Matter/Thread protocol to the smart thermostat or HVAC controller to inhibit combustion appliance ignition, preventing CO accumulation in the living space. The system operates on a CR123A lithium battery for 3+ years at one measurement cycle per hour.

## Technical Field

This invention relates to residential safety systems, specifically to automated monitoring of chimney and furnace exhaust flue integrity using acoustic and pressure-based sensing for obstruction detection and carbon monoxide prevention.

## Background

Carbon monoxide poisoning from blocked or malfunctioning residential flues kills approximately 430 Americans annually and sends another 50,000 to emergency departments (CDC). The Consumer Product Safety Commission estimates that 170 deaths per year result specifically from non-automotive consumer products, with furnaces, water heaters, and fireplaces among the leading sources. Blocked flues cause backdrafting, where combustion exhaust including CO is drawn into the living space rather than venting outdoors.

Common flue obstructions include:

- **Creosote accumulation:** The Chimney Safety Institute of America (CSIA) reports that creosote buildup is the leading cause of chimney fires, with approximately 25,000 chimney fires per year in the US. Stage 3 glazed creosote can reduce flue cross-section by 30-60% before visual inspection would detect the hazard.
- **Animal nests:** Birds (particularly chimney swifts, starlings, and sparrows), squirrels, and raccoons frequently nest in unused flues during spring and summer. NFPA 211 recommends annual inspection partly to detect animal intrusion, but most homeowners skip inspections for years.
- **Structural failure:** Clay tile flue liners crack from thermal shock and spall over decades. Mortar deterioration allows flue tiles to shift, partially or fully blocking the flue. Metal liners corrode, particularly in high-efficiency condensing appliances where acidic condensate attacks stainless steel.
- **Ice and snow caps:** In cold climates, moisture in exhaust gas can freeze at the flue termination, progressively restricting the opening. High-efficiency furnace PVC vent terminations are especially vulnerable, with documented cases of complete blockage within 48 hours during sustained sub-zero conditions.

Current detection methods are episodic and manual:

- **Visual inspection:** NFPA 211 recommends annual Level 1 chimney inspection. The CSIA certifies approximately 2,000 chimney sweeps nationwide. Average inspection cost: $125-300. Compliance rate among homeowners: estimated at fewer than 30%.
- **CO detectors:** Required by code in most jurisdictions under NFPA 720. CO detectors are last-resort alarms that trigger at 70 ppm sustained exposure (UL 2034 standard). They detect CO after it has already accumulated in the living space, providing no predictive or preventive capability. False negative rates increase significantly with detector age, as electrochemical sensor sensitivity degrades.
- **Draft gauges:** HVAC technicians use manometers to measure flue draft during service calls, typically once per year during heating season startup. This provides a single-point measurement that misses intermittent blockages and progressive degradation between service visits.

The art contains relevant but non-overlapping work in acoustic pipe inspection for industrial settings. Muggleton et al. (Journal of Sound and Vibration, 2006) demonstrated acoustic pulse reflectometry for detecting blockages in buried water mains, using frequencies below 500 Hz to characterize pipe wall condition. Sharp et al. (Journal of the Acoustical Society of America, 2015) developed acoustic resonance methods for assessing musical instrument bores. US6386037B1 (Siemens) describes acoustic monitoring of industrial exhaust stacks using permanently installed sensor arrays, but targets large-diameter (>1 m) industrial stacks, not residential flues (typically 15-30 cm diameter), and requires wired power and connectivity.

The gap in the art is a low-cost, battery-operated, consumer-grade system that: (a) continuously monitors residential flue condition using both acoustic and pressure modalities, (b) classifies obstruction type and estimates severity using on-device inference, (c) localizes the obstruction height within the flue, (d) predicts carbon monoxide risk before dangerous concentrations develop, and (e) integrates with smart home systems to preventively inhibit combustion appliance operation.

## Detailed Description

### 1. Sensor Module Hardware

The sensor module is designed for installation at the base of a chimney flue, furnace vent connector, or water heater draft hood. It comprises: a MEMS differential pressure sensor (e.g., Sensirion SDP810-500Pa, range +/-500 Pa, resolution 0.1 Pa, unit cost ~$8) with one port connected to the flue interior and the other to ambient room air via a short tube; a MEMS microphone (e.g., Knowles SPH0645LM4H, -26 dBFS sensitivity, 65 dB SNR, unit cost ~$1.50) mounted flush with the inner surface of the sensor module housing, oriented into the flue; a micro-loudspeaker (e.g., PUI Audio AS01808AO-3-R, 18 mm diameter, 8 ohm, unit cost ~$2) mounted adjacent to the microphone, also oriented into the flue; a microcontroller with DSP capability (e.g., ESP32-C6 with RISC-V core and 802.15.4 radio for Thread/Matter, unit cost ~$3.50); a CR123A lithium battery (3V, 1,500 mAh, operating temperature range -40C to 60C); and a temperature sensor (e.g., TMP117, +/-0.1C accuracy, unit cost ~$1.50) for compensating sound speed variation with flue temperature. Total bill-of-materials: approximately $22-28. The housing is stainless steel with a high-temperature silicone gasket, rated to 260C continuous exposure for installation in proximity to flue gas.

### 2. Acoustic Resonance Spectroscopy

A residential chimney flue behaves as an acoustic waveguide, approximately modeled as a cylindrical tube open at the top (termination cap or open sky) and partially closed at the bottom (damper or combustion chamber). The fundamental resonant frequency for this configuration follows:

f_n = (2n - 1) * c / (4L), for n = 1, 2, 3, ...

where c is the speed of sound (approximately 343 m/s at 20C, increasing to ~390 m/s at 200C flue gas temperature), and L is the effective acoustic length of the flue. For a typical 8-meter residential chimney, the fundamental resonant frequency is approximately 10.7 Hz (below the measurement range), with the third and fifth modes at 32 Hz and 53 Hz respectively. Higher-order modes at 100-2,000 Hz carry the diagnostic information, as they are more sensitive to cross-sectional area changes and localized obstructions.

The loudspeaker emits a logarithmic frequency-swept chirp from 50 Hz to 4,000 Hz over 200 ms, at approximately 70 dB SPL. The microphone records both the direct signal and all reflections for 500 ms total. On-device processing computes the impulse response of the flue via matched-filter cross-correlation between the emitted chirp and the recorded signal. From the impulse response, the system extracts: resonant mode frequencies (peaks in the frequency response magnitude), Q-factors for each mode (bandwidth at -3 dB), reflection coefficients at each impedance discontinuity (from time-domain reflection amplitudes), and the round-trip propagation delay to each reflector (localizing obstructions by height).

For a cylindrical waveguide of diameter d and length L, the cutoff frequency for the first non-planar mode is:

f_c = 1.841 * c / (pi * d)

For a standard 20 cm (8-inch) round flue, f_c approximately equals 1,005 Hz. Below this frequency, only planar modes propagate, simplifying the analysis. Above it, non-planar modes provide additional cross-sectional information useful for characterizing the spatial distribution of partial blockages.

### 3. Differential Pressure Draft Monitoring

The differential pressure sensor continuously measures the pressure difference between the flue interior and the occupied space. During combustion appliance operation, the hot exhaust gas column creates a natural draft described by the stack-effect equation:

dP = 0.0342 * h * (1/T_outdoor - 1/T_flue)

where dP is the draft pressure in Pa, h is the effective stack height in meters, and temperatures are in Kelvin. For an 8-meter flue with outdoor temperature of 0C (273 K) and flue gas temperature of 200C (473 K), the theoretical draft is approximately -12.6 Pa. Normal residential draft ranges from -5 Pa (marginal) to -25 Pa (strong). The sensor measures draft at 1 Hz during combustion appliance operation and at 0.1 Hz during idle periods.

An obstruction reducing the flue cross-sectional area by fraction beta increases the flow resistance approximately as (1 - beta)^(-2) for turbulent flow conditions (Reynolds number > 4,000, typical for operating flues). This produces a measurable reduction in draft pressure at the sensor location. The system maintains a running baseline of draft pressure versus outdoor temperature, wind speed (obtained from a local weather API via the Thread/Matter gateway), and appliance firing rate, detecting deviations that indicate developing obstructions.

### 4. Obstruction Classification

The on-device classifier is a random forest with 50 trees, maximum depth 12, trained on a feature vector comprising 18 acoustic features and 6 pressure features. Acoustic features include: frequencies of the first 6 resonant modes (6 features), Q-factors of the first 6 modes (6 features), total reflected energy normalized to emitted energy (1 feature), ratio of reflected energy above and below the cutoff frequency (1 feature), delay to the first strong reflection (1 feature), number of distinct reflection peaks above noise floor (1 feature), spectral centroid of the reflection pattern (1 feature), and spectral spread (1 feature). Pressure features include: mean draft pressure during last operating cycle (1 feature), draft pressure standard deviation (1 feature), draft pressure residual from stack-effect model (1 feature), draft startup transient time constant (1 feature), draft response to wind gusts (correlation coefficient with external wind speed, 1 feature), and draft pressure during idle (indicating reverse flow/backdraft tendency, 1 feature).

The classifier outputs five obstruction classes with associated confidence scores:

- **Clear (no obstruction):** Resonant modes match the baseline profile within +/-2%. Draft pressure follows the stack-effect model within +/-1.5 Pa.
- **Creosote/soot accumulation:** Progressive compression of higher-order modes toward the fundamental, because creosote deposits are distributed along the flue length and preferentially attenuate short-wavelength modes. Reflection pattern shows gradual impedance gradient rather than sharp discontinuity. Draft reduction is gradual and correlated with cumulative appliance runtime.
- **Animal nest/debris:** Abrupt impedance discontinuity producing a strong single reflection at a specific height. Modes above the obstruction height are strongly attenuated while modes below are preserved. Seasonal pattern (spring/summer onset). Draft reduction is sudden and may fluctuate as the animal modifies the nest.
- **Structural failure (liner collapse/mortar debris):** Broadband attenuation with characteristic double-reflection signature (debris pile at the base and structural discontinuity at the failure height). Multiple impedance discontinuities at irregular spacing. Draft pressure shows high variability as loose debris shifts.
- **Ice/snow cap:** Obstruction localized at the flue termination (maximum propagation delay). Reflection coefficient is strongly correlated with outdoor temperature below 0C. Draft pressure reduction appears only during freezing conditions and partially or fully resolves during thaws.

The classifier also estimates the percentage cross-sectional area reduction (0-100%) from the reflection coefficient magnitude and the draft pressure deficit, and localizes the obstruction height from the round-trip propagation delay (resolution +/-0.5 meters).

### 5. Carbon Monoxide Risk Scoring

The system computes a predictive CO risk score (0-100) by combining obstruction severity with operating conditions. The scoring model uses a logistic function:

Risk = 100 / (1 + exp(-(w1*beta + w2*dP_deficit + w3*T_diff + w4*V_wind + b)))

where beta is the estimated cross-sectional area reduction, dP_deficit is the observed draft deficit relative to the stack-effect model prediction, T_diff is the indoor-outdoor temperature difference (lower differences reduce natural draft, increasing backdraft risk), V_wind is the wind speed (high winds can cause pressure reversals in certain chimney geometries), and w1-w4 and b are parameters calibrated from the training dataset. The risk score exceeding 70 triggers an advisory notification; exceeding 85 triggers a preventive HVAC lockout command.

### 6. Smart HVAC Integration

The sensor module communicates via Thread (IEEE 802.15.4) to a Matter-compatible border router (e.g., a smart speaker or hub). When the CO risk score exceeds the lockout threshold (default: 85, user-configurable from 70-95), the system publishes a Matter command to the thermostat or HVAC controller cluster to inhibit combustion appliance operation. Specifically, it sets the HVAC system mode to "heat pump only" (for dual-fuel systems), "electric backup only," or "off" depending on the system configuration. The lockout persists until a subsequent acoustic measurement cycle confirms the obstruction has been cleared (cross-sectional area reduction below 15%) or until the homeowner manually overrides via the companion app with an explicit acknowledgment of the CO risk.

For homes without smart thermostats, the sensor module can directly interrupt the thermostat call-for-heat circuit via an optional relay module (normally closed, opens on lockout), installed inline with the thermostat wire at the furnace. This provides fail-safe operation without requiring a smart home ecosystem.

### 7. Power Management and Measurement Scheduling

The system operates on a single CR123A lithium battery with a target life of 3+ years. Power budget: the ESP32-C6 consumes approximately 5 uA in deep sleep; one acoustic measurement cycle (chirp emission, recording, DSP processing, classification) consumes approximately 15 mA for 2 seconds; one Thread transmission consumes approximately 10 mA for 50 ms; differential pressure sampling at 0.1 Hz during idle consumes approximately 3 uA average. At one acoustic measurement per hour and continuous low-rate pressure monitoring, the average current draw is approximately 12 uA, yielding a battery life of approximately 14,000 hours (4.7 years) from a 1,500 mAh cell.

The measurement frequency increases adaptively: when an obstruction trend is detected (any metric deviating more than 10% from baseline over a 72-hour window), the acoustic measurement rate increases to once per 15 minutes. During active combustion appliance operation (detected by the differential pressure exceeding -3 Pa), pressure is sampled at 1 Hz and acoustic measurements are triggered at cycle start and every 30 minutes during operation.

### 8. Calibration and Baseline Establishment

Upon installation, the system runs a calibration sequence of 10 acoustic measurements over 24 hours to establish the baseline resonant mode profile. The calibration captures the flue's specific geometry (length, diameter, liner material, bends, offsets, and damper position) as encoded in its acoustic signature. The system also records the baseline draft pressure versus outdoor temperature relationship over the first heating cycle. Subsequent measurements are compared against this per-installation baseline, making the system self-calibrating across the wide range of residential flue configurations (round clay tile, rectangular clay tile, oval stainless steel liner, double-wall B-vent, PVC/CPVC for high-efficiency condensing appliances). No manual configuration of flue dimensions is required.

## Claims

1. A system for detecting obstructions in residential chimney and furnace flues, comprising: a sensor module installed at or near the base of a flue, the module containing a loudspeaker, a microphone, a differential pressure sensor, and a microcontroller; wherein the loudspeaker periodically emits acoustic chirp signals into the flue, the microphone records the reflected acoustic response, and the microcontroller computes the flue's impulse response and extracts resonant mode parameters to detect changes indicative of obstruction.

2. The system of claim 1, wherein the resonant mode parameters extracted include mode frequencies, Q-factors, reflection coefficients, and round-trip propagation delays, and wherein changes in these parameters relative to a per-installation baseline are used to classify the obstruction type as one of: creosote accumulation, animal nest or debris, structural liner failure, or ice/snow cap formation.

3. The system of claim 1, wherein the differential pressure sensor simultaneously measures the flue draft pressure, and an on-device model compares the observed draft against a stack-effect prediction incorporating outdoor temperature, wind speed, and flue geometry to compute a draft pressure deficit indicative of obstruction severity.

4. The system of claim 1, wherein an on-device classifier fuses acoustic features and differential pressure features to estimate the percentage cross-sectional area reduction of the flue and localize the obstruction height within the flue from the round-trip propagation delay of reflected acoustic energy.

5. A method for preventing carbon monoxide poisoning from blocked residential flues, comprising: periodically measuring acoustic resonance and differential pressure in a flue; computing a predictive carbon monoxide risk score from obstruction severity, draft deficit, indoor-outdoor temperature difference, and wind speed; and when the risk score exceeds a threshold, transmitting a command to a smart thermostat or HVAC controller to inhibit combustion appliance ignition.

6. The method of claim 5, wherein the command is transmitted via a Thread/Matter protocol to set the HVAC system mode to a non-combustion heating mode or to open a relay interrupting the thermostat call-for-heat circuit.

7. The system of claim 1, wherein the acoustic measurement frequency increases adaptively from a baseline rate to a higher rate when any monitored parameter deviates from baseline by more than a configurable threshold, and further increases during active combustion appliance operation as detected by the differential pressure sensor.

8. The system of claim 1, wherein the system operates on a single lithium battery for at least three years by employing deep-sleep power management between measurement cycles, adaptive measurement scheduling, and low-duty-cycle wireless communication.

9. The system of claim 2, wherein ice/snow cap obstruction is distinguished from other obstruction types by correlation of the reflection coefficient at the flue termination with outdoor temperature below freezing, and by partial or complete resolution of the obstruction signature during thaw periods.

10. The system of claim 1, further comprising a self-calibration mode that establishes a per-installation acoustic baseline over the first 24 hours of operation by capturing the flue's specific geometry, liner material, bends, and damper position as encoded in the resonant mode profile, requiring no manual entry of flue dimensions.

11. The system of claim 1, wherein the obstruction classification includes a temporal trend component that distinguishes gradual accumulation processes (creosote, corrosion) from sudden events (animal intrusion, structural collapse) based on the rate of change of resonant mode parameters over a configurable observation window.

## Prior Art References

1. [CDC - Carbon Monoxide Poisoning](https://www.cdc.gov/co/default.htm) - 430 deaths, 50,000 ED visits annually in the US
2. [CPSC - Carbon Monoxide Information Center](https://www.cpsc.gov/Safety-Education/Safety-Education-Centers/Carbon-Monoxide-Information-Center) - 170 deaths/year from non-automotive consumer products
3. [CSIA - Chimney Fires](https://www.csia.org/chimneyfires.html) - 25,000 chimney fires/year, creosote as leading cause
4. [NFPA 211](https://www.nfpa.org/codes-and-standards/nfpa-211) - Standard for Chimneys, Fireplaces, Vents, and Solid Fuel-Burning Appliances
5. [NFPA 720](https://www.nfpa.org/codes-and-standards/nfpa-720) - Standard for Carbon Monoxide Detection in Residential Units
6. Muggleton et al., Journal of Sound and Vibration (2006) - Acoustic pulse reflectometry for buried water main blockage detection
7. Sharp et al., Journal of the Acoustical Society of America (2015) - Acoustic resonance methods for bore assessment
8. [US6386037B1](https://patents.google.com/patent/US6386037B1) - Siemens - Acoustic monitoring of industrial exhaust stacks
9. [ACHR News - PVC Vent Freezing](https://www.achrnews.com/articles/136189) - Documented high-efficiency furnace vent icing cases
10. [Sensirion SDP810-500Pa](https://www.sensirion.com/products/catalog/SDP810-500Pa) - Differential pressure sensor datasheet
11. [ESP32-C6 SoC](https://www.espressif.com/en/products/socs/esp32-c6) - RISC-V microcontroller with 802.15.4 and Wi-Fi 6
12. [OpenThread](https://openthread.io/) - Open-source Thread networking stack
13. [Matter (CSA)](https://csa-iot.org/all-solutions/matter/) - Smart home interoperability standard

## Implementation Notes

A reference implementation targets the ESP32-C6 microcontroller running FreeRTOS with the following firmware architecture: the chirp signal is generated from a pre-computed lookup table stored in flash (4 KB at 16 kHz sample rate for 200 ms); the matched-filter cross-correlation is computed in the frequency domain using a 16,384-point FFT (8,192-point chirp zero-padded); peak detection on the impulse response uses a CFAR (constant false alarm rate) algorithm with a guard band of 20 samples and a reference window of 100 samples; the random forest classifier occupies approximately 35 KB of flash (50 trees, max depth 12, 24 features); and the Thread networking stack uses the OpenThread library. Total firmware size: approximately 400 KB. RAM usage: approximately 80 KB peak during FFT computation.

Training data for the classifier can be generated synthetically using transfer matrix method (TMM) simulation of acoustic waveguides with parameterized obstructions (location, extent, acoustic impedance), validated against physical measurements in a range of flue configurations. The Sensirion SDP810 differential pressure sensor provides temperature-compensated output requiring no additional calibration.
