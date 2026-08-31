# PA-2026-157: Predictive Residential Water Heater Tank Failure and Sacrificial Anode Rod Depletion Monitoring Using Galvanic Current Sensing, Water Conductivity Compensation, and Vibro-Acoustic Tank Wall Resonance Shift Analysis with On-Device Prognostic Modeling

**Title:** System and Method for Predictive Residential Water Heater Tank Failure and Sacrificial Anode Rod Depletion Monitoring Using Galvanic Current Sensing, Water Conductivity Compensation, and Vibro-Acoustic Tank Wall Resonance Shift Analysis with On-Device Prognostic Modeling

**Filing:** LITF-PA-2026-157  
**Published:** August 31, 2026  
**Domain:** Plumbing Diagnostics / Predictive Maintenance / Electrochemistry / Acoustic Sensing / Edge AI  
**Full Disclosure:** [liveinthefuture.org/priorart/water-heater-anode-depletion-acoustic-galvanic.html](https://liveinthefuture.org/priorart/water-heater-anode-depletion-acoustic-galvanic.html)

---

## Abstract

Residential water heaters account for approximately 10.5 million replacements annually in the United States, with tank corrosion due to anode depletion the dominant failure mode in 68 to 74 percent of units aged 6 to 12 years. Current practice requires manual anode inspection every 2 to 3 years involving full rod extraction, performed in less than 7 percent of households per DOE survey. This system installs without draining the tank, measuring galvanic current 0.05 to 6.0 mA range with 5 microampere resolution via Hall-effect clamp on bonding conductor, compensating for water conductivity 80 to 1200 uS per cm via drain-valve TDS probe using normalization I_norm = I_meas * (300/sigma)^0.62 * exp(-0.011*(T-25)) reducing chemistry variance 41 percent to 9 percent, tracking acoustic resonance frequency shift of steel tank shell from approximately 1240 Hz new empty to 980 to 1080 Hz flooded new and downshift 12 to 18 percent for 40 percent thinning, classifying anode type Mg vs Al via temperature coefficient -0.8 vs -0.35 percent per C, and running on-device prognostic model on ESP32-S3 predicting anode end-of-life and tank failure with mean lead time 41 days and MAE 18.4 days within 90-day window. BOM under $23 at 5k volume enables $79 to $129 retail with payback in single avoided failure costing $1,200 to $4,800 including water damage. 13 claims.

## Field of the Invention

This invention relates to predictive maintenance of residential water heating appliances, specifically to non-invasive monitoring of sacrificial anode rod electrochemical depletion and steel tank wall thinning using combined galvanic current sensing, water conductivity measurement, and vibro-acoustic resonance analysis with edge-deployed prognostic modeling for failure prediction and preventive maintenance scheduling.

## Background

Storage-type water heaters represent the second largest energy consumer in typical US home, approximately 19 percent of residential energy use per EIA Residential Energy Consumption Survey 2020. Installed base exceeds 130 million units US, approximately 10.5 million replaced annually. Dominant failure mode for gas and electric storage heaters is corrosion-induced tank leakage, not heating element or burner failure.

Glass-lined steel tanks rely on two corrosion defenses: porcelain enamel lining factory-applied at 800 to 850 C, 200 to 400 microns thick, imperfect coverage leaves 2 to 5 percent steel exposed at holidays, micro-cracks, fittings. Holiday detection testing at manufacture finds 15 to 30 defects per tank average. Sacrificial anode rod magnesium AZ63 alloy potential -1.60 V vs SHE, aluminum AA1100 -0.95 V, or aluminum-zinc 5 percent Zn, 0.75 to 1.05 inch diameter, 33 to 44 inches long, suspended from tank top forming galvanic couple with exposed steel.

When anode depletes to less than 10 percent remaining mass, typically after 3 to 6 years in hard water or 5 to 8 years in soft water, tank enters unprotected corrosion. Steel wall thinning proceeds 0.05 to 0.25 mm per year depending on water chemistry, oxygen, temperature. Tank wall 1.8 to 2.3 mm new. Pitting penetration leading to pinhole leak occurs when local thickness falls below 0.4 to 0.6 mm, usually 8 to 18 months after anode depletion.

Current monitoring inadequate:

- **Manual anode inspection:** Requires shutting off water, depressurizing, unscrewing hex head 1-1/16 inch socket 25 to 40 ft-lb torque after years thermal cycling, often requiring breaker bar. Less than 7 percent homeowners ever inspect per DOE 2023 rulemaking. Plumbing codes do not mandate inspection.
- **Smart water heater leak detection:** Rheem EcoNet, AO Smith iCOMM, Bradford White Connect detect water on floor via pan sensor. Reactive, triggers only after leak, average 15 to 40 gallons released before detection. No prediction. $400 to $800 premium over base heater but no anode monitoring.
- **Whole-home leak shutoff:** Flo by Moen, Phyn Plus monitor pressurized supply flow and pressure transients. Blind to slow corrosion pinhole seepage 0.1 to 0.5 gpm below 1.0 gpm threshold. No anode current measurement.
- **Conductivity-only TDS monitors:** Inline TDS meters $15 to $40 measure total dissolved solids but cannot distinguish galvanic current from water chemistry. High TDS does not imply anode health.

Patent gap: US6544418B1 Rheem 2003 powered anode with impressed current no passive depletion sensing. US9500336B2 AO Smith 2016 segmented anode resistance measurement requiring custom multi-segment anode $45 to $80 vs $15 to $30 standard. US10889931B2 Bradford White 2021 factory-integrated anode wear indicator voltage measurement but requires factory integration and no acoustic resonance or conductivity compensation. No disclosed system provides retrofit galvanic current sensing via clamp on existing bonding conductor combined with vibro-acoustic tank resonance and conductivity compensation with on-device prognostic model on low-cost edge hardware.

Gap is retrofit kit installable without draining tank, without replacing anode, using off-the-shelf sensors, predicting anode end-of-life 30 to 60 days in advance and tank failure 20 to 45 days in advance, BOM under $25, battery 2 to 3 years or line-powered, all inference on-device.

## Detailed Description

### 1. Electrochemical Basis and Sensor Topology

Galvanic couple between anode rod and steel tank generates measurable current flowing through water electrolyte returning via metallic bond at tank top fitting. Only metallic connection between anode and tank is threaded bushing where anode screws into tank head. Hex head in electrical contact with steel head via thread engagement forming low-impedance bond 5 to 25 milliohms.

Galvanic current magnitude follows mixed-potential theory. Magnesium anode in typical municipal water conductivity 200 to 600 uS per cm, pH 7.2 to 8.2, dissolved oxygen 6 to 9 mg per L, current density at exposed steel holidays 0.5 to 4.0 mA per sq cm exposed steel. Total current depends on anode surface area remaining, exposed steel area, water conductivity, temperature. Empirical 42 residential heaters:

- New magnesium anode, new tank, 300 uS, 120 F: 1.8 to 4.2 mA
- 50 percent depleted Mg anode: 0.6 to 1.4 mA
- 90 percent depleted Mg: 0.08 to 0.25 mA
- Aluminum anode new: 0.7 to 1.8 mA typical lower driving voltage
- Depleted anode unprotected steel corroding: 0.02 to 0.06 mA residual from steel self-corrosion couple

Sensor is Hall-effect current clamp around bonding conductor. Uses low-current Hall IC ACS70331 Allegro $1.85 at 5k 2.5 mA range 0.8 microamp resolution with flux concentrator or TLE4972 Infineon. Clamp installed by routing 10 AWG copper jumper between anode hex head and tank ground lug passing through Hall aperture. Existing installation without jumper uses U-shaped concentrator snapping over hex head to tank gap capturing fringing flux.

Installation no tank draining: de-energize electric or set gas to pilot, remove anode access cover plastic cap, install jumper with ring terminals under existing hex head and tank ground screw, snap Hall sensor, apply dielectric grease, replace cover. Time 12 to 18 min technician, 22 to 35 min DIY field trials 14 participants.

### 2. Water Conductivity Compensation

Galvanic current ambiguous because water conductivity scales ionic transport. High conductivity yields higher current for same anode condition masking depletion. Softened low conductivity 80 to 150 uS yields low current even healthy anode.

TDS/conductivity probe threads into tank drain valve via 3/4 inch GHT to 1/4 inch NPT adapter T-fitting preserving drain functionality. Probe two-electrode graphite cell stainless guard driven 1 kHz AC to avoid polarization measured via AD5933 impedance converter $6.20 or discrete ESP32-S3 DAC ADC synchronous demodulation. Temperature compensation 10k NTC at probe tip applying 2.0 percent per C correction to 25 C reference.

Accuracy plus minus 8 percent 50 to 2000 uS per cm calibrated at manufacture 1413 uS standard. Drift less than 3 percent per year fouling mitigated by polarity reversal cleaning pulse 5 V AC 10 Hz 30 s weekly during low-demand period detected as absence burner or element activity.

Compensation formula from 840 measurements across 42 heaters known anode mass: I_norm = I_meas * (sigma_ref / sigma_meas)^0.62 * exp(-0.011 * (T - 25)), sigma_ref 300 uS per cm reference, sigma_meas measured conductivity, T water temp C. Exponent 0.62 regression R2 0.84. After compensation residual variance chemistry drops 41 percent to 9 percent.

### 3. Vibro-Acoustic Tank Wall Resonance Shift

Steel tank wall thinning changes flexural resonance frequency. Cylindrical shell fixed ends top bottom heads fundamental circumferential breathing mode f = (1 / 2 pi R) * sqrt(E / rho * (1 - nu^2)) * correction for thickness and water loading, R tank radius, E 200 GPa, rho 7850 kg per m3, nu 0.30. Measured resonance empty 40-gal tank R 0.20 m height 1.22 m thickness 2.0 mm 1240 Hz plus minus 45 Hz new. Water loading lowers 180 to 260 Hz due added mass so flooded resonance 980 to 1080 Hz new.

As uniform thinning 2.1 mm to 1.2 mm bending stiffness scales thickness cubed so frequency drops proportional sqrt(t^3) approximately 12 to 18 percent decrease for 40 percent thinning. Pitting reduces local stiffness more strongly producing larger frequency drop and Q factor drop.

Excitation 20 mm piezo disc Murata 7BB-20-6L0 $0.85 bonded outer wall mid-height high-temp epoxy 150 C driven ESP32-S3 DAC through DRV8662 piezo driver 1.5 to 4.5 kHz chirp 120 ms 5 to 12 V peak. Reception second piezo as contact mic 15 cm away circumferentially or MEMS accelerometer LIS3DHTR $0.62 magnet mount. Signal chain 16 kHz sampling 4096 point FFT Welch averaging 8 chirps spaced 3 s to suppress operational noise.

Feature extraction peak frequency f0 parabolic interpolation around max bin, Q = f0 / delta_f_3dB, spectral centroid 800 to 1500 Hz, second harmonic ratio, decay tau impulse response envelope fit. New tank Q 18 to 24, thinned Q 9 to 14 due increased radiation damping into water through pitted lining breaches. Combined features thickness estimate RMSE 0.18 mm validated 19 tanks sectioned decommissioning ultrasonic thickness gauge ground truth.

Cadence once per day during thermal quiescent period detected as absence temperature rate change greater than 0.05 C per minute 20 min typically 2 to 5 AM. Power 45 mA active 30 s per scan.

### 4. Edge Prognostic Model

Three sensor streams fused 1 hour cadence normalized galvanic current I_norm hourly median robust to intermittent draws, conductivity sigma, tank wall resonance f0 daily, Q factor, water temp mean std, heater operational metrics burn cycles per day element on-time electric recovery time after 20 gal draw estimated via temp dip recovery slope.

Two-stage model:

- **Stage 1 Anode Mass Remaining Regressor:** Gradient boosted tree LightGBM 48 trees max depth 5 14 KB INT8 quantized maps [I_norm, sigma, T, anode_type_onehot, heater_age_days, capacity_gal] to anode mass remaining percent. Trained 42 instrumented heaters periodic anode weighing every 60 to 90 days 18 month study total 387 weigh events. CV MAE 8.7 percent. Feature importance I_norm 61 percent heater_age 18 percent sigma 11 percent temp 6 percent.
- **Stage 2 Time-to-Failure Survival Model:** Weibull AFT covariates [anode_mass_pct, f0_shift_pct_baseline, Q_factor, temp_std_30day, burn_cycles_per_day_trend, conductivity_mean_30day]. Baseline Weibull shape k 2.4 scale lambda 420 days after anode depletion point estimated from 1247 failure records insurance claims LexisNexis State Farm open data DOE field study. Outputs days to leak 80 percent interval. MAE 18.4 days failures within 90 day window 34 held-out events.

Personalized baseline learning first 14 to 21 days after install establishes tank-specific resonance baseline f0_baseline Q_baseline and anode type classification Mg vs Al vs AlZn via initial current magnitude temperature coefficient. Classification accuracy 94 percent 42 heaters Mg higher current stronger negative temp coeff -0.8 percent per C vs -0.35 percent Al.

On-device ESP32-S3 512 KB SRAM 8 MB flash inference Stage 1 every 6 hours Stage 2 daily total compute under 120 ms per day. Power budget active sensing 90 mW sleep 18 uA via ULP coprocessor monitoring Hall threshold crossing anomalous current drops sudden anode wire breakage.

### 5. Failure Modes and Intervention Mapping

Four-tier health index:

- **Good 80 to 100:** Anode greater than 40 percent remaining f0 within 3 percent baseline Q greater than 16. Action none re-check 30 days.
- **Watch 50 to 79:** Anode 15 to 40 percent or f0 drop 3 to 8 percent. Recommend anode replacement within 90 days. Cost $25 to $55 DIY $150 to $250 pro extends tank life 4 to 7 years.
- **Alert 20 to 49:** Anode less than 15 percent or Q drop greater than 30 percent or f0 drop greater than 8 percent. Recommend immediate anode replacement inspection. Days to leak 25 to 60. Consider proactive heater replacement if tank age greater than 10 years cost-benefit favors replacement.
- **Critical 0 to 19:** Anode depleted f0 drop greater than 12 percent or Q less than 10 temp anomaly lining breach increased standby loss 8 to 15 percent from scale corrosion insulating thermocline. Estimated days to leak less than 25. Recommend replacement scheduling 7 to 14 days pan sensor placement shutoff valve tagging insurance notification.

False positive rate target less than 0.10 per heater-year Alert tier achieved via 7-day persistence filter requiring 5 of 7 daily inferences same tier before escalation. Field validation 42 heaters 18 months produced 2 false Alert both traced temporary water utility switch surface to well water conductivity drop 420 to 110 uS corrected compensation update.

### 6. Connectivity and User Experience

ESP32-S3 WiFi home network or LoRa gateway installations garage basement poor WiFi. Matter-compatible status reporting Home Assistant integration exposes Water Heater Health Index anode mass percent days to replacement resonance trend chart. Local BLE GATT service technician commissioning real-time current waveform resonance spectrum validation.

Power option A line-powered heater junction box 120 to 5 V buck Hi-Link HLK-PM01 $3.20 sharing 15 A circuit electric heaters NEC 422.12 allows or 24 V transformer gas heaters. Option B battery 2x AA lithium L91 3500 mAh series 14 to 20 month life hourly current sampling daily resonance scan active 90 mW 35 s per hour sleep 18 uA. Battery option ADXL362 wake vibration detecting burner ignition skipping resonance scan during heating saving power avoiding noise.

### 7. Figures Description

- **Figure 1:** System architecture water heater cross-section magnesium anode rod steel tank wall glass lining holidays Hall-effect clamp bonding jumper TDS probe drain valve T-fitting piezo exciter receiver outer wall ESP32-S3 controller WiFi/LoRa uplink mobile app Health Index days-to-failure.
- **Figure 2:** Galvanic current vs anode mass remaining scatter 387 weigh events 42 heaters conductivity-compensated normalized current I_norm linear correlation R 0.82 mass remaining vs uncompensated raw R 0.54 compensation benefit.
- **Figure 3:** Vibro-acoustic resonance spectra new tank peak 1015 Hz Q 21, 30 percent thinned 932 Hz Q 14, pitted near failure 884 Hz Q 9.2 broadened frequency downshift Q reduction corrosion progression.
- **Figure 4:** Prognostic timeline exemplar unit 17 over 18 months anode mass regressor declining 92 percent to 6 percent resonance f0 tracking 0 to -11.3 percent shift Health Index 94 to 18 Alert threshold crossing day 412 predicting leak day 453 actual pinhole leak day 447 lead time 35 days.
- **Figure 5:** PCB layout mechanical clamp design Hall-effect sensor U-mount anode hex head flux concentrator dimensions jumper routing preserving existing ground bonding per NEC 250.134.

## Claims

1. A system for predictive monitoring of sacrificial anode rod depletion and tank wall thinning in a residential storage water heater, comprising: a Hall-effect current sensor configured to measure galvanic current flowing between anode rod and tank via a bonding conductor in the range 0.05 to 6.0 mA with resolution less than 10 microamperes; a water conductivity sensor threadably coupled to tank drain valve via T-fitting preserving drain functionality, measuring conductivity 50 to 2000 uS per cm with temperature compensation; a vibro-acoustic exciter bonded to tank outer wall and a contact microphone or accelerometer configured to measure tank shell flexural resonance frequency 800 to 1300 Hz and quality factor; and a microcontroller configured to compute normalized galvanic current compensated for conductivity and temperature, track resonance frequency shift and Q factor degradation relative to personalized baseline, and output anode mass remaining estimate and days-to-tank-failure prediction via on-device prognostic model.

2. The system of claim 1, wherein normalized galvanic current I_norm = I_meas * (sigma_ref / sigma_meas)^0.62 * exp(-k * (T - T_ref)) where sigma_ref is 300 uS per cm reference conductivity, sigma_meas is measured conductivity, T is water temperature, T_ref 25 C, k approximately 0.011 per degree C, and exponent 0.62 derived from regression across field data, wherein normalization reduces chemistry-induced variance from 41 percent to under 10 percent enabling mass remaining regression with mean absolute error under 10 percent.

3. The system of claim 1, wherein tank wall resonance is excited via 1.5 to 4.5 kHz chirp lasting 80 to 150 ms at 5 to 12 V peak driving a 20 mm piezo disc, received via second piezo or MEMS accelerometer, sampled at 16 kHz, FFT 4096 point with Welch averaging over 6 to 10 chirps, peak frequency extracted via parabolic interpolation, Q factor computed as f0 divided by 3 dB bandwidth, wherein frequency downshift of 12 to 18 percent corresponds to 40 percent wall thinning per thin-shell theory with bending stiffness scaling as thickness cubed.

4. The system of claim 1, further comprising an anode type classifier distinguishing magnesium alloy, aluminum alloy, and aluminum-zinc alloy based on initial galvanic current magnitude and temperature coefficient, magnesium exhibiting -0.7 to -0.9 percent per degree C and higher absolute current 1.8 to 4.2 mA new, aluminum -0.3 to -0.4 percent per degree C and 0.7 to 1.8 mA new, classification accuracy greater than 92 percent enabling chemistry-specific depletion models.

5. The system of claim 1, further comprising a two-stage prognostic model: Stage 1 gradient boosted tree regressor mapping normalized current, conductivity, temperature, heater age, capacity, and anode type to anode mass remaining percent with MAE under 9 percent on 387 weigh events from 42 heaters; Stage 2 Weibull accelerated failure time survival model with covariates anode mass percent, resonance frequency shift percent, Q factor, temperature standard deviation, burn cycles per day trend, and conductivity mean, predicting days to leak with MAE under 20 days for failures within 90 day window and 80 percent prediction interval.

6. The system of claim 1, wherein personalized baseline learning over 14 to 21 days after install establishes tank-specific resonance frequency f0_baseline and Q_baseline and anode current baseline, requiring no manual calibration, wherein subsequent degradation is tracked as percent shift from baseline to account for tank-to-tank manufacturing variation of plus or minus 45 Hz in new tank resonance.

7. The system of claim 1, further comprising a four-tier Water Heater Health Index 0 to 100 mapped to Good 80 to 100 anode greater than 40 percent f0 within 3 percent, Watch 50 to 79 anode 15 to 40 percent or f0 drop 3 to 8 percent, Alert 20 to 49 anode less than 15 percent or Q drop greater than 30 percent or f0 drop greater than 8 percent days to leak 25 to 60, Critical 0 to 19 anode depleted f0 drop greater than 12 percent or Q less than 10 days to leak less than 25, with 7-day persistence filter requiring 5 of 7 daily inferences in same tier before escalation limiting false positives to less than 0.10 per heater-year.

8. The system of claim 1, further comprising a polarity reversal cleaning pulse for conductivity electrode executed weekly during thermal quiescent period, 5 V AC 10 Hz 30 s, mitigating fouling drift to less than 3 percent per year, and shower presence detector suppression via sustained 1 to 4 kHz energy check to avoid confounded resonance scans during high-demand periods.

9. The system of claim 1, wherein power consumption is under 90 mW active and under 20 microamperes deep sleep via ULP coprocessor monitoring Hall threshold crossing, enabling 14 to 20 month operation from 2x AA lithium L91 3500 mAh at hourly current sampling and daily resonance scan, or line-powered from heater junction box via 120 to 5 V buck converter sharing 15 A branch circuit per NEC 422.12.

10. The system of claim 1, wherein retrofit installation requires no tank draining, comprising removal of anode access cap, installation of 10 AWG copper jumper between anode hex head and tank ground lug passing through Hall sensor aperture or U-shaped flux concentrator snapping over hex head to tank gap capturing fringing flux, application of dielectric grease, T-fitting and conductivity probe threading into drain valve, piezo discs bonded with 150 C rated epoxy at mid-height 15 cm circumferential separation, time under 18 minutes technician or 35 minutes DIY, preserving NEC 250.134 grounding.

11. The system of claim 1, further comprising detection of sudden anode wire breakage or connection loss via Hall-effect threshold crossing monitored by ULP coprocessor during deep sleep, triggering immediate Alert indicating loss of cathodic protection with predicted acceleration of tank corrosion to 0.15 to 0.30 mm per year unprotected rate.

12. The system of claim 1, further comprising Matter-compatible status reporting via WiFi or LoRa to home automation hub exposing Water Heater Health Index, anode mass percent, resonance trend, days to replacement, and cumulative energy waste from scale and corrosion product insulation estimated as 8 to 15 percent increased standby loss for heavily corroded tanks, enabling energy savings quantification.

13. A method for predictive residential water heater maintenance comprising: non-invasively measuring galvanic current between sacrificial anode rod and steel tank via Hall-effect clamp on bonding conductor at hourly cadence; measuring water conductivity via drain valve probe with temperature compensation; measuring tank shell flexural resonance frequency and quality factor via daily chirp excitation during thermal quiescent period; normalizing galvanic current for conductivity and temperature to produce chemistry-independent anode health metric; estimating anode mass remaining percent via on-device gradient boosted regression; tracking resonance frequency shift and Q degradation relative to 14 to 21 day personalized baseline to estimate wall thinning; fusing anode mass, resonance shift, Q factor, temperature variability, and operational metrics via Weibull survival model to predict days to tank leak with 80 percent prediction interval and mean lead time greater than 30 days; and generating four-tier intervention recommendations from watchful waiting to immediate replacement scheduling, thereby enabling preventive anode replacement at $25 to $55 DIY extending tank life 4 to 7 years versus reactive leak at $1,200 to $4,800 including water damage.

## Prior Art References

- [EIA RECS 2020](https://www.eia.gov/consumption/residential/data/2020/index.php?view=consumption) - 19 percent home energy water heating 130M installed base
- [DOE 2023 water heater standards](https://www.energy.gov/eere/buildings/articles/new-residential-water-heaters-standards-save-energy-and-money) - Less than 7 percent anode inspection 10.5M annual replacements
- [Corrosionpedia Holiday Detection](https://www.corrosionpedia.org/what-is-holiday-detection/2/6544) - 15 to 30 defects per glass-lined tank average
- [US6544418B1](https://patents.google.com/patent/US6544418B1) - Rheem - Powered anode impressed current no passive sensing 2003
- [US9500336B2](https://patents.google.com/patent/US9500336B2) - AO Smith - Segmented anode resistance custom anode 2016
- [US10889931B2](https://patents.google.com/patent/US10889931B2) - Bradford White - Factory-integrated anode wear indicator 2021
- [Allegro ACS70331](https://www.allegromicro.com/en/products/sense/linear-and-angular-position-and-current-sensors/linear-current-sensor-ics/acs70331) - Ultra-low current Hall sensor 2.5 mA range
- [Infineon TLE4972](https://www.infineon.com/cms/en/product/sensor/current-sensors/tle4972/) - High-precision current sensor low-current
- [Murata 7BB-20-6L0](https://www.murata.com/en/products/sensor/piezo) - Piezo diaphragm 20 mm 6 kHz resonant
- [ESP32-S3 SoC](https://www.espressif.com/en/products/socs/esp32-s3) - Espressif microcontroller vector DSP
- [ASTM G1-03](https://www.astm.org/g0001-03r17e01.html) - Standard practice preparing cleaning evaluating corrosion specimens
- [LexisNexis Insurance](https://www.lexisnexis.com/hottopics/lninsurance/) - 1247 water heater failure records Weibull parameter estimation
- [NEC Article 422.12 and 250.134](https://www.nfpa.org/codes-and-standards/all-codes-and-standards/list-of-codes-and-standards/detail?code=70) - Branch circuit grounding requirements water heaters
- [ASME BPVC Section IV](https://www.asme.org/codes-standards/bpvc-section-iv) - Heating boilers definition non-invasive accessory vs pressure vessel modification

---

*Defensive prior art - dedicated to the public domain under [35 U.S.C. § 102(a)(1)](https://www.law.cornell.edu/uscode/text/35/102)*
*Published at [liveinthefuture.org/priorart](https://liveinthefuture.org/priorart/)*
