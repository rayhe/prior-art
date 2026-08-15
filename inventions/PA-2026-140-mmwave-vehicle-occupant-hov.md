# PA-2026-140: System and Method for Passive Vehicle Occupant Count Estimation Using Roadside Millimeter-Wave Radar Micro-Doppler Analysis with Convolutional Neural Network Classification for Automated High-Occupancy Vehicle Lane Enforcement

**Filing:** LITF-PA-2026-140  
**Domain:** Transportation / Radar Sensing / Edge AI  
**Published:** August 15, 2026  
**Type:** Defensive Prior Art Disclosure  

---

## Abstract

Disclosed is a system and method for passively estimating the number of human occupants inside moving vehicles using roadside-mounted millimeter-wave (mmWave) radar operating in the 76-81 GHz automotive band. The system exploits the fact that each living human body inside a vehicle cabin produces a distinct micro-Doppler signature caused by involuntary physiological motion: respiration (0.1-0.5 Hz thoracic displacement), cardiac mechanical activity (1-2 Hz chest wall vibration), and postural micro-sway (0.05-0.3 Hz center-of-mass oscillation). These signatures propagate through standard automotive glass and lightweight vehicle body panels with measurable attenuation but sufficient signal-to-noise ratio for detection at the 76-81 GHz band. A roadside radar unit illuminates passing vehicles with a frequency-modulated continuous wave (FMCW) chirp sequence, extracts range-Doppler maps at the vehicle's range bin, and isolates the micro-Doppler components from the dominant vehicle bulk motion via clutter cancellation. A convolutional neural network (CNN) classifier trained on labeled micro-Doppler spectrograms estimates the occupant count (1, 2, 3, 4+) with target accuracy exceeding 92% for the binary HOV-eligible classification (2+ occupants vs. 1 occupant). The system operates without cameras, captures no images of vehicle interiors or license plates, and produces only an integer occupant count and confidence score per vehicle transit event. This architecture enables privacy-preserving automated HOV lane enforcement that functions through tinted windows, at night, and in adverse weather conditions where camera-based systems fail.

## Field of the Invention

This invention relates to intelligent transportation systems, specifically to non-invasive, privacy-preserving vehicle occupancy detection using millimeter-wave radar micro-Doppler analysis for automated enforcement of high-occupancy vehicle lane regulations.

## Background

High-occupancy vehicle (HOV) lanes serve approximately 136 freeway corridors across 30 US metropolitan areas (FHWA), carrying an estimated 3.2 million daily commuters. The economic value of HOV time savings is approximately $2.5-4.1 billion annually (GAO-16-781), but violation rates undermine the system. Caltrans estimates HOV violation rates of 15-25% during peak hours on Bay Area freeways, with some corridors exceeding 40%.

Current enforcement approaches are expensive, labor-intensive, and unreliable:

- **Manual enforcement:** CHP officers visually inspect vehicles from roadside pulloffs or motorcycle patrols. Cost: approximately $250,000/year per dedicated HOV enforcement officer. Coverage: typically 2-4 hours/day on a given corridor. Detection rate: estimated 3-5% of violations.
- **Camera-based automated systems:** Infrared and visible-light cameras photograph vehicle interiors to count occupants. Pilot deployments achieved 85-90% accuracy under ideal conditions but degrade sharply with aftermarket window tinting, nighttime operation, sun glare, and large vehicle categories. These systems inherently capture facial images, raising BIPA/CCPA/GDPR privacy concerns.
- **Toll transponder self-declaration:** Some express lanes allow drivers to self-declare occupancy via a transponder switch. Compliance is unverified, creating a moral hazard.

Millimeter-wave radar has been validated for through-wall human detection in search-and-rescue and security screening contexts. Li et al. (IEEE TMTT 2017) demonstrated detection of human respiration through 20 cm concrete walls using 77 GHz FMCW radar. Alizadeh et al. (IEEE Access 2019) achieved multi-person vital sign monitoring through drywall at 60 GHz. Ahmad et al. (IEEE Sensors Journal 2021) demonstrated in-vehicle occupant detection using 77 GHz radar mounted inside the vehicle cabin for airbag deployment optimization.

The gap in the art is a complete roadside-deployed system that: (a) counts vehicle occupants from outside the vehicle using radar, (b) works through tinted windows and in all lighting/weather conditions, (c) captures no images and produces no personally identifiable information, (d) operates at highway speeds (25-80 mph), and (e) achieves sufficient accuracy for automated enforcement citations or dynamic tolling adjustment.

## Detailed Description

### 1. Radar Hardware Configuration

The roadside radar unit comprises a 76-81 GHz FMCW radar transceiver (e.g., Texas Instruments AWR2944 or equivalent) with a 4-transmit, 4-receive MIMO antenna array, mounted on a roadside gantry or pole at a height of 3-5 meters above the road surface. The antenna array is oriented at a depression angle of 15-30 degrees toward the adjacent travel lane, with the azimuth boresight perpendicular to the direction of travel.

Key radar parameters:
- Chirp bandwidth: 4 GHz (77-81 GHz), yielding range resolution of 3.75 cm
- Chirp duration: 60 μs, with 256 chirps per frame
- Frame rate: 20 Hz (50 ms per frame)
- Maximum unambiguous velocity: ±12.4 m/s (relative to vehicle bulk motion)
- Transmit power: 12 dBm EIRP (within FCC Part 95 limits for 76-81 GHz)
- Antenna gain: 18 dBi per element, with digital beamforming across the MIMO virtual array providing 2-degree angular resolution in azimuth

A co-located inductive loop detector or LIDAR trigger sensor detects vehicle presence and provides a bulk velocity estimate for Doppler compensation. Total bill-of-materials cost per unit: $800-1,200 at production volume.

### 2. Signal Acquisition and Vehicle Isolation

When a vehicle enters the detection zone (approximately 8 meters of travel lane centered on the radar boresight), the radar acquires 2-4 seconds of FMCW data depending on vehicle speed. The raw IF signal is digitized at 10 MSPS and processed through a standard FMCW pipeline: range FFT, Doppler FFT, producing a range-Doppler map per frame.

Vehicle isolation proceeds as follows:
1. **Bulk motion compensation:** The dominant Doppler component (vehicle body reflections) is estimated via peak detection in the Doppler spectrum at the vehicle's range bin. This bulk velocity is subtracted from all Doppler bins, centering the micro-Doppler signatures around zero velocity.
2. **Range gating:** The vehicle's extent in range is estimated from the range profile. Only range bins within this extent are retained.
3. **Clutter cancellation:** A 3-pulse MTI filter with Chebyshev weighting suppresses static clutter and the vehicle body's rigid-body motion, passing only micro-Doppler components with velocities between 0.1 mm/s and 50 mm/s relative to the vehicle frame.
4. **MIMO beamforming:** The 4x4 virtual array is beamformed to isolate returns from the vehicle cabin volume using Capon (MVDR) beamforming.

### 3. Micro-Doppler Feature Extraction

After clutter cancellation and range gating, the residual signal contains micro-Doppler components from human physiological motion inside the vehicle cabin:

- **Respiratory motion:** Thoracic displacement of 4-12 mm at 0.15-0.4 Hz (9-24 breaths/minute for adults at rest). Produces micro-Doppler velocities of 0.4-3.0 mm/s. Each occupant's breathing frequency and phase are independent, creating separable spectral peaks.
- **Cardiac mechanical activity:** Chest wall displacement of 0.2-0.5 mm at 1.0-1.7 Hz (60-100 BPM). Weaker than respiration by 15-20 dB but detectable at 77 GHz because the displacement is a non-negligible fraction of the 3.9 mm wavelength. Cardiac harmonic structure provides a distinctive spectral fingerprint per occupant.
- **Postural micro-sway:** Involuntary center-of-mass oscillation at 0.05-0.3 Hz with amplitude of 2-8 mm. Strongest for passengers not actively controlling the vehicle.
- **Voluntary micro-motion:** Head turns, arm gestures, phone manipulation produce intermittent micro-Doppler bursts at 5-50 mm/s providing additional evidence of distinct occupant positions.

The system computes a short-time Fourier transform of the clutter-cancelled signal over the 2-4 second observation window, using 512-sample Hamming windows with 75% overlap. The resulting micro-Doppler spectrogram (approximately 80 time bins x 128 frequency bins) is the primary input to the classification model.

### 4. Occupant Count Classification

A CNN classifier processes the micro-Doppler spectrogram to estimate occupant count:

- **Input:** 80x128 micro-Doppler spectrogram (real-valued, log-scaled)
- **Feature extractor:** 4 convolutional blocks (32/64/128/256 filters, 3x3 kernels, batch normalization, ReLU, 2x2 max pooling). Total parameters: approximately 1.2M.
- **Attention module:** Channel attention (squeeze-and-excitation block) to weight frequency bands carrying discriminative micro-Doppler information.
- **Classifier head:** Global average pooling, 256-unit dense layer with dropout (0.3), softmax output over 4 classes: 1 occupant, 2 occupants, 3 occupants, 4+ occupants.
- **Inference time:** <15 ms per vehicle on Jetson Orin Nano (INT8 quantized).

For binary HOV enforcement (1 vs. 2+ occupants), the model operates at a configurable decision threshold calibrated to achieve a false positive rate below 0.5% at a target 92% true positive rate.

### 5. Multi-Occupant Separation via Range-Angle Binning

The MIMO array's 2-degree angular resolution provides approximately 20 cm lateral resolution at 6-meter slant range, sufficient to resolve driver-side vs. passenger-side vs. rear-seat occupants. The system constructs a range-angle-Doppler datacube and applies independent component analysis (ICA) or non-negative matrix factorization (NMF) to separate co-located micro-Doppler sources with distinct spatial origins. Each separated component is individually analyzed for physiological motion consistency (must contain respiratory and/or cardiac components to count as a living occupant).

### 6. Vehicle Type Normalization

A two-stage pipeline handles vehicle type variation:
1. **Vehicle classification:** A random forest classifier on 12 radar-derived features categorizes vehicles into 8 classes (compact sedan through commercial van). Accuracy: approximately 88%.
2. **Type-specific occupant model:** The CNN maintains separate batch normalization parameters and classifier heads per vehicle class, sharing the convolutional feature extractor.

### 7. Deployment Architecture and Data Flow

Each enforcement point comprises 1-2 radar units per lane, edge compute, vehicle trigger sensor, and backhaul. Per-vehicle output: `{timestamp, lane_id, speed_estimate, vehicle_class, occupant_count_estimate, confidence_score, spectrogram_hash}`. No images. No license plate data. No PII.

For enforcement integration, the output is correlated with a separate, existing tolling/LPR system. The radar system never captures license plate information, maintaining an architectural privacy firewall.

### 8. Calibration and Validation

Initial 30-90 day shadow mode with ground-truth (manual officers + temporary cameras). Ongoing validation via random audit sampling (0.1-1% of flagged violations) and weekly calibration runs with known-occupancy vehicles.

### 9. Applications Beyond HOV Enforcement

- Dynamic tolling occupancy verification
- Emergency vehicle occupancy tracking
- Parking garage ride-share zone management
- Aggregate traffic planning occupancy statistics
- Child/pet left-in-vehicle detection (stationary variant)

## Claims

1. A system for passive estimation of vehicle occupant count, comprising: a roadside-mounted millimeter-wave FMCW radar transceiver operating in the 76-81 GHz band with a MIMO antenna array; a signal processing module that extracts micro-Doppler spectrograms from radar returns after compensating for vehicle bulk motion and cancelling static clutter; and an on-device neural network classifier that estimates the number of living human occupants inside a moving vehicle based on physiological micro-Doppler signatures including respiratory thoracic displacement, cardiac chest wall vibration, and postural micro-sway.

2. The system of claim 1, wherein the micro-Doppler signatures are extracted by subtracting the dominant Doppler component corresponding to vehicle rigid-body motion and applying a moving target indicator filter to isolate velocity components between 0.1 mm/s and 50 mm/s relative to the vehicle reference frame.

3. The system of claim 1, wherein the MIMO antenna array provides angular resolution sufficient to localize individual occupants within the vehicle cabin, and the system applies independent component analysis or non-negative matrix factorization to separate co-located micro-Doppler sources with overlapping frequency content but distinct spatial origins.

4. The system of claim 1, further comprising a vehicle type classifier that categorizes each detected vehicle and selects type-specific classification parameters for the occupant count estimator, accounting for differences in cabin geometry, glass angle, and body panel attenuation across vehicle categories.

5. The system of claim 1, wherein the neural network classifier is a convolutional neural network processing a time-frequency micro-Doppler spectrogram with a channel attention mechanism to weight frequency bands carrying discriminative physiological motion information, quantized to INT8 for deployment on embedded edge compute hardware.

6. A method for automated high-occupancy vehicle lane enforcement comprising: detecting a vehicle entering a monitoring zone via a trigger sensor; acquiring millimeter-wave FMCW radar data from a roadside radar unit for a duration of 2-4 seconds as the vehicle traverses the monitoring zone; compensating for vehicle bulk motion in the radar data; extracting a micro-Doppler spectrogram containing physiological motion signatures of vehicle occupants; classifying the spectrogram to estimate occupant count; and outputting a timestamped occupancy record containing only an integer occupant count and confidence score, with no image data, no license plate information, and no personally identifiable information captured by the radar system itself.

7. The method of claim 6, wherein the occupancy record is architecturally separated from vehicle identification, such that the radar-derived occupancy data is correlated with a separate, independently operated vehicle identification system at the enforcement point, maintaining a privacy firewall wherein the radar system alone cannot identify any individual or vehicle.

8. The method of claim 6, further comprising a living-occupant validation step that verifies each detected occupant exhibits respiratory and/or cardiac micro-Doppler signatures consistent with a living human, rejecting inanimate objects, mannequins, and animals below a configurable radar cross-section threshold.

9. The method of claim 6, further comprising a calibration protocol in which the system operates in shadow mode alongside ground-truth occupancy data for a calibration period, during which the classifier is fine-tuned on deployment-specific data including local vehicle mix, mounting geometry, and environmental conditions before transitioning to active enforcement.

10. The system of claim 1, adapted for detection of living occupants remaining in stationary vehicles in parking environments, wherein the absence of bulk vehicle motion simplifies micro-Doppler extraction and enables detection of child or pet presence in unattended vehicles via respiratory signature monitoring.

11. The system of claim 1, wherein the system operates through aftermarket window tinting with visible light transmission as low as 5%, through standard automotive safety glass and tempered glass, in nighttime conditions without active illumination, and in precipitation and fog conditions, by exploiting the transparency of automotive glazing materials at millimeter-wave frequencies.

## Implementation Notes

The 76-81 GHz band is allocated for vehicular radar under FCC Part 15.253 and ETSI EN 302 264. The 77 GHz wavelength (3.9 mm) provides the sensitivity needed to detect sub-millimeter chest wall displacements from respiratory and cardiac motion at ranges up to 10 meters through automotive glass.

Automotive glass attenuates 77 GHz signals by approximately 3-8 dB per surface (Yamada et al., IEEE VTC 2020). Factory-tinted and aftermarket window films operating on visible/IR absorption do not significantly affect mmWave transmission. This is the core advantage over camera-based systems.

The primary technical challenge is separating multiple occupants' micro-Doppler signatures when their respiratory frequencies overlap. MIMO spatial separation addresses driver vs. passenger side occupants, but co-located rear-seat passengers may require longer observation windows. The claimed accuracy targets (92% for binary, approximately 85% for 4-class) reflect this limitation.

The privacy advantage is structural: a camera system that promises not to store facial images still captures them; a radar system physically cannot capture images. There is no raw data to subpoena, no facial recognition to misuse, and no database of vehicle interior photographs to breach.

## Prior Art References

1. FHWA: HOV Facilities Overview (136 HOV corridors across 30 US metro areas)
2. GAO-16-781: Economic analysis of HOV lane utilization and enforcement
3. Caltrans HOV Program: HOV violation rate estimates for California corridors
4. Li et al., IEEE TMTT 2017: Through-wall human respiration detection using 77 GHz FMCW radar
5. Alizadeh et al., IEEE Access 2019: Multi-person vital sign monitoring at 60 GHz
6. Ahmad et al., IEEE Sensors Journal 2021: In-vehicle occupant detection using 77 GHz radar
7. US10867193B2 (Conduent): Camera-based vehicle occupancy detection system
8. US11120291B2 (Xerox/Conduent): Automated occupancy verification for managed lanes
9. Texas Instruments AWR2944: 76-81 GHz automotive radar transceiver SoC
10. FCC Part 15.253: Regulations for vehicular radar systems in 76-81 GHz
11. Yamada et al., IEEE VTC 2020: mmWave propagation loss through automotive glass materials
12. Illinois BIPA (740 ILCS 14): Biometric Information Privacy Act
13. California Consumer Privacy Act (CCPA): Consumer data privacy regulations
