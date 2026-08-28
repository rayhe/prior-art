# PA-2026-154: Cooperative Pedestrian Crossing Intent Prediction Using Smart Glasses and V2X Broadcast

**Title:** System and Method for Cooperative Pedestrian Crossing Intent Prediction and Vehicle Collision Risk Mitigation Using Smart Glasses Eye Gaze, Head Orientation, and Gait Kinematics with Vehicle-to-Everything Broadcast and On-Device Transformer Inference

**Filing:** LITF-PA-2026-154  
**Published:** August 28, 2026  
**Domain:** Wearables / Automotive Safety / V2X / Edge AI  
**Full Disclosure:** [liveinthefuture.org/priorart/cooperative-pedestrian-intent-v2x-smart-glasses.html](https://liveinthefuture.org/priorart/cooperative-pedestrian-intent-v2x-smart-glasses.html)

---

## Abstract

Pedestrian fatalities in the United States reached 7,522 in 2022 according to NHTSA, a 40-year high, with 75% occurring at non-intersection or un-signalized locations where driver expectation of crossing is low. Existing Advanced Driver Assistance Systems rely on vehicle-mounted cameras and radar to detect pedestrians already in the roadway, providing at most 0.6 to 0.9 seconds of warning at urban speeds. This system inverts the sensing direction. Smart glasses equipped with inward-facing eye cameras (120 Hz), a 6-axis IMU, and an outward-facing scene camera (30 Hz) continuously estimate three complementary signals: gaze scanning pattern toward oncoming traffic lanes, head yaw orientation relative to road axis, and gait phase transitions from steady walking to deceleration and weight-shift preparatory to stepping off the curb. A 4.2 million parameter temporal transformer running on the glasses application processor fuses these signals into a crossing probability scored every 100 ms, with a personalization layer that adapts decision thresholds to individual crossing behavior over 2 to 4 weeks of wear. When crossing probability exceeds 0.75, the glasses transmit a Pedestrian Safety Message via C-V2X PC5 sidelink or DSRC per SAE J2945/9, containing anonymized position, heading, crossing confidence, and time-to-curb estimate. Receiving vehicles integrate this message into their collision risk estimator 1.5 seconds earlier than vision-only detection would allow, enabling gentle braking at 0.2g rather than emergency braking at 0.8g. All inference runs on-device with no raw eye images leaving the glasses. 15 claims.

## Key Technical Claims

1. A system for cooperative pedestrian crossing intent prediction and vehicle collision risk mitigation, comprising: smart glasses worn by a pedestrian containing at least one inward-facing eye camera, a 6-axis IMU, an outward-facing scene camera, and a V2X radio; a processor executing software that extracts gaze scanning patterns, head orientation relative to road axis, and gait kinematic features from sensor data, fuses said features using a temporal machine learning model to produce a crossing probability and time-to-curb estimate, and broadcasts a Pedestrian Safety Message via the V2X radio when the crossing probability exceeds a threshold.

2. Gaze scanning pattern features including fixation detection on oncoming traffic lanes, saccade rate and amplitude histogram over a sliding window, and a road-aligned gaze score computed as the probability that current gaze fixation falls within the angular extent of oncoming traffic lanes estimated from scene camera lane detection or map data.

3. Head orientation features including head yaw angle relative to sidewalk longitudinal axis estimated from visual-inertial odometry trajectory, head yaw angular velocity with peak detection indicating active traffic scanning, and head turn frequency counting distinct turns toward traffic over a 5 second window.

4. Gait kinematic features including step frequency via FFT peak of vertical acceleration, vertical acceleration RMS amplitude normalized to subject-specific baseline, anteroposterior deceleration impulse integral over final step, distance to curb from VIO plus scene camera curb detection, and time since last full stop to distinguish crossing from bus stop or store entry.

5. Temporal machine learning model as a transformer encoder with 2 to 6 layers, 2 to 8 attention heads, causal masking, and 3 to 5 million parameters, taking as input 20 to 40 time steps of 12 to 24 features and producing crossing probability via sigmoid plus auxiliary heads for time-to-curb regression and crossing direction classification.

6. Personalization layer using a learned subject embedding concatenated to transformer input, fine-tuned on-device over 2 to 4 weeks using only final layer parameters updated via elastic weight consolidation to adapt to individual crossing behavior without catastrophic forgetting.

7. V2X radio supporting 3GPP Release 16 C-V2X PC5 sidelink Mode 4 in the 5.9 GHz ITS band at 20 dBm, broadcasting PSM per SAE J2945/9 at 10 Hz for 5 seconds after initial trigger then 2 Hz for 10 seconds, said messages signed using IEEE 1609.2 short-lived pseudonym certificates rotated every 5 minutes.

8. Pedestrian Safety Messages including vendor-specific Information Elements for crossing confidence as uint8 0-100, time-to-curb as uint16 milliseconds 0-5000, crossing direction as uint8 enumeration, and pedestrian height as uint8 centimeters for vehicle camera region-of-interest prioritization, with total message size under 100 bytes.

9. Fallback broadcast mode using Bluetooth Low Energy Extended Advertising with coded PHY S=8 at 125 kbps and 200 meter range when C-V2X sidelink is unavailable due to regulatory or hardware constraints.

10. Roadway geofencing module that enables eye cameras, scene camera, and V2X radio only when GNSS plus OpenStreetMap road graph indicates the pedestrian is within 15 meters of a roadway centerline, reducing average power to under 30 mW and limiting operation to approximately 18 percent of urban walking time.

11. Vehicle-side threat assessment module that computes time-to-collision to predicted crossing point and initiates brake pre-charge and gentle deceleration at 0.15 to 0.25g when TTC is less than 3.0 seconds and crossing confidence exceeds 0.6, replacing emergency braking at 0.8 to 1.0g that would otherwise occur at TTC less than 1.2 seconds.

12. All inward-facing eye images processed on-device and discarded immediately after feature extraction, with no raw eye images, iris patterns, or gaze videos stored or transmitted, and gaze features retained in a 3 second ring buffer as 6 floats per frame.

13. Distance to curb estimated via visual-inertial odometry fused with scene camera curb detection using Hough line transform on depth-discontinuity edges with accuracy plus or minus 0.3 meters at 5 meters range, combined with GNSS via error-state Kalman filter encoded at 1/10 microdegree resolution per SAE J2735.

14. Step frequency, vertical acceleration amplitude, and deceleration impulse features normalized per-subject using running mean and standard deviation over previous 10 minutes of wear to adapt to individual gait patterns without explicit calibration.

15. A method for cooperative pedestrian safety comprising continuously extracting gaze scanning, head orientation, and gait features from smart glasses; fusing via on-device temporal transformer to produce crossing probability every 100 ms; when probability exceeds 0.75 broadcasting PSM via C-V2X sidelink; at receiving vehicle computing TTC and initiating brake pre-charge and gentle deceleration when TTC is less than 3.0 seconds, thereby extending effective detection horizon by 1.5 seconds compared to vision-only and reducing required deceleration from 0.8g to 0.2g.

## Prior Art Distinguished

- **Vehicle-mounted pedestrian AEB (Mobileye, Bosch, ZF):** Euro NCAP / NHTSA systems use vehicle cameras to detect pedestrians already in roadway. IIHS 2023 testing found 27% crash reduction daylight but zero at night when 74% of fatalities occur. Cannot see through occlusions or infer intent before roadway entry. This disclosure uses pedestrian-worn sensors to predict intent 1.2 to 2.8 seconds before curb departure.

- **Smartphone-based V2P via cellular (3GPP Rel 16 Uu, Commsignia, Spoke):** Smartphone apps transmit position over cellular to cloud server which relays to vehicles. Latency 80 to 300 ms plus server relay, GPS accuracy 3 to 8 meters in urban canyons insufficient to distinguish sidewalk from roadway. No gaze or head orientation signal, so crossing intent cannot be inferred. This disclosure uses direct C-V2X sidelink bypassing cellular, with gaze and head orientation for intent inference.

- **Infrastructure-based pedestrian detection (FHWA Las Vegas / Tampa deployments):** LiDAR and thermal cameras at intersections detect pedestrians and broadcast via RSU. Reduces conflicts 15 to 20% at instrumented intersections but covers less than 2% of 4.2 million US intersections per FHWA HM-10. Mid-block crossings where 58% of fatalities occur remain uncovered. This disclosure works anywhere with no infrastructure.

- **Vehicle-observed intent prediction (Fang 2018, Rasouli PIE 2020):** Uses vehicle cameras to classify head orientation (0.72 AUC at 0.8s) or gait (0.78 accuracy at 0.5s). Limited by occlusion, distance, and night performance. No prior work uses inward-facing eye tracking from smart glasses for crossing intent, nor combines gaze, head, and gait into unified on-device transformer with direct sidelink broadcast from pedestrian device.

## Why This Matters

Pedestrian deaths are a sensing horizon problem. At 30 mph a vehicle travels 8 meters during the 0.6 second average perception-reaction time plus 10.5 meters braking at 0.8g on dry asphalt. A pedestrian stepping from behind a parked SUV at 2 meters is physically unavoidable using vehicle-only sensing. The 7,522 pedestrian fatalities in 2022 represent a public health crisis that vehicle AEB alone cannot solve, as IIHS testing proves zero benefit at night when three quarters of deaths occur.

Smart glasses are converging toward daily wear for display, AI assistant, and media capture. The sensors needed for crossing intent prediction (eye cameras, IMU, scene camera, GNSS) are already present for other purposes. Adding V2X broadcast as a safety service repurposes existing hardware with only 24 mW average power overhead when geofenced to roadways. The 4.2 million C-V2X equipped vehicles projected by end of 2026 represent early adopters, but the Bluetooth LE fallback extends coverage to aftermarket receivers.

This disclosure prevents any party from patenting the concept of using smart glasses gaze, head orientation, and gait to predict pedestrian crossing intent and broadcasting that intent via V2X sidelink for cooperative collision avoidance. The fundamental methods (gaze scanning pattern toward traffic lanes, head yaw relative to road axis, gait deceleration preparatory to curb departure), the transformer fusion architecture, the PSM extension with confidence and time-to-curb, the geofenced power management, and the vehicle-side gentle braking strategy are all placed into the public domain, ensuring any wearable or automotive manufacturer can implement this life-saving capability without licensing fees.

---

*Defensive prior art — dedicated to the public domain under [35 U.S.C. § 102(a)(1)](https://www.law.cornell.edu/uscode/text/35/102)*
