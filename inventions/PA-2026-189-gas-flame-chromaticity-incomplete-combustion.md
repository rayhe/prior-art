# PA-2026-189: Residential Gas Burner Incomplete-Combustion Detection via Flame Chromaticity and Flicker Analysis on Consumer Cameras

**Title:** System and Method for Residential Gas Burner Incomplete-Combustion Detection via Flame Chromaticity and Flicker Analysis on Consumer Cameras

**Filing:** LITF-PA-2026-189
**Published:** September 30, 2026
**Domain:** Appliances / Computer Vision / Home Safety
**Full Disclosure:** [liveinthefuture.org/priorart/gas-flame-chromaticity-incomplete-combustion.html](https://liveinthefuture.org/priorart/gas-flame-chromaticity-incomplete-combustion.html)
**License:** [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/) — Public Domain

> Prior Art Notice: This document is published openly as a technical disclosure
> under [35 U.S.C. Sec. 102(a)(1)](https://www.law.cornell.edu/uscode/text/35/102).
> Whether it constitutes prior art, and what it discloses, depends on the facts,
> including public accessibility, timing, and the disclosure's content.
> Publication does not establish novelty, patentability, freedom to operate, or
> public-domain status. This disclosure is offered as evidence of the state of
> the art for examiners and challengers to consider. This is general
> information, not legal advice.

---

## Abstract

Disclosed is a system that repurposes an ordinary consumer kitchen camera, such as a range-hood camera or a kitchen security camera, as a combustion-quality sensor for gas stovetops. On-device computer vision segments the flame of each burner, measures the chromaticity balance between blue chemiluminescence (complete combustion) and yellow-orange soot incandescence (incomplete combustion), and tracks flame flicker dynamics including lift-off from the burner ports. When a burner exhibits a persistent yellow-flame pattern during steady operation, the system issues a staged user advisory identifying the burner and the likely maintenance causes (clogged burner ports, misaligned cap, misadjusted air shutter). Transient confounders such as sodium flares from spilled food are rejected by persistence gating; camera white-balance drift is corrected with a reference patch; and static warm light sources are excluded by flicker-based flame localization. The disclosure optionally fuses the optical finding with a listed carbon-monoxide detector reading for corroboration. Automated gas shutoff is explicitly excluded from this disclosure.

## Technical Field

This disclosure relates to residential appliance safety and computer vision, specifically to the use of existing consumer cameras and on-device image analysis to detect incomplete combustion in gas burners from flame color and flame dynamics, and to staged advisory architectures that turn a leading optical indicator into a maintenance recommendation before combustion byproducts accumulate.

## Background

Gas cooking carries a well-documented indoor air quality burden. Gas stoves emit nitrogen dioxide, carbon monoxide, and fine particulate matter at levels the EPA and WHO consider unsafe (Beacon Journal / USA Today), and a Stanford-led study published in *Environmental Science & Technology* found that a single gas burner on high or an oven at 350 °F can raise indoor benzene above secondhand-smoke levels, with benzene forming directly in the flames (Stanford Report, June 2023). The U.S. Consumer Product Safety Commission issued a Request for Information on chronic hazards of gas range use after Commissioner Trumka called gas stoves a "hidden hazard" (Air Quality News).

The oldest diagnostic instrument for burner health is the human eye. A healthy gas flame burns blue; a yellow or orange flame indicates the burner is starved of air, producing incomplete combustion, soot, and carbon monoxide. The usual causes are mundane: grease and food clogging the burner ports, a misaligned burner cap, or a misadjusted air shutter (Family Fresh Recipes). Millions of households own this diagnostic signal and ignore it, because nobody watches their flames for the minutes it takes a bad pattern to matter.

Existing mitigations are reactive or professional-only. Carbon-monoxide detectors alarm only after CO has already accumulated in the room. Combustion analyzers that measure CO air-free, oxygen, and stack temperature are carried by HVAC technicians, not homeowners. Industrial flame monitoring — UV and IR flame scanners and flame-image combustion analysis — guards boilers and furnaces, not kitchens, and costs orders of magnitude more than a cooktop. No consumer product uses the cameras already installed in kitchens, in range hoods and on counters, to watch flame color as a leading indicator of incomplete combustion. This disclosure records that system.

## Detailed Description

### 1. Sensing architecture and burner registration

The sensor is any consumer camera with a view of the cooktop: a range-hood camera, a kitchen security camera, or a dedicated stovetop camera. During one-time setup, the user registers each burner by tapping its center in the camera view (or the system auto-discovers burner positions from the circular geometry of the grates and the ring pattern of a lit burner). Each burner receives a region of interest (ROI) mask. Frames are analyzed on-device at 5 to 10 frames per second, which is sufficient for a signal that evolves over minutes; no kitchen video leaves the device.

### 2. Flame segmentation

Within each burner ROI, flame pixels are segmented from the dark cooktop background by a combination of brightness thresholding and blue-channel dominance: flame pixels are bright and, in the healthy case, strongly blue. The segmentation is refined by temporal flicker (Section 4) so that static bright objects such as a polished pot are not counted as flame. The result is a per-burner flame mask updated each frame.

### 3. Chromaticity analysis: blue chemiluminescence versus soot incandescence

The physical basis of the measurement is that the two combustion regimes radiate differently. A clean premixed natural-gas flame is blue because of chemiluminescence from excited CH* and C2* radicals in the reaction zone. When combustion goes fuel-rich or air-starved, unburned carbon forms soot particles that glow yellow-orange by broadband black-body incandescence. The camera therefore sees the combustion quality directly: blue means the chemistry is completing; yellow means soot is forming and carbon monoxide is the likely co-product.

Flame pixels are converted to a perceptual color space (HSV or CIELAB) and classified into a blue class (hue near 200 to 240 degrees) and a yellow class (hue near 40 to 70 degrees with high saturation). The primary metric is the **yellow-tip fraction (YTF)**: the fraction of flame-mask pixels falling in the yellow class. A secondary metric is the **combustion quality index (CQI)**: the ratio of blue-band to yellow-band radiant energy within the flame mask. Example alert parameters, disclosed as one workable embodiment rather than a fixed boundary: an advisory is raised when YTF exceeds 0.35, or CQI falls below 2.0, sustained across a 90-second sliding window during steady burner operation. A small, steady yellow tip is normal (dust and trace sodium color the tips of an otherwise healthy flame); the system learns each burner's normal YTF during a calibration burn and alerts on deviation, not on an absolute value alone.

### 4. Flicker dynamics and flame lift-off

Color is corroborated by shape and motion. An air-starved burner produces a taller, lazier flame that can lift off the burner cap, detaching from the ports and floating above them, which is both a symptom and an amplifier of poor mixing. The system tracks flame height (vertical extent of the flame mask) and the **flame-root gap**: the distance between the burner cap plane and the lowest flame pixel. A sustained root gap larger than a disclosed example threshold of 8 millimeters during steady operation is recorded as lift-off. Flame flicker is measured by taking the Fourier spectrum of the flame mask's total intensity over a 10-second window; lifted, fuel-rich flames show elevated low-frequency flicker power relative to a seated blue flame. Flicker therefore serves two roles: it distinguishes live flame from static warm-colored objects (a tungsten under-cabinet light does not flicker at flame frequencies), and it provides an independent dynamics channel for the incomplete-combustion assessment.

### 5. Confounder handling

Three confounders are disclosed with their handling. First, **camera white balance**: consumer cameras shift white balance with scene lighting, which would corrupt any color measurement. The system either locks the camera white balance during setup or, preferably, normalizes against a small printed reference color patch affixed inside the range hood or on the backsplash, visible in frame; measured flame colors are corrected against the patch's known values each session. Second, **transient sodium flares**: spilled salt or food produces brilliant yellow-orange flares that are chemically unrelated to combustion quality. These are brief (seconds) and are rejected by the persistence gate: only a yellow pattern sustained for minutes across the sliding window can raise an advisory. Third, **occlusion**: a pot on the burner hides the flame. The system pauses scoring for a burner when the flame mask collapses while the burner is known to be on, and resumes when the flame is visible again; a burner that is never visible is reported as unmonitored rather than scored.

### 6. Staged advisory and escalation

Advisories are staged and attributed per burner. *Level one:* when the persistence gate trips, the user receives a notification naming the burner and the likely causes in plain language, for example: "Burner 2 has shown a persistent yellow flame for the last 10 minutes, consistent with incomplete combustion. Check for clogged burner ports or a misaligned cap, and confirm the flame returns to blue." *Level two:* if the pattern recurs across multiple cooking sessions, the system recommends professional service (burner orifice inspection, gas pressure, air-shutter adjustment). *Optional corroboration:* where a listed carbon-monoxide detector or indoor air-quality monitor is paired, the advisory notes whether elevated CO was observed during the same session, but the optical advisory does not depend on it. The system is advisory and diagnostic; it performs no gas shutoff and issues no diagnosis of any medical condition.

### 7. What is not claimed

For clarity of scope, this disclosure explicitly excludes: automated actuation of any gas valve or gas shutoff; detection or measurement of any specific gas concentration from the camera image alone (the camera measures a combustion-quality proxy, not parts per million); and any claim that the advisory replaces a listed carbon-monoxide detector, which remains the required safety device. A yellow flame from a correctly installed decorative or intentionally yellow-tipped burner (for example, certain wok or high-BTU burners specified by the manufacturer to run with yellow tips) is handled by the per-burner calibration baseline and is not an alert condition.

## Claims

1. A system for detecting incomplete combustion in a residential gas burner, comprising: a consumer camera positioned with a view of a cooktop; a burner registration defining a region of interest for each burner; a flame segmentation module producing a per-burner flame mask; a chromaticity analyzer classifying flame pixels into a blue chemiluminescence class and a yellow soot-incandescence class and computing a yellow-tip fraction; a persistence gate requiring the yellow-tip fraction to remain above an alert threshold across a multi-minute sliding window during steady burner operation; and an advisory module that issues a per-burner user advisory naming the burner and maintenance causes when the gate trips.
2. The system of claim 1, wherein the chromaticity analyzer computes a combustion quality index as the ratio of blue-band to yellow-band radiant energy within the flame mask, and wherein the persistence gate operates on the combustion quality index falling below a calibrated threshold.
3. The system of claim 1, further comprising a flicker dynamics module that measures a flame-root gap between the burner cap plane and the lowest flame pixel and the Fourier flicker spectrum of flame intensity, wherein a sustained flame-root gap above a lift-off threshold or elevated low-frequency flicker power corroborates the chromaticity finding.
4. The system of claim 3, wherein the flicker dynamics module distinguishes live flame from static warm-colored objects by the presence of flame-frequency intensity fluctuation, and excludes static objects from the flame mask.
5. The system of claim 1, further comprising a white-balance compensation module that normalizes measured flame colors against a reference color patch visible in the camera frame, or that locks the camera white balance at setup, to prevent lighting-induced chromaticity drift.
6. The system of claim 1, wherein the persistence gate rejects transient sodium flares from spilled food or salt, the flares being shorter in duration than the multi-minute sliding window, so that only sustained yellow patterns raise an advisory.
7. The system of claim 1, wherein the burner registration supports multiple burners with independent regions of interest, and the advisory names the specific burner exhibiting the persistent yellow pattern.
8. The system of claim 1, further comprising a calibration baseline captured per burner during a user-confirmed healthy burn, wherein the alert threshold is a deviation from the per-burner baseline, accommodating burners specified by the manufacturer to run with yellow tips.
9. The system of claim 1, further comprising an occlusion handler that pauses scoring for a burner when the flame mask collapses while the burner is on, and reports the burner as unmonitored rather than scored when the flame is never visible.
10. The system of claim 1, further comprising an escalation module that, upon recurrence of the persistent yellow pattern across multiple cooking sessions, recommends professional burner service, and that optionally notes corroborating readings from a paired listed carbon-monoxide detector without depending on them.
11. The system of claim 1, wherein all frame analysis is performed on-device with no kitchen video transmitted off the device, except for a user-initiated diagnostic snapshot.
12. A method for detecting incomplete combustion in a residential gas burner, comprising: capturing cooktop frames from a consumer camera; segmenting a per-burner flame mask; classifying flame pixels into blue chemiluminescence and yellow soot-incandescence classes; computing a yellow-tip fraction; requiring the fraction to remain above an alert threshold across a multi-minute window; compensating for white-balance drift with a reference patch; rejecting transient flares shorter than the window; and issuing a per-burner advisory naming maintenance causes; wherein the method performs no gas shutoff and measures no gas concentration from the image.

## Implementation Notes

A workable build runs on camera-adjacent compute: a range-hood system-on-chip, a smart-display processor, or a Raspberry-Pi-class hub receiving the camera stream over the local network. Five to ten frames per second is sufficient; the signal of interest evolves over minutes, and lower frame rates cut compute and thermal load. The per-burner baseline calibration can be as simple as a 60-second capture after the user confirms the burner is clean and the cap seated, stored as a reference YTF and CQI. The reference color patch can be a printed card supplied with the product or printed by the user; its known sRGB values are stored at setup. Notifications go through the existing smart-home app channel. Total bill of materials for a retrofit embodiment is a camera the household may already own, which is the point: the sensing hardware is already installed in millions of kitchens.

## Prior Art References

1. [Cooking on gas stoves emits benzene](https://news.stanford.edu/stories/2023/06/cooking-gas-stoves-emits-benzene-2), Stanford Report, June 2023: single gas burner on high or oven at 350 °F raised indoor benzene above secondhand-smoke levels; benzene forms in the flames (Lebel et al., *Environ. Sci. Technol.* 2023, DOI: 10.1021/acs.est.2c09289)
2. [US safety commission issues Request for Information on safety of gas cookers](https://airqualitynews.com/health/us-safety-commission-issues-request-for-information-on-safety-of-gas-cookers/), Air Quality News: CPSC chronic-hazard review of gas range use; Commissioner Trumka "hidden hazard" remarks
3. [Gas stove ban? US Consumer Product Safety Commission mulls it](https://www.beaconjournal.com/story/money/2023/01/10/gas-stove-ban-us/11022254002/), Beacon Journal / USA Today, January 2023: gas stoves emit nitrogen dioxide, carbon monoxide, and fine particulate matter at levels EPA and WHO deem unsafe
4. [Restoring a Gas Stove Burner Flame to Bright Blue](https://yum.familyfreshrecipes.com/wp/2025/08/12/restoring-a-gas-stove-burner-flame-to-bright-blue/), Family Fresh Recipes: yellow flame indicates incomplete combustion and possible carbon monoxide; causes include clogged ports, misaligned cap, misadjusted air shutter
5. Industrial flame monitoring and UV/IR flame safeguard controls for boilers and furnaces (e.g., Honeywell flame safeguard product lines): professional, high-cost flame supervision for industrial burners; no consumer cooktop embodiment using existing kitchen cameras
6. Carbon-monoxide detectors listed to UL 2034: reactive alarming after CO accumulation; complementary to, not a substitute for, the leading optical indicator disclosed here
