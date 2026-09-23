# PA-2026-180: Microwave Oven Door-Seal RF Leakage Detection via Ambient WiFi Channel State Information

**Title:** System and Method for Detecting Microwave Oven Radio-Frequency Leakage from Degraded Door Seals Using Ambient WiFi Channel State Information

**Filing:** LITF-PA-2026-180
**Published:** September 23, 2026
**Domain:** Smart Home / RF Sensing / Appliance Safety
**Full Disclosure:** [liveinthefuture.org/priorart/microwave-door-seal-rf-leakage-wifi-csi.html](https://liveinthefuture.org/priorart/microwave-door-seal-rf-leakage-wifi-csi.html)
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

Your WiFi router already knows your microwave is leaking. A microwave oven's magnetron radiates at 2.45 GHz, inside the 2.4 GHz WiFi band, with peak interference energy concentrated on WiFi channel 9 (2.452 GHz center). Door seals and RF choke structures degrade over an oven's lifetime through food-residue carbonization, hinge sag, latch wear, and choke corrosion, allowing increasing RF energy to escape the cavity. The federal performance standard, 21 CFR 1030.10, limits lifetime leakage to 5 mW/cm2 at 5 cm from the oven surface (1 mW/cm2 before sale), and the FDA recommends annual leakage checks that almost no household performs.

The disclosed system samples physical-layer statistics already measured by WiFi access points and mesh nodes (channel state information amplitude and phase per subcarrier, noise floor, retry counters, error vector magnitude). It detects oven operation windows from the magnetron's characteristic duty-cycle envelope, normalizes measured interference by the oven's electrical input power to produce a leakage-per-watt metric that separates door-seal degradation from magnetron aging, localizes the emission source through differential measurements across two or more nodes, and tracks the metric longitudinally against an installation baseline. Tiered alerts are referenced to the regulatory limit through a calibrated path-loss model. The result is continuous, zero-added-hardware seal monitoring using infrastructure the home already owns.

## Technical Field

This invention relates to appliance safety monitoring, specifically to continuous estimation of microwave oven RF leakage from degraded door seals using ambient RF sensing performed by existing WiFi network infrastructure, with input-power normalization to distinguish seal degradation from magnetron aging and longitudinal drift tracking referenced to regulatory emission limits.

## Background

Microwave ovens number in the hundreds of millions worldwide and sit in most American kitchens. Each contains a magnetron generating roughly 700 to 1,200 watts of RF energy at 2.45 GHz. That energy is supposed to stay inside the cavity, contained by the metal enclosure, the conductive mesh embedded in the door glass, and the quarter-wave choke structure around the door perimeter. Containment degrades: hinges sag over thousands of open-close cycles, latches wear, food residue carbonizes onto seal surfaces, and choke grooves corrode.

The regulatory framework anticipates this degradation. Under 21 CFR 1030.10, microwave oven leakage may not exceed 1 mW/cm2 at any point 5 cm or more from the external surface before sale, and 5 mW/cm2 at the same distance over the oven's useful life. The FDA's Office of Regulatory Affairs recommends that leakage be checked annually and not exceed 5 mW/cm2. The standard test procedure places a 275 ml water load in the cavity and runs the oven at full power while a calibrated survey meter is swept over the whole appliance, especially the door seals, at a fixed 50 mm standoff.

Field data shows the margin erodes with age. A Health Canada survey of 60 new and 103 used ovens found before-sale average leakage of 0.3 mW/cm2 without load and 0.08 mW/cm2 with a water load, against used-oven averages of 0.52 and 0.17 mW/cm2, with used units reaching the 5 mW/cm2 limit and maximum leakage most often found at the center of the door screen. The trend is clear: seals degrade, leakage rises, and the worst point is the door. Yet the annual check the FDA recommends is performed almost nowhere in homes. Handheld leakage meters cost $30 to $200, require the user to know the procedure, and sit in a drawer.

The interference path is equally well documented. The magnetron's 2.45 GHz emission overlaps the 2.4 GHz ISM band (2.400 to 2.4835 GHz) nearly completely. Spectrum analyzer measurements show an operating oven wiping out a large portion of the band, with pulses every 16 ms derived from the half-wave rectifier in transformer-type power supplies. An IFIP measurement study found oven energy concentrated on WiFi channels 6 through 12 with the peak on channel 9, at received levels of -80 to -60 dBm with roughly 50% duty cycle, and noted that ovens are shielded when new but "with use over time these ovens can leak some radiation." Consumer testing with a spectrum analyzer showed an operating oven collapsing nearby 2.4 GHz throughput from over 100 Mbps to about 3 Mbps.

Existing responses treat this interference as a networking problem to route around: enterprise WLAN systems detect the energy and move access points to cleaner channels, and researchers have demonstrated modifying the magnetron's magnetic field to make its emission benign to nearby electronics. Pulse-level models of oven interference exist for coexistence analysis. Separately, current-signature analysis of the oven's mains draw can assess magnetron health (see LITF-PA-2026-163), but the mains side cannot see RF escaping through the door.

No published system known to the disclosers inverts the interference from nuisance to signal: using the home's WiFi plant as a distributed, always-on RF leakage dosimeter that watches the oven's seals degrade over months and years, separates seal degradation from magnetron aging by normalizing against input power, localizes the source to reject external interferers, and reports in units referenced to the regulatory limit.

## Detailed Description

### 1. Physical-Layer Sensing Substrate

The system uses measurements that WiFi radios already produce. Each access point or mesh node continuously estimates, per 2.4 GHz channel: the noise floor from periodic channel surveys; per-subcarrier channel state information (amplitude and phase across OFDM subcarriers, where firmware exposes it); MAC-layer retry and error counters; and error vector magnitude on received frames. No payload content is inspected at any stage; all features are physical-layer or MAC-counter statistics, computed on the node or the home router.

Three deployment embodiments are disclosed. In the **router-firmware embodiment**, a package on OpenWrt-class firmware (or vendor firmware exposing equivalent telemetry) samples survey noise, retry counters, and CSI where the driver exposes it, timestamping samples at 1-second granularity. In the **dedicated-sniffer embodiment**, a low-cost 2.4 GHz receiver (e.g., an ESP32-class device, unit cost under $15) placed near the kitchen performs energy detection across channels 1 through 11 and reports band-energy time series; this covers homes whose APs lack CSI export. In the **roving-survey embodiment**, a smartphone application performs periodic passive scans (RSSI and channel-utilization reports available without privileges) while the user runs a guided calibration cook, producing a spot measurement rather than continuous monitoring. The router-firmware embodiment is preferred for continuous operation.

### 2. Oven-Operation Detection and Interferer Classification

The magnetron has a distinctive temporal-spectral signature. Transformer-type ovens (common in the installed base) pulse the magnetron at the mains rate: half-wave rectification produces RF bursts every 16.7 ms on 60 Hz mains (every 20 ms on 50 Hz mains) at roughly 50% duty. Inverter-type ovens drive the magnetron continuously during the cook, producing sustained wideband energy without the mains-rate pulsing. Both types additionally exhibit a slow envelope from the power-level control: a transformer oven set to 50% power typically cycles the magnetron on and off over tens of seconds, a second periodicity at the 0.03 to 0.1 Hz scale.

A classifier distinguishes the oven from common 2.4 GHz interferers using four features: occupied bandwidth (oven: wideband, spanning channels 6 through 12 with peak at channel 9; Bluetooth: 1 MHz frequency-hopping bursts at 1600 hops/s; Zigbee: 2 MHz DSSS at low duty; analog video senders and baby monitors: narrowband continuous carriers near 100% duty); fast duty cycle at the millisecond scale; slow duty cycle at the tens-of-seconds scale (oven power-level cycling); and decodability (neighbor WiFi shows valid 802.11 preambles; oven energy is non-decodable wideband noise). The classifier outputs oven-on windows with a type label (transformer or inverter) and a confidence score; windows below the confidence threshold are excluded from the leakage metric.

### 3. Input-Power Normalization: The Leakage-Per-Watt Metric

The central non-obvious step is normalization. Measured interference at the AP depends on two independent oven variables: how much RF the magnetron generates (which declines as the magnetron ages) and what fraction escapes through the seals (which rises as seals degrade). A system tracking raw interference alone cannot tell a dying magnetron from a failing seal, and the two demand opposite responses.

The system measures the oven's electrical input power during each cook, via a smart plug on the oven circuit (preferred) or via mains-level non-intrusive load monitoring keyed to the oven's distinctive kilowatt-scale step and harmonic signature. It then computes:

```
LPW = I_cook / P_in
```

where `I_cook` is the interference energy measured at the sensing node during the cook (noise-floor elevation in linear milliwatts, averaged over the cook window and restricted to the oven-attributed sub-band), and `P_in` is the mean electrical input power over the same window. Magnetron aging reduces both numerator and denominator proportionally, leaving LPW flat. Seal degradation raises the numerator at constant denominator, driving LPW upward. For transformer ovens, the metric is computed over the magnetron-on portions of the duty envelope only.

### 4. Multi-Node Differential Localization

With two or more sensing nodes (e.g., the kitchen AP plus a hallway mesh node), the system performs differential localization: for each oven-attributed event, it compares the interference elevation across nodes against the expected indoor path-loss ratio for the household oven's known position. A neighbor's oven inverts or scrambles that ratio.

The system additionally correlates candidate events with the household's own electrical measurements. An interference event with no coincident kilowatt-scale step on the home's mains (and no smart-plug draw on the oven circuit) is classified as external, logged but excluded from the household oven's leakage trend. This is the primary defense against false attribution in apartments and dense housing.

### 5. Installation Baseline, Drift Tracking, and Thresholds

At installation, the system learns a baseline LPW over the first 30 days of operation (or a minimum of 20 oven-attributed cooks), capturing the oven's as-found seal condition including its nonzero healthy leakage floor. The user enters the AP-to-oven distance at setup, establishing the geometry for the path-loss model.

An optional guided calibration cook sharpens the baseline: the user runs the oven for 60 seconds at full power with the standard 275 ml water load while the system records. Because the load and power are standardized, calibration cooks are comparable across months and across homes.

The system tracks the Seal Degradation Index (SDI), defined as the current 30-day median LPW divided by the baseline LPW, expressed in decibels. An **advisory alert** fires when SDI exceeds +4.8 dB (a threefold rise in leakage per watt) sustained across at least 5 cooks, recommending door-seal inspection and cleaning. An **action alert** fires when the screening-grade absolute estimate of section 6 approaches the 5 mW/cm2 lifetime limit within the stated uncertainty interval, recommending verification with a handheld leakage meter or professional service. Alert hysteresis and a minimum-cook count suppress transient false alarms.

### 6. Path-Loss Inversion to Regulatory Units

The system inverts measured interference to an estimated power density at 5 cm from the oven door: estimated leaked power `P_leak = S_node * 4*pi*d^2 / G`, where `S_node` is the interference power density at the sensing node, `d` is the node-to-oven distance, and `G` is an installation calibration constant absorbing antenna gains and the oven's directional leakage pattern, learned from the guided calibration cook. The 5 cm estimate follows from spherical spreading: `S_5cm = P_leak / (4*pi*25 cm^2)`.

Illustrative example: a node 3 meters from the oven measuring a cook-averaged interference elevation of -60 dBm, with calibration constant near unity, implies leaked power on the order of 0.1 mW in the node's direction, corresponding to roughly 0.3 mW/cm2 at 5 cm from the door. Indoor multipath makes any single absolute reading uncertain by roughly +/-6 to 10 dB; the absolute estimate is therefore reported as screening-grade with an explicit interval, while the SDI trend, which cancels the static multipath geometry, is the primary alert signal. Scope of validity: this is a screening and trending instrument, not a certification-grade survey meter, and action alerts recommend confirmatory measurement.

### 7. Applications

- **Homeowner monitoring:** Continuous background tracking closing the gap between the FDA's annual-check recommendation and actual household practice.
- **Multifamily property management:** A building-wide deployment flags degrading ovens across hundreds of units without entering apartments.
- **Commercial kitchens:** Automated logging of the leakage trend documents due diligence for health and safety inspections.
- **Insurance and warranty (opt-in):** With explicit user consent, the leakage trend supports appliance-breakdown coverage triage. Any third-party use must present the user the full input set behind the score and a contestation path, and coverage may not be denied on the screening estimate alone.

### 8. Figures Description

- **Figure 1:** System overview: kitchen with microwave oven, home WiFi access point and mesh node, smart plug on the oven circuit, and the processing pipeline from PHY statistics through interferer classification, power normalization, and alerting.
- **Figure 2:** Spectral-temporal signature panel: wideband oven interference spanning WiFi channels 6 through 12 peaking at channel 9, with the 16.7 ms mains-rate pulsing of a transformer oven and the continuous envelope of an inverter oven, contrasted with narrowband Bluetooth hopping and continuous analog video sender carriers.
- **Figure 3:** Leakage-per-watt decomposition: two trend lines over 24 months, one showing magnetron aging (interference and input power declining together, LPW flat) and one showing seal degradation (interference rising at constant input power, LPW rising).
- **Figure 4:** Multi-node differential localization geometry: two sensing nodes at different distances from the oven showing the expected path-loss ratio for the household oven versus the inverted ratio for a neighbor's oven.
- **Figure 5:** Seal Degradation Index trend with the installation baseline, the +4.8 dB advisory threshold, and the regulatory-limit-referenced action threshold with its uncertainty interval.

## Claims

1. A system for monitoring microwave oven radio-frequency leakage, comprising: at least one WiFi sensing node in a household, configured to sample physical-layer statistics including channel state information, noise floor, retry counters, and error vector magnitude on 2.4 GHz channels; an oven-operation detector that attributes interference windows to a microwave oven from the magnetron's duty-cycle envelope; an input-power sensor measuring the oven's electrical power draw during attributed windows; a normalization module computing a leakage-per-watt metric as measured interference energy divided by input power; and an alerting module that tracks the metric longitudinally against an installation baseline and generates alerts on sustained elevation.

2. The system of claim 1, wherein the oven-operation detector classifies transformer-type ovens by mains-rate RF pulsing at approximately 50% duty and inverter-type ovens by continuous wideband emission during cooks, and distinguishes both from Bluetooth frequency-hopping bursts, Zigbee transmissions, analog video sender carriers, and decodable neighbor WiFi by occupied bandwidth, fast and slow duty-cycle features, and signal decodability.

3. The system of claim 1, wherein the leakage-per-watt metric separates door-seal degradation, which raises measured interference at constant input power, from magnetron aging, which reduces interference and input power proportionally, leaving the metric unchanged.

4. The system of claim 1, further comprising at least two WiFi sensing nodes performing differential localization, wherein the ratio of interference elevations across nodes is compared against the expected indoor path-loss ratio for the household oven's known position, and events failing the ratio test or lacking coincident household power draw are classified as external and excluded from the leakage trend.

5. The system of claim 1, further comprising a path-loss inversion module that converts the measured interference to an estimated power density at 5 cm from the oven door using an installation calibration constant learned from a guided calibration cook, and references alert thresholds to the 5 mW/cm2 lifetime limit of 21 CFR 1030.10 with a stated uncertainty interval.

6. The system of claim 1, wherein the alerting module computes a Seal Degradation Index as the current median leakage-per-watt divided by the installation baseline, generating an advisory alert on sustained elevation exceeding approximately +4.8 dB and an action alert when the screening-grade absolute estimate approaches the regulatory limit within its uncertainty interval.

7. The system of claim 1, wherein the WiFi sensing node is implemented as firmware on the household's existing access point or mesh node, sampling survey noise, retry counters, and channel state information without inspecting payload content.

8. The system of claim 1, wherein the WiFi sensing node is implemented as a dedicated low-cost 2.4 GHz energy-detection receiver, and wherein the oven-operation detector runs on a home hub receiving the receiver's band-energy time series.

9. A method for detecting microwave oven door-seal degradation comprising: passively sampling WiFi physical-layer statistics during household operation; attributing interference windows to a microwave oven via magnetron duty-cycle envelope classification; measuring the oven's electrical input power during attributed windows; computing a leakage-per-watt metric normalized by the input power; establishing an installation baseline over initial operation; and generating tiered alerts on sustained longitudinal elevation of the metric relative to the baseline.

10. The method of claim 9, further comprising performing a guided calibration cook at full power with a standardized water load to establish a comparable cross-time reference, and using the calibration to learn a path-loss inversion constant mapping measured interference to estimated power density at 5 cm from the oven door.

11. The method of claim 9, further comprising multifamily deployment across units of a managed building, aggregating per-unit leakage trends into a maintenance schedule flagging ovens for seal service or replacement without entering units.

## Implementation Notes

The minimum viable implementation is a software package on OpenWrt-class router firmware plus a smart plug on the oven circuit. Sampling `iw survey` noise data and per-station retry counters at 1-second granularity, with CSI from debugfs where the driver exposes it, is sufficient for the oven-attribution classifier; subcarrier-level CSI improves noise rejection but is not required.

Measurement repeatability is the controlling specification. Bench testing should establish the cook-to-cook repeatability of the leakage-per-watt metric for a fixed oven and geometry (target: +/-1.5 dB, 95% interval), characterizing sensitivity to cookware load, food water content, food position, and AP client traffic. The slow power-level cycling envelope of transformer ovens must be handled by restricting the metric to magnetron-on segments; naive averaging over a whole cook at 50% power setting halves the metric and mimics improvement.

Known limitations, stated plainly. Indoor multipath makes absolute power-density estimates uncertain by roughly +/-6 to 10 dB, which is why the absolute reading is labeled screening-grade and the installation-canceled trend carries the alerts. Homes operating only on 5 GHz, or with no 2.4 GHz sensing node within roughly 10 meters of the oven, are out of scope. Metal-heavy commercial kitchens widen the absolute interval further, though trending still functions. A neighbor's oven in dense housing is the principal false-attribution risk, addressed by the differential-localization and power-correlation gates, but the gates are probabilistic, not perfect. Unusual loads (all-metal cookware, empty-cavity operation) can transiently move the metric; the multi-cook sustainment requirement exists for this reason.

The strongest case against this approach is that WiFi physical-layer metrics are too confounded by multipath, client traffic, AP channel changes, and geometry error to quantify leakage usefully, producing either false alarms or false reassurance. The design answers this in four places: per-installation baselining cancels static multipath; the trend rather than the absolute number drives alerts; absolute readings carry explicit uncertainty intervals and are labeled screening-grade; and action alerts recommend confirmatory measurement with a real survey meter rather than asserting a violation. The system claims early warning and triage, not certification.

Data governance: all features are physical-layer or MAC-counter statistics; no payload content is captured, stored, or transmitted. Processing runs on the home router or hub by default; cloud embodiments transmit only aggregated per-cook metrics with user consent. Multifamily deployments aggregate per-unit trends accessible only to building maintenance for the units they service, with tenant notice.

## Prior Art References

1. 21 CFR 1030.10 (https://www.ecfr.gov/current/title-21/chapter-I/subchapter-J/part-1030/section-1030.10): Performance standard for microwave ovens: 1 mW/cm2 pre-sale, 5 mW/cm2 lifetime limit at 5 cm
2. FDA Office of Regulatory Affairs (https://www.fda.gov/media/73598/download): Leakage should be checked annually and not exceed 5 mW/cm2
3. Health Canada, Radiation Leakage of Before-Sale and Used Microwave Ovens (2000) (https://www.canada.ca/en/health-canada/services/environmental-workplace-health/reports-publications/radiation/radiation-leakage-before-sale-used-microwave-ovens-health-canada-2000.html): Leakage rises with age; maximum most often at center of door screen
4. EDN, Wi-Fi Network Interference Analysis and Optimization (https://www.edn.com/wi-fi-network-interference-analysis-and-optimization/): Spectrum analyzer capture of oven interference; 16 ms pulsing from transformer supply
5. Mahanti et al., Ambient Interference Effects in Wi-Fi Networks (IFIP Networking 2010) (http://dl.ifip.org/db/conf/networking/networking2010/MahantiCWA10.pdf): Oven energy on channels 6-12 peaking at channel 9, -80 to -60 dBm, ~50% duty cycle
6. Macworld/MetaGeek (https://www.macworld.com/article/222241/fact-or-fiction-what-affects-wi-fi-speed.html): Operating oven collapses 2.4 GHz throughput from 100+ Mbps to ~3 Mbps
7. University of Michigan, Gilgenbach et al. (https://www.sciencedaily.com/releases/2003/12/031204073640.htm): Magnetron magnetic-field modification to reduce interference (mitigation approach)
8. IJACSA 2018, Real-Time Experimentation of WiFi Spectrum Utilization in Microwave Oven Noisy Environment (https://thesai.org/Downloads/Volume9No1/Paper_67-Real_Time_Experimentation_and_Analysis_of_WiFi_Spectrum.pdf): Pulse-level interference modeling for coexistence
9. Martindale, Requirements for Microwave Leakage Testing (https://professional-electrician.com/technical/requirements-for-microwave-leakage-testing-in-offices-martindale/): Standard test procedure: 50 mm standoff, sweep of door seals, 5 mW/cm2 limit
10. LITF-PA-2026-163, Microwave Magnetron Health via Current Signature (https://liveinthefuture.org/priorart/microwave-magnetron-health-current-signature.html): Companion disclosure: mains-side magnetron assessment, which cannot observe door-seal RF leakage
