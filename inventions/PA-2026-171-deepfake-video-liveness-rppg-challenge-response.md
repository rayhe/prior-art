# PA-2026-171: Deepfake Video Liveness Verification via Challenge-Response Remote Photoplethysmography

**Title:** System and Method for Deepfake Video Liveness Verification via Challenge-Response Remote Photoplethysmography

**Filing:** LITF-PA-2026-171
**Published:** September 14, 2026
**Domain:** AI Safety / Deepfake Detection
**Full Disclosure:** [liveinthefuture.org/priorart/deepfake-video-liveness-rppg-challenge-response.html](https://liveinthefuture.org/priorart/deepfake-video-liveness-rppg-challenge-response.html)
**License:** [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/) — Public Domain

> Prior Art Notice: This document is published as defensive prior art under
> [35 U.S.C. Sec. 102(a)(1)](https://www.law.cornell.edu/uscode/text/35/102).
> The inventions described herein are dedicated to the public domain as of the
> publication date above.

---

## Abstract

Disclosed is a system and method for verifying that a participant in a video stream is a live human rather than a deepfake, by turning the subject's own cardiovascular physiology into a challenge-response channel. A challenge generator produces a fresh pseudorandom illumination modulation waveform per session from a cryptographically generated seed. The modulation is emitted toward the subject's face via the viewing screen, a smart light, or an infrared illuminator, at a depth near or below the flicker-fusion threshold of human perception. A remote photoplethysmography (rPPG) extractor recovers the subject's pulse signal from skin-pixel intensity variations in the captured video. Because blood-perfused skin reflects the illumination challenge multiplicatively, the extracted rPPG signal of a genuine subject carries the challenge waveform impressed on the cardiac signal, and a lock-in correlator recovers the correlation between the known challenge and the observed signal. A generative model synthesizing a face has no access to the challenge and no physiological light-skin coupling model, so a deepfake, face-swap, reenactment, or replayed recording cannot reproduce the correlation. The method combines three independent gates: (1) physiological plausibility of the rPPG waveform (cardiac band energy, harmonic structure, heart-rate variability bounds), (2) multi-region phase coherence across face, neck, and hand regions bounded by pulse-transit-time delays, and (3) lock-in correlation with the fresh challenge. Verification completes in 8 to 15 seconds and operates entirely on the verifier side.

## Technical Field

This invention relates to computer vision and media forensics, specifically to liveness verification of human subjects in video streams using remote photoplethysmography extracted from skin-pixel color variations, combined with an active illumination challenge-response protocol that a synthetic or replayed video cannot satisfy.

## Background

Deepfakes have moved from a research curiosity to a fraud infrastructure. Voice-cloned CEOs authorize wire transfers, synthetic job candidates pass remote interviews, and real-time face-swap filters run live in video calls. Detectors trained on specific generators fail on the next generator; every forensic fingerprint is a feature the next model can learn to suppress. Physiological signals were supposed to be the exception: a face-swap model preserves pixels but not the subject's pulse, because it never saw the pulse. Ciftci, Demir, and Yin demonstrated exactly this in [FakeCatcher](https://pubmed.ncbi.nlm.nih.gov/32750816/) (IEEE TPAMI 2020), which detects synthetic portrait videos by measuring the spatial coherence and temporal consistency of biological signals extracted from facial regions.

The weakness of passive physiological detection is that it measures only absence: the fake lacks a coherent pulse. Generative models are already closing this gap. A generator trained with an rPPG consistency loss, or one that copies a real subject's rPPG waveform from a source recording into the target video, can present a plausible pulse. Once generators learn to fake the physiology, passive detection returns to the losing side of the arms race.

The known alternative is active illumination: phone face-unlock systems flash the screen to detect 3D liveness, rejecting printed photos and masks. These systems measure gross geometric and reflectance properties, not physiology, and verify geometry rather than identity continuity. They do not link the illumination challenge to a physiological signal.

Remote photoplethysmography itself is well established. [Verkruysse et al., 2008](https://pmc.ncbi.nlm.nih.gov/articles/PMC5695935/) showed the plethysmographic signal can be measured remotely from a human face using ambient light and a consumer camera. Poh, McDuff, and Picard (2010) introduced non-contact cardiac pulse measurement from webcam video using blind source separation on the RGB channels. These are measurement techniques containing no liveness challenge, no freshness guarantee, and no correlation of the physiological signal with an externally imposed, verifier-known stimulus.

The gap in the art is a liveness verification protocol in which the verifier injects a fresh, unpredictable stimulus into the scene and the stimulus can only be recovered from the video through a genuinely physiological coupling between light and blood-perfused skin. Such a protocol is not learnable by a generator, because the generator cannot predict the challenge (generated fresh per session and never transmitted to the subject), cannot reproduce the physiological coupling (it would require an accurate forward model of the subject's skin optics and hemodynamics), and cannot copy the answer from a previous session (the challenge differs every time).

## Detailed Description

### 1. System Architecture

The system comprises six functional modules: a challenge generator, an illumination emitter, a video acquisition unit, an rPPG extractor, a lock-in correlator, and a decision engine. In the primary embodiment, all modules execute on the verifier's device during a video call: the challenge generator derives a pseudorandom modulation waveform from a fresh 128-bit session seed; the emitter is the verifier-side display (or the subject's own screen, driven through the call application); the video acquisition unit is the camera capturing the subject; the rPPG extractor, correlator, and decision engine run locally on the verifier. The challenge waveform is never transmitted to the subject or included in any control data the subject's side could observe beyond the emitted light itself.

### 2. Challenge Generation

For each verification session, the challenge generator produces a seed using a cryptographically secure random number generator. The seed is expanded into a challenge waveform c(t) over a verification window T (default 12 seconds, configurable 8 to 20 seconds) using a seeded pseudorandom sequence shaped to concentrate energy in one or more narrow bands between 1 and 6 Hz, with a peak-to-peak modulation depth of 3 to 8 percent of emitter luminance. The band placement avoids the expected cardiac fundamental band of the subject's age bracket when an estimate is available, reducing cross-talk between the pulse signal and the challenge. The seed and waveform are stored verifier-side; the waveform itself is never disclosed to the subject, the calling application peer, or any network party.

When the verifier cannot control the subject's screen (e.g., a browser-based caller), the challenge generator supports a dual-band redundancy mode: two independent modulation bands are emitted simultaneously from different emitters (e.g., screen plus ambient room light via a smart-bulb API), and the lock-in correlator requires at least one band to correlate above threshold, with the requirement configurable to both bands for high-assurance sessions.

### 3. Illumination Embodiments

**Display modulation (preferred):** The subject's own display brightness or color balance is modulated at the challenge waveform, at 3 to 8 percent depth and 8 to 14 Hz, near or below the flicker-fusion threshold so the subject does not perceive the modulation or adapts to it within seconds. Green-channel-only modulation is preferred, since hemoglobin absorbs green light most strongly and rPPG extraction is dominated by the green channel, while achromatic brightness steps are the least perceptible to the viewer.

**Ambient smart-light modulation:** A controllable room light (e.g., a networked smart bulb) emits the challenge at 1 to 6 Hz with 5 to 10 percent depth. The lock-in correlator compensates for bulb slew-rate distortion by using the measured light waveform, recorded by the verifier's ambient-light sensor or recovered from a static background region in the video, as the correlation reference instead of the ideal waveform.

**Infrared modulation:** For cameras with near-infrared sensitivity (common in laptop webcams and face-unlock modules), an IR LED illuminator emits the challenge at 2 to 8 Hz with 5 to 12 percent depth. This embodiment is invisible to the human eye and usable in dark rooms; hemoglobin's IR absorption is weaker than in green, so the correlator uses longer integration windows and a higher default threshold on the cardiac morphology gate to compensate.

### 4. rPPG Extraction

The video acquisition unit captures the subject at a minimum of 30 frames per second. A face detector locates the face each frame and tracks skin regions of interest: forehead, left cheek, and right cheek, with the mouth and eye regions excluded. If the subject's hands are visible, a hand ROI is added. Per frame, spatial means of the RGB channels are computed for each ROI, forming raw traces r(t), g(t), b(t).

Traces are detrended (a smoothness-priors detrending with lambda between 10 and 100, or equivalently a high-pass filter at 0.4 Hz), and the rPPG signal s(t) is formed using the plane-orthogonal-to-skin (POS) projection: two orthogonal signals X = g - b and Y = g + b - 2r are combined as s = X + alpha * Y, where alpha is set adaptively as the ratio of standard deviations of X and Y over the window. For devices with limited compute, the classic Poh et al. 2010 approach is supported: independent component analysis on the three RGB traces with selection of the component showing the highest peak in the 0.7 to 4 Hz cardiac band.

### 5. Lock-In Correlation

The core detection step treats the subject's skin as a mixer: reflected intensity I(t) = E(t) * (R0 + Rp * p(t)), where E(t) is the illumination (including the challenge), R0 is baseline skin reflectance, Rp is the pulsatile amplitude, and p(t) is the cardiac waveform. The challenge waveform c(t) is multiplicative with both the reflectance baseline and the pulse, so band-pass filtering s(t) around the challenge band isolates a term proportional to c(t) * (R0 + Rp * p(t)). The lock-in correlator computes the normalized cross-correlation between the filtered extracted signal and the known challenge waveform over sliding 4-second windows with 50 percent overlap. A genuine subject produces correlation coefficients in the range 0.35 to 0.75; a synthetic or replayed video produces coefficients centered on zero with a standard deviation of approximately 0.15 for a 12-second window.

The decision threshold on correlation is set per embodiment and calibrated against a falsely-rejected-glass population (subjects wearing glasses, contact lenses, or makeup; low-light conditions) to bound the false-reject rate at 2 percent, then adjusted by the decision engine based on the other two gates.

### 6. The Three Gates

**Gate A, physiological plausibility:** The extracted rPPG must show cardiac-band energy (0.7 to 4 Hz) at least 6 dB above the out-of-band noise floor, a dominant peak consistent with 40 to 180 beats per minute, and inter-beat-interval coefficient of variation below 25 percent over the window. The waveform's first-harmonic to second-harmonic ratio must fall within published bounds for reflective PPG morphology.

**Gate B, multi-region coherence:** The rPPG waveforms from forehead, both cheeks, and any hand ROI must show the same cardiac fundamental within 0.05 Hz and pairwise phase offsets bounded by 300 milliseconds, consistent with pulse-transit time across body sites. Synthetic videos that paste a plausible global pulse typically fail Gate B because regional phase relationships are not preserved.

**Gate C, challenge lock-in:** The lock-in correlation of the extracted signal with the fresh challenge waveform must exceed the per-embodiment threshold. Gate C is the anti-replay gate: even a genuine recording of the subject from a previous session fails, because the previous session's illumination does not contain the current challenge.

The decision engine passes verification when all three gates pass; fails when any gate fails decisively (below its hard floor); and requests an extended challenge window (up to 20 seconds) when results are marginal, at most once per session before a definitive verdict.

### 7. Anti-Spoof Properties

- **Unpredictability:** The challenge is drawn from a 128-bit seed per session; a generator cannot pre-render a video satisfying it, and cannot learn a general mapping from challenges to valid rPPG because the valid response depends on the subject's specific skin optics and hemodynamics.
- **Freshness:** Recorded genuine footage of the subject from any prior session fails Gate C, since the embedded illumination differs from the current challenge. This covers the strongest realistic attack: a genuine past video of the target.
- **Passive observation is insufficient:** The attacker observing the emitted light sees only the modulated luminance, not the seed; reconstructing the exact 12-second waveform from a noisy visual observation and then synthesizing a physiologically correct rPPG response in real time requires inverting the subject's skin-optics model, which is underdetermined from observation alone.
- **No subject cooperation required:** The subject need not perform any action (no head turn, no spoken phrase), so the check runs invisibly during the first seconds of any video call.

### 8. Deployment Embodiments

- **Video-call client:** Integrated into a conferencing application; verification runs automatically at call start and periodically during the call, with the display as the emitter. Suitable for remote hiring interviews, executive video approvals, and customer onboarding.
- **KYC identity proofing:** A liveness check during remote identity verification, where the user holds the phone and the phone screen emits the challenge.
- **Smart-glasses / headset unlock:** An inward-facing IR camera with a modulated IR illuminator verifies that the wearer is a living person, resisting photo and mask attacks on the unlock path.
- **Forensic batch analysis:** A reference implementation in which a known-challenge illumination setup is used at capture time (e.g., a recorded witness statement under modulated room lighting); the recording later verifies against the logged challenge waveform, providing tamper evidence for the recording's liveness provenance.

### 9. Figures Description

- **Figure 1:** System block diagram showing the challenge generator, illumination emitter, subject, video acquisition unit, rPPG extractor, lock-in correlator, and decision engine with the three gates.
- **Figure 2:** Timing diagram of a 12-second verification session: challenge waveform, emitted luminance, extracted rPPG from cheek ROI showing both the cardiac pulsation and the impressed challenge envelope, and the sliding correlation coefficient rising above threshold for a genuine subject.
- **Figure 3:** Comparative waveforms for a genuine subject versus a face-swap deepfake under the same challenge: the genuine rPPG shows lock-in correlation of 0.52, the deepfake shows 0.03, with gate outcomes annotated.
- **Figure 4:** Multi-region phase diagram for forehead, cheeks, and hand ROIs showing pulse-transit-time-bounded phase offsets for a genuine subject, and uncorrelated phases for a synthetic video.

## Claims

1. A system for verifying liveness of a human subject in a video stream, comprising: a challenge generator producing a fresh pseudorandom illumination modulation waveform per verification session from a cryptographically generated seed; an illumination emitter modulating light incident on the subject's face according to the waveform; a video acquisition unit capturing video of the subject during modulation; a remote photoplethysmography extractor recovering a pulse signal from skin-pixel intensity variations in the video; a lock-in correlator computing correlation between the extracted pulse signal and the challenge waveform; and a decision engine passing liveness verification when the correlation exceeds a threshold.
2. The system of claim 1, wherein the illumination emitter is a display presenting the video call, modulating at 8 to 14 Hz with 3 to 8 percent luminance depth, and wherein the modulation is applied to the green color channel preferentially to maximize coupling with hemoglobin absorption while remaining below the subject's flicker-fusion perception threshold.
3. The system of claim 1, wherein the illumination emitter is a networked ambient light whose emitted waveform is measured by a light sensor and used as the correlation reference, compensating for emitter slew-rate distortion.
4. The system of claim 1, wherein the illumination emitter is a near-infrared illuminator invisible to the human eye, paired with an infrared-sensitive camera, the correlator using an extended integration window to compensate for weaker hemoglobin absorption in the infrared band.
5. The system of claim 1, wherein the decision engine applies three gates: (a) physiological plausibility of the extracted pulse, requiring cardiac-band energy at least 6 dB above the noise floor, a dominant peak between 40 and 180 beats per minute, and inter-beat-interval coefficient of variation below 25 percent; (b) multi-region phase coherence requiring identical cardiac fundamentals across forehead, cheek, and hand regions within 0.05 Hz with phase offsets bounded by 300 milliseconds; and (c) challenge lock-in correlation above a calibrated per-embodiment threshold; verification passing only when all three gates pass.
6. The system of claim 1, wherein the remote photoplethysmography extractor computes a plane-orthogonal-to-skin projection of the red, green, and blue channel traces from tracked skin regions, with mouth and eye regions excluded, and band-pass filters the result around the challenge band to isolate the challenge-impressed component.
7. A method for liveness verification in a video stream, comprising: generating a fresh pseudorandom illumination modulation waveform per session from a cryptographically generated seed kept verifier-side; emitting the waveform as light modulation toward the subject's face; capturing video of the subject; extracting a remote photoplethysmography signal from skin regions of the video; correlating the extracted signal with the waveform via lock-in correlation; and verifying liveness when the correlation exceeds a threshold, wherein a previously recorded video of the subject fails verification because its embedded illumination does not contain the current session's waveform.
8. The method of claim 7, wherein two independent modulation bands are emitted simultaneously from different emitters, and verification requires lock-in correlation above threshold for at least one band.
9. The method of claim 7, further comprising, when correlation is marginal, requesting at most one extended challenge window of up to 20 seconds before rendering a definitive verdict.
10. The system of claim 1, integrated into a video-conferencing application, wherein verification runs automatically during the first 8 to 15 seconds of a call without requiring any cooperative action from the subject.

## Prior Art References

1. [35 U.S.C. Sec. 102(a)(1)](https://www.law.cornell.edu/uscode/text/35/102) — Statutory basis for prior art and public disclosure
2. [Verkruysse et al., 2008](https://pmc.ncbi.nlm.nih.gov/articles/PMC5695935/) — Remote plethysmographic imaging using ambient light (Optics Express 16(26))
3. Poh, McDuff, and Picard, 2010 — "Non-contact, automated cardiac pulse measurements using video imaging and blind source separation," Optics Express 18(10):10762-10774
4. Poh, McDuff, and Picard, 2011 — "Advancements in noncontact, multiparameter physiological measurements using a webcam," IEEE Transactions on Biomedical Engineering 58:7-11
5. [Ciftci, Demir, and Yin, 2020](https://pubmed.ncbi.nlm.nih.gov/32750816/) — "FakeCatcher: Detection of Synthetic Portrait Videos using Biological Signals," IEEE TPAMI, DOI 10.1109/TPAMI.2020.3009287
6. [Ciftci, Demir, and Yin, 2019/2020](https://arxiv.org/pdf/1901.02212) — FakeCatcher arXiv preprint, arXiv:1901.02212
7. Wang, den Brinker, Stuijk, and de Haan, 2016 — "Algorithmic principles of remote-PPG," IEEE Transactions on Biomedical Engineering 64(7):1479-1491
