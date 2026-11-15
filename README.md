# Robust-Rolling-Shutter-Optical-Camera-Communications
This paper proposes pilot-assisted physics-informed learning (PAPIL) for rolling-shutter optical camera communication. It combines pilot-based timing and exposure calibration, waveform inversion, and residual learning to improve bit detection under simulated camera impairments and illumination drift.
# Primary findings (43,200 independent test payload bit positions)

| Regime (14,400 bits each) | Data-only ML BER | Calibrated physics BER | PAPIL BER | PAPIL errors |
| --- | ---: | ---: | ---: | ---: |
| Nominal | 0.025833 | 0.007917 | 0.000208 | 3 |
| Difficult | 0.094792 | 0.063472 | 0.011458 | 165 |
| Shifted exposure + clipping | 0.267917 | 0.156875 | 0.087569 | 1261 |

**These are simulation outputs.** They do not prove experimental optical-camera performance or the feasibility of a particular camera image-processing pipeline. 
