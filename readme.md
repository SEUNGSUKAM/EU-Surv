# Abstract

**Title: Event Uncertainty-Aware Deep Dynamic Survival Framework for Competing Risks**

Dynamic risk prediction using irregular longitudinal clinical records is essential for personalized decision support. However, existing deep dynamic survival models face three primary limitations: (i) sharp and temporally inconsistent risk distributions caused by supervising event times as single discrete bins; (ii) a lack of explicit mechanisms for modeling event-occurrence uncertainty, which implicitly assumes event inevitability; and (iii) unstable early-stage predictions due to sparse historical data. 

To address these challenges, we propose an **event uncertainty-aware deep dynamic survival framework** tailored for competing risks. Our approach introduces three key innovations:

1.  **Tail-Aware Likelihood Term:** To mitigate overconfidence from discretization, we implement a likelihood term that allocates probability mass beyond the observed event bin. This acts as temporal label smoothing, encouraging smoother cumulative risk curves and increasing entropy in discrete time.
2.  **Dual Risk Head Modules:** We decouple the prediction into a cause-specific time distribution head (conditioned on occurrence) and an event-occurrence distribution head. Their combination enables realistic, uncertainty-aware joint event-time probabilities.
3.  **Robust Representation Learning:** To stabilize early-stage predictions, we utilize an attention-based recurrent temporal network initialized with a static token derived from baseline covariates, ensuring robust performance even from the first clinical visit.

Experimental results on two competing-risk benchmarks (PBC2 and MIMIC-III) and a real-world hyperthyroidism cohort demonstrate that our framework significantly improves discrimination and calibration over state-of-the-art dynamic survival baselines.


## How to Run
To execute the model and reproduce the results, please run the following script:

python main.py
