# Deep Research Implementation: Structure-Constrained Systems Pharmacology

This folder contains the implementation of the "Deep Research" blueprint for analyzing Vemurafenib/Trametinib resistance.

## Core Concept
We bridge **Structural Biology (OpenFold3)** and **Systems Biology (ODE Models)** by translating structural variants into specific parameter sets.

## Files
1. **`DeepResearch_Sims.m`**: The main driver script. Run this to see the comparative resistance dynamics.
   - Loops through 5 key scenarios: WT, L505H (Dimer+), Splice (Constitutive Dimer), MEK_C121S (Drug Resistant), RAS_G12V (Upstream Drive).
   - Generates "Novel Results" plots showing how different structural mechanisms lead to distinct phenotypic rebound curves.
2. **`get_variant_modifiers.m`**: The "Translation Layer". Use this to map new OpenFold3 insights (e.g., "Mutation X weakens inhibitor binding by 50%") to model parameters.
3. **`Mapk_ODE_DR.m`**: A standalone version of the MAPK/PI3K ODE system, adapted for this research package.

## How to Run
1. Open MATLAB.
2. Navigate to `SCMPA/Matlab/DeepResearch`.
3. Run `DeepResearch_Sims`.

## Extensions
- To add a new mutation found by OpenFold3, add a case to `get_variant_modifiers.m`.
- To calibrate against new IC50 data, adjust the baseline parameters in `DeepResearch_Sims.m`.
