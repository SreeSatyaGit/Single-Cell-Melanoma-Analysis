% =============================================================================
% VEMURAFENIB EFFECTS ON MAPK AND PI3K/AKT/mTOR PATHWAYS
% BRAF^V600E, CRAF, and ARAF with Paradoxical Activation
% =============================================================================
%
% IMPORTANT: All concentrations and species values are normalized to [0, 1]
% - Initial conditions: Total pools = 1.0, distributed between active/inactive forms
% - Input signals: pRTK and vemurafenib normalized to [0, 1]
% - IC50: Normalized value (0.1 = 10% of maximum drug concentration)
%
% Model Description:
% This script models the effects of vemurafenib on:
% - MAPK pathway with BRAF^V600E mutation, CRAF, and ARAF isoforms
% - PI3K/AKT/mTOR pathway
% - Paradoxical activation of CRAF and ARAF by vemurafenib
% - Cross-talk mechanisms and adaptive responses
%
% PATHWAY 1: EGFR → MAPK (RAS-RAF-MEK-ERK)
%   - Shc recruitment and phosphorylation by pEGFR
%   - Grb2 binding and SOS activation
%   - RAS activation by SOS*
%   - BRAF^V600E constitutive activation (inhibited by vemurafenib)
%   - CRAF activation (RAS-dependent, AKT-inhibited, paradoxically activated by vemurafenib)
%   - ARAF activation (RAS-dependent, weak, minor paradoxical activation)
%   - BRAF-WT dimerization with CRAF/ARAF (paradoxical activation mechanism)
%   - Sequential MEK and ERK phosphorylation
%   - ERK-P negative feedback (relieved when vemurafenib reduces ERK-P)
%   - DUSP (dual-specificity phosphatase) feedback: pERK-induced DUSP transcription
%     and translation, DUSP enhances ERK dephosphorylation
%   - SPRY (Sprouty) feedback: pERK-induced SPRY transcription and translation,
%     SPRY sequesters Grb2, reducing SOS activation
%
% PATHWAY 2: EGFR → PI3K/AKT/mTOR
%   - PI3K activation by pEGFR/pHER3 (and RAS-GTP cross-talk)
%   - PIP2 to PIP3 conversion
%   - AKT dual phosphorylation (Thr308, Ser473)
%   - TSC1/TSC2 inhibition by AKT and ERK-P (cross-talk)
%   - Rheb-GTP accumulation and mTORC1 activation
%   - S6K and 4EBP1 phosphorylation
%   - pS6K negative feedback on PI3K
%
% VEMURAFENIB EFFECTS:
% 1. Direct inhibition: Vemurafenib inhibits BRAF^V600E kinase activity using Hill equation
% 2. Paradoxical activation: Vemurafenib can trans-activate CRAF and ARAF in dimers with BRAF-WT
% 3. Feedback relief: Reduced ERK-P relieves negative feedback on SOS/Shc
% 4. Adaptive signaling: RAS-GTP rebound → PI3K activation rebound
% 5. Cross-talk changes: Reduced ERK-P → reduced ERK→TSC2 enhancement
%
% CROSS-TALK MECHANISMS:
% 1. RAS → PI3K: RAS-GTP activates PI3K (rebound after vemurafenib)
% 2. ERK → TSC2/mTORC1: ERK-P phosphorylates TSC2 (reduced after vemurafenib)
% 3. AKT → CRAF: AKT inhibits CRAF (not BRAF^V600E or ARAF)
% 4. pS6K → PI3K: Negative feedback on PI3K activation
% 5. PTEN: Degrades PIP3, limiting AKT activation
%
% =============================================================================

clear all;
close all;
clc;

fprintf('═══════════════════════════════════════════════════════════════════════════\n');
fprintf('   VEMURAFENIB EFFECTS ON MAPK AND PI3K/AKT/mTOR PATHWAYS\n');
fprintf('   BRAF^{V600E}, CRAF, and ARAF with Paradoxical Activation\n');
fprintf('═══════════════════════════════════════════════════════════════════════════\n\n');

%% ============================================================================
% INPUT SIGNALS
% ============================================================================

% Define pRTK (pEGFR or pHER3) as a function of time (normalized to [0, 1])
% Option 1: Transient pulse (exponential decay, normalized)
pRTK_func = @(t) 1.0 * exp(-0.05 * t) .* (t >= 0);  % Normalized: starts at 1.0, decays

% Option 2: Step function (constant, normalized)
% pRTK_func = @(t) 1.0 * (t >= 0);  % Normalized: constant at 1.0

% Define vemurafenib concentration as a function of time (normalized to [0, 1])
% Option 1: Step function (constant drug concentration, normalized)
% Drug is applied at t = 1500 minutes
vemurafenib_func = @(t) 1.0 * (t >= 0);  % Normalized: drug = 1.0 (maximum) from t = 1500 min

% Option 2: Ramp function (gradual increase)
% vemurafenib_func = @(t) 1.0 * max(0, (t - 50) / 10) .* (t >= 50) .* (t <= 60) + 1.0 * (t > 60);

% Option 3: Time-dependent (decay)
% vemurafenib_func = @(t) 1.0 * exp(-0.02 * (t - 50)) .* (t >= 50);

fprintf('Using pRTK (pEGFR/pHER3) as input signal.\n');
fprintf('Using vemurafenib(t) as drug input signal.\n\n');

%% ============================================================================
% PARAMETERS
% ============================================================================

fprintf('Setting up parameters for both pathways with vemurafenib effects...\n');
fprintf('NOTE: All rate constants have been reduced by a factor of 10 to slow down reactions.\n\n');

% ============================================================================
% MAPK PATHWAY PARAMETERS
% ============================================================================

% Shc–Grb2–SOS Module 
k_on_Shc = 10e-5;      % nM^-1 min^-1 - Shc binding to pEGFR (base rate)
k_off_Shc = 10e-6;     % min^-1 - Dissociation of pEGFR:Shc
k_cat1 = 10e-3;         % min^-1 - Shc phosphorylation rate
k_cat_SOS = 7e-3;      % min^-1 - SOS activation rate (base rate)
k_on2 = 10e-5;         % nM^-1 min^-1 - Grb2 binding to pShc
k_off2 = 10e-4;        % min^-1 - Dissociation of pShc:Grb2
k_on3 = 10e-3;         % nM^-1 min^-1 - SOS binding to pShc:Grb2
k_off3 = 10e-3;        % min^-1 - Dissociation of ternary complex
k_deg4 = 10e-5;        % min^-1 - SOS* deactivation rate

% RAS Module 
k_SOS = 10e-3;          % min^-1 - RAS activation rate by SOS*
k_GTPase = 5e-3;       % min^-1 - RAS GTP hydrolysis rate (base rate)

% RAF Isoforms 
k_BRAF_mut = 10e-2;     % min^-1 - BRAF^V600E constitutive activation rate
k_BRAF_mut_deact = 10e-10; % min^-1 - BRAF^V600E* deactivation rate
k_CRAF_base = 10e-3;    % min^-1 - CRAF base activation rate (RAS-dependent)
k_CRAF_deact = 10e-4;  % min^-1 - CRAF* deactivation rate
k_ARAF_base = 10e-3;   % min^-1 - ARAF base activation rate (RAS-dependent, weak)
k_ARAF_deact = 10e-4;   % min^-1 - ARAF* deactivation rate

% BRAF-WT for dimerization 
k_BRAF_WT = 10e-3;      % min^-1 - BRAF-WT activation rate (RAS-dependent)
k_BRAF_WT_deact = 10e-5; % min^-1 - BRAF-WT* deactivation rate
k_dimer = 6e-3;        % nM^-1 min^-1 - Dimer formation rate (BRAF-WT with CRAF/ARAF)

% MEK and ERK Module 
k_MEK = 10e-3;          % min^-1 - MEK phosphorylation rate
k_MEK_deact = 10e-5;   % min^-1 - MEK-P dephosphorylation rate
k_ERK = 10e-3;          % min^-1 - ERK phosphorylation rate
k_ERK_deact = 10e-5;   % min^-1 - ERK-P dephosphorylation rate

% MAPK Feedback Parameters (ERK-P feedback)
alpha_ERK_Shc = 0.5;   % ERK-P inhibition of Shc binding
alpha_ERK_SOS = 0.5;   % ERK-P inhibition of SOS activation
alpha_ERK_GTPase = 0.2; % ERK-P enhancement of RAS GTPase

% ============================================================================
% PI3K/AKT/mTOR PATHWAY PARAMETERS
% ============================================================================

% PI3K Module (slowed down by factor of 10)
k_PI3K = 10e-3;         % min^-1 - PI3K activation rate by pRTK (base rate)
k_PI3K_deact = 10e-5;  % min^-1 - PI3K deactivation rate
k_PIP2_to_PIP3 = 0.05; % min^-1 - PIP2 to PIP3 conversion rate
k_PTEN = 0.01;         % min^-1 - PTEN-mediated PIP3 dephosphorylation

% Cross-talk: RAS → PI3K (slowed down by factor of 10)
k_PI3K_RAS = 0.02;     % min^-1 - PI3K activation rate by RAS-GTP

% AKT Module (slowed down by factor of 10)
k_AKT_recruit = 10e-3;  % min^-1 - AKT recruitment rate by PIP3
k_AKT_Thr308 = 10e-3;   % min^-1 - AKT Thr308 phosphorylation rate
k_AKT_Ser473 = 10e-3;   % min^-1 - AKT Ser473 phosphorylation rate
k_AKT_deact = 10e-5;   % min^-1 - AKT dephosphorylation rate

% TSC1/TSC2 and Rheb Module (slowed down by factor of 10)
k_TSC2_inh = 10e-2;     % min^-1 - TSC2 inhibition rate by pAKT (base rate)
k_TSC2_base = 10e-2;    % min^-1 - Base TSC2 inhibition rate
k_Rheb_act = 10e-3;     % min^-1 - Rheb-GTP activation rate
k_Rheb_GTPase = 10e-3;  % min^-1 - Rheb GTPase activity

% Cross-talk: ERK → TSC2/mTORC1
alpha3 = 0.2;          % ERK-P enhancement of TSC2 inhibition

% mTORC1 Module (slowed down by factor of 10)
k_mTORC1 = 10e-3;       % min^-1 - mTORC1 activation rate
k_mTORC1_deact = 10e-5; % min^-1 - mTORC1 deactivation rate

% S6K and 4EBP1 Module (slowed down by factor of 10)
k_S6K = 10e-3;          % min^-1 - S6K phosphorylation rate
k_S6K_deact = 10e-5;   % min^-1 - pS6K dephosphorylation rate
k_4EBP1 = 10-3;        % min^-1 - 4EBP1 phosphorylation rate
k_4EBP1_deact = 10e-5; % min^-1 - p4EBP1 dephosphorylation rate

% PI3K Feedback Parameters
alpha1 = 0.5;          % pS6K inhibition of PI3K activation

% Cross-talk: AKT → CRAF (CRAF only, not BRAF^V600E or ARAF)
alpha2 = 0.3;          % pAKT inhibition of CRAF activation

% ============================================================================
% VEMURAFENIB PARAMETERS (Hill Equation)
% ============================================================================

% Hill equation parameters for vemurafenib inhibition of BRAF^V600E (normalized)
IC50_vem = 0.4;        % Normalized [0,1] - Half-maximal inhibitory concentration (IC50)
                      % At 10% of max drug concentration, 50% inhibition occurs
Hill_n = 1.5;         % Hill coefficient (cooperativity, typically 1-2 for drugs)
% Hill equation inhibition: 
%k_BRAF_mut_eff = k_BRAF_mut * IC50^n / (IC50^n + (vemurafenib)^n);
% This gives: 0% inhibition at [Drug]=0, 50% at [Drug]=IC50, ~100% at [Drug]>>IC50
% All concentrations are normalized to [0, 1]

% Paradoxical activation parameters
gamma = 0.5;           % Paradoxical CRAF activation strength
delta = 0.1;           % Paradoxical ARAF activation strength (minor)

% ============================================================================
% DUSP AND SPRY FEEDBACK PARAMETERS
% ============================================================================

% DUSP (Dual-specificity phosphatase) parameters
Vtx_DUSP = 0.5;        % Normalized [0,1]/min - Maximum DUSP transcription rate
K_DUSP = 0.1;          % Normalized [0,1] - Hill Kd for pERK induction
n_DUSP = 2.0;          % Hill coefficient for DUSP transcription
k_deg_mDUSP = 0.2;     % min^-1 - DUSP mRNA degradation rate
k_tl_DUSP = 0.2;       % min^-1 - DUSP translation rate
k_deg_DUSP = 0.05;     % min^-1 - DUSP protein degradation rate
k_cat_DUSP = 0.5;      % min^-1 - DUSP catalytic rate constant
Vmax_DUSP = 1.0;       % min^-1 - Maximum DUSP dephosphorylation rate
Km_DUSP = 0.1;         % Normalized [0,1] - Michaelis-Menten Km for pERK

% SPRY (Sprouty) parameters
Vtx_SPRY = 0.3;        % Normalized [0,1]/min - Maximum SPRY transcription rate
K_SPRY = 0.1;          % Normalized [0,1] - Hill Kd for pERK induction
n_SPRY = 2.0;          % Hill coefficient for SPRY transcription
k_deg_mSPRY = 0.2;     % min^-1 - SPRY mRNA degradation rate
k_tl_SPRY = 0.2;       % min^-1 - SPRY translation rate
k_deg_SPRY = 0.05;     % min^-1 - SPRY protein degradation rate
k_bind_SPRY_Grb2 = 0.01; % Normalized [0,1]^-1 min^-1 - SPRY:Grb2 binding rate
k_unbind_SPRY_Grb2 = 0.1; % min^-1 - SPRY:Grb2 unbinding rate
k_deg_complex = 0.05;  % min^-1 - SPRY:Grb2 complex degradation rate
gamma_SPRY = 0.1;      % Normalized [0,1]^-1 - SPRY inhibition strength (multiplicative mode)
h_coop = 1.0;          % Cooperativity parameter for Grb2 effect

% Store all parameters in structure
params.k_on_Shc = k_on_Shc;
params.k_off_Shc = k_off_Shc;
params.k_cat1 = k_cat1;
params.k_cat_SOS = k_cat_SOS;
params.k_on2 = k_on2;
params.k_off2 = k_off2;
params.k_on3 = k_on3;
params.k_off3 = k_off3;
params.k_deg4 = k_deg4;
params.k_SOS = k_SOS;
params.k_GTPase = k_GTPase;
params.k_BRAF_mut = k_BRAF_mut;
params.k_BRAF_mut_deact = k_BRAF_mut_deact;
params.k_CRAF_base = k_CRAF_base;
params.k_CRAF_deact = k_CRAF_deact;
params.k_ARAF_base = k_ARAF_base;
params.k_ARAF_deact = k_ARAF_deact;
params.k_BRAF_WT = k_BRAF_WT;
params.k_BRAF_WT_deact = k_BRAF_WT_deact;
params.k_dimer = k_dimer;
params.k_MEK = k_MEK;
params.k_MEK_deact = k_MEK_deact;
params.k_ERK = k_ERK;
params.k_ERK_deact = k_ERK_deact;
params.alpha_ERK_Shc = alpha_ERK_Shc;
params.alpha_ERK_SOS = alpha_ERK_SOS;
params.alpha_ERK_GTPase = alpha_ERK_GTPase;
params.k_PI3K = k_PI3K;
params.k_PI3K_deact = k_PI3K_deact;
params.k_PI3K_RAS = k_PI3K_RAS;
params.k_PIP2_to_PIP3 = k_PIP2_to_PIP3;
params.k_PTEN = k_PTEN;
params.k_AKT_recruit = k_AKT_recruit;
params.k_AKT_Thr308 = k_AKT_Thr308;
params.k_AKT_Ser473 = k_AKT_Ser473;
params.k_AKT_deact = k_AKT_deact;
params.k_TSC2_inh = k_TSC2_inh;
params.k_TSC2_base = k_TSC2_base;
params.k_Rheb_act = k_Rheb_act;
params.k_Rheb_GTPase = k_Rheb_GTPase;
params.alpha3 = alpha3;
params.k_mTORC1 = k_mTORC1;
params.k_mTORC1_deact = k_mTORC1_deact;
params.k_S6K = k_S6K;
params.k_S6K_deact = k_S6K_deact;
params.k_4EBP1 = k_4EBP1;
params.k_4EBP1_deact = k_4EBP1_deact;
params.alpha1 = alpha1;
params.alpha2 = alpha2;
params.IC50_vem = IC50_vem;
params.Hill_n = Hill_n;
params.gamma = gamma;
params.delta = delta;
params.Vtx_DUSP = Vtx_DUSP;
params.K_DUSP = K_DUSP;
params.n_DUSP = n_DUSP;
params.k_deg_mDUSP = k_deg_mDUSP;
params.k_tl_DUSP = k_tl_DUSP;
params.k_deg_DUSP = k_deg_DUSP;
params.k_cat_DUSP = k_cat_DUSP;
params.Vmax_DUSP = Vmax_DUSP;
params.Km_DUSP = Km_DUSP;
params.Vtx_SPRY = Vtx_SPRY;
params.K_SPRY = K_SPRY;
params.n_SPRY = n_SPRY;
params.k_deg_mSPRY = k_deg_mSPRY;
params.k_tl_SPRY = k_tl_SPRY;
params.k_deg_SPRY = k_deg_SPRY;
params.k_bind_SPRY_Grb2 = k_bind_SPRY_Grb2;
params.k_unbind_SPRY_Grb2 = k_unbind_SPRY_Grb2;
params.k_deg_complex = k_deg_complex;
params.gamma_SPRY = gamma_SPRY;
params.h_coop = h_coop;
params.pRTK_func = pRTK_func;
params.vemurafenib_func = vemurafenib_func;

fprintf('Parameters configured for both pathways with vemurafenib effects.\n');
fprintf('DUSP and SPRY feedback parameters included.\n\n');

%% ============================================================================
% INITIAL CONDITIONS
% ============================================================================

fprintf('Setting initial conditions (normalized to [0, 1])...\n');

% MAPK Pathway species (normalized: total pools = 1.0)
% Shc pool: Shc + pEGFR:Shc + pShc = 1.0
Shc0 = 1.0;           % Normalized: inactive Shc
pEGFR_Shc0 = 0.0;     % Normalized: pEGFR:Shc complex
pShc0 = 0.0;          % Normalized: phosphorylated Shc

% Grb2 pool: Grb2 + pShc:Grb2 = 1.0
Grb2_0 = 1.0;         % Normalized: free Grb2
pShc_Grb2_0 = 0.0;    % Normalized: pShc:Grb2 complex

% SOS pool: SOS + pShc:Grb2:SOS + SOS* = 1.0
SOS_0 = 1.0;          % Normalized: inactive SOS
pShc_Grb2_SOS_0 = 0.0; % Normalized: ternary complex
SOS_star0 = 0.0;      % Normalized: active SOS*

% RAS pool: RAS_GDP + RAS_GTP = 1.0
RAS_GDP0 = 1.0;       % Normalized: inactive RAS-GDP
RAS_GTP0 = 0.0;       % Normalized: active RAS-GTP

% BRAF^V600E pool: BRAF_mut + BRAF_mut* = 1.0
BRAF_mut0 = 0.0;      % Normalized: inactive BRAF^V600E
BRAF_mut_star0 = 1.0; % Normalized: active BRAF^V600E* (constitutively active)

% CRAF pool: CRAF + CRAF* = 1.0
CRAF0 = 1.0;          % Normalized: inactive CRAF
CRAF_star0 = 0.0;     % Normalized: active CRAF*

% ARAF pool: ARAF + ARAF* = 1.0
ARAF0 = 1.0;          % Normalized: inactive ARAF
ARAF_star0 = 0.0;     % Normalized: active ARAF*

% BRAF-WT pool: BRAF_WT + BRAF_WT* = 1.0
BRAF_WT0 = 1.0;       % Normalized: inactive BRAF-WT
BRAF_WT_star0 = 0.0;  % Normalized: active BRAF-WT*

% Dimer pools (for paradoxical activation)
BRAF_WT_CRAF_dimer0 = 0.0;  % Normalized: BRAF-WT:CRAF dimer
BRAF_WT_ARAF_dimer0 = 0.0;  % Normalized: BRAF-WT:ARAF dimer

% MEK pool: MEK + MEK_P = 1.0
MEK0 = 1.0;           % Normalized: inactive MEK
MEK_P0 = 0.0;         % Normalized: phosphorylated MEK

% ERK pool: ERK + ERK_P = 1.0
ERK0 = 1.0;           % Normalized: inactive ERK
ERK_P0 = 0.0;         % Normalized: phosphorylated ERK

% PI3K/AKT/mTOR Pathway species (normalized: total pools = 1.0)
% PI3K pool: PI3K + PI3K_active = 1.0
PI3K0 = 1.0;          % Normalized: inactive PI3K
PI3K_active0 = 0.0;   % Normalized: active PI3K

% PIP pool: PIP2 + PIP3 = 1.0
PIP2_0 = 1.0;         % Normalized: PIP2
PIP3_0 = 0.0;         % Normalized: PIP3

% AKT pool: AKT + pAKT_Thr308 + pAKT_Ser473 + pAKT_full = 1.0
AKT0 = 1.0;           % Normalized: inactive AKT
pAKT_Thr308_0 = 0.0;  % Normalized: AKT phosphorylated at Thr308
pAKT_Ser473_0 = 0.0;  % Normalized: AKT phosphorylated at Ser473
pAKT_full_0 = 0.0;    % Normalized: AKT phosphorylated at both sites

% TSC2 pool: TSC2_active = 1.0 (simplified, active form)
TSC2_active0 = 1.0;   % Normalized: active TSC2

% Rheb pool: Rheb_GDP + Rheb_GTP = 1.0
Rheb_GTP0 = 0.0;      % Normalized: active Rheb-GTP

% mTORC1 pool: mTORC1 + mTORC1_active = 1.0
mTORC1_0 = 1.0;       % Normalized: inactive mTORC1
mTORC1_active0 = 0.0; % Normalized: active mTORC1

% S6K pool: S6K + pS6K = 1.0
S6K0 = 1.0;           % Normalized: inactive S6K
pS6K0 = 0.0;          % Normalized: phosphorylated S6K

% 4EBP1 pool: 4EBP1 + p4EBP1 = 1.0
EBP1_0 = 1.0;         % Normalized: inactive 4EBP1
p4EBP1_0 = 0.0;       % Normalized: phosphorylated 4EBP1

% DUSP and SPRY feedback species (normalized: start at zero, induced by pERK)
mDUSP0 = 0.0;         % Normalized: DUSP mRNA
DUSP0 = 0.0;          % Normalized: DUSP protein
mSPRY0 = 0.0;         % Normalized: SPRY mRNA
SPRY0 = 0.0;          % Normalized: SPRY protein
SPRY_Grb2_0 = 0.0;    % Normalized: SPRY:Grb2 complex

% Combined initial state vector: [MAPK species (24), PI3K species (16), DUSP/SPRY (5)]
y0 = [Shc0; pEGFR_Shc0; pShc0; Grb2_0; pShc_Grb2_0; SOS_0; ...
      pShc_Grb2_SOS_0; SOS_star0; RAS_GDP0; RAS_GTP0; ...
      BRAF_mut0; BRAF_mut_star0; CRAF0; CRAF_star0; ARAF0; ARAF_star0; ...
      BRAF_WT0; BRAF_WT_star0; BRAF_WT_CRAF_dimer0; BRAF_WT_ARAF_dimer0; ...
      MEK0; MEK_P0; ERK0; ERK_P0; ...
      PI3K0; PI3K_active0; PIP2_0; PIP3_0; AKT0; pAKT_Thr308_0; ...
      pAKT_Ser473_0; pAKT_full_0; TSC2_active0; Rheb_GTP0; ...
      mTORC1_0; mTORC1_active0; S6K0; pS6K0; EBP1_0; p4EBP1_0; ...
      mDUSP0; DUSP0; mSPRY0; SPRY0; SPRY_Grb2_0];

fprintf('Initial conditions set. BRAF^{V600E}* initially active.\n');
fprintf('DUSP and SPRY start at zero (induced by pERK).\n\n');

%% ============================================================================
% TIME SPAN
% ============================================================================

tspan = [0, 3000];  % minutes

%% ============================================================================
% SOLVE ODE SYSTEM
% ============================================================================

fprintf('═══════════════════════════════════════════════════════════════════════════\n');
fprintf('   SOLVING ODE SYSTEM WITH VEMURAFENIB INHIBITION\n');
fprintf('═══════════════════════════════════════════════════════════════════════════\n\n');

fprintf('Solving ODE system (this may take a moment)...\n');
options = odeset('RelTol', 1e-6, 'AbsTol', 1e-8);
[t, y] = ode45(@(t,y) vemurafenib_craf_araf_odes(t, y, params), tspan, y0, options);

% Extract MAPK pathway species
Shc = y(:, 1);
pEGFR_Shc = y(:, 2);
pShc = y(:, 3);
Grb2 = y(:, 4);
pShc_Grb2 = y(:, 5);
SOS = y(:, 6);
pShc_Grb2_SOS = y(:, 7);
SOS_star = y(:, 8);
RAS_GDP = y(:, 9);
RAS_GTP = y(:, 10);
BRAF_mut = y(:, 11);
BRAF_mut_star = y(:, 12);
CRAF = y(:, 13);
CRAF_star = y(:, 14);
ARAF = y(:, 15);
ARAF_star = y(:, 16);
BRAF_WT = y(:, 17);
BRAF_WT_star = y(:, 18);
BRAF_WT_CRAF_dimer = y(:, 19);
BRAF_WT_ARAF_dimer = y(:, 20);
MEK = y(:, 21);
MEK_P = y(:, 22);
ERK = y(:, 23);
ERK_P = y(:, 24);

% Extract PI3K/AKT/mTOR pathway species
PI3K = y(:, 25);
PI3K_active = y(:, 26);
PIP2 = y(:, 27);
PIP3 = y(:, 28);
AKT = y(:, 29);
pAKT_Thr308 = y(:, 30);
pAKT_Ser473 = y(:, 31);
pAKT_full = y(:, 32);
TSC2_active = y(:, 33);
Rheb_GTP = y(:, 34);
mTORC1 = y(:, 35);
mTORC1_active = y(:, 36);
S6K = y(:, 37);
pS6K = y(:, 38);
EBP1 = y(:, 39);
p4EBP1 = y(:, 40);

% Extract DUSP and SPRY feedback species
mDUSP = y(:, 41);
DUSP = y(:, 42);
mSPRY = y(:, 43);
SPRY = y(:, 44);
SPRY_Grb2 = y(:, 45);

% Calculate input signals over time
pRTK_signal = arrayfun(pRTK_func, t);
vemurafenib_signal = arrayfun(vemurafenib_func, t);

fprintf('Simulation completed successfully!\n\n');

%% ============================================================================
% PLOTTING
% ============================================================================

fprintf('═══════════════════════════════════════════════════════════════════════════\n');
fprintf('   GENERATING VISUALIZATIONS\n');
fprintf('═══════════════════════════════════════════════════════════════════════════\n\n');

% Figure 1: Vemurafenib, BRAF, RAS, MEK, and ERK Species
figure('Position', [50, 50, 1400, 900]);

% Vemurafenib concentration
subplot(2, 3, 1);
plot(t, vemurafenib_signal, 'k-', 'LineWidth', 2.5, 'DisplayName', 'Vemurafenib');
xlabel('Time (minutes)', 'FontSize', 12);
ylabel('Normalized Concentration [0,1]', 'FontSize', 12);
title('Vemurafenib', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 10);
grid on;
xlim([0, tspan(2)]);
yline(0, 'k--', 'LineWidth', 1);
hold off;

% ALL RAS Species
subplot(2, 3, 2);
plot(t, RAS_GDP, 'c--', 'LineWidth', 1.5, 'DisplayName', 'RAS-GDP');
hold on;
plot(t, RAS_GTP, 'c-', 'LineWidth', 2.5, 'DisplayName', 'RAS-GTP');
xlabel('Time (minutes)', 'FontSize', 12);
ylabel('Normalized Concentration [0,1]', 'FontSize', 12);
title('ALL RAS Species', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 10);
grid on;
xlim([0, tspan(2)]);
hold off;

% ALL BRAF Species
subplot(2, 3, 3);
plot(t, BRAF_mut, 'r--', 'LineWidth', 1.5, 'DisplayName', 'BRAF^{V600E}');
hold on;
plot(t, BRAF_mut_star, 'r-', 'LineWidth', 2.5, 'DisplayName', 'BRAF^{V600E}*');
plot(t, BRAF_WT, 'b--', 'LineWidth', 1.5, 'DisplayName', 'BRAF-WT');
plot(t, BRAF_WT_star, 'b-', 'LineWidth', 2, 'DisplayName', 'BRAF-WT*');
xlabel('Time (minutes)', 'FontSize', 12);
ylabel('Normalized Concentration [0,1]', 'FontSize', 12);
title('ALL BRAF Species', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 10);
grid on;
xlim([0, tspan(2)]);
hold off;

% MEK and pMEK
subplot(2, 3, 4);
plot(t, MEK, 'y--', 'LineWidth', 1.5, 'DisplayName', 'MEK');
hold on;
plot(t, MEK_P, 'y-', 'LineWidth', 2.5, 'DisplayName', 'pMEK (MEK-P)');
xlabel('Time (minutes)', 'FontSize', 12);
ylabel('Normalized Concentration [0,1]', 'FontSize', 12);
title('MEK and pMEK', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 10);
grid on;
xlim([0, tspan(2)]);
hold off;

% ERK and pERK
subplot(2, 3, 5);
plot(t, ERK, 'g--', 'LineWidth', 1.5, 'DisplayName', 'ERK');
hold on;
plot(t, ERK_P, 'g-', 'LineWidth', 2.5, 'DisplayName', 'pERK (ERK-P)');
xlabel('Time (minutes)', 'FontSize', 12);
ylabel('Normalized Concentration [0,1]', 'FontSize', 12);
title('ERK and pERK', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 10);
grid on;
xlim([0, tspan(2)]);
hold off;

% Combined MAPK cascade
subplot(2, 3, 6);
plot(t, RAS_GTP, 'c-', 'LineWidth', 2, 'DisplayName', 'RAS-GTP');
hold on;
plot(t, MEK_P, 'y-', 'LineWidth', 2, 'DisplayName', 'pMEK');
plot(t, ERK_P, 'g-', 'LineWidth', 2.5, 'DisplayName', 'pERK');
xlabel('Time (minutes)', 'FontSize', 12);
ylabel('Normalized Concentration [0,1]', 'FontSize', 12);
title('MAPK Cascade: RAS → MEK → ERK', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 10);
grid on;
xlim([0, tspan(2)]);
hold off;

sgtitle('BRAF, RAS, MEK, and ERK Species Dynamics', ...
        'FontSize', 16, 'FontWeight', 'bold');

% Figure 2: DUSP and SPRY Feedback Dynamics
figure('Position', [100, 100, 1400, 900]);

% DUSP mRNA and Protein
subplot(2, 3, 1);
plot(t, mDUSP, 'r--', 'LineWidth', 1.5, 'DisplayName', 'mDUSP (mRNA)');
hold on;
plot(t, DUSP, 'r-', 'LineWidth', 2.5, 'DisplayName', 'DUSP (protein)');
xlabel('Time (minutes)', 'FontSize', 12);
ylabel('Normalized Concentration [0,1]', 'FontSize', 12);
title('DUSP Expression', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 10);
grid on;
xlim([0, tspan(2)]);
hold off;

% SPRY mRNA and Protein
subplot(2, 3, 2);
plot(t, mSPRY, 'b--', 'LineWidth', 1.5, 'DisplayName', 'mSPRY (mRNA)');
hold on;
plot(t, SPRY, 'b-', 'LineWidth', 2.5, 'DisplayName', 'SPRY (protein)');
xlabel('Time (minutes)', 'FontSize', 12);
ylabel('Normalized Concentration [0,1]', 'FontSize', 12);
title('SPRY Expression', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 10);
grid on;
xlim([0, tspan(2)]);
hold off;

% SPRY:Grb2 Complex
subplot(2, 3, 3);
plot(t, SPRY_Grb2, 'm-', 'LineWidth', 2.5, 'DisplayName', 'SPRY:Grb2');
hold on;
Grb2_free_plot = max(0, 1.0 - pShc_Grb2 - SPRY_Grb2);  % Free Grb2
plot(t, Grb2_free_plot, 'c-', 'LineWidth', 2, 'DisplayName', 'Grb2_{free}');
xlabel('Time (minutes)', 'FontSize', 12);
ylabel('Normalized Concentration [0,1]', 'FontSize', 12);
title('SPRY:Grb2 Complex and Free Grb2', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 10);
grid on;
xlim([0, tspan(2)]);
hold off;

% pERK with DUSP Feedback Effect
subplot(2, 3, 4);
plot(t, ERK_P, 'g-', 'LineWidth', 2.5, 'DisplayName', 'pERK');
hold on;
plot(t, DUSP, 'r--', 'LineWidth', 2, 'DisplayName', 'DUSP');
xlabel('Time (minutes)', 'FontSize', 12);
ylabel('Normalized Concentration [0,1]', 'FontSize', 12);
title('pERK and DUSP Feedback', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 10);
grid on;
xlim([0, tspan(2)]);
hold off;

% SOS Activation and SPRY Feedback
subplot(2, 3, 5);
plot(t, SOS_star, 'Color', [1 0.5 0], 'LineWidth', 2.5, 'DisplayName', 'SOS*');
hold on;
plot(t, SPRY, 'b--', 'LineWidth', 2, 'DisplayName', 'SPRY');
xlabel('Time (minutes)', 'FontSize', 12);
ylabel('Normalized Concentration [0,1]', 'FontSize', 12);
title('SOS* Activation and SPRY Feedback', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 10);
grid on;
xlim([0, tspan(2)]);
hold off;

% Combined Feedback: pERK, DUSP, and SPRY
subplot(2, 3, 6);
yyaxis left;
plot(t, ERK_P, 'g-', 'LineWidth', 2.5, 'DisplayName', 'pERK');
ylabel('pERK (normalized [0,1])', 'FontSize', 12);
hold on;
yyaxis right;
plot(t, DUSP, 'r--', 'LineWidth', 2, 'DisplayName', 'DUSP');
plot(t, SPRY, 'b--', 'LineWidth', 2, 'DisplayName', 'SPRY');
ylabel('DUSP/SPRY (normalized [0,1])', 'FontSize', 12);
xlabel('Time (minutes)', 'FontSize', 12);
title('Combined Feedback: pERK, DUSP, and SPRY', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 10);
grid on;
xlim([0, tspan(2)]);
hold off;

sgtitle('DUSP and SPRY Feedback Dynamics', ...
        'FontSize', 16, 'FontWeight', 'bold');

% Calculate contributions for summary statistics
contribution_BRAF_mut = k_MEK * BRAF_mut_star .* MEK;
contribution_CRAF = k_MEK * CRAF_star .* MEK;
contribution_ARAF = k_MEK * ARAF_star .* MEK;
contribution_BRAF_WT = k_MEK * BRAF_WT_star .* MEK;

%% ============================================================================
% SUMMARY STATISTICS
% ============================================================================

fprintf('═══════════════════════════════════════════════════════════════════════════\n');
fprintf('   SUMMARY STATISTICS\n');
fprintf('═══════════════════════════════════════════════════════════════════════════\n\n');

% Find drug addition time
drug_time_idx = find(vemurafenib_signal > 0, 1);
drug_time_idx = max(1, min(drug_time_idx, length(t)));  % Default to last time if not found
drug_time = t(drug_time_idx);

fprintf('VEMURAFENIB TREATMENT:\n');
fprintf('  Drug added at:        t = %.2f minutes\n', drug_time);
fprintf('  Peak drug concentration: %.4f (normalized [0,1])\n', max(vemurafenib_signal));

fprintf('\nMAPK PATHWAY - RAF ISOFORMS:\n');
% Handle case where drug starts at t=0
before_idx = max(1, drug_time_idx - 1);
[max_BRAF_mut_star, idx_BRAF_mut] = max(BRAF_mut_star);
[max_CRAF_star, idx_CRAF] = max(CRAF_star);
[max_ARAF_star, idx_ARAF] = max(ARAF_star);
[max_MEK_P, idx_MEK] = max(MEK_P);
[max_ERK_P, idx_ERK] = max(ERK_P);
fprintf('  Peak BRAF^{V600E}*:    %.4f (normalized [0,1]) at t = %.2f min\n', max_BRAF_mut_star, t(idx_BRAF_mut));
fprintf('  Peak CRAF*:            %.4f (normalized [0,1]) at t = %.2f min\n', max_CRAF_star, t(idx_CRAF));
fprintf('  Peak ARAF*:            %.4f (normalized [0,1]) at t = %.2f min\n', max_ARAF_star, t(idx_ARAF));
fprintf('  Peak MEK-P:            %.4f (normalized [0,1]) at t = %.2f min\n', max_MEK_P, t(idx_MEK));
fprintf('  Peak ERK-P:            %.4f (normalized [0,1]) at t = %.2f min\n', max_ERK_P, t(idx_ERK));

fprintf('\nPI3K/AKT/mTOR PATHWAY:\n');
[max_PIP3, idx_PIP3] = max(PIP3);
[max_pAKT_full, idx_pAKT] = max(pAKT_full);
[max_mTORC1_active, idx_mTOR] = max(mTORC1_active);
[max_pS6K, idx_S6K] = max(pS6K);
[max_p4EBP1, idx_4EBP1] = max(p4EBP1);
fprintf('  Peak PIP3:            %.4f (normalized [0,1]) at t = %.2f min\n', max_PIP3, t(idx_PIP3));
fprintf('  Peak pAKT(Full):      %.4f (normalized [0,1]) at t = %.2f min\n', max_pAKT_full, t(idx_pAKT));
fprintf('  Peak mTORC1_active:   %.4f (normalized [0,1]) at t = %.2f min\n', max_mTORC1_active, t(idx_mTOR));
fprintf('  Peak pS6K:            %.4f (normalized [0,1]) at t = %.2f min\n', max_pS6K, t(idx_S6K));
fprintf('  Peak p4EBP1:          %.4f (normalized [0,1]) at t = %.2f min\n', max_p4EBP1, t(idx_4EBP1));

fprintf('\nDUSP AND SPRY FEEDBACK:\n');
[max_mDUSP, idx_mDUSP] = max(mDUSP);
[max_DUSP, idx_DUSP] = max(DUSP);
[max_mSPRY, idx_mSPRY] = max(mSPRY);
[max_SPRY, idx_SPRY] = max(SPRY);
[max_SPRY_Grb2, idx_SPRY_Grb2] = max(SPRY_Grb2);
fprintf('  Peak mDUSP:           %.4f (normalized [0,1]) at t = %.2f min\n', max_mDUSP, t(idx_mDUSP));
fprintf('  Peak DUSP:            %.4f (normalized [0,1]) at t = %.2f min\n', max_DUSP, t(idx_DUSP));
fprintf('  Peak mSPRY:           %.4f (normalized [0,1]) at t = %.2f min\n', max_mSPRY, t(idx_mSPRY));
fprintf('  Peak SPRY:            %.4f (normalized [0,1]) at t = %.2f min\n', max_SPRY, t(idx_SPRY));
fprintf('  Peak SPRY:Grb2:       %.4f (normalized [0,1]) at t = %.2f min\n', max_SPRY_Grb2, t(idx_SPRY_Grb2));
% Calculate minimum free Grb2
min_Grb2_free = min(max(0, 1.0 - pShc_Grb2 - SPRY_Grb2));
fprintf('  Minimum Grb2_free:    %.4f (normalized [0,1])\n', min_Grb2_free);

fprintf('\nPARADOXICAL ACTIVATION:\n');
fprintf('  Peak BRAF-WT:CRAF dimer: %.4f (normalized [0,1])\n', max(BRAF_WT_CRAF_dimer));
fprintf('  Peak BRAF-WT:ARAF dimer: %.4f (normalized [0,1])\n', max(BRAF_WT_ARAF_dimer));
fprintf('  CRAF* contribution to MEK-P: %.1f%%\n', ...
        100 * max(contribution_CRAF) / (max(contribution_BRAF_mut) + max(contribution_CRAF) + max(contribution_ARAF) + max(contribution_BRAF_WT)));
fprintf('  ARAF* contribution to MEK-P: %.1f%%\n', ...
        100 * max(contribution_ARAF) / (max(contribution_BRAF_mut) + max(contribution_CRAF) + max(contribution_ARAF) + max(contribution_BRAF_WT)));

fprintf('\nCROSS-TALK AND FEEDBACK STRENGTHS:\n');
fprintf('  Vemurafenib inhibition (Hill): IC50 = %.4f (normalized [0,1]), Hill n = %.2f\n', IC50_vem, Hill_n);
fprintf('  Paradoxical CRAF activation: gamma = %.2f\n', gamma);
fprintf('  Paradoxical ARAF activation: delta = %.2f\n', delta);
fprintf('  RAS → PI3K:           k_PI3K_RAS = %.2f\n', k_PI3K_RAS);
fprintf('  ERK → TSC2:           alpha3 = %.2f\n', alpha3);
fprintf('  AKT → CRAF:           alpha2 = %.2f\n', alpha2);
fprintf('  pS6K → PI3K:         alpha1 = %.2f\n', alpha1);
fprintf('  DUSP transcription:   Vtx_DUSP = %.2f, K_DUSP = %.2f, n_DUSP = %.2f\n', Vtx_DUSP, K_DUSP, n_DUSP);
fprintf('  SPRY transcription:   Vtx_SPRY = %.2f, K_SPRY = %.2f, n_SPRY = %.2f\n', Vtx_SPRY, K_SPRY, n_SPRY);
fprintf('  DUSP catalytic:       Vmax_DUSP = %.2f, Km_DUSP = %.2f\n', Vmax_DUSP, Km_DUSP);
fprintf('  SPRY:Grb2 binding:    k_bind = %.4f, k_unbind = %.2f\n', k_bind_SPRY_Grb2, k_unbind_SPRY_Grb2);

fprintf('\n═══════════════════════════════════════════════════════════════════════════\n');
fprintf('   SIMULATION COMPLETED SUCCESSFULLY!\n');
fprintf('═══════════════════════════════════════════════════════════════════════════\n\n');

%% ============================================================================
% SAVE OUTPUTS
% ============================================================================

save('vemurafenib_craf_araf_output.mat', 't', 'ERK_P', 'pAKT_full', 'pS6K', 'p4EBP1', ...
     'BRAF_mut_star', 'CRAF_star', 'ARAF_star', 'BRAF_WT_star', 'MEK_P', 'RAS_GTP', ...
     'PIP3', 'Rheb_GTP', 'mTORC1_active', 'pRTK_signal', 'vemurafenib_signal', ...
     'BRAF_WT_CRAF_dimer', 'BRAF_WT_ARAF_dimer', ...
     'mDUSP', 'DUSP', 'mSPRY', 'SPRY', 'SPRY_Grb2');
fprintf('Saved outputs to vemurafenib_craf_araf_output.mat\n');

%% ============================================================================
% ODE FUNCTION
% ============================================================================

function dydt = vemurafenib_craf_araf_odes(t, y, p)
    % Integrated ODE system for MAPK and PI3K/AKT/mTOR pathways with vemurafenib
    % Includes BRAF^V600E, CRAF, ARAF, and BRAF-WT with paradoxical activation
    % Includes DUSP and SPRY feedback mechanisms
    % y = [MAPK: Shc, pEGFR_Shc, pShc, Grb2, pShc_Grb2, SOS, pShc_Grb2_SOS, SOS_star,
    %      RAS_GDP, RAS_GTP, BRAF_mut, BRAF_mut_star, CRAF, CRAF_star, ARAF, ARAF_star,
    %      BRAF_WT, BRAF_WT_star, BRAF_WT_CRAF_dimer, BRAF_WT_ARAF_dimer,
    %      MEK, MEK_P, ERK, ERK_P;
    %      PI3K: PI3K, PI3K_active, PIP2, PIP3, AKT, pAKT_Thr308, pAKT_Ser473, pAKT_full,
    %      TSC2_active, Rheb_GTP, mTORC1, mTORC1_active, S6K, pS6K, 4EBP1, p4EBP1;
    %      DUSP/SPRY: mDUSP, DUSP, mSPRY, SPRY, SPRY_Grb2]
    
    % Extract MAPK pathway species
    Shc = y(1);
    pEGFR_Shc = y(2);
    pShc = y(3);
    Grb2 = y(4);
    pShc_Grb2 = y(5);
    SOS = y(6);
    pShc_Grb2_SOS = y(7);
    SOS_star = y(8);
    RAS_GDP = y(9);
    RAS_GTP = y(10);
    BRAF_mut = y(11);
    BRAF_mut_star = y(12);
    CRAF = y(13);
    CRAF_star = y(14);
    ARAF = y(15);
    ARAF_star = y(16);
    BRAF_WT = y(17);
    BRAF_WT_star = y(18);
    BRAF_WT_CRAF_dimer = y(19);
    BRAF_WT_ARAF_dimer = y(20);
    MEK = y(21);
    MEK_P = y(22);
    ERK = y(23);
    ERK_P = y(24);
    
    % Extract PI3K/AKT/mTOR pathway species
    PI3K = y(25);
    PI3K_active = y(26);
    PIP2 = y(27);
    PIP3 = y(28);
    AKT = y(29);
    pAKT_Thr308 = y(30);
    pAKT_Ser473 = y(31);
    pAKT_full = y(32);
    TSC2_active = y(33);
    Rheb_GTP = y(34);
    mTORC1 = y(35);
    mTORC1_active = y(36);
    S6K = y(37);
    pS6K = y(38);
    EBP1 = y(39);
    p4EBP1 = y(40);
    
    % Extract DUSP and SPRY feedback species
    mDUSP = y(41);
    DUSP = y(42);
    mSPRY = y(43);
    SPRY = y(44);
    SPRY_Grb2 = y(45);
    
    % Get input signals at current time
    pRTK = p.pRTK_func(t);
    vemurafenib = p.vemurafenib_func(t);
    
    % ========================================================================
    % CALCULATE FEEDBACK-MODULATED AND CROSS-TALK RATE CONSTANTS
    % ========================================================================
    
    % MAPK pathway feedbacks (ERK-P) - feedback is relieved when ERK-P drops
    k_on_Shc_eff = p.k_on_Shc / (1 + p.alpha_ERK_Shc * ERK_P);
    k_cat_SOS_eff = p.k_cat_SOS / (1 + p.alpha_ERK_SOS * ERK_P);
    k_GTPase_eff = p.k_GTPase + p.alpha_ERK_GTPase * ERK_P;
    
    % VEMURAFENIB INHIBITION: BRAF^V600E activity reduced using Hill equation
    IC50_n = p.IC50_vem^p.Hill_n;
    vem_n = vemurafenib^p.Hill_n;
    k_BRAF_mut_eff = p.k_BRAF_mut * IC50_n / (IC50_n + vem_n);
   
    
    % CRAF activation: base rate + paradoxical activation - AKT inhibition
    % Paradoxical activation: vemurafenib + BRAF-WT* dimer → CRAF* activation
    k_CRAF_eff = p.k_CRAF_base * (1 + p.gamma * vemurafenib * BRAF_WT_star) / (1 + p.alpha2 * pAKT_full);
    
    % ARAF activation: base rate + minor paradoxical activation
    k_ARAF_eff = p.k_ARAF_base * (1 + p.delta * vemurafenib * BRAF_WT_star);
    
    % BRAF-WT activation (RAS-dependent)
    k_BRAF_WT_eff = p.k_BRAF_WT;
    
    % PI3K pathway feedback (pS6K)
    k_PI3K_eff = p.k_PI3K / (1 + p.alpha1 * pS6K);
    
    % Cross-talk: ERK → TSC2 (ERK-P enhances TSC2 inhibition)
    k_TSC2_ERK = p.k_TSC2_base + p.alpha3 * ERK_P;
    
    % ========================================================================
    % DUSP AND SPRY TRANSCRIPTIONAL INDUCTION BY pERK
    % ========================================================================
    
    % DUSP transcription (Hill-type induction by pERK)
    ERK_P_n_DUSP = ERK_P^p.n_DUSP;
    K_DUSP_n = p.K_DUSP^p.n_DUSP;
    v_tx_DUSP = p.Vtx_DUSP * (ERK_P_n_DUSP / (K_DUSP_n + ERK_P_n_DUSP));
    
    % SPRY transcription (Hill-type induction by pERK)
    ERK_P_n_SPRY = ERK_P^p.n_SPRY;
    K_SPRY_n = p.K_SPRY^p.n_SPRY;
    v_tx_SPRY = p.Vtx_SPRY * (ERK_P_n_SPRY / (K_SPRY_n + ERK_P_n_SPRY));
    
    % ========================================================================
    % DUSP AND SPRY FEEDBACK ON MAPK PATHWAY
    % ========================================================================
    
    % DUSP catalytic dephosphorylation of pERK (Michaelis-Menten)
    % Use smooth transition: when Km_DUSP > 0 use Michaelis-Menten, else use first-order
    % Heaviside-like function: H(x) = 0.5 * (1 + sign(x)) for x > 0, else 0
    H_Km = 0.5 * (1 + sign(p.Km_DUSP - eps));  % 1 if Km_DUSP > eps, else 0
    Km_DUSP_eff = max(eps, p.Km_DUSP);
    k_deph_ERK_DUSP = H_Km * (p.Vmax_DUSP * DUSP / (Km_DUSP_eff + ERK_P)) + ...
                      (1 - H_Km) * (p.k_cat_DUSP * DUSP);  % First-order when Km <= eps
    k_ERK_deact_eff = p.k_ERK_deact + k_deph_ERK_DUSP;
    
    % SPRY action: sequestration of Grb2 (reduce free Grb2 available for SOS)
    % Calculate free Grb2 (total Grb2 pool = 1.0 normalized, minus bound forms)
    Grb2_total = 1.0;  % Normalized total Grb2 pool
    Grb2_free = max(0, Grb2_total - pShc_Grb2 - SPRY_Grb2);
    
    % SPRY feedback on SOS activation (via Grb2 sequestration)
    % Effective SOS activation rate depends on free Grb2
    % Use max(eps, Grb2_total) to avoid division by zero
    Grb2_total_eff = max(eps, Grb2_total);
    Grb2_ratio = max(0, Grb2_free / Grb2_total_eff);
    k_on2_eff = p.k_on2 * (Grb2_ratio^p.h_coop);  % Cooperativity effect
    
    % ========================================================================
    % MAPK PATHWAY REACTIONS
    % ========================================================================
    
    % Shc–Grb2–SOS module
    r1_forward = k_on_Shc_eff * pRTK * Shc;
    r1_reverse = p.k_off_Shc * pEGFR_Shc;
    r1_cat = p.k_cat1 * pEGFR_Shc;
    r2_forward = k_on2_eff * pShc * Grb2_free;  % Use free Grb2 (SPRY feedback)
    r2_reverse = p.k_off2 * pShc_Grb2;
    r3_forward = p.k_on3 * pShc_Grb2 * SOS;
    r3_reverse = p.k_off3 * pShc_Grb2_SOS;
    r3_cat = k_cat_SOS_eff * pShc_Grb2_SOS;
    r4 = p.k_deg4 * SOS_star;
    
    % RAS module
    r5 = p.k_SOS * SOS_star * RAS_GDP;
    r6 = k_GTPase_eff * RAS_GTP;
    
    % BRAF^V600E constitutive activation (inhibited by vemurafenib)
    r7 = k_BRAF_mut_eff * BRAF_mut;
    r8 = p.k_BRAF_mut_deact * BRAF_mut_star;
    
    % CRAF activation (RAS-dependent, AKT-inhibited, paradoxically activated)
    r9 = k_CRAF_eff * RAS_GTP * CRAF;
    r10 = p.k_CRAF_deact * CRAF_star;
    
    % ARAF activation (RAS-dependent, weak, minor paradoxical activation)
    r11 = k_ARAF_eff * RAS_GTP * ARAF;
    r12 = p.k_ARAF_deact * ARAF_star;
    
    % BRAF-WT activation (RAS-dependent)
    r13 = k_BRAF_WT_eff * RAS_GTP * BRAF_WT;
    r14 = p.k_BRAF_WT_deact * BRAF_WT_star;
    
    % Dimer formation for paradoxical activation
    r15 = p.k_dimer * BRAF_WT_star * CRAF;  % BRAF-WT:CRAF dimer
    r16 = p.k_dimer * BRAF_WT_star * ARAF;  % BRAF-WT:ARAF dimer
    r17 = 0.1 * BRAF_WT_CRAF_dimer;  % Dimer dissociation (simplified)
    r18 = 0.1 * BRAF_WT_ARAF_dimer;  % Dimer dissociation (simplified)
    
    % Paradoxical activation from dimers (when vemurafenib is present)
    % Use max(0, vemurafenib) to ensure non-negative, multiplied by step function
    vemurafenib_eff = max(0, vemurafenib);
    r19 = p.gamma * vemurafenib_eff * BRAF_WT_CRAF_dimer;  % CRAF* from dimer
    r20 = p.delta * vemurafenib_eff * BRAF_WT_ARAF_dimer;  % ARAF* from dimer
    
    % MEK phosphorylation by all RAF isoforms
    r21 = p.k_MEK * BRAF_mut_star * MEK;  % BRAF^V600E contribution
    r22 = p.k_MEK * CRAF_star * MEK;  % CRAF contribution
    r23 = p.k_MEK * ARAF_star * MEK;  % ARAF contribution (weak)
    r24 = p.k_MEK * BRAF_WT_star * MEK;  % BRAF-WT contribution
    r25 = p.k_MEK_deact * MEK_P;
    
    % ERK phosphorylation
    r26 = p.k_ERK * MEK_P * ERK;
    r27 = k_ERK_deact_eff * ERK_P;  % Enhanced by DUSP feedback
    
    % ========================================================================
    % DUSP AND SPRY REACTIONS
    % ========================================================================
    
    % DUSP mRNA dynamics
    r_DUSP_tx = v_tx_DUSP;  % Transcription (Hill-type by pERK)
    r_DUSP_mRNA_deg = p.k_deg_mDUSP * mDUSP;  % mRNA degradation
    
    % DUSP protein dynamics
    r_DUSP_tl = p.k_tl_DUSP * mDUSP;  % Translation
    r_DUSP_deg = p.k_deg_DUSP * DUSP;  % Protein degradation
    
    % SPRY mRNA dynamics
    r_SPRY_tx = v_tx_SPRY;  % Transcription (Hill-type by pERK)
    r_SPRY_mRNA_deg = p.k_deg_mSPRY * mSPRY;  % mRNA degradation
    
    % SPRY protein dynamics and Grb2 binding
    r_SPRY_tl = p.k_tl_SPRY * mSPRY;  % Translation
    r_SPRY_deg = p.k_deg_SPRY * SPRY;  % Protein degradation
    r_SPRY_bind = p.k_bind_SPRY_Grb2 * SPRY * Grb2_free;  % SPRY:Grb2 binding
    r_SPRY_unbind = p.k_unbind_SPRY_Grb2 * SPRY_Grb2;  % SPRY:Grb2 unbinding
    r_SPRY_Grb2_deg = p.k_deg_complex * SPRY_Grb2;  % Complex degradation
    
    % ========================================================================
    % PI3K/AKT/mTOR PATHWAY REACTIONS
    % ========================================================================
    
    % PI3K module
    r28 = k_PI3K_eff * pRTK * PI3K + p.k_PI3K_RAS * RAS_GTP * PI3K;  % Cross-talk: RAS → PI3K
    r29 = p.k_PI3K_deact * PI3K_active;
    r30 = p.k_PIP2_to_PIP3 * PI3K_active * PIP2;
    r31 = p.k_PTEN * PIP3;
    
    % AKT module
    r32 = p.k_AKT_recruit * PIP3 * AKT;
    r33 = p.k_AKT_Thr308 * PIP3 * AKT;
    r34 = p.k_AKT_Ser473 * pAKT_Thr308;
    r35 = p.k_AKT_Ser473 * PIP3 * AKT;
    r36 = p.k_AKT_Thr308 * pAKT_Ser473;
    r37_Thr308 = p.k_AKT_deact * pAKT_Thr308;
    r37_Ser473 = p.k_AKT_deact * pAKT_Ser473;
    r37_full = p.k_AKT_deact * pAKT_full;
    
    % TSC2/Rheb module
    r38 = (p.k_TSC2_inh * pAKT_full + k_TSC2_ERK * ERK_P) * TSC2_active;  % Cross-talk: ERK → TSC2
    Rheb_GDP = 1.0 - Rheb_GTP;  % Normalized: total Rheb pool = 1.0
    r39 = p.k_Rheb_act * Rheb_GDP / (1 + TSC2_active);
    r40 = p.k_Rheb_GTPase * Rheb_GTP * (1 + TSC2_active);  % Updated for normalized values
    
    % mTORC1 module
    r41 = p.k_mTORC1 * Rheb_GTP * mTORC1;
    r42 = p.k_mTORC1_deact * mTORC1_active;
    
    % S6K and 4EBP1 module
    r43 = p.k_S6K * mTORC1_active * S6K;
    r44 = p.k_S6K_deact * pS6K;
    r45 = p.k_4EBP1 * mTORC1_active * EBP1;
    r46 = p.k_4EBP1_deact * p4EBP1;
    
    % ========================================================================
    % ODEs FOR MAPK PATHWAY
    % ========================================================================
    
    dShc_dt = -r1_forward + r1_reverse;
    dpEGFR_Shc_dt = r1_forward - r1_reverse - r1_cat;
    dpShc_dt = r1_cat - r2_forward + r2_reverse;
    dGrb2_dt = -r2_forward + r2_reverse - r_SPRY_bind + r_SPRY_unbind;  % SPRY binding feedback
    dpShc_Grb2_dt = r2_forward - r2_reverse - r3_forward + r3_reverse + r3_cat;
    dSOS_dt = -r3_forward + r3_reverse + r4;
    dpShc_Grb2_SOS_dt = r3_forward - r3_reverse - r3_cat;
    dSOS_star_dt = r3_cat - r4;
    dRAS_GDP_dt = -r5 + r6;
    dRAS_GTP_dt = r5 - r6 - r9 - r11 - r13;  % Used by CRAF, ARAF, and BRAF-WT
    dBRAF_mut_dt = -r7 + r8;
    dBRAF_mut_star_dt = r7 - r8;
    dCRAF_dt = -r9 - r15 + r10 + r17;
    dCRAF_star_dt = r9 + r19 - r10;  % Includes paradoxical activation
    dARAF_dt = -r11 - r16 + r12 + r18;
    dARAF_star_dt = r11 + r20 - r12;  % Includes paradoxical activation
    dBRAF_WT_dt = -r13 + r14 - r15 - r16 + r17 + r18;
    dBRAF_WT_star_dt = r13 - r14 - r15 - r16;  % Used for dimer formation
    dBRAF_WT_CRAF_dimer_dt = r15 - r17 - r19;
    dBRAF_WT_ARAF_dimer_dt = r16 - r18 - r20;
    dMEK_dt = -r21 - r22 - r23 - r24 + r25;
    dMEK_P_dt = r21 + r22 + r23 + r24 - r25;
    dERK_dt = -r26 + r27;
    dERK_P_dt = r26 - r27;
    
    % ========================================================================
    % ODEs FOR PI3K/AKT/mTOR PATHWAY
    % ========================================================================
    
    dPI3K_dt = -r28 + r29;
    dPI3K_active_dt = r28 - r29;
    dPIP2_dt = -r30 + r31;
    dPIP3_dt = r30 - r31;
    dAKT_dt = -r32 - r33 - r35 + r37_Thr308 + r37_Ser473 + r37_full;
    dpAKT_Thr308_dt = r32 + r33 - r34 - r37_Thr308;
    dpAKT_Ser473_dt = r35 - r36 - r37_Ser473;
    dpAKT_full_dt = r34 + r36 - r37_full;
    dTSC2_active_dt = -r38;
    dRheb_GTP_dt = r39 - r40;
    dmTORC1_dt = -r41 + r42;
    dmTORC1_active_dt = r41 - r42;
    dS6K_dt = -r43 + r44;
    dpS6K_dt = r43 - r44;
    dEBP1_dt = -r45 + r46;
    dp4EBP1_dt = r45 - r46;
    
    % ========================================================================
    % ODEs FOR DUSP AND SPRY FEEDBACK
    % ========================================================================
    
    % DUSP mRNA
    dmDUSP_dt = r_DUSP_tx - r_DUSP_mRNA_deg;
    
    % DUSP protein
    dDUSP_dt = r_DUSP_tl - r_DUSP_deg;
    
    % SPRY mRNA
    dmSPRY_dt = r_SPRY_tx - r_SPRY_mRNA_deg;
    
    % SPRY protein
    dSPRY_dt = r_SPRY_tl - r_SPRY_deg - r_SPRY_bind + r_SPRY_unbind;
    
    % SPRY:Grb2 complex
    dSPRY_Grb2_dt = r_SPRY_bind - r_SPRY_unbind - r_SPRY_Grb2_deg;
    
    % Return all derivatives
    dydt = [dShc_dt; dpEGFR_Shc_dt; dpShc_dt; dGrb2_dt; dpShc_Grb2_dt; ...
            dSOS_dt; dpShc_Grb2_SOS_dt; dSOS_star_dt; ...
            dRAS_GDP_dt; dRAS_GTP_dt; ...
            dBRAF_mut_dt; dBRAF_mut_star_dt; ...
            dCRAF_dt; dCRAF_star_dt; dARAF_dt; dARAF_star_dt; ...
            dBRAF_WT_dt; dBRAF_WT_star_dt; ...
            dBRAF_WT_CRAF_dimer_dt; dBRAF_WT_ARAF_dimer_dt; ...
            dMEK_dt; dMEK_P_dt; dERK_dt; dERK_P_dt; ...
            dPI3K_dt; dPI3K_active_dt; dPIP2_dt; dPIP3_dt; ...
            dAKT_dt; dpAKT_Thr308_dt; dpAKT_Ser473_dt; dpAKT_full_dt; ...
            dTSC2_active_dt; dRheb_GTP_dt; dmTORC1_dt; dmTORC1_active_dt; ...
            dS6K_dt; dpS6K_dt; dEBP1_dt; dp4EBP1_dt; ...
            dmDUSP_dt; dDUSP_dt; dmSPRY_dt; dSPRY_dt; dSPRY_Grb2_dt];
end

