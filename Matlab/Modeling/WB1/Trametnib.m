% =============================================================================
% MAPK/PI3K PATHWAY MODEL WITH TRAMETINIB
% =============================================================================
% Version: 2.0 | Author: Anandivada Lab | Last Updated: 2026-01-09
%
% DESCRIPTION:
%   Comprehensive ODE model of MAPK and PI3K/AKT/mTOR signaling pathways
%   with Trametinib treatment (MEK Inhibitor).
%   Consolidated with Vemurafenib.m structure for consistancy.
%
% KEY FEATURES:
%   • EGFR/Her2/Her3/IGFR/PDGFR → RAS → RAF → MEK → ERK cascade
%   • PI3K → AKT → mTOR → 4EBP1/S6K pathway
%   • DUSP and SPRY negative feedback loops
%   • Trametinib inhibition of MEK (with displacement mechanism)
%   • Paradoxical activation logic included (inactive when Vemurafenib = 0)
%
% MODEL SIZE:
%   • 68 Species (state variables)
%   • 68 Parameters (rate constants)
%   • Time units: seconds (experimental data in hours, converted)
%
% =============================================================================

clear all;
close all;
clc;

fprintf('═══════════════════════════════════════════════════════════════════════════\n');
fprintf('   MAPK/PI3K PATHWAY MODEL WITH TRAMETINIB\n');
fprintf('   Version 2.0 - Unified Structure\n');
fprintf('═══════════════════════════════════════════════════════════════════════════\n\n');

%% ============================================================================
%  SECTION 1: MODEL PARAMETERS
%  ============================================================================
%  Organized by pathway module for clarity. All rates in min^-1.
% =============================================================================

fprintf('Setting up model parameters...\n');

% ----------------------------------------------------------------------------
% 1.1 RTK/EGFR Module Parameters
% ----------------------------------------------------------------------------
params.RTK.k_on = 10e-5;          % EGFR binding rate (min^-1)
params.RTK.k_off = 10e-6;         % EGFR dissociation rate (min^-1)
params.RTK.k_cat = 10e-3;         % EGFR phosphorylation rate (min^-1)
params.RTK.k_degrad = 10e-7;      % EGFR degradation rate (min^-1)
params.RTK.k_ERK_inhib = 20e-6;   % ERK feedback on EGFR (min^-1)
params.RTK.k_Her2_act = 10e-5;    % Her2 activation rate (min^-1)
params.RTK.k_Her3_act = 10e-5;    % Her3 activation rate (min^-1)
params.RTK.k_PDGFR_act = 10e-5;   % PDGFR activation rate (min^-1)

% ----------------------------------------------------------------------------
% 1.2 Shc/Grb2/SOS Module Parameters
% ----------------------------------------------------------------------------
params.SOS.k_Shc_dephos = 10e-5;      % Shc dephosphorylation (min^-1)
params.SOS.k_ptp = 7e-7;              % PTP activity (min^-1)
params.SOS.k_Grb2_bind = 36e-6;      % Grb2 binding to pShc (min^-1)
params.SOS.k_Sprty_inhib = 10e-5;    % SPRY inhibition of Grb2 (min^-1)
params.SOS.k_SOS_bind = 10e-5;       % SOS binding to Grb2 (min^-1)
params.SOS.k_ERK_phos = 17e-6;       % ERK phosphorylation of SOS (min^-1)

% ----------------------------------------------------------------------------
% 1.3 RAS Module Parameters
% ----------------------------------------------------------------------------
params.RAS.k_act = 10e-5;            % RAS activation by SOS* (min^-1)

% ----------------------------------------------------------------------------
% 1.4 RAF Module Parameters
% ----------------------------------------------------------------------------
params.RAF.k_CRAF_act = 10e-4;       % CRAF activation by RAS-GTP (min^-1)
params.RAF.k_ERK_phos_CRAF = 15.7342e-6;  % ERK phosphorylation of pCRAF (min^-1)
params.RAF.k_CRAF_degrad = 1.7342e-4;    % pCRAF degradation (min^-1)

% ----------------------------------------------------------------------------
% 1.5 MEK Module Parameters
% ----------------------------------------------------------------------------
params.MEK.k_phos = 7e-6;            % MEK phosphorylation by RAF (min^-1)
params.MEK.k_ERK_phos = 20.7342e-5;  % ERK phosphorylation of pMEK (min^-1)
params.MEK.k_degrad = 1.7342e-8;     % pMEK degradation (min^-1)
params.MEK.k_BRAF_route = 8e-6;      % BRAF^P → MEK (min^-1)
params.MEK.k_CRAF_route = 8e-6;      % CRAF* → MEK (min^-1)

% ----------------------------------------------------------------------------
% 1.6 ERK Module Parameters
% ----------------------------------------------------------------------------
params.ERK.k_phos = 20e-5;           % ERK phosphorylation by MEK (min^-1)
params.ERK.k_DUSP_inhib = 2.7342e-4; % DUSP inhibition of pERK (min^-1)
params.ERK.k_degrad = 1.7342e-7;     % pERK degradation (min^-1)
params.ERK.k_DUSP_dephos = 1e-5;     % DUSP dephosphorylation of pERK (min^-1)

% ----------------------------------------------------------------------------
% 1.7 DUSP Feedback Module Parameters
% ----------------------------------------------------------------------------
params.DUSP.k_max_tx = 1.11e-8;      % Maximum DUSP transcription rate (min^-1)
params.DUSP.k_DUSP_stop = 1.7342e-04; % DUSP stop signal (min^-1)
params.DUSP.k_DUSP_deg = 1e-6;       % DUSP degradation by ERK (min^-1)

% ----------------------------------------------------------------------------
% 1.8 SPRY Feedback Module Parameters
% ----------------------------------------------------------------------------
params.SPRY.k_max_tx = 1.7e-06;      % Maximum SPRY transcription rate (min^-1)
params.SPRY.k_come_down = 5.5000e-5; % SPRY degradation (min^-1)
params.SPRY.k_mRNA_decay = 8e-5;     % SPRY mRNA decay (min^-1)

% ----------------------------------------------------------------------------
% 1.9 BRAF Module Parameters
% ----------------------------------------------------------------------------
params.BRAF.k_inhib = 10e-2;         % BRAF inhibition rate (min^-1)

% ----------------------------------------------------------------------------
% 1.10 KSR Scaffold Module Parameters
% ----------------------------------------------------------------------------
params.KSR.k_phos = 5e-6;            % KSR phosphorylation by RAF (min^-1)
params.KSR.k_dephos = 5e-6;          % KSR dephosphorylation (min^-1)
params.KSR.k_MEK_route = 3e-6;      % pKSR → MEK phosphorylation (min^-1)

% ----------------------------------------------------------------------------
% 1.11 Trametinib (MEK Inhibitor) Parameters
% ----------------------------------------------------------------------------
params.Trametinib.conc = 1e-6;       % 1 uM Trametinib
params.Trametinib.Ki_RAF = 1e-9;     % Effective Ki (Molar) - High potency
params.Trametinib.Ki_KSR = 1e-9;     % Ki for KSR-mediated phosphorylation
params.Trametinib.Hill_n = 2.0;      % Hill coefficient
params.Trametinib.K_displace = 0.05; % Sensitivity to Upstream Load (Displacement)

% ----------------------------------------------------------------------------
% 1.12 Vemurafenib and Paradoxical Activation Parameters
% ----------------------------------------------------------------------------
params.Vemurafenib.conc = 0.0;           % Vemurafenib concentration (STRICTLY 0)
params.Vemurafenib.IC50 = 0.4;           % IC50 for BRAF^V600E inhibition
params.Vemurafenib.Hill_n = 1.5;         % Hill coefficient
params.Paradox.k_dimer_form = 0.0;      % Paradoxical dimers disabled (Vem=0)
params.Paradox.k_dimer_dissoc = 1e-1;    % Quick dissociation of any residual dimers
params.Paradox.gamma = 0.0;              % Paradoxical activation strength disabled

% ----------------------------------------------------------------------------
% 1.13 PI3K/AKT/mTOR Module Parameters
% ----------------------------------------------------------------------------
params.PI3K.k_p85_bind_EGFR = 10e-4;   % p85 binding to pEGFR
params.PI3K.k_p85_bind_Her2 = 10e-4;   % p85 binding to pHer2
params.PI3K.k_p85_bind_Her3 = 10e-4;   % p85 binding to pHer3
params.PI3K.k_p85_bind_IGFR = 10e-4;   % p85 binding to pIGFR
params.PI3K.k_p85_bind_PDGFR = 10e-4;  % p85 binding to pPDGFR
params.PI3K.k_p85_unbind = 10e-5;      % p85 unbinding
params.PI3K.k_PI3K_recruit = 10e-4;    % PI3K recruitment
params.PI3K.k_MTOR_feedback = 10e-4;   % mTOR feedback on PI3K
params.PI3K.k_PIP2_to_PIP3 = 10e-4;    % PIP2 to PIP3 conversion
params.PI3K.k_PTEN = 10e-5;             % PTEN
params.AKT.k_act = 10e-5;              % AKT activation
params.AKT.k_degrad = 10e-7;           % AKT degradation
params.mTOR.kb1 = 10e-8;               % mTOR activation
params.mTOR.k43b1 = 10e-3;             % mTOR activation
params.mTOR.k_4EBP1 = 10e-5;           % 4EBP1 phosphorylation
params.mTOR.k_4EBP1_dephos = 10e-5;   % p4EBP1 dephosphorylation
params.mTOR.k_S6K = 10e-5;               % S6K phosphorylation
params.mTOR.k_S6K_dephos = 10e-5;        % pS6K dephosphorylation

% ----------------------------------------------------------------------------
% 1.14 General Degradation Parameters
% ----------------------------------------------------------------------------
params.Degrad.general = 10e-7;      % General degradation rate

% Convert parameter structure to vector for optimization
% Order matches Vemurafenib.m params (1-67) + K_displace (68)
p = params;
params_vector = [
    p.RTK.k_on, p.RTK.k_off, p.RTK.k_cat, ...
    p.RAF.k_CRAF_act, p.MEK.k_phos, p.ERK.k_phos, ...
    p.RTK.k_degrad, p.RTK.k_ERK_inhib, p.SOS.k_Shc_dephos, p.SOS.k_ptp, p.SOS.k_Grb2_bind, ...
    p.SOS.k_Sprty_inhib, p.SOS.k_SOS_bind, p.SOS.k_ERK_phos, ...
    p.RAF.k_ERK_phos_CRAF, p.RAF.k_CRAF_degrad, p.MEK.k_ERK_phos, p.MEK.k_degrad, ...
    p.ERK.k_DUSP_inhib, p.ERK.k_degrad, p.BRAF.k_inhib, p.DUSP.k_DUSP_stop, p.DUSP.k_max_tx, ...
    p.SPRY.k_max_tx, p.SPRY.k_come_down, p.Degrad.general, p.SPRY.k_mRNA_decay, ...
    p.DUSP.k_max_tx, p.SPRY.k_max_tx, ...
    p.ERK.k_DUSP_dephos, p.DUSP.k_DUSP_deg, ...
    p.RTK.k_Her2_act, p.RTK.k_Her3_act, ...
    p.PI3K.k_p85_bind_EGFR, p.PI3K.k_p85_bind_Her2, p.PI3K.k_p85_bind_Her3, p.PI3K.k_p85_bind_IGFR, ...
    p.PI3K.k_p85_unbind, p.PI3K.k_PI3K_recruit, ...
    p.PI3K.k_MTOR_feedback, ...
    p.PI3K.k_PIP2_to_PIP3, p.PI3K.k_PTEN, ...
    p.AKT.k_act, p.AKT.k_degrad, ...
    p.mTOR.kb1, p.mTOR.k43b1, p.mTOR.k_4EBP1, p.mTOR.k_4EBP1_dephos, ...
    p.KSR.k_phos, p.KSR.k_dephos, ...
    p.MEK.k_BRAF_route, p.MEK.k_CRAF_route, p.KSR.k_MEK_route, ...
    p.Trametinib.conc, p.Trametinib.Ki_RAF, p.Trametinib.Ki_KSR, p.Trametinib.Hill_n, ...
    p.Vemurafenib.conc, p.Paradox.k_dimer_form, p.Paradox.k_dimer_dissoc, ...
    p.Paradox.gamma, p.Vemurafenib.IC50, p.Vemurafenib.Hill_n, ...
    p.RTK.k_PDGFR_act, p.PI3K.k_p85_bind_PDGFR, ...
    p.mTOR.k_S6K, p.mTOR.k_S6K_dephos, ...
    p.Trametinib.K_displace  % Param 68: Added for Trametinib
];

% Verify parameter vector length
if length(params_vector) ~= 68
    error('Parameter vector has %d elements, expected 68!', length(params_vector));
end

fprintf('Parameters organized by pathway modules. Total: %d parameters.\n\n', length(params_vector));

%% ============================================================================
%  SECTION 2: INITIAL CONDITIONS
%  ============================================================================

fprintf('Setting initial conditions...\n');

% RTK/EGFR Module (INITIAL BASAL LEVELS)
IC.EGFR = [1.0, 0.35, 0.35];    IC.Her2 = [1.0, 0.245, 0.245];  IC.Her3 = [1.0, 0.203, 0.203];
IC.SHC = [1.0, 0.0, 1.0];       IC.Grb2_SOS = [0.0, 0.0];

% RAS Module
IC.HRAS = [0.0, 0.0];           IC.NRAS = [0.0, 0.0];           IC.KRAS = [1.0, 0.0, 1.0];

% RAF, MEK, ERK
IC.CRAF = [0.8, 0.366];         IC.BRAF = [1.0, 1.0];
IC.MEK = [1.0, 1.759];          IC.ERK = [1.0, 2.903];

% Feedback & Degradation
IC.DUSP = [1.0, 2.677];         IC.SPRY = [1.0, 1.0];
IC.pERK_degrad = [1.0];         IC.pMEK_degrad = [1.0];         IC.pCRAF_degrad = [1.0]; IC.DUSP_stop = [1.0];

% PI3K/AKT/mTOR
IC.IGFR = [1.0, 0.0, 0.0];      IC.IRS = [1.0, 0.0];
IC.p85 = [1.0];
IC.p85_EGFR = [0.1]; IC.p85_Her2 = [0.1]; IC.p85_Her3 = [0.1]; IC.p85_IGFR = [0.1];
IC.PI3K = [1.0, 0.2];           IC.PIP = [1.0, 0.1];
IC.AKT = [1.0, 0.513];          IC.FOXO = [0.0];
IC.mTORC = [1.0, 0.5];          IC.frebp1 = [1.0, 0.5, 1.002];

% KSR & Paradox
IC.KSR = [1.0, 0.0];            IC.BRAF_CRAF_dimer = [0.0];

% New Species (PDGFR, S6K)
IC.PDGFR = [1.0, 0.474, 0.474]; IC.S6K = [1.0, 1.432];    IC.p85_PDGFR = [0.1];

% Construct State Vector (68 Species)
y0 = [
    IC.EGFR, IC.Her2, IC.Her3, IC.SHC, IC.Grb2_SOS, ...
    IC.HRAS, IC.NRAS, IC.KRAS, ...
    IC.CRAF, IC.BRAF, ...
    IC.MEK, IC.ERK, ...
    IC.DUSP, IC.SPRY, ...
    IC.pERK_degrad, IC.pMEK_degrad, IC.pCRAF_degrad, IC.DUSP_stop, ...
    IC.IGFR, IC.IRS, ...
    IC.p85, IC.p85_EGFR, IC.p85_Her2, IC.p85_Her3, IC.p85_IGFR, ...
    IC.PI3K, IC.PIP, IC.AKT, IC.FOXO, IC.mTORC, IC.frebp1, ...
    IC.KSR, IC.BRAF_CRAF_dimer, ...
    IC.PDGFR, IC.S6K, IC.p85_PDGFR
];

fprintf('Initial conditions set. Total species: %d\n\n', length(y0));

%% ============================================================================
%  SECTION 3: EXPERIMENTAL DATA
%  ============================================================================
%  Data for Trametinib Treatment
% =============================================================================

fprintf('Loading experimental data...\n');
timeStamps_hours = [0, 1, 4, 8, 24, 48];
timeStamps_seconds = timeStamps_hours * 3600;

% Trametinib Data from Original File
% Note: Using same field names as Vemurafenib.m format for consistency in logic
expData_raw.panRAS = [0.729420212,0.767968882,0.807297102,0.834879564,1.269288689,1.498069973];
expData_raw.pMEK = [2.024063616,0.501888368,0.442535778,0.508820348,0.671938823,0.983710872];
expData_raw.pERK = [3.487158237,0.114217722,0.022162422,0.03726381,0.40696761,1.204092044];
expData_raw.DUSP = [3.054033606,3.118384551,1.178142126,0.367382855,0.031574513,0.092390719];
expData_raw.pEGFR = [0.35120201,0.545885712,0.642780201,0.359436744,0.009710876,0.01311083];
expData_raw.pCRAF = [0.233819043,0.451384259,0.355935204,0.786954922,0.946694075,0.26220219];
expData_raw.pAKT = [0.549427631,0.642783939,1.046735362,0.944355203,0.479593107,0.310063914];
expData_raw.p4ebp1 = [0.984913436,1.263725259,1.355812598,1.380725265,0.941991484,0.281153536];
expData_raw.pS6k = [1.432459522,1.520433646,1.542177411,1.248505245,0.109963216,0.013374136];
expData_raw.her2 = [0.245236744,0.177917339,0.239075259,0.306884773,1.066654783,1.005085151];
expData_raw.her3 = [0.203233765,0.194358998,0.303475212,0.674083831,0.89702403,0.459831389];
expData_raw.pDGFR = [0.474174188,0.492132953,0.743620725,1.266460499,2.514722273,2.482761079];
% Normalize
species_names = fieldnames(expData_raw);
expData_norm = struct();
for i = 1:length(species_names)
    data = expData_raw.(species_names{i});
    expData_norm.(species_names{i}) = (data - min(data)) / (max(data) - min(data) + eps);
end

fprintf('Experimental data normalized.\n\n');

%% ============================================================================
%  SECTION 4: PARAMETER OPTIMIZATION SETUP
%  ============================================================================

fprintf('Setting up parameter optimization...\n');

params0 = params_vector;
num_params = length(params0);
lb = 1e-12 * ones(num_params, 1);
ub = 1.0 * ones(num_params, 1); % EXPANDED search space (up to 1.0 min^-1)

% Trametinib / Vemurafenib Concentrations (Fixed or constrained)
lb(54) = 1e-8; ub(54) = 1e-4; % Trametinib Concentration
lb(58) = 0.0;  ub(58) = 0.0;  % Vemurafenib (STRICTLY 0)
lb(61) = 0.0;  ub(61) = 0.0;  % Paradox strength (STRICTLY 0)

% Drug Affinity Bounds (Search space coverage for nanomolar to millimolar)
lb(55) = 1e-12; ub(55) = 1.0; % Ki_RAF
lb(56) = 1e-12; ub(56) = 1.0; % Ki_KSR

% K_displace Constraint
lb(68) = 1e-4; ub(68) = 1000.0; % Sensitivity to Upstream Load

% Physical constraints (Hill, IC50)
lb(57) = 0.5;  ub(57) = 5.0;  % Tram Hill
lb(63) = 0.01; ub(63) = 0.99; % Vem IC50

opts = optimoptions(@fmincon, ...
    'Algorithm', 'sqp', ...
    'Display', 'iter', ...
    'MaxIterations', 5000, ...
    'MaxFunctionEvaluations', 500000, ...
    'FunctionTolerance', 1e-10, ...
    'StepTolerance', 1e-14, ...
    'UseParallel', false);

%% ============================================================================
%  SECTION 5 & 6: RUN OPTIMIZATION AND SIMULATE
%  ============================================================================

% Optimization and Simulation Options
ode_opts = odeset('RelTol', 1e-6, 'AbsTol', 1e-8, 'NonNegative', 1:68);

fprintf('Running Global Parameter Optimization (GlobalSearch)...\n');
tic;

% Create the optimization problem for GlobalSearch
% This wraps the existing fmincon setup
problem = createOptimProblem('fmincon', ...
    'objective', @(p) objectiveFunction_all(p, timeStamps_seconds, expData_norm, y0), ...
    'x0', params0, ...
    'lb', lb, ...
    'ub', ub, ...
    'options', opts);

% Initialize the GlobalSearch object
% NumTrialPoints: Total points tested | NumStageOnePoints: Points used to identify promising regions
gs = GlobalSearch('Display', 'iter', ...
                 'NumTrialPoints', 200, ...
                 'NumStageOnePoints', 50, ...
                 'StartPointsToRun', 'bounds-ineqs');

% Run GlobalSearch
[optimizedParams, errorOpt] = run(gs, problem);

toc;

fprintf('Optimized Parameters:\n'); disp(optimizedParams);
fprintf('Minimum Fit Error: %.6e\n\n', errorOpt);

fprintf('Simulating with optimized parameters...\n');
[T_all, Y_all] = ode15s(@(t,y) Mapk_ODE(t, y, optimizedParams), timeStamps_seconds, y0, ode_opts);

% Check for simulation failure
if length(T_all) ~= length(timeStamps_seconds)
    warning('Full simulation failed. Results will be truncated or invalid.');
    % Pad Y_all with NaNs if needed to avoid size mismatch errors below
    rows_needed = length(timeStamps_seconds) - size(Y_all, 1);
    if rows_needed > 0
        Y_all = [Y_all; NaN(rows_needed, 68)];
    end
end

% Helper Normalization Function (Matches experimental Min-Max scaling)
min_max_norm = @(v) (v - min(v)) ./ (max(v) - min(v) + eps);

% Evaluate Fit
model_outputs.pEGFR = min_max_norm(Y_all(:,3));
model_outputs.panRAS = min_max_norm(Y_all(:,16) + Y_all(:,18) + Y_all(:,20));
model_outputs.pCRAF = min_max_norm(Y_all(:,23));
model_outputs.pMEK = min_max_norm(Y_all(:,27));
model_outputs.pERK = min_max_norm(Y_all(:,29));
model_outputs.DUSP = min_max_norm(Y_all(:,31));
model_outputs.pAKT = min_max_norm(Y_all(:,53));
model_outputs.p4EBP1 = min_max_norm(Y_all(:,59));
model_outputs.pHer2 = min_max_norm(Y_all(:,6));
model_outputs.pHer3 = min_max_norm(Y_all(:,9));
model_outputs.pDGFR = min_max_norm(Y_all(:,65));
model_outputs.pS6K = min_max_norm(Y_all(:,67));

% Calculate fit errors (display purposes)
fit_errors.pEGFR = abs(model_outputs.pEGFR - expData_norm.pEGFR(:));
fit_errors.panRAS = abs(model_outputs.panRAS - expData_norm.panRAS(:));
fit_errors.pCRAF = abs(model_outputs.pCRAF - expData_norm.pCRAF(:));
fit_errors.pMEK = abs(model_outputs.pMEK - expData_norm.pMEK(:));
fit_errors.pERK = abs(model_outputs.pERK - expData_norm.pERK(:));
fit_errors.DUSP = abs(model_outputs.DUSP - expData_norm.DUSP(:));
fit_errors.pAKT = abs(model_outputs.pAKT - expData_norm.pAKT(:));
fit_errors.p4EBP1 = abs(model_outputs.p4EBP1 - expData_norm.p4ebp1(:));
fit_errors.pHer2 = abs(model_outputs.pHer2 - expData_norm.her2(:));
fit_errors.pHer3 = abs(model_outputs.pHer3 - expData_norm.her3(:));
fit_errors.pDGFR = abs(model_outputs.pDGFR - expData_norm.pDGFR(:));
fit_errors.pS6K = abs(model_outputs.pS6K - expData_norm.pS6k(:));

fprintf('Fit Error: %.4f\n\n', errorOpt);

%% ============================================================================
%  SECTION 7: VISUALIZATION
%  ============================================================================

fprintf('Generating plots...\n');
tFine_hours = linspace(0, 48, 400);
[T_fine, Y_fine] = ode15s(@(t,y) Mapk_ODE(t, y, optimizedParams), tFine_hours*3600, y0, ode_opts);

% Extract Smooth
model_smooth.pEGFR = min_max_norm(Y_fine(:,3));
model_smooth.panRAS = min_max_norm(Y_fine(:,16) + Y_fine(:,18) + Y_fine(:,20));
model_smooth.pCRAF = min_max_norm(Y_fine(:,23));
model_smooth.pMEK = min_max_norm(Y_fine(:,27));
model_smooth.pERK = min_max_norm(Y_fine(:,29));
model_smooth.DUSP = min_max_norm(Y_fine(:,31));
model_smooth.pAKT = min_max_norm(Y_fine(:,53));
model_smooth.p4EBP1 = min_max_norm(Y_fine(:,59));
model_smooth.pHer2 = min_max_norm(Y_fine(:,6));
model_smooth.pHer3 = min_max_norm(Y_fine(:,9));
model_smooth.pDGFR = min_max_norm(Y_fine(:,65));
model_smooth.pS6K = min_max_norm(Y_fine(:,67));

% Figure 1: Model Fit
figure('Name', 'Model Fit - Trametinib', 'Position', [50, 50, 1600, 1000]);
species_to_plot = {'pEGFR', 'pCRAF', 'pMEK', 'pERK', 'DUSP', 'pAKT', 'p4EBP1', 'panRAS', 'pHer2', 'pHer3', 'pDGFR', 'pS6K'};
exp_fields = {'pEGFR', 'pCRAF', 'pMEK', 'pERK', 'DUSP', 'pAKT', 'p4ebp1', 'panRAS', 'her2', 'her3', 'pDGFR', 'pS6k'};

for i = 1:length(species_to_plot)
    subplot(3, 4, i);
    spec = species_to_plot{i};
    field = exp_fields{i};
    plot(tFine_hours, model_smooth.(spec), 'g-', 'LineWidth', 3); hold on;
    plot(timeStamps_hours, expData_norm.(field), 'ro', 'LineWidth', 2, 'MarkerSize', 8);
    title(spec); xlabel('Time (h)'); grid on;
end
sgtitle('Trametinib Model Fit Results');

% Fit Error Summary Bar Chart
figure('Name', 'Fit Error Summary', 'Position', [100, 100, 800, 600]);
total_errors = [sum(fit_errors.pEGFR), sum(fit_errors.pCRAF), ...
                sum(fit_errors.pMEK), sum(fit_errors.pERK), sum(fit_errors.DUSP), ...
                sum(fit_errors.pAKT), sum(fit_errors.p4EBP1), sum(fit_errors.panRAS), ...
                sum(fit_errors.pHer2), sum(fit_errors.pHer3), sum(fit_errors.pDGFR), sum(fit_errors.pS6K)];
protein_names = {'pEGFR', 'pCRAF', 'pMEK', 'pERK', 'DUSP', 'pAKT', 'p4EBP1', 'panRAS', 'pHer2', 'pHer3', 'pDGFR', 'pS6K'};
bar(1:length(protein_names), total_errors, 'FaceColor', [0.2, 0.6, 0.8], 'EdgeColor', 'k');
set(gca, 'XTick', 1:length(protein_names), 'XTickLabel', protein_names);
ylabel('Total Fit Error', 'FontSize', 12);
title('Fit Errors by Protein', 'FontSize', 14, 'FontWeight', 'bold');
xtickangle(45); grid on;

% Save parameters
save('trained_Tram_params.mat', 'optimizedParams');
fprintf('Simulation Complete. Results saved.\n');


%% ============================================================================
%  ODE FUNCTION (Unified with Vemurafenib + Trametinib Logic)
%  ============================================================================
function dydt = Mapk_ODE(t, y, p)
    % Initialize
    dydt = zeros(68, 1);
    
    % Unpack Mapk Parameters (1-53) Same as Vemurafenib.m
    ka1=p(1); kr1=p(2); kc1=p(3); kpCraf=p(4); kpMek=p(5); kpErk=p(6); kDegradEgfr=p(7); 
    kErkInbEgfr=p(8); kShcDephos=p(9); kptpDeg=p(10); kGrb2CombShc=p(11); kSprtyInbGrb2=p(12); 
    kSosCombGrb2=p(13); kErkPhosSos=p(14); kErkPhosPcraf=p(15); kPcrafDegrad=p(16); 
    kErkPhosMek=p(17); kMekDegrad=p(18); kDuspInbErk=p(19); kErkDeg=p(20); kinbBraf=p(21); 
    kDuspStop=p(22); kDusps=p(23); kSproutyForm=p(24); kSprtyComeDown=p(25); kdegrad=p(26); 
    km_Sprty_decay=p(27); km_Dusp=p(28); km_Sprty=p(29); kErkDephos=p(30); kDuspDeg=p(31);
    kHer2_act=p(32); kHer3_act=p(33); k_p85_bind_EGFR=p(34); k_p85_bind_Her2=p(35); 
    k_p85_bind_Her3=p(36); k_p85_bind_IGFR=p(37); k_p85_unbind=p(38); k_PI3K_recruit=p(39); 
    kMTOR_Feedback=p(40); k_PIP2_to_PIP3=p(41); k_PTEN=p(42); kAkt=p(43); kdegradAKT=p(44); 
    kb1=p(45); k43b1=p(46); k4ebp1=p(47); k_4EBP1_dephos=p(48); kKSRphos=p(49); kKSRdephos=p(50);
    kMekByBraf=p(51); kMekByCraf=p(52); kMekByKSR=p(53);
    
    % Drug Parameters
    Tram=p(54); K_tram_RAF=p(55); K_tram_KSR=p(56); n_tram=p(57);
    Vemurafenib=p(58); kDimerForm=p(59); kDimerDissoc=p(60); kParadoxCRAF=p(61); 
    IC50_vem=p(62); Hill_n_vem=p(63);
    
    % New Parameters
    kPDGFR_act=p(64); k_p85_bind_PDGFR=p(65); kS6K_phos=p(66); kS6K_dephos=p(67);
    K_displace=p(68); % Added for Trametinib
    
    % --- ODEs ---
    
    % 1. RTK Module
    dydt(1) = -ka1*y(1) + kr1*y(2);
    dydt(2) = ka1*y(1) - kr1*y(2) - kc1*y(2);
    dydt(3) = kc1*y(2) - kDegradEgfr*y(3) - kErkInbEgfr*y(29)*y(3);
    
    dydt(4) = -kHer2_act*y(4) + kr1*y(5);
    dydt(5) = kHer2_act*y(4) - kr1*y(5) - kc1*y(5);
    dydt(6) = kc1*y(5) - kDegradEgfr*y(6) - kErkInbEgfr*y(29)*y(6);
    
    dydt(7) = -kHer3_act*y(7) + kr1*y(8);
    dydt(8) = kHer3_act*y(7) - kr1*y(8) - kc1*y(8);
    dydt(9) = kc1*y(8) - kDegradEgfr*y(9) - kErkInbEgfr*y(29)*y(9);
    
    % 2. Shc/Grb2/SOS
    dydt(10) = -ka1*y(3)*y(10);
    dydt(11) = ka1*y(3)*y(10) - kShcDephos*y(12)*y(11);
    dydt(12) = -kptpDeg*y(11)*y(12);                                   % pShc
    
    dydt(13) = kGrb2CombShc*y(11)*y(3) - kSprtyInbGrb2*y(33)*y(13);    % Grb2 (FIXED: use SPRY protein y(33))
    dydt(14) = kSosCombGrb2*y(13)*y(11) - kErkPhosSos*y(29)*y(14);     % pShc:Grb2:SOS (FIXED: use pERK y(29))
    
    % 3. RAS
    dydt(15) = -ka1*y(14)*y(15); dydt(16) = ka1*y(14)*y(15);
    dydt(17) = -ka1*y(14)*y(17); dydt(18) = ka1*y(14)*y(17);
    dydt(19) = -ka1*y(14)*y(19);
    dydt(20) = ka1*y(14)*y(19) - ka1*y(20)*y(21);
    dydt(21) = -ka1*y(20)*y(21);
    
    % 4. RAF / Vemurafenib Logic
    IC50_n = IC50_vem^Hill_n_vem; Vem_n = Vemurafenib^Hill_n_vem;
    kBRAF_eff = ka1 * IC50_n / (IC50_n + Vem_n + eps);
    paradox_activation = kParadoxCRAF * Vemurafenib * y(62);
    
    dydt(22) = -kpCraf*y(20)*y(22) + kErkPhosPcraf*y(29)*y(23) + kPcrafDegrad*y(23)*y(36) - kDimerForm*y(25)*y(22)*Vemurafenib + kDimerDissoc*y(62);
    dydt(23) = kpCraf*y(20)*y(22) - kErkPhosPcraf*y(29)*y(23) - kPcrafDegrad*y(23)*y(36) + paradox_activation;
    dydt(24) = -kBRAF_eff*y(24)*y(20) - kDimerForm*y(25)*y(22)*Vemurafenib + kDimerDissoc*y(62);
    dydt(25) = kBRAF_eff*y(24)*y(20) - kinbBraf*y(25) - kDimerForm*y(25)*y(22)*Vemurafenib + kDimerDissoc*y(62);
    dydt(62) = kDimerForm*y(25)*y(22)*Vemurafenib - kDimerDissoc*y(62) - kPcrafDegrad*y(62)*y(36);
    
    % ========================================================================
    % MODULE 5 & 6: MEK/ERK MODULE WITH TRAMETINIB MECHANISM
    % ========================================================================
    % MECHANISM:
    % 1. Trametinib binds MEK allosterically, preventing it from phosphorylating ERK.
    % 2. It does NOT prevent MEK from being phosphorylated by RAF (pMEK accumulation).
    % 3. "Scaffold Protection": High upstream flux (RAS/RAF/KSR) reduces drug efficacy
    %    by effectively outcompeting the inhibitor or stabilizing active conformations.
    
    % A. Quantify Upstream Signaling Load (RAS + RAFs)
    panRAS_active = y(16) + y(18) + y(20); % HRAS-GTP + NRAS-GTP + KRAS-GTP
    CRAF_total = y(22) + y(23);           % CRAF + pCRAF
    BRAF_total = y(24) + y(25);           % BRAF + BRAF^P
    
    % B. Calculate Resistance Factor (Displacement by Flux)
    % Higher load = Higher effective Ki (lower potency)
    % Using (Load/K)^2 to model threshold-based scaffold protection
    Stream_Load = panRAS_active + CRAF_total + BRAF_total;
    Ki_effective = K_tram_RAF * (1 + (Stream_Load / K_displace)^2);
    
    % C. Calculate Inhibitor Efficacy (Fraction of MEK that remains active)
    f_MEK_activity = 1 / (max(0.01, 1 + (Tram / max(eps, Ki_effective))^n_tram));
    
    % D. MEK Phosphorylation (RAF -> MEK)
    % DOES NOT include f_MEK_activity because Trametinib allows pMEK formation.
    % pMEK levels are determined by RAF activity vs Phosphatases.
    raf_to_mek = (kpMek*y(23) + kMekByBraf*y(25) + kMekByCraf*y(23) + kpMek*y(62));
    ksr_to_mek = (kMekByKSR * y(61));
    
    dydt(26) = -(raf_to_mek + ksr_to_mek)*y(26) + kErkPhosMek*y(29)*y(27) + kMekDegrad*y(27)*y(35);
    dydt(27) = (raf_to_mek + ksr_to_mek)*y(26) - kErkPhosMek*y(29)*y(27) - kMekDegrad*y(27)*y(35);
    
    % E. ERK Phosphorylation (pMEK -> pERK)
    % THIS is where Trametinib acts (blocking catalysis).
    % Modulated by f_MEK_activity (which includes the resistance logic).
    erk_activation = kpErk * y(27) * y(28) * f_MEK_activity;
    
    dydt(28) = -erk_activation + kDuspInbErk*y(31)*y(29) + kErkDeg*y(29)*y(34) + kErkDephos*y(31)*y(29);
    dydt(29) = erk_activation - kDuspInbErk*y(31)*y(29) - kErkDeg*y(29)*y(34) - kErkDephos*y(31)*y(29);
    
    % 7. Feedback & Degradation
    dydt(30) = km_Dusp*y(29)/(1 + (km_Dusp/kDusps)*y(29)) - kDuspStop*y(30)*y(37) - kDuspDeg*y(30)*y(29);
    dydt(31) = -kDuspStop*y(30)*y(31);
    dydt(32) = km_Sprty*y(29)/(1 + (km_Sprty/kSproutyForm)*y(29)) - kSprtyComeDown*y(32)*y(33);
    dydt(33) = -kSprtyComeDown*y(32)*y(33);
    dydt(34) = -kErkDeg*y(29)*y(34); dydt(35) = -kMekDegrad*y(27)*y(35); 
    dydt(36) = -kPcrafDegrad*y(23)*y(36); dydt(37) = -kDuspStop*y(30)*y(37);
    
    % 10. PI3K/AKT/mTOR
    dydt(38) = -ka1*y(38) + kr1*y(39); dydt(39) = ka1*y(38) - kr1*y(39) - kc1*y(39);
    dydt(40) = kc1*y(39) - kErkInbEgfr*y(29)*y(40);
    dydt(41) = -ka1*y(3)*y(41); dydt(42) = ka1*y(3)*y(41);
    
    % p85:RTK complexes
    dydt(44) = k_p85_bind_EGFR*y(3)*y(43) - k_p85_unbind*y(44);  % p85:pEGFR
    dydt(45) = k_p85_bind_Her2*y(6)*y(43) - k_p85_unbind*y(45);  % p85:pHer2
    dydt(46) = k_p85_bind_Her3*y(9)*y(43) - k_p85_unbind*y(46);  % p85:pHer3
    dydt(47) = k_p85_bind_IGFR*y(40)*y(43) - k_p85_unbind*y(47); % p85:pIGFR
    dydt(68) = k_p85_bind_PDGFR*y(65)*y(43) - k_p85_unbind*y(68); % p85:pPDGFR
    
    % PI3K module - recruitment by p85:RTK complexes
    % Total p85:RTK complexes recruit PI3K (p110 catalytic subunit)
    total_p85_RTK = y(44) + y(45) + y(46) + y(47) + y(68);
    dydt(48) = -k_PI3K_recruit*total_p85_RTK*y(48) + kMTOR_Feedback*y(56)*y(49);
    dydt(49) = k_PI3K_recruit*total_p85_RTK*y(48) - kMTOR_Feedback*y(56)*y(49);
    
    dydt(50) = -k_PIP2_to_PIP3*y(49)*y(50) + k_PTEN*y(51);
    dydt(51) = k_PIP2_to_PIP3*y(49)*y(50) - k_PTEN*y(51);
    
    dydt(52) = -kAkt*y(51)*y(52) + kdegradAKT*y(53);
    dydt(53) = kAkt*y(51)*y(52) - kdegradAKT*y(53);
    
    % FOXO module - FOXO (y(54))
    % FOXO is inactivated by pAKT (simplified model)
    % Linked to kAkt (p43) instead of generic ka1
    % Robust denominator to prevent singularity if FOXO goes negative
    dydt(54) = (max(0, 1 - y(53))) * kAkt / (max(0.1, 1 + (y(54) / 15e-5)));
    
    % mTORC module - mTORC (y(55)) and mTORC_active (y(56))
    % mTORC activation by pAKT - Linked to kAkt (p43) instead of generic ka1
    dydt(55) = -kAkt * y(53) * y(55) + kdegrad * y(56);  % mTORC (inactive)
    dydt(56) = kAkt * y(53) * y(55) - kdegrad * y(56);   % mTORC_active (activated by pAKT)
    
    dydt(57) = -k4ebp1*y(56)*y(57) + kb1*y(58) + k_4EBP1_dephos*y(59);
    dydt(58) = k4ebp1*y(56)*y(57) - kb1*y(58) - k43b1*y(58);
    dydt(59) = k43b1*y(58) - k_4EBP1_dephos*y(59);
    
    % KSR
    dydt(60) = -kKSRphos*(y(16)+y(18)+y(21))*y(60) + kKSRdephos*y(61);
    dydt(61) = kKSRphos*(y(16)+y(18)+y(21))*y(60) - kKSRdephos*y(61);
    
    % PDGFR
    dydt(63) = -kPDGFR_act*y(63) + kr1*y(64);
    dydt(64) = kPDGFR_act*y(63) - kr1*y(64) - kc1*y(64);
    dydt(65) = kc1*y(64) - kDegradEgfr*y(65) - kErkInbEgfr*y(29)*y(65);
    
    % S6K
    dydt(66) = -kS6K_phos*y(56)*y(66) + kS6K_dephos*y(67);
    dydt(67) = kS6K_phos*y(56)*y(66) - kS6K_dephos*y(67);
    
    % p85:PDGFR
    dydt(68) = k_p85_bind_PDGFR*y(65)*y(43) - k_p85_unbind*y(68);
end

%% ============================================================================
%  OBJECTIVE FUNCTION
%  ============================================================================
function err = objectiveFunction_all(p, timeStamps_seconds, expData_norm, y0)
    % Set non-negative options for optimization internal calls
    opt_ode = odeset('RelTol', 1e-5, 'AbsTol', 1e-7, 'NonNegative', 1:68);
    try
        [T, Y] = ode15s(@(t,y) Mapk_ODE(t, y, p), timeStamps_seconds, y0, opt_ode);
        if length(T) ~= length(timeStamps_seconds); err = 1e9; return; end
    catch
        err = 1e9; return;
    end
    
    % Extract
    m_pEGFR=Y(:,3); m_panRAS=Y(:,16)+Y(:,18)+Y(:,20); m_pCRAF=Y(:,23); 
    m_pMEK=Y(:,27); m_pERK=Y(:,29); m_DUSP=Y(:,31); m_pAKT=Y(:,53); m_p4EBP1=Y(:,59);
    m_pHer2=Y(:,6); m_pHer3=Y(:,9); m_pDGFR=Y(:,65); m_pS6K=Y(:,67);
    
    % Normalize ALL species to [0,1] range (Min-Max) exactly as plotted
    % Subtract min and divide by range to ensure dots and lines are on same scale
    norm=@(v) (v - min(v)) ./ (max(v) - min(v) + 1e-6);
    
    m_pEGFR=norm(Y(:,3)); m_panRAS=norm(Y(:,16)+Y(:,18)+Y(:,20)); m_pCRAF=norm(Y(:,23)); 
    m_pMEK=norm(Y(:,27)); m_pERK=norm(Y(:,29)); m_DUSP=norm(Y(:,31)); m_pAKT=norm(Y(:,53)); m_p4EBP1=norm(Y(:,59));
    m_pHer2=norm(Y(:,6)); m_pHer3=norm(Y(:,9)); m_pDGFR=norm(Y(:,65)); m_pS6K=norm(Y(:,67));
    
    % Weighted Error (Matching the biological sensitivity identified in blotting)
    w=struct('EGFR',1,'MEK',5,'ERK',3,'DUSP',2,'CRAF',2,'AKT',3,'p4EBP1',4,'panRAS',4, ...
             'Her2',3,'Her3',2,'PDGFR',5,'S6K',4);
    err = 0;
    err = err + w.EGFR*sum((m_pEGFR - expData_norm.pEGFR(:)).^2);
    err = err + w.panRAS*sum((m_panRAS - expData_norm.panRAS(:)).^2);
    err = err + w.MEK*sum((m_pMEK - expData_norm.pMEK(:)).^2);
    err = err + w.ERK*sum((m_pERK - expData_norm.pERK(:)).^2);
    err = err + w.DUSP*sum((m_DUSP - expData_norm.DUSP(:)).^2);
    err = err + w.CRAF*sum((m_pCRAF - expData_norm.pCRAF(:)).^2);
    err = err + w.AKT*sum((m_pAKT - expData_norm.pAKT(:)).^2);
    err = err + w.p4EBP1*sum((m_p4EBP1 - expData_norm.p4ebp1(:)).^2);
    err = err + w.Her2*sum((m_pHer2 - expData_norm.her2(:)).^2);
    err = err + w.Her3*sum((m_pHer3 - expData_norm.her3(:)).^2);
    err = err + w.PDGFR*sum((m_pDGFR - expData_norm.pDGFR(:)).^2);
    err = err + w.S6K*sum((m_pS6K - expData_norm.pS6k(:)).^2);
end
