% =============================================================================
% MAPK/PI3K PATHWAY MODEL WITH VEMURAFENIB AND PARADOXICAL ACTIVATION
% =============================================================================
%
% Model Description:
% This script implements a comprehensive ODE model of the MAPK and PI3K/AKT/mTOR
% signaling pathways with vemurafenib treatment and paradoxical activation.
%
% Key Features:
% - EGFR → RAS → RAF → MEK → ERK cascade
% - PI3K → AKT → mTOR pathway
% - DUSP and SPRY negative feedback loops
% - Vemurafenib inhibition of BRAF^V600E (Hill equation)
% - Paradoxical activation: vemurafenib-induced CRAF activation via dimerization
% - KSR scaffold protein for MEK phosphorylation
% - Parameter optimization to fit experimental data
%
% Model Structure:
% - 62 species (49 original + 11 new: Her2, Her3, p85, p85 complexes)
% - 60 parameters (52 original + 8 new: Her2/Her3 activation, p85 binding/recruitment)
% - Time units: seconds (experimental data in hours, converted)
%
% =============================================================================

clear all;
close all;
clc;

fprintf('═══════════════════════════════════════════════════════════════════════════\n');
fprintf('   MAPK/PI3K PATHWAY MODEL WITH VEMURAFENIB\n');
fprintf('   Paradoxical Activation Included\n');
fprintf('═══════════════════════════════════════════════════════════════════════════\n\n');

%% ============================================================================
% SECTION 1: MODEL PARAMETERS
% ============================================================================
% Organized by pathway/module for clarity

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
% Note: RAS isoforms (HRAS, NRAS, KRAS) share the same activation rate

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
params.Trametinib.conc = 1e-6;       % Trametinib concentration
params.Trametinib.Ki_RAF = 5e-1;     % Ki for RAF→MEK inhibition
params.Trametinib.Ki_KSR = 5e-12;    % Ki for KSR→MEK inhibition
params.Trametinib.Hill_n = 2;        % Hill coefficient

% ----------------------------------------------------------------------------
% 1.12 Vemurafenib and Paradoxical Activation Parameters
% ----------------------------------------------------------------------------
params.Vemurafenib.conc = 1.0;           % Vemurafenib concentration (normalized [0,1])
params.Vemurafenib.IC50 = 0.4;           % IC50 for BRAF^V600E inhibition (normalized [0,1])
params.Vemurafenib.Hill_n = 1.5;         % Hill coefficient for inhibition
params.Paradox.k_dimer_form = 6e-6;      % Dimer formation rate (BRAF-WT* + CRAF → dimer)
params.Paradox.k_dimer_dissoc = 1e-5;    % Dimer dissociation rate (min^-1)
params.Paradox.gamma = 0.5;              % Paradoxical CRAF activation strength

% ----------------------------------------------------------------------------
% 1.13 PI3K/AKT/mTOR Module Parameters (EXPANDED with p85 recruitment)
% ----------------------------------------------------------------------------
% p85 regulatory subunit binding to phosphotyrosine sites
params.PI3K.k_p85_bind_EGFR = 10e-4;   % p85 binding to pEGFR (min^-1)
params.PI3K.k_p85_bind_Her2 = 10e-4;   % p85 binding to pHer2 (min^-1)
params.PI3K.k_p85_bind_Her3 = 10e-4;   % p85 binding to pHer3 (min^-1)
params.PI3K.k_p85_bind_IGFR = 10e-4;   % p85 binding to pIGFR (min^-1)
params.PI3K.k_p85_unbind = 10e-5;      % p85 unbinding from RTKs (min^-1)
% PI3K recruitment and activation
params.PI3K.k_PI3K_recruit = 10e-4;    % PI3K recruitment by p85:RTK (min^-1)
params.PI3K.k_MTOR_feedback = 10e-4;   % mTOR feedback on PI3K (min^-1)
% PIP2/PIP3 conversion - BOOSTED for proper AKT activation
params.PI3K.k_PIP2_to_PIP3 = 50e-4;    % PIP2 to PIP3 conversion by active PI3K (min^-1) [INCREASED 5x]
params.PI3K.k_PTEN = 2e-5;             % PTEN-mediated PIP3 to PIP2 dephosphorylation (min^-1) [REDUCED 5x]
params.AKT.k_act = 50e-5;              % AKT activation rate (min^-1) [INCREASED 5x]
params.AKT.k_degrad = 5e-7;            % AKT degradation (min^-1) [REDUCED 2x]
params.mTOR.kb1 = 10e-8;               % mTOR activation parameter
params.mTOR.k43b1 = 10e-3;             % mTOR activation parameter
params.mTOR.k_4EBP1 = 10e-5;           % 4EBP1 phosphorylation rate by mTORC (min^-1)
params.mTOR.k_4EBP1_dephos = 10e-5;   % p4EBP1 dephosphorylation rate (min^-1)

% ----------------------------------------------------------------------------
% 1.14 General Degradation Parameters
% ----------------------------------------------------------------------------
params.Degrad.general = 10e-7;      % General degradation rate (min^-1)

% Convert parameter structure to vector for optimization
% Order must match objectiveFunction_all parameter unpacking
p = params;
params_vector = [
    p.RTK.k_on, p.RTK.k_off, p.RTK.k_cat, ...
    p.RAF.k_CRAF_act, p.MEK.k_phos, p.ERK.k_phos, ...
    p.RTK.k_degrad, p.RTK.k_ERK_inhib, p.SOS.k_Shc_dephos, p.SOS.k_ptp, p.SOS.k_Grb2_bind, ...
    p.SOS.k_Sprty_inhib, p.SOS.k_SOS_bind, p.SOS.k_ERK_phos, ...
    p.RAF.k_ERK_phos_CRAF, p.RAF.k_CRAF_degrad, p.MEK.k_ERK_phos, p.MEK.k_degrad, ...
    p.ERK.k_DUSP_inhib, p.ERK.k_degrad, p.BRAF.k_inhib, p.DUSP.k_DUSP_stop, p.DUSP.k_max_tx, ...
    p.SPRY.k_max_tx, p.SPRY.k_come_down, p.Degrad.general, p.SPRY.k_mRNA_decay, ...
    p.DUSP.k_max_tx, p.SPRY.k_max_tx, ...  % km_Dusp and km_Sprty (same as k_max_tx for Hill equation)
    p.ERK.k_DUSP_dephos, p.DUSP.k_DUSP_deg, ...
    p.RTK.k_Her2_act, p.RTK.k_Her3_act, ...  % NEW: Her2/Her3 activation
    p.PI3K.k_p85_bind_EGFR, p.PI3K.k_p85_bind_Her2, p.PI3K.k_p85_bind_Her3, p.PI3K.k_p85_bind_IGFR, ...  % NEW: p85 binding
    p.PI3K.k_p85_unbind, p.PI3K.k_PI3K_recruit, ...  % NEW: p85 unbinding and PI3K recruitment
    p.PI3K.k_MTOR_feedback, ...
    p.PI3K.k_PIP2_to_PIP3, p.PI3K.k_PTEN, ...  % NEW: PIP2/PIP3 conversion
    p.AKT.k_act, p.AKT.k_degrad, ...
    p.mTOR.kb1, p.mTOR.k43b1, p.mTOR.k_4EBP1, p.mTOR.k_4EBP1_dephos, ...  % 4EBP1 phosphorylation and dephosphorylation
    p.KSR.k_phos, p.KSR.k_dephos, ...
    p.MEK.k_BRAF_route, p.MEK.k_CRAF_route, p.KSR.k_MEK_route, ...
    p.Trametinib.conc, p.Trametinib.Ki_RAF, p.Trametinib.Ki_KSR, p.Trametinib.Hill_n, ...
    p.Vemurafenib.conc, p.Paradox.k_dimer_form, p.Paradox.k_dimer_dissoc, ...
    p.Paradox.gamma, p.Vemurafenib.IC50, p.Vemurafenib.Hill_n
];

% Verify parameter vector length
if length(params_vector) ~= 63
    error('Parameter vector has %d elements, expected 63!', length(params_vector));
end

fprintf('Parameters organized by pathway modules. Total: %d parameters.\n\n', length(params_vector));

%% ============================================================================
% SECTION 2: INITIAL CONDITIONS
% ============================================================================

fprintf('Setting initial conditions...\n');

% Define initial conditions for each species
% Format: [inactive, intermediate, active] or [inactive, active]

% RTK/EGFR Module (EXPANDED)
IC.EGFR = [1.0, 0.0, 0.0];          % EGFR, EGFR:ligand, pEGFR
IC.Her2 = [1.0, 0.0, 0.0];          % Her2, Her2:ligand, pHer2 (NEW)
IC.Her3 = [1.0, 0.0, 0.0];          % Her3, Her3:ligand, pHer3 (NEW)
IC.SHC = [1.0, 0.0, 1.0];           % Shc, pEGFR:Shc, pShc
IC.Grb2_SOS = [0.0, 0.0];           % Grb2, pShc:Grb2:SOS

% RAS Module
IC.HRAS = [0.0, 0.0];               % HRAS-GDP, HRAS-GTP
IC.NRAS = [0.0, 0.0];               % NRAS-GDP, NRAS-GTP
IC.KRAS = [1.0, 0.0, 1.0];          % KRAS-GDP, intermediate, KRAS-GTP

% RAF Module
IC.CRAF = [0.8, 0.2];               % CRAF, pCRAF
IC.BRAF = [1.0, 1.0];               % BRAF, BRAF^P (BRAF^V600E*)

% MEK/ERK Module
IC.MEK = [1.0, 1.0];                % MEK, pMEK
IC.ERK = [1.0, 0.8];                % ERK, pERK

% Feedback Module
IC.DUSP = [1.0, 1.0];               % DUSP mRNA, DUSP protein
IC.SPRY = [1.0, 1.0];               % SPRY mRNA, SPRY protein

% Degradation Trackers
IC.pERK_degrad = [1.0];
IC.pMEK_degrad = [1.0];
IC.pCRAF_degrad = [1.0];
IC.DUSP_stop = [1.0];

% PI3K/AKT/mTOR Module (EXPANDED with p85 recruitment)
IC.IGFR = [1.0, 0.0, 0.0];          % IGFR, IGFR:ligand, pIGFR
IC.IRS = [1.0, 0.0];                % IRS, pIRS
IC.p85 = [1.0];                     % p85 regulatory subunit (free) (NEW)
IC.p85_EGFR = [0.0];                % p85:pEGFR complex (NEW)
IC.p85_Her2 = [0.0];                 % p85:pHer2 complex (NEW)
IC.p85_Her3 = [0.0];                 % p85:pHer3 complex (NEW)
IC.p85_IGFR = [0.0];                 % p85:pIGFR complex (NEW)
IC.PI3K = [0.8, 0.2];               % PI3K (p110), PI3K_active - Start with some basal activity
IC.PIP = [0.95, 0.05];              % PIP2, PIP3 - Start with small PIP3 pool
IC.AKT = [0.8, 0.2];                % AKT, pAKT - Start with some basal pAKT
IC.FOXO = [0.5];                    % FOXO - Start at moderate level
IC.mTORC = [0.7, 0.3];              % mTORC, mTORC_active - Basal mTOR activity
IC.frebp1 = [0.3, 0.1, 0.6];        % 4EBP1, intermediate, p4EBP1 - Mostly phosphorylated at baseline
IC.PTEN = [0.7];                     % PTEN - Active phosphatase (NEW)

% KSR Module
IC.KSR = [1.0, 0.0];                % KSR, pKSR

% Paradoxical Activation
IC.BRAF_CRAF_dimer = [0.0];         % BRAF-WT:CRAF dimer

% Combine all initial conditions into state vector
% New species order: EGFR(1-3), Her2(4-6), Her3(7-9), Shc(10-12), Grb2_SOS(13-14),
% HRAS(15-16), NRAS(17-18), KRAS(19-21), CRAF(22-23), BRAF(24-25),
% MEK(26-27), ERK(28-29), DUSP(30-31), SPRY(32-33),
% Degrad(34-37), IGFR(38-40), IRS(41-42),
% p85(43), p85 complexes(44-47), PI3K(48-49), PIP(50-51), AKT(52-53),
% FOXO(54), mTORC(55-56), 4EBP1(57-59), KSR(60-61), Dimer(62), PTEN(63)
% Total: 63 species
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
    IC.KSR, IC.BRAF_CRAF_dimer, IC.PTEN
];

fprintf('Initial conditions set. Total species: %d\n\n', length(y0));

%% ============================================================================
% SECTION 3: EXPERIMENTAL DATA
% ============================================================================

fprintf('Loading experimental data...\n');

% Experimental time points (hours)
timeStamps_hours = [0, 1, 4, 8, 24, 48];
timeStamps_seconds = timeStamps_hours * 3600;  % Convert to seconds

% Raw experimental data (not normalized)
% Note: RAS_GTP removed from optimization as requested

expData_raw.pMEK = [1.444352623,1.3679024,0.045830147,0.083831418,1.367721545,1.36831541];
expData_raw.pERK = [1.114022078,0.026924306,0.021997086,0.032017915,0.027554675,0.065078457];
expData_raw.DUSP = [1.000155324,0.940131419,0.939828443,0.939338766,0.938512135,0.00044688];
expData_raw.pEGFR = [0.300182236,0.252046981,0.22856473,0.2028871,0.046178942,0.069217078];
expData_raw.pCRAF = [0.168209685,0.219892222,0.142801944,0.317505379,0.108380258,0.102267192];
expData_raw.pAKT = [1.236801158,0.778330635,1.167145085,1.166994592,1.166919345,1.167124558];
expData_raw.p4ebp1 = [1.052785913,0.989544247,0.98951505,0.989536989,0.988999207,0.988657075];

% Normalize experimental data to [0, 1] using min-max normalization
species_names = fieldnames(expData_raw);
expData_norm = struct();
for i = 1:length(species_names)
    data = expData_raw.(species_names{i});
    data_min = min(data);
    data_max = max(data);
    expData_norm.(species_names{i}) = (data - data_min) / (data_max - data_min + eps);
end

fprintf('Experimental data normalized. Time points: %d\n\n', length(timeStamps_hours));

%% ============================================================================
% SECTION 4: PARAMETER OPTIMIZATION SETUP
% ============================================================================

fprintf('Setting up parameter optimization...\n');

% Initial parameter guess (from params_vector)
params0 = params_vector;

% Parameter bounds
num_params = length(params0);

% RELAXED GLOBAL BOUNDS to improve fit accuracy
% Previous bounds [1e-8, 1e-4] were too restrictive for enzymatic rates (e.g., k_cat)
lb = 1e-12 * ones(num_params, 1);
ub = 1e-3 * ones(num_params, 1);  % Increased to allow faster rates

% Specific constraints for physical parameters (Concentrations, Hill coeffs)
% Vemurafenib parameters: FIXED concentration at 1.0, optimize IC50 and Hill
lb(58) = 1.0;      % Vemurafenib concentration FIXED at 1.0
ub(58) = 1.0; 
lb(62) = 0.01;     % IC50_vem
ub(62) = 0.99;
lb(63) = 0.5;      % Hill_n_vem
ub(63) = 5.0;

% Custom bounds for pMEK related parameters
% Indices: 5 (k_phos), 17 (k_ERK_phos), 18 (k_degrad), 51 (BRAF_route), 52 (CRAF_route), 53 (KSR_route)
lb([5, 17, 18, 51, 52, 53]) = 1e-12; 
ub([5, 17, 18, 51, 52, 53]) = 1e-3; % Allow wider range for MEK dynamics

% Custom bounds for p4EBP1 related parameters
% Indices: 47 (k_4EBP1), 48 (k_4EBP1_dephos)
lb([47, 48]) = 1e-12;
ub([47, 48]) = 1e-3; 

% Optimization options - Enhanced for better convergence
opts = optimoptions(@fmincon, ...
    'Algorithm', 'sqp', ...
    'Display', 'iter', ...
    'MaxIterations', 2000, ...      % Increased from 150 to ensure convergence
    'MaxFunctionEvaluations', 100000, ... % Sufficient evaluations
    'FunctionTolerance', 1e-7, ...  % Tighter tolerance
    'StepTolerance', 1e-9);

fprintf('Optimization setup complete. Parameters to optimize: %d\n\n', num_params);

%% ============================================================================
% SECTION 5: RUN PARAMETER OPTIMIZATION
% ============================================================================

fprintf('═══════════════════════════════════════════════════════════════════════════\n');
fprintf('   RUNNING PARAMETER OPTIMIZATION\n');
fprintf('═══════════════════════════════════════════════════════════════════════════\n\n');

tic;
[optimizedParams, errorOpt] = fmincon( ...
    @(p) objectiveFunction_all(p, timeStamps_seconds, expData_norm, y0), ...
    params0, [], [], [], [], lb, ub, [], opts);
optimization_time = toc;

fprintf('\nOptimization completed in %.2f seconds.\n', optimization_time);
fprintf('Optimized Parameters:\n');
disp(optimizedParams);
fprintf('Minimum Fit Error: %.6e\n\n', errorOpt);

% Save optimized parameters immediately
save('trained_Vem_params.mat', 'optimizedParams');
fprintf('Optimized parameters saved to trained_Vem_params.mat\n');

%% ============================================================================
% SECTION 6: SIMULATE WITH OPTIMIZED PARAMETERS
% ============================================================================

fprintf('Simulating with optimized parameters...\n');

% Simulate at experimental time points
% Simulate at experimental time points
[T_all, Y_all] = ode15s(@(t,y) Mapk_ODE(t, y, optimizedParams), ...
                       timeStamps_seconds, y0);

% Normalization helper function
normit = @(v, mode) v ./ max(eps, ...
    (strcmpi(mode,'first'))*v(1) + ...
    (strcmpi(mode,'last')) *v(end) + ...
    (strcmpi(mode,'max'))  *max(v));

% Extract and normalize model outputs
model_outputs.pEGFR = normit(Y_all(:,3), 'max');
model_outputs.panRAS = normit(Y_all(:,16) + Y_all(:,18) + Y_all(:,20), 'last'); % panRAS = HRAS-GTP + NRAS-GTP + KRAS-GTP
model_outputs.pCRAF = normit(Y_all(:,23), 'max');  % Updated index: pCRAF is now y(23)
model_outputs.pMEK = normit(Y_all(:,27), 'first');  % Updated index: pMEK is now y(27)
model_outputs.pERK = normit(Y_all(:,29), 'first');  % Updated index: pERK is now y(29)
model_outputs.DUSP = normit(Y_all(:,31), 'max');  % Updated index: DUSP is now y(31)
model_outputs.pAKT = normit(Y_all(:,53), 'max');  % Updated index: pAKT is now y(53)
model_outputs.p4EBP1 = normit(Y_all(:,59), 'max');  % Updated index: p4EBP1 is now y(59)


% Calculate fit errors
fit_errors = struct();
fit_errors.pEGFR = abs(model_outputs.pEGFR - expData_norm.pEGFR(:));
fit_errors.pCRAF = abs(model_outputs.pCRAF - expData_norm.pCRAF(:));
fit_errors.pMEK = abs(model_outputs.pMEK - expData_norm.pMEK(:));
fit_errors.pERK = abs(model_outputs.pERK - expData_norm.pERK(:));
fit_errors.DUSP = abs(model_outputs.DUSP - expData_norm.DUSP(:));
fit_errors.pAKT = abs(model_outputs.pAKT - expData_norm.pAKT(:));
fit_errors.p4EBP1 = abs(model_outputs.p4EBP1 - expData_norm.p4ebp1(:));
fit_errors.panRAS = zeros(size(fit_errors.pEGFR));  % Dummy for plotting (no exp data)

% Display fit results
fprintf('═══════════════════════════════════════════════════════════════════════════\n');
fprintf('   MODEL FIT RESULTS\n');
fprintf('═══════════════════════════════════════════════════════════════════════════\n');
fprintf('Total Fit Error: %.4f\n', errorOpt);
fprintf('Individual Protein Errors (sum across all timepoints):\n');
fprintf('  pEGFR: %.4f\n', sum(fit_errors.pEGFR));
%fprintf('  panRAS: %.4f\n', sum(fit_errors.panRAS));
fprintf('  pCRAF: %.4f\n', sum(fit_errors.pCRAF));
fprintf('  pMEK:  %.4f\n', sum(fit_errors.pMEK));
fprintf('  pERK:  %.4f\n', sum(fit_errors.pERK));
fprintf('  DUSP:  %.4f\n', sum(fit_errors.DUSP));
fprintf('  pAKT:  %.4f\n', sum(fit_errors.pAKT));
fprintf('  p4EBP1: %.4f\n', sum(fit_errors.p4EBP1));
fprintf('\n');

%% ============================================================================
% SECTION 7: VISUALIZATION
% ============================================================================

fprintf('Generating visualization plots...\n');

% Generate smooth time course for plotting
tFine_hours = linspace(0, 48, 400);
tFine_seconds = tFine_hours * 3600;

[T_fine, Y_fine] = ode15s(@(t,y) Mapk_ODE(t, y, optimizedParams), ...
                         tFine_seconds, y0);

% Normalize smooth model outputs
model_smooth.pEGFR = normit(Y_fine(:,3), 'max');
model_smooth.panRAS = normit(Y_fine(:,16) + Y_fine(:,18) + Y_fine(:,20), 'last');
model_smooth.pCRAF = normit(Y_fine(:,23), 'max');  % Updated index
model_smooth.pMEK = normit(Y_fine(:,27), 'first');  % Updated index
model_smooth.pERK = normit(Y_fine(:,29), 'first');  % Updated index
model_smooth.DUSP = normit(Y_fine(:,31), 'max');  % Updated index
model_smooth.pAKT = normit(Y_fine(:,53), 'max');  % Updated index
model_smooth.p4EBP1 = normit(Y_fine(:,59), 'max');  % Updated index

% Plotting colors
data_color = [0.8, 0.2, 0.2];    % Red for experimental data
model_color = [0.2, 0.6, 0.2];   % Green for model fit

% Create figure for all species
figure('Name', 'Model Fit - All Species', 'Position', [50, 50, 1600, 1000]);

species_to_plot = {'pEGFR', 'pCRAF', 'pMEK', 'pERK', 'DUSP', 'pAKT', 'p4EBP1'};
% Mapping between display names and experimental data field names (handle case differences)
expData_field_map = containers.Map({'pEGFR', 'pCRAF', 'pMEK', 'pERK', 'DUSP', 'pAKT', 'p4EBP1'}, ...
                                    {'pEGFR', 'pCRAF', 'pMEK', 'pERK', 'DUSP', 'pAKT', 'p4ebp1'});

for i = 1:length(species_to_plot)
    subplot(3, 3, i);
    species = species_to_plot{i};
    exp_field = expData_field_map(species);  % Get correct field name for experimental data
    
    % Plot model fit
    plot(tFine_hours, model_smooth.(species), '-', 'Color', model_color, ...
         'LineWidth', 3, 'DisplayName', 'Model Fit');
    hold on;
    
    % Plot experimental data
    plot(timeStamps_hours, expData_norm.(exp_field), 'o', 'Color', data_color, ...
         'MarkerSize', 10, 'MarkerFaceColor', data_color, 'LineWidth', 2, ...
         'DisplayName', 'Experimental');
    
    xlabel('Time (hours)', 'FontSize', 12);
    ylabel(species, 'FontSize', 12);
    title(sprintf('%s Model Fit', species), 'FontSize', 14, 'FontWeight', 'bold');
    legend('Location', 'best', 'FontSize', 11);
    grid on;
    xlim([0, 50]);
    set(gca, 'FontSize', 11);
    hold off;
end

% Fit error summary plot
subplot(3, 3, 9);
total_errors = [sum(fit_errors.pEGFR), sum(fit_errors.pCRAF), ...
                sum(fit_errors.pMEK), sum(fit_errors.pERK), sum(fit_errors.DUSP), ...
                sum(fit_errors.pAKT), sum(fit_errors.p4EBP1)];
protein_names = {'pEGFR', 'pCRAF', 'pMEK', 'pERK', 'DUSP', 'pAKT', 'p4EBP1'};
bar(1:length(protein_names), total_errors, 'FaceColor', [0.2, 0.6, 0.8], 'EdgeColor', 'k');
set(gca, 'XTick', 1:length(protein_names), 'XTickLabel', protein_names);
ylabel('Total Fit Error', 'FontSize', 12);
title('Fit Errors by Protein', 'FontSize', 14, 'FontWeight', 'bold');
xtickangle(45);
grid on;
set(gca, 'FontSize', 11);

sgtitle('MAPK/PI3K Pathway Model Fit Results', 'FontSize', 16, 'FontWeight', 'bold');

%% ============================================================================
% PARADOXICAL ACTIVATION VISUALIZATION
% ============================================================================

fprintf('Generating paradoxical activation plots...\n');

% Extract paradoxical activation species from simulation (updated indices)
paradox_species.dimer = Y_fine(:,62);           % BRAF-WT:CRAF dimer (now y(62))
paradox_species.pCRAF = Y_fine(:,23);           % pCRAF (includes paradoxical activation, now y(23))
paradox_species.CRAF = Y_fine(:,22);            % CRAF (inactive, now y(22))
paradox_species.BRAF_P = Y_fine(:,25);          % BRAF^P (BRAF^V600E*, now y(25))
paradox_species.BRAF = Y_fine(:,24);            % BRAF (inactive, now y(24))

% Get vemurafenib concentration from optimized parameters (correct indices)
Vemurafenib_conc = optimizedParams(58);  % Vemurafenib concentration (parameter 58)
kParadoxCRAF = optimizedParams(61);       % Paradoxical activation strength (parameter 61)

% Calculate paradoxical activation rate over time
paradox_activation_rate = kParadoxCRAF * Vemurafenib_conc * paradox_species.dimer;

% Normalize for visualization (optional - can show raw values too)
paradox_species.dimer_norm = paradox_species.dimer ./ (max(paradox_species.dimer) + eps);
paradox_species.pCRAF_norm = normit(paradox_species.pCRAF, 'max');
paradox_species.CRAF_norm = normit(paradox_species.CRAF, 'max');
paradox_species.BRAF_P_norm = normit(paradox_species.BRAF_P, 'max');
paradox_species.BRAF_norm = normit(paradox_species.BRAF, 'max');
paradox_activation_rate_norm = paradox_activation_rate ./ (max(paradox_activation_rate) + eps);

% Create figure for paradoxical activation
figure('Name', 'Paradoxical Activation Dynamics', 'Position', [100, 100, 1400, 900]);

% Plot 1: Vemurafenib concentration and BRAF-WT:CRAF dimer
subplot(2, 3, 1);
yyaxis left
plot(tFine_hours, ones(size(tFine_hours)) * Vemurafenib_conc, '-', 'Color', [0.8, 0.2, 0.2], ...
     'LineWidth', 3, 'DisplayName', 'Vemurafenib');
ylabel('Vemurafenib Concentration (normalized)', 'FontSize', 11);
ylim([0, 1.1]);
yyaxis right
plot(tFine_hours, paradox_species.dimer_norm, '-', 'Color', [0.2, 0.6, 0.8], ...
     'LineWidth', 2.5, 'DisplayName', 'BRAF-WT:CRAF Dimer');
ylabel('Dimer Concentration (normalized)', 'FontSize', 11);
xlabel('Time (hours)', 'FontSize', 11);
title('Vemurafenib and Dimer Formation', 'FontSize', 13, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 10);
grid on;
set(gca, 'FontSize', 10);

% Plot 2: CRAF and pCRAF dynamics
subplot(2, 3, 2);
plot(tFine_hours, paradox_species.CRAF_norm, '-', 'Color', [0.6, 0.6, 0.6], ...
     'LineWidth', 2, 'DisplayName', 'CRAF (inactive)');
hold on;
plot(tFine_hours, paradox_species.pCRAF_norm, '-', 'Color', [0.2, 0.8, 0.2], ...
     'LineWidth', 2.5, 'DisplayName', 'pCRAF (active)');
xlabel('Time (hours)', 'FontSize', 11);
ylabel('Concentration (normalized)', 'FontSize', 11);
title('CRAF Activation Dynamics', 'FontSize', 13, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 10);
grid on;
hold off;
set(gca, 'FontSize', 10);

% Plot 3: BRAF species
subplot(2, 3, 3);
plot(tFine_hours, paradox_species.BRAF_norm, '-', 'Color', [0.8, 0.6, 0.2], ...
     'LineWidth', 2, 'DisplayName', 'BRAF (inactive)');
hold on;
plot(tFine_hours, paradox_species.BRAF_P_norm, '-', 'Color', [0.9, 0.3, 0.1], ...
     'LineWidth', 2.5, 'DisplayName', 'BRAF^P (BRAF^V600E*)');
xlabel('Time (hours)', 'FontSize', 11);
ylabel('Concentration (normalized)', 'FontSize', 11);
title('BRAF Dynamics with Vemurafenib Inhibition', 'FontSize', 13, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 10);
grid on;
hold off;
set(gca, 'FontSize', 10);

% Plot 4: Paradoxical activation rate
subplot(2, 3, 4);
plot(tFine_hours, paradox_activation_rate_norm, '-', 'Color', [0.7, 0.1, 0.7], ...
     'LineWidth', 2.5, 'DisplayName', 'Paradoxical Activation Rate');
xlabel('Time (hours)', 'FontSize', 11);
ylabel('Activation Rate (normalized)', 'FontSize', 11);
title('Paradoxical CRAF Activation Rate', 'FontSize', 13, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 10);
grid on;
set(gca, 'FontSize', 10);

% Plot 5: Dimer formation and dissociation
subplot(2, 3, 5);
kDimerForm = optimizedParams(59);  % Dimer formation rate (parameter 59)
kDimerDissoc = optimizedParams(60);  % Dimer dissociation rate (parameter 60)
dimer_formation_rate = kDimerForm * paradox_species.BRAF_P .* paradox_species.CRAF * Vemurafenib_conc;
dimer_dissociation_rate = kDimerDissoc * paradox_species.dimer;

% Normalize rates for visualization
dimer_formation_rate_norm = dimer_formation_rate ./ (max(dimer_formation_rate) + eps);
dimer_dissociation_rate_norm = dimer_dissociation_rate ./ (max(dimer_dissociation_rate) + eps);

plot(tFine_hours, dimer_formation_rate_norm, '-', 'Color', [0.2, 0.7, 0.3], ...
     'LineWidth', 2, 'DisplayName', 'Dimer Formation');
hold on;
plot(tFine_hours, dimer_dissociation_rate_norm, '-', 'Color', [0.7, 0.2, 0.2], ...
     'LineWidth', 2, 'DisplayName', 'Dimer Dissociation');
xlabel('Time (hours)', 'FontSize', 11);
ylabel('Rate (normalized)', 'FontSize', 11);
title('Dimer Formation vs Dissociation', 'FontSize', 13, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 10);
grid on;
hold off;
set(gca, 'FontSize', 10);

% Plot 6: Combined view - pCRAF with and without paradoxical activation
subplot(2, 3, 6);
% Calculate pCRAF contribution from normal activation (RAS-dependent)
kpCraf = optimizedParams(4);
RAS_GTP = Y_fine(:,20);  % Updated: KRAS-GTP is now y(20)
normal_CRAF_activation = kpCraf * RAS_GTP .* paradox_species.CRAF;
normal_CRAF_activation_norm = normal_CRAF_activation ./ (max(normal_CRAF_activation) + eps);

plot(tFine_hours, normal_CRAF_activation_norm, '--', 'Color', [0.5, 0.5, 0.5], ...
     'LineWidth', 2, 'DisplayName', 'Normal CRAF Activation (RAS-dependent)');
hold on;
plot(tFine_hours, paradox_activation_rate_norm, '--', 'Color', [0.7, 0.1, 0.7], ...
     'LineWidth', 2, 'DisplayName', 'Paradoxical Activation');
plot(tFine_hours, paradox_species.pCRAF_norm, '-', 'Color', [0.2, 0.8, 0.2], ...
     'LineWidth', 2.5, 'DisplayName', 'Total pCRAF');
xlabel('Time (hours)', 'FontSize', 11);
ylabel('Concentration/Rate (normalized)', 'FontSize', 11);
title('CRAF Activation: Normal vs Paradoxical', 'FontSize', 13, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 10);
grid on;
hold off;
set(gca, 'FontSize', 10);

sgtitle('Paradoxical Activation Mechanism', 'FontSize', 16, 'FontWeight', 'bold');

% Display paradoxical activation parameters
fprintf('═══════════════════════════════════════════════════════════════════════════\n');
fprintf('   PARADOXICAL ACTIVATION PARAMETERS\n');
fprintf('═══════════════════════════════════════════════════════════════════════════\n');
fprintf('Vemurafenib Concentration: %.4f (normalized [0,1])\n', Vemurafenib_conc);
fprintf('Paradoxical Activation Strength (gamma): %.4f\n', kParadoxCRAF);
fprintf('Dimer Formation Rate: %.6e (min^-1)\n', optimizedParams(59));
fprintf('Dimer Dissociation Rate: %.6e (min^-1)\n', optimizedParams(60));
fprintf('Maximum Dimer Concentration: %.6e\n', max(paradox_species.dimer));
fprintf('Maximum Paradoxical Activation Rate: %.6e\n', max(paradox_activation_rate));
fprintf('\n');

%% ============================================================================
% FIGURE 3: NEW ADDITIONS - Her2, Her3, p85, and PI3K Recruitment
% ============================================================================

fprintf('Generating Figure 3: Her2, Her3, p85, and PI3K Recruitment...\n');

% Extract new species from simulation
new_species.pEGFR = Y_fine(:,3);           % pEGFR
new_species.pHer2 = Y_fine(:,6);           % pHer2 (NEW)
new_species.pHer3 = Y_fine(:,9);           % pHer3 (NEW)
new_species.pIGFR = Y_fine(:,40);          % pIGFR

% p85 species
new_species.p85_free = Y_fine(:,43);       % Free p85 (NEW)
new_species.p85_EGFR = Y_fine(:,44);       % p85:pEGFR complex (NEW)
new_species.p85_Her2 = Y_fine(:,45);       % p85:pHer2 complex (NEW)
new_species.p85_Her3 = Y_fine(:,46);       % p85:pHer3 complex (NEW)
new_species.p85_IGFR = Y_fine(:,47);       % p85:pIGFR complex (NEW)

% Calculate total p85:RTK complexes
new_species.total_p85_RTK = new_species.p85_EGFR + new_species.p85_Her2 + ...
                             new_species.p85_Her3 + new_species.p85_IGFR;

% PI3K species
new_species.PI3K_inactive = Y_fine(:,48);  % PI3K (p110, inactive)
new_species.PI3K_active = Y_fine(:,49);    % PI3K_active (recruited)

% Normalize for visualization
new_species.pEGFR_norm = normit(new_species.pEGFR, 'max');
new_species.pHer2_norm = normit(new_species.pHer2, 'max');
new_species.pHer3_norm = normit(new_species.pHer3, 'max');
new_species.pIGFR_norm = normit(new_species.pIGFR, 'max');
new_species.p85_free_norm = normit(new_species.p85_free, 'max');
new_species.p85_EGFR_norm = normit(new_species.p85_EGFR, 'max');
new_species.p85_Her2_norm = normit(new_species.p85_Her2, 'max');
new_species.p85_Her3_norm = normit(new_species.p85_Her3, 'max');
new_species.p85_IGFR_norm = normit(new_species.p85_IGFR, 'max');
new_species.total_p85_RTK_norm = normit(new_species.total_p85_RTK, 'max');
new_species.PI3K_inactive_norm = normit(new_species.PI3K_inactive, 'max');
new_species.PI3K_active_norm = normit(new_species.PI3K_active, 'max');

% Create Figure 3
figure('Name', 'Her2, Her3, p85, and PI3K Recruitment', 'Position', [150, 150, 1600, 1200]);

% Plot 1: RTK Activation (EGFR, Her2, Her3, IGFR)
subplot(3, 3, 1);
plot(tFine_hours, new_species.pEGFR_norm, '-', 'Color', [0.2, 0.6, 0.8], ...
     'LineWidth', 2.5, 'DisplayName', 'pEGFR');
hold on;
plot(tFine_hours, new_species.pHer2_norm, '-', 'Color', [0.8, 0.2, 0.2], ...
     'LineWidth', 2.5, 'DisplayName', 'pHer2');
plot(tFine_hours, new_species.pHer3_norm, '-', 'Color', [0.2, 0.8, 0.2], ...
     'LineWidth', 2.5, 'DisplayName', 'pHer3');
plot(tFine_hours, new_species.pIGFR_norm, '-', 'Color', [0.8, 0.6, 0.2], ...
     'LineWidth', 2.5, 'DisplayName', 'pIGFR');
xlabel('Time (hours)', 'FontSize', 11);
ylabel('Concentration (normalized)', 'FontSize', 11);
title('RTK Activation: EGFR, Her2, Her3, IGFR', 'FontSize', 13, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 10);
grid on;
hold off;
set(gca, 'FontSize', 10);

% Plot 2: Free p85 and Total p85:RTK Complexes
subplot(3, 3, 2);
yyaxis left
plot(tFine_hours, new_species.p85_free_norm, '-', 'Color', [0.5, 0.5, 0.5], ...
     'LineWidth', 2.5, 'DisplayName', 'Free p85');
ylabel('Free p85 (normalized)', 'FontSize', 11);
yyaxis right
plot(tFine_hours, new_species.total_p85_RTK_norm, '-', 'Color', [0.7, 0.1, 0.7], ...
     'LineWidth', 2.5, 'DisplayName', 'Total p85:RTK Complexes');
ylabel('Total p85:RTK Complexes (normalized)', 'FontSize', 11);
xlabel('Time (hours)', 'FontSize', 11);
title('p85 Regulatory Subunit Dynamics', 'FontSize', 13, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 10);
grid on;
set(gca, 'FontSize', 10);

% Plot 3: Individual p85:RTK Complexes
subplot(3, 3, 3);
plot(tFine_hours, new_species.p85_EGFR_norm, '-', 'Color', [0.2, 0.6, 0.8], ...
     'LineWidth', 2, 'DisplayName', 'p85:pEGFR');
hold on;
plot(tFine_hours, new_species.p85_Her2_norm, '-', 'Color', [0.8, 0.2, 0.2], ...
     'LineWidth', 2, 'DisplayName', 'p85:pHer2');
plot(tFine_hours, new_species.p85_Her3_norm, '-', 'Color', [0.2, 0.8, 0.2], ...
     'LineWidth', 2, 'DisplayName', 'p85:pHer3');
plot(tFine_hours, new_species.p85_IGFR_norm, '-', 'Color', [0.8, 0.6, 0.2], ...
     'LineWidth', 2, 'DisplayName', 'p85:pIGFR');
xlabel('Time (hours)', 'FontSize', 11);
ylabel('Complex Concentration (normalized)', 'FontSize', 11);
title('p85:RTK Complex Formation', 'FontSize', 13, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 10);
grid on;
hold off;
set(gca, 'FontSize', 10);

% Plot 4: PI3K Recruitment and Activation
subplot(3, 3, 4);
yyaxis left
plot(tFine_hours, new_species.PI3K_inactive_norm, '--', 'Color', [0.6, 0.6, 0.6], ...
     'LineWidth', 2, 'DisplayName', 'PI3K (inactive)');
hold on;
plot(tFine_hours, new_species.PI3K_active_norm, '-', 'Color', [0.2, 0.8, 0.2], ...
     'LineWidth', 2.5, 'DisplayName', 'PI3K (active, recruited)');
ylabel('PI3K Concentration (normalized)', 'FontSize', 11);
yyaxis right
plot(tFine_hours, new_species.total_p85_RTK_norm, '-', 'Color', [0.7, 0.1, 0.7], ...
     'LineWidth', 2, 'DisplayName', 'Total p85:RTK (recruiter)');
ylabel('Total p85:RTK Complexes (normalized)', 'FontSize', 11);
xlabel('Time (hours)', 'FontSize', 11);
title('PI3K Recruitment by p85:RTK Complexes', 'FontSize', 13, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 10);
grid on;
hold off;
set(gca, 'FontSize', 10);

% Plot 5: p85 Binding Rates (calculated from parameters)
subplot(3, 3, 5);
k_p85_bind_EGFR = optimizedParams(34);
k_p85_bind_Her2 = optimizedParams(35);
k_p85_bind_Her3 = optimizedParams(36);
k_p85_bind_IGFR = optimizedParams(37);
k_p85_unbind = optimizedParams(38);

% Calculate binding rates
binding_rate_EGFR = k_p85_bind_EGFR * new_species.pEGFR .* new_species.p85_free;
binding_rate_Her2 = k_p85_bind_Her2 * new_species.pHer2 .* new_species.p85_free;
binding_rate_Her3 = k_p85_bind_Her3 * new_species.pHer3 .* new_species.p85_free;
binding_rate_IGFR = k_p85_bind_IGFR * new_species.pIGFR .* new_species.p85_free;

% Normalize rates
max_rate = max([max(binding_rate_EGFR), max(binding_rate_Her2), ...
                max(binding_rate_Her3), max(binding_rate_IGFR)]) + eps;
binding_rate_EGFR_norm = binding_rate_EGFR / max_rate;
binding_rate_Her2_norm = binding_rate_Her2 / max_rate;
binding_rate_Her3_norm = binding_rate_Her3 / max_rate;
binding_rate_IGFR_norm = binding_rate_IGFR / max_rate;

plot(tFine_hours, binding_rate_EGFR_norm, '-', 'Color', [0.2, 0.6, 0.8], ...
     'LineWidth', 2, 'DisplayName', 'p85:pEGFR binding');
hold on;
plot(tFine_hours, binding_rate_Her2_norm, '-', 'Color', [0.8, 0.2, 0.2], ...
     'LineWidth', 2, 'DisplayName', 'p85:pHer2 binding');
plot(tFine_hours, binding_rate_Her3_norm, '-', 'Color', [0.2, 0.8, 0.2], ...
     'LineWidth', 2, 'DisplayName', 'p85:pHer3 binding');
plot(tFine_hours, binding_rate_IGFR_norm, '-', 'Color', [0.8, 0.6, 0.2], ...
     'LineWidth', 2, 'DisplayName', 'p85:pIGFR binding');
xlabel('Time (hours)', 'FontSize', 11);
ylabel('Binding Rate (normalized)', 'FontSize', 11);
title('p85 Binding Rates to RTKs', 'FontSize', 13, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 10);
grid on;
hold off;
set(gca, 'FontSize', 10);

% Plot 6: PI3K Recruitment Rate
subplot(3, 3, 6);
k_PI3K_recruit = optimizedParams(39);
PI3K_recruitment_rate = k_PI3K_recruit * new_species.total_p85_RTK .* new_species.PI3K_inactive;
PI3K_recruitment_rate_norm = PI3K_recruitment_rate ./ (max(PI3K_recruitment_rate) + eps);

plot(tFine_hours, PI3K_recruitment_rate_norm, '-', 'Color', [0.2, 0.8, 0.2], ...
     'LineWidth', 2.5, 'DisplayName', 'PI3K Recruitment Rate');
xlabel('Time (hours)', 'FontSize', 11);
ylabel('Recruitment Rate (normalized)', 'FontSize', 11);
title('PI3K Recruitment Rate by p85:RTK', 'FontSize', 13, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 10);
grid on;
set(gca, 'FontSize', 10);

% Plot 7: p85 Distribution (pie chart at final time point)
subplot(3, 3, 7);
final_idx = length(new_species.p85_free);
p85_distribution = [
    new_species.p85_free(final_idx);
    new_species.p85_EGFR(final_idx);
    new_species.p85_Her2(final_idx);
    new_species.p85_Her3(final_idx);
    new_species.p85_IGFR(final_idx)
];
p85_labels = {'Free p85', 'p85:pEGFR', 'p85:pHer2', 'p85:pHer3', 'p85:pIGFR'};
p85_colors = [0.5 0.5 0.5; 0.2 0.6 0.8; 0.8 0.2 0.2; 0.2 0.8 0.2; 0.8 0.6 0.2];
pie(p85_distribution, p85_labels);
colormap(gca, p85_colors);
title('p85 Distribution at Final Time Point', 'FontSize', 13, 'FontWeight', 'bold');
set(gca, 'FontSize', 10);

% Plot 8: RTK Contribution to PI3K Recruitment
subplot(3, 3, 8);
% Calculate contribution of each RTK to total p85:RTK
contribution_EGFR = new_species.p85_EGFR ./ (new_species.total_p85_RTK + eps);
contribution_Her2 = new_species.p85_Her2 ./ (new_species.total_p85_RTK + eps);
contribution_Her3 = new_species.p85_Her3 ./ (new_species.total_p85_RTK + eps);
contribution_IGFR = new_species.p85_IGFR ./ (new_species.total_p85_RTK + eps);

area(tFine_hours, [contribution_EGFR, contribution_Her2, contribution_Her3, contribution_IGFR]);
colormap(gca, [0.2 0.6 0.8; 0.8 0.2 0.2; 0.2 0.8 0.2; 0.8 0.6 0.2]);
xlabel('Time (hours)', 'FontSize', 11);
ylabel('Relative Contribution', 'FontSize', 11);
title('RTK Contribution to Total p85:RTK Complexes', 'FontSize', 13, 'FontWeight', 'bold');
legend('p85:pEGFR', 'p85:pHer2', 'p85:pHer3', 'p85:pIGFR', 'Location', 'best', 'FontSize', 10);
grid on;
set(gca, 'FontSize', 10);

% Plot 9: Summary - Key Metrics
subplot(3, 3, 9);
% Calculate key metrics
max_p85_RTK = max(new_species.total_p85_RTK);
max_PI3K_active = max(new_species.PI3K_active);
avg_p85_bound = mean(new_species.total_p85_RTK);
avg_PI3K_active = mean(new_species.PI3K_active);

metrics_text = {
    sprintf('Max Total p85:RTK: %.4f', max_p85_RTK);
    sprintf('Max PI3K Active: %.4f', max_PI3K_active);
    sprintf('Avg p85:RTK: %.4f', avg_p85_bound);
    sprintf('Avg PI3K Active: %.4f', avg_PI3K_active);
    sprintf('Max p85:pEGFR: %.4f', max(new_species.p85_EGFR));
    sprintf('Max p85:pHer2: %.4f', max(new_species.p85_Her2));
    sprintf('Max p85:pHer3: %.4f', max(new_species.p85_Her3));
    sprintf('Max p85:pIGFR: %.4f', max(new_species.p85_IGFR));
};

text(0.1, 0.5, metrics_text, 'FontSize', 11, 'VerticalAlignment', 'middle');
axis off;
title('Key Metrics Summary', 'FontSize', 13, 'FontWeight', 'bold');

sgtitle('Her2, Her3, p85 Recruitment, and PI3K Activation', 'FontSize', 16, 'FontWeight', 'bold');

% Display summary statistics
fprintf('═══════════════════════════════════════════════════════════════════════════\n');
fprintf('   NEW ADDITIONS SUMMARY\n');
fprintf('═══════════════════════════════════════════════════════════════════════════\n');
fprintf('Maximum RTK Activation:\n');
fprintf('  pEGFR: %.4f\n', max(new_species.pEGFR));
fprintf('  pHer2: %.4f\n', max(new_species.pHer2));
fprintf('  pHer3: %.4f\n', max(new_species.pHer3));
fprintf('  pIGFR: %.4f\n', max(new_species.pIGFR));
fprintf('\n');
fprintf('p85 Dynamics:\n');
fprintf('  Maximum Free p85: %.4f\n', max(new_species.p85_free));
fprintf('  Maximum Total p85:RTK Complexes: %.4f\n', max_p85_RTK);
fprintf('  Average p85:RTK Complexes: %.4f\n', avg_p85_bound);
fprintf('\n');
fprintf('PI3K Recruitment:\n');
fprintf('  Maximum PI3K Active: %.4f\n', max_PI3K_active);
fprintf('  Average PI3K Active: %.4f\n', avg_PI3K_active);
fprintf('  Maximum PI3K Recruitment Rate: %.6e\n', max(PI3K_recruitment_rate));
fprintf('\n');

fprintf('Visualization complete.\n\n');

%% ============================================================================
% FIGURE 4: ALL PI3K PATHWAY SPECIES
% ============================================================================

fprintf('Generating Figure 4: All PI3K Pathway Species...\n');

% Extract all PI3K pathway species from simulation
pi3k_species.IGFR = Y_fine(:,38);          % IGFR
pi3k_species.IGFR_ligand = Y_fine(:,39);  % IGFR:ligand
pi3k_species.pIGFR = Y_fine(:,40);        % pIGFR
pi3k_species.IRS = Y_fine(:,41);          % IRS
pi3k_species.pIRS = Y_fine(:,42);         % pIRS
pi3k_species.p85_free = Y_fine(:,43);     % Free p85
pi3k_species.p85_EGFR = Y_fine(:,44);     % p85:pEGFR complex
pi3k_species.p85_Her2 = Y_fine(:,45);     % p85:pHer2 complex
pi3k_species.p85_Her3 = Y_fine(:,46);     % p85:pHer3 complex
pi3k_species.p85_IGFR = Y_fine(:,47);     % p85:pIGFR complex
pi3k_species.PI3K_inactive = Y_fine(:,48); % PI3K (p110, inactive)
pi3k_species.PI3K_active = Y_fine(:,49);   % PI3K_active
pi3k_species.PIP2 = Y_fine(:,50);         % PIP2
pi3k_species.PIP3 = Y_fine(:,51);         % PIP3
pi3k_species.AKT = Y_fine(:,52);          % AKT
pi3k_species.pAKT = Y_fine(:,53);         % pAKT
pi3k_species.FOXO = Y_fine(:,54);         % FOXO
pi3k_species.mTORC = Y_fine(:,55);        % mTORC
pi3k_species.mTORC_active = Y_fine(:,56); % mTORC_active
pi3k_species.frebp1 = Y_fine(:,57);       % 4EBP1
pi3k_species.frebp1_inter = Y_fine(:,58);  % 4EBP1 intermediate
pi3k_species.p4EBP1 = Y_fine(:,59);       % p4EBP1

% Calculate total p85:RTK complexes
pi3k_species.total_p85_RTK = pi3k_species.p85_EGFR + pi3k_species.p85_Her2 + ...
                             pi3k_species.p85_Her3 + pi3k_species.p85_IGFR;

% Create figure with subplots
figure('Name', 'All PI3K Pathway Species', 'Position', [200, 200, 1800, 1400]);

% Plot 1: IGFR Module
subplot(4, 4, 1);
plot(tFine_hours, pi3k_species.IGFR, '-', 'Color', [0.2, 0.6, 0.8], 'LineWidth', 2, 'DisplayName', 'IGFR');
hold on;
plot(tFine_hours, pi3k_species.IGFR_ligand, '-', 'Color', [0.4, 0.7, 0.9], 'LineWidth', 2, 'DisplayName', 'IGFR:ligand');
plot(tFine_hours, pi3k_species.pIGFR, '-', 'Color', [0.8, 0.6, 0.2], 'LineWidth', 2.5, 'DisplayName', 'pIGFR');
xlabel('Time (hours)', 'FontSize', 10);
ylabel('Concentration', 'FontSize', 10);
title('IGFR Module', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 8);
grid on;

% Plot 2: IRS Module
subplot(4, 4, 2);
plot(tFine_hours, pi3k_species.IRS, '-', 'Color', [0.3, 0.5, 0.7], 'LineWidth', 2, 'DisplayName', 'IRS');
hold on;
plot(tFine_hours, pi3k_species.pIRS, '-', 'Color', [0.7, 0.3, 0.5], 'LineWidth', 2.5, 'DisplayName', 'pIRS');
xlabel('Time (hours)', 'FontSize', 10);
ylabel('Concentration', 'FontSize', 10);
title('IRS Module', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 8);
grid on;

% Plot 3: p85 Regulatory Subunit
subplot(4, 4, 3);
plot(tFine_hours, pi3k_species.p85_free, '-', 'Color', [0.5, 0.5, 0.5], 'LineWidth', 2.5, 'DisplayName', 'Free p85');
hold on;
plot(tFine_hours, pi3k_species.total_p85_RTK, '-', 'Color', [0.7, 0.1, 0.7], 'LineWidth', 2.5, 'DisplayName', 'Total p85:RTK');
xlabel('Time (hours)', 'FontSize', 10);
ylabel('Concentration', 'FontSize', 10);
title('p85 Regulatory Subunit', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 8);
grid on;

% Plot 4: p85:RTK Complexes
subplot(4, 4, 4);
plot(tFine_hours, pi3k_species.p85_EGFR, '-', 'Color', [0.2, 0.6, 0.8], 'LineWidth', 2, 'DisplayName', 'p85:pEGFR');
hold on;
plot(tFine_hours, pi3k_species.p85_Her2, '-', 'Color', [0.8, 0.2, 0.2], 'LineWidth', 2, 'DisplayName', 'p85:pHer2');
plot(tFine_hours, pi3k_species.p85_Her3, '-', 'Color', [0.2, 0.8, 0.2], 'LineWidth', 2, 'DisplayName', 'p85:pHer3');
plot(tFine_hours, pi3k_species.p85_IGFR, '-', 'Color', [0.8, 0.6, 0.2], 'LineWidth', 2, 'DisplayName', 'p85:pIGFR');
xlabel('Time (hours)', 'FontSize', 10);
ylabel('Concentration', 'FontSize', 10);
title('p85:RTK Complexes', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 8);
grid on;

% Plot 5: PI3K
subplot(4, 4, 5);
plot(tFine_hours, pi3k_species.PI3K_inactive, '--', 'Color', [0.6, 0.6, 0.6], 'LineWidth', 2, 'DisplayName', 'PI3K (inactive)');
hold on;
plot(tFine_hours, pi3k_species.PI3K_active, '-', 'Color', [0.2, 0.8, 0.2], 'LineWidth', 2.5, 'DisplayName', 'PI3K (active)');
xlabel('Time (hours)', 'FontSize', 10);
ylabel('Concentration', 'FontSize', 10);
title('PI3K Activation', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 8);
grid on;

% Plot 6: PIP2/PIP3
subplot(4, 4, 6);
plot(tFine_hours, pi3k_species.PIP2, '-', 'Color', [0.4, 0.6, 0.9], 'LineWidth', 2, 'DisplayName', 'PIP2');
hold on;
plot(tFine_hours, pi3k_species.PIP3, '-', 'Color', [0.9, 0.4, 0.6], 'LineWidth', 2.5, 'DisplayName', 'PIP3');
xlabel('Time (hours)', 'FontSize', 10);
ylabel('Concentration', 'FontSize', 10);
title('Phosphoinositides (PIP2/PIP3)', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 8);
grid on;

% Plot 7: AKT Module
subplot(4, 4, 7);
plot(tFine_hours, pi3k_species.AKT, '-', 'Color', [0.3, 0.5, 0.7], 'LineWidth', 2, 'DisplayName', 'AKT');
hold on;
plot(tFine_hours, pi3k_species.pAKT, '-', 'Color', [0.7, 0.2, 0.3], 'LineWidth', 2.5, 'DisplayName', 'pAKT');
xlabel('Time (hours)', 'FontSize', 10);
ylabel('Concentration', 'FontSize', 10);
title('AKT Module', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 8);
grid on;

% Plot 8: FOXO
subplot(4, 4, 8);
plot(tFine_hours, pi3k_species.FOXO, '-', 'Color', [0.6, 0.3, 0.8], 'LineWidth', 2.5, 'DisplayName', 'FOXO');
xlabel('Time (hours)', 'FontSize', 10);
ylabel('Concentration', 'FontSize', 10);
title('FOXO Transcription Factor', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 8);
grid on;

% Plot 9: mTORC Module
subplot(4, 4, 9);
plot(tFine_hours, pi3k_species.mTORC, '-', 'Color', [0.4, 0.7, 0.4], 'LineWidth', 2, 'DisplayName', 'mTORC');
hold on;
plot(tFine_hours, pi3k_species.mTORC_active, '-', 'Color', [0.2, 0.9, 0.2], 'LineWidth', 2.5, 'DisplayName', 'mTORC (active)');
xlabel('Time (hours)', 'FontSize', 10);
ylabel('Concentration', 'FontSize', 10);
title('mTORC Module', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 8);
grid on;

% Plot 10: 4EBP1 Module
subplot(4, 4, 10);
plot(tFine_hours, pi3k_species.frebp1, '-', 'Color', [0.5, 0.5, 0.5], 'LineWidth', 2, 'DisplayName', '4EBP1');
hold on;
plot(tFine_hours, pi3k_species.frebp1_inter, '--', 'Color', [0.7, 0.7, 0.7], 'LineWidth', 2, 'DisplayName', '4EBP1 (intermediate)');
plot(tFine_hours, pi3k_species.p4EBP1, '-', 'Color', [0.9, 0.5, 0.1], 'LineWidth', 2.5, 'DisplayName', 'p4EBP1');
xlabel('Time (hours)', 'FontSize', 10);
ylabel('Concentration', 'FontSize', 10);
title('4EBP1 Module', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 8);
grid on;

% Plot 11: RTK Activation Summary (for context)
subplot(4, 4, 11);
pEGFR = Y_fine(:,3);
pHer2 = Y_fine(:,6);
pHer3 = Y_fine(:,9);
plot(tFine_hours, pEGFR, '-', 'Color', [0.2, 0.6, 0.8], 'LineWidth', 2, 'DisplayName', 'pEGFR');
hold on;
plot(tFine_hours, pHer2, '-', 'Color', [0.8, 0.2, 0.2], 'LineWidth', 2, 'DisplayName', 'pHer2');
plot(tFine_hours, pHer3, '-', 'Color', [0.2, 0.8, 0.2], 'LineWidth', 2, 'DisplayName', 'pHer3');
plot(tFine_hours, pi3k_species.pIGFR, '-', 'Color', [0.8, 0.6, 0.2], 'LineWidth', 2, 'DisplayName', 'pIGFR');
xlabel('Time (hours)', 'FontSize', 10);
ylabel('Concentration', 'FontSize', 10);
title('RTK Activation (Context)', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 8);
grid on;

% Plot 12: PI3K Pathway Flow (Normalized)
subplot(4, 4, 12);
% Normalize for visualization
p85_total_norm = normit(pi3k_species.total_p85_RTK, 'max');
PI3K_active_norm = normit(pi3k_species.PI3K_active, 'max');
PIP3_norm = normit(pi3k_species.PIP3, 'max');
pAKT_norm = normit(pi3k_species.pAKT, 'max');
mTORC_active_norm = normit(pi3k_species.mTORC_active, 'max');
p4EBP1_norm = normit(pi3k_species.p4EBP1, 'max');

plot(tFine_hours, p85_total_norm, '-', 'Color', [0.7, 0.1, 0.7], 'LineWidth', 2, 'DisplayName', 'p85:RTK');
hold on;
plot(tFine_hours, PI3K_active_norm, '-', 'Color', [0.2, 0.8, 0.2], 'LineWidth', 2, 'DisplayName', 'PI3K*');
plot(tFine_hours, PIP3_norm, '-', 'Color', [0.9, 0.4, 0.6], 'LineWidth', 2, 'DisplayName', 'PIP3');
plot(tFine_hours, pAKT_norm, '-', 'Color', [0.7, 0.2, 0.3], 'LineWidth', 2, 'DisplayName', 'pAKT');
plot(tFine_hours, mTORC_active_norm, '-', 'Color', [0.2, 0.9, 0.2], 'LineWidth', 2, 'DisplayName', 'mTORC*');
plot(tFine_hours, p4EBP1_norm, '-', 'Color', [0.9, 0.5, 0.1], 'LineWidth', 2, 'DisplayName', 'p4EBP1');
xlabel('Time (hours)', 'FontSize', 10);
ylabel('Normalized Concentration', 'FontSize', 10);
title('PI3K Pathway Flow (Normalized)', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 7);
grid on;

% Plot 13: p85 Distribution
subplot(4, 4, 13);
p85_total_all = pi3k_species.p85_free + pi3k_species.total_p85_RTK;
p85_free_pct = (pi3k_species.p85_free ./ (p85_total_all + eps)) * 100;
p85_bound_pct = (pi3k_species.total_p85_RTK ./ (p85_total_all + eps)) * 100;

plot(tFine_hours, p85_free_pct, '-', 'Color', [0.5, 0.5, 0.5], 'LineWidth', 2.5, 'DisplayName', 'Free p85 (%)');
hold on;
plot(tFine_hours, p85_bound_pct, '-', 'Color', [0.7, 0.1, 0.7], 'LineWidth', 2.5, 'DisplayName', 'Bound p85 (%)');
xlabel('Time (hours)', 'FontSize', 10);
ylabel('Percentage (%)', 'FontSize', 10);
title('p85 Distribution', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 8);
ylim([0, 100]);
grid on;

% Plot 14: PIP2/PIP3 Ratio
subplot(4, 4, 14);
PIP_ratio = pi3k_species.PIP3 ./ (pi3k_species.PIP2 + eps);
plot(tFine_hours, PIP_ratio, '-', 'Color', [0.6, 0.3, 0.8], 'LineWidth', 2.5, 'DisplayName', 'PIP3/PIP2 Ratio');
xlabel('Time (hours)', 'FontSize', 10);
ylabel('Ratio', 'FontSize', 10);
title('PIP3/PIP2 Ratio', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 8);
grid on;

% Plot 15: AKT/mTORC/4EBP1 Cascade (Normalized)
subplot(4, 4, 15);
plot(tFine_hours, pAKT_norm, '-', 'Color', [0.7, 0.2, 0.3], 'LineWidth', 2.5, 'DisplayName', 'pAKT');
hold on;
plot(tFine_hours, mTORC_active_norm, '-', 'Color', [0.2, 0.9, 0.2], 'LineWidth', 2.5, 'DisplayName', 'mTORC*');
plot(tFine_hours, p4EBP1_norm, '-', 'Color', [0.9, 0.5, 0.1], 'LineWidth', 2.5, 'DisplayName', 'p4EBP1');
xlabel('Time (hours)', 'FontSize', 10);
ylabel('Normalized Concentration', 'FontSize', 10);
title('AKT → mTORC → 4EBP1 Cascade', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 8);
grid on;

% Plot 16: Summary Statistics
subplot(4, 4, 16);
axis off;
[~, idx_p85] = max(pi3k_species.total_p85_RTK);
[~, idx_PI3K] = max(pi3k_species.PI3K_active);
[~, idx_pAKT] = max(pi3k_species.pAKT);
[~, idx_p4EBP1] = max(pi3k_species.p4EBP1);
stats_text = {
    sprintf('PI3K Pathway Summary:');
    sprintf('');
    sprintf('Max p85:RTK: %.4f', max(pi3k_species.total_p85_RTK));
    sprintf('Max PI3K*: %.4f', max(pi3k_species.PI3K_active));
    sprintf('Max PIP3: %.4f', max(pi3k_species.PIP3));
    sprintf('Max pAKT: %.4f', max(pi3k_species.pAKT));
    sprintf('Max mTORC*: %.4f', max(pi3k_species.mTORC_active));
    sprintf('Max p4EBP1: %.4f', max(pi3k_species.p4EBP1));
    sprintf('');
    sprintf('Time to Peak:');
    sprintf('  p85:RTK: %.2f h', tFine_hours(idx_p85));
    sprintf('  PI3K*: %.2f h', tFine_hours(idx_PI3K));
    sprintf('  pAKT: %.2f h', tFine_hours(idx_pAKT));
    sprintf('  p4EBP1: %.2f h', tFine_hours(idx_p4EBP1));
};

text(0.1, 0.5, stats_text, 'FontSize', 10, 'VerticalAlignment', 'middle');
title('PI3K Pathway Summary', 'FontSize', 12, 'FontWeight', 'bold');

sgtitle('All PI3K Pathway Species', 'FontSize', 16, 'FontWeight', 'bold');

fprintf('Figure 4: All PI3K Pathway Species generated successfully.\n\n');

%% ============================================================================
% SECTION 8: ODE FUNCTION
% ============================================================================

function dydt = Mapk_ODE(t, y, p)
    % MAPK/PI3K Pathway ODE System with Vemurafenib and Paradoxical Activation
    % EXPANDED with Her2, Her3, and p85-mediated PI3K recruitment
    %
    % Inputs:
    %   t - time (seconds)
    %   y - state vector (62 species)
    %   p - parameter vector (60 parameters)
    %
    % Outputs:
    %   dydt - derivatives of all species
    %
    % State Vector (y):
    %   [1-3]   EGFR module: EGFR, EGFR:ligand, pEGFR
    %   [4-6]   Her2 module: Her2, Her2:ligand, pHer2 (NEW)
    %   [7-9]   Her3 module: Her3, Her3:ligand, pHer3 (NEW)
    %   [10-12] Shc module: Shc, pEGFR:Shc, pShc
    %   [13-14] Grb2/SOS: Grb2, pShc:Grb2:SOS
    %   [15-16] HRAS: HRAS-GDP, HRAS-GTP
    %   [17-18] NRAS: NRAS-GDP, NRAS-GTP
    %   [19-21] KRAS: KRAS-GDP, intermediate, KRAS-GTP
    %   [22-23] CRAF: CRAF, pCRAF
    %   [24-25] BRAF: BRAF, BRAF^P (BRAF^V600E*)
    %   [26-27] MEK: MEK, pMEK
    %   [28-29] ERK: ERK, pERK
    %   [30-31] DUSP: mDUSP, DUSP
    %   [32-33] SPRY: mSPRY, SPRY
    %   [34-37] Degradation trackers
    %   [38-40] IGFR module: IGFR, IGFR:ligand, pIGFR
    %   [41-42] IRS: IRS, pIRS
    %   [43]    p85: p85 regulatory subunit (free) (NEW)
    %   [44-47] p85 complexes: p85:pEGFR, p85:pHer2, p85:pHer3, p85:pIGFR (NEW)
    %   [48-49] PI3K: PI3K (p110), PI3K_active
    %   [50-51] PIP: PIP2, PIP3
    %   [52-53] AKT: AKT, pAKT
    %   [54]    FOXO
    %   [55-56] mTORC: mTORC, mTORC_active
    %   [57-59] 4EBP1 module: 4EBP1, intermediate, p4EBP1
    %   [60-61] KSR: KSR, pKSR
    %   [62]    BRAF-WT:CRAF dimer
    
    % Unpack parameters
    ka1 = p(1);  kr1 = p(2);  kc1 = p(3);
    kpCraf = p(4); kpMek = p(5); kpErk = p(6);
    kDegradEgfr = p(7); kErkInbEgfr = p(8); kShcDephos = p(9); kptpDeg = p(10); kGrb2CombShc = p(11);
    kSprtyInbGrb2 = p(12); kSosCombGrb2 = p(13); kErkPhosSos = p(14);
    kErkPhosPcraf = p(15); kPcrafDegrad = p(16); kErkPhosMek = p(17); kMekDegrad = p(18);
    kDuspInbErk = p(19); kErkDeg = p(20); kinbBraf = p(21); kDuspStop = p(22); kDusps = p(23);
    kSproutyForm = p(24); kSprtyComeDown = p(25); kdegrad = p(26); km_Sprty_decay = p(27); km_Dusp = p(28); km_Sprty = p(29);
    kErkDephos = p(30); kDuspDeg = p(31);
    kHer2_act = p(32); kHer3_act = p(33);  % NEW: Her2/Her3 activation (updated indices)
    k_p85_bind_EGFR = p(34); k_p85_bind_Her2 = p(35); k_p85_bind_Her3 = p(36); k_p85_bind_IGFR = p(37);  % NEW: p85 binding (updated indices)
    k_p85_unbind = p(38); k_PI3K_recruit = p(39);  % NEW: p85 unbinding and PI3K recruitment (updated indices)
    kMTOR_Feedback = p(40);
    k_PIP2_to_PIP3 = p(41); k_PTEN = p(42);  % NEW: PIP2/PIP3 conversion parameters
    kAkt = p(43); kdegradAKT = p(44);
    kb1 = p(45); k43b1 = p(46); k4ebp1 = p(47); k_4EBP1_dephos = p(48);  % 4EBP1 parameters
    kKSRphos = p(49); kKSRdephos = p(50);
    kMekByBraf = p(51); kMekByCraf = p(52); kMekByKSR = p(53);
    Tram = p(54); K_tram_RAF = p(55); K_tram_KSR = p(56); n_tram = p(57);
    Vemurafenib = p(58); kDimerForm = p(59); kDimerDissoc = p(60);
    kParadoxCRAF = p(61); IC50_vem = p(62); Hill_n_vem = p(63);
    
    % Initialize derivatives
    dydt = zeros(63, 1);  % Updated from 62 to 63 (added PTEN)
    
    % ========================================================================
    % MODULE 1: RTK SIGNALING (EXPANDED: EGFR, Her2, Her3)
    % ========================================================================
    % EGFR
    dydt(1) = -ka1*y(1) + kr1*y(2);                                    % EGFR
    dydt(2) = ka1*y(1) - kr1*y(2) - kc1*y(2);                          % EGFR:ligand
    dydt(3) = kc1*y(2) - kDegradEgfr*y(3) - kErkInbEgfr*y(29)*y(3);    % pEGFR (with ERK feedback)
    
    % Her2 (NEW) - Enhanced activation when pERK is low (adaptive response)
    % Biology: ERK normally inhibits RTK signaling; loss of ERK → RTK upregulation
    % Adaptive factor: 1.0 (baseline) + 5.0*(1-pERK), so 6x max when pERK=0
    kHer2_adaptive = kHer2_act * (1.0 + 5.0 * max(0, 1 - y(29)));
    dydt(4) = -kHer2_adaptive*y(4) + kr1*y(5);                              % Her2
    dydt(5) = kHer2_adaptive*y(4) - kr1*y(5) - kc1*y(5);                    % Her2:ligand
    dydt(6) = kc1*y(5) - kDegradEgfr*y(6) - kErkInbEgfr*y(29)*y(6);    % pHer2 (with ERK feedback)
    
    % Her3 (NEW) - Enhanced activation when pERK is low (adaptive response)
    kHer3_adaptive = kHer3_act * (1.0 + 5.0 * max(0, 1 - y(29)));
    dydt(7) = -kHer3_adaptive*y(7) + kr1*y(8);                              % Her3
    dydt(8) = kHer3_adaptive*y(7) - kr1*y(8) - kc1*y(8);                    % Her3:ligand
    dydt(9) = kc1*y(8) - kDegradEgfr*y(9) - kErkInbEgfr*y(29)*y(9);    % pHer3 (with ERK feedback)
    
    % ========================================================================
    % MODULE 2: Shc/Grb2/SOS SIGNALING
    % ========================================================================
    dydt(10) = -ka1*y(3)*y(10);                                        % Shc
    dydt(11) = ka1*y(3)*y(10) - kShcDephos*y(12)*y(11);                % pEGFR:Shc
    dydt(12) = -kptpDeg*y(11)*y(12);                                   % pShc
    
    dydt(13) = kGrb2CombShc*y(11)*y(3) - kSprtyInbGrb2*y(27)*y(13);    % Grb2 (with SPRY feedback)
    dydt(14) = kSosCombGrb2*y(13)*y(11) - kErkPhosSos*y(25)*y(14);     % pShc:Grb2:SOS (with ERK feedback)
    
    % ========================================================================
    % MODULE 3: RAS ACTIVATION
    % ========================================================================
    dydt(15) = -ka1*y(14)*y(15);                                       % HRAS-GDP
    dydt(16) = ka1*y(14)*y(15);                                        % HRAS-GTP
    dydt(17) = -ka1*y(14)*y(17);                                       % NRAS-GDP
    dydt(18) = ka1*y(14)*y(17);                                        % NRAS-GTP
    dydt(19) = -ka1*y(14)*y(19);                                       % KRAS-GDP
    dydt(20) = ka1*y(14)*y(19) - ka1*y(20)*y(21);                      % KRAS-GTP
    dydt(21) = -ka1*y(20)*y(21);                                       % KRAS intermediate
    
    % ========================================================================
    % MODULE 4: RAF SIGNALING WITH PARADOXICAL ACTIVATION
    % ========================================================================
    
    % Vemurafenib inhibition of BRAF^V600E (Hill equation)
    IC50_n = IC50_vem^Hill_n_vem;
    Vem_n = Vemurafenib^Hill_n_vem;
    kBRAF_eff = ka1 * IC50_n / (IC50_n + Vem_n + eps);  % Effective BRAF activation (inhibited)
    
    % Paradoxical activation: vemurafenib promotes BRAF-WT*:CRAF dimer → CRAF* activation
    paradox_activation = kParadoxCRAF * Vemurafenib * y(62);  % y(62) = BRAF-WT:CRAF dimer
    
    % CRAF dynamics
    dydt(22) = -kpCraf*y(20)*y(22) + kErkPhosPcraf*y(29)*y(23) + kPcrafDegrad*y(23)*y(36) ...
               - kDimerForm*y(25)*y(22)*Vemurafenib + kDimerDissoc*y(62);  % CRAF (used in dimer)
    dydt(23) = kpCraf*y(20)*y(22) - kErkPhosPcraf*y(29)*y(23) - kPcrafDegrad*y(23)*y(36) ...
               + paradox_activation;  % pCRAF (with paradoxical activation)
    
    % BRAF dynamics (with vemurafenib inhibition)
    dydt(24) = -kBRAF_eff*y(24)*y(20) - kDimerForm*y(25)*y(22)*Vemurafenib + kDimerDissoc*y(62);
    dydt(25) = kBRAF_eff*y(24)*y(20) - kinbBraf*y(25) - kDimerForm*y(25)*y(22)*Vemurafenib + kDimerDissoc*y(62);
    
    % BRAF-WT:CRAF dimer dynamics (y(62) = dimer complex)
    dydt(62) = kDimerForm*y(25)*y(22)*Vemurafenib - kDimerDissoc*y(62) - kPcrafDegrad*y(62)*y(36);
    
    % ========================================================================
    % MODULE 5: MEK PHOSPHORYLATION
    % ========================================================================
    % MEK can be phosphorylated by CRAF, BRAF, or KSR
    raf_to_mek = (kpMek*y(23) + kMekByBraf*y(25) + kMekByCraf*y(23));
    ksr_to_mek = (kMekByKSR * y(61));
    
    dydt(26) = -(raf_to_mek + ksr_to_mek)*y(26) + kErkPhosMek*y(29)*y(27) + kMekDegrad*y(27)*y(35);
    dydt(27) = (raf_to_mek + ksr_to_mek)*y(26) - kErkPhosMek*y(29)*y(27) - kMekDegrad*y(27)*y(35);
    
    % ========================================================================
    % MODULE 6: ERK PHOSPHORYLATION
    % ========================================================================
    dydt(28) = -kpErk*y(27)*y(28) + kDuspInbErk*y(31)*y(29) + kErkDeg*y(29)*y(34) + kErkDephos*y(31)*y(29);
    dydt(29) = kpErk*y(27)*y(28) - kDuspInbErk*y(31)*y(29) - kErkDeg*y(29)*y(34) - kErkDephos*y(31)*y(29);
    
    % ========================================================================
    % MODULE 7: DUSP FEEDBACK
    % ========================================================================
    dydt(30) = km_Dusp*y(29)/(1 + (km_Dusp/kDusps)*y(29)) - kDuspStop*y(30)*y(37) - kDuspDeg*y(30)*y(29);
    dydt(31) = -kDuspStop*y(30)*y(31);
    
    % ========================================================================
    % MODULE 8: SPRY FEEDBACK
    % ========================================================================
    dydt(32) = km_Sprty*y(29)/(1 + (km_Sprty/kSproutyForm)*y(29)) - kSprtyComeDown*y(32)*y(33);
    dydt(33) = -kSprtyComeDown*y(32)*y(33);
    
    % ========================================================================
    % MODULE 9: DEGRADATION TRACKERS
    % ========================================================================
    dydt(34) = -kErkDeg*y(29)*y(34);      % pERK degradation tracker
    dydt(35) = -kMekDegrad*y(27)*y(35);   % pMEK degradation tracker
    dydt(36) = -kPcrafDegrad*y(23)*y(36); % pCRAF degradation tracker
    dydt(37) = -kDuspStop*y(30)*y(37);    % DUSP stop tracker
    
    % ========================================================================
    % MODULE 10: PI3K/AKT/mTOR SIGNALING
    % ========================================================================
    % IGFR module (with adaptive upregulation)
    % Biology: When MAPK is blocked, cells upregulate insulin/IGF signaling as survival mechanism
    % Adaptive factor: 1.0 + 3.0*(1-pERK), so 4x max when pERK=0
    kIGFR_adaptive = ka1 * (1.0 + 3.0 * max(0, 1 - y(29)));
    
    dydt(38) = -kIGFR_adaptive*y(38) + kr1*y(39);
    dydt(39) = kIGFR_adaptive*y(38) - kr1*y(39) - kc1*y(39);
    dydt(40) = kc1*y(39) - kErkInbEgfr*y(29)*y(40);  % pIGFR
    
    % IRS module (with mTOR S6K negative feedback)
    % Biology: Active mTORC1→S6K phosphorylates IRS1 on inhibitory residues
    % When mTORC is low, IRS1 is de-repressed → enhanced PI3K activation
    % Feedback strength: 0.8 means 80% inhibition at max mTOR
    k_IRS_eff = ka1 * (1.0 - 0.8 * y(56));  % Strongly reduced by active mTORC (y(56))
    
    dydt(41) = -k_IRS_eff*y(3)*y(41);  % IRS1 phosphorylation by pEGFR
    dydt(42) = k_IRS_eff*y(3)*y(41);   % pIRS1 (enhanced when mTORC is low)
    
    % p85 regulatory subunit binding to phosphotyrosine sites
    % p85 binds to: pEGFR (y(3)), pHer2 (y(6)), pHer3 (y(9)), pIGFR (y(40))
    % Free p85 (y(43))
    dydt(43) = -k_p85_bind_EGFR*y(3)*y(43) ...
               - k_p85_bind_Her2*y(6)*y(43) ...
               - k_p85_bind_Her3*y(9)*y(43) ...
               - k_p85_bind_IGFR*y(40)*y(43) ...
               + k_p85_unbind*(y(44) + y(45) + y(46) + y(47));
    
    % p85:RTK complexes
    dydt(44) = k_p85_bind_EGFR*y(3)*y(43) - k_p85_unbind*y(44);  % p85:pEGFR
    dydt(45) = k_p85_bind_Her2*y(6)*y(43) - k_p85_unbind*y(45);  % p85:pHer2
    dydt(46) = k_p85_bind_Her3*y(9)*y(43) - k_p85_unbind*y(46);  % p85:pHer3
    dydt(47) = k_p85_bind_IGFR*y(40)*y(43) - k_p85_unbind*y(47); % p85:pIGFR
    
    % PI3K module - recruitment by p85:RTK complexes
    % Total p85:RTK complexes recruit PI3K (p110 catalytic subunit)
    total_p85_RTK = y(44) + y(45) + y(46) + y(47);
    
    dydt(48) = -k_PI3K_recruit * total_p85_RTK * y(48) ...
               + kMTOR_Feedback * y(56) * y(49);  % PI3K (p110, inactive)
    dydt(49) = k_PI3K_recruit * total_p85_RTK * y(48) ...
               - kMTOR_Feedback * y(56) * y(49);  % PI3K_active (recruited to membrane)
    
    % PIP module - PIP2 (y(50)) and PIP3 (y(51))
    % PIP2 → PIP3 conversion by active PI3K
    % PIP3 → PIP2 dephosphorylation by PTEN
    dydt(50) = -k_PIP2_to_PIP3 * y(49) * y(50) ...  % PIP2 consumed by active PI3K
               + k_PTEN * y(51);                     % PIP3 dephosphorylated back to PIP2 by PTEN
    dydt(51) = k_PIP2_to_PIP3 * y(49) * y(50) ...   % PIP2 converted to PIP3 by active PI3K
               - k_PTEN * y(51);                     % PIP3 dephosphorylated by PTEN
    
    % AKT module - AKT (y(52)) and pAKT (y(53))
    % AKT activation by PIP3 (simplified: PIP3 recruits and activates AKT)
    % In reality: PIP3 recruits AKT to membrane, PDK1 phosphorylates Thr308, mTORC2 phosphorylates Ser473
    dydt(52) = -kAkt * y(51) * y(52) ...      % AKT consumed by PIP3-mediated activation
               + kdegradAKT * y(53);           % pAKT dephosphorylated back to AKT
    dydt(53) = kAkt * y(51) * y(52) ...       % AKT activated by PIP3
               - kdegradAKT * y(53);           % pAKT dephosphorylated/degraded
    
    % FOXO module - FOXO (y(54))
    % FOXO is inactivated by pAKT (simplified model)
    dydt(54) = (1 - y(53)) * ka1 / (1 + (y(54) / 15e-5));  % FOXO dynamics (inhibited by pAKT)
    
    % mTORC module - mTORC (y(55)) and mTORC_active (y(56))
    % mTORC activation by pAKT
    dydt(55) = -ka1 * y(53) * y(55) + kdegrad * y(56);  % mTORC (inactive)
    dydt(56) = ka1 * y(53) * y(55) - kdegrad * y(56);   % mTORC_active (activated by pAKT)
    
    % 4EBP1 module - 4EBP1 (y(57)), intermediate (y(58)), p4EBP1 (y(59))
    % 4EBP1 phosphorylation by active mTORC
    dydt(57) = -k4ebp1 * y(56) * y(57) + kb1 * y(58) + k_4EBP1_dephos * y(59);  % 4EBP1 (unphosphorylated) - receives dephosphorylated p4EBP1
    dydt(58) = k4ebp1 * y(56) * y(57) - kb1 * y(58) - k43b1 * y(58);  % 4EBP1 intermediate
    dydt(59) = k43b1 * y(58) - k_4EBP1_dephos * y(59);  % p4EBP1 (fully phosphorylated) - can be dephosphorylated
    
    % ========================================================================
    % MODULE 11: KSR SCAFFOLD
    % ========================================================================
    % KSR (y(60)) and pKSR (y(61))
    % KSR phosphorylation by RAS-GTP (HRAS-GTP: y(16), NRAS-GTP: y(18), KRAS-GTP: y(21))
    dydt(60) = -kKSRphos * (y(16) + y(18) + y(21)) * y(60) + kKSRdephos * y(61);  % KSR
    dydt(61) = kKSRphos * (y(16) + y(18) + y(21)) * y(60) - kKSRdephos * y(61);   % pKSR
    
    
    % ========================================================================
    % MODULE 12: PTEN DYNAMICS (NEW - y(63))
    % ========================================================================
    % PTEN is regulated by:
    %   - FOXO transcriptional activation (positive)
    %   - pAKT-mediated inhibition (negative feedback)
    % Biological rationale: FOXO activates PTEN transcription,
    % but pAKT phosphorylates and inactivates PTEN
    k_PTEN_basal = 5e-6;     % Basal PTEN production
    k_PTEN_FOXO = 10e-5;     % FOXO-mediated PTEN upregulation
    k_PTEN_degrad = 5e-6;    % PTEN degradation
    k_PTEN_AKT_inhib = 20e-5; % pAKT-mediated PTEN inhibition
    
    dydt(63) = k_PTEN_basal + k_PTEN_FOXO * y(54) ...  % Basal + FOXO activation
               - k_PTEN_degrad * y(63) ...              % Degradation
               - k_PTEN_AKT_inhib * y(53) * y(63);      % pAKT inhibits PTEN
    
    % Note: dydt(62) for BRAF-WT:CRAF dimer is defined above in RAF module
end

%% ============================================================================
% SECTION 9: OBJECTIVE FUNCTION FOR OPTIMIZATION
% ============================================================================

function err = objectiveFunction_all(p, timeStamps_seconds, expData_norm, y0)
    % Objective function for parameter optimization
    %
    % Inputs:
    %   p - parameter vector (52 parameters)
    %   timeStamps_seconds - experimental time points (seconds)
    %   expData_norm - normalized experimental data (structure)
    %   y0 - initial conditions
    %
    % Outputs:
    %   err - sum of squared residuals
    
    % Integrate ODE system at experimental time points
    % Integrate ODE system at experimental time points
    % Use ode15s for stiff systems
    try
        [T, Y] = ode15s(@(t,y) Mapk_ODE(t, y, p), timeStamps_seconds, y0);
        
        % Check if integration completed successfully
        if length(T) ~= length(timeStamps_seconds)
            % ODE solver failed to integrate up to the final time point
            % Return a large penalty
            err = 1e9;
            return;
        end
    catch
        % Solver crashed
        err = 1e9;
        return;
    end
    
    % Extract model outputs (updated indices for new species order)
    m_pEGFR = Y(:,3);
    m_panRAS = Y(:,16) + Y(:,18) + Y(:,20); % panRAS = HRAS-GTP + NRAS-GTP + KRAS-GTP
    m_pCRAF = Y(:,23);  % Updated: pCRAF is now y(23)
    m_pMEK = Y(:,27);   % Updated: pMEK is now y(27)
    m_pERK = Y(:,29);   % Updated: pERK is now y(29)
    m_DUSP = Y(:,31);   % Updated: DUSP is now y(31)
    m_pAKT = Y(:,53);   % Updated: pAKT is now y(53)
    m_p4EBP1 = Y(:,59); % Updated: p4EBP1 is now y(59)
    % m_RAS_GTP = Y(:,20);  % Removed from optimization (KRAS-GTP is now y(20))
    
    % Normalization function
    norm = @(v, mode) v ./ max(eps, (mode=="first")*v(1) + (mode=="last")*v(end) + (mode=="max")*max(v));
    
    % Normalize model outputs
    m_pEGFR = norm(m_pEGFR, 'max');
    m_panRAS = norm(m_panRAS, 'last');
    m_pMEK = norm(m_pMEK, 'first');
    m_pERK = norm(m_pERK, 'first');
    m_DUSP = norm(m_DUSP, 'max');
    m_pCRAF = norm(m_pCRAF, 'max');
    m_pAKT = norm(m_pAKT, 'max');
    m_p4EBP1 = norm(m_p4EBP1, 'max');  % p4EBP1
    % m_RAS_GTP = norm(m_RAS_GTP, 'last');  % Removed from optimization
    
    % Calculate weighted sum of squared residuals
    % Calculate weighted sum of squared residuals
    w = struct('EGFR', 1, 'MEK', 1, 'ERK', 3, 'DUSP', 3, 'CRAF', 1, 'AKT', 1, 'p4EBP1', 1, 'panRAS', 1);
    
    err = 0;
    err = err + w.EGFR * sum((m_pEGFR - expData_norm.pEGFR(:)).^2);
    %err = err + w.panRAS * sum((m_panRAS - expData_norm.panRAS(:)).^2);
    err = err + w.MEK * sum((m_pMEK - expData_norm.pMEK(:)).^2);
    err = err + w.ERK * sum((m_pERK - expData_norm.pERK(:)).^2);
    err = err + w.DUSP * sum((m_DUSP - expData_norm.DUSP(:)).^2);
    err = err + w.CRAF * sum((m_pCRAF - expData_norm.pCRAF(:)).^2);
    err = err + w.AKT * sum((m_pAKT - expData_norm.pAKT(:)).^2);
    err = err + w.p4EBP1 * sum((m_p4EBP1 - expData_norm.p4ebp1(:)).^2);
end

fprintf('═══════════════════════════════════════════════════════════════════════════\n');
fprintf('   SIMULATION COMPLETED SUCCESSFULLY!\n');
fprintf('═══════════════════════════════════════════════════════════════════════════\n\n');
% Save trained parameters for transfer learning
save('trained_Vem_params.mat', 'optimizedParams');
fprintf('Trained parameters saved to trained_Vem_params.mat\n');
