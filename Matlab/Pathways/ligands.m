% =============================================================================
% EGFR-HER3 RECEPTOR SIGNALING MODEL
% Ordinary Differential Equations (ODEs) with Mass-Action Kinetics
% =============================================================================
%
% Model Description:
% This script models the EGFR-HER3 receptor signaling module including:
% - Ligand binding to EGFR
% - EGFR homodimerization and phosphorylation
% - EGFR-HER3 heterodimerization and phosphorylation
% - Dephosphorylation reactions
%
% Species:
% E   = unbound EGFR monomer
% L   = extracellular EGF ligand (constant)
% EL  = EGF-bound EGFR monomer
% EE  = EGFR-EGFR dimer
% EEp = phosphorylated EGFR-EGFR dimer
% H   = unbound HER3 monomer
% EH  = EGFR-HER3 heterodimer
% EHp = phosphorylated EGFR-HER3 heterodimer
%
% =============================================================================

clear all;
close all;
clc;

%% ============================================================================
% PARAMETERS
% ============================================================================

% Ligand binding to EGFR
kon = 0.01;      % nM^-1 min^-1
koff = 0.01;     % min^-1

% EGFR homodimerization
kd = 0.02;       % nM^-1 min^-1
kdm = 0.05;      % min^-1

% Phosphorylation
kphos_EE = 0.5;  % min^-1 (EGFR dimer phosphorylation)

% EGFR-HER3 heterodimerization
kd_EH = 0.01;    % nM^-1 min^-1
kdm_EH = 0.05;   % min^-1

% HER3 phosphorylation
kphos_EH = 0.3;  % min^-1

% Dephosphorylation
kdeph = 0.1;     % min^-1

% Ligand concentration (constant)
L = 10;          % nM

% Store parameters in structure for passing to ODE function
params.kon = kon;
params.koff = koff;
params.kd = kd;
params.kdm = kdm;
params.kphos_EE = kphos_EE;
params.kd_EH = kd_EH;
params.kdm_EH = kdm_EH;
params.kphos_EH = kphos_EH;
params.kdeph = kdeph;
params.L = L;

%% ============================================================================
% INITIAL CONDITIONS
% ============================================================================

E0 = 10;    % nM - unbound EGFR monomer
EL0 = 0;     % nM - EGF-bound EGFR monomer (initially zero)
EE0 = 0;     % nM - EGFR-EGFR dimer (initially zero)
EEp0 = 0;    % nM - phosphorylated EGFR-EGFR dimer (initially zero)
H0 = 5;      % nM - unbound HER3 monomer
EH0 = 0;     % nM - EGFR-HER3 heterodimer (initially zero)
EHp0 = 0;    % nM - phosphorylated EGFR-HER3 heterodimer (initially zero)

% Initial state vector: [E, EL, EE, EEp, H, EH, EHp]
y0 = [E0; EL0; EE0; EEp0; H0; EH0; EHp0];

%% ============================================================================
% TIME SPAN
% ============================================================================

tspan = [0, 200];  % minutes

%% ============================================================================
% SOLVE ODE SYSTEM
% ============================================================================

fprintf('Solving ODE system...\n');
options = odeset('RelTol', 1e-6, 'AbsTol', 1e-8);
[t, y] = ode45(@(t,y) egfr_her3_odes(t, y, params), tspan, y0, options);

% Extract species concentrations
E = y(:, 1);
EL = y(:, 2);
EE = y(:, 3);
EEp = y(:, 4);
H = y(:, 5);
EH = y(:, 6);
EHp = y(:, 7);

fprintf('Simulation completed.\n');
%% ============================================================================
% SAVE pEGFR (EEp) FOR USE IN ADAPTOR MODEL
% ============================================================================

% Save time and EEp (phosphorylated EGFR) for use in adaptor model
save('egfr_output.mat', 't', 'EEp');
fprintf('Saved pEGFR (EEp) values to egfr_output.mat\n');
fprintf('  Time points: %d\n', length(t));
fprintf('  EEp range: %.4f to %.4f nM\n', min(EEp), max(EEp));

%% ============================================================================
% PLOTTING
% ============================================================================

% Create figure for main plots
figure('Position', [100, 100, 1200, 800]);

% Plot 1: Phosphorylated EGFR (EEp)
subplot(2, 2, 1);
plot(t, EEp, 'b-', 'LineWidth', 2);
xlabel('Time (minutes)', 'FontSize', 12);
ylabel('Concentration (nM)', 'FontSize', 12);
title('Phosphorylated EGFR-EGFR Dimer (EEp)', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
xlim([0, 200]);

% Plot 2: Phosphorylated HER3 (EHp)
subplot(2, 2, 2);
plot(t, EHp, 'r-', 'LineWidth', 2);
xlabel('Time (minutes)', 'FontSize', 12);
ylabel('Concentration (nM)', 'FontSize', 12);
title('Phosphorylated EGFR-HER3 Heterodimer (EHp)', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
xlim([0, 200]);

% Plot 3: Both phosphorylated species together
subplot(2, 2, 3);
plot(t, EEp, 'b-', 'LineWidth', 2, 'DisplayName', 'EEp (Phosphorylated EGFR)');
hold on;
plot(t, EHp, 'r-', 'LineWidth', 2, 'DisplayName', 'EHp (Phosphorylated HER3)');
xlabel('Time (minutes)', 'FontSize', 12);
ylabel('Concentration (nM)', 'FontSize', 12);
title('Phosphorylated Species Comparison', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 10);
grid on;
xlim([0, 200]);
hold off;

% Plot 4: All species overview
subplot(2, 2, 4);
plot(t, E, 'k-', 'LineWidth', 1.5, 'DisplayName', 'E (EGFR)');
hold on;
plot(t, EL, 'g-', 'LineWidth', 1.5, 'DisplayName', 'EL (EGFR-Ligand)');
plot(t, EE, 'c-', 'LineWidth', 1.5, 'DisplayName', 'EE (EGFR Dimer)');
plot(t, EEp, 'b-', 'LineWidth', 1.5, 'DisplayName', 'EEp (Phospho EGFR)');
plot(t, H, 'm-', 'LineWidth', 1.5, 'DisplayName', 'H (HER3)');
plot(t, EH, 'y-', 'LineWidth', 1.5, 'DisplayName', 'EH (Heterodimer)');
plot(t, EHp, 'r-', 'LineWidth', 1.5, 'DisplayName', 'EHp (Phospho HER3)');
xlabel('Time (minutes)', 'FontSize', 12);
ylabel('Concentration (nM)', 'FontSize', 12);
title('All Species Time Courses', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 9);
grid on;
xlim([0, 200]);
hold off;

sgtitle('EGFR-HER3 Receptor Signaling Model', 'FontSize', 16, 'FontWeight', 'bold');



%% ============================================================================
% STEADY-STATE ANALYSIS
% ============================================================================

fprintf('\n=== Steady-State Concentrations (at t = 200 min) ===\n');
fprintf('E (EGFR monomer):        %.4f nM\n', E(end));
fprintf('EL (EGFR-Ligand):        %.4f nM\n', EL(end));
fprintf('EE (EGFR dimer):         %.4f nM\n', EE(end));
fprintf('EEp (Phospho EGFR):      %.4f nM\n', EEp(end));
fprintf('H (HER3 monomer):        %.4f nM\n', H(end));
fprintf('EH (Heterodimer):        %.4f nM\n', EH(end));
fprintf('EHp (Phospho HER3):      %.4f nM\n', EHp(end));

fprintf('\n=== Total EGFR Conservation Check ===\n');
total_EGFR = E(end) + EL(end) + 2*EE(end) + 2*EEp(end) + EH(end) + EHp(end);
fprintf('Total EGFR (conserved):   %.4f nM (should be ~%.1f nM)\n', total_EGFR, E0);

fprintf('\n=== Total HER3 Conservation Check ===\n');
total_HER3 = H(end) + EH(end) + EHp(end);
fprintf('Total HER3 (conserved):  %.4f nM (should be ~%.1f nM)\n', total_HER3, H0);

%% ============================================================================
% ODE FUNCTION
% ============================================================================

function dydt = egfr_her3_odes(t, y, p)
    % ODE system for EGFR-HER3 receptor signaling
    % y = [E, EL, EE, EEp, H, EH, EHp]
    % p = parameters structure
    
    % Extract species
    E = y(1);
    EL = y(2);
    EE = y(3);
    EEp = y(4);
    H = y(5);
    EH = y(6);
    EHp = y(7);
    
    % Ligand concentration (constant)
    L = p.L;
    
    % Reaction rates
    % 1. E + L ↔ EL
    r1_forward = p.kon * E * L;
    r1_reverse = p.koff * EL;
    
    % 2. EL + EL ↔ EE
    r2_forward = p.kd * EL * EL;
    r2_reverse = p.kdm * EE;
    
    % 3. EE → EEp (phosphorylation)
    r3 = p.kphos_EE * EE;
    
    % 4. EEp → EE (dephosphorylation)
    r4 = p.kdeph * EEp;
    
    % 5. EL + H ↔ EH
    r5_forward = p.kd_EH * EL * H;
    r5_reverse = p.kdm_EH * EH;
    
    % 6. EH → EHp (phosphorylation)
    r6 = p.kphos_EH * EH;
    
    % 7. EHp → EH (dephosphorylation)
    r7 = p.kdeph * EHp;
    
    % ODEs for each species
    dE_dt = -r1_forward + r1_reverse;
    
    dEL_dt = r1_forward - r1_reverse ...
             - 2*r2_forward + 2*r2_reverse ...
             - r5_forward + r5_reverse;
    
    dEE_dt = r2_forward - r2_reverse ...
             - r3 + r4;
    
    dEEp_dt = r3 - r4;
    
    dH_dt = -r5_forward + r5_reverse;
    
    dEH_dt = r5_forward - r5_reverse ...
             - r6 + r7;
    
    dEHp_dt = r6 - r7;
    
    % Return derivatives
    dydt = [dE_dt; dEL_dt; dEE_dt; dEEp_dt; dH_dt; dEH_dt; dEHp_dt];
end

