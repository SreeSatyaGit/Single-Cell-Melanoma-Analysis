% =============================================================================
% COMPLETE MAPK SIGNALING PATHWAY MODEL
% Integrated EGFR-HER3 → Shc-Grb2-SOS → RAS-RAF-MEK-ERK Cascade
% Ordinary Differential Equations (ODEs) with Mass-Action Kinetics
% =============================================================================
%
% This script models the complete MAPK signaling pathway from receptor
% activation to ERK phosphorylation, integrating three modules:
%
% Module 1: EGFR-HER3 Receptor Signaling
%   - Ligand binding, dimerization, and phosphorylation
%   - Output: pEGFR (EEp)
%
% Module 2: Shc-Grb2-SOS Adaptor Cascade
%   - Shc recruitment and phosphorylation by pEGFR
%   - Grb2 binding and SOS activation
%   - Output: SOS* (active SOS)
%
% Module 3: RAS-RAF-MEK-ERK (MAPK) Cascade
%   - RAS activation by SOS*
%   - RAF, MEK, and ERK sequential phosphorylation
%   - Output: ERK-P (active ERK)
%
% =============================================================================

clear all;
close all;
clc;

fprintf('═══════════════════════════════════════════════════════════════════════════\n');
fprintf('   COMPLETE MAPK SIGNALING PATHWAY MODEL\n');
fprintf('═══════════════════════════════════════════════════════════════════════════\n\n');

%% ============================================================================
% MODULE 1: EGFR-HER3 RECEPTOR SIGNALING
% ============================================================================

fprintf('═══════════════════════════════════════════════════════════════════════════\n');
fprintf('   MODULE 1: EGFR-HER3 RECEPTOR SIGNALING\n');
fprintf('═══════════════════════════════════════════════════════════════════════════\n\n');

% Parameters
kon = 0.01;      % nM^-1 min^-1
koff = 0.01;     % min^-1
kd = 0.02;       % nM^-1 min^-1
kdm = 0.05;      % min^-1
kphos_EE = 0.5;  % min^-1
kd_EH = 0.01;    % nM^-1 min^-1
kdm_EH = 0.05;   % min^-1
kphos_EH = 0.3;  % min^-1
kdeph = 0.1;     % min^-1
L = 10;          % nM

params1.kon = kon;
params1.koff = koff;
params1.kd = kd;
params1.kdm = kdm;
params1.kphos_EE = kphos_EE;
params1.kd_EH = kd_EH;
params1.kdm_EH = kdm_EH;
params1.kphos_EH = kphos_EH;
params1.kdeph = kdeph;
params1.L = L;

% Initial conditions
E0 = 10;    % nM
EL0 = 0;
EE0 = 0;
EEp0 = 0;
H0 = 5;     % nM
EH0 = 0;
EHp0 = 0;

y0_1 = [E0; EL0; EE0; EEp0; H0; EH0; EHp0];
tspan = [0, 200];  % minutes

% Solve ODE system
fprintf('Solving EGFR-HER3 ODE system...\n');
options = odeset('RelTol', 1e-6, 'AbsTol', 1e-8);
[t1, y1] = ode45(@(t,y) egfr_her3_odes(t, y, params1), tspan, y0_1, options);

% Extract species
E = y1(:, 1);
EL = y1(:, 2);
EE = y1(:, 3);
EEp = y1(:, 4);  % pEGFR - output for Module 2
H = y1(:, 5);
EH = y1(:, 6);
EHp = y1(:, 7);

fprintf('Module 1 completed. pEGFR (EEp) range: %.4f to %.4f nM\n\n', min(EEp), max(EEp));

% Create interpolation function for pEGFR
pEGFR_func = @(t) interp1(t1, EEp, t, 'linear', EEp(end));

%% ============================================================================
% MODULE 2: SHC-GRB2-SOS ADAPTOR CASCADE
% ============================================================================

fprintf('═══════════════════════════════════════════════════════════════════════════\n');
fprintf('   MODULE 2: SHC-GRB2-SOS ADAPTOR CASCADE\n');
fprintf('═══════════════════════════════════════════════════════════════════════════\n\n');

% Parameters
k_on1 = 0.1;      % nM^-1 min^-1
k_off1 = 0.05;    % min^-1
k_cat1 = 0.3;      % min^-1
k_on2 = 0.15;      % nM^-1 min^-1
k_off2 = 0.08;     % min^-1
k_on3 = 0.12;      % nM^-1 min^-1
k_off3 = 0.06;     % min^-1
k_cat3 = 0.25;     % min^-1
k_deg4 = 0.02;     % min^-1

params2.k_on1 = k_on1;
params2.k_off1 = k_off1;
params2.k_cat1 = k_cat1;
params2.k_on2 = k_on2;
params2.k_off2 = k_off2;
params2.k_on3 = k_on3;
params2.k_off3 = k_off3;
params2.k_cat3 = k_cat3;
params2.k_deg4 = k_deg4;
params2.pEGFR_func = pEGFR_func;

% Initial conditions
Shc0 = 10.0;
pEGFR_Shc0 = 0.0;
pShc0 = 0.0;
Grb2_0 = 8.0;
pShc_Grb2_0 = 0.0;
SOS_0 = 5.0;
pShc_Grb2_SOS_0 = 0.0;
SOS_star0 = 0.0;

y0_2 = [Shc0; pEGFR_Shc0; pShc0; Grb2_0; pShc_Grb2_0; SOS_0; pShc_Grb2_SOS_0; SOS_star0];

% Solve ODE system
fprintf('Solving Shc-Grb2-SOS ODE system...\n');
[t2, y2] = ode45(@(t,y) shc_grb2_sos_odes(t, y, params2), tspan, y0_2, options);

% Extract species
Shc = y2(:, 1);
pEGFR_Shc = y2(:, 2);
pShc = y2(:, 3);
Grb2 = y2(:, 4);
pShc_Grb2 = y2(:, 5);
SOS = y2(:, 6);
pShc_Grb2_SOS = y2(:, 7);
SOS_star = y2(:, 8);  % Active SOS - output for Module 3

fprintf('Module 2 completed. SOS* range: %.4f to %.4f nM\n\n', min(SOS_star), max(SOS_star));

% Create interpolation function for SOS_star
SOS_star_func = @(t) interp1(t2, SOS_star, t, 'linear', SOS_star(end));

%% ============================================================================
% MODULE 3: RAS-RAF-MEK-ERK (MAPK) CASCADE
% ============================================================================

fprintf('═══════════════════════════════════════════════════════════════════════════\n');
fprintf('   MODULE 3: RAS-RAF-MEK-ERK (MAPK) CASCADE\n');
fprintf('═══════════════════════════════════════════════════════════════════════════\n\n');

% Parameters
k_SOS = 0.5;        % min^-1
k_GTPase = 0.1;     % min^-1
k_RAF_act = 0.3;    % min^-1
k_RAF_deact = 0.05; % min^-1
k_MEK = 0.5;        % min^-1
k_MEK_deact = 0.05; % min^-1
k_ERK = 0.5;        % min^-1
k_ERK_deact = 0.05; % min^-1

params3.k_SOS = k_SOS;
params3.k_GTPase = k_GTPase;
params3.k_RAF_act = k_RAF_act;
params3.k_RAF_deact = k_RAF_deact;
params3.k_MEK = k_MEK;
params3.k_MEK_deact = k_MEK_deact;
params3.k_ERK = k_ERK;
params3.k_ERK_deact = k_ERK_deact;
params3.SOS_star_func = SOS_star_func;

% Initial conditions
RAS_GDP0 = 10.0;
RAS_GTP0 = 0.0;
RAF0 = 5.0;
RAF_star0 = 0.0;
MEK0 = 5.0;
MEK_P0 = 0.0;
ERK0 = 5.0;
ERK_P0 = 0.0;

y0_3 = [RAS_GDP0; RAS_GTP0; RAF0; RAF_star0; MEK0; MEK_P0; ERK0; ERK_P0];

% Solve ODE system
fprintf('Solving RAS-RAF-MEK-ERK ODE system...\n');
[t3, y3] = ode45(@(t,y) mapk_cascade_odes(t, y, params3), tspan, y0_3, options);

% Extract species
RAS_GDP = y3(:, 1);
RAS_GTP = y3(:, 2);
RAF = y3(:, 3);
RAF_star = y3(:, 4);
MEK = y3(:, 5);
MEK_P = y3(:, 6);
ERK = y3(:, 7);
ERK_P = y3(:, 8);  % Final output: Active ERK

fprintf('Module 3 completed. ERK-P range: %.4f to %.4f nM\n\n', min(ERK_P), max(ERK_P));

%% ============================================================================
% COMPREHENSIVE VISUALIZATION
% ============================================================================

fprintf('═══════════════════════════════════════════════════════════════════════════\n');
fprintf('   GENERATING VISUALIZATIONS\n');
fprintf('═══════════════════════════════════════════════════════════════════════════\n\n');

% Figure 1: Complete Pathway Overview
figure('Position', [50, 50, 1600, 1000]);

% Module 1: EGFR-HER3
subplot(3, 3, 1);
plot(t1, EEp, 'b-', 'LineWidth', 2);
xlabel('Time (minutes)', 'FontSize', 10);
ylabel('Concentration (nM)', 'FontSize', 10);
title('Module 1: pEGFR (EEp)', 'FontSize', 12, 'FontWeight', 'bold');
grid on;
xlim([0, 200]);

subplot(3, 3, 2);
plot(t1, EHp, 'r-', 'LineWidth', 2);
xlabel('Time (minutes)', 'FontSize', 10);
ylabel('Concentration (nM)', 'FontSize', 10);
title('Module 1: pHER3 (EHp)', 'FontSize', 12, 'FontWeight', 'bold');
grid on;
xlim([0, 200]);

subplot(3, 3, 3);
plot(t1, EEp, 'b-', 'LineWidth', 2, 'DisplayName', 'EEp');
hold on;
plot(t1, EHp, 'r-', 'LineWidth', 2, 'DisplayName', 'EHp');
xlabel('Time (minutes)', 'FontSize', 10);
ylabel('Concentration (nM)', 'FontSize', 10);
title('Module 1: All Phosphorylated', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 9);
grid on;
xlim([0, 200]);
hold off;

% Module 2: Shc-Grb2-SOS
subplot(3, 3, 4);
plot(t2, pShc, 'b-', 'LineWidth', 2, 'DisplayName', 'pShc');
hold on;
plot(t2, pShc_Grb2, 'g-', 'LineWidth', 2, 'DisplayName', 'pShc:Grb2');
xlabel('Time (minutes)', 'FontSize', 10);
ylabel('Concentration (nM)', 'FontSize', 10);
title('Module 2: Shc-Grb2 Complexes', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 9);
grid on;
xlim([0, 200]);
hold off;

subplot(3, 3, 5);
plot(t2, SOS_star, 'r-', 'LineWidth', 2.5);
xlabel('Time (minutes)', 'FontSize', 10);
ylabel('Concentration (nM)', 'FontSize', 10);
title('Module 2: SOS* (Output)', 'FontSize', 12, 'FontWeight', 'bold');
grid on;
xlim([0, 200]);

subplot(3, 3, 6);
plot(t2, SOS, 'b-', 'LineWidth', 2, 'DisplayName', 'SOS');
hold on;
plot(t2, SOS_star, 'r-', 'LineWidth', 2, 'DisplayName', 'SOS*');
xlabel('Time (minutes)', 'FontSize', 10);
ylabel('Concentration (nM)', 'FontSize', 10);
title('Module 2: SOS Species', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 9);
grid on;
xlim([0, 200]);
hold off;

% Module 3: MAPK Cascade
subplot(3, 3, 7);
plot(t3, RAS_GTP, 'b-', 'LineWidth', 2, 'DisplayName', 'RAS-GTP');
hold on;
plot(t3, RAF_star, 'g-', 'LineWidth', 2, 'DisplayName', 'RAF*');
xlabel('Time (minutes)', 'FontSize', 10);
ylabel('Concentration (nM)', 'FontSize', 10);
title('Module 3: RAS & RAF', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 9);
grid on;
xlim([0, 200]);
hold off;

subplot(3, 3, 8);
plot(t3, MEK_P, 'm-', 'LineWidth', 2, 'DisplayName', 'MEK-P');
hold on;
plot(t3, ERK_P, 'r-', 'LineWidth', 2.5, 'DisplayName', 'ERK-P');
xlabel('Time (minutes)', 'FontSize', 10);
ylabel('Concentration (nM)', 'FontSize', 10);
title('Module 3: MEK & ERK', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 9);
grid on;
xlim([0, 200]);
hold off;

subplot(3, 3, 9);
plot(t3, ERK_P, 'r-', 'LineWidth', 2.5);
xlabel('Time (minutes)', 'FontSize', 10);
ylabel('Concentration (nM)', 'FontSize', 10);
title('Module 3: ERK-P (Final Output)', 'FontSize', 12, 'FontWeight', 'bold');
grid on;
xlim([0, 200]);

sgtitle('Complete MAPK Signaling Pathway', 'FontSize', 16, 'FontWeight', 'bold');

% Figure 2: Signal Flow Through Pathway
figure('Position', [100, 100, 1400, 800]);

% Interpolate all signals to common time vector for comparison
t_common = linspace(0, 200, 1000)';
EEp_common = interp1(t1, EEp, t_common, 'linear', EEp(end));
SOS_star_common = interp1(t2, SOS_star, t_common, 'linear', SOS_star(end));
ERK_P_common = interp1(t3, ERK_P, t_common, 'linear', ERK_P(end));

subplot(2, 2, 1);
plot(t_common, EEp_common, 'b-', 'LineWidth', 2, 'DisplayName', 'pEGFR (EEp)');
hold on;
plot(t_common, SOS_star_common, 'g-', 'LineWidth', 2, 'DisplayName', 'SOS*');
plot(t_common, ERK_P_common, 'r-', 'LineWidth', 2.5, 'DisplayName', 'ERK-P');
xlabel('Time (minutes)', 'FontSize', 12);
ylabel('Concentration (nM)', 'FontSize', 12);
title('Signal Flow Through Complete Pathway', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 10);
grid on;
xlim([0, 200]);
hold off;

subplot(2, 2, 2);
yyaxis left;
plot(t_common, EEp_common, 'b-', 'LineWidth', 2);
ylabel('pEGFR (nM)', 'FontSize', 12);
yyaxis right;
plot(t_common, SOS_star_common, 'g-', 'LineWidth', 2);
ylabel('SOS* (nM)', 'FontSize', 12);
xlabel('Time (minutes)', 'FontSize', 12);
title('Module 1 → Module 2', 'FontSize', 12, 'FontWeight', 'bold');
grid on;
xlim([0, 200]);

subplot(2, 2, 3);
yyaxis left;
plot(t_common, SOS_star_common, 'g-', 'LineWidth', 2);
ylabel('SOS* (nM)', 'FontSize', 12);
yyaxis right;
plot(t_common, ERK_P_common, 'r-', 'LineWidth', 2.5);
ylabel('ERK-P (nM)', 'FontSize', 12);
xlabel('Time (minutes)', 'FontSize', 12);
title('Module 2 → Module 3', 'FontSize', 12, 'FontWeight', 'bold');
grid on;
xlim([0, 200]);

subplot(2, 2, 4);
RAS_GTP_common = interp1(t3, RAS_GTP, t_common, 'linear', RAS_GTP(end));
RAF_star_common = interp1(t3, RAF_star, t_common, 'linear', RAF_star(end));
MEK_P_common = interp1(t3, MEK_P, t_common, 'linear', MEK_P(end));
plot(t_common, EEp_common, 'b-', 'LineWidth', 1.5, 'DisplayName', 'pEGFR');
hold on;
plot(t_common, SOS_star_common, 'g-', 'LineWidth', 1.5, 'DisplayName', 'SOS*');
plot(t_common, RAS_GTP_common, 'c-', 'LineWidth', 1.5, 'DisplayName', 'RAS-GTP');
plot(t_common, RAF_star_common, 'm-', 'LineWidth', 1.5, 'DisplayName', 'RAF*');
plot(t_common, MEK_P_common, 'y-', 'LineWidth', 1.5, 'DisplayName', 'MEK-P');
plot(t_common, ERK_P_common, 'r-', 'LineWidth', 2.5, 'DisplayName', 'ERK-P');
xlabel('Time (minutes)', 'FontSize', 12);
ylabel('Concentration (nM)', 'FontSize', 12);
title('All Key Species', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 9);
grid on;
xlim([0, 200]);
hold off;

sgtitle('Complete MAPK Pathway: Signal Transmission', 'FontSize', 16, 'FontWeight', 'bold');

%% ============================================================================
% SUMMARY STATISTICS
% ============================================================================

fprintf('═══════════════════════════════════════════════════════════════════════════\n');
fprintf('   SUMMARY STATISTICS\n');
fprintf('═══════════════════════════════════════════════════════════════════════════\n\n');

fprintf('MODULE 1 (EGFR-HER3):\n');
fprintf('  Peak pEGFR (EEp):     %.4f nM at t = %.2f min\n', max(EEp), t1(EEp == max(EEp)));
fprintf('  Peak pHER3 (EHp):     %.4f nM at t = %.2f min\n', max(EHp), t1(EHp == max(EHp)));
fprintf('  Steady-state EEp:     %.4f nM\n', EEp(end));

fprintf('\nMODULE 2 (Shc-Grb2-SOS):\n');
fprintf('  Peak SOS*:            %.4f nM at t = %.2f min\n', max(SOS_star), t2(SOS_star == max(SOS_star)));
fprintf('  Steady-state SOS*:    %.4f nM\n', SOS_star(end));

fprintf('\nMODULE 3 (RAS-RAF-MEK-ERK):\n');
fprintf('  Peak RAS-GTP:         %.4f nM at t = %.2f min\n', max(RAS_GTP), t3(RAS_GTP == max(RAS_GTP)));
fprintf('  Peak RAF*:            %.4f nM at t = %.2f min\n', max(RAF_star), t3(RAF_star == max(RAF_star)));
fprintf('  Peak MEK-P:           %.4f nM at t = %.2f min\n', max(MEK_P), t3(MEK_P == max(MEK_P)));
fprintf('  Peak ERK-P:           %.4f nM at t = %.2f min\n', max(ERK_P), t3(ERK_P == max(ERK_P)));
fprintf('  Steady-state ERK-P:   %.4f nM\n', ERK_P(end));

fprintf('\n═══════════════════════════════════════════════════════════════════════════\n');
fprintf('   SIMULATION COMPLETED SUCCESSFULLY!\n');
fprintf('═══════════════════════════════════════════════════════════════════════════\n\n');

%% ============================================================================
% SAVE OUTPUTS
% ============================================================================

save('fullMAPK_output.mat', 't1', 'EEp', 'EHp', 't2', 'SOS_star', 't3', 'ERK_P', ...
     'RAS_GTP', 'RAF_star', 'MEK_P');
fprintf('Saved all outputs to fullMAPK_output.mat\n');

%% ============================================================================
% ODE FUNCTIONS
% ============================================================================

function dydt = egfr_her3_odes(t, y, p)
    % ODE system for EGFR-HER3 receptor signaling
    % y = [E, EL, EE, EEp, H, EH, EHp]
    
    E = y(1);
    EL = y(2);
    EE = y(3);
    EEp = y(4);
    H = y(5);
    EH = y(6);
    EHp = y(7);
    
    L = p.L;
    
    r1_forward = p.kon * E * L;
    r1_reverse = p.koff * EL;
    r2_forward = p.kd * EL * EL;
    r2_reverse = p.kdm * EE;
    r3 = p.kphos_EE * EE;
    r4 = p.kdeph * EEp;
    r5_forward = p.kd_EH * EL * H;
    r5_reverse = p.kdm_EH * EH;
    r6 = p.kphos_EH * EH;
    r7 = p.kdeph * EHp;
    
    dE_dt = -r1_forward + r1_reverse;
    dEL_dt = r1_forward - r1_reverse - 2*r2_forward + 2*r2_reverse - r5_forward + r5_reverse;
    dEE_dt = r2_forward - r2_reverse - r3 + r4;
    dEEp_dt = r3 - r4;
    dH_dt = -r5_forward + r5_reverse;
    dEH_dt = r5_forward - r5_reverse - r6 + r7;
    dEHp_dt = r6 - r7;
    
    dydt = [dE_dt; dEL_dt; dEE_dt; dEEp_dt; dH_dt; dEH_dt; dEHp_dt];
end

function dydt = shc_grb2_sos_odes(t, y, p)
    % ODE system for Shc-Grb2-SOS adaptor-mediated signal transduction cascade
    % y = [Shc, pEGFR_Shc, pShc, Grb2, pShc_Grb2, SOS, pShc_Grb2_SOS, SOS_star]
    
    Shc = y(1);
    pEGFR_Shc = y(2);
    pShc = y(3);
    Grb2 = y(4);
    pShc_Grb2 = y(5);
    SOS = y(6);
    pShc_Grb2_SOS = y(7);
    SOS_star = y(8);
    
    pEGFR = p.pEGFR_func(t);
    
    r1_forward = p.k_on1 * pEGFR * Shc;
    r1_reverse = p.k_off1 * pEGFR_Shc;
    r1_cat = p.k_cat1 * pEGFR_Shc;
    r2_forward = p.k_on2 * pShc * Grb2;
    r2_reverse = p.k_off2 * pShc_Grb2;
    r3_forward = p.k_on3 * pShc_Grb2 * SOS;
    r3_reverse = p.k_off3 * pShc_Grb2_SOS;
    r3_cat = p.k_cat3 * pShc_Grb2_SOS;
    r4 = p.k_deg4 * SOS_star;
    
    dShc_dt = -r1_forward + r1_reverse;
    dpEGFR_Shc_dt = r1_forward - r1_reverse - r1_cat;
    dpShc_dt = r1_cat - r2_forward + r2_reverse;
    dGrb2_dt = -r2_forward + r2_reverse;
    dpShc_Grb2_dt = r2_forward - r2_reverse - r3_forward + r3_reverse + r3_cat;
    dSOS_dt = -r3_forward + r3_reverse + r4;
    dpShc_Grb2_SOS_dt = r3_forward - r3_reverse - r3_cat;
    dSOS_star_dt = r3_cat - r4;
    
    dydt = [dShc_dt; dpEGFR_Shc_dt; dpShc_dt; dGrb2_dt; ...
            dpShc_Grb2_dt; dSOS_dt; dpShc_Grb2_SOS_dt; dSOS_star_dt];
end

function dydt = mapk_cascade_odes(t, y, p)
    % ODE system for RAS-RAF-MEK-ERK (MAPK) signaling cascade
    % y = [RAS_GDP, RAS_GTP, RAF, RAF_star, MEK, MEK_P, ERK, ERK_P]
    
    RAS_GDP = y(1);
    RAS_GTP = y(2);
    RAF = y(3);
    RAF_star = y(4);
    MEK = y(5);
    MEK_P = y(6);
    ERK = y(7);
    ERK_P = y(8);
    
    SOS_star = p.SOS_star_func(t);
    
    r1 = p.k_SOS * SOS_star * RAS_GDP;
    r2 = p.k_GTPase * RAS_GTP;
    r3 = p.k_RAF_act * RAS_GTP * RAF;
    r4 = p.k_RAF_deact * RAF_star;
    r5 = p.k_MEK * RAF_star * MEK;
    r6 = p.k_MEK_deact * MEK_P;
    r7 = p.k_ERK * MEK_P * ERK;
    r8 = p.k_ERK_deact * ERK_P;
    
    dRAS_GDP_dt = -r1 + r2;
    dRAS_GTP_dt = r1 - r2 - r3;
    dRAF_dt = -r3 + r4;
    dRAF_star_dt = r3 - r4;
    dMEK_dt = -r5 + r6;
    dMEK_P_dt = r5 - r6;
    dERK_dt = -r7 + r8;
    dERK_P_dt = r7 - r8;
    
    dydt = [dRAS_GDP_dt; dRAS_GTP_dt; dRAF_dt; dRAF_star_dt; ...
            dMEK_dt; dMEK_P_dt; dERK_dt; dERK_P_dt];
end

