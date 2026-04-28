% =============================================================================
% INTEGRATED MAPK AND PI3K/AKT/mTOR PATHWAY MODEL
% Complete EGFR → MAPK and PI3K/AKT/mTOR Signaling with Feedback Control
% =============================================================================
%
% Model Description:
% This script models both major signaling pathways downstream of EGFR:
%
% PATHWAY 1: EGFR → MAPK (RAS-RAF-MEK-ERK)
%   - Shc recruitment and phosphorylation by pEGFR
%   - Grb2 binding and SOS activation
%   - RAS activation by SOS*
%   - Sequential RAF, MEK, and ERK phosphorylation
%   - pERK negative feedback on upstream components
%
% PATHWAY 2: EGFR → PI3K/AKT/mTOR
%   - PI3K activation by pEGFR/pHER3
%   - PIP2 to PIP3 conversion
%   - AKT dual phosphorylation (Thr308, Ser473)
%   - TSC1/TSC2 inhibition and Rheb-GTP accumulation
%   - mTORC1 activation
%   - S6K and 4EBP1 phosphorylation
%   - S6K negative feedback on PI3K
%
% Key Features:
% - Both pathways share the same input signal (pEGFR/pRTK)
% - Independent but parallel signaling cascades
% - Each pathway has its own negative feedback mechanisms
% - Comprehensive visualization of both pathways
%
% =============================================================================

clear all;
close all;
clc;

fprintf('═══════════════════════════════════════════════════════════════════════════\n');
fprintf('   INTEGRATED MAPK AND PI3K/AKT/mTOR PATHWAY MODEL\n');
fprintf('═══════════════════════════════════════════════════════════════════════════\n\n');

%% ============================================================================
% INPUT SIGNAL: pEGFR(t) or pRTK(t)
% ============================================================================

% Define pRTK (pEGFR or pHER3) as a function of time
% Option 1: Transient pulse (exponential decay)
pRTK_func = @(t) 2.0 * exp(-0.05 * t) .* (t >= 0);

% Option 2: Step function (constant)
% pRTK_func = @(t) 1.5 * (t >= 0);

% Option 3: Load from previous simulation
% if exist('egfr_output.mat', 'file')
%     load('egfr_output.mat', 't', 'EEp', 'EHp');
%     t_rtk = t;
%     pRTK_data = EEp + 0.5 * EHp;  % Combined pEGFR and pHER3
%     pRTK_func = @(t) interp1(t_rtk, pRTK_data, t, 'linear', pRTK_data(end));
% end

fprintf('Using pRTK (pEGFR/pHER3) as input signal for both pathways.\n\n');

%% ============================================================================
% PARAMETERS
% ============================================================================

fprintf('Setting up parameters for both pathways...\n');

% ============================================================================
% MAPK PATHWAY PARAMETERS
% ============================================================================

% Shc–Grb2–SOS Module
k_on1 = 0.01;      % nM^-1 min^-1 - Shc binding to pEGFR
k_off1 = 0.01;     % min^-1 - Dissociation of pEGFR:Shc
k_cat1 = 0.5;      % min^-1 - Shc phosphorylation rate
k_on2 = 0.05;      % nM^-1 min^-1 - Grb2 binding to pShc
k_off2 = 0.05;     % min^-1 - Dissociation of pShc:Grb2
k_on3 = 0.05;      % nM^-1 min^-1 - SOS binding to pShc:Grb2
k_off3 = 0.05;     % min^-1 - Dissociation of ternary complex
k_cat3 = 0.5;      % min^-1 - SOS activation rate
k_deg4 = 0.05;     % min^-1 - SOS* deactivation rate

% RAS–RAF–MEK–ERK Module
k_SOS = 0.5;        % min^-1 - RAS activation rate by SOS*
k_GTPase = 0.1;     % min^-1 - RAS GTP hydrolysis rate
k_RAF_act = 0.3;    % min^-1 - RAF activation rate
k_RAF_deact = 0.05; % min^-1 - RAF* deactivation rate
k_MEK = 0.5;        % min^-1 - MEK phosphorylation rate
k_MEK_deact = 0.05; % min^-1 - MEK-P dephosphorylation rate
k_ERK = 0.5;        % min^-1 - ERK phosphorylation rate
k_ERK_deact = 0.05; % min^-1 - ERK-P dephosphorylation rate

% MAPK Feedback Parameters
alpha1 = 0.5;       % ERK-P inhibition of Shc binding
alpha2 = 0.5;       % ERK-P inhibition of SOS activation
alpha3 = 0.5;       % ERK-P inhibition of RAF activation
alpha4 = 0.2;       % ERK-P enhancement of RAS GTPase

% ============================================================================
% PI3K/AKT/mTOR PATHWAY PARAMETERS
% ============================================================================

% PI3K Module
k_PI3K_act = 0.5;   % min^-1 - PI3K activation rate by pRTK
k_PI3K_deact = 0.05; % min^-1 - PI3K deactivation rate
k_PIP2_to_PIP3 = 0.5; % min^-1 - PIP2 to PIP3 conversion rate
k_PTEN = 0.1;       % min^-1 - PTEN-mediated PIP3 dephosphorylation

% AKT Module
k_AKT_recruit = 0.3; % min^-1 - AKT recruitment rate by PIP3
k_AKT_Thr308 = 0.5;  % min^-1 - AKT Thr308 phosphorylation rate
k_AKT_Ser473 = 0.3;  % min^-1 - AKT Ser473 phosphorylation rate
k_AKT_deact = 0.05;  % min^-1 - AKT dephosphorylation rate

% TSC1/TSC2 and Rheb Module
k_TSC2_inh = 0.5;   % min^-1 - TSC2 inhibition rate by pAKT
k_Rheb_act = 0.5;    % min^-1 - Rheb-GTP activation rate
k_Rheb_GTPase = 0.1; % min^-1 - Rheb GTPase activity

% mTORC1 Module
k_mTORC1_act = 0.5;  % min^-1 - mTORC1 activation rate
k_mTORC1_deact = 0.05; % min^-1 - mTORC1 deactivation rate

% S6K and 4EBP1 Module
k_S6K = 0.5;         % min^-1 - S6K phosphorylation rate
k_S6K_deact = 0.05;  % min^-1 - pS6K dephosphorylation rate
k_4EBP1 = 0.5;       % min^-1 - 4EBP1 phosphorylation rate
k_4EBP1_deact = 0.05; % min^-1 - p4EBP1 dephosphorylation rate

% PI3K Feedback Parameters
alpha_PI3K = 0.5;   % pS6K inhibition of PI3K activation

% Store all parameters in structure
params.k_on1 = k_on1;
params.k_off1 = k_off1;
params.k_cat1 = k_cat1;
params.k_on2 = k_on2;
params.k_off2 = k_off2;
params.k_on3 = k_on3;
params.k_off3 = k_off3;
params.k_cat3 = k_cat3;
params.k_deg4 = k_deg4;
params.k_SOS = k_SOS;
params.k_GTPase = k_GTPase;
params.k_RAF_act = k_RAF_act;
params.k_RAF_deact = k_RAF_deact;
params.k_MEK = k_MEK;
params.k_MEK_deact = k_MEK_deact;
params.k_ERK = k_ERK;
params.k_ERK_deact = k_ERK_deact;
params.alpha1 = alpha1;
params.alpha2 = alpha2;
params.alpha3 = alpha3;
params.alpha4 = alpha4;
params.k_PI3K_act = k_PI3K_act;
params.k_PI3K_deact = k_PI3K_deact;
params.k_PIP2_to_PIP3 = k_PIP2_to_PIP3;
params.k_PTEN = k_PTEN;
params.k_AKT_recruit = k_AKT_recruit;
params.k_AKT_Thr308 = k_AKT_Thr308;
params.k_AKT_Ser473 = k_AKT_Ser473;
params.k_AKT_deact = k_AKT_deact;
params.k_TSC2_inh = k_TSC2_inh;
params.k_Rheb_act = k_Rheb_act;
params.k_Rheb_GTPase = k_Rheb_GTPase;
params.k_mTORC1_act = k_mTORC1_act;
params.k_mTORC1_deact = k_mTORC1_deact;
params.k_S6K = k_S6K;
params.k_S6K_deact = k_S6K_deact;
params.k_4EBP1 = k_4EBP1;
params.k_4EBP1_deact = k_4EBP1_deact;
params.alpha_PI3K = alpha_PI3K;
params.pRTK_func = pRTK_func;

fprintf('Parameters configured for both pathways.\n\n');

%% ============================================================================
% INITIAL CONDITIONS
% ============================================================================

fprintf('Setting initial conditions...\n');

% MAPK Pathway species
Shc0 = 10.0;
pEGFR_Shc0 = 0.0;
pShc0 = 0.0;
Grb2_0 = 10.0;
pShc_Grb2_0 = 0.0;
SOS_0 = 5.0;
pShc_Grb2_SOS_0 = 0.0;
SOS_star0 = 0.0;
RAS_GDP0 = 10.0;
RAS_GTP0 = 0.0;
RAF0 = 5.0;
RAF_star0 = 0.0;
MEK0 = 5.0;
MEK_P0 = 0.0;
ERK0 = 5.0;
ERK_P0 = 0.0;

% PI3K/AKT/mTOR Pathway species
PI3K0 = 5.0;
PI3K_active0 = 0.0;
PIP2_0 = 10.0;
PIP3_0 = 0.0;
AKT0 = 5.0;
pAKT_Thr308_0 = 0.0;
pAKT_Ser473_0 = 0.0;
pAKT_full_0 = 0.0;
TSC2_active0 = 5.0;
Rheb_GTP0 = 0.0;
mTORC1_0 = 5.0;
mTORC1_active0 = 0.0;
S6K0 = 5.0;
pS6K0 = 0.0;
EBP1_0 = 5.0;
p4EBP1_0 = 0.0;

% Combined initial state vector: [MAPK species (16), PI3K species (16)]
y0 = [Shc0; pEGFR_Shc0; pShc0; Grb2_0; pShc_Grb2_0; SOS_0; ...
      pShc_Grb2_SOS_0; SOS_star0; RAS_GDP0; RAS_GTP0; RAF0; ...
      RAF_star0; MEK0; MEK_P0; ERK0; ERK_P0; ...
      PI3K0; PI3K_active0; PIP2_0; PIP3_0; AKT0; pAKT_Thr308_0; ...
      pAKT_Ser473_0; pAKT_full_0; TSC2_active0; Rheb_GTP0; ...
      mTORC1_0; mTORC1_active0; S6K0; pS6K0; EBP1_0; p4EBP1_0];

fprintf('Initial conditions set for both pathways.\n\n');

%% ============================================================================
% TIME SPAN
% ============================================================================

tspan = [0, 200];  % minutes

%% ============================================================================
% SOLVE ODE SYSTEM
% ============================================================================

fprintf('═══════════════════════════════════════════════════════════════════════════\n');
fprintf('   SOLVING INTEGRATED ODE SYSTEM\n');
fprintf('═══════════════════════════════════════════════════════════════════════════\n\n');

fprintf('Solving ODE system (this may take a moment)...\n');
options = odeset('RelTol', 1e-6, 'AbsTol', 1e-8);
[t, y] = ode45(@(t,y) integrated_pathways_odes(t, y, params), tspan, y0, options);

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
RAF = y(:, 11);
RAF_star = y(:, 12);
MEK = y(:, 13);
MEK_P = y(:, 14);
ERK = y(:, 15);
ERK_P = y(:, 16);

% Extract PI3K/AKT/mTOR pathway species
PI3K = y(:, 17);
PI3K_active = y(:, 18);
PIP2 = y(:, 19);
PIP3 = y(:, 20);
AKT = y(:, 21);
pAKT_Thr308 = y(:, 22);
pAKT_Ser473 = y(:, 23);
pAKT_full = y(:, 24);
TSC2_active = y(:, 25);
Rheb_GTP = y(:, 26);
mTORC1 = y(:, 27);
mTORC1_active = y(:, 28);
S6K = y(:, 29);
pS6K = y(:, 30);
EBP1 = y(:, 31);
p4EBP1 = y(:, 32);

% Calculate pRTK input signal over time
pRTK_signal = arrayfun(pRTK_func, t);

fprintf('Simulation completed successfully!\n\n');

%% ============================================================================
% PLOTTING
% ============================================================================

fprintf('═══════════════════════════════════════════════════════════════════════════\n');
fprintf('   GENERATING VISUALIZATIONS\n');
fprintf('═══════════════════════════════════════════════════════════════════════════\n\n');

% Figure 1: MAPK Pathway Overview
figure('Position', [50, 50, 1600, 1000]);

% MAPK Pathway - Left column
subplot(3, 3, 1);
plot(t, pShc, 'b-', 'LineWidth', 2, 'DisplayName', 'pShc');
hold on;
plot(t, SOS_star, 'g-', 'LineWidth', 2, 'DisplayName', 'SOS*');
xlabel('Time (minutes)', 'FontSize', 10);
ylabel('Concentration (nM)', 'FontSize', 10);
title('MAPK: Adaptor Module', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 9);
grid on;
xlim([0, 200]);
hold off;

subplot(3, 3, 2);
plot(t, RAS_GTP, 'c-', 'LineWidth', 2, 'DisplayName', 'RAS-GTP');
hold on;
plot(t, RAF_star, 'm-', 'LineWidth', 2, 'DisplayName', 'RAF*');
xlabel('Time (minutes)', 'FontSize', 10);
ylabel('Concentration (nM)', 'FontSize', 10);
title('MAPK: RAS & RAF', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 9);
grid on;
xlim([0, 200]);
hold off;

subplot(3, 3, 3);
plot(t, MEK_P, 'y-', 'LineWidth', 2, 'DisplayName', 'MEK-P');
hold on;
plot(t, ERK_P, 'r-', 'LineWidth', 2.5, 'DisplayName', 'ERK-P');
xlabel('Time (minutes)', 'FontSize', 10);
ylabel('Concentration (nM)', 'FontSize', 10);
title('MAPK: MEK & ERK', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 9);
grid on;
xlim([0, 200]);
hold off;

% PI3K Pathway - Middle column
subplot(3, 3, 4);
plot(t, PI3K_active, 'b-', 'LineWidth', 2, 'DisplayName', 'PI3K_active');
hold on;
plot(t, PIP3, 'r-', 'LineWidth', 2, 'DisplayName', 'PIP3');
xlabel('Time (minutes)', 'FontSize', 10);
ylabel('Concentration (nM)', 'FontSize', 10);
title('PI3K: PI3K & PIP3', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 9);
grid on;
xlim([0, 200]);
hold off;

subplot(3, 3, 5);
plot(t, pAKT_full, 'b-', 'LineWidth', 2.5, 'DisplayName', 'pAKT(Full)');
hold on;
plot(t, Rheb_GTP, 'g-', 'LineWidth', 2, 'DisplayName', 'Rheb-GTP');
xlabel('Time (minutes)', 'FontSize', 10);
ylabel('Concentration (nM)', 'FontSize', 10);
title('PI3K: AKT & Rheb', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 9);
grid on;
xlim([0, 200]);
hold off;

subplot(3, 3, 6);
plot(t, mTORC1_active, 'm-', 'LineWidth', 2, 'DisplayName', 'mTORC1_active');
hold on;
plot(t, pS6K, 'r-', 'LineWidth', 2, 'DisplayName', 'pS6K');
plot(t, p4EBP1, 'b-', 'LineWidth', 2, 'DisplayName', 'p4EBP1');
xlabel('Time (minutes)', 'FontSize', 10);
ylabel('Concentration (nM)', 'FontSize', 10);
title('PI3K: mTORC1 Outputs', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 9);
grid on;
xlim([0, 200]);
hold off;

% Comparison plots - Right column
subplot(3, 3, 7);
plot(t, pRTK_signal, 'k-', 'LineWidth', 2);
xlabel('Time (minutes)', 'FontSize', 10);
ylabel('Concentration (nM)', 'FontSize', 10);
title('Input Signal: pRTK', 'FontSize', 12, 'FontWeight', 'bold');
grid on;
xlim([0, 200]);

subplot(3, 3, 8);
plot(t, ERK_P, 'r-', 'LineWidth', 2.5, 'DisplayName', 'ERK-P (MAPK)');
hold on;
plot(t, pAKT_full, 'b-', 'LineWidth', 2.5, 'DisplayName', 'pAKT (PI3K)');
xlabel('Time (minutes)', 'FontSize', 10);
ylabel('Concentration (nM)', 'FontSize', 10);
title('Key Outputs Comparison', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 9);
grid on;
xlim([0, 200]);
hold off;

subplot(3, 3, 9);
plot(t, ERK_P, 'r-', 'LineWidth', 2, 'DisplayName', 'ERK-P');
hold on;
plot(t, pS6K, 'g-', 'LineWidth', 2, 'DisplayName', 'pS6K');
plot(t, p4EBP1, 'b-', 'LineWidth', 2, 'DisplayName', 'p4EBP1');
xlabel('Time (minutes)', 'FontSize', 10);
ylabel('Concentration (nM)', 'FontSize', 10);
title('Final Pathway Outputs', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 9);
grid on;
xlim([0, 200]);
hold off;

sgtitle('Integrated MAPK and PI3K/AKT/mTOR Pathways', 'FontSize', 16, 'FontWeight', 'bold');

% Figure 2: Detailed Pathway Comparison
figure('Position', [100, 100, 1400, 900]);

% Pathway flow comparison
subplot(2, 3, 1);
plot(t, SOS_star, 'g-', 'LineWidth', 2, 'DisplayName', 'SOS*');
hold on;
plot(t, RAS_GTP, 'c-', 'LineWidth', 2, 'DisplayName', 'RAS-GTP');
plot(t, RAF_star, 'm-', 'LineWidth', 2, 'DisplayName', 'RAF*');
plot(t, MEK_P, 'y-', 'LineWidth', 2, 'DisplayName', 'MEK-P');
plot(t, ERK_P, 'r-', 'LineWidth', 2.5, 'DisplayName', 'ERK-P');
xlabel('Time (minutes)', 'FontSize', 12);
ylabel('Concentration (nM)', 'FontSize', 12);
title('MAPK Pathway Flow', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 10);
grid on;
xlim([0, 200]);
hold off;

subplot(2, 3, 2);
plot(t, PIP3, 'c-', 'LineWidth', 2, 'DisplayName', 'PIP3');
hold on;
plot(t, pAKT_full, 'b-', 'LineWidth', 2, 'DisplayName', 'pAKT(Full)');
plot(t, Rheb_GTP, 'g-', 'LineWidth', 2, 'DisplayName', 'Rheb-GTP');
plot(t, mTORC1_active, 'm-', 'LineWidth', 2, 'DisplayName', 'mTORC1_active');
xlabel('Time (minutes)', 'FontSize', 12);
ylabel('Concentration (nM)', 'FontSize', 12);
title('PI3K/AKT/mTOR Pathway Flow', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 10);
grid on;
xlim([0, 200]);
hold off;

% Input-output relationships
subplot(2, 3, 3);
yyaxis left;
plot(t, pRTK_signal, 'k-', 'LineWidth', 2);
ylabel('pRTK (nM)', 'FontSize', 12);
yyaxis right;
plot(t, ERK_P, 'r-', 'LineWidth', 2, 'DisplayName', 'ERK-P');
hold on;
plot(t, pAKT_full, 'b-', 'LineWidth', 2, 'DisplayName', 'pAKT');
ylabel('Concentration (nM)', 'FontSize', 12);
xlabel('Time (minutes)', 'FontSize', 12);
title('Input-Output: Both Pathways', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 10);
grid on;
xlim([0, 200]);
hold off;

% Feedback effects
subplot(2, 3, 4);
k_on1_eff = k_on1 ./ (1 + alpha1 * ERK_P);
k_PI3K_eff = k_PI3K_act ./ (1 + alpha_PI3K * pS6K);
plot(t, k_on1_eff, 'r-', 'LineWidth', 2, 'DisplayName', 'k_{on1} (MAPK feedback)');
hold on;
plot(t, k_PI3K_eff, 'b-', 'LineWidth', 2, 'DisplayName', 'k_{PI3K} (PI3K feedback)');
xlabel('Time (minutes)', 'FontSize', 12);
ylabel('Effective Rate Constant', 'FontSize', 12);
title('Negative Feedback Effects', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 10);
grid on;
xlim([0, 200]);
hold off;

% Final outputs comparison
subplot(2, 3, 5);
plot(t, ERK_P, 'r-', 'LineWidth', 2.5, 'DisplayName', 'ERK-P (MAPK output)');
hold on;
plot(t, pS6K, 'g-', 'LineWidth', 2, 'DisplayName', 'pS6K (PI3K output)');
plot(t, p4EBP1, 'b-', 'LineWidth', 2, 'DisplayName', 'p4EBP1 (PI3K output)');
xlabel('Time (minutes)', 'FontSize', 12);
ylabel('Concentration (nM)', 'FontSize', 12);
title('Final Pathway Outputs', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 10);
grid on;
xlim([0, 200]);
hold off;

% Cascade amplification comparison
subplot(2, 3, 6);
% Normalize to peak values for comparison
ERK_P_norm = ERK_P / max(ERK_P);
pAKT_norm = pAKT_full / max(pAKT_full);
pS6K_norm = pS6K / max(pS6K);
plot(t, ERK_P_norm, 'r-', 'LineWidth', 2.5, 'DisplayName', 'ERK-P (normalized)');
hold on;
plot(t, pAKT_norm, 'b-', 'LineWidth', 2, 'DisplayName', 'pAKT (normalized)');
plot(t, pS6K_norm, 'g-', 'LineWidth', 2, 'DisplayName', 'pS6K (normalized)');
xlabel('Time (minutes)', 'FontSize', 12);
ylabel('Normalized Concentration', 'FontSize', 12);
title('Normalized Output Comparison', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 10);
grid on;
xlim([0, 200]);
hold off;

sgtitle('Integrated Pathways: Detailed Analysis', 'FontSize', 16, 'FontWeight', 'bold');

%% ============================================================================
% SUMMARY STATISTICS
% ============================================================================

fprintf('═══════════════════════════════════════════════════════════════════════════\n');
fprintf('   SUMMARY STATISTICS\n');
fprintf('═══════════════════════════════════════════════════════════════════════════\n\n');

fprintf('MAPK PATHWAY:\n');
[max_SOS_star, idx_SOS] = max(SOS_star);
[max_RAS_GTP, idx_RAS] = max(RAS_GTP);
[max_RAF_star, idx_RAF] = max(RAF_star);
[max_MEK_P, idx_MEK] = max(MEK_P);
[max_ERK_P, idx_ERK] = max(ERK_P);
fprintf('  Peak SOS*:            %.4f nM at t = %.2f min\n', max_SOS_star, t(idx_SOS));
fprintf('  Peak RAS-GTP:         %.4f nM at t = %.2f min\n', max_RAS_GTP, t(idx_RAS));
fprintf('  Peak RAF*:            %.4f nM at t = %.2f min\n', max_RAF_star, t(idx_RAF));
fprintf('  Peak MEK-P:           %.4f nM at t = %.2f min\n', max_MEK_P, t(idx_MEK));
fprintf('  Peak ERK-P:           %.4f nM at t = %.2f min\n', max_ERK_P, t(idx_ERK));

fprintf('\nPI3K/AKT/mTOR PATHWAY:\n');
[max_PIP3, idx_PIP3] = max(PIP3);
[max_pAKT_full, idx_pAKT] = max(pAKT_full);
[max_Rheb_GTP, idx_Rheb] = max(Rheb_GTP);
[max_mTORC1_active, idx_mTOR] = max(mTORC1_active);
[max_pS6K, idx_S6K] = max(pS6K);
[max_p4EBP1, idx_4EBP1] = max(p4EBP1);
fprintf('  Peak PIP3:            %.4f nM at t = %.2f min\n', max_PIP3, t(idx_PIP3));
fprintf('  Peak pAKT(Full):      %.4f nM at t = %.2f min\n', max_pAKT_full, t(idx_pAKT));
fprintf('  Peak Rheb-GTP:        %.4f nM at t = %.2f min\n', max_Rheb_GTP, t(idx_Rheb));
fprintf('  Peak mTORC1_active:   %.4f nM at t = %.2f min\n', max_mTORC1_active, t(idx_mTOR));
fprintf('  Peak pS6K:            %.4f nM at t = %.2f min\n', max_pS6K, t(idx_S6K));
fprintf('  Peak p4EBP1:          %.4f nM at t = %.2f min\n', max_p4EBP1, t(idx_4EBP1));

fprintf('\nSTEADY-STATE VALUES (at t = 200 min):\n');
fprintf('  ERK-P:                %.4f nM\n', ERK_P(end));
fprintf('  pAKT(Full):           %.4f nM\n', pAKT_full(end));
fprintf('  pS6K:                 %.4f nM\n', pS6K(end));
fprintf('  p4EBP1:               %.4f nM\n', p4EBP1(end));

fprintf('\n═══════════════════════════════════════════════════════════════════════════\n');
fprintf('   SIMULATION COMPLETED SUCCESSFULLY!\n');
fprintf('═══════════════════════════════════════════════════════════════════════════\n\n');

%% ============================================================================
% SAVE OUTPUTS
% ============================================================================

save('integrated_mapk_pi3k_output.mat', 't', 'ERK_P', 'pAKT_full', 'pS6K', ...
     'p4EBP1', 'SOS_star', 'RAS_GTP', 'RAF_star', 'MEK_P', 'PIP3', ...
     'Rheb_GTP', 'mTORC1_active', 'pRTK_signal');
fprintf('Saved outputs to integrated_mapk_pi3k_output.mat\n');

%% ============================================================================
% ODE FUNCTION
% ============================================================================

function dydt = integrated_pathways_odes(t, y, p)
    % Integrated ODE system for both MAPK and PI3K/AKT/mTOR pathways
    % y = [MAPK: Shc, pEGFR_Shc, pShc, Grb2, pShc_Grb2, SOS, pShc_Grb2_SOS, SOS_star,
    %      RAS_GDP, RAS_GTP, RAF, RAF_star, MEK, MEK_P, ERK, ERK_P;
    %      PI3K: PI3K, PI3K_active, PIP2, PIP3, AKT, pAKT_Thr308, pAKT_Ser473, pAKT_full,
    %      TSC2_active, Rheb_GTP, mTORC1, mTORC1_active, S6K, pS6K, 4EBP1, p4EBP1]
    
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
    RAF = y(11);
    RAF_star = y(12);
    MEK = y(13);
    MEK_P = y(14);
    ERK = y(15);
    ERK_P = y(16);
    
    % Extract PI3K/AKT/mTOR pathway species
    PI3K = y(17);
    PI3K_active = y(18);
    PIP2 = y(19);
    PIP3 = y(20);
    AKT = y(21);
    pAKT_Thr308 = y(22);
    pAKT_Ser473 = y(23);
    pAKT_full = y(24);
    TSC2_active = y(25);
    Rheb_GTP = y(26);
    mTORC1 = y(27);
    mTORC1_active = y(28);
    S6K = y(29);
    pS6K = y(30);
    EBP1 = y(31);
    p4EBP1 = y(32);
    
    % Get pRTK input signal at current time
    pRTK = p.pRTK_func(t);
    
    % ========================================================================
    % CALCULATE FEEDBACK-MODULATED RATE CONSTANTS
    % ========================================================================
    
    % MAPK pathway feedbacks (ERK-P)
    k_on1_eff = p.k_on1 / (1 + p.alpha1 * ERK_P);
    k_cat3_eff = p.k_cat3 / (1 + p.alpha2 * ERK_P);
    k_RAF_act_eff = p.k_RAF_act / (1 + p.alpha3 * ERK_P);
    k_GTPase_eff = p.k_GTPase + p.alpha4 * ERK_P;
    
    % PI3K pathway feedback (pS6K)
    k_PI3K_act_eff = p.k_PI3K_act / (1 + p.alpha_PI3K * pS6K);
    
    % ========================================================================
    % MAPK PATHWAY REACTIONS
    % ========================================================================
    
    % Shc–Grb2–SOS module
    r1_forward = k_on1_eff * pRTK * Shc;
    r1_reverse = p.k_off1 * pEGFR_Shc;
    r1_cat = p.k_cat1 * pEGFR_Shc;
    r2_forward = p.k_on2 * pShc * Grb2;
    r2_reverse = p.k_off2 * pShc_Grb2;
    r3_forward = p.k_on3 * pShc_Grb2 * SOS;
    r3_reverse = p.k_off3 * pShc_Grb2_SOS;
    r3_cat = k_cat3_eff * pShc_Grb2_SOS;
    r4 = p.k_deg4 * SOS_star;
    
    % RAS–RAF–MEK–ERK module
    r5 = p.k_SOS * SOS_star * RAS_GDP;
    r6 = k_GTPase_eff * RAS_GTP;
    r7 = k_RAF_act_eff * RAS_GTP * RAF;
    r8 = p.k_RAF_deact * RAF_star;
    r9 = p.k_MEK * RAF_star * MEK;
    r10 = p.k_MEK_deact * MEK_P;
    r11 = p.k_ERK * MEK_P * ERK;
    r12 = p.k_ERK_deact * ERK_P;
    
    % ========================================================================
    % PI3K/AKT/mTOR PATHWAY REACTIONS
    % ========================================================================
    
    % PI3K module
    r13 = k_PI3K_act_eff * pRTK * PI3K;
    r14 = p.k_PI3K_deact * PI3K_active;
    r15 = p.k_PIP2_to_PIP3 * PI3K_active * PIP2;
    r16 = p.k_PTEN * PIP3;
    
    % AKT module
    r17 = p.k_AKT_recruit * PIP3 * AKT;
    r18 = p.k_AKT_Thr308 * PIP3 * AKT;
    r19 = p.k_AKT_Ser473 * pAKT_Thr308;
    r20 = p.k_AKT_Ser473 * PIP3 * AKT;
    r21 = p.k_AKT_Thr308 * pAKT_Ser473;
    r22_Thr308 = p.k_AKT_deact * pAKT_Thr308;
    r22_Ser473 = p.k_AKT_deact * pAKT_Ser473;
    r22_full = p.k_AKT_deact * pAKT_full;
    
    % TSC2/Rheb module
    r23 = p.k_TSC2_inh * pAKT_full * TSC2_active;
    Rheb_GDP = 5.0 - Rheb_GTP;
    r24 = p.k_Rheb_act * Rheb_GDP / (1 + TSC2_active);
    r25 = p.k_Rheb_GTPase * Rheb_GTP * (1 + 5 * TSC2_active);
    
    % mTORC1 module
    r26 = p.k_mTORC1_act * Rheb_GTP * mTORC1;
    r27 = p.k_mTORC1_deact * mTORC1_active;
    
    % S6K and 4EBP1 module
    r28 = p.k_S6K * mTORC1_active * S6K;
    r29 = p.k_S6K_deact * pS6K;
    r30 = p.k_4EBP1 * mTORC1_active * EBP1;
    r31 = p.k_4EBP1_deact * p4EBP1;
    
    % ========================================================================
    % ODEs FOR MAPK PATHWAY
    % ========================================================================
    
    dShc_dt = -r1_forward + r1_reverse;
    dpEGFR_Shc_dt = r1_forward - r1_reverse - r1_cat;
    dpShc_dt = r1_cat - r2_forward + r2_reverse;
    dGrb2_dt = -r2_forward + r2_reverse;
    dpShc_Grb2_dt = r2_forward - r2_reverse - r3_forward + r3_reverse + r3_cat;
    dSOS_dt = -r3_forward + r3_reverse + r4;
    dpShc_Grb2_SOS_dt = r3_forward - r3_reverse - r3_cat;
    dSOS_star_dt = r3_cat - r4;
    dRAS_GDP_dt = -r5 + r6;
    dRAS_GTP_dt = r5 - r6 - r7;
    dRAF_dt = -r7 + r8;
    dRAF_star_dt = r7 - r8;
    dMEK_dt = -r9 + r10;
    dMEK_P_dt = r9 - r10;
    dERK_dt = -r11 + r12;
    dERK_P_dt = r11 - r12;
    
    % ========================================================================
    % ODEs FOR PI3K/AKT/mTOR PATHWAY
    % ========================================================================
    
    dPI3K_dt = -r13 + r14;
    dPI3K_active_dt = r13 - r14;
    dPIP2_dt = -r15 + r16;
    dPIP3_dt = r15 - r16;
    dAKT_dt = -r17 - r18 - r20 + r22_Thr308 + r22_Ser473 + r22_full;
    dpAKT_Thr308_dt = r17 + r18 - r19 - r22_Thr308;
    dpAKT_Ser473_dt = r20 - r21 - r22_Ser473;
    dpAKT_full_dt = r19 + r21 - r22_full;
    dTSC2_active_dt = -r23;
    dRheb_GTP_dt = r24 - r25;
    dmTORC1_dt = -r26 + r27;
    dmTORC1_active_dt = r26 - r27;
    dS6K_dt = -r28 + r29;
    dpS6K_dt = r28 - r29;
    dEBP1_dt = -r30 + r31;
    dp4EBP1_dt = r30 - r31;
    
    % Return all derivatives
    dydt = [dShc_dt; dpEGFR_Shc_dt; dpShc_dt; dGrb2_dt; dpShc_Grb2_dt; ...
            dSOS_dt; dpShc_Grb2_SOS_dt; dSOS_star_dt; ...
            dRAS_GDP_dt; dRAS_GTP_dt; dRAF_dt; dRAF_star_dt; ...
            dMEK_dt; dMEK_P_dt; dERK_dt; dERK_P_dt; ...
            dPI3K_dt; dPI3K_active_dt; dPIP2_dt; dPIP3_dt; ...
            dAKT_dt; dpAKT_Thr308_dt; dpAKT_Ser473_dt; dpAKT_full_dt; ...
            dTSC2_active_dt; dRheb_GTP_dt; dmTORC1_dt; dmTORC1_active_dt; ...
            dS6K_dt; dpS6K_dt; dEBP1_dt; dp4EBP1_dt];
end

