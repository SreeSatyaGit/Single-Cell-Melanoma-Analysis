% =============================================================================
% EGFR → MAPK PATHWAY WITH pERK NEGATIVE FEEDBACK
% Complete Shc–Grb2–SOS → RAS–RAF–MEK–ERK Cascade with Feedback Control
% =============================================================================
%
% Model Description:
% This script models the complete EGFR → MAPK signaling pathway including:
% - Shc recruitment and phosphorylation by pEGFR
% - Grb2 binding and SOS activation
% - RAS activation by SOS*
% - Sequential RAF, MEK, and ERK phosphorylation
% - pERK (ERK-P) negative feedback on upstream components
%
% Negative Feedback Mechanisms:
% 1. ERK-P reduces Shc binding to pEGFR
% 2. ERK-P reduces SOS activation rate
% 3. ERK-P reduces RAF activation rate
% 4. ERK-P increases RAS GTPase activity
%
% =============================================================================

clear all;
close all;
clc;

fprintf('═══════════════════════════════════════════════════════════════════════════\n');
fprintf('   EGFR → MAPK PATHWAY WITH pERK NEGATIVE FEEDBACK\n');
fprintf('═══════════════════════════════════════════════════════════════════════════\n\n');

%% ============================================================================
% INPUT SIGNAL: pEGFR(t)
% ============================================================================

% Define pEGFR as a function of time
% Option 1: Transient pulse (exponential decay)
pEGFR_func = @(t) 2.0 * exp(-0.05 * t) .* (t >= 0);

% Option 2: Step function (constant)
% pEGFR_func = @(t) 1.5 * (t >= 0);

% Option 3: Load from previous simulation
% if exist('egfr_output.mat', 'file')
%     load('egfr_output.mat', 't', 'EEp');
%     t_egfr = t;
%     EEp_egfr = EEp;
%     pEGFR_func = @(t) interp1(t_egfr, EEp_egfr, t, 'linear', EEp_egfr(end));
% end

fprintf('Using pEGFR(t) as input signal.\n\n');

%% ============================================================================
% PARAMETERS
% ============================================================================

fprintf('Setting up parameters...\n');

% ============================================================================
% Shc–Grb2–SOS Module Parameters
% ============================================================================

% Reaction 1: pEGFR + Shc ⇌ pEGFR:Shc → pEGFR + pShc
k_on1 = 0.01;      % nM^-1 min^-1 - Shc binding to pEGFR (base rate)
k_off1 = 0.01;     % min^-1 - Dissociation of pEGFR:Shc
k_cat1 = 0.5;      % min^-1 - Phosphorylation rate (Shc → pShc)

% Reaction 2: pShc + Grb2 ⇌ pShc:Grb2
k_on2 = 0.05;      % nM^-1 min^-1 - Grb2 binding to pShc
k_off2 = 0.05;     % min^-1 - Dissociation of pShc:Grb2

% Reaction 3: pShc:Grb2 + SOS ⇌ pShc:Grb2:SOS → pShc:Grb2 + SOS*
k_on3 = 0.05;      % nM^-1 min^-1 - SOS binding to pShc:Grb2
k_off3 = 0.05;     % min^-1 - Dissociation of ternary complex
k_cat3 = 0.5;      % min^-1 - SOS activation rate (base rate)

% Reaction 4: SOS* → SOS (deactivation)
k_deg4 = 0.05;     % min^-1 - SOS* deactivation rate

% ============================================================================
% RAS–RAF–MEK–ERK Module Parameters
% ============================================================================

% Reaction 5: RAS-GDP + SOS* → RAS-GTP + SOS*
k_SOS = 0.5;        % min^-1 - RAS activation rate by SOS* (base rate)

% Reaction 6: RAS-GTP → RAS-GDP (GTP hydrolysis)
k_GTPase = 0.1;     % min^-1 - RAS intrinsic GTP hydrolysis rate (base rate)

% Reaction 7: RAS-GTP + RAF → RAF* (RAF activation)
k_RAF_act = 0.3;    % min^-1 - RAF activation rate by RAS-GTP (base rate)

% Reaction 8: RAF* → RAF (dephosphorylation)
k_RAF_deact = 0.05; % min^-1 - RAF* deactivation rate

% Reaction 9: RAF* + MEK → RAF* + MEK-P
k_MEK = 0.5;        % min^-1 - MEK phosphorylation rate by RAF*

% Reaction 10: MEK-P → MEK (dephosphorylation)
k_MEK_deact = 0.05; % min^-1 - MEK-P dephosphorylation rate

% Reaction 11: MEK-P + ERK → MEK-P + ERK-P
k_ERK = 0.5;        % min^-1 - ERK phosphorylation rate by MEK-P

% Reaction 12: ERK-P → ERK (dephosphorylation)
k_ERK_deact = 0.05; % min^-1 - ERK-P dephosphorylation rate

% ============================================================================
% Negative Feedback Parameters
% ============================================================================

alpha1 = 0.5;       % Feedback strength: ERK-P inhibition of Shc binding
alpha2 = 0.5;       % Feedback strength: ERK-P inhibition of SOS activation
alpha3 = 0.5;       % Feedback strength: ERK-P inhibition of RAF activation
alpha4 = 0.2;       % Feedback strength: ERK-P enhancement of RAS GTPase

% Store parameters in structure
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
params.pEGFR_func = pEGFR_func;

fprintf('Parameters configured.\n\n');

%% ============================================================================
% INITIAL CONDITIONS
% ============================================================================

fprintf('Setting initial conditions...\n');

% Shc–Grb2–SOS species
Shc0 = 10.0;          % nM - inactive Shc
pEGFR_Shc0 = 0.0;     % nM - pEGFR:Shc complex
pShc0 = 0.0;          % nM - phosphorylated Shc
Grb2_0 = 10.0;        % nM - unbound Grb2
pShc_Grb2_0 = 0.0;    % nM - pShc:Grb2 complex
SOS_0 = 5.0;          % nM - inactive SOS
pShc_Grb2_SOS_0 = 0.0; % nM - ternary complex
SOS_star0 = 0.0;      % nM - active SOS

% RAS–RAF–MEK–ERK species
RAS_GDP0 = 10.0;      % nM - inactive RAS (GDP-bound)
RAS_GTP0 = 0.0;       % nM - active RAS (GTP-bound)
RAF0 = 5.0;           % nM - inactive RAF
RAF_star0 = 0.0;      % nM - active RAF
MEK0 = 5.0;           % nM - inactive MEK
MEK_P0 = 0.0;         % nM - phosphorylated MEK
ERK0 = 5.0;           % nM - inactive ERK
ERK_P0 = 0.0;         % nM - phosphorylated ERK (pERK)

% Initial state vector: [Shc, pEGFR_Shc, pShc, Grb2, pShc_Grb2, SOS, 
%                        pShc_Grb2_SOS, SOS_star, RAS_GDP, RAS_GTP, RAF, 
%                        RAF_star, MEK, MEK_P, ERK, ERK_P]
y0 = [Shc0; pEGFR_Shc0; pShc0; Grb2_0; pShc_Grb2_0; SOS_0; ...
      pShc_Grb2_SOS_0; SOS_star0; RAS_GDP0; RAS_GTP0; RAF0; ...
      RAF_star0; MEK0; MEK_P0; ERK0; ERK_P0];

fprintf('Initial conditions set.\n\n');

%% ============================================================================
% TIME SPAN
% ============================================================================

tspan = [0, 200];  % minutes

%% ============================================================================
% SOLVE ODE SYSTEM
% ============================================================================

fprintf('═══════════════════════════════════════════════════════════════════════════\n');
fprintf('   SOLVING ODE SYSTEM\n');
fprintf('═══════════════════════════════════════════════════════════════════════════\n\n');

fprintf('Solving ODE system (this may take a moment)...\n');
options = odeset('RelTol', 1e-6, 'AbsTol', 1e-8);
[t, y] = ode45(@(t,y) mapk_feedback_odes(t, y, params), tspan, y0, options);

% Extract species concentrations
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
ERK_P = y(:, 16);  % pERK - key output and feedback signal

% Calculate pEGFR input signal over time
pEGFR_signal = arrayfun(pEGFR_func, t);

fprintf('Simulation completed successfully!\n\n');

%% ============================================================================
% PLOTTING
% ============================================================================

fprintf('═══════════════════════════════════════════════════════════════════════════\n');
fprintf('   GENERATING VISUALIZATIONS\n');
fprintf('═══════════════════════════════════════════════════════════════════════════\n\n');

% Figure 1: Key Species Time Courses
figure('Position', [100, 100, 1400, 900]);

% Plot 1: Shc species
subplot(2, 3, 1);
plot(t, Shc, 'b-', 'LineWidth', 2, 'DisplayName', 'Shc');
hold on;
plot(t, pShc, 'r-', 'LineWidth', 2, 'DisplayName', 'pShc');
xlabel('Time (minutes)', 'FontSize', 11);
ylabel('Concentration (nM)', 'FontSize', 11);
title('Shc Species', 'FontSize', 13, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 9);
grid on;
xlim([0, 200]);
hold off;

% Plot 2: SOS species
subplot(2, 3, 2);
plot(t, SOS, 'b-', 'LineWidth', 2, 'DisplayName', 'SOS');
hold on;
plot(t, SOS_star, 'r-', 'LineWidth', 2, 'DisplayName', 'SOS*');
xlabel('Time (minutes)', 'FontSize', 11);
ylabel('Concentration (nM)', 'FontSize', 11);
title('SOS Species', 'FontSize', 13, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 9);
grid on;
xlim([0, 200]);
hold off;

% Plot 3: RAS species
subplot(2, 3, 3);
plot(t, RAS_GDP, 'b-', 'LineWidth', 2, 'DisplayName', 'RAS-GDP');
hold on;
plot(t, RAS_GTP, 'r-', 'LineWidth', 2, 'DisplayName', 'RAS-GTP');
xlabel('Time (minutes)', 'FontSize', 11);
ylabel('Concentration (nM)', 'FontSize', 11);
title('RAS Species', 'FontSize', 13, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 9);
grid on;
xlim([0, 200]);
hold off;

% Plot 4: RAF species
subplot(2, 3, 4);
plot(t, RAF, 'b-', 'LineWidth', 2, 'DisplayName', 'RAF');
hold on;
plot(t, RAF_star, 'r-', 'LineWidth', 2, 'DisplayName', 'RAF*');
xlabel('Time (minutes)', 'FontSize', 11);
ylabel('Concentration (nM)', 'FontSize', 11);
title('RAF Species', 'FontSize', 13, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 9);
grid on;
xlim([0, 200]);
hold off;

% Plot 5: MEK species
subplot(2, 3, 5);
plot(t, MEK, 'b-', 'LineWidth', 2, 'DisplayName', 'MEK');
hold on;
plot(t, MEK_P, 'r-', 'LineWidth', 2, 'DisplayName', 'MEK-P');
xlabel('Time (minutes)', 'FontSize', 11);
ylabel('Concentration (nM)', 'FontSize', 11);
title('MEK Species', 'FontSize', 13, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 9);
grid on;
xlim([0, 200]);
hold off;

% Plot 6: ERK species (including pERK feedback signal)
subplot(2, 3, 6);
plot(t, ERK, 'b-', 'LineWidth', 2, 'DisplayName', 'ERK');
hold on;
plot(t, ERK_P, 'r-', 'LineWidth', 2.5, 'DisplayName', 'ERK-P (pERK)');
xlabel('Time (minutes)', 'FontSize', 11);
ylabel('Concentration (nM)', 'FontSize', 11);
title('ERK Species (Feedback Signal)', 'FontSize', 13, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 9);
grid on;
xlim([0, 200]);
hold off;

sgtitle('EGFR → MAPK Pathway with pERK Negative Feedback', ...
        'FontSize', 15, 'FontWeight', 'bold');

% Figure 2: Cascade Progression and Feedback Effects
figure('Position', [150, 150, 1400, 800]);

% Plot 1: Cascade progression (all active species)
subplot(2, 2, 1);
plot(t, pShc, 'b-', 'LineWidth', 2, 'DisplayName', 'pShc');
hold on;
plot(t, SOS_star, 'g-', 'LineWidth', 2, 'DisplayName', 'SOS*');
plot(t, RAS_GTP, 'c-', 'LineWidth', 2, 'DisplayName', 'RAS-GTP');
plot(t, RAF_star, 'm-', 'LineWidth', 2, 'DisplayName', 'RAF*');
plot(t, MEK_P, 'y-', 'LineWidth', 2, 'DisplayName', 'MEK-P');
plot(t, ERK_P, 'r-', 'LineWidth', 2.5, 'DisplayName', 'ERK-P (pERK)');
xlabel('Time (minutes)', 'FontSize', 12);
ylabel('Concentration (nM)', 'FontSize', 12);
title('Cascade Progression: All Active Species', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 10);
grid on;
xlim([0, 200]);
hold off;

% Plot 2: Input signal and key outputs
subplot(2, 2, 2);
yyaxis left;
plot(t, pEGFR_signal, 'k-', 'LineWidth', 2, 'DisplayName', 'pEGFR (input)');
ylabel('pEGFR Concentration (nM)', 'FontSize', 12);
yyaxis right;
plot(t, SOS_star, 'g-', 'LineWidth', 2, 'DisplayName', 'SOS*');
hold on;
plot(t, ERK_P, 'r-', 'LineWidth', 2.5, 'DisplayName', 'ERK-P (output)');
ylabel('Concentration (nM)', 'FontSize', 12);
xlabel('Time (minutes)', 'FontSize', 12);
title('Input-Output Relationship', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 10);
grid on;
xlim([0, 200]);
hold off;

% Plot 3: Feedback effects visualization
subplot(2, 2, 3);
% Calculate effective rates (with feedback) over time
k_on1_eff = k_on1 ./ (1 + alpha1 * ERK_P);
k_cat3_eff = k_cat3 ./ (1 + alpha2 * ERK_P);
k_RAF_act_eff = k_RAF_act ./ (1 + alpha3 * ERK_P);
k_GTPase_eff = k_GTPase + alpha4 * ERK_P;

plot(t, k_on1_eff, 'b-', 'LineWidth', 2, 'DisplayName', 'k_{on1} (Shc binding)');
hold on;
plot(t, k_cat3_eff, 'g-', 'LineWidth', 2, 'DisplayName', 'k_{cat3} (SOS activation)');
plot(t, k_RAF_act_eff, 'm-', 'LineWidth', 2, 'DisplayName', 'k_{RAF,act}');
plot(t, k_GTPase_eff, 'r-', 'LineWidth', 2, 'DisplayName', 'k_{GTPase}');
xlabel('Time (minutes)', 'FontSize', 12);
ylabel('Effective Rate Constant', 'FontSize', 12);
title('Feedback Effects on Rate Constants', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 10);
grid on;
xlim([0, 200]);
hold off;

% Plot 4: Comparison of upstream vs downstream (showing feedback delay)
subplot(2, 2, 4);
plot(t, SOS_star, 'g-', 'LineWidth', 2, 'DisplayName', 'SOS*');
hold on;
plot(t, RAS_GTP, 'c-', 'LineWidth', 2, 'DisplayName', 'RAS-GTP');
plot(t, RAF_star, 'm-', 'LineWidth', 2, 'DisplayName', 'RAF*');
plot(t, ERK_P, 'r-', 'LineWidth', 2.5, 'DisplayName', 'ERK-P');
xlabel('Time (minutes)', 'FontSize', 12);
ylabel('Concentration (nM)', 'FontSize', 12);
title('Feedback Regulation: Upstream vs Downstream', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 10);
grid on;
xlim([0, 200]);
hold off;

sgtitle('pERK Negative Feedback Effects', 'FontSize', 16, 'FontWeight', 'bold');

%% ============================================================================
% SUMMARY STATISTICS
% ============================================================================

fprintf('═══════════════════════════════════════════════════════════════════════════\n');
fprintf('   SUMMARY STATISTICS\n');
fprintf('═══════════════════════════════════════════════════════════════════════════\n\n');

fprintf('KEY SPECIES PEAK VALUES:\n');
[max_pShc, idx_pShc] = max(pShc);
[max_SOS_star, idx_SOS] = max(SOS_star);
[max_RAS_GTP, idx_RAS] = max(RAS_GTP);
[max_RAF_star, idx_RAF] = max(RAF_star);
[max_MEK_P, idx_MEK] = max(MEK_P);
[max_ERK_P, idx_ERK] = max(ERK_P);

fprintf('  pShc:              %.4f nM at t = %.2f min\n', max_pShc, t(idx_pShc));
fprintf('  SOS*:               %.4f nM at t = %.2f min\n', max_SOS_star, t(idx_SOS));
fprintf('  RAS-GTP:            %.4f nM at t = %.2f min\n', max_RAS_GTP, t(idx_RAS));
fprintf('  RAF*:               %.4f nM at t = %.2f min\n', max_RAF_star, t(idx_RAF));
fprintf('  MEK-P:              %.4f nM at t = %.2f min\n', max_MEK_P, t(idx_MEK));
fprintf('  ERK-P (pERK):       %.4f nM at t = %.2f min\n', max_ERK_P, t(idx_ERK));

fprintf('\nSTEADY-STATE VALUES (at t = 200 min):\n');
fprintf('  pShc:              %.4f nM\n', pShc(end));
fprintf('  SOS*:               %.4f nM\n', SOS_star(end));
fprintf('  RAS-GTP:            %.4f nM\n', RAS_GTP(end));
fprintf('  RAF*:               %.4f nM\n', RAF_star(end));
fprintf('  MEK-P:              %.4f nM\n', MEK_P(end));
fprintf('  ERK-P (pERK):       %.4f nM\n', ERK_P(end));

fprintf('\nFEEDBACK STRENGTHS:\n');
fprintf('  alpha1 (Shc binding):     %.2f\n', alpha1);
fprintf('  alpha2 (SOS activation):  %.2f\n', alpha2);
fprintf('  alpha3 (RAF activation):  %.2f\n', alpha3);
fprintf('  alpha4 (RAS GTPase):      %.2f\n', alpha4);

fprintf('\n═══════════════════════════════════════════════════════════════════════════\n');
fprintf('   SIMULATION COMPLETED SUCCESSFULLY!\n');
fprintf('═══════════════════════════════════════════════════════════════════════════\n\n');

%% ============================================================================
% SAVE OUTPUTS
% ============================================================================

save('mapk_feedback_output.mat', 't', 'pShc', 'SOS_star', 'RAS_GTP', ...
     'RAF_star', 'MEK_P', 'ERK_P', 'pEGFR_signal');
fprintf('Saved outputs to mapk_feedback_output.mat\n');

%% ============================================================================
% ODE FUNCTION
% ============================================================================

function dydt = mapk_feedback_odes(t, y, p)
    % ODE system for EGFR → MAPK pathway with pERK negative feedback
    % y = [Shc, pEGFR_Shc, pShc, Grb2, pShc_Grb2, SOS, pShc_Grb2_SOS, SOS_star,
    %      RAS_GDP, RAS_GTP, RAF, RAF_star, MEK, MEK_P, ERK, ERK_P]
    % p = parameters structure
    
    % Extract species
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
    ERK_P = y(16);  % pERK - feedback signal
    
    % Get pEGFR input signal at current time
    pEGFR = p.pEGFR_func(t);
    
    % ========================================================================
    % CALCULATE FEEDBACK-MODULATED RATE CONSTANTS
    % ========================================================================
    
    % Feedback 1: ERK-P reduces Shc binding to pEGFR
    k_on1_eff = p.k_on1 / (1 + p.alpha1 * ERK_P);
    
    % Feedback 2: ERK-P reduces SOS activation rate
    k_cat3_eff = p.k_cat3 / (1 + p.alpha2 * ERK_P);
    
    % Feedback 3: ERK-P reduces RAF activation rate
    k_RAF_act_eff = p.k_RAF_act / (1 + p.alpha3 * ERK_P);
    
    % Feedback 4: ERK-P increases RAS GTPase activity
    k_GTPase_eff = p.k_GTPase + p.alpha4 * ERK_P;
    
    % ========================================================================
    % SHC–GRB2–SOS MODULE REACTIONS
    % ========================================================================
    
    % Reaction 1: pEGFR + Shc ⇌ pEGFR:Shc → pEGFR + pShc
    r1_forward = k_on1_eff * pEGFR * Shc;  % Feedback-modulated
    r1_reverse = p.k_off1 * pEGFR_Shc;
    r1_cat = p.k_cat1 * pEGFR_Shc;
    
    % Reaction 2: pShc + Grb2 ⇌ pShc:Grb2
    r2_forward = p.k_on2 * pShc * Grb2;
    r2_reverse = p.k_off2 * pShc_Grb2;
    
    % Reaction 3: pShc:Grb2 + SOS ⇌ pShc:Grb2:SOS → pShc:Grb2 + SOS*
    r3_forward = p.k_on3 * pShc_Grb2 * SOS;
    r3_reverse = p.k_off3 * pShc_Grb2_SOS;
    r3_cat = k_cat3_eff * pShc_Grb2_SOS;  % Feedback-modulated
    
    % Reaction 4: SOS* → SOS (deactivation)
    r4 = p.k_deg4 * SOS_star;
    
    % ========================================================================
    % RAS–RAF–MEK–ERK MODULE REACTIONS
    % ========================================================================
    
    % Reaction 5: RAS-GDP + SOS* → RAS-GTP + SOS*
    r5 = p.k_SOS * SOS_star * RAS_GDP;
    
    % Reaction 6: RAS-GTP → RAS-GDP (GTP hydrolysis)
    r6 = k_GTPase_eff * RAS_GTP;  % Feedback-modulated
    
    % Reaction 7: RAS-GTP + RAF → RAF* (RAF activation)
    r7 = k_RAF_act_eff * RAS_GTP * RAF;  % Feedback-modulated
    
    % Reaction 8: RAF* → RAF (dephosphorylation)
    r8 = p.k_RAF_deact * RAF_star;
    
    % Reaction 9: RAF* + MEK → RAF* + MEK-P
    r9 = p.k_MEK * RAF_star * MEK;
    
    % Reaction 10: MEK-P → MEK (dephosphorylation)
    r10 = p.k_MEK_deact * MEK_P;
    
    % Reaction 11: MEK-P + ERK → MEK-P + ERK-P
    r11 = p.k_ERK * MEK_P * ERK;
    
    % Reaction 12: ERK-P → ERK (dephosphorylation)
    r12 = p.k_ERK_deact * ERK_P;
    
    % ========================================================================
    % ODEs FOR EACH SPECIES
    % ========================================================================
    
    % Shc–Grb2–SOS module
    dShc_dt = -r1_forward + r1_reverse;
    dpEGFR_Shc_dt = r1_forward - r1_reverse - r1_cat;
    dpShc_dt = r1_cat - r2_forward + r2_reverse;
    dGrb2_dt = -r2_forward + r2_reverse;
    dpShc_Grb2_dt = r2_forward - r2_reverse - r3_forward + r3_reverse + r3_cat;
    dSOS_dt = -r3_forward + r3_reverse + r4;
    dpShc_Grb2_SOS_dt = r3_forward - r3_reverse - r3_cat;
    dSOS_star_dt = r3_cat - r4;
    
    % RAS–RAF–MEK–ERK module
    dRAS_GDP_dt = -r5 + r6;
    dRAS_GTP_dt = r5 - r6 - r7;
    dRAF_dt = -r7 + r8;
    dRAF_star_dt = r7 - r8;
    dMEK_dt = -r9 + r10;
    dMEK_P_dt = r9 - r10;
    dERK_dt = -r11 + r12;
    dERK_P_dt = r11 - r12;
    
    % Return derivatives
    dydt = [dShc_dt; dpEGFR_Shc_dt; dpShc_dt; dGrb2_dt; dpShc_Grb2_dt; ...
            dSOS_dt; dpShc_Grb2_SOS_dt; dSOS_star_dt; ...
            dRAS_GDP_dt; dRAS_GTP_dt; dRAF_dt; dRAF_star_dt; ...
            dMEK_dt; dMEK_P_dt; dERK_dt; dERK_P_dt];
end

