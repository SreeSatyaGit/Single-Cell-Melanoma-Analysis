% =============================================================================
% RAS-RAF-MEK-ERK (MAPK) SIGNALING CASCADE MODEL
% Ordinary Differential Equations (ODEs) with Mass-Action Kinetics
% =============================================================================
%
% Model Description:
% This script models the canonical MAPK signaling cascade downstream of
% active SOS (SOS*), including:
% - RAS activation (RAS-GDP → RAS-GTP) by SOS*
% - RAF activation by RAS-GTP
% - MEK activation (phosphorylation) by RAF*
% - ERK activation (phosphorylation) by MEK-P
% - Deactivation/dephosphorylation reactions
%
% Species:
% RAS_GDP    = inactive RAS (GDP-bound)
% RAS_GTP    = active RAS (GTP-bound)
% RAF        = inactive RAF
% RAF_star   = active RAF (phosphorylated/dimerized)
% MEK        = inactive MEK
% MEK_P      = phosphorylated MEK (active)
% ERK        = inactive ERK
% ERK_P      = phosphorylated ERK (active)
%
% Input Signal:
% SOS_star(t) = active SOS concentration over time (external input)
%
% =============================================================================

clear all;
close all;
clc;

%% ============================================================================
% INPUT SIGNAL: SOS_star(t) - Load from Shc-Grb2-SOS model
% ============================================================================

% Load SOS_star (pSOS) from previous Shc-Grb2-SOS adaptor model simulation
if exist('sos_output.mat', 'file')
    % Load with different variable names to avoid conflicts
    load('sos_output.mat', 't', 'SOS_star');
    t_sos = t;  % Rename to avoid conflict with anonymous function parameter
    SOS_star_data = SOS_star;
    
    fprintf('Loaded SOS_star (pSOS) from sos_output.mat\n');
    fprintf('  Time points: %d\n', length(t_sos));
    fprintf('  SOS_star range: %.4f to %.4f nM\n', min(SOS_star_data), max(SOS_star_data));
    
    % Create interpolation function
    % Use loaded t_sos and SOS_star_data, function parameter is t
    SOS_star_func = @(t) interp1(t_sos, SOS_star_data, t, 'linear', SOS_star_data(end));
    
    fprintf('Using SOS_star from Shc-Grb2-SOS model as input signal.\n');
else
    % Fallback: Use exponential decay if file not found
    warning('sos_output.mat not found. Using default exponential decay for SOS_star.');
    SOS_star_func = @(t) 1.0 * exp(-0.05 * t) .* (t >= 0);
    fprintf('Using default SOS_star function.\n');
end

%% ============================================================================
% PARAMETERS
% ============================================================================

% Reaction 1: RAS-GDP + SOS* → RAS-GTP + SOS* (RAS activation)
k_SOS = 0.5;        % min^-1 - RAS activation rate by SOS*

% Reaction 2: RAS-GTP → RAS-GDP (GTP hydrolysis)
k_GTPase = 0.1;     % min^-1 - RAS intrinsic GTP hydrolysis rate

% Reaction 3: RAS-GTP + RAF → RAS-GTP:RAF → RAF* (RAF activation)
k_RAF_act = 0.3;    % min^-1 - RAF activation rate by RAS-GTP
% Note: Simplified as single-step activation (could be expanded to include complex)

% Reaction 4: RAF* → RAF (dephosphorylation)
k_RAF_deact = 0.05; % min^-1 - RAF* deactivation rate

% Reaction 5: RAF* + MEK → RAF* + MEK-P (MEK phosphorylation)
k_MEK = 0.5;        % min^-1 - MEK phosphorylation rate by RAF*

% Reaction 6: MEK-P → MEK (dephosphorylation)
k_MEK_deact = 0.05; % min^-1 - MEK-P dephosphorylation rate

% Reaction 7: MEK-P + ERK → MEK-P + ERK-P (ERK phosphorylation)
k_ERK = 0.5;        % min^-1 - ERK phosphorylation rate by MEK-P

% Reaction 8: ERK-P → ERK (dephosphorylation)
k_ERK_deact = 0.05; % min^-1 - ERK-P dephosphorylation rate

% Store parameters in structure
params.k_SOS = k_SOS;
params.k_GTPase = k_GTPase;
params.k_RAF_act = k_RAF_act;
params.k_RAF_deact = k_RAF_deact;
params.k_MEK = k_MEK;
params.k_MEK_deact = k_MEK_deact;
params.k_ERK = k_ERK;
params.k_ERK_deact = k_ERK_deact;
params.SOS_star_func = SOS_star_func;

%% ============================================================================
% INITIAL CONDITIONS
% ============================================================================

RAS_GDP0 = 10.0;    % nM - inactive RAS (GDP-bound)
RAS_GTP0 = 0.0;     % nM - active RAS (GTP-bound, initially zero)
RAF0 = 5.0;         % nM - inactive RAF
RAF_star0 = 0.0;    % nM - active RAF (initially zero)
MEK0 = 5.0;         % nM - inactive MEK
MEK_P0 = 0.0;       % nM - phosphorylated MEK (initially zero)
ERK0 = 5.0;         % nM - inactive ERK
ERK_P0 = 0.0;       % nM - phosphorylated ERK (initially zero)

% Initial state vector: [RAS_GDP, RAS_GTP, RAF, RAF_star, MEK, MEK_P, ERK, ERK_P]
y0 = [RAS_GDP0; RAS_GTP0; RAF0; RAF_star0; MEK0; MEK_P0; ERK0; ERK_P0];

%% ============================================================================
% TIME SPAN
% ============================================================================

tspan = [0, 200];  % minutes

%% ============================================================================
% SOLVE ODE SYSTEM
% ============================================================================

fprintf('\nSolving ODE system for RAS-RAF-MEK-ERK cascade...\n');
options = odeset('RelTol', 1e-6, 'AbsTol', 1e-8);
[t, y] = ode45(@(t,y) mapk_cascade_odes(t, y, params), tspan, y0, options);

% Extract species concentrations
RAS_GDP = y(:, 1);
RAS_GTP = y(:, 2);
RAF = y(:, 3);
RAF_star = y(:, 4);
MEK = y(:, 5);
MEK_P = y(:, 6);
ERK = y(:, 7);
ERK_P = y(:, 8);

% Calculate SOS_star input signal over time
SOS_star_signal = arrayfun(SOS_star_func, t);

fprintf('Simulation completed.\n');

%% ============================================================================
% PLOTTING
% ============================================================================

% Create figure for main plots
figure('Position', [100, 100, 1400, 900]);

% Plot 1: RAS species (RAS-GDP, RAS-GTP)
subplot(2, 3, 1);
plot(t, RAS_GDP, 'b-', 'LineWidth', 2, 'DisplayName', 'RAS-GDP (inactive)');
hold on;
plot(t, RAS_GTP, 'r-', 'LineWidth', 2, 'DisplayName', 'RAS-GTP (active)');
xlabel('Time (minutes)', 'FontSize', 11);
ylabel('Concentration (nM)', 'FontSize', 11);
title('RAS Activation Dynamics', 'FontSize', 13, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 9);
grid on;
xlim([0, 200]);
hold off;

% Plot 2: RAF species (RAF, RAF*)
subplot(2, 3, 2);
plot(t, RAF, 'b-', 'LineWidth', 2, 'DisplayName', 'RAF (inactive)');
hold on;
plot(t, RAF_star, 'r-', 'LineWidth', 2, 'DisplayName', 'RAF* (active)');
xlabel('Time (minutes)', 'FontSize', 11);
ylabel('Concentration (nM)', 'FontSize', 11);
title('RAF Activation Dynamics', 'FontSize', 13, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 9);
grid on;
xlim([0, 200]);
hold off;

% Plot 3: MEK species (MEK, MEK-P)
subplot(2, 3, 3);
plot(t, MEK, 'b-', 'LineWidth', 2, 'DisplayName', 'MEK (inactive)');
hold on;
plot(t, MEK_P, 'r-', 'LineWidth', 2, 'DisplayName', 'MEK-P (active)');
xlabel('Time (minutes)', 'FontSize', 11);
ylabel('Concentration (nM)', 'FontSize', 11);
title('MEK Activation Dynamics', 'FontSize', 13, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 9);
grid on;
xlim([0, 200]);
hold off;

% Plot 4: ERK species (ERK, ERK-P)
subplot(2, 3, 4);
plot(t, ERK, 'b-', 'LineWidth', 2, 'DisplayName', 'ERK (inactive)');
hold on;
plot(t, ERK_P, 'r-', 'LineWidth', 2, 'DisplayName', 'ERK-P (active)');
xlabel('Time (minutes)', 'FontSize', 11);
ylabel('Concentration (nM)', 'FontSize', 11);
title('ERK Activation Dynamics', 'FontSize', 13, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 9);
grid on;
xlim([0, 200]);
hold off;

% Plot 5: Input signal SOS_star(t)
subplot(2, 3, 5);
plot(t, SOS_star_signal, 'k-', 'LineWidth', 2);
xlabel('Time (minutes)', 'FontSize', 11);
ylabel('Concentration (nM)', 'FontSize', 11);
title('Input Signal: SOS* (t)', 'FontSize', 13, 'FontWeight', 'bold');
grid on;
xlim([0, 200]);

% Plot 6: Cascade progression (all active species)
subplot(2, 3, 6);
plot(t, RAS_GTP, 'b-', 'LineWidth', 2, 'DisplayName', 'RAS-GTP');
hold on;
plot(t, RAF_star, 'g-', 'LineWidth', 2, 'DisplayName', 'RAF*');
plot(t, MEK_P, 'm-', 'LineWidth', 2, 'DisplayName', 'MEK-P');
plot(t, ERK_P, 'r-', 'LineWidth', 2.5, 'DisplayName', 'ERK-P');
xlabel('Time (minutes)', 'FontSize', 11);
ylabel('Concentration (nM)', 'FontSize', 11);
title('MAPK Cascade Progression', 'FontSize', 13, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 9);
grid on;
xlim([0, 200]);
hold off;

sgtitle('RAS-RAF-MEK-ERK (MAPK) Signaling Cascade Model', ...
        'FontSize', 15, 'FontWeight', 'bold');

% Create separate figure for detailed cascade visualization
figure('Position', [150, 150, 1200, 600]);

% Comparison plot: Input vs Output
subplot(1, 2, 1);
yyaxis left;
plot(t, SOS_star_signal, 'k-', 'LineWidth', 2, 'DisplayName', 'SOS* (input)');
ylabel('SOS* Concentration (nM)', 'FontSize', 12);
yyaxis right;
plot(t, ERK_P, 'r-', 'LineWidth', 2.5, 'DisplayName', 'ERK-P (output)');
ylabel('ERK-P Concentration (nM)', 'FontSize', 12);
xlabel('Time (minutes)', 'FontSize', 12);
title('Input-Output Relationship', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 10);
grid on;
xlim([0, 200]);

% Cascade amplification
subplot(1, 2, 2);
plot(t, RAS_GTP, 'b-', 'LineWidth', 2, 'DisplayName', 'RAS-GTP');
hold on;
plot(t, RAF_star, 'g-', 'LineWidth', 2, 'DisplayName', 'RAF*');
plot(t, MEK_P, 'm-', 'LineWidth', 2, 'DisplayName', 'MEK-P');
plot(t, ERK_P, 'r-', 'LineWidth', 2.5, 'DisplayName', 'ERK-P');
xlabel('Time (minutes)', 'FontSize', 12);
ylabel('Concentration (nM)', 'FontSize', 12);
title('Signal Amplification Through Cascade', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 10);
grid on;
xlim([0, 200]);
hold off;

%% ============================================================================
% STEADY-STATE ANALYSIS
% ============================================================================

fprintf('\n=== Steady-State Concentrations (at t = 200 min) ===\n');
fprintf('RAS-GDP:            %.4f nM\n', RAS_GDP(end));
fprintf('RAS-GTP:            %.4f nM\n', RAS_GTP(end));
fprintf('RAF:                %.4f nM\n', RAF(end));
fprintf('RAF*:               %.4f nM\n', RAF_star(end));
fprintf('MEK:                %.4f nM\n', MEK(end));
fprintf('MEK-P:              %.4f nM\n', MEK_P(end));
fprintf('ERK:                %.4f nM\n', ERK(end));
fprintf('ERK-P:              %.4f nM\n', ERK_P(end));

fprintf('\n=== Conservation Checks ===\n');
total_RAS = RAS_GDP(end) + RAS_GTP(end);
fprintf('Total RAS:           %.4f nM (initial: %.1f nM)\n', total_RAS, RAS_GDP0);

total_RAF = RAF(end) + RAF_star(end);
fprintf('Total RAF:           %.4f nM (initial: %.1f nM)\n', total_RAF, RAF0);

total_MEK = MEK(end) + MEK_P(end);
fprintf('Total MEK:           %.4f nM (initial: %.1f nM)\n', total_MEK, MEK0);

total_ERK = ERK(end) + ERK_P(end);
fprintf('Total ERK:           %.4f nM (initial: %.1f nM)\n', total_ERK, ERK0);

fprintf('\n=== Peak Activation ===\n');
[max_RAS_GTP, idx_RAS] = max(RAS_GTP);
[max_RAF_star, idx_RAF] = max(RAF_star);
[max_MEK_P, idx_MEK] = max(MEK_P);
[max_ERK_P, idx_ERK] = max(ERK_P);

fprintf('Peak RAS-GTP:        %.4f nM at t = %.2f minutes\n', max_RAS_GTP, t(idx_RAS));
fprintf('Peak RAF*:           %.4f nM at t = %.2f minutes\n', max_RAF_star, t(idx_RAF));
fprintf('Peak MEK-P:          %.4f nM at t = %.2f minutes\n', max_MEK_P, t(idx_MEK));
fprintf('Peak ERK-P:          %.4f nM at t = %.2f minutes\n', max_ERK_P, t(idx_ERK));

%% ============================================================================
% SAVE ERK-P FOR USE IN DOWNSTREAM MODELS
% ============================================================================

% Save time and ERK_P (active ERK) for use in downstream models
save('mapk_output.mat', 't', 'ERK_P');
fprintf('\nSaved ERK-P values to mapk_output.mat\n');
fprintf('  Time points: %d\n', length(t));
fprintf('  ERK-P range: %.4f to %.4f nM\n', min(ERK_P), max(ERK_P));
fprintf('  Peak ERK-P: %.4f nM at t = %.2f minutes\n', max(ERK_P), t(ERK_P == max(ERK_P)));

%% ============================================================================
% ODE FUNCTION
% ============================================================================

function dydt = mapk_cascade_odes(t, y, p)
    % ODE system for RAS-RAF-MEK-ERK (MAPK) signaling cascade
    % y = [RAS_GDP, RAS_GTP, RAF, RAF_star, MEK, MEK_P, ERK, ERK_P]
    % p = parameters structure
    
    % Extract species
    RAS_GDP = y(1);
    RAS_GTP = y(2);
    RAF = y(3);
    RAF_star = y(4);
    MEK = y(5);
    MEK_P = y(6);
    ERK = y(7);
    ERK_P = y(8);
    
    % Get SOS_star input signal at current time
    SOS_star = p.SOS_star_func(t);
    
    % ========================================================================
    % REACTION 1: RAS-GDP + SOS* → RAS-GTP + SOS* (RAS activation)
    % ========================================================================
    % SOS* catalyzes GDP-GTP exchange on RAS
    % Rate: k_SOS * SOS_star * RAS_GDP
    r1 = p.k_SOS * SOS_star * RAS_GDP;
    
    % ========================================================================
    % REACTION 2: RAS-GTP → RAS-GDP (GTP hydrolysis)
    % ========================================================================
    % Intrinsic and GAP-mediated GTP hydrolysis
    % Rate: k_GTPase * RAS_GTP
    r2 = p.k_GTPase * RAS_GTP;
    
    % ========================================================================
    % REACTION 3: RAS-GTP + RAF → RAF* (RAF activation)
    % ========================================================================
    % RAS-GTP recruits and activates RAF (simplified as single step)
    % Rate: k_RAF_act * RAS_GTP * RAF
    r3 = p.k_RAF_act * RAS_GTP * RAF;
    
    % ========================================================================
    % REACTION 4: RAF* → RAF (dephosphorylation)
    % ========================================================================
    % RAF* deactivation
    % Rate: k_RAF_deact * RAF_star
    r4 = p.k_RAF_deact * RAF_star;
    
    % ========================================================================
    % REACTION 5: RAF* + MEK → RAF* + MEK-P (MEK phosphorylation)
    % ========================================================================
    % RAF* phosphorylates MEK (RAF* acts as catalyst, not consumed)
    % Rate: k_MEK * RAF_star * MEK
    r5 = p.k_MEK * RAF_star * MEK;
    
    % ========================================================================
    % REACTION 6: MEK-P → MEK (dephosphorylation)
    % ========================================================================
    % MEK-P deactivation
    % Rate: k_MEK_deact * MEK_P
    r6 = p.k_MEK_deact * MEK_P;
    
    % ========================================================================
    % REACTION 7: MEK-P + ERK → MEK-P + ERK-P (ERK phosphorylation)
    % ========================================================================
    % MEK-P phosphorylates ERK (MEK-P acts as catalyst, not consumed)
    % Rate: k_ERK * MEK_P * ERK
    r7 = p.k_ERK * MEK_P * ERK;
    
    % ========================================================================
    % REACTION 8: ERK-P → ERK (dephosphorylation)
    % ========================================================================
    % ERK-P deactivation
    % Rate: k_ERK_deact * ERK_P
    r8 = p.k_ERK_deact * ERK_P;
    
    % ========================================================================
    % ODEs FOR EACH SPECIES
    % ========================================================================
    
    % d[RAS-GDP]/dt: consumed by activation, produced by GTP hydrolysis
    dRAS_GDP_dt = -r1 + r2;
    
    % d[RAS-GTP]/dt: produced by activation, consumed by GTP hydrolysis and RAF activation
    dRAS_GTP_dt = r1 - r2 - r3;
    
    % d[RAF]/dt: consumed by activation, produced by deactivation
    dRAF_dt = -r3 + r4;
    
    % d[RAF*]/dt: produced by activation, consumed by deactivation
    dRAF_star_dt = r3 - r4;
    
    % d[MEK]/dt: consumed by phosphorylation, produced by dephosphorylation
    dMEK_dt = -r5 + r6;
    
    % d[MEK-P]/dt: produced by phosphorylation, consumed by dephosphorylation
    dMEK_P_dt = r5 - r6;
    
    % d[ERK]/dt: consumed by phosphorylation, produced by dephosphorylation
    dERK_dt = -r7 + r8;
    
    % d[ERK-P]/dt: produced by phosphorylation, consumed by dephosphorylation
    dERK_P_dt = r7 - r8;
    
    % Return derivatives
    dydt = [dRAS_GDP_dt; dRAS_GTP_dt; dRAF_dt; dRAF_star_dt; ...
            dMEK_dt; dMEK_P_dt; dERK_dt; dERK_P_dt];
end

