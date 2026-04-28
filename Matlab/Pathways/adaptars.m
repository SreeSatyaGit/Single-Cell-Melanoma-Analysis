% =============================================================================
% SHC-GRB2-SOS ADAPTOR-MEDIATED SIGNAL TRANSDUCTION CASCADE
% Ordinary Differential Equations (ODEs) Model
% =============================================================================
%
% Model Description:
% This script models the adaptor-mediated signal transduction cascade
% downstream of phosphorylated EGFR (pEGFR), involving:
% - Shc recruitment and phosphorylation by pEGFR
% - Grb2 binding to phosphorylated Shc
% - SOS recruitment and activation via Grb2
% - SOS* deactivation
%
% Species:
% Shc          = inactive Shc adaptor protein
% pEGFR_Shc    = pEGFR:Shc complex (transient)
% pShc         = phosphorylated Shc
% Grb2         = unbound Grb2 adaptor protein
% pShc_Grb2    = pShc:Grb2 complex
% SOS          = inactive SOS (GEF)
% pShc_Grb2_SOS = pShc:Grb2:SOS ternary complex
% SOS_star     = active SOS (SOS*)
%
% Input Signal:
% pEGFR(t) = phosphorylated EGFR concentration over time (external input)
%
% =============================================================================

close all;
clc;

%% ============================================================================
% INPUT SIGNAL: pEGFR(t) - Load from EGFR-HER3 model
% ============================================================================

% Load pEGFR (EEp) from previous EGFR-HER3 model simulation
if exist('egfr_output.mat', 'file')
    % Load with different variable names to avoid conflicts
    load('egfr_output.mat', 't', 'EEp');
    t_egfr = t;  % Rename to avoid conflict with anonymous function parameter
    EEp_egfr = EEp;
    
    fprintf('Loaded pEGFR (EEp) from egfr_output.mat\n');
    fprintf('  Time points: %d\n', length(t_egfr));
    fprintf('  EEp range: %.4f to %.4f nM\n', min(EEp_egfr), max(EEp_egfr));
    
    % Create interpolation function
    % Use loaded t_egfr and EEp_egfr, function parameter is t
    pEGFR_func = @(t) interp1(t_egfr, EEp_egfr, t, 'linear', EEp_egfr(end));
    
    fprintf('Using pEGFR from EGFR-HER3 model as input signal.\n');
else
    % Fallback: Use exponential decay if file not found
    warning('egfr_output.mat not found. Using default exponential decay for pEGFR.');
    pEGFR_func = @(t) 2.0 * exp(-0.05 * t) .* (t >= 0);
end

%% ============================================================================
% PARAMETERS
% ============================================================================

% Reaction 1: pEGFR + Shc ⇌ pEGFR:Shc → pEGFR + pShc
k_on1 = 0.1;      % nM^-1 min^-1 - Shc binding to pEGFR
k_off1 = 0.05;    % min^-1 - Dissociation of pEGFR:Shc
k_cat1 = 0.3;     % min^-1 - Phosphorylation rate (Shc → pShc)

% Reaction 2: pShc + Grb2 ⇌ pShc:Grb2
k_on2 = 0.15;     % nM^-1 min^-1 - Grb2 binding to pShc
k_off2 = 0.08;    % min^-1 - Dissociation of pShc:Grb2

% Reaction 3: pShc:Grb2 + SOS ⇌ pShc:Grb2:SOS → pShc:Grb2 + SOS*
k_on3 = 0.12;     % nM^-1 min^-1 - SOS binding to pShc:Grb2
k_off3 = 0.06;    % min^-1 - Dissociation of ternary complex
k_cat3 = 0.25;    % min^-1 - SOS activation rate (SOS → SOS*)

% Reaction 4: SOS* → SOS (deactivation)
k_deg4 = 0.02;    % min^-1 - SOS* deactivation rate

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
params.pEGFR_func = pEGFR_func;

%% ============================================================================
% INITIAL CONDITIONS
% ============================================================================

Shc0 = 10.0;          % nM - inactive Shc
pEGFR_Shc0 = 0.0;     % nM - pEGFR:Shc complex (initially zero)
pShc0 = 0.0;          % nM - phosphorylated Shc (initially zero)
Grb2_0 = 8.0;         % nM - unbound Grb2
pShc_Grb2_0 = 0.0;    % nM - pShc:Grb2 complex (initially zero)
SOS_0 = 5.0;          % nM - inactive SOS
pShc_Grb2_SOS_0 = 0.0; % nM - ternary complex (initially zero)
SOS_star0 = 0.0;      % nM - active SOS (initially zero)

% Initial state vector: [Shc, pEGFR_Shc, pShc, Grb2, pShc_Grb2, SOS, pShc_Grb2_SOS, SOS_star]
y0 = [Shc0; pEGFR_Shc0; pShc0; Grb2_0; pShc_Grb2_0; SOS_0; pShc_Grb2_SOS_0; SOS_star0];

%% ============================================================================
% TIME SPAN
% ============================================================================

tspan = [0, 200];  % minutes

%% ============================================================================
% SOLVE ODE SYSTEM
% ============================================================================

fprintf('Solving ODE system for Shc-Grb2-SOS cascade...\n');
options = odeset('RelTol', 1e-6, 'AbsTol', 1e-8);
[t, y] = ode45(@(t,y) shc_grb2_sos_odes(t, y, params), tspan, y0, options);

% Extract species concentrations
Shc = y(:, 1);
pEGFR_Shc = y(:, 2);
pShc = y(:, 3);
Grb2 = y(:, 4);
pShc_Grb2 = y(:, 5);
SOS = y(:, 6);
pShc_Grb2_SOS = y(:, 7);
SOS_star = y(:, 8);

% Calculate pEGFR input signal over time
pEGFR_signal = arrayfun(pEGFR_func, t);

fprintf('Simulation completed.\n');

%% ============================================================================
% PLOTTING
% ============================================================================

% Create figure for main plots
figure('Position', [100, 100, 1400, 900]);

% Plot 1: Shc species (Shc, pShc, pEGFR:Shc)
subplot(2, 3, 1);
plot(t, Shc, 'b-', 'LineWidth', 2, 'DisplayName', 'Shc');
hold on;
plot(t, pShc, 'r-', 'LineWidth', 2, 'DisplayName', 'pShc');
plot(t, pEGFR_Shc, 'g--', 'LineWidth', 1.5, 'DisplayName', 'pEGFR:Shc');
xlabel('Time (minutes)', 'FontSize', 11);
ylabel('Concentration (nM)', 'FontSize', 11);
title('Shc Species Dynamics', 'FontSize', 13, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 9);
grid on;
xlim([0, 200]);
hold off;

% Plot 2: Grb2 species (Grb2, pShc:Grb2)
subplot(2, 3, 2);
plot(t, Grb2, 'b-', 'LineWidth', 2, 'DisplayName', 'Grb2');
hold on;
plot(t, pShc_Grb2, 'r-', 'LineWidth', 2, 'DisplayName', 'pShc:Grb2');
xlabel('Time (minutes)', 'FontSize', 11);
ylabel('Concentration (nM)', 'FontSize', 11);
title('Grb2 Species Dynamics', 'FontSize', 13, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 9);
grid on;
xlim([0, 200]);
hold off;

% Plot 3: SOS species (SOS, SOS*, pShc:Grb2:SOS)
subplot(2, 3, 3);
plot(t, SOS, 'b-', 'LineWidth', 2, 'DisplayName', 'SOS (inactive)');
hold on;
plot(t, SOS_star, 'r-', 'LineWidth', 2, 'DisplayName', 'SOS* (active)');
plot(t, pShc_Grb2_SOS, 'g--', 'LineWidth', 1.5, 'DisplayName', 'pShc:Grb2:SOS');
xlabel('Time (minutes)', 'FontSize', 11);
ylabel('Concentration (nM)', 'FontSize', 11);
title('SOS Species Dynamics', 'FontSize', 13, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 9);
grid on;
xlim([0, 200]);
hold off;

% Plot 4: Input signal pEGFR(t)
subplot(2, 3, 4);
plot(t, pEGFR_signal, 'k-', 'LineWidth', 2);
xlabel('Time (minutes)', 'FontSize', 11);
ylabel('Concentration (nM)', 'FontSize', 11);
title('Input Signal: pEGFR(t)', 'FontSize', 13, 'FontWeight', 'bold');
grid on;
xlim([0, 200]);

% Plot 5: Key output - Active SOS (SOS*)
subplot(2, 3, 5);
plot(t, SOS_star, 'r-', 'LineWidth', 2.5);
xlabel('Time (minutes)', 'FontSize', 11);
ylabel('Concentration (nM)', 'FontSize', 11);
title('Active SOS (SOS*) - Key Output', 'FontSize', 13, 'FontWeight', 'bold');
grid on;
xlim([0, 200]);

% Plot 6: All species overview
subplot(2, 3, 6);
plot(t, Shc, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Shc');
hold on;
plot(t, pShc, 'r-', 'LineWidth', 1.5, 'DisplayName', 'pShc');
plot(t, Grb2, 'c-', 'LineWidth', 1.5, 'DisplayName', 'Grb2');
plot(t, pShc_Grb2, 'm-', 'LineWidth', 1.5, 'DisplayName', 'pShc:Grb2');
plot(t, SOS, 'g-', 'LineWidth', 1.5, 'DisplayName', 'SOS');
plot(t, SOS_star, 'k-', 'LineWidth', 2, 'DisplayName', 'SOS*');
plot(t, pShc_Grb2_SOS, 'y-', 'LineWidth', 1.5, 'DisplayName', 'pShc:Grb2:SOS');
xlabel('Time (minutes)', 'FontSize', 11);
ylabel('Concentration (nM)', 'FontSize', 11);
title('All Species Overview', 'FontSize', 13, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 8);
grid on;
xlim([0, 200]);
hold off;

sgtitle('Shc-Grb2-SOS Adaptor-Mediated Signal Transduction Cascade', ...
        'FontSize', 15, 'FontWeight', 'bold');

% Create separate figure for detailed comparison
figure('Position', [150, 150, 1200, 600]);

% Comparison plot: Input vs Output
subplot(1, 2, 1);
yyaxis left;
plot(t, pEGFR_signal, 'k-', 'LineWidth', 2, 'DisplayName', 'pEGFR (input)');
ylabel('pEGFR Concentration (nM)', 'FontSize', 12);
yyaxis right;
plot(t, SOS_star, 'r-', 'LineWidth', 2, 'DisplayName', 'SOS* (output)');
ylabel('SOS* Concentration (nM)', 'FontSize', 12);
xlabel('Time (minutes)', 'FontSize', 12);
title('Input-Output Relationship', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 10);
grid on;
xlim([0, 200]);

% Cascade progression
subplot(1, 2, 2);
plot(t, pShc, 'b-', 'LineWidth', 2, 'DisplayName', 'pShc');
hold on;
plot(t, pShc_Grb2, 'g-', 'LineWidth', 2, 'DisplayName', 'pShc:Grb2');
plot(t, pShc_Grb2_SOS, 'm-', 'LineWidth', 2, 'DisplayName', 'pShc:Grb2:SOS');
plot(t, SOS_star, 'r-', 'LineWidth', 2.5, 'DisplayName', 'SOS*');
xlabel('Time (minutes)', 'FontSize', 12);
ylabel('Concentration (nM)', 'FontSize', 12);
title('Cascade Progression', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 10);
grid on;
xlim([0, 200]);
hold off;

%% ============================================================================
% STEADY-STATE ANALYSIS
% ============================================================================

fprintf('\n=== Steady-State Concentrations (at t = 200 min) ===\n');
fprintf('Shc:                %.4f nM\n', Shc(end));
fprintf('pEGFR:Shc:          %.4f nM\n', pEGFR_Shc(end));
fprintf('pShc:               %.4f nM\n', pShc(end));
fprintf('Grb2:               %.4f nM\n', Grb2(end));
fprintf('pShc:Grb2:          %.4f nM\n', pShc_Grb2(end));
fprintf('SOS:                %.4f nM\n', SOS(end));
fprintf('pShc:Grb2:SOS:      %.4f nM\n', pShc_Grb2_SOS(end));
fprintf('SOS* (active):      %.4f nM\n', SOS_star(end));

fprintf('\n=== Conservation Checks ===\n');
total_Shc = Shc(end) + pEGFR_Shc(end) + pShc(end);
fprintf('Total Shc:          %.4f nM (initial: %.1f nM)\n', total_Shc, Shc0);

total_Grb2 = Grb2(end) + pShc_Grb2(end) + pShc_Grb2_SOS(end);
fprintf('Total Grb2:         %.4f nM (initial: %.1f nM)\n', total_Grb2, Grb2_0);

total_SOS = SOS(end) + pShc_Grb2_SOS(end) + SOS_star(end);
fprintf('Total SOS:          %.4f nM (initial: %.1f nM)\n', total_SOS, SOS_0);

fprintf('\n=== Peak Activation ===\n');
[max_SOS_star, idx_max] = max(SOS_star);
t_max = t(idx_max);
fprintf('Peak SOS* concentration: %.4f nM at t = %.2f minutes\n', max_SOS_star, t_max);

%% ============================================================================
% SAVE pSOS (SOS_star) FOR USE IN DOWNSTREAM MODELS
% ============================================================================

% Save time and SOS_star (active SOS) for use in downstream models (e.g., Ras activation)
save('sos_output.mat', 't', 'SOS_star');
fprintf('\nSaved pSOS (SOS_star) values to sos_output.mat\n');
fprintf('  Time points: %d\n', length(t));
fprintf('  SOS_star range: %.4f to %.4f nM\n', min(SOS_star), max(SOS_star));
fprintf('  Peak SOS_star: %.4f nM at t = %.2f minutes\n', max(SOS_star), t(SOS_star == max(SOS_star)));

%% ============================================================================
% ODE FUNCTION
% ============================================================================

function dydt = shc_grb2_sos_odes(t, y, p)
    % ODE system for Shc-Grb2-SOS adaptor-mediated signal transduction cascade
    % y = [Shc, pEGFR_Shc, pShc, Grb2, pShc_Grb2, SOS, pShc_Grb2_SOS, SOS_star]
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
    
    % Get pEGFR input signal at current time
    pEGFR = p.pEGFR_func(t);
    
    % ========================================================================
    % REACTION 1: pEGFR + Shc ⇌ pEGFR:Shc → pEGFR + pShc
    % ========================================================================
    % Forward: pEGFR + Shc → pEGFR:Shc
    r1_forward = p.k_on1 * pEGFR * Shc;
    
    % Reverse: pEGFR:Shc → pEGFR + Shc
    r1_reverse = p.k_off1 * pEGFR_Shc;
    
    % Catalytic: pEGFR:Shc → pEGFR + pShc (phosphorylation)
    r1_cat = p.k_cat1 * pEGFR_Shc;
    
    % ========================================================================
    % REACTION 2: pShc + Grb2 ⇌ pShc:Grb2
    % ========================================================================
    % Forward: pShc + Grb2 → pShc:Grb2
    r2_forward = p.k_on2 * pShc * Grb2;
    
    % Reverse: pShc:Grb2 → pShc + Grb2
    r2_reverse = p.k_off2 * pShc_Grb2;
    
    % ========================================================================
    % REACTION 3: pShc:Grb2 + SOS ⇌ pShc:Grb2:SOS → pShc:Grb2 + SOS*
    % ========================================================================
    % Forward: pShc:Grb2 + SOS → pShc:Grb2:SOS
    r3_forward = p.k_on3 * pShc_Grb2 * SOS;
    
    % Reverse: pShc:Grb2:SOS → pShc:Grb2 + SOS
    r3_reverse = p.k_off3 * pShc_Grb2_SOS;
    
    % Catalytic: pShc:Grb2:SOS → pShc:Grb2 + SOS* (SOS activation)
    r3_cat = p.k_cat3 * pShc_Grb2_SOS;
    
    % ========================================================================
    % REACTION 4: SOS* → SOS (deactivation)
    % ========================================================================
    r4 = p.k_deg4 * SOS_star;
    
    % ========================================================================
    % ODEs FOR EACH SPECIES
    % ========================================================================
    
    % d[Shc]/dt: consumed by binding to pEGFR, produced by dissociation
    dShc_dt = -r1_forward + r1_reverse;
    
    % d[pEGFR:Shc]/dt: formed from pEGFR+Shc, lost by dissociation and phosphorylation
    dpEGFR_Shc_dt = r1_forward - r1_reverse - r1_cat;
    
    % d[pShc]/dt: produced by phosphorylation, consumed by binding to Grb2
    dpShc_dt = r1_cat - r2_forward + r2_reverse;
    
    % d[Grb2]/dt: consumed by binding to pShc, produced by dissociation
    dGrb2_dt = -r2_forward + r2_reverse;
    
    % d[pShc:Grb2]/dt: formed from pShc+Grb2, lost by dissociation and SOS binding
 % Also produced when ternary complex dissociates or activates
    dpShc_Grb2_dt = r2_forward - r2_reverse - r3_forward + r3_reverse + r3_cat;
    
    % d[SOS]/dt: consumed by binding to pShc:Grb2, produced by dissociation and deactivation
    dSOS_dt = -r3_forward + r3_reverse + r4;
    
    % d[pShc:Grb2:SOS]/dt: formed from pShc:Grb2+SOS, lost by dissociation and activation
    dpShc_Grb2_SOS_dt = r3_forward - r3_reverse - r3_cat;
    
    % d[SOS*]/dt: produced by activation, lost by deactivation
    dSOS_star_dt = r3_cat - r4;
    
    % Return derivatives
    dydt = [dShc_dt; dpEGFR_Shc_dt; dpShc_dt; dGrb2_dt; ...
            dpShc_Grb2_dt; dSOS_dt; dpShc_Grb2_SOS_dt; dSOS_star_dt];
end

