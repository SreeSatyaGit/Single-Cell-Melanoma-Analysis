% =============================================================================
% PI3K/AKT/mTOR SIGNALING PATHWAY MODEL
% Complete Pathway from RTK to pS6K and p4EBP1 with Negative Feedback
% =============================================================================
%
% Model Description:
% This script models the PI3K/AKT/mTOR signaling cascade including:
% - PI3K activation by phosphorylated RTKs (pEGFR/pHER3)
% - PIP2 to PIP3 conversion
% - AKT recruitment and dual phosphorylation (Thr308, Ser473)
% - TSC1/TSC2 inhibition and Rheb-GTP accumulation
% - mTORC1 activation
% - S6K and 4EBP1 phosphorylation
% - Negative feedback loops (S6K→IRS1/PI3K, PTEN)
%
% Key Features:
% - AKT requires dual phosphorylation (Thr308 by PDK1, Ser473 by mTORC2)
% - mTORC1 phosphorylates S6K (Thr389) and 4EBP1 (Thr37/46)
% - S6K provides negative feedback on PI3K activation
% - PTEN dephosphorylates PIP3 back to PIP2
%
% =============================================================================

clear all;
close all;
clc;

fprintf('═══════════════════════════════════════════════════════════════════════════\n');
fprintf('   PI3K/AKT/mTOR SIGNALING PATHWAY MODEL\n');
fprintf('═══════════════════════════════════════════════════════════════════════════\n\n');

%% ============================================================================
% INPUT SIGNAL: pEGFR(t) or pHER3(t)
% ============================================================================

% Define pRTK (pEGFR or pHER3) as a function of time
% Option 1: Transient pulse (exponential decay)
pRTK_func = @(t) 2.0 * exp(-0.05 * t) .* (t >= 0);

% Option 2: Step function (constant)
% pRTK_func = @(t) 1.5 * (t >= 0);

% Option 3: Load from previous simulation (e.g., from EGFR-HER3 model)
% if exist('egfr_output.mat', 'file')
%     load('egfr_output.mat', 't', 'EEp', 'EHp');
%     t_rtk = t;
%     % Combine pEGFR and pHER3 (or use individually)
%     pRTK_data = EEp + 0.5 * EHp;  % Example: weighted combination
%     pRTK_func = @(t) interp1(t_rtk, pRTK_data, t, 'linear', pRTK_data(end));
% end

fprintf('Using pRTK (pEGFR/pHER3) as input signal.\n\n');

%% ============================================================================
% PARAMETERS
% ============================================================================

fprintf('Setting up parameters...\n');

% ============================================================================
% PI3K Module Parameters
% ============================================================================

% Reaction 1: pRTK + PI3K → pRTK:PI3K → PI3K_active
k_PI3K_act = 0.5;   % min^-1 - PI3K activation rate by pRTK (base rate)
k_PI3K_deact = 0.05; % min^-1 - PI3K deactivation rate

% Reaction 2: PI3K_active converts PIP2 → PIP3
k_PIP2_to_PIP3 = 0.5; % min^-1 - PIP2 to PIP3 conversion rate

% Reaction 3: PTEN dephosphorylates PIP3 → PIP2
k_PTEN = 0.1;       % min^-1 - PTEN-mediated PIP3 dephosphorylation rate

% ============================================================================
% AKT Module Parameters
% ============================================================================

% Reaction 4: PIP3 recruits AKT to membrane (simplified as activation step)
k_AKT_recruit = 0.3; % min^-1 - AKT recruitment rate by PIP3

% Reaction 5: PDK1 phosphorylates AKT at Thr308
k_AKT_Thr308 = 0.5; % min^-1 - AKT Thr308 phosphorylation rate by PDK1

% Reaction 6: mTORC2 phosphorylates AKT at Ser473
k_AKT_Ser473 = 0.3; % min^-1 - AKT Ser473 phosphorylation rate by mTORC2

% Note: Fully active AKT requires both Thr308 and Ser473 phosphorylation
% We'll track: AKT, pAKT_Thr308, pAKT_Ser473, pAKT_full (both sites)

% Reaction 7: AKT dephosphorylation
k_AKT_deact = 0.05; % min^-1 - AKT dephosphorylation rate

% ============================================================================
% TSC1/TSC2 and Rheb Module Parameters
% ============================================================================

% Reaction 8: pAKT phosphorylates TSC2 → inhibits TSC1/TSC2 complex
k_TSC2_inh = 0.5;   % min^-1 - TSC2 inhibition rate by pAKT

% Reaction 9: TSC1/TSC2 inhibition → Rheb-GTP accumulation
k_Rheb_act = 0.5;   % min^-1 - Rheb-GTP activation rate

% Reaction 10: Rheb-GTP → Rheb-GDP (GTP hydrolysis)
k_Rheb_GTPase = 0.1; % min^-1 - Rheb GTPase activity

% ============================================================================
% mTORC1 Module Parameters
% ============================================================================

% Reaction 11: Rheb-GTP activates mTORC1
k_mTORC1_act = 0.5; % min^-1 - mTORC1 activation rate by Rheb-GTP

% Reaction 12: mTORC1 deactivation
k_mTORC1_deact = 0.05; % min^-1 - mTORC1 deactivation rate

% ============================================================================
% S6K Module Parameters
% ============================================================================

% Reaction 13: mTORC1 phosphorylates S6K at Thr389
k_S6K = 0.5;        % min^-1 - S6K Thr389 phosphorylation rate by mTORC1

% Reaction 14: S6K dephosphorylation
k_S6K_deact = 0.05; % min^-1 - pS6K dephosphorylation rate

% ============================================================================
% 4EBP1 Module Parameters
% ============================================================================

% Reaction 15: mTORC1 phosphorylates 4EBP1 at Thr37/46
k_4EBP1 = 0.5;      % min^-1 - 4EBP1 Thr37/46 phosphorylation rate by mTORC1

% Reaction 16: 4EBP1 dephosphorylation
k_4EBP1_deact = 0.05; % min^-1 - p4EBP1 dephosphorylation rate

% ============================================================================
% Negative Feedback Parameters
% ============================================================================

alpha1 = 0.5;       % Feedback strength: pS6K inhibition of PI3K activation

% Store parameters in structure
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
params.alpha1 = alpha1;
params.pRTK_func = pRTK_func;

fprintf('Parameters configured.\n\n');

%% ============================================================================
% INITIAL CONDITIONS
% ============================================================================

fprintf('Setting initial conditions...\n');

% PI3K species
PI3K0 = 5.0;           % nM - inactive PI3K
PI3K_active0 = 0.0;     % nM - active PI3K

% PIP species
PIP2_0 = 10.0;          % nM - PIP2 (substrate)
PIP3_0 = 0.0;           % nM - PIP3 (product)

% AKT species (simplified: track AKT, pAKT_Thr308, pAKT_Ser473, pAKT_full)
AKT0 = 5.0;             % nM - unphosphorylated AKT
pAKT_Thr308_0 = 0.0;    % nM - AKT phosphorylated at Thr308 only
pAKT_Ser473_0 = 0.0;    % nM - AKT phosphorylated at Ser473 only
pAKT_full_0 = 0.0;      % nM - AKT phosphorylated at both sites (fully active)

% TSC1/TSC2 and Rheb species
TSC2_active0 = 5.0;     % nM - active TSC2 (inhibits Rheb)
Rheb_total = 5.0;       % nM - total Rheb pool (constant)
Rheb_GTP0 = 0.0;        % nM - active Rheb-GTP

% mTORC1 species
mTORC1_0 = 5.0;         % nM - inactive mTORC1
mTORC1_active0 = 0.0;   % nM - active mTORC1

% S6K species
S6K0 = 5.0;             % nM - unphosphorylated S6K
pS6K0 = 0.0;            % nM - S6K phosphorylated at Thr389

% 4EBP1 species
EBP1_0 = 5.0;           % nM - unphosphorylated 4EBP1
p4EBP1_0 = 0.0;         % nM - 4EBP1 phosphorylated at Thr37/46

% Initial state vector: [PI3K, PI3K_active, PIP2, PIP3, AKT, pAKT_Thr308,
%                        pAKT_Ser473, pAKT_full, TSC2_active, Rheb_GTP,
%                        mTORC1, mTORC1_active, S6K, pS6K, 4EBP1, p4EBP1]
y0 = [PI3K0; PI3K_active0; PIP2_0; PIP3_0; AKT0; pAKT_Thr308_0; ...
      pAKT_Ser473_0; pAKT_full_0; TSC2_active0; Rheb_GTP0; ...
      mTORC1_0; mTORC1_active0; S6K0; pS6K0; EBP1_0; p4EBP1_0];

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
[t, y] = ode45(@(t,y) pi3k_akt_mtor_odes(t, y, params), tspan, y0, options);

% Extract species concentrations
PI3K = y(:, 1);
PI3K_active = y(:, 2);
PIP2 = y(:, 3);
PIP3 = y(:, 4);
AKT = y(:, 5);
pAKT_Thr308 = y(:, 6);
pAKT_Ser473 = y(:, 7);
pAKT_full = y(:, 8);  % Fully active AKT (both sites phosphorylated)
TSC2_active = y(:, 9);
Rheb_GTP = y(:, 10);
mTORC1 = y(:, 11);
mTORC1_active = y(:, 12);
S6K = y(:, 13);
pS6K = y(:, 14);
EBP1 = y(:, 15);
p4EBP1 = y(:, 16);

% Calculate pRTK input signal over time
pRTK_signal = arrayfun(pRTK_func, t);

fprintf('Simulation completed successfully!\n\n');

%% ============================================================================
% PLOTTING
% ============================================================================

fprintf('═══════════════════════════════════════════════════════════════════════════\n');
fprintf('   GENERATING VISUALIZATIONS\n');
fprintf('═══════════════════════════════════════════════════════════════════════════\n\n');

% Figure 1: Key Species Time Courses
figure('Position', [100, 100, 1400, 1000]);

% Plot 1: PI3K and PIP species
subplot(3, 3, 1);
plot(t, PI3K, 'b-', 'LineWidth', 2, 'DisplayName', 'PI3K');
hold on;
plot(t, PI3K_active, 'r-', 'LineWidth', 2, 'DisplayName', 'PI3K_active');
xlabel('Time (minutes)', 'FontSize', 10);
ylabel('Concentration (nM)', 'FontSize', 10);
title('PI3K Species', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 9);
grid on;
xlim([0, 200]);
hold off;

subplot(3, 3, 2);
plot(t, PIP2, 'b-', 'LineWidth', 2, 'DisplayName', 'PIP2');
hold on;
plot(t, PIP3, 'r-', 'LineWidth', 2, 'DisplayName', 'PIP3');
xlabel('Time (minutes)', 'FontSize', 10);
ylabel('Concentration (nM)', 'FontSize', 10);
title('PIP2/PIP3 Species', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 9);
grid on;
xlim([0, 200]);
hold off;

% Plot 2: AKT species
subplot(3, 3, 3);
plot(t, AKT, 'b-', 'LineWidth', 1.5, 'DisplayName', 'AKT');
hold on;
plot(t, pAKT_Thr308, 'g-', 'LineWidth', 1.5, 'DisplayName', 'pAKT(Thr308)');
plot(t, pAKT_Ser473, 'm-', 'LineWidth', 1.5, 'DisplayName', 'pAKT(Ser473)');
plot(t, pAKT_full, 'r-', 'LineWidth', 2.5, 'DisplayName', 'pAKT(Full)');
xlabel('Time (minutes)', 'FontSize', 10);
ylabel('Concentration (nM)', 'FontSize', 10);
title('AKT Phosphorylation States', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 8);
grid on;
xlim([0, 200]);
hold off;

% Plot 3: TSC2 and Rheb
subplot(3, 3, 4);
plot(t, TSC2_active, 'b-', 'LineWidth', 2, 'DisplayName', 'TSC2_active');
hold on;
plot(t, Rheb_GTP, 'r-', 'LineWidth', 2, 'DisplayName', 'Rheb-GTP');
xlabel('Time (minutes)', 'FontSize', 10);
ylabel('Concentration (nM)', 'FontSize', 10);
title('TSC2/Rheb Species', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 9);
grid on;
xlim([0, 200]);
hold off;

% Plot 4: mTORC1
subplot(3, 3, 5);
plot(t, mTORC1, 'b-', 'LineWidth', 2, 'DisplayName', 'mTORC1');
hold on;
plot(t, mTORC1_active, 'r-', 'LineWidth', 2, 'DisplayName', 'mTORC1_active');
xlabel('Time (minutes)', 'FontSize', 10);
ylabel('Concentration (nM)', 'FontSize', 10);
title('mTORC1 Species', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 9);
grid on;
xlim([0, 200]);
hold off;

% Plot 5: S6K
subplot(3, 3, 6);
plot(t, S6K, 'b-', 'LineWidth', 2, 'DisplayName', 'S6K');
hold on;
plot(t, pS6K, 'r-', 'LineWidth', 2, 'DisplayName', 'pS6K(Thr389)');
xlabel('Time (minutes)', 'FontSize', 10);
ylabel('Concentration (nM)', 'FontSize', 10);
title('S6K Species', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 9);
grid on;
xlim([0, 200]);
hold off;

% Plot 6: 4EBP1
subplot(3, 3, 7);
plot(t, EBP1, 'b-', 'LineWidth', 2, 'DisplayName', '4EBP1');
hold on;
plot(t, p4EBP1, 'r-', 'LineWidth', 2, 'DisplayName', 'p4EBP1(Thr37/46)');
xlabel('Time (minutes)', 'FontSize', 10);
ylabel('Concentration (nM)', 'FontSize', 10);
title('4EBP1 Species', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 9);
grid on;
xlim([0, 200]);
hold off;

% Plot 7: Input signal
subplot(3, 3, 8);
plot(t, pRTK_signal, 'k-', 'LineWidth', 2);
xlabel('Time (minutes)', 'FontSize', 10);
ylabel('Concentration (nM)', 'FontSize', 10);
title('Input Signal: pRTK', 'FontSize', 12, 'FontWeight', 'bold');
grid on;
xlim([0, 200]);

% Plot 8: Cascade progression
subplot(3, 3, 9);
plot(t, pAKT_full, 'b-', 'LineWidth', 2, 'DisplayName', 'pAKT(Full)');
hold on;
plot(t, mTORC1_active, 'g-', 'LineWidth', 2, 'DisplayName', 'mTORC1_active');
plot(t, pS6K, 'm-', 'LineWidth', 2, 'DisplayName', 'pS6K');
plot(t, p4EBP1, 'r-', 'LineWidth', 2, 'DisplayName', 'p4EBP1');
xlabel('Time (minutes)', 'FontSize', 10);
ylabel('Concentration (nM)', 'FontSize', 10);
title('Cascade Progression', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 9);
grid on;
xlim([0, 200]);
hold off;

sgtitle('PI3K/AKT/mTOR Signaling Pathway', 'FontSize', 16, 'FontWeight', 'bold');

% Figure 2: Detailed Analysis
figure('Position', [150, 150, 1400, 800]);

% Plot 1: Pathway flow
subplot(2, 2, 1);
plot(t, PIP3, 'c-', 'LineWidth', 2, 'DisplayName', 'PIP3');
hold on;
plot(t, pAKT_full, 'b-', 'LineWidth', 2, 'DisplayName', 'pAKT(Full)');
plot(t, Rheb_GTP, 'g-', 'LineWidth', 2, 'DisplayName', 'Rheb-GTP');
plot(t, mTORC1_active, 'm-', 'LineWidth', 2, 'DisplayName', 'mTORC1_active');
xlabel('Time (minutes)', 'FontSize', 12);
ylabel('Concentration (nM)', 'FontSize', 12);
title('Pathway Flow: PI3K → AKT → mTORC1', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 10);
grid on;
xlim([0, 200]);
hold off;

% Plot 2: Downstream outputs
subplot(2, 2, 2);
plot(t, pS6K, 'r-', 'LineWidth', 2.5, 'DisplayName', 'pS6K(Thr389)');
hold on;
plot(t, p4EBP1, 'b-', 'LineWidth', 2.5, 'DisplayName', 'p4EBP1(Thr37/46)');
xlabel('Time (minutes)', 'FontSize', 12);
ylabel('Concentration (nM)', 'FontSize', 12);
title('mTORC1 Downstream Outputs', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 10);
grid on;
xlim([0, 200]);
hold off;

% Plot 3: Feedback effects
subplot(2, 2, 3);
% Calculate effective PI3K activation rate (with feedback)
k_PI3K_eff = k_PI3K_act ./ (1 + alpha1 * pS6K);

plot(t, k_PI3K_eff, 'r-', 'LineWidth', 2, 'DisplayName', 'k_{PI3K,eff} (with feedback)');
hold on;
yline(k_PI3K_act, 'b--', 'LineWidth', 1.5, 'DisplayName', 'k_{PI3K} (base rate)');
xlabel('Time (minutes)', 'FontSize', 12);
ylabel('Effective Rate Constant', 'FontSize', 12);
title('S6K Feedback on PI3K Activation', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 10);
grid on;
xlim([0, 200]);
hold off;

% Plot 4: Input-output relationship
subplot(2, 2, 4);
yyaxis left;
plot(t, pRTK_signal, 'k-', 'LineWidth', 2, 'DisplayName', 'pRTK (input)');
ylabel('pRTK Concentration (nM)', 'FontSize', 12);
yyaxis right;
plot(t, pAKT_full, 'b-', 'LineWidth', 2, 'DisplayName', 'pAKT(Full)');
hold on;
plot(t, pS6K, 'r-', 'LineWidth', 2, 'DisplayName', 'pS6K');
plot(t, p4EBP1, 'm-', 'LineWidth', 2, 'DisplayName', 'p4EBP1');
ylabel('Concentration (nM)', 'FontSize', 12);
xlabel('Time (minutes)', 'FontSize', 12);
title('Input-Output Relationship', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 10);
grid on;
xlim([0, 200]);
hold off;

sgtitle('PI3K/AKT/mTOR Pathway: Detailed Analysis', 'FontSize', 16, 'FontWeight', 'bold');

%% ============================================================================
% SUMMARY STATISTICS
% ============================================================================

fprintf('═══════════════════════════════════════════════════════════════════════════\n');
fprintf('   SUMMARY STATISTICS\n');
fprintf('═══════════════════════════════════════════════════════════════════════════\n\n');

fprintf('KEY SPECIES PEAK VALUES:\n');
[max_PIP3, idx_PIP3] = max(PIP3);
[max_pAKT_full, idx_pAKT] = max(pAKT_full);
[max_Rheb_GTP, idx_Rheb] = max(Rheb_GTP);
[max_mTORC1_active, idx_mTOR] = max(mTORC1_active);
[max_pS6K, idx_S6K] = max(pS6K);
[max_p4EBP1, idx_4EBP1] = max(p4EBP1);

fprintf('  PIP3:               %.4f nM at t = %.2f min\n', max_PIP3, t(idx_PIP3));
fprintf('  pAKT(Full):         %.4f nM at t = %.2f min\n', max_pAKT_full, t(idx_pAKT));
fprintf('  Rheb-GTP:           %.4f nM at t = %.2f min\n', max_Rheb_GTP, t(idx_Rheb));
fprintf('  mTORC1_active:      %.4f nM at t = %.2f min\n', max_mTORC1_active, t(idx_mTOR));
fprintf('  pS6K(Thr389):       %.4f nM at t = %.2f min\n', max_pS6K, t(idx_S6K));
fprintf('  p4EBP1(Thr37/46):   %.4f nM at t = %.2f min\n', max_p4EBP1, t(idx_4EBP1));

fprintf('\nSTEADY-STATE VALUES (at t = 200 min):\n');
fprintf('  PIP3:               %.4f nM\n', PIP3(end));
fprintf('  pAKT(Full):         %.4f nM\n', pAKT_full(end));
fprintf('  Rheb-GTP:           %.4f nM\n', Rheb_GTP(end));
fprintf('  mTORC1_active:      %.4f nM\n', mTORC1_active(end));
fprintf('  pS6K(Thr389):       %.4f nM\n', pS6K(end));
fprintf('  p4EBP1(Thr37/46):   %.4f nM\n', p4EBP1(end));

fprintf('\nFEEDBACK STRENGTHS:\n');
fprintf('  alpha1 (S6K → PI3K): %.2f\n', alpha1);
fprintf('  PTEN activity:       %.2f min^-1\n', k_PTEN);

fprintf('\n═══════════════════════════════════════════════════════════════════════════\n');
fprintf('   SIMULATION COMPLETED SUCCESSFULLY!\n');
fprintf('═══════════════════════════════════════════════════════════════════════════\n\n');

%% ============================================================================
% SAVE OUTPUTS
% ============================================================================

save('pi3k_akt_mtor_output.mat', 't', 'pAKT_full', 'mTORC1_active', ...
     'pS6K', 'p4EBP1', 'PIP3', 'Rheb_GTP', 'pRTK_signal');
fprintf('Saved outputs to pi3k_akt_mtor_output.mat\n');

%% ============================================================================
% ODE FUNCTION
% ============================================================================

function dydt = pi3k_akt_mtor_odes(t, y, p)
    % ODE system for PI3K/AKT/mTOR signaling pathway
    % y = [PI3K, PI3K_active, PIP2, PIP3, AKT, pAKT_Thr308, pAKT_Ser473,
    %      pAKT_full, TSC2_active, Rheb_GTP, mTORC1, mTORC1_active,
    %      S6K, pS6K, 4EBP1, p4EBP1]
    % p = parameters structure
    
    % Extract species
    PI3K = y(1);
    PI3K_active = y(2);
    PIP2 = y(3);
    PIP3 = y(4);
    AKT = y(5);
    pAKT_Thr308 = y(6);
    pAKT_Ser473 = y(7);
    pAKT_full = y(8);
    TSC2_active = y(9);
    Rheb_GTP = y(10);
    mTORC1 = y(11);
    mTORC1_active = y(12);
    S6K = y(13);
    pS6K = y(14);
    EBP1 = y(15);
    p4EBP1 = y(16);
    
    % Get pRTK input signal at current time
    pRTK = p.pRTK_func(t);
    
    % ========================================================================
    % CALCULATE FEEDBACK-MODULATED RATE CONSTANTS
    % ========================================================================
    
    % Feedback: pS6K reduces PI3K activation (S6K → IRS1/PI3K feedback)
    k_PI3K_act_eff = p.k_PI3K_act / (1 + p.alpha1 * pS6K);
    
    % ========================================================================
    % PI3K MODULE REACTIONS
    % ========================================================================
    
    % Reaction 1: pRTK + PI3K → PI3K_active
    r1 = k_PI3K_act_eff * pRTK * PI3K;  % Feedback-modulated
    
    % Reaction 2: PI3K_active → PI3K (deactivation)
    r2 = p.k_PI3K_deact * PI3K_active;
    
    % Reaction 3: PI3K_active converts PIP2 → PIP3
    r3 = p.k_PIP2_to_PIP3 * PI3K_active * PIP2;
    
    % Reaction 4: PTEN dephosphorylates PIP3 → PIP2
    r4 = p.k_PTEN * PIP3;
    
    % ========================================================================
    % AKT MODULE REACTIONS
    % ========================================================================
    
    % Reaction 5: PIP3 recruits AKT (simplified: AKT → pAKT_Thr308)
    % Note: In reality, PIP3 recruits AKT, then PDK1 phosphorylates Thr308
    r5 = p.k_AKT_recruit * PIP3 * AKT;  % Creates pAKT_Thr308
    
    % Reaction 6: PDK1 phosphorylates AKT at Thr308
    r6 = p.k_AKT_Thr308 * PIP3 * AKT;  % Alternative pathway
    
    % Reaction 7: mTORC2 phosphorylates pAKT_Thr308 at Ser473 → pAKT_full
    r7 = p.k_AKT_Ser473 * pAKT_Thr308;
    
    % Reaction 8: mTORC2 can also phosphorylate AKT directly at Ser473
    r8 = p.k_AKT_Ser473 * PIP3 * AKT;  % Creates pAKT_Ser473
    
    % Reaction 9: pAKT_Ser473 can be phosphorylated at Thr308 → pAKT_full
    r9 = p.k_AKT_Thr308 * pAKT_Ser473;
    
    % Reaction 10: Dephosphorylation reactions
    r10_Thr308 = p.k_AKT_deact * pAKT_Thr308;  % pAKT_Thr308 → AKT
    r10_Ser473 = p.k_AKT_deact * pAKT_Ser473;  % pAKT_Ser473 → AKT
    r10_full = p.k_AKT_deact * pAKT_full;      % pAKT_full → can go to either intermediate
    
    % ========================================================================
    % TSC1/TSC2 AND RHEB MODULE REACTIONS
    % ========================================================================
    
    % Reaction 11: pAKT_full phosphorylates TSC2 → inhibits TSC1/TSC2
    r11 = p.k_TSC2_inh * pAKT_full * TSC2_active;
    
    % Reaction 12: Rheb-GTP accumulation from Rheb pool (inhibited by active TSC2)
    % Assume constant Rheb_total pool, so Rheb_GDP = Rheb_total - Rheb_GTP
    % TSC2 acts as GAP, so when TSC2 is active, Rheb activation is reduced
    Rheb_GDP = 5.0 - Rheb_GTP;  % Constant total Rheb pool (5.0 nM)
    r12 = p.k_Rheb_act * Rheb_GDP / (1 + TSC2_active);
    
    % Reaction 13: Rheb-GTP → Rheb-GDP (GTP hydrolysis, enhanced by active TSC2)
    % TSC2_active acts as GAP, promoting GTP hydrolysis
    r13 = p.k_Rheb_GTPase * Rheb_GTP * (1 + 5 * TSC2_active);  % TSC2 strongly promotes hydrolysis
    
    % ========================================================================
    % mTORC1 MODULE REACTIONS
    % ========================================================================
    
    % Reaction 14: Rheb-GTP activates mTORC1
    r14 = p.k_mTORC1_act * Rheb_GTP * mTORC1;
    
    % Reaction 15: mTORC1 deactivation
    r15 = p.k_mTORC1_deact * mTORC1_active;
    
    % ========================================================================
    % S6K MODULE REACTIONS
    % ========================================================================
    
    % Reaction 16: mTORC1_active phosphorylates S6K at Thr389
    r16 = p.k_S6K * mTORC1_active * S6K;
    
    % Reaction 17: S6K dephosphorylation
    r17 = p.k_S6K_deact * pS6K;
    
    % ========================================================================
    % 4EBP1 MODULE REACTIONS
    % ========================================================================
    
    % Reaction 18: mTORC1_active phosphorylates 4EBP1 at Thr37/46
    r18 = p.k_4EBP1 * mTORC1_active * EBP1;
    
    % Reaction 19: 4EBP1 dephosphorylation
    r19 = p.k_4EBP1_deact * p4EBP1;
    
    % ========================================================================
    % ODEs FOR EACH SPECIES
    % ========================================================================
    
    % PI3K module
    dPI3K_dt = -r1 + r2;
    dPI3K_active_dt = r1 - r2;
    
    % PIP species
    dPIP2_dt = -r3 + r4;
    dPIP3_dt = r3 - r4;
    
    % AKT module (simplified phosphorylation scheme)
    % AKT can be phosphorylated at Thr308 or Ser473, then both sites → fully active
    dAKT_dt = -r5 - r6 - r8 + r10_Thr308 + r10_Ser473 + r10_full;
    dpAKT_Thr308_dt = r5 + r6 - r7 - r10_Thr308;
    dpAKT_Ser473_dt = r8 - r9 - r10_Ser473;
    dpAKT_full_dt = r7 + r9 - r10_full;
    
    % TSC2/Rheb module
    dTSC2_active_dt = -r11;
    dRheb_GTP_dt = r12 - r13;
    
    % mTORC1 module
    dmTORC1_dt = -r14 + r15;
    dmTORC1_active_dt = r14 - r15;
    
    % S6K module
    dS6K_dt = -r16 + r17;
    dpS6K_dt = r16 - r17;
    
    % 4EBP1 module
    dEBP1_dt = -r18 + r19;
    dp4EBP1_dt = r18 - r19;
    
    % Return derivatives
    dydt = [dPI3K_dt; dPI3K_active_dt; dPIP2_dt; dPIP3_dt; ...
            dAKT_dt; dpAKT_Thr308_dt; dpAKT_Ser473_dt; dpAKT_full_dt; ...
            dTSC2_active_dt; dRheb_GTP_dt; dmTORC1_dt; dmTORC1_active_dt; ...
            dS6K_dt; dpS6K_dt; dEBP1_dt; dp4EBP1_dt];
end

