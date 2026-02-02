% =============================================================================
% CANCER SIGNALING PATHWAY MODEL (PINN-BASED)
% =============================================================================
% 
% Use this script to simulate the Melanoma signaling pathway dynamics 
% using the ODE system and learned kinetic parameters from PINN training.
%
% This model includes MAPK and PI3K pathways with comprehensive feedback 
% and crosstalk mechanisms, matching the physics structure used for training.
% =============================================================================

clear all;
close all;
clc;

%% 1. Learned Kinetic Parameters
% These values were derived from training the Physics-Informed Neural Network (PINN).
k.IC50            = 0.6555;
k.IC50_vem        = 0.6793;
k.Km              = 0.1081;
k.Km_akt_mek      = 1.1644;
k.Km_dusp         = 0.6443;
k.Km_erk_rtk      = 0.4050;
k.Km_s6k          = 0.5116;
k.hill_coeff      = 1.8186;
k.k_4ebp1_act     = 0.6895;
k.k_4ebp1_comp    = 0.3664;
k.k_4ebp1_deg     = 0.5349;
k.k_akt_act       = 0.7001;
k.k_akt_deg       = 0.2700;
k.k_akt_mek       = 0.1388;
k.k_akt_raf       = 0.7912;
k.k_akt_rtk       = 0.2472;
k.k_craf_act      = 1.0962;
k.k_craf_deg      = 0.5246;
k.k_dusp_cat      = 0.6068;
k.k_dusp_deg      = 0.5608;
k.k_dusp_synth    = 0.6695;
k.k_egfr_dephos   = 0.5073;
k.k_egfr_phos     = 0.0000;
k.k_erk_act       = 1.1760;
k.k_erk_deg       = 0.4415;
k.k_erk_pi3k      = 0.7909;
k.k_erk_rtk       = 0.2067;
k.k_erk_sos       = 0.4892;
k.k_her_dephos    = 0.1521;
k.k_her_phos      = 0.0000;
k.k_igf_dephos    = 0.0818;
k.k_igf_phos      = -0.0001;
k.k_mek_act       = 0.7730;
k.k_mek_deg       = 0.3550;
k.k_mek_inhib     = 0.1508;
k.k_raf_pi3k      = -0.0696;
k.k_s6k_act       = 0.7736;
k.k_s6k_deg       = 0.6233;
k.k_s6k_irs       = 0.6331;
k.k_s6k_mtor      = 0.2536;
k.k_vem_paradox   = -0.0001;
k.n_dusp          = 2.0639;
k.vem_optimal     = 0.1469;

% Structural constants
k.w_her3 = 1.5; 

%% 2. Simulation Environment
% Drug Concentrations (normalized [0, 1])
Vemurafenib = 1.0; 
Trametinib = 0.0;
PI3K_inhibitor = 1.0;

% Time range (hours)
t_span = [0 48];
t_eval = linspace(0, 48, 200);

% Initial Conditions (Scaled values at t=0)
% [pEGFR, HER2, HER3, IGF1R, pCRAF, pMEK, pERK, DUSP6, pAKT, pS6K, p4EBP1]
y0 = [0.85, 0.75, 0.60, 0.90, 0.50, 0.40, 0.30, 0.70, 0.45, 0.35, 0.55];

%% 3. Numerical Integration (ODE15s)
[T, Y] = ode15s(@(t, y) cancer_signaling_ode(t, y, k, Vemurafenib, Trametinib, PI3K_inhibitor), t_eval, y0);

%% 4. Data Visualization
figure('Name', 'Melanoma Signaling Pathway Dynamics', 'Position', [50, 50, 1600, 1000]);
species = {'pEGFR', 'pHER2', 'pHER3', 'pIGF1R', 'pCRAF', 'pMEK', 'pERK', 'DUSP6', 'pAKT', 'pS6K', 'p4EBP1'};
colors = lines(11);

for i = 1:11
    subplot(3, 4, i);
    plot(T, Y(:, i), 'LineWidth', 2.5, 'Color', colors(i, :));
    xlabel('Time (hours)', 'FontSize', 10);
    ylabel('Signal (norm)', 'FontSize', 10);
    title(species{i}, 'FontSize', 12, 'FontWeight', 'bold');
    grid on;
    ylim([0, max(1.1, max(Y(:,i)) * 1.2)]);
    set(gca, 'FontSize', 10);
end

sgtitle(sprintf('PINN-Based Model Simulation | Vem: %.1f, Tram: %.1f, PI3Ki: %.1f', ...
        Vemurafenib, Trametinib, PI3K_inhibitor), 'FontSize', 16, 'FontWeight', 'bold');

%% 5. Mathematical Model (ODE System)
function dydt = cancer_signaling_ode(t, y, k, Vem, Tram, PI3Ki)
    % Species Mapping
    pEGFR = y(1);  HER2 = y(2);  HER3 = y(3);  IGF1R = y(4);
    pCRAF = y(5);  pMEK = y(6);  pERK = y(7);  DUSP6 = y(8);
    pAKT = y(9);   pS6K = y(10); p4EBP1 = y(11);

    % Common terms
    n = k.hill_coeff;
    Km = k.Km;
    eps = 1e-8;

    % --- 5.1 Drug Effects ---
    Tram_effect = (Tram^n) / (k.IC50^n + Tram^n + eps);
    PI3Ki_effect = (PI3Ki^n) / (k.IC50^n + PI3Ki^n + eps);
    Vem_term = (Vem^n) / (k.IC50_vem^n + Vem^n + eps);
    Vem_activation = k.k_vem_paradox * Vem * exp(-((Vem - k.vem_optimal)^2) / (2 * 0.15^2));

    % --- 5.2 Negative Feedbacks ---
    % ERK to Receptor Feedback
    ERK_feedback = (k.k_erk_rtk * pERK) / (k.Km_erk_rtk + pERK + eps);
    
    % ERK-mediated SOS inhibition
    ERK_to_SOS_inh = (k.k_erk_sos * pERK) / (Km + pERK + eps);
    
    % AKT-mediated RTK feedback
    AKT_to_RTK_feed = (k.k_akt_rtk * pAKT) / (Km + pAKT + eps);

    % --- 5.3 RTK Dynamics ---
    % ERK -ve feedback increases the unphosphorylated (inactive) pool by 
    % accelerating the dephosphorylation rate of active receptors.
    % Unphosphorylated Conc = (1 - active_receptor)
    dpEGFR = k.k_egfr_phos * (1.0 - pEGFR) - (k.k_egfr_dephos + ERK_feedback) * pEGFR;
    dHER2  = k.k_her_phos * (1.0 - HER2) - (k.k_her_dephos + ERK_feedback) * HER2;
    dHER3  = k.k_her_phos * (1.0 - HER3) - (k.k_her_dephos + ERK_feedback) * HER3;
    dIGF1R = k.k_igf_phos * (1.0 - IGF1R) - (k.k_igf_dephos + ERK_feedback) * IGF1R;

    % Integrated RTK Signal
    RTK_total = pEGFR + HER2 + k.w_her3 * HER3 + IGF1R;
    
    % Ras Activation (Influenced by SOS inhibition and AKT feedback)
    RAS_GTP = RTK_total * (1.0 - ERK_to_SOS_inh) * (1.0 - AKT_to_RTK_feed);

    % --- 5.4 MAPK Pathway (Strict Mass Action) ---
    AKT_to_RAF_inh = (k.k_akt_raf * pAKT) / (Km + pAKT + eps);
    % Activation of CRAF pool (1 - pCRAF)
    dpCRAF = (k.k_craf_act * RAS_GTP * (1.0 - Vem_term) + Vem_activation) * (1.0 - pCRAF) ...
             - (k.k_craf_deg + AKT_to_RAF_inh) * pCRAF;

    AKT_to_MEK_prom = (k.k_akt_mek * pAKT) / (k.Km_akt_mek + pAKT + eps);
    MEK_sub_inh = (k.k_mek_inhib * pMEK) / (Km + pMEK + eps);
    % Activation of MEK pool (1 - pMEK)
    dpMEK = (k.k_mek_act * pCRAF * (1.0 - Tram_effect) * (1.0 - AKT_to_RAF_inh) + AKT_to_MEK_prom) * (1.0 - pMEK) ...
            - (k.k_mek_deg + MEK_sub_inh) * pMEK;

    DUSP6_activity = (k.k_dusp_cat * DUSP6) / (Km + DUSP6 + eps);
    % Activation of ERK pool (1 - pERK)
    dpERK = k.k_erk_act * pMEK * (1.0 - pERK) - (k.k_erk_deg + DUSP6_activity) * pERK;

    % Net Synthesis of DUSP6
    DUSP6_synth = (k.k_dusp_synth * (pERK^k.n_dusp)) / (k.Km_dusp^k.n_dusp + pERK^k.n_dusp + eps);
    dDUSP6 = DUSP6_synth - k.k_dusp_deg * DUSP6;

    % --- 5.5 PI3K/AKT/mTOR Pathway (Strict Mass Action) ---
    % mTOR/S6K Feedback
    S6K_to_IRS1_inh = (k.k_s6k_irs * pS6K) / (k.Km_s6k + pS6K + eps);
    S6K_to_mTOR_feed = (k.k_s6k_mtor * pS6K) / (Km + pS6K + eps);
    mTOR_cumulative_feed = S6K_to_IRS1_inh + S6K_to_mTOR_feed;
    
    ERK_to_PI3K_inh = (k.k_erk_pi3k * pERK) / (Km + pERK + eps);
    RAF_to_PI3K_act = (k.k_raf_pi3k * pCRAF) / (Km + pCRAF + eps);
    
    PI3K_input = RTK_total * (1.0 - ERK_to_PI3K_inh) * (1.0 - mTOR_cumulative_feed);
    % Activation of AKT pool (1 - pAKT)
    dpAKT = k.k_akt_act * (PI3K_input + RAF_to_PI3K_act) * (1.0 - PI3Ki_effect) * (1.0 - pAKT) ...
            - (k.k_akt_deg + mTOR_cumulative_feed) * pAKT;

    mTOR_4EBP1_comp = (k.k_4ebp1_comp * p4EBP1) / (Km + p4EBP1 + eps);
    mTOR_activity = pAKT * (1.0 - mTOR_4EBP1_comp);
    % Activation of S6K pool (1 - pS6K)
    dpS6K = k.k_s6k_act * mTOR_activity * (1.0 - pS6K) - k.k_s6k_deg * pS6K;
    
    % Activation of 4EBP1 pool (1 - p4EBP1)
    dp4EBP1 = k.k_4ebp1_act * pAKT * (1.0 - p4EBP1) - k.k_4ebp1_deg * p4EBP1;

    dydt = [dpEGFR; dHER2; dHER3; dIGF1R; dpCRAF; dpMEK; dpERK; dDUSP6; dpAKT; dpS6K; dp4EBP1];
end
