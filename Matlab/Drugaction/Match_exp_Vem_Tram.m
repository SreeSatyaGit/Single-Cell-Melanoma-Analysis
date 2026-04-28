% == == == == == == == == == == == == == == == == == == == == == == == == == ==
    == == == ==
    % MAPK / PI3K Plotting Script — v9 % Matches MAPK_optimizer_v9.m exactly.%
        Fits 9 species : pEGFR,
    pCRAF, pMEK, pERK, DUSP6, pAKT, % IGF1R, HER2,
    HER3 % == == == == == == == == == == == == == == == == == == == == == == ==
        == == == == == == == close all;
clc;

% % == == == == == == == == == == == == == == == == == == == == == == == == ==
    == == == == == == == == ==
    % % SECTION 2 — UNPACK PARAMETERS(77 elements) % % == == == == == == == ==
    == == == == == == == == == == == == == == == == == == == == == == == == ==
    == p = optimizedParams;

ka1 = 1e-4;
kr1 = 1e-3;
kc1 = 1e-2;
kpCraf = 1e-4;
kpMek = 2e-6;
kpErk = 4e-5;
kDegradEgfr = p(7);
kErkInbEgfr = p(8);
kShcDephos = p(9);
kptpDeg = p(10);
kGrb2CombShc = p(11);
kSprtyInbGrb2 = p(12);
kSosCombGrb2 = 5e-8;
kErkPhosSos = 3e-4;
kErkPhosPcraf = 8.57e-5;
kPcrafDegrad = p(16);
kErkPhosMek = 1.07e-5;
kMekDegrad = p(18);
kDuspInbErk = 2.73e-5;
kErkDeg = p(20);
kinbBraf = p(21);
kDuspStop = p(22);
kDusps = p(23);
kSproutyForm = p(24);
kSprtyComeDown = p(25);
kdegrad = p(26);
km_Sprty_decay = p(27);
km_Dusp = p(28);
km_Sprty = p(29);
kEGFRPI3k = p(30);
kMTOR_Feedback = 1e-4;
Ki_vemu = p(32);
n_vemu = p(33);
Vemu = p(34);
kHER2_EGFR_form = p(35);
kHER2_EGFR_diss = p(36);
kNRG_bind = p(37);
NRG = p(38);
kHER2_HER3_form = p(39);
kHER2_HER3_diss = p(40);
kHER23_PI3K = p(41);
kHER2EGFR_Grb2 = p(42);
kRasAct = p(43);
kRasGAP = 1e-4;
kBCform = p(45);
kBCdiss = p(46);
kBCact = p(47);
kHER2EGFR_Shc = p(48);
kHER23_Shc = p(49);
kIGF1R_Shc = p(50);
knf1_KRAS = 1e-6;
Km_nf1_KRAS = p(52);
knf1_HRAS = p(53);
Km_nf1_HRAS = p(54);
knf1_NRAS = p(55);
Km_nf1_NRAS = p(56);
kSynHER2 = p(57);
Km_HER2 = p(58);
kSynHER3 = p(59);
Km_HER3 = p(60);
kSynIGF1R = p(61);
Km_IGF1R = p(62);
kSynFoxo = 8e-3;
kDegFoxo = 5e-4;
Ki_foxo = p(65);
n_foxo = p(66);
Km_foxo = p(67);
kDegRTK = p(68);
Ki_tram = 1e-4;
n_tram = 1;
Tram = 1e-4;
kIRS_PI3K = p(72);
kRasPI3K = p(73);
kPI3K_AKT = p(74);
kPTEN = p(75);
Ki_PTEN = p(76);
kDephos_AKT = p(77);

% % == == == == == == == == == == == == == == == == == == == == == == == == ==
    == == == == == == == == == % % SECTION 3 — EXPERIMENTAL DATA % % == == == ==
    == == == == == == == == == == == == == == == == == == == == == == == == ==
    == == == == == timeStamps = [ 0, 1, 4, 8, 24, 48 ];

pEGFR_Exp = [
  0.222379739, 0.622877159, 0.629217784, 0.533530834, 0.022513609, 0.010036399
];
pCRAF_Exp = [
  0.234376572, 0.641878896, 0.567434544, 0.406320223, 0.582899195, 0.25113447
];
pMEK_Exp = [
  1.936660577, 0.029380652, 0.012873835, 0.03390921, 0.095155796, 0.944936578
];
pERK_Exp = [
  3.273353557, 0.075717978, 0.011570416, 0.00642985, 0.041863585, 0.91621491
];
DUSP_Exp = [
  2.854207662, 2.842703936, 1.163746208, 0.332720449, 0.030434242, 0.094073888
];
expAKT = [
  0.527301325, 0.614645732, 0.95895017, 0.895019432, 0.412820453, 0.269891704
];

IGF1R_Exp = [
  1.180034579, 0.967927178, 0.808905442, 0.781013289, 0.41928501, 0.870763253
];
Her2_Exp = [
  0.306924546, 0.275751955, 0.32171108, 0.23070312, 1.013023288, 1.045536401
];
Her3_Exp = [
  0.295284147, 0.285719072, 0.385045943, 0.582261781, 0.751301308, 0.264889608
];

% Normalised for plotting
pEGFR_norm = rescale(pEGFR_Exp);
pCRAF_norm = rescale(pCRAF_Exp);
pMEK_norm = rescale(pMEK_Exp);
pERK_norm = rescale(pERK_Exp);
DUSP_norm = rescale(DUSP_Exp);
AKT_norm = rescale(expAKT);
IGF1R_norm = rescale(IGF1R_Exp);
HER2_norm = rescale(Her2_Exp);
HER3_norm = rescale(Her3_Exp);

% % == == == == == == == == == == == == == == == == == == == == == == == == ==
    == == == == == == == == ==
    % % SECTION 4 — INITIAL CONDITIONS(50 states) %
        % All 9 fitted species initialised to normalised experimental t =
    0 values % % == == == == == == == == == == == == == == == == == == == == ==
    == == == == == == == == == == == == == RAS_GTP_frac = 0.1;

EGFR = [ 1 - pEGFR_norm(1), 0, pEGFR_norm(1) ];
SHC = [ 1, 0, 1 ];
grb_sos = [ 0, 0 ];
HRAS = [ 0, 0 ];
NRAS = [ 0, 0 ];
KRAS = [ 1 - RAS_GTP_frac, RAS_GTP_frac, 1 ];
CRAF = [ 1 - pCRAF_norm(1), pCRAF_norm(1) ];
BRAF = [ 0, 1 ];
MEK = [ 1, pMEK_norm(1) ];
% inactive = 1(substrate pool full)ERK = [ 1 - pERK_norm(1), pERK_norm(1) ];
DUSP = [ 0.1, DUSP_norm(1) ];
% y(24) = active DUSP SRTY = [ 1, 1 ];
pERKDegrad = [1];
pMEKDegrad = [1];
pCRAFDegrad = [1];
DUSPStop = [1];
IGFR = [ 1 - IGF1R_norm(1), 0, IGF1R_norm(1) ];
IRS = [ 1, 0 ];
pi3k = [ 1, 0 ];
AKT = [ 1 - AKT_norm(1), AKT_norm(1) ];
FOXO = [0];
MTORC = [ 1, 0 ];
HER2 = [ HER2_norm(1), 0 ];
HER3 = [ 1 - HER3_norm(1), HER3_norm(1), 0 ];
% conservation : inactive + NRG - bound + dimer BCRAF_dimer =
    [ 1 - pCRAF_norm(1), 1 - pCRAF_norm(1) ];

y0 = [EGFR, SHC, grb_sos, HRAS, NRAS, KRAS, CRAF, BRAF, MEK, ERK, ...
      DUSP, SRTY, pERKDegrad, pMEKDegrad, pCRAFDegrad, DUSPStop, ...
      IGFR, IRS, pi3k, AKT, FOXO, MTORC, ...
      HER2, HER3, BCRAF_dimer];
% 50 states

        % %
    == == == == == == == == == == == == == == == == == == == == == == == == ==
    == == == == == == == == == % % SECTION 5 — SOLVE ODE % % == == == == == ==
    == == == == == == == == == == == == == == == == == == == == == == == == ==
    == == == tStamps = timeStamps * 3600;
tFine = sort(unique([ linspace(0, 48 * 3600, 2000), tStamps ]));
odeOpts = odeset('RelTol', 1e-7, 'AbsTol', 1e-10, 'MaxStep', 500);

fprintf('Solving ODE...\n');
[ T, Y ] = ode15s(
    @(t, y) Mapk(
        t, y, ... ka1, kr1, kc1, kpCraf, kpMek, kpErk, ... kDegradEgfr,
        kErkInbEgfr, kShcDephos, kptpDeg, ... kGrb2CombShc, kSprtyInbGrb2,
        kSosCombGrb2, kErkPhosSos, ... kErkPhosPcraf, kPcrafDegrad, kErkPhosMek,
        kMekDegrad, ... kDuspInbErk, kErkDeg, kinbBraf, kDuspStop, kDusps,
        ... kSproutyForm, kSprtyComeDown, kdegrad, km_Sprty_decay, ... km_Dusp,
        km_Sprty, kEGFRPI3k, kMTOR_Feedback, ... Ki_vemu, n_vemu, Vemu,
        ... kHER2_EGFR_form, kHER2_EGFR_diss, kNRG_bind, NRG,
        ... kHER2_HER3_form, kHER2_HER3_diss, kHER23_PI3K, kHER2EGFR_Grb2,
        ... kRasAct, kRasGAP, ... kBCform, kBCdiss, kBCact, ... kHER2EGFR_Shc,
        kHER23_Shc, kIGF1R_Shc, ... knf1_KRAS, Km_nf1_KRAS, knf1_HRAS,
        Km_nf1_HRAS, knf1_NRAS, Km_nf1_NRAS, ... kSynHER2, Km_HER2, kSynHER3,
        Km_HER3, kSynIGF1R, Km_IGF1R, ... kSynFoxo, kDegFoxo, Ki_foxo, n_foxo,
        Km_foxo, kDegRTK, ... Ki_tram, n_tram, Tram, kIRS_PI3K, kRasPI3K,
        kPI3K_AKT, ... kPTEN, Ki_PTEN, kDephos_AKT),
    ... tFine, y0, odeOpts);

fprintf('Done. %d time points.\n', length(T));

T_h = T / 3600;
ts_h = timeStamps;
C = @(col) max(Y( :, col), 0);

% % == == == == == == == == == == == == == == == == == == == == == == == == ==
    == == == == == == == == ==
    % % SECTION 5b — AUC CALCULATIONS(pAKT and pERK) % % == == == == == == == ==
    == == == == == == == == == == == == == == == == == == == == == == == == ==
    ==

    % Model AUC — full time course AUC_pERK_model = trapz(T_h, C(23));
AUC_pAKT_model = trapz(T_h, C(40));

% Model AUC — normalised(comparable to experiment)
        AUC_pERK_model_norm = trapz(T_h, rescale_safe(C(23)));
AUC_pAKT_model_norm = trapz(T_h, rescale_safe(C(40)));

% Experimental AUC — sparse time points only AUC_pERK_exp =
    trapz(ts_h, pERK_norm);
AUC_pAKT_exp = trapz(ts_h, AKT_norm);

fprintf('\n===== AUC Summary =====\n');
fprintf('  pERK  — Model (raw):    %.4f\n', AUC_pERK_model);
fprintf('  pERK  — Model (norm):   %.4f\n', AUC_pERK_model_norm);
fprintf('  pERK  — Experiment:     %.4f\n', AUC_pERK_exp);
fprintf('  pAKT  — Model (raw):    %.4f\n', AUC_pAKT_model);
fprintf('  pAKT  — Model (norm):   %.4f\n', AUC_pAKT_model_norm);
fprintf('  pAKT  — Experiment:     %.4f\n', AUC_pAKT_exp);
fprintf('=======================\n\n');

% % == == == == == == == == == == == == == == == == == == == == == == == == ==
    == == == == == == == == == % % SECTION 6 — PLOT SETTINGS % % == == == == ==
    == == == == == == == == == == == == == == == == == == == == == == == == ==
    == == == == lw = 2.5;
ms = 9;
mfc = [0.92 0.25 0.20];
mec = [0.10 0.28 0.60];
mk = {'d', 'MarkerSize', ms, 'MarkerFaceColor', mfc, 'MarkerEdgeColor',
      mec, 'LineWidth',  1.5};

CL = struct(... 'EGFR', [0.00 0.45 0.70], ... 'CRAF', [0.90 0.62 0.00],
            ... 'MEK', [0.00 0.62 0.45], ... 'ERK', [0.80 0.10 0.40],
            ... 'DUSP', [0.35 0.70 0.35], ... 'AKT', [0.60 0.40 0.80],
            ... 'HRAS', [0.20 0.60 0.85], ... 'NRAS', [0.90 0.45 0.10],
            ... 'KRAS', [0.20 0.75 0.45], ... 'HER2', [0.85 0.20 0.50],
            ... 'HER3', [0.40 0.20 0.80], ... 'IGFR', [0.70 0.50 0.10],
            ... 'FOXO', [0.55 0.35 0.70], ... 'DIMER', [0.85 0.35 0.10]);

% % == == == == == == == == == == == == == == == == == == == == == == == == ==
    == == == == == == == == ==
    % % FIGURES 1 - 6 : Original 6 fitted species — model vs experiment % % ==
    == == == == == == == == == == == == == == == == == == == == == == == == ==
    == == == == == == == == figure(1);
plot(T_h, rescale_safe(C(3)), 'Color', CL.EGFR, 'LineWidth', lw, 'DisplayName',
     'Model');
hold on;
plot(ts_h, pEGFR_norm, mk{ : }, 'DisplayName', 'Experiment');
fmt('Time (h)', 'Normalised', 'Fig 1 — pEGFR');

figure(2);
plot(T_h, rescale_safe(C(17)), 'Color', CL.CRAF, 'LineWidth', lw, 'DisplayName',
     'Model');
hold on;
plot(ts_h, pCRAF_norm, mk{ : }, 'DisplayName', 'Experiment');
fmt('Time (h)', 'Normalised', 'Fig 2 — pCRAF');

figure(3);
plot(T_h, rescale_safe(C(21)), 'Color', CL.MEK, 'LineWidth', lw, 'DisplayName',
     'Model');
hold on;
plot(ts_h, pMEK_norm, mk{ : }, 'DisplayName', 'Experiment');
fmt('Time (h)', 'Normalised', 'Fig 3 — pMEK');

figure(4);
plot(T_h, rescale_safe(C(23)), 'Color', CL.ERK, 'LineWidth', lw, 'DisplayName',
     'Model');
hold on;
plot(ts_h, pERK_norm, mk{ : }, 'DisplayName', 'Experiment');
fmt('Time (h)', 'Normalised', 'Fig 4 — pERK');

figure(5);
plot(T_h, rescale_safe(C(24)), 'Color', CL.DUSP, 'LineWidth', lw, 'DisplayName',
     'Model');
hold on;
plot(ts_h, DUSP_norm, mk{ : }, 'DisplayName', 'Experiment');
fmt('Time (h)', 'Normalised', 'Fig 5 — DUSP6');

figure(6);
plot(T_h, rescale_safe(C(40)), 'Color', CL.AKT, 'LineWidth', lw, 'DisplayName',
     'Model');
hold on;
plot(ts_h, AKT_norm, mk{ : }, 'DisplayName', 'Experiment');
fmt('Time (h)', 'Normalised', 'Fig 6 — pAKT');

% % == == == == == == == == == == == == == == == == == == == == == == == == ==
    == == == == == == == == ==
    % % FIGURES 7 - 8 : RAS isoforms and BRAF : CRAF dimer % % == == == == == ==
    == == == == == == == == == == == == == == == == == == == == == == == == ==
    == == == figure(7);
plot(T_h, C(10), 'Color', CL.HRAS, 'LineWidth', lw, 'DisplayName', 'HRAS-GTP');
hold on;
plot(T_h, C(12), 'Color', CL.NRAS, 'LineWidth', lw, 'DisplayName', 'NRAS-GTP');
plot(T_h, C(14), 'Color', CL.KRAS, 'LineWidth', lw, 'DisplayName', 'KRAS-GTP');
fmt('Time (h)', 'Amount', 'Fig 7 — Active RAS isoforms (GTP-bound)');

figure(8);
plot(T_h, C(49), '--', 'Color', CL.DIMER, 'LineWidth', lw, 'DisplayName',
     'Inactive dimer');
hold on;
plot(T_h, C(50), '-', 'Color', CL.DIMER, 'LineWidth', lw + 0.5, 'DisplayName',
     'Active dimer');
fmt('Time (h)', 'Amount', 'Fig 8 — BRAF:CRAF heterodimer');

% % == == == == == == == == == == == == == == == == == == == == == == == == ==
    == == == == == == == == ==
    % % FIGURE 9 : FOXO - driven receptor upregulation(model only) % % == == ==
    == == == == == == == == == == == == == == == == == == == == == == == == ==
    == == == == == == figure(9);
subplot(3, 1, 1);
plot(T_h, C(44), '--', 'Color', CL.HER2, 'LineWidth', lw, 'DisplayName',
     'HER2 monomer');
hold on;
plot(T_h, C(45), '-', 'Color', CL.HER2, 'LineWidth', lw, 'DisplayName',
     'HER2:EGFR dimer');
xlabel('Time (h)');
ylabel('Amount');
title('HER2');
legend;
grid on;
box on;
xlim([-1 50]);
xticks([0 1 4 8 24 48]);

subplot(3, 1, 2);
plot(T_h, C(46), '--', 'Color', CL.HER3, 'LineWidth', lw, 'DisplayName',
     'HER3 inactive');
hold on;
plot(T_h, C(48), '-', 'Color', CL.HER3, 'LineWidth', lw, 'DisplayName',
     'HER2:HER3 dimer');
xlabel('Time (h)');
ylabel('Amount');
title('HER3');
legend;
grid on;
box on;
xlim([-1 50]);
xticks([0 1 4 8 24 48]);

subplot(3, 1, 3);
plot(T_h, C(32), '--', 'Color', CL.IGFR, 'LineWidth', lw, 'DisplayName',
     'IGF1R inactive');
hold on;
plot(T_h, C(34), '-', 'Color', CL.IGFR, 'LineWidth', lw, 'DisplayName',
     'pIGF1R');
xlabel('Time (h)');
ylabel('Amount');
title('IGF1R');
legend;
grid on;
box on;
xlim([-1 50]);
xticks([0 1 4 8 24 48]);
sgtitle('Fig 9 — FOXO-driven receptor upregulation', 'FontSize', 13,
        'FontWeight', 'bold');

% % == == == == == == == == == == == == == == == == == == == == == == == == ==
    == == == == == == == == == % % FIGURE 10 : FOXO vs pAKT % % == == == == ==
    == == == == == == == == == == == == == == == == == == == == == == == == ==
    == == == == figure(10);
yyaxis left;
plot(T_h, C(41), 'Color', CL.FOXO, 'LineWidth', lw, 'DisplayName', 'FOXO');
ylabel('FOXO amount');
yyaxis right;
plot(T_h, C(40), 'Color', CL.AKT, 'LineWidth', lw, 'LineStyle', '--',
     'DisplayName', 'pAKT');
ylabel('pAKT amount');
fmt('Time (h)', '', 'Fig 10 — FOXO vs pAKT');

% % == == == == == == == == == == == == == == == == == == == == == == == == ==
    == == == == == == == == == % % FIGURES 12 - 14 : IGF1R,
    HER2,
    HER3 model vs experiment % % == == == == == == == == == == == == == == == ==
        == == == == == == == == == == == == == == == == == == figure(12);
plot(T_h, rescale_safe(C(32)), 'Color', CL.IGFR, 'LineWidth', lw, 'DisplayName',
     'Model');
hold on;
plot(ts_h, IGF1R_norm, mk{ : }, 'DisplayName', 'Experiment');
fmt('Time (h)', 'Normalised', 'Fig 12 — IGF1R');

figure(13);
plot(T_h, rescale_safe(C(44)), 'Color', CL.HER2, 'LineWidth', lw, 'DisplayName',
     'Model');
hold on;
plot(ts_h, HER2_norm, mk{ : }, 'DisplayName', 'Experiment');
fmt('Time (h)', 'Normalised', 'Fig 13 — HER2');

figure(14);
plot(T_h, rescale_safe(C(46)), 'Color', CL.HER3, 'LineWidth', lw, 'DisplayName',
     'Model');
hold on;
plot(ts_h, HER3_norm, mk{ : }, 'DisplayName', 'Experiment');
fmt('Time (h)', 'Normalised', 'Fig 14 — HER3');

% % == == == == == == == == == == == == == == == == == == == == == == == == ==
    == == == == == == == == ==
    % % FIGURE 15 : AUC Visualisation — pERK and pAKT % % == == == == == == ==
    == == == == == == == == == == == == == == == == == == == == == == == == ==
    == == figure(15);
set(gcf, 'Position', [100 100 1100 500]);

subplot(1, 2, 1);
fill([T_h; flipud(T_h)], ...[rescale_safe(C(23)); zeros(size(T_h))], ... CL.ERK,
     'FaceAlpha', 0.18, 'EdgeColor', 'none', 'DisplayName', 'Model AUC');
hold on;
plot(T_h, rescale_safe(C(23)), 'Color', CL.ERK, 'LineWidth', lw, 'DisplayName',
     ... sprintf('Model (AUC = %.2f)', AUC_pERK_model_norm));
fill([ ts_h, fliplr(ts_h) ], [ pERK_norm, zeros(1, numel(ts_h)) ], ... mec,
     'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
plot(ts_h, pERK_norm, '--', 'Color', mec, 'LineWidth', 1.2, 'HandleVisibility',
     'off');
plot(ts_h, pERK_norm, mk{ : }, 'DisplayName',
     ... sprintf('Experiment (AUC = %.2f)', AUC_pERK_exp));
text(0.97, 0.95,
     sprintf('\\DeltaAUC = %.2f', AUC_pERK_model_norm - AUC_pERK_exp),
     ... 'Units', 'normalized', 'HorizontalAlignment', 'right', ... 'FontSize',
     10, 'FontWeight', 'bold', 'Color', [0.3 0.3 0.3]);
fmt('Time (h)', 'Normalised', 'Fig 15a — pERK AUC');

subplot(1, 2, 2);
fill([T_h; flipud(T_h)], ...[rescale_safe(C(40)); zeros(size(T_h))], ... CL.AKT,
     'FaceAlpha', 0.18, 'EdgeColor', 'none', 'DisplayName', 'Model AUC');
hold on;
plot(T_h, rescale_safe(C(40)), 'Color', CL.AKT, 'LineWidth', lw, 'DisplayName',
     ... sprintf('Model (AUC = %.2f)', AUC_pAKT_model_norm));
fill([ ts_h, fliplr(ts_h) ], [ AKT_norm, zeros(1, numel(ts_h)) ], ... mec,
     'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
plot(ts_h, AKT_norm, '--', 'Color', mec, 'LineWidth', 1.2, 'HandleVisibility',
     'off');
plot(ts_h, AKT_norm, mk{ : }, 'DisplayName',
     ... sprintf('Experiment (AUC = %.2f)', AUC_pAKT_exp));
text(0.97, 0.95,
     sprintf('\\DeltaAUC = %.2f', AUC_pAKT_model_norm - AUC_pAKT_exp),
     ... 'Units', 'normalized', 'HorizontalAlignment', 'right', ... 'FontSize',
     10, 'FontWeight', 'bold', 'Color', [0.3 0.3 0.3]);
fmt('Time (h)', 'Normalised', 'Fig 15b — pAKT AUC');

sgtitle('Fig 15 — Area Under Curve: Model vs Experiment', 'FontSize', 14,
        'FontWeight', 'bold');

% % == == == == == == == == == == == == == == == == == == == == == == == == ==
    == == == == == == == == ==
    % % FIGURE 11 : Summary panel(3x4 — all 9 fitted species + FOXO) % % == ==
    == == == == == == == == == == == == == == == == == == == == == == == == ==
    == == == == == == == figure(11);
set(gcf, 'Position', [50 50 1400 1000]);

subplot(3, 4, 1);
plot(T_h, rescale_safe(C(3)), 'Color', CL.EGFR, 'LineWidth', lw);
hold on;
plot(ts_h, pEGFR_norm, mk{ : });
title('pEGFR');
grid on;
box on;
xlim([-1 50]);
xticks([0 1 4 8 24 48]);

subplot(3, 4, 2);
plot(T_h, rescale_safe(C(17)), 'Color', CL.CRAF, 'LineWidth', lw);
hold on;
plot(ts_h, pCRAF_norm, mk{ : });
title('pCRAF');
grid on;
box on;
xlim([-1 50]);
xticks([0 1 4 8 24 48]);

subplot(3, 4, 3);
plot(T_h, rescale_safe(C(21)), 'Color', CL.MEK, 'LineWidth', lw);
hold on;
plot(ts_h, pMEK_norm, mk{ : });
title('pMEK');
grid on;
box on;
xlim([-1 50]);
xticks([0 1 4 8 24 48]);

subplot(3, 4, 4);
plot(T_h, rescale_safe(C(23)), 'Color', CL.ERK, 'LineWidth', lw);
hold on;
plot(ts_h, pERK_norm, mk{ : });
title('pERK');
grid on;
box on;
xlim([-1 50]);
xticks([0 1 4 8 24 48]);

subplot(3, 4, 5);
plot(T_h, rescale_safe(C(24)), 'Color', CL.DUSP, 'LineWidth', lw);
hold on;
plot(ts_h, DUSP_norm, mk{ : });
title('DUSP6');
grid on;
box on;
xlim([-1 50]);
xticks([0 1 4 8 24 48]);

subplot(3, 4, 6);
plot(T_h, rescale_safe(C(40)), 'Color', CL.AKT, 'LineWidth', lw);
hold on;
plot(ts_h, AKT_norm, mk{ : });
title('pAKT');
grid on;
box on;
xlim([-1 50]);
xticks([0 1 4 8 24 48]);

subplot(3, 4, 7);
plot(T_h, rescale_safe(C(32)), 'Color', CL.IGFR, 'LineWidth', lw);
hold on;
plot(ts_h, IGF1R_norm, mk{ : });
title('IGF1R');
grid on;
box on;
xlim([-1 50]);
xticks([0 1 4 8 24 48]);

subplot(3, 4, 8);
plot(T_h, rescale_safe(C(44)), 'Color', CL.HER2, 'LineWidth', lw);
hold on;
plot(ts_h, HER2_norm, mk{ : });
title('HER2');
grid on;
box on;
xlim([-1 50]);
xticks([0 1 4 8 24 48]);

subplot(3, 4, 9);
plot(T_h, rescale_safe(C(46)), 'Color', CL.HER3, 'LineWidth', lw);
hold on;
plot(ts_h, HER3_norm, mk{ : });
title('HER3');
grid on;
box on;
xlim([-1 50]);
xticks([0 1 4 8 24 48]);

subplot(3, 4, 10);
plot(T_h, C(10), 'Color', CL.HRAS, 'LineWidth', lw, 'DisplayName', 'HRAS');
hold on;
plot(T_h, C(12), 'Color', CL.NRAS, 'LineWidth', lw, 'DisplayName', 'NRAS');
plot(T_h, C(14), 'Color', CL.KRAS, 'LineWidth', lw, 'DisplayName', 'KRAS');
title('RAS-GTP');
xlabel('Time (h)');
legend('FontSize', 7);
grid on;
box on;
xlim([-1 50]);
xticks([0 1 4 8 24 48]);

subplot(3, 4, 11);
plot(T_h, C(8), '--', 'Color', CL.DIMER, 'LineWidth', lw, 'DisplayName', 'SOS');
title('SOS');
xlabel('Time (h)');
legend('FontSize', 7);
grid on;
box on;
xlim([-1 50]);
xticks([0 1 4 8 24 48]);

subplot(3, 4, 12);
plot(T_h, C(41), 'Color', CL.FOXO, 'LineWidth', lw);
title('FOXO');
xlabel('Time (h)');
grid on;
box on;
xlim([-1 50]);
xticks([0 1 4 8 24 48]);

sgtitle('Fig 11 — Model Summary (9 fitted species)', 'FontSize', 14,
        'FontWeight', 'bold');

fprintf('All figures generated.\n');

% % == == == == == == == == == == == == == == == == == == == == == == == == ==
    == == == == == == == == == % % ODE FUNCTION(50 states, 77 parameters) % % ==
    == == == == == == == == == == == == == == == == == == == == == == == == ==
    == == == == == == == == function dy = Mapk(
    t, y, ... ka1, kr1, kc1, kpCraf, kpMek, kpErk, ... kDegradEgfr, kErkInbEgfr,
    kShcDephos, kptpDeg, ... kGrb2CombShc, kSprtyInbGrb2, kSosCombGrb2,
    kErkPhosSos, ... kErkPhosPcraf, kPcrafDegrad, kErkPhosMek, kMekDegrad,
    ... kDuspInbErk, kErkDeg, kinbBraf, kDuspStop, kDusps, ... kSproutyForm,
    kSprtyComeDown, kdegrad, km_Sprty_decay, km_Dusp, km_Sprty, ... kEGFRPI3k,
    kMTOR_Feedback, Ki_vemu, n_vemu, Vemu, ... kHER2_EGFR_form, kHER2_EGFR_diss,
    kNRG_bind, NRG, ... kHER2_HER3_form, kHER2_HER3_diss, kHER23_PI3K,
    kHER2EGFR_Grb2, ... kRasAct, kRasGAP, ... kBCform, kBCdiss, kBCact,
    ... kHER2EGFR_Shc, kHER23_Shc, kIGF1R_Shc, ... knf1_KRAS, Km_nf1_KRAS,
    knf1_HRAS, Km_nf1_HRAS, knf1_NRAS, Km_nf1_NRAS, ... kSynHER2, Km_HER2,
    kSynHER3, Km_HER3, kSynIGF1R, Km_IGF1R, ... kSynFoxo, kDegFoxo, Ki_foxo,
    n_foxo, Km_foxo, kDegRTK, ... Ki_tram, n_tram, Tram, kIRS_PI3K, kRasPI3K,
    kPI3K_AKT, ... kPTEN, Ki_PTEN, kDephos_AKT)

    dy = zeros(50, 1);

vemu_inh = Ki_vemu ^ n_vemu / (Ki_vemu ^ n_vemu + Vemu ^ n_vemu);
vemu_on = 1 - vemu_inh;
tram_inh = Ki_tram ^ n_tram / (Ki_tram ^ n_tram + Tram ^ n_tram);
RAS_GTP = y(10) + y(12) + y(14);
NF1_KRAS = knf1_KRAS * y(15) * y(14) / (Km_nf1_KRAS + y(14));
NF1_HRAS = knf1_HRAS * y(15) * y(10) / (Km_nf1_HRAS + y(10));
NF1_NRAS = knf1_NRAS * y(15) * y(12) / (Km_nf1_NRAS + y(12));
AKT_inh_foxo = y(40) ^ n_foxo / (Ki_foxo ^ n_foxo + y(40) ^ n_foxo);

% -- -EGFR-- - dy(1) = -ka1 * y(1) + kr1 * y(2);
dy(2) = ka1 * y(1) - kr1 * y(2) - kc1 * y(2);
dy(3) = kc1 * y(2) - kDegradEgfr * y(3) - kErkInbEgfr * y(23) * y(3);

% -- -SHC-- - dy(4) = -ka1 * y(3) * y(4) - kHER2EGFR_Shc * y(45) * y(4)... -
                      kHER23_Shc * y(48) * y(4) - kIGF1R_Shc * y(34) * y(4);
dy(5) = ka1 * y(3) * y(4) + kHER2EGFR_Shc * y(45) * y(4)... +
        kHER23_Shc * y(48) * y(4) + kIGF1R_Shc * y(34) * y(4)... -
        kShcDephos * y(6) * y(5);
dy(6) = -kptpDeg * y(5) * y(6);

% -- -GRB2 / SOS-- -
    dy(7) = kGrb2CombShc * y(5) * y(3) + kHER2EGFR_Grb2 * y(5) * y(45)... -
            kSprtyInbGrb2 * y(21) * y(7);
dy(8) = kSosCombGrb2 * y(7) * y(5) - kErkPhosSos * y(23) * y(8);

% -- -HRAS-- - dy(9) = -kRasAct * y(8) * y(9) + kRasGAP * y(10) + NF1_HRAS;
dy(10) = kRasAct * y(8) * y(9) - kRasGAP * y(10) - NF1_HRAS;

% -- -NRAS-- - dy(11) = -kRasAct * y(8) * y(11) + kRasGAP * y(12) + NF1_NRAS;
dy(12) = kRasAct * y(8) * y(11) - kRasGAP * y(12) - NF1_NRAS;

% -- -KRAS-- - dy(13) = -kRasAct * y(8) * y(13) + kRasGAP * y(14) + NF1_KRAS;
dy(14) = kRasAct * y(8) * y(13) - kRasGAP * y(14) - NF1_KRAS;
dy(15) = 0;
% NF1 constant

        % -- -CRAF-- -
    dy(16) = -kpCraf * RAS_GTP * y(16) + kErkPhosPcraf * y(23) * y(17)... +
             kPcrafDegrad * y(17) * y(30) -
             kBCform * vemu_on * y(18) * y(16)... + kBCdiss * (y(49) + y(50));
dy(17) = kpCraf * RAS_GTP * y(16) - kErkPhosPcraf * y(23) * y(17) -
         kPcrafDegrad * y(17) * y(30);

% -- -BRAF-- - dy(18) = -ka1 * y(18) - kBCform * vemu_on * y(18) * y(16) +
                        kBCdiss * (y(49) + y(50));
dy(19) = ka1 * y(18) * vemu_inh - kinbBraf * y(19);

% -- -BRAF : CRAF dimer-- - dy(49) = kBCform * vemu_on * y(18) * y(16) -
                                     kBCact * RAS_GTP * y(49) - kBCdiss * y(49);
dy(50) =
    kBCact * RAS_GTP * y(49) - kErkPhosPcraf * y(23) * y(50) - kBCdiss * y(50);

% -- -MEK-- - dy(20) = -kpMek * (y(17) + y(19) + y(50)) * y(20) * tram_inh... +
                       kErkPhosMek * y(23) * y(21) + kMekDegrad * y(21) * y(29);
dy(21) = kpMek * (y(17) + y(19) + y(50)) * y(20) * tram_inh... -
         kErkPhosMek * y(23) * y(21) - kMekDegrad * y(21) * y(29);

% -- -ERK-- - dy(22) = -kpErk * y(21) * y(22) +
                       kDuspInbErk * y(24) * y(23) + kErkDeg * y(23) * y(28);
dy(23) = kpErk * y(21) * y(22) -
         kDuspInbErk * y(24) * y(23) - kErkDeg * y(23) * y(28);

% -- -DUSP6 / Sprouty-- -
    dy(24) = km_Dusp * y(23) / (1 + (km_Dusp / kDusps) * y(23)) - kDuspStop *
                                                                      y(24) *
                                                                      y(31);
dy(25) = -kDuspStop * y(24) * y(25);
dy(26) = km_Sprty * y(23) / (1 + (km_Sprty / kSproutyForm) * y(23)) -
         kSprtyComeDown * y(26) * y(27);
dy(27) = -kSprtyComeDown * y(26) * y(27);

% -- -Degradation sentinels-- - dy(28) = -kErkDeg * y(23) * y(28);
dy(29) = -kMekDegrad * y(21) * y(29);
dy(30) = -kPcrafDegrad * y(17) * y(30);
dy(31) = -kDuspStop * y(24) * y(31);

% -- -IGF1R-- - dy(32) = kSynIGF1R * (y(41) / (Km_IGF1R + y(41))) -
                         kDegRTK * y(32) - ka1 * y(32) + kr1 * y(33);
dy(33) = ka1 * y(32) - kr1 * y(33) - kc1 * y(33);
dy(34) = kc1 * y(33) - kErkInbEgfr * y(23) * y(34);

% -- -IRS / PI3K-- - dy(35) = -ka1 * y(3) * y(35) - kEGFRPI3k * y(34) * y(35);
dy(36) = ka1 * y(3) * y(35) + kEGFRPI3k * y(34) * y(35) -
         kMTOR_Feedback * y(43) * y(36);

% -- -PIP3 with PTEN Level 2(PTEN inhibited by high PIP3)-- - PTEN_active =
    kPTEN / (1 + y(38) / Ki_PTEN);

dy(37) = -(kIRS_PI3K * y(36) * y(37) + kRasPI3K * y(14) * y(37) +
           kHER23_PI3K * y(48) * y(37))... +
         kdegrad * y(38);
dy(38) = kIRS_PI3K * y(36) * y(37) + kRasPI3K * y(14) * y(37) +
         kHER23_PI3K * y(48) * y(37)... - kdegrad * y(38)... -
         PTEN_active * y(38);

% -- -AKT(mass - conserving cycling via dedicated phosphatase kDephos_AKT)-- -
    % d / dt(y39 + y40) = 0  — total AKT conserved
    dy(39) = -kPI3K_AKT * y(38) * y(39) + kDephos_AKT * y(40);
dy(40) = kPI3K_AKT * y(38) * y(39) - kDephos_AKT * y(40);

% -- -FOXO-- - dy(41) =
    kSynFoxo * (1 - AKT_inh_foxo) / (1 + (y(41) / Km_foxo)) - kDegFoxo * y(41);

% -- -mTORC-- - dy(42) = -ka1 * y(40) * y(42) + kdegrad * y(43);
dy(43) = ka1 * y(40) * y(42) - kdegrad * y(43);

% -- -HER2-- - dy(44) =
    kSynHER2 * (y(41) / (Km_HER2 + y(41))) - kDegRTK * y(44)... -
    kHER2_EGFR_form * y(3) * y(44) - kHER2_HER3_form * y(47) * y(44)... +
    kHER2_EGFR_diss * y(45) + kHER2_HER3_diss * y(48);
dy(45) = kHER2_EGFR_form * y(3) * y(44) - kHER2_EGFR_diss * y(45);

% -- -HER3(with ERK - mediated negative feedback)-- -
    dy(46) = kSynHER3 * (y(41) / (Km_HER3 + y(41))) - kDegRTK * y(46)... -
             kNRG_bind * NRG * y(46) - kErkInbEgfr * y(23) * y(46);
dy(47) = kNRG_bind * NRG * y(46) - kHER2_HER3_form * y(44) * y(47) +
         kHER2_HER3_diss * y(48);
dy(48) = kHER2_HER3_form * y(44) * y(47) - kHER2_HER3_diss * y(48);

end % Mapk

        % %
    == == == == == == == == == == == == == == == == == == == == == == == == ==
    == == == == == == == == == % % HELPERS % % == == == == == == == == == == ==
    == == == == == == == == == == == == == == == == == == == == == == ==
    function fmt(xlab, ylab, ttl) xlabel(xlab, 'FontSize', 11);
ylabel(ylab, 'FontSize', 11);
title(ttl, 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 9);
xlim([-1 50]);
xticks([0 1 4 8 24 48]);
grid on;
box on;
end

    function v = rescale_safe(x) r = max(x) - min(x);
if r
  < eps v = zeros(size(x));
else
  v = (x - min(x)) / r;
end end