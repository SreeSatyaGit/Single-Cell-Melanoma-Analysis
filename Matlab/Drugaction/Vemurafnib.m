% == == == == == == == == == == == == == == == == == == == == == == == == == ==
    == == == ==
    % MAPK /
        PI3K ODE System(50 states, 68 parameters) % Experimental data : pEGFR,
    pCRAF, pMEK, pERK, DUSP6, pAKT, % IGF1R, HER2,
    HER3(9 species fitted) % == == == == == == == == == == == == == == == == ==
        == == == == == == == == == == == == ==

        % % == == == == == == == == == == == == == == == == == == == == == == ==
        == == == == == == == == == == ==
        % % SECTION 1 — KINETIC PARAMETERS % % == == == == == == == == == == ==
        == == == == == == == == == == == == == == == == == == == == == == ==

        % -- -Core-- - ka1 = 1e-4;
kr1 = 1e-3;
kc1 = 1e-2;

% -- -MAPK cascade-- - kpCraf = 1e-4;
kpMek = 2e-6;
kpErk = 4e-5;
kDegradEgfr = 5.2537e-05;
kErkInbEgfr = 0.0017;
kShcDephos = 0.0043;
kptpDeg = 0.3063;
kGrb2CombShc = 0.0015;
kSprtyInbGrb2 = 0.0014;
kSosCombGrb2 = 5e-8;
kErkPhosSos = 3e-4;
kErkPhosPcraf = 8.57e-5;
kPcrafDegrad = 0.0167;
kErkPhosMek = 1.07e-5;
kMekDegrad = 0.6850;
kDuspInbErk = 2.73e-5;
kErkDeg = 0.0024;
kinbBraf = 5e-2;
kDuspStop = 1.3792e-04;
kDusps = 0.0266;
kSproutyForm = 1.7e-6;
kSprtyComeDown = 5.5e-5;
kdegrad = 2.8153e-04;
km_Sprty_decay = 8e-5;
km_Dusp = 1.2700e-04;
km_Sprty = 1.0000e-06;
kEGFRPI3k = 0.0090;
kMTOR_Feedback = 0.0175;

% -- -RAS / SOS-- - kRasAct = 0.0193;
kRasGAP = 1e-4;

% -- -Vemurafenib-- - Ki_vemu = 1.0000e-06;
n_vemu = 1.5;
Vemu = 1e-6;

% -- -HER2 / HER3-- - kHER2_EGFR_form = 0.0651;
kHER2_EGFR_diss = 0.0018;
kNRG_bind = 7.2533e-05;
NRG = 0.0285;
kHER2_HER3_form = 4.1082e-04;
kHER2_HER3_diss = 0.0377;
kHER23_PI3K = 0.0041;
kHER2EGFR_Grb2 = 0.0122;

% -- -BRAF : CRAF heterodimer-- - kBCform = 0.0020;
kBCdiss = 5.0000e-05;
kBCact = 0.0020;

% -- -SHC multi - receptor inputs-- - kHER2EGFR_Shc = 0.0354;
kHER23_Shc = 0.0027;
kIGF1R_Shc = 6.2581e-04;

% -- -NF1 isoform - specific GAP-- - knf1_KRAS = 1e-6;
Km_nf1_KRAS = 0.4999;
knf1_HRAS = 0.0015;
Km_nf1_HRAS = 0.4999;
knf1_NRAS = 0.0017;
Km_nf1_NRAS = 0.5001;

% -- -FOXO - driven receptor synthesis-- - kSynHER2 = 0.0077;
Km_HER2 = 0.0163;
kSynHER3 = 2.6964e-04;
Km_HER3 = 0.0109;
kSynIGF1R = 4.9880e-04;
Km_IGF1R = 0.0182;

% -- -FOXO dynamics-- - kSynFoxo = 8e-3;
kDegFoxo = 5e-4;
Ki_foxo = 0.5995;
n_foxo = 2.0;
Km_foxo = 0.0493;

% -- -Basal RTK degradation-- - kDegRTK = 3.6896e-05;

% % == == == == == == == == == == == == == == == == == == == == == == == == ==
    == == == == == == == == == % % SECTION 2 — EXPERIMENTAL DATA % % == == == ==
    == == == == == == == == == == == == == == == == == == == == == == == == ==
    == == == == == pEGFR_Exp = [
  0.291928893, 0.392400458, 0.265016688, 0.394238749, 0.006158316, 0.008115099
];
pCRAF_Exp = [
  0.366397596, 0.537106733, 0.465541704, 0.586732657, 1.102322681, 0.269181259
];
pMEK_Exp = [
  1.75938884, 0.170160085, 0.095112609, 0.201000276, 0.219207054, 0.502831668
];
pERK_Exp = [
  2.903209735, 0.207867788, 0.303586121, 0.805254439, 1.408362153, 1.847606441
];
DUSP_Exp = [
  2.677161325, 2.782754577, 1.130758062, 0.395642757, 0.828575853, 0.916618219
];
expAKT = [ 1.080071, 0.900024, 1.066681, 0.899337, 1.21576, 1.333254 ];
IGF1R_Exp = [
  0.96024414, 1.070624757, 1.126551098, 1.157072001, 0.734617533, 0.457922622
];
Her2_Exp = [
  0.245236744, 0.177917339, 0.239075259, 0.306884773, 1.066654783, 1.005085151
];
Her3_Exp = [
  0.203233765, 0.194358998, 0.303475212, 0.674083831, 0.89702403, 0.459831389
];

timeStamps = [ 0, 1, 4, 8, 24, 48 ];

% Normalise all experimental data mekExpVals = rescale(pMEK_Exp);
PexpErkVals = rescale(pERK_Exp);
expDusp = rescale(DUSP_Exp);
expPEGFR = rescale(pEGFR_Exp);
expCRAF = rescale(pCRAF_Exp);
expAKT = rescale(expAKT);
expIGF1R = rescale(IGF1R_Exp);
% NEW expHER2 = rescale(Her2_Exp);
% NEW expHER3 = rescale(Her3_Exp);
% NEW

        % %
    == == == == == == == == == == == == == == == == == == == == == == == == ==
    == == == == == == == == == % % SECTION 3 — INITIAL CONDITIONS(50 states) % %
    == == == == == == == == == == == == == == == == == == == == == == == == ==
    == == == == == == == == == RAS_GTP_frac = 0.1;

EGFR = [ 1 - pEGFR_Exp(1), 0, pEGFR_Exp(1) ];
SHC = [ 1, 0, 1 ];
grb_sos = [ 0, 0 ];
HRAS = [ 1 - RAS_GTP_frac, RAS_GTP_frac ];
NRAS = [ 1 - RAS_GTP_frac, RAS_GTP_frac ];
KRAS = [ 1 - RAS_GTP_frac, RAS_GTP_frac, 1 ];
CRAF = [ 1 - pCRAF_Exp(1), pCRAF_Exp(1) ];
BRAF = [ 0, 1 ];
MEK = [ 1, 1 ];
ERK = [ 1 - PexpErkVals(1), PexpErkVals(1) ];
DUSP = [ 1 - expDusp(1), expDusp(1) ];
SRTY = [ 1, 1 ];
pERKDegrad = [1];
pMEKDegrad = [1];
pCRAFDegrad = [1];
DUSPStop = [1];
IGFR = [ 1, 0, 0 ];
IRS = [ 1, 0 ];
pi3k = [ 1, 0 ];
AKT = [ 1 - expAKT(1), expAKT(1) ];
FOXO = [0];
MTORC = [ 1, 0 ];
HER2 = [ 1, 0 ];
HER3 = [ 1, 0, 0 ];
BCRAF_dimer = [ 0, 0 ];

y0 = [EGFR, SHC, grb_sos, HRAS, NRAS, KRAS, CRAF, BRAF, MEK, ERK, ...
      DUSP, SRTY, pERKDegrad, pMEKDegrad, pCRAFDegrad, DUSPStop, ...
      IGFR, IRS, pi3k, AKT, FOXO, MTORC, ...
      HER2, HER3, BCRAF_dimer];
% 50 states

        % %
    == == == == == == == == == == == == == == == == == == == == == == == == ==
    == == == == == == == == == % % SECTION 4 — PARAMETER VECTOR(68 elements) % %
    == == == == == == == == == == == == == == == == == == == == == == == == ==
    == == == == == == == == == params0 = [
  ka1,
  kr1,
  kc1,
  ... kpCraf,
  kpMek,
  kpErk,
  ... kDegradEgfr,
  kErkInbEgfr,
  kShcDephos,
  kptpDeg,
  ... kGrb2CombShc,
  kSprtyInbGrb2,
  kSosCombGrb2,
  kErkPhosSos,
  ... kErkPhosPcraf,
  kPcrafDegrad,
  kErkPhosMek,
  kMekDegrad,
  ... kDuspInbErk,
  kErkDeg,
  kinbBraf,
  kDuspStop,
  ... kDusps,
  kSproutyForm,
  kSprtyComeDown,
  kdegrad,
  ... km_Sprty_decay,
  km_Dusp,
  km_Sprty,
  kEGFRPI3k,
  kMTOR_Feedback,
  ... Ki_vemu,
  n_vemu,
  Vemu,
  ... kHER2_EGFR_form,
  kHER2_EGFR_diss,
  kNRG_bind,
  NRG,
  ... kHER2_HER3_form,
  kHER2_HER3_diss,
  kHER23_PI3K,
  kHER2EGFR_Grb2,
  ... kRasAct,
  kRasGAP,
  ... kBCform,
  kBCdiss,
  kBCact,
  ... kHER2EGFR_Shc,
  kHER23_Shc,
  kIGF1R_Shc,
  ... knf1_KRAS,
  Km_nf1_KRAS,
  knf1_HRAS,
  Km_nf1_HRAS,
  knf1_NRAS,
  Km_nf1_NRAS,
  ... kSynHER2,
  Km_HER2,
  kSynHER3,
  Km_HER3,
  kSynIGF1R,
  Km_IGF1R,
  ... kSynFoxo,
  kDegFoxo,
  Ki_foxo,
  n_foxo,
  Km_foxo,
  ... kDegRTK
];

nParams = length(params0);   % = 68


%% ====================================================================
%%  SECTION 5 — BOUNDS  (same for all parameters)
%% ====================================================================
lb = 1e-12 * ones(1, nParams);
ub = 1e2 * ones(1, nParams);

params0 = max(params0, lb + 1e-8 * (ub - lb));
params0 = min(params0, ub - 1e-8 * (ub - lb));

% % == == == == == == == == == == == == == == == == == == == == == == == == ==
    == == == == == == == == ==
    % % SECTION 6 — OPTIMISATION % % Objective now fits 9 species : % % pEGFR,
    pCRAF, pMEK, pERK, DUSP6, pAKT, IGF1R, HER2,
    HER3 % % == == == == == == == == == == == == == == == == == == == == == ==
        == == == == == == == == == == == == options =
        optimoptions(@fmincon, ... 'Algorithm', 'sqp', ... 'Display', 'iter',
                     ... 'MaxIterations', 300, ... 'MaxFunctionEvaluations',
                     50000, ... 'OptimalityTolerance', 1e-6,
                     ... 'StepTolerance', 1e-8, ... 'FiniteDifferenceType',
                     'central', ... 'ScaleProblem', 'obj-and-constr');

[ optimizedParams, errorOpt ] =
    fmincon(... @(p) objectiveFunction(
                p, timeStamps, ... mekExpVals, PexpErkVals, expDusp, expPEGFR,
                expCRAF, expAKT, ... expIGF1R, expHER2, expHER3, y0),
            ... params0, [], [], [], [], lb, ub, [], options);

paramNames = {... 'ka1',
              'kr1',
              'kc1',
              ... 'kpCraf',
              'kpMek',
              'kpErk',
              ... 'kDegradEgfr',
              'kErkInbEgfr',
              'kShcDephos',
              'kptpDeg',
              ... 'kGrb2CombShc',
              'kSprtyInbGrb2',
              'kSosCombGrb2',
              'kErkPhosSos',
              ... 'kErkPhosPcraf',
              'kPcrafDegrad',
              'kErkPhosMek',
              'kMekDegrad',
              ... 'kDuspInbErk',
              'kErkDeg',
              'kinbBraf',
              'kDuspStop',
              ... 'kDusps',
              'kSproutyForm',
              'kSprtyComeDown',
              'kdegrad',
              ... 'km_Sprty_decay',
              'km_Dusp',
              'km_Sprty',
              'kEGFRPI3k',
              'kMTOR_Feedback',
              ... 'Ki_vemu',
              'n_vemu',
              'Vemu',
              ... 'kHER2_EGFR_form',
              'kHER2_EGFR_diss',
              'kNRG_bind',
              'NRG',
              ... 'kHER2_HER3_form',
              'kHER2_HER3_diss',
              'kHER23_PI3K',
              'kHER2EGFR_Grb2',
              ... 'kRasAct',
              'kRasGAP',
              ... 'kBCform',
              'kBCdiss',
              'kBCact',
              ... 'kHER2EGFR_Shc',
              'kHER23_Shc',
              'kIGF1R_Shc',
              ... 'knf1_KRAS',
              'Km_nf1_KRAS',
              'knf1_HRAS',
              'Km_nf1_HRAS',
              'knf1_NRAS',
              'Km_nf1_NRAS',
              ... 'kSynHER2',
              'Km_HER2',
              'kSynHER3',
              'Km_HER3',
              'kSynIGF1R',
              'Km_IGF1R',
              ... 'kSynFoxo',
              'kDegFoxo',
              'Ki_foxo',
              'n_foxo',
              'Km_foxo',
              ... 'kDegRTK'};

fprintf('\nOptimized parameters:\n');
for
  i = 1 : nParams fprintf('  p(%2d) %-20s = %.6g\n', i, paramNames{i},
                          optimizedParams(i));
end fprintf('Minimum error: %g\n', errorOpt);

% % == == == == == == == == == == == == == == == == == == == == == == == == ==
    == == == == == == == == == % % ODE FUNCTION(50 states, 68 parameters) % % ==
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
    n_foxo, Km_foxo, kDegRTK)

    dy = zeros(50, 1);

vemu_inh = Ki_vemu ^ n_vemu / (Ki_vemu ^ n_vemu + Vemu ^ n_vemu);
vemu_on = 1 - vemu_inh;
RAS_GTP = y(10) + y(12) + y(14);
NF1_KRAS = knf1_KRAS * y(15) * y(14) / (Km_nf1_KRAS + y(14));
NF1_HRAS = knf1_HRAS * y(15) * y(10) / (Km_nf1_HRAS + y(10));
NF1_NRAS = knf1_NRAS * y(15) * y(12) / (Km_nf1_NRAS + y(12));
AKT_inh_foxo = y(40) ^ n_foxo / (Ki_foxo ^ n_foxo + y(40) ^ n_foxo);

dy(1) = -ka1 * y(1) + kr1 * y(2);
dy(2) = ka1 * y(1) - kr1 * y(2) - kc1 * y(2);
dy(3) = kc1 * y(2) - kDegradEgfr * y(3) - kErkInbEgfr * y(23) * y(3);

dy(4) = -ka1 * y(3) * y(4) - kHER2EGFR_Shc * y(45) * y(4)... -
        kHER23_Shc * y(48) * y(4) - kIGF1R_Shc * y(34) * y(4);
dy(5) = ka1 * y(3) * y(4) + kHER2EGFR_Shc * y(45) * y(4)... +
        kHER23_Shc * y(48) * y(4) + kIGF1R_Shc * y(34) * y(4)... -
        kShcDephos * y(6) * y(5);
dy(6) = -kptpDeg * y(5) * y(6);

dy(7) = kGrb2CombShc * y(5) * y(3) + kHER2EGFR_Grb2 * y(5) * y(45)... -
        kSprtyInbGrb2 * y(21) * y(7);
dy(8) = kSosCombGrb2 * y(7) * y(5) - kErkPhosSos * y(23) * y(8);

dy(9) = -kRasAct * y(8) * y(9) + kRasGAP * y(10) + NF1_HRAS;
dy(10) = kRasAct * y(8) * y(9) - kRasGAP * y(10) - NF1_HRAS;
dy(11) = -kRasAct * y(8) * y(11) + kRasGAP * y(12) + NF1_NRAS;
dy(12) = kRasAct * y(8) * y(11) - kRasGAP * y(12) - NF1_NRAS;
dy(13) = -kRasAct * y(8) * y(13) + kRasGAP * y(14) + NF1_KRAS;
dy(14) = kRasAct * y(8) * y(13) - kRasGAP * y(14) - NF1_KRAS;
dy(15) = 0;

dy(16) = -kpCraf * RAS_GTP * y(16) + kErkPhosPcraf * y(23) * y(17)... +
         kPcrafDegrad * y(17) * y(30) - kBCform * vemu_on * y(18) * y(16)... +
         kBCdiss * (y(49) + y(50));
dy(17) = kpCraf * RAS_GTP * y(16) - kErkPhosPcraf * y(23) * y(17) -
         kPcrafDegrad * y(17) * y(30);
dy(18) = -ka1 * y(18) - kBCform * vemu_on * y(18) * y(16) +
         kBCdiss * (y(49) + y(50));
dy(19) = ka1 * y(18) * vemu_inh - kinbBraf * y(19);

dy(49) = kBCform * vemu_on * y(18) * y(16) - kBCact * RAS_GTP * y(49) -
         kBCdiss * y(49);
dy(50) =
    kBCact * RAS_GTP * y(49) - kErkPhosPcraf * y(23) * y(50) - kBCdiss * y(50);

dy(20) = -kpMek * (y(17) + y(19) + y(50)) * y(20)... +
         kErkPhosMek * y(23) * y(21)... + kMekDegrad * y(21) * y(29);

dy(21) = kpMek * (y(17) + y(19) + y(50)) * y(20)... -
         kErkPhosMek * y(23) * y(21)... - kMekDegrad * y(21) * y(29);
dy(22) = -kpErk * y(21) * y(22) +
         kDuspInbErk * y(24) * y(23) + kErkDeg * y(23) * y(28);
dy(23) = kpErk * y(21) * y(22) -
         kDuspInbErk * y(24) * y(23) - kErkDeg * y(23) * y(28);

dy(24) = km_Dusp * y(23) / (1 + (km_Dusp / kDusps) * y(23)) - kDuspStop *
                                                                  y(24) * y(31);
dy(25) = -kDuspStop * y(24) * y(25);
dy(26) = km_Sprty * y(23) / (1 + (km_Sprty / kSproutyForm) * y(23)) -
         kSprtyComeDown * y(26) * y(27);
dy(27) = -kSprtyComeDown * y(26) * y(27);

dy(28) = -kErkDeg * y(23) * y(28);
dy(29) = -kMekDegrad * y(21) * y(29);
dy(30) = -kPcrafDegrad * y(17) * y(30);
dy(31) = -kDuspStop * y(24) * y(31);

dy(32) = kSynIGF1R * (y(41) / (Km_IGF1R + y(41))) - kDegRTK * y(32) -
         ka1 * y(32) + kr1 * y(33);
dy(33) = ka1 * y(32) - kr1 * y(33) - kc1 * y(33);
dy(34) = kc1 * y(33) - kErkInbEgfr * y(23) * y(34);
dy(35) = -ka1 * y(3) * y(35) - kEGFRPI3k * y(34) * y(35);
dy(36) = ka1 * y(3) * y(35) + kEGFRPI3k * y(34) * y(35) -
         kMTOR_Feedback * y(43) * y(36);
dy(37) = -(ka1 * y(36) * y(37) + ka1 * y(14) * y(37) +
           kHER23_PI3K * y(48) * y(37)) +
         kdegrad * y(38);
dy(38) = ka1 * y(36) * y(37) + ka1 * y(14) * y(37) +
         kHER23_PI3K * y(48) * y(37) - kdegrad * y(38);
dy(39) = -ka1 * y(38) * y(39) + kdegrad * y(40);
dy(40) = ka1 * y(38) * y(39) - kdegrad * y(40);
dy(41) = kSynFoxo * (1 - AKT_inh_foxo) / (1 + (y(41) / Km_foxo)) - kDegFoxo *
                                                                       y(41);
dy(42) = -ka1 * y(40) * y(42) + kdegrad * y(43);
dy(43) = ka1 * y(40) * y(42) - kdegrad * y(43);

dy(44) = kSynHER2 * (y(41) / (Km_HER2 + y(41))) - kDegRTK * y(44)... -
         kHER2_EGFR_form * y(3) * y(44) - kHER2_HER3_form * y(47) * y(44)... +
         kHER2_EGFR_diss * y(45) + kHER2_HER3_diss * y(48);
dy(45) = kHER2_EGFR_form * y(3) * y(44) - kHER2_EGFR_diss * y(45);
dy(46) = kSynHER3 * (y(41) / (Km_HER3 + y(41))) - kDegRTK * y(46) -
         kNRG_bind * NRG * y(46);
dy(47) = kNRG_bind * NRG * y(46) - kHER2_HER3_form * y(44) * y(47) +
         kHER2_HER3_diss * y(48);
dy(48) = kHER2_HER3_form * y(44) * y(47) - kHER2_HER3_diss * y(48);

end % Mapk

        % %
    == == == == == == == == == == == == == == == == == == == == == == == == ==
    == == == == == == == == ==
    % % OBJECTIVE FUNCTION % % Fits 9 species : pEGFR,
    pCRAF, pMEK, pERK, DUSP6, pAKT, % % IGF1R(y32), HER2 monomer(y44),
    HER3 inactive(y46) % % % % Model state mapping : % % y(3) =
        pEGFR % %
        y(17) = pCRAF % %
                y(21) = pMEK % %
                        y(23) = pERK % %
                                y(24) = DUSP6 % % y(40) =
                                            pAKT % % y(32) =
                                                IGF1R inactive <
                                                --NEW %
                                                    % y(44) =
                                                    HER2 monomer <
                                                    --NEW %
                                                        % y(46) =
                                                        HER3 inactive < --NEW
                                                                            % %
                                                        == == == == == == == ==
                                                        == == == == == == == ==
                                                        == == == == == == == ==
                                                        == == == == == == == ==
                                                        == == function err =
                                                            objectiveFunction(
                                                                p, timeStamps,
                                                                ... mekExpVals,
                                                                PexpErkVals,
                                                                expDusp,
                                                                expPEGFR,
                                                                expCRAF, expAKT,
                                                                ... expIGF1R,
                                                                expHER2,
                                                                expHER3, y0)

                                                            % Unpack
                                                            68 parameters ka1 =
                                                                p(1);
kr1 = p(2);
kc1 = p(3);
kpCraf = p(4);
kpMek = p(5);
kpErk = p(6);
kDegradEgfr = p(7);
kErkInbEgfr = p(8);
kShcDephos = p(9);
kptpDeg = p(10);
kGrb2CombShc = p(11);
kSprtyInbGrb2 = p(12);
kSosCombGrb2 = p(13);
kErkPhosSos = p(14);
kErkPhosPcraf = p(15);
kPcrafDegrad = p(16);
kErkPhosMek = p(17);
kMekDegrad = p(18);
kDuspInbErk = p(19);
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
kMTOR_Feedback = p(31);
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
kRasGAP = p(44);
kBCform = p(45);
kBCdiss = p(46);
kBCact = p(47);
kHER2EGFR_Shc = p(48);
kHER23_Shc = p(49);
kIGF1R_Shc = p(50);
knf1_KRAS = p(51);
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
kSynFoxo = p(63);
kDegFoxo = p(64);
Ki_foxo = p(65);
n_foxo = p(66);
Km_foxo = p(67);
kDegRTK = p(68);

odeOpts = odeset('RelTol', 1e-6, 'AbsTol', 1e-9, 'MaxStep', 500);
try[T, YO] = ode15s(
    @(t, y)
        Mapk(t, y, ... ka1, kr1, kc1, kpCraf, kpMek, kpErk, ... kDegradEgfr,
             kErkInbEgfr, kShcDephos, kptpDeg, ... kGrb2CombShc, kSprtyInbGrb2,
             kSosCombGrb2, kErkPhosSos, ... kErkPhosPcraf, kPcrafDegrad,
             kErkPhosMek, kMekDegrad, ... kDuspInbErk, kErkDeg, kinbBraf,
             kDuspStop, kDusps, ... kSproutyForm, kSprtyComeDown, kdegrad,
             km_Sprty_decay, ... km_Dusp, km_Sprty, kEGFRPI3k, kMTOR_Feedback,
             ... Ki_vemu, n_vemu, Vemu, ... kHER2_EGFR_form, kHER2_EGFR_diss,
             kNRG_bind, NRG, ... kHER2_HER3_form, kHER2_HER3_diss, kHER23_PI3K,
             kHER2EGFR_Grb2, ... kRasAct, kRasGAP, ... kBCform, kBCdiss, kBCact,
             ... kHER2EGFR_Shc, kHER23_Shc, kIGF1R_Shc, ... knf1_KRAS,
             Km_nf1_KRAS, knf1_HRAS, Km_nf1_HRAS, knf1_NRAS, Km_nf1_NRAS,
             ... kSynHER2, Km_HER2, kSynHER3, Km_HER3, kSynIGF1R, Km_IGF1R,
             ... kSynFoxo, kDegFoxo, Ki_foxo, n_foxo, Km_foxo, kDegRTK),
    ... timeStamps * 3600, y0, odeOpts);
catch err = 1e6;
return;
end

    ts = timeStamps * 3600;
safe = @(col) max(interp1(T, YO( :, col), ts, 'linear', 'extrap'), 0);

% Original 6 residuals errorEGFR = sum((rescale_safe(safe(3)) - expPEGFR).^ 2);
errorCRAF = sum((rescale_safe(safe(17)) - expCRAF).^ 2);
errorMek = sum((rescale_safe(safe(21)) - mekExpVals).^ 2);
errorPErk = sum((rescale_safe(safe(23)) - PexpErkVals).^ 2);
errorDusp = sum((rescale_safe(safe(24)) - expDusp).^ 2);
errorAKT = sum((rescale_safe(safe(40)) - expAKT).^ 2);

% New 3 residuals — IGF1R inactive(y32), HER2 monomer(y44),
    HER3
    inactive(y46) errorIGF1R = sum((rescale_safe(safe(32)) - expIGF1R).^ 2);
errorHER2 = sum((rescale_safe(safe(44)) - expHER2).^ 2);
errorHER3 = sum((rescale_safe(safe(46)) - expHER3).^ 2);

err = errorPErk + errorAKT + errorEGFR + errorCRAF + ... errorMek + errorDusp +
      errorIGF1R + errorHER2 + errorHER3;
end

        % %
    == == == == == == == == == == == == == == == == == == == == == == == == ==
    == == == == == == == == == % % HELPER % % == == == == == == == == == == ==
    == == == == == == == == == == == == == == == == == == == == == == ==
    function v = rescale_safe(x) r = max(x) - min(x);
if r
  < eps v = zeros(size(x));
else
  v = (x - min(x)) / r;
end end