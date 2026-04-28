function dydt = Mapk_ODE_DR(t, y, p)
    % MAPK/PI3K Pathway ODE System with Vemurafenib and Paradoxical Activation
    % Adapted for Deep Research Implementation
    
    % Unpack parameters
    ka1 = p(1);  kr1 = p(2);  kc1 = p(3);
    kpCraf = p(4); kpMek = p(5); kpErk = p(6);
    kDegradEgfr = p(7); kErkInbEgfr = p(8); kShcDephos = p(9); kptpDeg = p(10); kGrb2CombShc = p(11);
    kSprtyInbGrb2 = p(12); kSosCombGrb2 = p(13); kErkPhosSos = p(14);
    kErkPhosPcraf = p(15); kPcrafDegrad = p(16); kErkPhosMek = p(17); kMekDegrad = p(18);
    kDuspInbErk = p(19); kErkDeg = p(20); kinbBraf = p(21); kDuspStop = p(22); kDusps = p(23);
    kSproutyForm = p(24); kSprtyComeDown = p(25); kdegrad = p(26); km_Sprty_decay = p(27); km_Dusp = p(28); km_Sprty = p(29);
    kErkDephos = p(30); kDuspDeg = p(31);
    kHer2_act = p(32); kHer3_act = p(33);
    k_p85_bind_EGFR = p(34); k_p85_bind_Her2 = p(35); k_p85_bind_Her3 = p(36); k_p85_bind_IGFR = p(37);
    k_p85_unbind = p(38); k_PI3K_recruit = p(39);
    kMTOR_Feedback = p(40);
    k_PIP2_to_PIP3 = p(41); k_PTEN = p(42);
    kAkt = p(43); kdegradAKT = p(44);
    kb1 = p(45); k43b1 = p(46); k4ebp1 = p(47); k_4EBP1_dephos = p(48);
    kKSRphos = p(49); kKSRdephos = p(50);
    kMekByBraf = p(51); kMekByCraf = p(52); kMekByKSR = p(53);
    Tram = p(54); K_tram_RAF = p(55); K_tram_KSR = p(56); n_tram = p(57);
    Vemurafenib = p(58); kDimerForm = p(59); kDimerDissoc = p(60);
    kParadoxCRAF = p(61); IC50_vem = p(62); Hill_n_vem = p(63);
    
    % Initialize derivatives
    dydt = zeros(62, 1);
    
    % ========================================================================
    % MODULE 1: RTK SIGNALING (EXPANDED: EGFR, Her2, Her3)
    % ========================================================================
    % EGFR
    dydt(1) = -ka1*y(1) + kr1*y(2);                                    % EGFR
    dydt(2) = ka1*y(1) - kr1*y(2) - kc1*y(2);                          % EGFR:ligand
    dydt(3) = kc1*y(2) - kDegradEgfr*y(3) - kErkInbEgfr*y(29)*y(3);    % pEGFR (with ERK feedback)
    
    % Her2 (NEW)
    dydt(4) = -kHer2_act*y(4) + kr1*y(5);                              % Her2
    dydt(5) = kHer2_act*y(4) - kr1*y(5) - kc1*y(5);                    % Her2:ligand
    dydt(6) = kc1*y(5) - kDegradEgfr*y(6) - kErkInbEgfr*y(29)*y(6);    % pHer2 (with ERK feedback)
    
    % Her3 (NEW)
    dydt(7) = -kHer3_act*y(7) + kr1*y(8);                              % Her3
    dydt(8) = kHer3_act*y(7) - kr1*y(8) - kc1*y(8);                    % Her3:ligand
    dydt(9) = kc1*y(8) - kDegradEgfr*y(9) - kErkInbEgfr*y(29)*y(9);    % pHer3 (with ERK feedback)
    
    % ========================================================================
    % MODULE 2: Shc/Grb2/SOS SIGNALING
    % ========================================================================
    dydt(10) = -ka1*y(3)*y(10);                                        % Shc
    dydt(11) = ka1*y(3)*y(10) - kShcDephos*y(12)*y(11);                % pEGFR:Shc
    dydt(12) = -kptpDeg*y(11)*y(12);                                   % pShc
    
    dydt(13) = kGrb2CombShc*y(11)*y(3) - kSprtyInbGrb2*y(27)*y(13);    % Grb2 (with SPRY feedback)
    dydt(14) = kSosCombGrb2*y(13)*y(11) - kErkPhosSos*y(25)*y(14);     % pShc:Grb2:SOS (with ERK feedback)
    
    % ========================================================================
    % MODULE 3: RAS ACTIVATION
    % ========================================================================
    dydt(15) = -ka1*y(14)*y(15);                                       % HRAS-GDP
    dydt(16) = ka1*y(14)*y(15);                                        % HRAS-GTP
    dydt(17) = -ka1*y(14)*y(17);                                       % NRAS-GDP
    dydt(18) = ka1*y(14)*y(17);                                        % NRAS-GTP
    dydt(19) = -ka1*y(14)*y(19);                                       % KRAS-GDP
    dydt(20) = ka1*y(14)*y(19) - ka1*y(20)*y(21);                      % KRAS-GTP
    dydt(21) = -ka1*y(20)*y(21);                                       % KRAS intermediate
    
    % ========================================================================
    % MODULE 4: RAF SIGNALING WITH PARADOXICAL ACTIVATION
    % ========================================================================
    
    % Vemurafenib inhibition of BRAF^V600E (Hill equation)
    IC50_n = IC50_vem^Hill_n_vem;
    Vem_n = Vemurafenib^Hill_n_vem;
    kBRAF_eff = ka1 * IC50_n / (IC50_n + Vem_n + eps);  % Effective BRAF activation (inhibited)
    
    % Paradoxical activation: vemurafenib promotes BRAF-WT*:CRAF dimer → CRAF* activation
    paradox_activation = kParadoxCRAF * Vemurafenib * y(62);  % y(62) = BRAF-WT:CRAF dimer
    
    % CRAF dynamics
    dydt(22) = -kpCraf*y(20)*y(22) + kErkPhosPcraf*y(29)*y(23) + kPcrafDegrad*y(23)*y(36) ...
               - kDimerForm*y(25)*y(22)*Vemurafenib + kDimerDissoc*y(62);  % CRAF (used in dimer)
    dydt(23) = kpCraf*y(20)*y(22) - kErkPhosPcraf*y(29)*y(23) - kPcrafDegrad*y(23)*y(36) ...
               + paradox_activation;  % pCRAF (with paradoxical activation)
    
    % BRAF dynamics (with vemurafenib inhibition)
    dydt(24) = -kBRAF_eff*y(24)*y(20) - kDimerForm*y(25)*y(22)*Vemurafenib + kDimerDissoc*y(62);
    dydt(25) = kBRAF_eff*y(24)*y(20) - kinbBraf*y(25) - kDimerForm*y(25)*y(22)*Vemurafenib + kDimerDissoc*y(62);
    
    % BRAF-WT:CRAF dimer dynamics (y(62) = dimer complex)
    dydt(62) = kDimerForm*y(25)*y(22)*Vemurafenib - kDimerDissoc*y(62) - kPcrafDegrad*y(62)*y(36);
    
    % ========================================================================
    % MODULE 5: MEK PHOSPHORYLATION
    % ========================================================================
    % MEK can be phosphorylated by CRAF, BRAF, or KSR
    raf_to_mek = (kpMek*y(23) + kMekByBraf*y(25) + kMekByCraf*y(23));
    ksr_to_mek = (kMekByKSR * y(61));
    
    dydt(26) = -(raf_to_mek + ksr_to_mek)*y(26) + kErkPhosMek*y(29)*y(27) + kMekDegrad*y(27)*y(35);
    dydt(27) = (raf_to_mek + ksr_to_mek)*y(26) - kErkPhosMek*y(29)*y(27) - kMekDegrad*y(27)*y(35);
    
    % ========================================================================
    % MODULE 6: ERK PHOSPHORYLATION
    % ========================================================================
    dydt(28) = -kpErk*y(27)*y(28) + kDuspInbErk*y(31)*y(29) + kErkDeg*y(29)*y(34) + kErkDephos*y(31)*y(29);
    dydt(29) = kpErk*y(27)*y(28) - kDuspInbErk*y(31)*y(29) - kErkDeg*y(29)*y(34) - kErkDephos*y(31)*y(29);
    
    % ========================================================================
    % MODULE 7: DUSP FEEDBACK
    % ========================================================================
    dydt(30) = km_Dusp*y(29)/(1 + (km_Dusp/kDusps)*y(29)) - kDuspStop*y(30)*y(37) - kDuspDeg*y(30)*y(29);
    dydt(31) = -kDuspStop*y(30)*y(31);
    
    % ========================================================================
    % MODULE 8: SPRY FEEDBACK
    % ========================================================================
    dydt(32) = km_Sprty*y(29)/(1 + (km_Sprty/kSproutyForm)*y(29)) - kSprtyComeDown*y(32)*y(33);
    dydt(33) = -kSprtyComeDown*y(32)*y(33);
    
    % ========================================================================
    % MODULE 9: DEGRADATION TRACKERS
    % ========================================================================
    dydt(34) = -kErkDeg*y(29)*y(34);      % pERK degradation tracker
    dydt(35) = -kMekDegrad*y(27)*y(35);   % pMEK degradation tracker
    dydt(36) = -kPcrafDegrad*y(23)*y(36); % pCRAF degradation tracker
    dydt(37) = -kDuspStop*y(30)*y(37);    % DUSP stop tracker
    
    % ========================================================================
    % MODULE 10: PI3K/AKT/mTOR SIGNALING
    % ========================================================================
    % IGFR module
    dydt(38) = -ka1*y(38) + kr1*y(39);
    dydt(39) = ka1*y(38) - kr1*y(39) - kc1*y(39);
    dydt(40) = kc1*y(39) - kErkInbEgfr*y(29)*y(40);  % pIGFR
    
    % IRS module (moved before p85)
    dydt(41) = -ka1*y(3)*y(41);  % IRS
    dydt(42) = ka1*y(3)*y(41);   % pIRS
    
    % p85 regulatory subunit binding to phosphotyrosine sites
    % p85 binds to: pEGFR (y(3)), pHer2 (y(6)), pHer3 (y(9)), pIGFR (y(40))
    % Free p85 (y(43))
    dydt(43) = -k_p85_bind_EGFR*y(3)*y(43) ...
               - k_p85_bind_Her2*y(6)*y(43) ...
               - k_p85_bind_Her3*y(9)*y(43) ...
               - k_p85_bind_IGFR*y(40)*y(43) ...
               + k_p85_unbind*(y(44) + y(45) + y(46) + y(47));
    
    % p85:RTK complexes
    dydt(44) = k_p85_bind_EGFR*y(3)*y(43) - k_p85_unbind*y(44);  % p85:pEGFR
    dydt(45) = k_p85_bind_Her2*y(6)*y(43) - k_p85_unbind*y(45);  % p85:pHer2
    dydt(46) = k_p85_bind_Her3*y(9)*y(43) - k_p85_unbind*y(46);  % p85:pHer3
    dydt(47) = k_p85_bind_IGFR*y(40)*y(43) - k_p85_unbind*y(47); % p85:pIGFR
    
    % PI3K module - recruitment by p85:RTK complexes
    % Total p85:RTK complexes recruit PI3K (p110 catalytic subunit)
    total_p85_RTK = y(44) + y(45) + y(46) + y(47);
    
    dydt(48) = -k_PI3K_recruit * total_p85_RTK * y(48) ...
               + kMTOR_Feedback * y(56) * y(49);  % PI3K (p110, inactive)
    dydt(49) = k_PI3K_recruit * total_p85_RTK * y(48) ...
               - kMTOR_Feedback * y(56) * y(49);  % PI3K_active (recruited to membrane)
    
    % PIP module - PIP2 (y(50)) and PIP3 (y(51))
    % PIP2 → PIP3 conversion by active PI3K
    % PIP3 → PIP2 dephosphorylation by PTEN
    dydt(50) = -k_PIP2_to_PIP3 * y(49) * y(50) ...  % PIP2 consumed by active PI3K
               + k_PTEN * y(51);                     % PIP3 dephosphorylated back to PIP2 by PTEN
    dydt(51) = k_PIP2_to_PIP3 * y(49) * y(50) ...   % PIP2 converted to PIP3 by active PI3K
               - k_PTEN * y(51);                     % PIP3 dephosphorylated by PTEN
    
    % AKT module - AKT (y(52)) and pAKT (y(53))
    % AKT activation by PIP3 (simplified: PIP3 recruits and activates AKT)
    % In reality: PIP3 recruits AKT to membrane, PDK1 phosphorylates Thr308, mTORC2 phosphorylates Ser473
    dydt(52) = -kAkt * y(51) * y(52) ...      % AKT consumed by PIP3-mediated activation
               + kdegradAKT * y(53);           % pAKT dephosphorylated back to AKT
    dydt(53) = kAkt * y(51) * y(52) ...       % AKT activated by PIP3
               - kdegradAKT * y(53);           % pAKT dephosphorylated/degraded
    
    % FOXO module - FOXO (y(54))
    % FOXO is inactivated by pAKT (simplified model)
    dydt(54) = (1 - y(53)) * ka1 / (1 + (y(54) / 15e-5));  % FOXO dynamics (inhibited by pAKT)
    
    % mTORC module - mTORC (y(55)) and mTORC_active (y(56))
    % mTORC activation by pAKT
    dydt(55) = -ka1 * y(53) * y(55) + kdegrad * y(56);  % mTORC (inactive)
    dydt(56) = ka1 * y(53) * y(55) - kdegrad * y(56);   % mTORC_active (activated by pAKT)
    
    % 4EBP1 module - 4EBP1 (y(57)), intermediate (y(58)), p4EBP1 (y(59))
    % 4EBP1 phosphorylation by active mTORC
    dydt(57) = -k4ebp1 * y(56) * y(57) + kb1 * y(58) + k_4EBP1_dephos * y(59);  % 4EBP1 (unphosphorylated) - receives dephosphorylated p4EBP1
    dydt(58) = k4ebp1 * y(56) * y(57) - kb1 * y(58) - k43b1 * y(58);  % 4EBP1 intermediate
    dydt(59) = k43b1 * y(58) - k_4EBP1_dephos * y(59);  % p4EBP1 (fully phosphorylated) - can be dephosphorylated
    
    % ========================================================================
    % MODULE 11: KSR SCAFFOLD
    % ========================================================================
    % KSR (y(60)) and pKSR (y(61))
    % KSR phosphorylation by RAS-GTP (HRAS-GTP: y(16), NRAS-GTP: y(18), KRAS-GTP: y(21))
    dydt(60) = -kKSRphos * (y(16) + y(18) + y(21)) * y(60) + kKSRdephos * y(61);  % KSR
    dydt(61) = kKSRphos * (y(16) + y(18) + y(21)) * y(60) - kKSRdephos * y(61);   % pKSR
    
end
