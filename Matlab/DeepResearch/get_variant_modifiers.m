function [params_new, variant_desc] = get_variant_modifiers(params_base, variant_name)
    % GET_VARIANT_MODIFIERS Simulates OpenFold3 structural insights by modifying model parameters
    %
    % Inputs:
    %   params_base: The idealized scalar parameter vector (63 elements)
    %   variant_name: String name of the variant (e.g., 'WT', 'V600E', 'L505H', 'RAS_G12V', 'MEK_C121S')
    %
    % Outputs:
    %   params_new: Modified parameter vector reflecting structural changes
    %   variant_desc: Description of the structural mechanism (for plotting)
    %
    % Structural Mapping Logic (simulating OpenFold3 ranking):
    % - Dimer Interface Mutations: Affect k_dimer_form (59) and k_dimer_dissoc (60)
    % - Drug Binding Domain Mutations: Affect IC50_vem (62), Ki_RAF (55)
    % - Catalytic Mutations: Affect k_cat (3), kpMek (5), etc.
    % - RAS Mutations: Affect activation/deactivation via Initial Conditions (handled in main script) or rate shifts
    
    params_new = params_base;
    variant_desc = 'Wild Type';
    
    % Parameter Indices (based on Vemurafenib.m mapping):
    % 58: Vem_conc, 59: kDimerForm, 60: kDimerDissoc, 61: Gamma (Paradox), 62: IC50_vem
    % 55: Ki_RAF (Trametinib)
    % 4: kpCraf (CRAF activation), 5: kpMek (MEK phosphorylation)
    
    switch variant_name
        case 'WT'
            % No changes
            variant_desc = 'Wild Type (Baseline)';
            
        case 'V600E'
            % V600E is constitutively active as a monomer, but also inhibited by Vemurafenib
            % In our model, V600E is often represented by initial condition BRAF_P
            % But here we might assume V600E destabilizes the inactive state
            % For this specific model structure, V600E is the 'Target' of Vemurafenib.
            % Let's assume the "WT" in this switch is actually the V600E cell line baseline.
            variant_desc = 'BRAF V600E Baseline';
            
        case 'L505H'
            % A known "Dimer Enhancer" mutation in RAF
            % Mechanism: Stabilizes the dimer interface
            % OpenFold3 Prediction: Computed DeltaDeltaG of dimer interface < -2.0 kcal/mol
            % Model translation: Lower k_dimer_dissoc significantly.
            
            % Decrease dissociation by 10x (stronger dimer)
            params_new(60) = params_base(60) * 0.1; 
            % Increase formation slightly (faster association)
            params_new(59) = params_base(59) * 2.0;
            
            variant_desc = 'L505H (Dimer Stabilized)';
            
        case 'Splice_Variant'
            % p61-BRAF-V600E (Splice variant)
            % Mechanism: Constitutive dimerization, resistant to Vemurafenib monomer binding
            % OpenFold3 Poses: Show steric clash with Vemurafenib in the dimer context or forced dimerization
            
            % Massive increase in dimer stability
            params_new(60) = params_base(60) * 0.01;
            params_new(59) = params_base(59) * 10.0;
            % Reduced Vemurafenib binding (IC50 shift)
            params_new(62) = params_base(62) * 50.0; % High resistance
            
            variant_desc = 'p61 Splice (Constitutive Dimer)';
            
        case 'MEK_C121S'
            % MEK mutation conferring Trametinib resistance
            % Mechanism: Blocks allosteric binding pocket of Trametinib
            % OpenFold3 Prediction: Trametinib clashes with Serine residue
            
            % Increase Trametinib Ki (lower affinity)
            % Parameter 55: Ki_RAF
            params_new(55) = params_base(55) * 100.0; % 100-fold resistance
            
            variant_desc = 'MEK C121S (Trametinib Resistant)';
            
        case 'RAS_G12V'
             % RAS mutation (GTP locked)
             % This is usually modeled by Initial Conditions (High RAS-GTP), but we can also
             % increase RAS activation rate or decrease hydrolysis
             % Params 15,17,19 are GDP->GTP rates (driven by SOS).
             % Params 16,18,20 are GTP hydrolysis (wait, let's check ODE).
             % Mapk_ODE: dydt(16) = ka1*y(14)*y(15); % HRAS-GTP formation
             % It seems implicit hydrolysis isn't detailed or is balanced.
             % Actually, looking at Mapk_ODE: dydt(20) = ka1*y(14)*y(19) - ka1*y(20)*y(21);
             % It seems RAS dynamics are driven by SOS.
             % To simulate G12V 'Lock', we effectively effectively ignore this parameter here
             % and handle it in the SIMULATION script by setting Initial Conditions y0(20/21) to 1.0
             variant_desc = 'RAS G12V (Hyperactive Upstream)';
             
        otherwise
            warning('Variant not recognized: %s. Using Baseline.', variant_name);
    end
end
