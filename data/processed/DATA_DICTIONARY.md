# Harmonized Electrochemical Database — Data Dictionary

All tables use standardized physical units: Corrosion Potentials ($E_{\text{corr}}$) in **V vs. SHE**, Corrosion Current Densities ($i_{\text{corr}}$) in **µA/cm²**, and Tafel Slopes ($\beta_a, \beta_c$) in strictly positive **mV/dec**.

## Table: `electrochemical_database`

| Column | Role | Unit | Type | Non-Null | Description |
| :--- | :--- | :--- | :--- | ---: | :--- |
| `record_id` | identifier | dimensionless | `object` | 180 | Unique electrochemical measurement record identifier. |
| `material_id` | identifier | dimensionless | `object` | 180 | Canonical Metal identifier matching symbol. |
| `solution_id` | identifier | dimensionless | `object` | 180 | Unique Solution formulation identifier. |
| `metal` | categorical | dimensionless | `object` | 180 | Conductive metallic element or alloy film (Cu, Co, W, Ru, Ti, Ta, Al, Mo). |
| `is_target_metal` | attribute | dimensionless | `object` | 147 | Harmonized attribute 'is_target_metal' in electrochemical_database. |
| `material_assignment_method` | attribute | dimensionless | `object` | 147 | Harmonized attribute 'material_assignment_method' in electrochemical_database. |
| `paper_primary_material` | attribute | dimensionless | `object` | 147 | Harmonized attribute 'paper_primary_material' in electrochemical_database. |
| `material_mention_counts` | attribute | dimensionless | `object` | 147 | Harmonized attribute 'material_mention_counts' in electrochemical_database. |
| `sample_name` | attribute | dimensionless | `object` | 109 | Harmonized attribute 'sample_name' in electrochemical_database. |
| `metal_form` | attribute | dimensionless | `object` | 128 | Harmonized attribute 'metal_form' in electrochemical_database. |
| `surface_preparation` | attribute | dimensionless | `object` | 124 | Harmonized attribute 'surface_preparation' in electrochemical_database. |
| `solution_description` | attribute | dimensionless | `object` | 136 | Harmonized attribute 'solution_description' in electrochemical_database. |
| `pH` | numeric | pH units | `float64` | 180 | Hydrogen ion activity (-log10[H+]) of the Solution. |
| `temperature_C` | numeric | °C | `float64` | 180 | Operating temperature of the Exposure Condition in degrees Celsius. |
| `reference_electrode` | categorical | dimensionless | `object` | 180 | Original experimental reference electrode before conversion to SHE. |
| `Ecorr_V_SHE` | numeric | V vs. SHE | `float64` | 180 | Standardized Corrosion Potential (E_corr) converted to Volts vs. Standard Hydrogen Electrode. |
| `Ecorr_V_original` | numeric | V vs. ref | `float64` | 180 | Originally reported open-circuit Corrosion Potential prior to SHE conversion. |
| `icorr_uA_cm2` | numeric | µA/cm² | `float64` | 180 | Standardized Corrosion Current Density (i_corr) at E_corr in microamperes per square centimeter. |
| `log_icorr` | attribute | dimensionless | `float64` | 147 | Harmonized attribute 'log_icorr' in electrochemical_database. |
| `beta_a_mV_dec` | numeric | mV/dec | `float64` | 180 | Strictly positive anodic Tafel Slope (beta_a) in millivolts per decade of current density. |
| `beta_c_mV_dec` | numeric | mV/dec | `float64` | 180 | Strictly positive cathodic Tafel Slope (beta_c) in millivolts per decade of current density. |
| `tafel_slopes_general` | attribute | dimensionless | `object` | 2 | Harmonized attribute 'tafel_slopes_general' in electrochemical_database. |
| `Rp_ohm_cm2` | attribute | dimensionless | `float64` | 11 | Harmonized attribute 'Rp_ohm_cm2' in electrochemical_database. |
| `Rp_unit` | attribute | dimensionless | `object` | 11 | Harmonized attribute 'Rp_unit' in electrochemical_database. |
| `CR_reported_value` | attribute | dimensionless | `float64` | 24 | Harmonized attribute 'CR_reported_value' in electrochemical_database. |
| `CR_reported_unit` | attribute | dimensionless | `object` | 24 | Harmonized attribute 'CR_reported_unit' in electrochemical_database. |
| `CR_calculated_nm_min` | attribute | dimensionless | `float64` | 147 | Harmonized attribute 'CR_calculated_nm_min' in electrochemical_database. |
| `measurement_method` | attribute | dimensionless | `object` | 128 | Harmonized attribute 'measurement_method' in electrochemical_database. |
| `scan_rate` | attribute | dimensionless | `object` | 9 | Harmonized attribute 'scan_rate' in electrochemical_database. |
| `immersion_time` | attribute | dimensionless | `object` | 74 | Harmonized attribute 'immersion_time' in electrochemical_database. |
| `aeration_condition` | attribute | dimensionless | `object` | 18 | Harmonized attribute 'aeration_condition' in electrochemical_database. |
| `reaction_type` | attribute | dimensionless | `object` | 147 | Harmonized attribute 'reaction_type' in electrochemical_database. |
| `reactions` | attribute | dimensionless | `object` | 147 | Harmonized attribute 'reactions' in electrochemical_database. |
| `source_type` | attribute | dimensionless | `object` | 180 | Harmonized attribute 'source_type' in electrochemical_database. |
| `source_reference` | attribute | dimensionless | `object` | 180 | Harmonized attribute 'source_reference' in electrochemical_database. |
| `data_extraction_method` | attribute | dimensionless | `object` | 147 | Harmonized attribute 'data_extraction_method' in electrochemical_database. |
| `data_quality_score` | quality_metric | score [0, 1] | `float64` | 180 | Raw ingestion data quality score prior to imputation penalty. |
| `quality_notes` | attribute | dimensionless | `object` | 147 | Harmonized attribute 'quality_notes' in electrochemical_database. |
| `notes` | attribute | dimensionless | `float64` | 0 | Harmonized attribute 'notes' in electrochemical_database. |
| `_source_metal` | attribute | dimensionless | `object` | 147 | Harmonized attribute '_source_metal' in electrochemical_database. |
| `solution_name` | text | dimensionless | `object` | 176 | Human-readable chemical formulation summary for the Solution. |
| `solution_category` | categorical | dimensionless | `object` | 176 | pH regime category of the Solution (acidic, neutral, alkaline, etc.). |
| `materials` | attribute | dimensionless | `object` | 143 | Harmonized attribute 'materials' in electrochemical_database. |
| `base_solvent` | categorical | dimensionless | `object` | 176 | Primary liquid solvent (typically DI water). |
| `oxidizer_type` | categorical | dimensionless | `object` | 180 | Primary oxidizing agent in the Solution (e.g., H2O2, HNO3, None). |
| `oxidizer_concentration` | numeric | wt% or M | `float64` | 180 | Numerical concentration of the oxidizing agent (0.0 when oxidizer_type is None). |
| `oxidizer_concentration_unit` | categorical | unit string | `object` | 46 | Concentration unit for oxidizer_concentration (e.g., wt%, M, mM). |
| `acid_type` | categorical | dimensionless | `object` | 108 | Acid component in the Solution formulation. |
| `acid_concentration` | numeric | M or wt% | `float64` | 108 | Concentration of the acid component. |
| `acid_concentration_unit` | categorical | unit string | `object` | 108 | Concentration unit for acid_concentration. |
| `base_type` | attribute | dimensionless | `object` | 5 | Harmonized attribute 'base_type' in electrochemical_database. |
| `base_concentration` | attribute | dimensionless | `float64` | 5 | Harmonized attribute 'base_concentration' in electrochemical_database. |
| `base_concentration_unit` | attribute | dimensionless | `object` | 5 | Harmonized attribute 'base_concentration_unit' in electrochemical_database. |
| `HF_concentration_wt_pct` | attribute | dimensionless | `float64` | 0 | Harmonized attribute 'HF_concentration_wt_pct' in electrochemical_database. |
| `chelating_agent_type` | categorical | dimensionless | `object` | 49 | Complexing or chelating agent (e.g., citric acid, glycine, oxalic acid, EDTA). |
| `chelating_agent_concentration` | numeric | mM or wt% | `float64` | 49 | Concentration of the complexing/chelating agent. |
| `chelating_agent_concentration_unit` | categorical | unit string | `object` | 49 | Concentration unit for chelating_agent_concentration. |
| `inhibitor_type` | categorical | dimensionless | `object` | 39 | Corrosion inhibitor chemical species (e.g., BTA). |
| `inhibitor_concentration` | numeric | mM or wt% | `float64` | 39 | Concentration of the corrosion inhibitor. |
| `inhibitor_concentration_unit` | categorical | unit string | `object` | 39 | Concentration unit for inhibitor_concentration. |
| `organic_solvent_type` | attribute | dimensionless | `object` | 2 | Harmonized attribute 'organic_solvent_type' in electrochemical_database. |
| `organic_solvent_concentration` | attribute | dimensionless | `float64` | 2 | Harmonized attribute 'organic_solvent_concentration' in electrochemical_database. |
| `electrolyte_additive` | attribute | dimensionless | `object` | 62 | Harmonized attribute 'electrolyte_additive' in electrochemical_database. |
| `electrolyte_additive_concentration` | attribute | dimensionless | `float64` | 62 | Harmonized attribute 'electrolyte_additive_concentration' in electrochemical_database. |
| `electrolyte_additive_concentration_unit` | attribute | dimensionless | `object` | 62 | Harmonized attribute 'electrolyte_additive_concentration_unit' in electrochemical_database. |
| `dissolved_oxygen_ppm` | attribute | dimensionless | `float64` | 0 | Harmonized attribute 'dissolved_oxygen_ppm' in electrochemical_database. |
| `conductivity_mS_cm` | attribute | dimensionless | `float64` | 0 | Harmonized attribute 'conductivity_mS_cm' in electrochemical_database. |
| `ionic_strength_mol_L` | attribute | dimensionless | `float64` | 0 | Harmonized attribute 'ionic_strength_mol_L' in electrochemical_database. |
| `ORP_mV` | numeric | mV | `float64` | 33 | Solution Oxidation-Reduction Potential. |
| `deaerated` | attribute | dimensionless | `object` | 143 | Harmonized attribute 'deaerated' in electrochemical_database. |
| `source` | attribute | dimensionless | `object` | 143 | Harmonized attribute 'source' in electrochemical_database. |
| `material_name` | text | dimensionless | `object` | 180 | Full English name of the Metal. |
| `symbol` | categorical | dimensionless | `object` | 180 | Chemical symbol of the Metal. |
| `category` | categorical | dimensionless | `object` | 180 | Semiconductor film functional category (metal, barrier, liner). |
| `density_g_cm3` | numeric | g/cm³ | `float64` | 180 | Bulk mass density of the Metal film. |
| `molar_mass_g_mol` | numeric | g/mol | `float64` | 180 | Atomic or molar mass of the Metal. |
| `valence` | numeric | e⁻/atom | `float64` | 180 | Dissolution Valence (z): electrons transferred per dissolved Metal atom in Faraday's law. |
| `equivalent_weight_g_eq` | numeric | g/eq | `float64` | 180 | Electrochemical equivalent weight (molar_mass / valence). |
| `standard_potential_V_SHE` | numeric | V vs. SHE | `float64` | 180 | Standard reduction potential E° of the Metal in V vs. SHE. |
| `melting_point_C` | numeric | °C | `float64` | 180 | Melting temperature of the Metal at 1 atm. |
| `electronegativity_pauling` | numeric | Pauling scale | `float64` | 180 | Pauling electronegativity of the Metal. |
| `work_function_eV` | numeric | eV | `float64` | 180 | Electronic work function of the Metal surface in electron-volts. |
| `oxide_type` | text | chemical formula | `object` | 180 | Primary surface oxide formulas formed on the Metal. |
| `oxide_stability` | categorical | tier | `object` | 180 | Qualitative thermodynamic stability tier of the native oxide (low, medium, high). |
| `passivation_tendency` | numeric | index [0, 1] | `float64` | 180 | Dimensionless tendency of the Metal to form a self-limiting passive oxide film. |
| `pourbaix_stable_pH_min` | numeric | pH units | `float64` | 180 | Lower pH boundary of the Metal's passive oxide stability window. |
| `pourbaix_stable_pH_max` | numeric | pH units | `float64` | 180 | Upper pH boundary of the Metal's passive oxide stability window. |
| `crystal_structure` | categorical | lattice type | `object` | 180 | Room-temperature crystal structure (FCC, BCC, HCP). |
| `thermal_conductivity_W_mK` | numeric | W/(m·K) | `float64` | 180 | Bulk thermal conductivity of the Metal. |
| `electrical_resistivity_ohm_cm` | numeric | Ω·cm | `float64` | 180 | Bulk electrical resistivity of the Metal. |
| `common_corrosion_products` | text | chemical species | `object` | 180 | Typical dissolved ionic or solid corrosion products. |
| `base_element` | attribute | dimensionless | `object` | 147 | Harmonized attribute 'base_element' in electrochemical_database. |
| `pH_effective` | attribute | dimensionless | `float64` | 147 | Harmonized attribute 'pH_effective' in electrochemical_database. |
| `pH_source` | attribute | dimensionless | `object` | 147 | Harmonized attribute 'pH_source' in electrochemical_database. |
| `elem_atomic_number` | attribute | dimensionless | `float64` | 147 | Harmonized attribute 'elem_atomic_number' in electrochemical_database. |
| `elem_atomic_mass` | attribute | dimensionless | `float64` | 147 | Harmonized attribute 'elem_atomic_mass' in electrochemical_database. |
| `elem_pauling_en` | attribute | dimensionless | `float64` | 147 | Harmonized attribute 'elem_pauling_en' in electrochemical_database. |
| `elem_work_function_eV` | attribute | dimensionless | `float64` | 147 | Harmonized attribute 'elem_work_function_eV' in electrochemical_database. |
| `elem_standard_potential_V_SHE` | attribute | dimensionless | `float64` | 147 | Harmonized attribute 'elem_standard_potential_V_SHE' in electrochemical_database. |
| `elem_melting_point_K` | attribute | dimensionless | `float64` | 147 | Harmonized attribute 'elem_melting_point_K' in electrochemical_database. |
| `elem_molar_volume_cm3` | attribute | dimensionless | `float64` | 147 | Harmonized attribute 'elem_molar_volume_cm3' in electrochemical_database. |
| `mp_material_id` | attribute | dimensionless | `object` | 147 | Harmonized attribute 'mp_material_id' in electrochemical_database. |
| `mp_work_function_eV` | attribute | dimensionless | `float64` | 147 | Harmonized attribute 'mp_work_function_eV' in electrochemical_database. |
| `mp_surface_energy_J_m2` | attribute | dimensionless | `float64` | 147 | Harmonized attribute 'mp_surface_energy_J_m2' in electrochemical_database. |
| `mp_formation_energy_eV_atom` | attribute | dimensionless | `float64` | 147 | Harmonized attribute 'mp_formation_energy_eV_atom' in electrochemical_database. |
| `mp_band_gap_eV` | attribute | dimensionless | `float64` | 147 | Harmonized attribute 'mp_band_gap_eV' in electrochemical_database. |
| `thermo_E_onset_V_SHE` | attribute | dimensionless | `float64` | 147 | Harmonized attribute 'thermo_E_onset_V_SHE' in electrochemical_database. |
| `thermo_corrosion_regime` | attribute | dimensionless | `object` | 145 | Harmonized attribute 'thermo_corrosion_regime' in electrochemical_database. |
| `thermo_stable_phase` | attribute | dimensionless | `object` | 145 | Harmonized attribute 'thermo_stable_phase' in electrochemical_database. |
| `thermo_source` | attribute | dimensionless | `object` | 147 | Harmonized attribute 'thermo_source' in electrochemical_database. |
| `mp_pourbaix_E_onset_V_SHE` | attribute | dimensionless | `float64` | 147 | Harmonized attribute 'mp_pourbaix_E_onset_V_SHE' in electrochemical_database. |
| `mp_stable_phase_at_Ecorr` | attribute | dimensionless | `object` | 145 | Harmonized attribute 'mp_stable_phase_at_Ecorr' in electrochemical_database. |
| `mp_corrosion_regime` | attribute | dimensionless | `object` | 145 | Harmonized attribute 'mp_corrosion_regime' in electrochemical_database. |
| `mp_delta_G_pbx_eV_atom` | attribute | dimensionless | `float64` | 145 | Harmonized attribute 'mp_delta_G_pbx_eV_atom' in electrochemical_database. |
| `delta_E_overpotential_V` | attribute | dimensionless | `float64` | 145 | Harmonized attribute 'delta_E_overpotential_V' in electrochemical_database. |
| `log10_icorr_uA_cm2` | numeric | log10(µA/cm²) | `float64` | 180 | Base-10 logarithm of standardized Corrosion Current Density i_corr. |
| `beta_a_imputed` | boolean_flag | boolean | `bool` | 180 | True if anodic Tafel slope beta_a was missing and imputed via group median or 60 mV/dec fallback. |
| `beta_c_imputed` | boolean_flag | boolean | `bool` | 180 | True if cathodic Tafel slope beta_c was missing and imputed via group median or 120 mV/dec fallback. |
| `is_imputed` | boolean_flag | boolean | `bool` | 180 | Quality flag indicating whether either Tafel slope (beta_a or beta_c) was imputed. |
| `quality_score` | quality_metric | score [0, 1] | `float64` | 180 | Data reliability score in [0, 1], automatically decremented when is_imputed is True. |

## Table: `material_properties`

| Column | Role | Unit | Type | Non-Null | Description |
| :--- | :--- | :--- | :--- | ---: | :--- |
| `metal` | categorical | dimensionless | `object` | 8 | Conductive metallic element or alloy film (Cu, Co, W, Ru, Ti, Ta, Al, Mo). |
| `material_id` | identifier | dimensionless | `object` | 8 | Canonical Metal identifier matching symbol. |
| `material_name` | text | dimensionless | `object` | 8 | Full English name of the Metal. |
| `symbol` | categorical | dimensionless | `object` | 8 | Chemical symbol of the Metal. |
| `category` | categorical | dimensionless | `object` | 8 | Semiconductor film functional category (metal, barrier, liner). |
| `density_g_cm3` | numeric | g/cm³ | `float64` | 8 | Bulk mass density of the Metal film. |
| `molar_mass_g_mol` | numeric | g/mol | `float64` | 8 | Atomic or molar mass of the Metal. |
| `valence` | numeric | e⁻/atom | `int64` | 8 | Dissolution Valence (z): electrons transferred per dissolved Metal atom in Faraday's law. |
| `equivalent_weight_g_eq` | numeric | g/eq | `float64` | 8 | Electrochemical equivalent weight (molar_mass / valence). |
| `standard_potential_V_SHE` | numeric | V vs. SHE | `float64` | 8 | Standard reduction potential E° of the Metal in V vs. SHE. |
| `melting_point_C` | numeric | °C | `float64` | 8 | Melting temperature of the Metal at 1 atm. |
| `electronegativity_pauling` | numeric | Pauling scale | `float64` | 8 | Pauling electronegativity of the Metal. |
| `work_function_eV` | numeric | eV | `float64` | 8 | Electronic work function of the Metal surface in electron-volts. |
| `oxide_type` | text | chemical formula | `object` | 8 | Primary surface oxide formulas formed on the Metal. |
| `oxide_stability` | categorical | tier | `object` | 8 | Qualitative thermodynamic stability tier of the native oxide (low, medium, high). |
| `passivation_tendency` | numeric | index [0, 1] | `float64` | 8 | Dimensionless tendency of the Metal to form a self-limiting passive oxide film. |
| `pourbaix_stable_pH_min` | numeric | pH units | `float64` | 8 | Lower pH boundary of the Metal's passive oxide stability window. |
| `pourbaix_stable_pH_max` | numeric | pH units | `float64` | 8 | Upper pH boundary of the Metal's passive oxide stability window. |
| `crystal_structure` | categorical | lattice type | `object` | 8 | Room-temperature crystal structure (FCC, BCC, HCP). |
| `thermal_conductivity_W_mK` | numeric | W/(m·K) | `float64` | 8 | Bulk thermal conductivity of the Metal. |
| `electrical_resistivity_ohm_cm` | numeric | Ω·cm | `float64` | 8 | Bulk electrical resistivity of the Metal. |
| `common_corrosion_products` | text | chemical species | `object` | 8 | Typical dissolved ionic or solid corrosion products. |
| `notes` | attribute | dimensionless | `float64` | 0 | Harmonized attribute 'notes' in material_properties. |
| `_source_metal` | attribute | dimensionless | `object` | 8 | Harmonized attribute '_source_metal' in material_properties. |

## Table: `solution_properties`

| Column | Role | Unit | Type | Non-Null | Description |
| :--- | :--- | :--- | :--- | ---: | :--- |
| `solution_id` | identifier | dimensionless | `object` | 149 | Unique Solution formulation identifier. |
| `solution_name` | text | dimensionless | `object` | 148 | Human-readable chemical formulation summary for the Solution. |
| `solution_category` | categorical | dimensionless | `object` | 148 | pH regime category of the Solution (acidic, neutral, alkaline, etc.). |
| `materials` | attribute | dimensionless | `object` | 144 | Harmonized attribute 'materials' in solution_properties. |
| `is_target_metal` | attribute | dimensionless | `object` | 144 | Harmonized attribute 'is_target_metal' in solution_properties. |
| `base_solvent` | categorical | dimensionless | `object` | 148 | Primary liquid solvent (typically DI water). |
| `pH` | numeric | pH units | `float64` | 149 | Hydrogen ion activity (-log10[H+]) of the Solution. |
| `temperature_C` | numeric | °C | `float64` | 149 | Operating temperature of the Exposure Condition in degrees Celsius. |
| `oxidizer_type` | categorical | dimensionless | `object` | 149 | Primary oxidizing agent in the Solution (e.g., H2O2, HNO3, None). |
| `oxidizer_concentration` | numeric | wt% or M | `float64` | 149 | Numerical concentration of the oxidizing agent (0.0 when oxidizer_type is None). |
| `oxidizer_concentration_unit` | categorical | unit string | `object` | 30 | Concentration unit for oxidizer_concentration (e.g., wt%, M, mM). |
| `acid_type` | categorical | dimensionless | `object` | 52 | Acid component in the Solution formulation. |
| `acid_concentration` | numeric | M or wt% | `float64` | 52 | Concentration of the acid component. |
| `acid_concentration_unit` | categorical | unit string | `object` | 52 | Concentration unit for acid_concentration. |
| `base_type` | attribute | dimensionless | `object` | 31 | Harmonized attribute 'base_type' in solution_properties. |
| `base_concentration` | attribute | dimensionless | `float64` | 31 | Harmonized attribute 'base_concentration' in solution_properties. |
| `base_concentration_unit` | attribute | dimensionless | `object` | 31 | Harmonized attribute 'base_concentration_unit' in solution_properties. |
| `HF_concentration_wt_pct` | attribute | dimensionless | `float64` | 1 | Harmonized attribute 'HF_concentration_wt_pct' in solution_properties. |
| `chelating_agent_type` | categorical | dimensionless | `object` | 21 | Complexing or chelating agent (e.g., citric acid, glycine, oxalic acid, EDTA). |
| `chelating_agent_concentration` | numeric | mM or wt% | `float64` | 21 | Concentration of the complexing/chelating agent. |
| `chelating_agent_concentration_unit` | categorical | unit string | `object` | 21 | Concentration unit for chelating_agent_concentration. |
| `inhibitor_type` | categorical | dimensionless | `object` | 15 | Corrosion inhibitor chemical species (e.g., BTA). |
| `inhibitor_concentration` | numeric | mM or wt% | `float64` | 15 | Concentration of the corrosion inhibitor. |
| `inhibitor_concentration_unit` | categorical | unit string | `object` | 15 | Concentration unit for inhibitor_concentration. |
| `organic_solvent_type` | attribute | dimensionless | `object` | 1 | Harmonized attribute 'organic_solvent_type' in solution_properties. |
| `organic_solvent_concentration` | attribute | dimensionless | `float64` | 1 | Harmonized attribute 'organic_solvent_concentration' in solution_properties. |
| `electrolyte_additive` | attribute | dimensionless | `object` | 67 | Harmonized attribute 'electrolyte_additive' in solution_properties. |
| `electrolyte_additive_concentration` | attribute | dimensionless | `float64` | 67 | Harmonized attribute 'electrolyte_additive_concentration' in solution_properties. |
| `electrolyte_additive_concentration_unit` | attribute | dimensionless | `object` | 67 | Harmonized attribute 'electrolyte_additive_concentration_unit' in solution_properties. |
| `dissolved_oxygen_ppm` | attribute | dimensionless | `float64` | 0 | Harmonized attribute 'dissolved_oxygen_ppm' in solution_properties. |
| `conductivity_mS_cm` | attribute | dimensionless | `float64` | 0 | Harmonized attribute 'conductivity_mS_cm' in solution_properties. |
| `ionic_strength_mol_L` | attribute | dimensionless | `float64` | 0 | Harmonized attribute 'ionic_strength_mol_L' in solution_properties. |
| `ORP_mV` | numeric | mV | `float64` | 4 | Solution Oxidation-Reduction Potential. |
| `deaerated` | attribute | dimensionless | `object` | 144 | Harmonized attribute 'deaerated' in solution_properties. |
| `source` | attribute | dimensionless | `object` | 144 | Harmonized attribute 'source' in solution_properties. |
| `notes` | attribute | dimensionless | `float64` | 0 | Harmonized attribute 'notes' in solution_properties. |
| `_source_metal` | attribute | dimensionless | `object` | 144 | Harmonized attribute '_source_metal' in solution_properties. |

## Table: `galvanic_series`

| Column | Role | Unit | Type | Non-Null | Description |
| :--- | :--- | :--- | :--- | ---: | :--- |
| `solution_id` | identifier | dimensionless | `object` | 71 | Unique Solution formulation identifier. |
| `metal` | categorical | dimensionless | `object` | 71 | Conductive metallic element or alloy film (Cu, Co, W, Ru, Ti, Ta, Al, Mo). |
| `solution_name` | text | dimensionless | `object` | 70 | Human-readable chemical formulation summary for the Solution. |
| `pH` | numeric | pH units | `float64` | 71 | Hydrogen ion activity (-log10[H+]) of the Solution. |
| `oxidizer_type` | categorical | dimensionless | `object` | 71 | Primary oxidizing agent in the Solution (e.g., H2O2, HNO3, None). |
| `oxidizer_concentration` | numeric | wt% or M | `float64` | 71 | Numerical concentration of the oxidizing agent (0.0 when oxidizer_type is None). |
| `temperature_C` | numeric | °C | `float64` | 71 | Operating temperature of the Exposure Condition in degrees Celsius. |
| `Ecorr_V_SHE` | numeric | V vs. SHE | `float64` | 71 | Standardized Corrosion Potential (E_corr) converted to Volts vs. Standard Hydrogen Electrode. |
| `icorr_uA_cm2` | numeric | µA/cm² | `float64` | 71 | Standardized Corrosion Current Density (i_corr) at E_corr in microamperes per square centimeter. |
| `beta_a_mV_dec` | numeric | mV/dec | `float64` | 71 | Strictly positive anodic Tafel Slope (beta_a) in millivolts per decade of current density. |
| `beta_c_mV_dec` | numeric | mV/dec | `float64` | 71 | Strictly positive cathodic Tafel Slope (beta_c) in millivolts per decade of current density. |
| `record_count` | numeric | count | `int64` | 71 | Number of harmonized experimental records aggregated for that (Solution, Metal) pair. |
| `galvanic_rank` | numeric | integer rank (1..N) | `int64` | 71 | Monotonic Galvanic Series rank within a Solution (1 = most Active Metal, N = most Noble Metal). |
| `metals_in_solution` | numeric | count | `int64` | 71 | Total distinct Metals evaluated in that Solution within the Galvanic Series. |
| `relative_nobility` | categorical | role label | `object` | 71 | Qualitative galvanic role within the Solution (Most Active, Intermediate, Most Noble). |
