# Harmonized Electrochemical Database — Data Dictionary

All tables use standardized physical units: Corrosion Potentials ($E_{\text{corr}}$) in **V vs. SHE**, Corrosion Current Densities ($i_{\text{corr}}$) in **µA/cm²**, and Tafel Slopes ($\beta_a, \beta_c$) in strictly positive **mV/dec**.

## Table: `electrochemical_database`

| Column | Role | Unit | Type | Non-Null | Description |
| :--- | :--- | :--- | :--- | ---: | :--- |
| `record_id` | identifier | dimensionless | `object` | 160 | Unique electrochemical measurement record identifier. |
| `material_id` | identifier | dimensionless | `object` | 160 | Canonical Metal identifier matching symbol. |
| `solution_id` | identifier | dimensionless | `object` | 160 | Unique Solution formulation identifier. |
| `metal` | categorical | dimensionless | `object` | 160 | Conductive metallic element or alloy film (Cu, Co, W, Ru, Ti, Ta, Al, Mo). |
| `sample_name` | text | dimensionless | `object` | 109 | Original experimental sample or specimen designation reported in the source publication. |
| `metal_form` | categorical | dimensionless | `object` | 128 | Physical morphology or deposition form of the Metal (thin film, PVD, ALD, bulk, foil, rod). |
| `surface_preparation` | text | dimensionless | `object` | 124 | Pre-exposure surface conditioning or polishing treatment (e.g., CMP, mechanical polish, annealed). |
| `solution_description` | text | dimensionless | `object` | 136 | Raw text description of the Solution composition from the literature source. |
| `pH` | numeric | pH units | `float64` | 160 | Hydrogen ion activity (-log10[H+]) of the Solution. |
| `temperature_C` | numeric | °C | `float64` | 160 | Operating temperature of the Exposure Condition in degrees Celsius. |
| `reference_electrode` | categorical | dimensionless | `object` | 160 | Original experimental reference electrode before conversion to SHE. |
| `Ecorr_V_SHE` | numeric | V vs. SHE | `float64` | 160 | Standardized Corrosion Potential (E_corr) converted to Volts vs. Standard Hydrogen Electrode. |
| `Ecorr_V_original` | numeric | V vs. ref | `float64` | 160 | Originally reported open-circuit Corrosion Potential prior to SHE conversion. |
| `icorr_uA_cm2` | numeric | µA/cm² | `float64` | 160 | Standardized Corrosion Current Density (i_corr) at E_corr in microamperes per square centimeter. |
| `log_icorr` | numeric | log10(A/cm²) | `float64` | 147 | Raw base-10 logarithm of corrosion current density in A/cm² from literature compilation. |
| `beta_a_mV_dec` | numeric | mV/dec | `float64` | 160 | Strictly positive anodic Tafel Slope (beta_a) in millivolts per decade of current density. |
| `beta_c_mV_dec` | numeric | mV/dec | `float64` | 160 | Strictly positive cathodic Tafel Slope (beta_c) in millivolts per decade of current density. |
| `tafel_slopes_general` | text | mV/dec | `object` | 2 | Unassigned or general Tafel slope values extracted from literature text. |
| `Rp_ohm_cm2` | numeric | Ω·cm² | `float64` | 11 | Linear polarization resistance (R_p) in ohm-square centimeters. |
| `Rp_unit` | categorical | unit string | `object` | 11 | Reported physical unit for linear polarization resistance R_p. |
| `CR_reported_value` | numeric | reported unit | `float64` | 24 | Corrosion Rate value as originally reported in the source publication. |
| `CR_reported_unit` | categorical | unit string | `object` | 24 | Physical unit for CR_reported_value (e.g., nm/min, mm/year, Å/min). |
| `CR_calculated_nm_min` | numeric | nm/min | `float64` | 147 | Faraday-derived Corrosion Rate in nanometers per minute. |
| `measurement_method` | text | dimensionless | `object` | 128 | Experimental electrochemical or analytical measurement techniques (potentiodynamic, EIS, LPR, ICP-MS). |
| `scan_rate` | text | mV/s | `object` | 9 | Potentiodynamic polarization potential sweep rate. |
| `immersion_time` | text | time unit | `object` | 74 | Experimental immersion or pre-conditioning duration reported in the study. |
| `aeration_condition` | categorical | dimensionless | `object` | 18 | Dissolved gas saturation state during testing (aerated, deaerated, N2/Ar purged). |
| `reaction_type` | categorical | dimensionless | `object` | 147 | Primary electrochemical process classification (corrosion, passivity, etching). |
| `reactions` | text | dimensionless | `object` | 147 | Half-cell reactions active under the Exposure Condition (HER, ORR, OER, metal dissolution). |
| `source_type` | categorical | dimensionless | `object` | 160 | Literature or benchmark source classification (journal, seed_benchmark). |
| `source_reference` | text | DOI / ID | `object` | 160 | Publication DOI, table row identifier, or benchmark reference string. |
| `data_extraction_method` | categorical | dimensionless | `object` | 147 | Ingestion extraction pipeline method (table_column_mapped, regex, manual_curation). |
| `data_quality_score` | quality_metric | score [0, 1] | `float64` | 160 | Raw ingestion data quality score prior to imputation penalty. |
| `quality_notes` | text | dimensionless | `object` | 147 | Automated ingestion audit notes regarding unit inference or missing metadata. |
| `notes` | text | dimensionless | `float64` | 0 | Curator annotations or supplementary notes. |
| `solution_name` | text | dimensionless | `object` | 156 | Human-readable chemical formulation summary for the Solution. |
| `solution_category` | categorical | dimensionless | `object` | 156 | pH regime category of the Solution (acidic, neutral, alkaline, etc.). |
| `materials` | text | dimensionless | `object` | 143 | Semicolon-delimited list of Metals evaluated with this Solution. |
| `base_solvent` | categorical | dimensionless | `object` | 156 | Primary liquid solvent (typically DI water). |
| `oxidizer_type` | categorical | dimensionless | `object` | 160 | Primary oxidizing agent in the Solution (e.g., H2O2, HNO3, None). |
| `oxidizer_concentration` | numeric | wt% or M | `float64` | 160 | Numerical concentration of the oxidizing agent (0.0 when oxidizer_type is None). |
| `oxidizer_concentration_unit` | categorical | unit string | `object` | 26 | Concentration unit for oxidizer_concentration (e.g., wt%, M, mM). |
| `acid_type` | categorical | dimensionless | `object` | 88 | Acid component in the Solution formulation (e.g., H2SO4, HCl, HF, HClO4, citric acid). |
| `acid_concentration` | numeric | M or wt% | `float64` | 88 | Concentration of the acid component. |
| `acid_concentration_unit` | categorical | unit string | `object` | 88 | Concentration unit for acid_concentration. |
| `base_type` | categorical | dimensionless | `object` | 5 | Alkaline base component in the Solution formulation (e.g., KOH, NaOH, TMAH). |
| `base_concentration` | numeric | M or wt% | `float64` | 5 | Concentration of the alkaline base component. |
| `base_concentration_unit` | categorical | unit string | `object` | 5 | Concentration unit for base_concentration. |
| `HF_concentration_wt_pct` | numeric | wt% | `float64` | 0 | Hydrofluoric acid (HF) weight percent concentration in the Solution. |
| `chelating_agent_type` | categorical | dimensionless | `object` | 29 | Complexing or chelating agent (e.g., citric acid, glycine, oxalic acid, EDTA). |
| `chelating_agent_concentration` | numeric | mM or wt% | `float64` | 29 | Concentration of the complexing/chelating agent. |
| `chelating_agent_concentration_unit` | categorical | unit string | `object` | 29 | Concentration unit for chelating_agent_concentration. |
| `inhibitor_type` | categorical | dimensionless | `object` | 19 | Corrosion inhibitor chemical species (e.g., BTA). |
| `inhibitor_concentration` | numeric | mM or wt% | `float64` | 19 | Concentration of the corrosion inhibitor. |
| `inhibitor_concentration_unit` | categorical | unit string | `object` | 19 | Concentration unit for inhibitor_concentration. |
| `organic_solvent_type` | categorical | dimensionless | `object` | 2 | Co-solvent organic species in the Solution (e.g., ethylene glycol, IPA). |
| `organic_solvent_concentration` | numeric | wt% or mM | `float64` | 2 | Concentration of the organic co-solvent. |
| `supporting_salt_type` | categorical | dimensionless | `object` | 62 | Supporting ionic salt species in the Solution (e.g., NaCl, Na2SO4, KNO3, KCl). |
| `supporting_salt_concentration` | numeric | M or wt% | `float64` | 62 | Concentration of the supporting ionic salt. |
| `supporting_salt_concentration_unit` | categorical | unit string | `object` | 62 | Concentration unit for supporting_salt_concentration. |
| `dissolved_oxygen_ppm` | numeric | ppm | `float64` | 0 | Dissolved oxygen concentration in parts per million. |
| `conductivity_mS_cm` | numeric | mS/cm | `float64` | 0 | Ionic electrical conductivity of the Solution in millisiemens per centimeter. |
| `ionic_strength_mol_L` | numeric | mol/L | `float64` | 0 | Total molar ionic strength of the Solution. |
| `ORP_mV` | numeric | mV | `float64` | 13 | Solution Oxidation-Reduction Potential in millivolts. |
| `deaerated` | boolean_flag | boolean | `object` | 143 | True if the Solution was explicitly purged of dissolved oxygen prior to measurement. |
| `source` | text | DOI / ID | `object` | 143 | Aggregated publication DOI and table row provenance identifiers. |
| `material_name` | text | dimensionless | `object` | 160 | Full English name of the Metal. |
| `symbol` | categorical | dimensionless | `object` | 160 | Chemical symbol of the Metal. |
| `category` | categorical | dimensionless | `object` | 160 | Semiconductor film functional category (metal, barrier, liner). |
| `density_g_cm3` | numeric | g/cm³ | `float64` | 160 | Bulk mass density of the Metal film. |
| `molar_mass_g_mol` | numeric | g/mol | `float64` | 160 | Atomic or molar mass of the Metal. |
| `valence` | numeric | e⁻/atom | `int64` | 160 | Dissolution Valence (z): electrons transferred per dissolved Metal atom in Faraday's law. |
| `equivalent_weight_g_eq` | numeric | g/eq | `float64` | 160 | Electrochemical equivalent weight (molar_mass_g_mol / valence). |
| `standard_potential_V_SHE` | numeric | V vs. SHE | `float64` | 160 | Standard reduction potential E° of the Metal in V vs. SHE. |
| `melting_point_C` | numeric | °C | `float64` | 160 | Melting temperature of the Metal at 1 atm in degrees Celsius. |
| `electronegativity_pauling` | numeric | Pauling scale | `float64` | 160 | Pauling electronegativity of the Metal. |
| `work_function_eV` | numeric | eV | `float64` | 160 | Electronic work function of the Metal surface in electron-volts. |
| `oxide_type` | text | chemical formula | `object` | 160 | Primary surface oxide formulas formed on the Metal. |
| `oxide_stability` | categorical | tier | `object` | 160 | Qualitative thermodynamic stability tier of the native oxide (low, medium, high). |
| `passivation_tendency` | numeric | index [0, 1] | `float64` | 160 | Dimensionless tendency of the Metal to form a self-limiting passive oxide film. |
| `pourbaix_stable_pH_min` | numeric | pH units | `float64` | 160 | Lower pH boundary of the Metal's passive oxide stability window. |
| `pourbaix_stable_pH_max` | numeric | pH units | `float64` | 160 | Upper pH boundary of the Metal's passive oxide stability window. |
| `crystal_structure` | categorical | lattice type | `object` | 160 | Room-temperature crystal structure (FCC, BCC, HCP). |
| `thermal_conductivity_W_mK` | numeric | W/(m·K) | `float64` | 160 | Bulk thermal conductivity of the Metal. |
| `electrical_resistivity_ohm_cm` | numeric | Ω·cm | `float64` | 160 | Bulk electrical resistivity of the Metal. |
| `common_corrosion_products` | text | chemical species | `object` | 160 | Typical dissolved ionic or solid corrosion products. |
| `base_element` | categorical | dimensionless | `object` | 147 | Primary metallic element symbol for elemental descriptor lookup. |
| `pH_effective` | numeric | pH units | `float64` | 147 | Effective pH used for thermodynamic calculations (measured or inferred from acid/base molarity). |
| `pH_source` | categorical | dimensionless | `object` | 147 | Provenance of pH_effective (measured_echem, inferred_strong_acid, inferred_neutral, etc.). |
| `elem_atomic_number` | numeric | atomic number Z | `float64` | 147 | Atomic number (proton count) of the Metal element. |
| `elem_atomic_mass` | numeric | g/mol | `float64` | 147 | Standard atomic weight of the Metal element. |
| `elem_pauling_en` | numeric | Pauling scale | `float64` | 147 | Elemental Pauling electronegativity. |
| `elem_work_function_eV` | numeric | eV | `float64` | 147 | Polycrystalline elemental electronic work function in electron-volts. |
| `elem_standard_potential_V_SHE` | numeric | V vs. SHE | `float64` | 147 | Elemental standard reduction potential in V vs. SHE. |
| `elem_melting_point_K` | numeric | K | `float64` | 147 | Elemental melting temperature in Kelvin. |
| `elem_molar_volume_cm3` | numeric | cm³/mol | `float64` | 147 | Elemental molar volume in cubic centimeters per mole. |
| `mp_material_id` | identifier | Materials Project ID | `object` | 147 | Materials Project crystal structure identifier. |
| `mp_work_function_eV` | numeric | eV | `float64` | 147 | DFT-computed surface work function from Materials Project in electron-volts. |
| `mp_surface_energy_J_m2` | numeric | J/m² | `float64` | 147 | DFT-computed cleavage/surface energy in Joules per square meter. |
| `mp_formation_energy_eV_atom` | numeric | eV/atom | `float64` | 147 | DFT-computed formation enthalpy per atom in electron-volts. |
| `mp_band_gap_eV` | numeric | eV | `float64` | 147 | Electronic band gap energy in electron-volts (0.0 eV for metallic conductors). |
| `thermo_E_onset_V_SHE` | numeric | V vs. SHE | `float64` | 147 | Thermodynamic Pourbaix anodic dissolution or oxidation onset potential in V vs. SHE. |
| `thermo_corrosion_regime` | categorical | Pourbaix regime | `object` | 145 | Thermodynamic Pourbaix stability regime (immunity, corrosion, passivation). |
| `thermo_stable_phase` | text | chemical phase | `object` | 145 | Thermodynamically favored aqueous ion or solid oxide phase at the Exposure Condition. |
| `thermo_source` | categorical | dimensionless | `object` | 147 | Provenance of the Pourbaix thermodynamic assessment (e.g., mp_pourbaix). |
| `mp_pourbaix_E_onset_V_SHE` | numeric | V vs. SHE | `float64` | 147 | Materials Project Pourbaix oxidation onset potential at pH_effective in V vs. SHE. |
| `mp_stable_phase_at_Ecorr` | text | chemical phase | `object` | 145 | Materials Project Pourbaix stable phase evaluated at (pH_effective, Ecorr_V_SHE). |
| `mp_corrosion_regime` | categorical | Pourbaix regime | `object` | 145 | Materials Project Pourbaix stability classification at Ecorr_V_SHE (immunity, corrosion, passivation). |
| `mp_delta_G_pbx_eV_atom` | numeric | eV/atom | `float64` | 145 | Pourbaix decomposition free energy (driving force above convex hull) at Ecorr_V_SHE in eV/atom. |
| `delta_E_overpotential_V` | numeric | V | `float64` | 145 | Anodic overpotential above thermodynamic oxidation onset (Ecorr_V_SHE - thermo_E_onset_V_SHE) in Volts. |
| `log10_icorr_uA_cm2` | numeric | log10(µA/cm²) | `float64` | 160 | Base-10 logarithm of standardized Corrosion Current Density i_corr in µA/cm². |
| `beta_a_imputed` | boolean_flag | boolean | `bool` | 160 | True if anodic Tafel slope beta_a was missing and imputed via group median or 60 mV/dec fallback. |
| `beta_c_imputed` | boolean_flag | boolean | `bool` | 160 | True if cathodic Tafel slope beta_c was missing and imputed via group median or 120 mV/dec fallback. |
| `is_imputed` | boolean_flag | boolean | `bool` | 160 | Quality flag indicating whether either Tafel slope (beta_a or beta_c) was imputed. |
| `quality_score` | quality_metric | score [0, 1] | `float64` | 160 | Data reliability score in [0, 1], automatically decremented when is_imputed is True. |

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
| `equivalent_weight_g_eq` | numeric | g/eq | `float64` | 8 | Electrochemical equivalent weight (molar_mass_g_mol / valence). |
| `standard_potential_V_SHE` | numeric | V vs. SHE | `float64` | 8 | Standard reduction potential E° of the Metal in V vs. SHE. |
| `melting_point_C` | numeric | °C | `float64` | 8 | Melting temperature of the Metal at 1 atm in degrees Celsius. |
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
| `notes` | text | dimensionless | `float64` | 0 | Curator annotations or supplementary notes. |

## Table: `solution_properties`

| Column | Role | Unit | Type | Non-Null | Description |
| :--- | :--- | :--- | :--- | ---: | :--- |
| `solution_id` | identifier | dimensionless | `object` | 149 | Unique Solution formulation identifier. |
| `solution_name` | text | dimensionless | `object` | 148 | Human-readable chemical formulation summary for the Solution. |
| `solution_category` | categorical | dimensionless | `object` | 148 | pH regime category of the Solution (acidic, neutral, alkaline, etc.). |
| `materials` | text | dimensionless | `object` | 144 | Semicolon-delimited list of Metals evaluated with this Solution. |
| `base_solvent` | categorical | dimensionless | `object` | 148 | Primary liquid solvent (typically DI water). |
| `pH` | numeric | pH units | `float64` | 149 | Hydrogen ion activity (-log10[H+]) of the Solution. |
| `oxidizer_type` | categorical | dimensionless | `object` | 149 | Primary oxidizing agent in the Solution (e.g., H2O2, HNO3, None). |
| `oxidizer_concentration` | numeric | wt% or M | `float64` | 149 | Numerical concentration of the oxidizing agent (0.0 when oxidizer_type is None). |
| `oxidizer_concentration_unit` | categorical | unit string | `object` | 30 | Concentration unit for oxidizer_concentration (e.g., wt%, M, mM). |
| `acid_type` | categorical | dimensionless | `object` | 52 | Acid component in the Solution formulation (e.g., H2SO4, HCl, HF, HClO4, citric acid). |
| `acid_concentration` | numeric | M or wt% | `float64` | 52 | Concentration of the acid component. |
| `acid_concentration_unit` | categorical | unit string | `object` | 52 | Concentration unit for acid_concentration. |
| `base_type` | categorical | dimensionless | `object` | 31 | Alkaline base component in the Solution formulation (e.g., KOH, NaOH, TMAH). |
| `base_concentration` | numeric | M or wt% | `float64` | 31 | Concentration of the alkaline base component. |
| `base_concentration_unit` | categorical | unit string | `object` | 31 | Concentration unit for base_concentration. |
| `HF_concentration_wt_pct` | numeric | wt% | `float64` | 1 | Hydrofluoric acid (HF) weight percent concentration in the Solution. |
| `chelating_agent_type` | categorical | dimensionless | `object` | 21 | Complexing or chelating agent (e.g., citric acid, glycine, oxalic acid, EDTA). |
| `chelating_agent_concentration` | numeric | mM or wt% | `float64` | 21 | Concentration of the complexing/chelating agent. |
| `chelating_agent_concentration_unit` | categorical | unit string | `object` | 21 | Concentration unit for chelating_agent_concentration. |
| `inhibitor_type` | categorical | dimensionless | `object` | 15 | Corrosion inhibitor chemical species (e.g., BTA). |
| `inhibitor_concentration` | numeric | mM or wt% | `float64` | 15 | Concentration of the corrosion inhibitor. |
| `inhibitor_concentration_unit` | categorical | unit string | `object` | 15 | Concentration unit for inhibitor_concentration. |
| `organic_solvent_type` | categorical | dimensionless | `object` | 1 | Co-solvent organic species in the Solution (e.g., ethylene glycol, IPA). |
| `organic_solvent_concentration` | numeric | wt% or mM | `float64` | 1 | Concentration of the organic co-solvent. |
| `supporting_salt_type` | categorical | dimensionless | `object` | 67 | Supporting ionic salt species in the Solution (e.g., NaCl, Na2SO4, KNO3, KCl). |
| `supporting_salt_concentration` | numeric | M or wt% | `float64` | 67 | Concentration of the supporting ionic salt. |
| `supporting_salt_concentration_unit` | categorical | unit string | `object` | 67 | Concentration unit for supporting_salt_concentration. |
| `dissolved_oxygen_ppm` | numeric | ppm | `float64` | 0 | Dissolved oxygen concentration in parts per million. |
| `conductivity_mS_cm` | numeric | mS/cm | `float64` | 0 | Ionic electrical conductivity of the Solution in millisiemens per centimeter. |
| `ionic_strength_mol_L` | numeric | mol/L | `float64` | 0 | Total molar ionic strength of the Solution. |
| `ORP_mV` | numeric | mV | `float64` | 4 | Solution Oxidation-Reduction Potential in millivolts. |
| `deaerated` | boolean_flag | boolean | `object` | 144 | True if the Solution was explicitly purged of dissolved oxygen prior to measurement. |
| `source` | text | DOI / ID | `object` | 144 | Aggregated publication DOI and table row provenance identifiers. |
| `notes` | text | dimensionless | `float64` | 0 | Curator annotations or supplementary notes. |

## Table: `galvanic_series`

| Column | Role | Unit | Type | Non-Null | Description |
| :--- | :--- | :--- | :--- | ---: | :--- |
| `solution_id` | identifier | dimensionless | `object` | 51 | Unique Solution formulation identifier. |
| `metal` | categorical | dimensionless | `object` | 51 | Conductive metallic element or alloy film (Cu, Co, W, Ru, Ti, Ta, Al, Mo). |
| `solution_name` | text | dimensionless | `object` | 50 | Human-readable chemical formulation summary for the Solution. |
| `pH` | numeric | pH units | `float64` | 51 | Hydrogen ion activity (-log10[H+]) of the Solution. |
| `oxidizer_type` | categorical | dimensionless | `object` | 51 | Primary oxidizing agent in the Solution (e.g., H2O2, HNO3, None). |
| `oxidizer_concentration` | numeric | wt% or M | `float64` | 51 | Numerical concentration of the oxidizing agent (0.0 when oxidizer_type is None). |
| `temperature_C` | numeric | °C | `float64` | 51 | Operating temperature of the Exposure Condition in degrees Celsius. |
| `Ecorr_V_SHE` | numeric | V vs. SHE | `float64` | 51 | Standardized Corrosion Potential (E_corr) converted to Volts vs. Standard Hydrogen Electrode. |
| `icorr_uA_cm2` | numeric | µA/cm² | `float64` | 51 | Standardized Corrosion Current Density (i_corr) at E_corr in microamperes per square centimeter. |
| `beta_a_mV_dec` | numeric | mV/dec | `float64` | 51 | Strictly positive anodic Tafel Slope (beta_a) in millivolts per decade of current density. |
| `beta_c_mV_dec` | numeric | mV/dec | `float64` | 51 | Strictly positive cathodic Tafel Slope (beta_c) in millivolts per decade of current density. |
| `record_count` | numeric | count | `int64` | 51 | Number of harmonized experimental records aggregated for that (Solution, Metal) pair. |
| `galvanic_rank` | numeric | integer rank (1..N) | `int64` | 51 | Monotonic Galvanic Series rank within a Solution (1 = most Active Metal, N = most Noble Metal). |
| `metals_in_solution` | numeric | count | `int64` | 51 | Total distinct Metals evaluated in that Solution within the Galvanic Series. |
| `relative_nobility` | categorical | role label | `object` | 51 | Qualitative galvanic role within the Solution (Most Active, Intermediate, Most Noble). |
