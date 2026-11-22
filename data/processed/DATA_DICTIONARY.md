# Harmonized Electrochemical Database — Data Dictionary

All tables use standardized physical units: Corrosion Potentials ($E_{\text{corr}}$) in **V vs. SHE**, Corrosion Current Densities ($i_{\text{corr}}$) in **µA/cm²**, and Tafel Slopes ($\beta_a, \beta_c$) in strictly positive **mV/dec**.

## Table: `electrochemical_database`

| Column | Role | Unit | Type | Non-Null | Description |
| :--- | :--- | :--- | :--- | ---: | :--- |
| `record_id` | identifier | dimensionless | `object` | 948 | Unique electrochemical measurement record identifier. |
| `material_id` | identifier | dimensionless | `object` | 948 | Canonical Metal identifier matching symbol. |
| `solution_id` | identifier | dimensionless | `object` | 948 | Unique Solution formulation identifier. |
| `metal` | categorical | dimensionless | `object` | 948 | Conductive metallic element or alloy film (Cu, Co, W, Ru, Ti, Ta, Al, Mo). |
| `sample_name` | text | dimensionless | `object` | 711 | Original experimental sample or specimen designation reported in the source publication. |
| `metal_form` | categorical | dimensionless | `object` | 843 | Physical morphology or deposition form of the Metal (thin film, PVD, ALD, bulk, foil, rod). |
| `surface_preparation` | text | dimensionless | `object` | 802 | Pre-exposure surface conditioning or polishing treatment (e.g., CMP, mechanical polish, annealed). |
| `solution_description` | text | dimensionless | `object` | 798 | Raw text description of the Solution composition from the literature source. |
| `pH` | numeric | pH units | `float64` | 948 | Hydrogen ion activity (-log10[H+]) of the Solution. |
| `temperature_C` | numeric | °C | `float64` | 948 | Operating temperature of the Exposure Condition in degrees Celsius. |
| `reference_electrode` | categorical | dimensionless | `object` | 948 | Original experimental reference electrode before conversion to SHE. |
| `Ecorr_V_SHE` | numeric | V vs. SHE | `float64` | 948 | Standardized Corrosion Potential (E_corr) converted to Volts vs. Standard Hydrogen Electrode. |
| `Ecorr_V_original` | numeric | V vs. ref | `float64` | 948 | Originally reported open-circuit Corrosion Potential prior to SHE conversion. |
| `icorr_uA_cm2` | numeric | µA/cm² | `float64` | 948 | Standardized Corrosion Current Density (i_corr) at E_corr in microamperes per square centimeter. |
| `log_icorr` | numeric | log10(A/cm²) | `float64` | 948 | Raw base-10 logarithm of corrosion current density in A/cm² from literature compilation. |
| `beta_a_mV_dec` | numeric | mV/dec | `float64` | 948 | Strictly positive anodic Tafel Slope (beta_a) in millivolts per decade of current density. |
| `beta_c_mV_dec` | numeric | mV/dec | `float64` | 948 | Strictly positive cathodic Tafel Slope (beta_c) in millivolts per decade of current density. |
| `tafel_slopes_general` | text | mV/dec | `object` | 7 | Unassigned or general Tafel slope values extracted from literature text. |
| `Rp_ohm_cm2` | numeric | Ω·cm² | `float64` | 253 | Linear polarization resistance (R_p) in ohm-square centimeters. |
| `Rp_unit` | categorical | unit string | `object` | 253 | Reported physical unit for linear polarization resistance R_p. |
| `CR_reported_value` | numeric | reported unit | `float64` | 183 | Corrosion Rate value as originally reported in the source publication. |
| `CR_reported_unit` | categorical | unit string | `object` | 183 | Physical unit for CR_reported_value (e.g., nm/min, mm/year, Å/min). |
| `CR_calculated_nm_min` | numeric | nm/min | `float64` | 948 | Faraday-derived Corrosion Rate in nanometers per minute. |
| `measurement_method` | text | dimensionless | `object` | 843 | Experimental electrochemical or analytical measurement techniques (potentiodynamic, EIS, LPR, ICP-MS). |
| `scan_rate` | text | mV/s | `object` | 461 | Potentiodynamic polarization potential sweep rate. |
| `immersion_time` | text | time unit | `object` | 505 | Experimental immersion or pre-conditioning duration reported in the study. |
| `aeration_condition` | categorical | dimensionless | `object` | 151 | Dissolved gas saturation state during testing (aerated, deaerated, N2/Ar purged). |
| `reaction_type` | categorical | dimensionless | `object` | 948 | Primary electrochemical process classification (corrosion, passivity, etching). |
| `reactions` | text | dimensionless | `object` | 948 | Half-cell reactions active under the Exposure Condition (HER, ORR, OER, metal dissolution). |
| `source_type` | categorical | dimensionless | `object` | 948 | Literature or benchmark source classification (journal, seed_benchmark). |
| `source_reference` | text | DOI / ID | `object` | 948 | Publication DOI, table row identifier, or benchmark reference string. |
| `data_extraction_method` | categorical | dimensionless | `object` | 948 | Ingestion extraction pipeline method (table_column_mapped, regex, manual_curation). |
| `data_quality_score` | quality_metric | score [0, 1] | `float64` | 948 | Raw ingestion data quality score prior to imputation penalty. |
| `quality_notes` | text | dimensionless | `object` | 948 | Automated ingestion audit notes regarding unit inference or missing metadata. |
| `notes` | text | dimensionless | `float64` | 0 | Curator annotations or supplementary notes. |
| `solution_name` | text | dimensionless | `object` | 840 | Human-readable chemical formulation summary for the Solution. |
| `solution_category` | categorical | dimensionless | `object` | 840 | pH regime category of the Solution (acidic, neutral, alkaline, etc.). |
| `materials` | text | dimensionless | `object` | 840 | Semicolon-delimited list of Metals evaluated with this Solution. |
| `base_solvent` | categorical | dimensionless | `object` | 840 | Primary liquid solvent (typically DI water). |
| `oxidizer_type` | categorical | dimensionless | `object` | 948 | Primary oxidizing agent in the Solution (e.g., H2O2, HNO3, None). |
| `oxidizer_concentration` | numeric | wt% or M | `float64` | 948 | Numerical concentration of the oxidizing agent (0.0 when oxidizer_type is None). |
| `oxidizer_concentration_unit` | categorical | unit string | `object` | 127 | Concentration unit for oxidizer_concentration (e.g., wt%, M, mM). |
| `acid_type` | categorical | dimensionless | `object` | 337 | Acid component in the Solution formulation (e.g., H2SO4, HCl, HF, HClO4, citric acid). |
| `acid_concentration` | numeric | M or wt% | `float64` | 337 | Concentration of the acid component. |
| `acid_concentration_unit` | categorical | unit string | `object` | 337 | Concentration unit for acid_concentration. |
| `base_type` | categorical | dimensionless | `object` | 85 | Alkaline base component in the Solution formulation (e.g., KOH, NaOH, TMAH). |
| `base_concentration` | numeric | M or wt% | `float64` | 85 | Concentration of the alkaline base component. |
| `base_concentration_unit` | categorical | unit string | `object` | 85 | Concentration unit for base_concentration. |
| `HF_concentration_wt_pct` | numeric | wt% | `float64` | 15 | Hydrofluoric acid (HF) weight percent concentration in the Solution. |
| `chelating_agent_type` | categorical | dimensionless | `object` | 36 | Complexing or chelating agent (e.g., citric acid, glycine, oxalic acid, EDTA). |
| `chelating_agent_concentration` | numeric | mM or wt% | `float64` | 36 | Concentration of the complexing/chelating agent. |
| `chelating_agent_concentration_unit` | categorical | unit string | `object` | 36 | Concentration unit for chelating_agent_concentration. |
| `inhibitor_type` | categorical | dimensionless | `object` | 11 | Corrosion inhibitor chemical species (e.g., BTA). |
| `inhibitor_concentration` | numeric | mM or wt% | `float64` | 11 | Concentration of the corrosion inhibitor. |
| `inhibitor_concentration_unit` | categorical | unit string | `object` | 11 | Concentration unit for inhibitor_concentration. |
| `organic_solvent_type` | categorical | dimensionless | `object` | 5 | Co-solvent organic species in the Solution (e.g., ethylene glycol, IPA). |
| `organic_solvent_concentration` | numeric | wt% or mM | `float64` | 5 | Concentration of the organic co-solvent. |
| `supporting_salt_type` | categorical | dimensionless | `object` | 568 | Supporting ionic salt species in the Solution (e.g., NaCl, Na2SO4, KNO3, KCl). |
| `supporting_salt_concentration` | numeric | M or wt% | `float64` | 568 | Concentration of the supporting ionic salt. |
| `supporting_salt_concentration_unit` | categorical | unit string | `object` | 568 | Concentration unit for supporting_salt_concentration. |
| `dissolved_oxygen_ppm` | numeric | ppm | `float64` | 0 | Dissolved oxygen concentration in parts per million. |
| `conductivity_mS_cm` | numeric | mS/cm | `float64` | 0 | Ionic electrical conductivity of the Solution in millisiemens per centimeter. |
| `ionic_strength_mol_L` | numeric | mol/L | `float64` | 0 | Total molar ionic strength of the Solution. |
| `ORP_mV` | numeric | mV | `float64` | 0 | Solution Oxidation-Reduction Potential in millivolts. |
| `deaerated` | boolean_flag | boolean | `object` | 840 | True if the Solution was explicitly purged of dissolved oxygen prior to measurement. |
| `source` | text | DOI / ID | `object` | 840 | Aggregated publication DOI and table row provenance identifiers. |
| `material_name` | text | dimensionless | `object` | 948 | Full English name of the Metal. |
| `symbol` | categorical | dimensionless | `object` | 948 | Chemical symbol of the Metal. |
| `category` | categorical | dimensionless | `object` | 948 | Semiconductor film functional category (metal, barrier, liner). |
| `density_g_cm3` | numeric | g/cm³ | `float64` | 948 | Bulk mass density of the Metal film. |
| `molar_mass_g_mol` | numeric | g/mol | `float64` | 948 | Atomic or molar mass of the Metal. |
| `valence` | numeric | e⁻/atom | `int64` | 948 | Dissolution Valence (z): electrons transferred per dissolved Metal atom in Faraday's law. |
| `equivalent_weight_g_eq` | numeric | g/eq | `float64` | 948 | Electrochemical equivalent weight (molar_mass_g_mol / valence). |
| `standard_potential_V_SHE` | numeric | V vs. SHE | `float64` | 948 | Standard reduction potential E° of the Metal in V vs. SHE. |
| `melting_point_C` | numeric | °C | `float64` | 948 | Melting temperature of the Metal at 1 atm in degrees Celsius. |
| `electronegativity_pauling` | numeric | Pauling scale | `float64` | 948 | Pauling electronegativity of the Metal. |
| `work_function_eV` | numeric | eV | `float64` | 948 | Electronic work function of the Metal surface in electron-volts. |
| `oxide_type` | text | chemical formula | `object` | 948 | Primary surface oxide formulas formed on the Metal. |
| `oxide_stability` | categorical | tier | `object` | 948 | Qualitative thermodynamic stability tier of the native oxide (low, medium, high). |
| `passivation_tendency` | numeric | index [0, 1] | `float64` | 948 | Dimensionless tendency of the Metal to form a self-limiting passive oxide film. |
| `pourbaix_stable_pH_min` | numeric | pH units | `float64` | 948 | Lower pH boundary of the Metal's passive oxide stability window. |
| `pourbaix_stable_pH_max` | numeric | pH units | `float64` | 948 | Upper pH boundary of the Metal's passive oxide stability window. |
| `crystal_structure` | categorical | lattice type | `object` | 948 | Room-temperature crystal structure (FCC, BCC, HCP). |
| `thermal_conductivity_W_mK` | numeric | W/(m·K) | `float64` | 948 | Bulk thermal conductivity of the Metal. |
| `electrical_resistivity_ohm_cm` | numeric | Ω·cm | `float64` | 948 | Bulk electrical resistivity of the Metal. |
| `common_corrosion_products` | text | chemical species | `object` | 948 | Typical dissolved ionic or solid corrosion products. |
| `base_element` | categorical | dimensionless | `object` | 948 | Primary metallic element symbol for elemental descriptor lookup. |
| `pH_effective` | numeric | pH units | `float64` | 948 | Effective pH used for thermodynamic calculations (measured or inferred from acid/base molarity). |
| `pH_source` | categorical | dimensionless | `object` | 948 | Provenance of pH_effective (measured_echem, inferred_strong_acid, inferred_neutral, etc.). |
| `elem_atomic_number` | numeric | atomic number Z | `float64` | 948 | Atomic number (proton count) of the Metal element. |
| `elem_atomic_mass` | numeric | g/mol | `float64` | 948 | Standard atomic weight of the Metal element. |
| `elem_pauling_en` | numeric | Pauling scale | `float64` | 948 | Elemental Pauling electronegativity. |
| `elem_work_function_eV` | numeric | eV | `float64` | 948 | Polycrystalline elemental electronic work function in electron-volts. |
| `elem_standard_potential_V_SHE` | numeric | V vs. SHE | `float64` | 948 | Elemental standard reduction potential in V vs. SHE. |
| `elem_melting_point_K` | numeric | K | `float64` | 948 | Elemental melting temperature in Kelvin. |
| `elem_molar_volume_cm3` | numeric | cm³/mol | `float64` | 948 | Elemental molar volume in cubic centimeters per mole. |
| `mp_material_id` | identifier | Materials Project ID | `object` | 948 | Materials Project crystal structure identifier. |
| `mp_work_function_eV` | numeric | eV | `float64` | 941 | DFT-computed surface work function from Materials Project in electron-volts. |
| `mp_surface_energy_J_m2` | numeric | J/m² | `float64` | 941 | DFT-computed cleavage/surface energy in Joules per square meter. |
| `mp_formation_energy_eV_atom` | numeric | eV/atom | `float64` | 948 | DFT-computed formation enthalpy per atom in electron-volts. |
| `mp_band_gap_eV` | numeric | eV | `float64` | 948 | Electronic band gap energy in electron-volts (0.0 eV for metallic conductors). |
| `thermo_E_onset_V_SHE` | numeric | V vs. SHE | `float64` | 948 | Thermodynamic Pourbaix anodic dissolution or oxidation onset potential in V vs. SHE. |
| `thermo_corrosion_regime` | categorical | Pourbaix regime | `object` | 945 | Thermodynamic Pourbaix stability regime (immunity, corrosion, passivation). |
| `thermo_stable_phase` | text | chemical phase | `object` | 945 | Thermodynamically favored aqueous ion or solid oxide phase at the Exposure Condition. |
| `thermo_source` | categorical | dimensionless | `object` | 948 | Provenance of the Pourbaix thermodynamic assessment (e.g., mp_pourbaix). |
| `mp_pourbaix_E_onset_V_SHE` | numeric | V vs. SHE | `float64` | 948 | Materials Project Pourbaix oxidation onset potential at pH_effective in V vs. SHE. |
| `mp_stable_phase_at_Ecorr` | text | chemical phase | `object` | 945 | Materials Project Pourbaix stable phase evaluated at (pH_effective, Ecorr_V_SHE). |
| `mp_corrosion_regime` | categorical | Pourbaix regime | `object` | 945 | Materials Project Pourbaix stability classification at Ecorr_V_SHE (immunity, corrosion, passivation). |
| `mp_delta_G_pbx_eV_atom` | numeric | eV/atom | `float64` | 945 | Pourbaix decomposition free energy (driving force above convex hull) at Ecorr_V_SHE in eV/atom. |
| `delta_E_overpotential_V` | numeric | V | `float64` | 945 | Anodic overpotential above thermodynamic oxidation onset (Ecorr_V_SHE - thermo_E_onset_V_SHE) in Volts. |
| `log10_icorr_uA_cm2` | numeric | log10(µA/cm²) | `float64` | 948 | Base-10 logarithm of standardized Corrosion Current Density i_corr in µA/cm². |
| `beta_a_imputed` | boolean_flag | boolean | `bool` | 948 | True if anodic Tafel slope beta_a was missing and imputed via group median or 60 mV/dec fallback. |
| `beta_c_imputed` | boolean_flag | boolean | `bool` | 948 | True if cathodic Tafel slope beta_c was missing and imputed via group median or 120 mV/dec fallback. |
| `is_imputed` | boolean_flag | boolean | `bool` | 948 | Quality flag indicating whether either Tafel slope (beta_a or beta_c) was imputed. |
| `quality_score` | quality_metric | score [0, 1] | `float64` | 948 | Data reliability score in [0, 1], automatically decremented when is_imputed is True. |

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
| `solution_id` | identifier | dimensionless | `object` | 528 | Unique Solution formulation identifier. |
| `solution_name` | text | dimensionless | `object` | 527 | Human-readable chemical formulation summary for the Solution. |
| `solution_category` | categorical | dimensionless | `object` | 527 | pH regime category of the Solution (acidic, neutral, alkaline, etc.). |
| `materials` | text | dimensionless | `object` | 527 | Semicolon-delimited list of Metals evaluated with this Solution. |
| `base_solvent` | categorical | dimensionless | `object` | 527 | Primary liquid solvent (typically DI water). |
| `pH` | numeric | pH units | `float64` | 528 | Hydrogen ion activity (-log10[H+]) of the Solution. |
| `oxidizer_type` | categorical | dimensionless | `object` | 528 | Primary oxidizing agent in the Solution (e.g., H2O2, HNO3, None). |
| `oxidizer_concentration` | numeric | wt% or M | `float64` | 528 | Numerical concentration of the oxidizing agent (0.0 when oxidizer_type is None). |
| `oxidizer_concentration_unit` | categorical | unit string | `object` | 86 | Concentration unit for oxidizer_concentration (e.g., wt%, M, mM). |
| `acid_type` | categorical | dimensionless | `object` | 189 | Acid component in the Solution formulation (e.g., H2SO4, HCl, HF, HClO4, citric acid). |
| `acid_concentration` | numeric | M or wt% | `float64` | 189 | Concentration of the acid component. |
| `acid_concentration_unit` | categorical | unit string | `object` | 189 | Concentration unit for acid_concentration. |
| `base_type` | categorical | dimensionless | `object` | 112 | Alkaline base component in the Solution formulation (e.g., KOH, NaOH, TMAH). |
| `base_concentration` | numeric | M or wt% | `float64` | 112 | Concentration of the alkaline base component. |
| `base_concentration_unit` | categorical | unit string | `object` | 112 | Concentration unit for base_concentration. |
| `HF_concentration_wt_pct` | numeric | wt% | `float64` | 6 | Hydrofluoric acid (HF) weight percent concentration in the Solution. |
| `chelating_agent_type` | categorical | dimensionless | `object` | 37 | Complexing or chelating agent (e.g., citric acid, glycine, oxalic acid, EDTA). |
| `chelating_agent_concentration` | numeric | mM or wt% | `float64` | 37 | Concentration of the complexing/chelating agent. |
| `chelating_agent_concentration_unit` | categorical | unit string | `object` | 37 | Concentration unit for chelating_agent_concentration. |
| `inhibitor_type` | categorical | dimensionless | `object` | 22 | Corrosion inhibitor chemical species (e.g., BTA). |
| `inhibitor_concentration` | numeric | mM or wt% | `float64` | 22 | Concentration of the corrosion inhibitor. |
| `inhibitor_concentration_unit` | categorical | unit string | `object` | 22 | Concentration unit for inhibitor_concentration. |
| `organic_solvent_type` | categorical | dimensionless | `object` | 2 | Co-solvent organic species in the Solution (e.g., ethylene glycol, IPA). |
| `organic_solvent_concentration` | numeric | wt% or mM | `float64` | 2 | Concentration of the organic co-solvent. |
| `supporting_salt_type` | categorical | dimensionless | `object` | 285 | Supporting ionic salt species in the Solution (e.g., NaCl, Na2SO4, KNO3, KCl). |
| `supporting_salt_concentration` | numeric | M or wt% | `float64` | 285 | Concentration of the supporting ionic salt. |
| `supporting_salt_concentration_unit` | categorical | unit string | `object` | 285 | Concentration unit for supporting_salt_concentration. |
| `dissolved_oxygen_ppm` | numeric | ppm | `float64` | 0 | Dissolved oxygen concentration in parts per million. |
| `conductivity_mS_cm` | numeric | mS/cm | `float64` | 0 | Ionic electrical conductivity of the Solution in millisiemens per centimeter. |
| `ionic_strength_mol_L` | numeric | mol/L | `float64` | 0 | Total molar ionic strength of the Solution. |
| `ORP_mV` | numeric | mV | `float64` | 0 | Solution Oxidation-Reduction Potential in millivolts. |
| `deaerated` | boolean_flag | boolean | `object` | 527 | True if the Solution was explicitly purged of dissolved oxygen prior to measurement. |
| `source` | text | DOI / ID | `object` | 527 | Aggregated publication DOI and table row provenance identifiers. |
| `notes` | text | dimensionless | `float64` | 0 | Curator annotations or supplementary notes. |

## Table: `galvanic_series`

| Column | Role | Unit | Type | Non-Null | Description |
| :--- | :--- | :--- | :--- | ---: | :--- |
| `solution_id` | identifier | dimensionless | `object` | 202 | Unique Solution formulation identifier. |
| `metal` | categorical | dimensionless | `object` | 202 | Conductive metallic element or alloy film (Cu, Co, W, Ru, Ti, Ta, Al, Mo). |
| `solution_name` | text | dimensionless | `object` | 195 | Human-readable chemical formulation summary for the Solution. |
| `pH` | numeric | pH units | `float64` | 202 | Hydrogen ion activity (-log10[H+]) of the Solution. |
| `oxidizer_type` | categorical | dimensionless | `object` | 202 | Primary oxidizing agent in the Solution (e.g., H2O2, HNO3, None). |
| `oxidizer_concentration` | numeric | wt% or M | `float64` | 202 | Numerical concentration of the oxidizing agent (0.0 when oxidizer_type is None). |
| `temperature_C` | numeric | °C | `float64` | 202 | Operating temperature of the Exposure Condition in degrees Celsius. |
| `Ecorr_V_SHE` | numeric | V vs. SHE | `float64` | 202 | Standardized Corrosion Potential (E_corr) converted to Volts vs. Standard Hydrogen Electrode. |
| `icorr_uA_cm2` | numeric | µA/cm² | `float64` | 202 | Standardized Corrosion Current Density (i_corr) at E_corr in microamperes per square centimeter. |
| `beta_a_mV_dec` | numeric | mV/dec | `float64` | 202 | Strictly positive anodic Tafel Slope (beta_a) in millivolts per decade of current density. |
| `beta_c_mV_dec` | numeric | mV/dec | `float64` | 202 | Strictly positive cathodic Tafel Slope (beta_c) in millivolts per decade of current density. |
| `record_count` | numeric | count | `int64` | 202 | Number of harmonized experimental records aggregated for that (Solution, Metal) pair. |
| `galvanic_rank` | numeric | integer rank (1..N) | `int64` | 202 | Monotonic Galvanic Series rank within a Solution (1 = most Active Metal, N = most Noble Metal). |
| `metals_in_solution` | numeric | count | `int64` | 202 | Total distinct Metals evaluated in that Solution within the Galvanic Series. |
| `relative_nobility` | categorical | role label | `object` | 202 | Qualitative galvanic role within the Solution (Most Active, Intermediate, Most Noble). |
