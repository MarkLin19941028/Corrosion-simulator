# Harmonized Electrochemical Database — Data Dictionary

All tables use standardized physical units: Corrosion Potentials ($E_{\text{corr}}$) in **V vs. SHE**, Corrosion Current Densities ($i_{\text{corr}}$) in **µA/cm²**, and Tafel Slopes ($\beta_a, \beta_c$) in strictly positive **mV/dec**.

## Table: `electrochemical_database`

| Column | Role | Unit | Type | Non-Null | Description |
| :--- | :--- | :--- | :--- | ---: | :--- |
| `record_id` | identifier | dimensionless | `object` | 3182 | Unique electrochemical measurement record identifier. |
| `material_id` | identifier | dimensionless | `object` | 3182 | Canonical Metal identifier matching symbol. |
| `solution_id` | identifier | dimensionless | `object` | 3182 | Unique Solution formulation identifier. |
| `metal` | categorical | dimensionless | `object` | 3182 | Conductive metallic element, compound, or alloy film identifier (e.g., Cu, Co, W, Ru, Ti, Ta, Al, Mo, TiN, Fe, Ni). |
| `material_class` | categorical | dimensionless | `object` | 3182 | High-level material classification (e.g., pure_metal, alloy, nitride, barrier). |
| `sample_name` | text | dimensionless | `object` | 2621 | Original experimental sample or specimen designation reported in the source publication. |
| `metal_form` | categorical | dimensionless | `object` | 2629 | Physical morphology or deposition form of the Metal (thin film, PVD, ALD, bulk, foil, rod). |
| `surface_preparation` | text | dimensionless | `object` | 2411 | Pre-exposure surface conditioning or polishing treatment (e.g., CMP, mechanical polish, annealed). |
| `solution_description` | text | dimensionless | `object` | 2715 | Raw text description of the Solution composition from the literature source. |
| `pH` | numeric | pH units | `float64` | 3182 | Hydrogen ion activity (-log10[H+]) of the Solution. |
| `temperature_C` | numeric | °C | `float64` | 3182 | Operating temperature of the Exposure Condition in degrees Celsius. |
| `reference_electrode` | categorical | dimensionless | `object` | 3182 | Original experimental reference electrode before conversion to SHE. |
| `Ecorr_V_SHE` | numeric | V vs. SHE | `float64` | 3182 | Standardized Corrosion Potential (E_corr) converted to Volts vs. Standard Hydrogen Electrode. |
| `Ecorr_V_original` | numeric | V vs. ref | `float64` | 3182 | Originally reported open-circuit Corrosion Potential prior to SHE conversion. |
| `icorr_uA_cm2` | numeric | µA/cm² | `float64` | 3182 | Standardized Corrosion Current Density (i_corr) at E_corr in microamperes per square centimeter. |
| `log_icorr` | numeric | log10(A/cm²) | `float64` | 2934 | Raw base-10 logarithm of corrosion current density in A/cm² from literature compilation. |
| `beta_a_mV_dec` | numeric | mV/dec | `float64` | 3182 | Strictly positive anodic Tafel Slope (beta_a) in millivolts per decade of current density. |
| `beta_c_mV_dec` | numeric | mV/dec | `float64` | 3182 | Strictly positive cathodic Tafel Slope (beta_c) in millivolts per decade of current density. |
| `tafel_slopes_general` | text | mV/dec | `object` | 32 | Unassigned or general Tafel slope values extracted from literature text. |
| `Rp_ohm_cm2` | numeric | Ω·cm² | `float64` | 816 | Linear polarization resistance (R_p) in ohm-square centimeters. |
| `Rp_unit` | categorical | unit string | `object` | 816 | Reported physical unit for linear polarization resistance R_p. |
| `CR_reported_value` | numeric | reported unit | `float64` | 780 | Corrosion Rate value as originally reported in the source publication. |
| `CR_reported_unit` | categorical | unit string | `object` | 780 | Physical unit for CR_reported_value (e.g., nm/min, mm/year, Å/min). |
| `CR_calculated_nm_min` | numeric | nm/min | `float64` | 2217 | Faraday-derived Corrosion Rate in nanometers per minute. |
| `measurement_method` | text | dimensionless | `object` | 2632 | Experimental electrochemical or analytical measurement techniques (potentiodynamic, EIS, LPR, ICP-MS). |
| `scan_rate` | text | mV/s | `object` | 1373 | Potentiodynamic polarization potential sweep rate. |
| `immersion_time` | text | time unit | `object` | 1349 | Experimental immersion or pre-conditioning duration reported in the study. |
| `aeration_condition` | categorical | dimensionless | `object` | 609 | Dissolved gas saturation state during testing (aerated, deaerated, N2/Ar purged). |
| `reaction_type` | categorical | dimensionless | `object` | 3182 | Primary electrochemical process classification (corrosion, passivity, etching). |
| `reactions` | text | dimensionless | `object` | 3182 | Half-cell reactions active under the Exposure Condition (HER, ORR, OER, metal dissolution). |
| `source_type` | categorical | dimensionless | `object` | 3182 | Literature or benchmark source classification (journal, seed_benchmark). |
| `source_reference` | text | DOI / ID | `object` | 3182 | Publication DOI, table row identifier, or benchmark reference string. |
| `data_extraction_method` | categorical | dimensionless | `object` | 3182 | Ingestion extraction pipeline method (table_column_mapped, regex, manual_curation). |
| `data_quality_score` | quality_metric | score [0, 1] | `float64` | 3182 | Raw ingestion data quality score prior to imputation penalty. |
| `quality_notes` | text | dimensionless | `object` | 3182 | Automated ingestion audit notes regarding unit inference or missing metadata. |
| `notes` | text | dimensionless | `float64` | 0 | Curator annotations or supplementary notes. |
| `solution_name` | text | dimensionless | `object` | 2888 | Human-readable chemical formulation summary for the Solution. |
| `solution_category` | categorical | dimensionless | `object` | 2888 | pH regime category of the Solution (acidic, neutral, alkaline, etc.). |
| `materials` | text | dimensionless | `object` | 2888 | Semicolon-delimited list of Metals evaluated with this Solution. |
| `base_solvent` | categorical | dimensionless | `object` | 2888 | Primary liquid solvent (typically DI water). |
| `oxidizer_type` | categorical | dimensionless | `object` | 3182 | Primary oxidizing agent in the Solution (e.g., H2O2, HNO3, None). |
| `oxidizer_concentration` | numeric | wt% or M | `float64` | 3182 | Numerical concentration of the oxidizing agent (0.0 when oxidizer_type is None). |
| `oxidizer_concentration_unit` | categorical | unit string | `object` | 285 | Concentration unit for oxidizer_concentration (e.g., wt%, M, mM). |
| `acid_type` | categorical | dimensionless | `object` | 1251 | Acid component in the Solution formulation (e.g., H2SO4, HCl, HF, HClO4, citric acid). |
| `acid_concentration` | numeric | M or wt% | `float64` | 1251 | Concentration of the acid component. |
| `acid_concentration_unit` | categorical | unit string | `object` | 1251 | Concentration unit for acid_concentration. |
| `base_type` | categorical | dimensionless | `object` | 286 | Alkaline base component in the Solution formulation (e.g., KOH, NaOH, TMAH). |
| `base_concentration` | numeric | M or wt% | `float64` | 286 | Concentration of the alkaline base component. |
| `base_concentration_unit` | categorical | unit string | `object` | 286 | Concentration unit for base_concentration. |
| `HF_concentration_wt_pct` | numeric | wt% | `float64` | 50 | Hydrofluoric acid (HF) weight percent concentration in the Solution. |
| `chelating_agent_type` | categorical | dimensionless | `object` | 94 | Complexing or chelating agent (e.g., citric acid, glycine, oxalic acid, EDTA). |
| `chelating_agent_concentration` | numeric | mM or wt% | `float64` | 94 | Concentration of the complexing/chelating agent. |
| `chelating_agent_concentration_unit` | categorical | unit string | `object` | 94 | Concentration unit for chelating_agent_concentration. |
| `inhibitor_type` | categorical | dimensionless | `object` | 204 | Corrosion inhibitor chemical species (e.g., BTA). |
| `inhibitor_concentration` | numeric | mM or wt% | `float64` | 204 | Concentration of the corrosion inhibitor. |
| `inhibitor_concentration_unit` | categorical | unit string | `object` | 204 | Concentration unit for inhibitor_concentration. |
| `organic_solvent_type` | categorical | dimensionless | `object` | 3182 | Co-solvent organic species in the Solution (e.g., ethylene glycol, IPA). |
| `organic_solvent_concentration` | numeric | wt% or mM | `float64` | 6 | Concentration of the organic co-solvent. |
| `supporting_salt_type` | categorical | dimensionless | `object` | 1779 | Supporting ionic salt species in the Solution (e.g., NaCl, Na2SO4, KNO3, KCl). |
| `supporting_salt_concentration` | numeric | M or wt% | `float64` | 1779 | Concentration of the supporting ionic salt. |
| `supporting_salt_concentration_unit` | categorical | unit string | `object` | 1779 | Concentration unit for supporting_salt_concentration. |
| `dissolved_oxygen_ppm` | numeric | ppm | `float64` | 0 | Dissolved oxygen concentration in parts per million. |
| `conductivity_mS_cm` | numeric | mS/cm | `float64` | 0 | Ionic electrical conductivity of the Solution in millisiemens per centimeter. |
| `ionic_strength_mol_L` | numeric | mol/L | `float64` | 0 | Total molar ionic strength of the Solution. |
| `ORP_mV` | numeric | mV | `float64` | 0 | Solution Oxidation-Reduction Potential in millivolts. |
| `deaerated` | boolean_flag | boolean | `object` | 2888 | True if the Solution was explicitly purged of dissolved oxygen prior to measurement. |
| `components_json` | text | json | `object` | 2888 | Structured JSON breakdown of Solution chemical components and concentrations. |
| `source` | text | DOI / ID | `object` | 2888 | Aggregated publication DOI and table row provenance identifiers. |
| `material_name` | text | dimensionless | `object` | 2399 | Full English name of the Metal. |
| `symbol` | categorical | dimensionless | `object` | 2399 | Chemical symbol of the Metal. |
| `category` | categorical | dimensionless | `object` | 2399 | Semiconductor film functional category (metal, barrier, liner, semiconductor). |
| `density_g_cm3` | numeric | g/cm³ | `float64` | 3182 | Bulk mass density of the Metal film. |
| `molar_mass_g_mol` | numeric | g/mol | `float64` | 3182 | Atomic or molar mass of the Metal. |
| `valence` | numeric | e⁻/atom | `float64` | 3182 | Dissolution Valence (z): electrons transferred per dissolved Metal atom in Faraday's law. |
| `equivalent_weight_g_eq` | numeric | g/eq | `float64` | 3182 | Electrochemical equivalent weight (molar_mass_g_mol / valence). |
| `standard_potential_V_SHE` | numeric | V vs. SHE | `float64` | 2399 | Standard reduction potential E° of the Metal in V vs. SHE. |
| `melting_point_C` | numeric | °C | `float64` | 2399 | Melting temperature of the Metal at 1 atm in degrees Celsius. |
| `electronegativity_pauling` | numeric | Pauling scale | `float64` | 2399 | Pauling electronegativity of the Metal. |
| `work_function_eV` | numeric | eV | `float64` | 2399 | Electronic work function of the Metal surface in electron-volts. |
| `oxide_type` | text | chemical formula | `object` | 2399 | Primary surface oxide formulas formed on the Metal. |
| `oxide_stability` | categorical | tier | `object` | 2399 | Qualitative thermodynamic stability tier of the native oxide (low, medium, high). |
| `passivation_tendency` | numeric | index [0, 1] | `float64` | 2399 | Dimensionless tendency of the Metal to form a self-limiting passive oxide film. |
| `pourbaix_stable_pH_min` | numeric | pH units | `float64` | 2399 | Lower pH boundary of the Metal's passive oxide stability window. |
| `pourbaix_stable_pH_max` | numeric | pH units | `float64` | 2399 | Upper pH boundary of the Metal's passive oxide stability window. |
| `crystal_structure` | categorical | lattice type | `object` | 2399 | Room-temperature crystal structure (FCC, BCC, HCP). |
| `thermal_conductivity_W_mK` | numeric | W/(m·K) | `float64` | 2399 | Bulk thermal conductivity of the Metal. |
| `electrical_resistivity_ohm_cm` | numeric | Ω·cm | `float64` | 2399 | Bulk electrical resistivity of the Metal. |
| `common_corrosion_products` | text | chemical species | `object` | 2399 | Typical dissolved ionic or solid corrosion products. |
| `base_element` | categorical | dimensionless | `object` | 3179 | Primary metallic element symbol for elemental descriptor lookup. |
| `pH_effective` | numeric | pH units | `float64` | 3182 | Effective pH used for thermodynamic calculations (measured or inferred from acid/base molarity). |
| `pH_source` | categorical | dimensionless | `object` | 3182 | Provenance of pH_effective (measured_echem, inferred_strong_acid, inferred_neutral, etc.). |
| `elem_atomic_number` | numeric | atomic number Z | `float64` | 3179 | Atomic number (proton count) of the Metal element. |
| `elem_atomic_mass` | numeric | g/mol | `float64` | 3179 | Standard atomic weight of the Metal element. |
| `elem_pauling_en` | numeric | Pauling scale | `float64` | 3179 | Elemental Pauling electronegativity. |
| `elem_work_function_eV` | numeric | eV | `float64` | 3179 | Polycrystalline elemental electronic work function in electron-volts. |
| `elem_standard_potential_V_SHE` | numeric | V vs. SHE | `float64` | 3179 | Elemental standard reduction potential in V vs. SHE. |
| `elem_melting_point_K` | numeric | K | `float64` | 3179 | Elemental melting temperature in Kelvin. |
| `elem_molar_volume_cm3` | numeric | cm³/mol | `float64` | 3179 | Elemental molar volume in cubic centimeters per mole. |
| `mp_material_id` | identifier | Materials Project ID | `object` | 3177 | Materials Project crystal structure identifier. |
| `mp_work_function_eV` | numeric | eV | `float64` | 3144 | DFT-computed surface work function from Materials Project in electron-volts. |
| `mp_surface_energy_J_m2` | numeric | J/m² | `float64` | 3144 | DFT-computed cleavage/surface energy in Joules per square meter. |
| `mp_formation_energy_eV_atom` | numeric | eV/atom | `float64` | 3177 | DFT-computed formation enthalpy per atom in electron-volts. |
| `mp_band_gap_eV` | numeric | eV | `float64` | 3177 | Electronic band gap energy in electron-volts (0.0 eV for metallic conductors). |
| `thermo_E_onset_V_SHE` | numeric | V vs. SHE | `float64` | 3179 | Thermodynamic Pourbaix anodic dissolution or oxidation onset potential in V vs. SHE. |
| `thermo_corrosion_regime` | categorical | Pourbaix regime | `object` | 3179 | Thermodynamic Pourbaix stability regime (immunity, corrosion, passivation). |
| `thermo_stable_phase` | text | chemical phase | `object` | 3179 | Thermodynamically favored aqueous ion or solid oxide phase at the Exposure Condition. |
| `thermo_source` | categorical | dimensionless | `object` | 3182 | Provenance of the Pourbaix thermodynamic assessment (e.g., mp_pourbaix). |
| `mp_pourbaix_E_onset_V_SHE` | numeric | V vs. SHE | `float64` | 3179 | Materials Project Pourbaix oxidation onset potential at pH_effective in V vs. SHE. |
| `mp_stable_phase_at_Ecorr` | text | chemical phase | `object` | 3179 | Materials Project Pourbaix stable phase evaluated at (pH_effective, Ecorr_V_SHE). |
| `mp_corrosion_regime` | categorical | Pourbaix regime | `object` | 3179 | Materials Project Pourbaix stability classification at Ecorr_V_SHE (immunity, corrosion, passivation). |
| `mp_delta_G_pbx_eV_atom` | numeric | eV/atom | `float64` | 3179 | Pourbaix decomposition free energy (driving force above convex hull) at Ecorr_V_SHE in eV/atom. |
| `delta_E_overpotential_V` | numeric | V | `float64` | 3179 | Anodic overpotential above thermodynamic oxidation onset (Ecorr_V_SHE - thermo_E_onset_V_SHE) in Volts. |
| `organic_solvent_concentration_unit` | categorical | unit string | `object` | 6 | Concentration unit for organic_solvent_concentration (e.g., wt%, vol%, mM). |
| `water_content_wt_pct` | numeric | wt% | `float64` | 3182 | Water mass percentage in the Solution solvent system in [0, 100]. |
| `water_content_source` | categorical | dimensionless | `object` | 3182 | Provenance of water_content_wt_pct (reported, inferred_aqueous, or inferred_regime_default). |
| `solvent_regime` | categorical | dimensionless | `object` | 3182 | Water-content solvent regime classification (aqueous >= 90 wt%, semi_aqueous 1-90 wt%, non_aqueous < 1 wt%). |
| `solvent_subclass` | categorical | dimensionless | `object` | 3182 | Chemical-family or pH subclass of the Solution (acidic/neutral/alkaline/buffered for aqueous; glycol/alcohol/polar_aprotic/organic_acid/other_organic otherwise). |
| `icorr_source` | categorical | dimensionless | `object` | 3182 | Provenance of icorr_uA_cm2 (measured, derived_from_CR via Faraday law, or derived_from_Rp via Stern-Geary). |
| `log10_icorr_uA_cm2` | numeric | log10(µA/cm²) | `float64` | 3182 | Base-10 logarithm of standardized Corrosion Current Density i_corr in µA/cm². |
| `has_complete_pair` | boolean_flag | boolean | `bool` | 3182 | True when both Ecorr_V_SHE and icorr_uA_cm2 are present and valid for full polarization and lookup. |
| `beta_a_imputed` | boolean_flag | boolean | `bool` | 3182 | True if anodic Tafel slope beta_a was missing and imputed via group median or 60 mV/dec fallback. |
| `beta_c_imputed` | boolean_flag | boolean | `bool` | 3182 | True if cathodic Tafel slope beta_c was missing and imputed via group median or 120 mV/dec fallback. |
| `is_imputed` | boolean_flag | boolean | `bool` | 3182 | Quality flag indicating whether either Tafel slope (beta_a or beta_c) was imputed. |
| `quality_score` | quality_metric | score [0, 1] | `float64` | 3182 | Data reliability score in [0, 1], automatically decremented when is_imputed is True. |

## Table: `material_properties`

| Column | Role | Unit | Type | Non-Null | Description |
| :--- | :--- | :--- | :--- | ---: | :--- |
| `metal` | categorical | dimensionless | `object` | 25 | Conductive metallic element, compound, or alloy film identifier (e.g., Cu, Co, W, Ru, Ti, Ta, Al, Mo, TiN, Fe, Ni). |
| `material_id` | identifier | dimensionless | `object` | 25 | Canonical Metal identifier matching symbol. |
| `material_name` | text | dimensionless | `object` | 25 | Full English name of the Metal. |
| `symbol` | categorical | dimensionless | `object` | 25 | Chemical symbol of the Metal. |
| `category` | categorical | dimensionless | `object` | 25 | Semiconductor film functional category (metal, barrier, liner, semiconductor). |
| `density_g_cm3` | numeric | g/cm³ | `float64` | 25 | Bulk mass density of the Metal film. |
| `molar_mass_g_mol` | numeric | g/mol | `float64` | 25 | Atomic or molar mass of the Metal. |
| `valence` | numeric | e⁻/atom | `float64` | 25 | Dissolution Valence (z): electrons transferred per dissolved Metal atom in Faraday's law. |
| `equivalent_weight_g_eq` | numeric | g/eq | `float64` | 25 | Electrochemical equivalent weight (molar_mass_g_mol / valence). |
| `standard_potential_V_SHE` | numeric | V vs. SHE | `float64` | 25 | Standard reduction potential E° of the Metal in V vs. SHE. |
| `melting_point_C` | numeric | °C | `float64` | 13 | Melting temperature of the Metal at 1 atm in degrees Celsius. |
| `electronegativity_pauling` | numeric | Pauling scale | `float64` | 25 | Pauling electronegativity of the Metal. |
| `work_function_eV` | numeric | eV | `float64` | 25 | Electronic work function of the Metal surface in electron-volts. |
| `oxide_type` | text | chemical formula | `object` | 13 | Primary surface oxide formulas formed on the Metal. |
| `oxide_stability` | categorical | tier | `object` | 13 | Qualitative thermodynamic stability tier of the native oxide (low, medium, high). |
| `passivation_tendency` | numeric | index [0, 1] | `float64` | 25 | Dimensionless tendency of the Metal to form a self-limiting passive oxide film. |
| `pourbaix_stable_pH_min` | numeric | pH units | `float64` | 25 | Lower pH boundary of the Metal's passive oxide stability window. |
| `pourbaix_stable_pH_max` | numeric | pH units | `float64` | 25 | Upper pH boundary of the Metal's passive oxide stability window. |
| `crystal_structure` | categorical | lattice type | `object` | 13 | Room-temperature crystal structure (FCC, BCC, HCP). |
| `thermal_conductivity_W_mK` | numeric | W/(m·K) | `float64` | 13 | Bulk thermal conductivity of the Metal. |
| `electrical_resistivity_ohm_cm` | numeric | Ω·cm | `float64` | 13 | Bulk electrical resistivity of the Metal. |
| `common_corrosion_products` | text | chemical species | `object` | 13 | Typical dissolved ionic or solid corrosion products. |
| `notes` | text | dimensionless | `object` | 1 | Curator annotations or supplementary notes. |
| `base_element` | categorical | dimensionless | `object` | 25 | Primary metallic element symbol for elemental descriptor lookup. |
| `tier` | categorical | dimensionless | `object` | 25 | Material catalog tier (core semiconductor metal vs. extended material). |
| `has_faraday_properties` | boolean | dimensionless | `bool` | 25 | True when density and equivalent weight are available for Faraday corrosion-rate conversion. |

## Table: `solution_properties`

| Column | Role | Unit | Type | Non-Null | Description |
| :--- | :--- | :--- | :--- | ---: | :--- |
| `solution_id` | identifier | dimensionless | `object` | 1300 | Unique Solution formulation identifier. |
| `solution_name` | text | dimensionless | `object` | 1299 | Human-readable chemical formulation summary for the Solution. |
| `solution_category` | categorical | dimensionless | `object` | 1299 | pH regime category of the Solution (acidic, neutral, alkaline, etc.). |
| `materials` | text | dimensionless | `object` | 1299 | Semicolon-delimited list of Metals evaluated with this Solution. |
| `base_solvent` | categorical | dimensionless | `object` | 1299 | Primary liquid solvent (typically DI water). |
| `pH` | numeric | pH units | `float64` | 1276 | Hydrogen ion activity (-log10[H+]) of the Solution. |
| `oxidizer_type` | categorical | dimensionless | `object` | 1300 | Primary oxidizing agent in the Solution (e.g., H2O2, HNO3, None). |
| `oxidizer_concentration` | numeric | wt% or M | `float64` | 1300 | Numerical concentration of the oxidizing agent (0.0 when oxidizer_type is None). |
| `oxidizer_concentration_unit` | categorical | unit string | `object` | 175 | Concentration unit for oxidizer_concentration (e.g., wt%, M, mM). |
| `acid_type` | categorical | dimensionless | `object` | 520 | Acid component in the Solution formulation (e.g., H2SO4, HCl, HF, HClO4, citric acid). |
| `acid_concentration` | numeric | M or wt% | `float64` | 520 | Concentration of the acid component. |
| `acid_concentration_unit` | categorical | unit string | `object` | 520 | Concentration unit for acid_concentration. |
| `base_type` | categorical | dimensionless | `object` | 233 | Alkaline base component in the Solution formulation (e.g., KOH, NaOH, TMAH). |
| `base_concentration` | numeric | M or wt% | `float64` | 233 | Concentration of the alkaline base component. |
| `base_concentration_unit` | categorical | unit string | `object` | 233 | Concentration unit for base_concentration. |
| `HF_concentration_wt_pct` | numeric | wt% | `float64` | 46 | Hydrofluoric acid (HF) weight percent concentration in the Solution. |
| `chelating_agent_type` | categorical | dimensionless | `object` | 70 | Complexing or chelating agent (e.g., citric acid, glycine, oxalic acid, EDTA). |
| `chelating_agent_concentration` | numeric | mM or wt% | `float64` | 70 | Concentration of the complexing/chelating agent. |
| `chelating_agent_concentration_unit` | categorical | unit string | `object` | 70 | Concentration unit for chelating_agent_concentration. |
| `inhibitor_type` | categorical | dimensionless | `object` | 130 | Corrosion inhibitor chemical species (e.g., BTA). |
| `inhibitor_concentration` | numeric | mM or wt% | `float64` | 130 | Concentration of the corrosion inhibitor. |
| `inhibitor_concentration_unit` | categorical | unit string | `object` | 130 | Concentration unit for inhibitor_concentration. |
| `organic_solvent_type` | categorical | dimensionless | `object` | 1300 | Co-solvent organic species in the Solution (e.g., ethylene glycol, IPA). |
| `organic_solvent_concentration` | numeric | wt% or mM | `float64` | 14 | Concentration of the organic co-solvent. |
| `supporting_salt_type` | categorical | dimensionless | `object` | 790 | Supporting ionic salt species in the Solution (e.g., NaCl, Na2SO4, KNO3, KCl). |
| `supporting_salt_concentration` | numeric | M or wt% | `float64` | 790 | Concentration of the supporting ionic salt. |
| `supporting_salt_concentration_unit` | categorical | unit string | `object` | 790 | Concentration unit for supporting_salt_concentration. |
| `dissolved_oxygen_ppm` | numeric | ppm | `float64` | 0 | Dissolved oxygen concentration in parts per million. |
| `conductivity_mS_cm` | numeric | mS/cm | `float64` | 0 | Ionic electrical conductivity of the Solution in millisiemens per centimeter. |
| `ionic_strength_mol_L` | numeric | mol/L | `float64` | 0 | Total molar ionic strength of the Solution. |
| `ORP_mV` | numeric | mV | `float64` | 0 | Solution Oxidation-Reduction Potential in millivolts. |
| `deaerated` | boolean_flag | boolean | `object` | 1299 | True if the Solution was explicitly purged of dissolved oxygen prior to measurement. |
| `components_json` | text | json | `object` | 1299 | Structured JSON breakdown of Solution chemical components and concentrations. |
| `source` | text | DOI / ID | `object` | 1299 | Aggregated publication DOI and table row provenance identifiers. |
| `notes` | text | dimensionless | `float64` | 0 | Curator annotations or supplementary notes. |
| `organic_solvent_concentration_unit` | categorical | unit string | `object` | 14 | Concentration unit for organic_solvent_concentration (e.g., wt%, vol%, mM). |
| `water_content_wt_pct` | numeric | wt% | `float64` | 1300 | Water mass percentage in the Solution solvent system in [0, 100]. |
| `water_content_source` | categorical | dimensionless | `object` | 1300 | Provenance of water_content_wt_pct (reported, inferred_aqueous, or inferred_regime_default). |
| `solvent_regime` | categorical | dimensionless | `object` | 1300 | Water-content solvent regime classification (aqueous >= 90 wt%, semi_aqueous 1-90 wt%, non_aqueous < 1 wt%). |
| `solvent_subclass` | categorical | dimensionless | `object` | 1300 | Chemical-family or pH subclass of the Solution (acidic/neutral/alkaline/buffered for aqueous; glycol/alcohol/polar_aprotic/organic_acid/other_organic otherwise). |

## Table: `galvanic_series`

| Column | Role | Unit | Type | Non-Null | Description |
| :--- | :--- | :--- | :--- | ---: | :--- |
| `solution_id` | identifier | dimensionless | `object` | 711 | Unique Solution formulation identifier. |
| `metal` | categorical | dimensionless | `object` | 711 | Conductive metallic element, compound, or alloy film identifier (e.g., Cu, Co, W, Ru, Ti, Ta, Al, Mo, TiN, Fe, Ni). |
| `solution_name` | text | dimensionless | `object` | 696 | Human-readable chemical formulation summary for the Solution. |
| `pH` | numeric | pH units | `float64` | 711 | Hydrogen ion activity (-log10[H+]) of the Solution. |
| `oxidizer_type` | categorical | dimensionless | `object` | 711 | Primary oxidizing agent in the Solution (e.g., H2O2, HNO3, None). |
| `oxidizer_concentration` | numeric | wt% or M | `float64` | 711 | Numerical concentration of the oxidizing agent (0.0 when oxidizer_type is None). |
| `temperature_C` | numeric | °C | `float64` | 711 | Operating temperature of the Exposure Condition in degrees Celsius. |
| `Ecorr_V_SHE` | numeric | V vs. SHE | `float64` | 711 | Standardized Corrosion Potential (E_corr) converted to Volts vs. Standard Hydrogen Electrode. |
| `icorr_uA_cm2` | numeric | µA/cm² | `float64` | 711 | Standardized Corrosion Current Density (i_corr) at E_corr in microamperes per square centimeter. |
| `beta_a_mV_dec` | numeric | mV/dec | `float64` | 711 | Strictly positive anodic Tafel Slope (beta_a) in millivolts per decade of current density. |
| `beta_c_mV_dec` | numeric | mV/dec | `float64` | 711 | Strictly positive cathodic Tafel Slope (beta_c) in millivolts per decade of current density. |
| `record_count` | numeric | count | `int64` | 711 | Number of harmonized experimental records aggregated for that (Solution, Metal) pair. |
| `galvanic_rank` | numeric | integer rank (1..N) | `int64` | 711 | Monotonic Galvanic Series rank within a Solution (1 = most Active Metal, N = most Noble Metal). |
| `metals_in_solution` | numeric | count | `int64` | 711 | Total distinct Metals evaluated in that Solution within the Galvanic Series. |
| `relative_nobility` | categorical | role label | `object` | 711 | Qualitative galvanic role within the Solution (Most Active, Intermediate, Most Noble). |
