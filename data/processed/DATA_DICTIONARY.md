# Harmonized Electrochemical Database — Data Dictionary

All tables use standardized physical units: Corrosion Potentials ($E_{\text{corr}}$) in **V vs. SHE**, Corrosion Current Densities ($i_{\text{corr}}$) in **µA/cm²**, and Tafel Slopes ($\beta_a, \beta_c$) in strictly positive **mV/dec**.

## Table: `electrochemical_database`

| Column | Role | Unit | Type | Non-Null | Description |
| :--- | :--- | :--- | :--- | ---: | :--- |
| `record_id` | identifier | dimensionless | `object` | 1741 | Unique electrochemical measurement record identifier. |
| `material_id` | identifier | dimensionless | `object` | 1741 | Canonical Metal identifier matching symbol. |
| `solution_id` | identifier | dimensionless | `object` | 1741 | Unique Solution formulation identifier. |
| `metal` | categorical | dimensionless | `object` | 1741 | Conductive metallic element, compound, or alloy film identifier (e.g., Cu, Co, W, Ru, Ti, Ta, Al, Mo, TiN, Fe, Ni). |
| `material_class` | categorical | dimensionless | `object` | 1741 | High-level material classification (e.g., pure_metal, alloy, nitride, barrier). |
| `sample_name` | text | dimensionless | `object` | 1392 | Original experimental sample or specimen designation reported in the source publication. |
| `metal_form` | categorical | dimensionless | `object` | 1482 | Physical morphology or deposition form of the Metal (thin film, PVD, ALD, bulk, foil, rod). |
| `surface_preparation` | text | dimensionless | `object` | 1320 | Pre-exposure surface conditioning or polishing treatment (e.g., CMP, mechanical polish, annealed). |
| `solution_description` | text | dimensionless | `object` | 1274 | Raw text description of the Solution composition from the literature source. |
| `pH` | numeric | pH units | `float64` | 1741 | Hydrogen ion activity (-log10[H+]) of the Solution. |
| `temperature_C` | numeric | °C | `float64` | 1741 | Operating temperature of the Exposure Condition in degrees Celsius. |
| `reference_electrode` | categorical | dimensionless | `object` | 1741 | Original experimental reference electrode before conversion to SHE. |
| `Ecorr_V_SHE` | numeric | V vs. SHE | `float64` | 1741 | Standardized Corrosion Potential (E_corr) converted to Volts vs. Standard Hydrogen Electrode. |
| `Ecorr_V_original` | numeric | V vs. ref | `float64` | 1741 | Originally reported open-circuit Corrosion Potential prior to SHE conversion. |
| `icorr_uA_cm2` | numeric | µA/cm² | `float64` | 1741 | Standardized Corrosion Current Density (i_corr) at E_corr in microamperes per square centimeter. |
| `log_icorr` | numeric | log10(A/cm²) | `float64` | 1579 | Raw base-10 logarithm of corrosion current density in A/cm² from literature compilation. |
| `beta_a_mV_dec` | numeric | mV/dec | `float64` | 1741 | Strictly positive anodic Tafel Slope (beta_a) in millivolts per decade of current density. |
| `beta_c_mV_dec` | numeric | mV/dec | `float64` | 1741 | Strictly positive cathodic Tafel Slope (beta_c) in millivolts per decade of current density. |
| `tafel_slopes_general` | text | mV/dec | `object` | 29 | Unassigned or general Tafel slope values extracted from literature text. |
| `Rp_ohm_cm2` | numeric | Ω·cm² | `float64` | 427 | Linear polarization resistance (R_p) in ohm-square centimeters. |
| `Rp_unit` | categorical | unit string | `object` | 427 | Reported physical unit for linear polarization resistance R_p. |
| `CR_reported_value` | numeric | reported unit | `float64` | 516 | Corrosion Rate value as originally reported in the source publication. |
| `CR_reported_unit` | categorical | unit string | `object` | 516 | Physical unit for CR_reported_value (e.g., nm/min, mm/year, Å/min). |
| `CR_calculated_nm_min` | numeric | nm/min | `float64` | 1377 | Faraday-derived Corrosion Rate in nanometers per minute. |
| `measurement_method` | text | dimensionless | `object` | 1485 | Experimental electrochemical or analytical measurement techniques (potentiodynamic, EIS, LPR, ICP-MS). |
| `scan_rate` | text | mV/s | `object` | 822 | Potentiodynamic polarization potential sweep rate. |
| `immersion_time` | text | time unit | `object` | 721 | Experimental immersion or pre-conditioning duration reported in the study. |
| `aeration_condition` | categorical | dimensionless | `object` | 335 | Dissolved gas saturation state during testing (aerated, deaerated, N2/Ar purged). |
| `reaction_type` | categorical | dimensionless | `object` | 1741 | Primary electrochemical process classification (corrosion, passivity, etching). |
| `reactions` | text | dimensionless | `object` | 1741 | Half-cell reactions active under the Exposure Condition (HER, ORR, OER, metal dissolution). |
| `source_type` | categorical | dimensionless | `object` | 1741 | Literature or benchmark source classification (journal, seed_benchmark). |
| `source_reference` | text | DOI / ID | `object` | 1741 | Publication DOI, table row identifier, or benchmark reference string. |
| `data_extraction_method` | categorical | dimensionless | `object` | 1741 | Ingestion extraction pipeline method (table_column_mapped, regex, manual_curation). |
| `data_quality_score` | quality_metric | score [0, 1] | `float64` | 1741 | Raw ingestion data quality score prior to imputation penalty. |
| `quality_notes` | text | dimensionless | `object` | 1741 | Automated ingestion audit notes regarding unit inference or missing metadata. |
| `notes` | text | dimensionless | `float64` | 0 | Curator annotations or supplementary notes. |
| `solution_name` | text | dimensionless | `object` | 1331 | Human-readable chemical formulation summary for the Solution. |
| `solution_category` | categorical | dimensionless | `object` | 1331 | pH regime category of the Solution (acidic, neutral, alkaline, etc.). |
| `materials` | text | dimensionless | `object` | 1331 | Semicolon-delimited list of Metals evaluated with this Solution. |
| `base_solvent` | categorical | dimensionless | `object` | 1331 | Primary liquid solvent (typically DI water). |
| `oxidizer_type` | categorical | dimensionless | `object` | 1741 | Primary oxidizing agent in the Solution (e.g., H2O2, HNO3, None). |
| `oxidizer_concentration` | numeric | wt% or M | `float64` | 1741 | Numerical concentration of the oxidizing agent (0.0 when oxidizer_type is None). |
| `oxidizer_concentration_unit` | categorical | unit string | `object` | 189 | Concentration unit for oxidizer_concentration (e.g., wt%, M, mM). |
| `acid_type` | categorical | dimensionless | `object` | 465 | Acid component in the Solution formulation (e.g., H2SO4, HCl, HF, HClO4, citric acid). |
| `acid_concentration` | numeric | M or wt% | `float64` | 465 | Concentration of the acid component. |
| `acid_concentration_unit` | categorical | unit string | `object` | 465 | Concentration unit for acid_concentration. |
| `base_type` | categorical | dimensionless | `object` | 160 | Alkaline base component in the Solution formulation (e.g., KOH, NaOH, TMAH). |
| `base_concentration` | numeric | M or wt% | `float64` | 160 | Concentration of the alkaline base component. |
| `base_concentration_unit` | categorical | unit string | `object` | 160 | Concentration unit for base_concentration. |
| `HF_concentration_wt_pct` | numeric | wt% | `float64` | 45 | Hydrofluoric acid (HF) weight percent concentration in the Solution. |
| `chelating_agent_type` | categorical | dimensionless | `object` | 57 | Complexing or chelating agent (e.g., citric acid, glycine, oxalic acid, EDTA). |
| `chelating_agent_concentration` | numeric | mM or wt% | `float64` | 57 | Concentration of the complexing/chelating agent. |
| `chelating_agent_concentration_unit` | categorical | unit string | `object` | 57 | Concentration unit for chelating_agent_concentration. |
| `inhibitor_type` | categorical | dimensionless | `object` | 23 | Corrosion inhibitor chemical species (e.g., BTA). |
| `inhibitor_concentration` | numeric | mM or wt% | `float64` | 23 | Concentration of the corrosion inhibitor. |
| `inhibitor_concentration_unit` | categorical | unit string | `object` | 23 | Concentration unit for inhibitor_concentration. |
| `organic_solvent_type` | categorical | dimensionless | `object` | 5 | Co-solvent organic species in the Solution (e.g., ethylene glycol, IPA). |
| `organic_solvent_concentration` | numeric | wt% or mM | `float64` | 5 | Concentration of the organic co-solvent. |
| `supporting_salt_type` | categorical | dimensionless | `object` | 813 | Supporting ionic salt species in the Solution (e.g., NaCl, Na2SO4, KNO3, KCl). |
| `supporting_salt_concentration` | numeric | M or wt% | `float64` | 813 | Concentration of the supporting ionic salt. |
| `supporting_salt_concentration_unit` | categorical | unit string | `object` | 813 | Concentration unit for supporting_salt_concentration. |
| `dissolved_oxygen_ppm` | numeric | ppm | `float64` | 0 | Dissolved oxygen concentration in parts per million. |
| `conductivity_mS_cm` | numeric | mS/cm | `float64` | 0 | Ionic electrical conductivity of the Solution in millisiemens per centimeter. |
| `ionic_strength_mol_L` | numeric | mol/L | `float64` | 0 | Total molar ionic strength of the Solution. |
| `ORP_mV` | numeric | mV | `float64` | 0 | Solution Oxidation-Reduction Potential in millivolts. |
| `deaerated` | boolean_flag | boolean | `object` | 1331 | True if the Solution was explicitly purged of dissolved oxygen prior to measurement. |
| `components_json` | text | json | `object` | 1331 | Structured JSON breakdown of Solution chemical components and concentrations. |
| `source` | text | DOI / ID | `object` | 1331 | Aggregated publication DOI and table row provenance identifiers. |
| `material_name` | text | dimensionless | `object` | 1516 | Full English name of the Metal. |
| `symbol` | categorical | dimensionless | `object` | 1516 | Chemical symbol of the Metal. |
| `category` | categorical | dimensionless | `object` | 1516 | Semiconductor film functional category (metal, barrier, liner, semiconductor). |
| `density_g_cm3` | numeric | g/cm³ | `float64` | 1741 | Bulk mass density of the Metal film. |
| `molar_mass_g_mol` | numeric | g/mol | `float64` | 1741 | Atomic or molar mass of the Metal. |
| `valence` | numeric | e⁻/atom | `float64` | 1741 | Dissolution Valence (z): electrons transferred per dissolved Metal atom in Faraday's law. |
| `equivalent_weight_g_eq` | numeric | g/eq | `float64` | 1741 | Electrochemical equivalent weight (molar_mass_g_mol / valence). |
| `standard_potential_V_SHE` | numeric | V vs. SHE | `float64` | 1516 | Standard reduction potential E° of the Metal in V vs. SHE. |
| `melting_point_C` | numeric | °C | `float64` | 1516 | Melting temperature of the Metal at 1 atm in degrees Celsius. |
| `electronegativity_pauling` | numeric | Pauling scale | `float64` | 1516 | Pauling electronegativity of the Metal. |
| `work_function_eV` | numeric | eV | `float64` | 1516 | Electronic work function of the Metal surface in electron-volts. |
| `oxide_type` | text | chemical formula | `object` | 1516 | Primary surface oxide formulas formed on the Metal. |
| `oxide_stability` | categorical | tier | `object` | 1516 | Qualitative thermodynamic stability tier of the native oxide (low, medium, high). |
| `passivation_tendency` | numeric | index [0, 1] | `float64` | 1516 | Dimensionless tendency of the Metal to form a self-limiting passive oxide film. |
| `pourbaix_stable_pH_min` | numeric | pH units | `float64` | 1516 | Lower pH boundary of the Metal's passive oxide stability window. |
| `pourbaix_stable_pH_max` | numeric | pH units | `float64` | 1516 | Upper pH boundary of the Metal's passive oxide stability window. |
| `crystal_structure` | categorical | lattice type | `object` | 1516 | Room-temperature crystal structure (FCC, BCC, HCP). |
| `thermal_conductivity_W_mK` | numeric | W/(m·K) | `float64` | 1516 | Bulk thermal conductivity of the Metal. |
| `electrical_resistivity_ohm_cm` | numeric | Ω·cm | `float64` | 1516 | Bulk electrical resistivity of the Metal. |
| `common_corrosion_products` | text | chemical species | `object` | 1516 | Typical dissolved ionic or solid corrosion products. |
| `base_element` | categorical | dimensionless | `object` | 1738 | Primary metallic element symbol for elemental descriptor lookup. |
| `pH_effective` | numeric | pH units | `float64` | 1741 | Effective pH used for thermodynamic calculations (measured or inferred from acid/base molarity). |
| `pH_source` | categorical | dimensionless | `object` | 1741 | Provenance of pH_effective (measured_echem, inferred_strong_acid, inferred_neutral, etc.). |
| `elem_atomic_number` | numeric | atomic number Z | `float64` | 1738 | Atomic number (proton count) of the Metal element. |
| `elem_atomic_mass` | numeric | g/mol | `float64` | 1738 | Standard atomic weight of the Metal element. |
| `elem_pauling_en` | numeric | Pauling scale | `float64` | 1738 | Elemental Pauling electronegativity. |
| `elem_work_function_eV` | numeric | eV | `float64` | 1738 | Polycrystalline elemental electronic work function in electron-volts. |
| `elem_standard_potential_V_SHE` | numeric | V vs. SHE | `float64` | 1738 | Elemental standard reduction potential in V vs. SHE. |
| `elem_melting_point_K` | numeric | K | `float64` | 1738 | Elemental melting temperature in Kelvin. |
| `elem_molar_volume_cm3` | numeric | cm³/mol | `float64` | 1738 | Elemental molar volume in cubic centimeters per mole. |
| `mp_material_id` | identifier | Materials Project ID | `object` | 1738 | Materials Project crystal structure identifier. |
| `mp_work_function_eV` | numeric | eV | `float64` | 1722 | DFT-computed surface work function from Materials Project in electron-volts. |
| `mp_surface_energy_J_m2` | numeric | J/m² | `float64` | 1722 | DFT-computed cleavage/surface energy in Joules per square meter. |
| `mp_formation_energy_eV_atom` | numeric | eV/atom | `float64` | 1738 | DFT-computed formation enthalpy per atom in electron-volts. |
| `mp_band_gap_eV` | numeric | eV | `float64` | 1738 | Electronic band gap energy in electron-volts (0.0 eV for metallic conductors). |
| `thermo_E_onset_V_SHE` | numeric | V vs. SHE | `float64` | 1738 | Thermodynamic Pourbaix anodic dissolution or oxidation onset potential in V vs. SHE. |
| `thermo_corrosion_regime` | categorical | Pourbaix regime | `object` | 1738 | Thermodynamic Pourbaix stability regime (immunity, corrosion, passivation). |
| `thermo_stable_phase` | text | chemical phase | `object` | 1738 | Thermodynamically favored aqueous ion or solid oxide phase at the Exposure Condition. |
| `thermo_source` | categorical | dimensionless | `object` | 1741 | Provenance of the Pourbaix thermodynamic assessment (e.g., mp_pourbaix). |
| `mp_pourbaix_E_onset_V_SHE` | numeric | V vs. SHE | `float64` | 1738 | Materials Project Pourbaix oxidation onset potential at pH_effective in V vs. SHE. |
| `mp_stable_phase_at_Ecorr` | text | chemical phase | `object` | 1738 | Materials Project Pourbaix stable phase evaluated at (pH_effective, Ecorr_V_SHE). |
| `mp_corrosion_regime` | categorical | Pourbaix regime | `object` | 1738 | Materials Project Pourbaix stability classification at Ecorr_V_SHE (immunity, corrosion, passivation). |
| `mp_delta_G_pbx_eV_atom` | numeric | eV/atom | `float64` | 1738 | Pourbaix decomposition free energy (driving force above convex hull) at Ecorr_V_SHE in eV/atom. |
| `delta_E_overpotential_V` | numeric | V | `float64` | 1738 | Anodic overpotential above thermodynamic oxidation onset (Ecorr_V_SHE - thermo_E_onset_V_SHE) in Volts. |
| `icorr_source` | categorical | dimensionless | `object` | 1741 | Provenance of icorr_uA_cm2 (measured, derived_from_CR via Faraday law, or derived_from_Rp via Stern-Geary). |
| `log10_icorr_uA_cm2` | numeric | log10(µA/cm²) | `float64` | 1741 | Base-10 logarithm of standardized Corrosion Current Density i_corr in µA/cm². |
| `has_complete_pair` | boolean_flag | boolean | `bool` | 1741 | True when both Ecorr_V_SHE and icorr_uA_cm2 are present and valid for full polarization and lookup. |
| `beta_a_imputed` | boolean_flag | boolean | `bool` | 1741 | True if anodic Tafel slope beta_a was missing and imputed via group median or 60 mV/dec fallback. |
| `beta_c_imputed` | boolean_flag | boolean | `bool` | 1741 | True if cathodic Tafel slope beta_c was missing and imputed via group median or 120 mV/dec fallback. |
| `is_imputed` | boolean_flag | boolean | `bool` | 1741 | Quality flag indicating whether either Tafel slope (beta_a or beta_c) was imputed. |
| `quality_score` | quality_metric | score [0, 1] | `float64` | 1741 | Data reliability score in [0, 1], automatically decremented when is_imputed is True. |

## Table: `material_properties`

| Column | Role | Unit | Type | Non-Null | Description |
| :--- | :--- | :--- | :--- | ---: | :--- |
| `metal` | categorical | dimensionless | `object` | 22 | Conductive metallic element, compound, or alloy film identifier (e.g., Cu, Co, W, Ru, Ti, Ta, Al, Mo, TiN, Fe, Ni). |
| `material_id` | identifier | dimensionless | `object` | 22 | Canonical Metal identifier matching symbol. |
| `material_name` | text | dimensionless | `object` | 22 | Full English name of the Metal. |
| `symbol` | categorical | dimensionless | `object` | 22 | Chemical symbol of the Metal. |
| `category` | categorical | dimensionless | `object` | 22 | Semiconductor film functional category (metal, barrier, liner, semiconductor). |
| `density_g_cm3` | numeric | g/cm³ | `float64` | 22 | Bulk mass density of the Metal film. |
| `molar_mass_g_mol` | numeric | g/mol | `float64` | 22 | Atomic or molar mass of the Metal. |
| `valence` | numeric | e⁻/atom | `float64` | 22 | Dissolution Valence (z): electrons transferred per dissolved Metal atom in Faraday's law. |
| `equivalent_weight_g_eq` | numeric | g/eq | `float64` | 22 | Electrochemical equivalent weight (molar_mass_g_mol / valence). |
| `standard_potential_V_SHE` | numeric | V vs. SHE | `float64` | 22 | Standard reduction potential E° of the Metal in V vs. SHE. |
| `melting_point_C` | numeric | °C | `float64` | 11 | Melting temperature of the Metal at 1 atm in degrees Celsius. |
| `electronegativity_pauling` | numeric | Pauling scale | `float64` | 22 | Pauling electronegativity of the Metal. |
| `work_function_eV` | numeric | eV | `float64` | 22 | Electronic work function of the Metal surface in electron-volts. |
| `oxide_type` | text | chemical formula | `object` | 11 | Primary surface oxide formulas formed on the Metal. |
| `oxide_stability` | categorical | tier | `object` | 11 | Qualitative thermodynamic stability tier of the native oxide (low, medium, high). |
| `passivation_tendency` | numeric | index [0, 1] | `float64` | 22 | Dimensionless tendency of the Metal to form a self-limiting passive oxide film. |
| `pourbaix_stable_pH_min` | numeric | pH units | `float64` | 22 | Lower pH boundary of the Metal's passive oxide stability window. |
| `pourbaix_stable_pH_max` | numeric | pH units | `float64` | 22 | Upper pH boundary of the Metal's passive oxide stability window. |
| `crystal_structure` | categorical | lattice type | `object` | 11 | Room-temperature crystal structure (FCC, BCC, HCP). |
| `thermal_conductivity_W_mK` | numeric | W/(m·K) | `float64` | 11 | Bulk thermal conductivity of the Metal. |
| `electrical_resistivity_ohm_cm` | numeric | Ω·cm | `float64` | 11 | Bulk electrical resistivity of the Metal. |
| `common_corrosion_products` | text | chemical species | `object` | 11 | Typical dissolved ionic or solid corrosion products. |
| `notes` | text | dimensionless | `object` | 1 | Curator annotations or supplementary notes. |
| `base_element` | categorical | dimensionless | `object` | 22 | Primary metallic element symbol for elemental descriptor lookup. |
| `tier` | categorical | dimensionless | `object` | 22 | Material catalog tier (core semiconductor metal vs. extended material). |
| `has_faraday_properties` | boolean | dimensionless | `bool` | 22 | True when density and equivalent weight are available for Faraday corrosion-rate conversion. |

## Table: `solution_properties`

| Column | Role | Unit | Type | Non-Null | Description |
| :--- | :--- | :--- | :--- | ---: | :--- |
| `solution_id` | identifier | dimensionless | `object` | 640 | Unique Solution formulation identifier. |
| `solution_name` | text | dimensionless | `object` | 626 | Human-readable chemical formulation summary for the Solution. |
| `solution_category` | categorical | dimensionless | `object` | 626 | pH regime category of the Solution (acidic, neutral, alkaline, etc.). |
| `materials` | text | dimensionless | `object` | 626 | Semicolon-delimited list of Metals evaluated with this Solution. |
| `base_solvent` | categorical | dimensionless | `object` | 626 | Primary liquid solvent (typically DI water). |
| `pH` | numeric | pH units | `float64` | 631 | Hydrogen ion activity (-log10[H+]) of the Solution. |
| `oxidizer_type` | categorical | dimensionless | `object` | 640 | Primary oxidizing agent in the Solution (e.g., H2O2, HNO3, None). |
| `oxidizer_concentration` | numeric | wt% or M | `float64` | 640 | Numerical concentration of the oxidizing agent (0.0 when oxidizer_type is None). |
| `oxidizer_concentration_unit` | categorical | unit string | `object` | 97 | Concentration unit for oxidizer_concentration (e.g., wt%, M, mM). |
| `acid_type` | categorical | dimensionless | `object` | 217 | Acid component in the Solution formulation (e.g., H2SO4, HCl, HF, HClO4, citric acid). |
| `acid_concentration` | numeric | M or wt% | `float64` | 217 | Concentration of the acid component. |
| `acid_concentration_unit` | categorical | unit string | `object` | 217 | Concentration unit for acid_concentration. |
| `base_type` | categorical | dimensionless | `object` | 88 | Alkaline base component in the Solution formulation (e.g., KOH, NaOH, TMAH). |
| `base_concentration` | numeric | M or wt% | `float64` | 88 | Concentration of the alkaline base component. |
| `base_concentration_unit` | categorical | unit string | `object` | 88 | Concentration unit for base_concentration. |
| `HF_concentration_wt_pct` | numeric | wt% | `float64` | 28 | Hydrofluoric acid (HF) weight percent concentration in the Solution. |
| `chelating_agent_type` | categorical | dimensionless | `object` | 40 | Complexing or chelating agent (e.g., citric acid, glycine, oxalic acid, EDTA). |
| `chelating_agent_concentration` | numeric | mM or wt% | `float64` | 40 | Concentration of the complexing/chelating agent. |
| `chelating_agent_concentration_unit` | categorical | unit string | `object` | 40 | Concentration unit for chelating_agent_concentration. |
| `inhibitor_type` | categorical | dimensionless | `object` | 21 | Corrosion inhibitor chemical species (e.g., BTA). |
| `inhibitor_concentration` | numeric | mM or wt% | `float64` | 21 | Concentration of the corrosion inhibitor. |
| `inhibitor_concentration_unit` | categorical | unit string | `object` | 21 | Concentration unit for inhibitor_concentration. |
| `organic_solvent_type` | categorical | dimensionless | `object` | 3 | Co-solvent organic species in the Solution (e.g., ethylene glycol, IPA). |
| `organic_solvent_concentration` | numeric | wt% or mM | `float64` | 3 | Concentration of the organic co-solvent. |
| `supporting_salt_type` | categorical | dimensionless | `object` | 355 | Supporting ionic salt species in the Solution (e.g., NaCl, Na2SO4, KNO3, KCl). |
| `supporting_salt_concentration` | numeric | M or wt% | `float64` | 355 | Concentration of the supporting ionic salt. |
| `supporting_salt_concentration_unit` | categorical | unit string | `object` | 355 | Concentration unit for supporting_salt_concentration. |
| `dissolved_oxygen_ppm` | numeric | ppm | `float64` | 0 | Dissolved oxygen concentration in parts per million. |
| `conductivity_mS_cm` | numeric | mS/cm | `float64` | 0 | Ionic electrical conductivity of the Solution in millisiemens per centimeter. |
| `ionic_strength_mol_L` | numeric | mol/L | `float64` | 0 | Total molar ionic strength of the Solution. |
| `ORP_mV` | numeric | mV | `float64` | 0 | Solution Oxidation-Reduction Potential in millivolts. |
| `deaerated` | boolean_flag | boolean | `object` | 626 | True if the Solution was explicitly purged of dissolved oxygen prior to measurement. |
| `components_json` | text | json | `object` | 626 | Structured JSON breakdown of Solution chemical components and concentrations. |
| `source` | text | DOI / ID | `object` | 626 | Aggregated publication DOI and table row provenance identifiers. |
| `notes` | text | dimensionless | `float64` | 0 | Curator annotations or supplementary notes. |

## Table: `galvanic_series`

| Column | Role | Unit | Type | Non-Null | Description |
| :--- | :--- | :--- | :--- | ---: | :--- |
| `solution_id` | identifier | dimensionless | `object` | 385 | Unique Solution formulation identifier. |
| `metal` | categorical | dimensionless | `object` | 385 | Conductive metallic element, compound, or alloy film identifier (e.g., Cu, Co, W, Ru, Ti, Ta, Al, Mo, TiN, Fe, Ni). |
| `solution_name` | text | dimensionless | `object` | 364 | Human-readable chemical formulation summary for the Solution. |
| `pH` | numeric | pH units | `float64` | 385 | Hydrogen ion activity (-log10[H+]) of the Solution. |
| `oxidizer_type` | categorical | dimensionless | `object` | 385 | Primary oxidizing agent in the Solution (e.g., H2O2, HNO3, None). |
| `oxidizer_concentration` | numeric | wt% or M | `float64` | 385 | Numerical concentration of the oxidizing agent (0.0 when oxidizer_type is None). |
| `temperature_C` | numeric | °C | `float64` | 385 | Operating temperature of the Exposure Condition in degrees Celsius. |
| `Ecorr_V_SHE` | numeric | V vs. SHE | `float64` | 385 | Standardized Corrosion Potential (E_corr) converted to Volts vs. Standard Hydrogen Electrode. |
| `icorr_uA_cm2` | numeric | µA/cm² | `float64` | 385 | Standardized Corrosion Current Density (i_corr) at E_corr in microamperes per square centimeter. |
| `beta_a_mV_dec` | numeric | mV/dec | `float64` | 385 | Strictly positive anodic Tafel Slope (beta_a) in millivolts per decade of current density. |
| `beta_c_mV_dec` | numeric | mV/dec | `float64` | 385 | Strictly positive cathodic Tafel Slope (beta_c) in millivolts per decade of current density. |
| `record_count` | numeric | count | `int64` | 385 | Number of harmonized experimental records aggregated for that (Solution, Metal) pair. |
| `galvanic_rank` | numeric | integer rank (1..N) | `int64` | 385 | Monotonic Galvanic Series rank within a Solution (1 = most Active Metal, N = most Noble Metal). |
| `metals_in_solution` | numeric | count | `int64` | 385 | Total distinct Metals evaluated in that Solution within the Galvanic Series. |
| `relative_nobility` | categorical | role label | `object` | 385 | Qualitative galvanic role within the Solution (Most Active, Intermediate, Most Noble). |
