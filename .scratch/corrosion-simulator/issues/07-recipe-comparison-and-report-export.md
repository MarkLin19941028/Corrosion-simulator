# 07: Multi-Recipe Comparison & Multi-Format Report Export

**What to build:** A decision-support and automated reporting extension to the interactive platform that enables process engineers to define and compare 2 to 5 candidate **Cleaning Recipes** side by side (across single-metal or bimetallic galvanic criteria), visualizes comparative performance via summary tables, grouped bar charts, and multi-axis radar charts, automatically recommends the lowest-risk **Cleaning Recipe**, and generates comprehensive downloadable reports in HTML, PDF, CSV, JSON, PNG, and SVG formats containing input conditions, parameter provenance (Database vs. ML), figures, corrosion/galvanic metrics, uncertainties, **Confidence Tiers**, **Applicability Domain** warnings, and recipe comparison rankings, accompanied by complete project documentation.

**Blocked by:** 06: Interactive Streamlit Platform — Single-Metal & Galvanic Analysis Views

**Status:** ready-for-agent

- [ ] Users can define 2 to 5 candidate **Cleaning Recipes** (varying **Solution** formulation, pH, temperature, and **Immersion Time**) and evaluate them across a chosen **Metal** or bimetallic pair.
- [ ] The **Recipe Comparison** view renders a side-by-side metrics table, interactive bar chart, and normalized radar chart comparing **Initial Corrosion Rate ($\text{CR}_0$)**, final **Corrosion Depth**, **Thickness Loss Risk** score and tier, **Time to Threshold**, and (when coupled) **Acceleration Factor**, and clearly highlights the recommended lowest-risk **Cleaning Recipe**.
- [ ] The report generator produces self-contained HTML reports as well as PDF, CSV, JSON, PNG, and SVG exports containing full input summaries, parameter provenance (`Database` vs. `ML`), Tafel and time-resolved figures, uncertainty intervals, **Confidence Tiers**, **Applicability Domain** warnings, experimental recommendations, and comparison results.
- [ ] Project documentation (`README.md`) provides clear setup, data placement, notebook execution, test suite, and Streamlit launch instructions, and end-to-end automated tests verify multi-recipe ranking and all report export formats.
