# 03: Solvent System UI controls, non-aqueous physics warnings, and report metadata

**What to build:** Expose Solvent System controls (`Aqueous / Semi-aqueous / Non-aqueous`, organic co-solvent selector, and water-content wt% slider) in the Streamlit sidebar and Cleaning Recipe comparison flow, display a prominent physics-limitation warning banner whenever a `non_aqueous` (`< 1%` wt% water) Exposure Condition is simulated, and include solvent regime metadata in all exported technical reports.

**Blocked by:** 02 (Solvent physical descriptors and water-weighted Pourbaix ML features)

**Status:** ready-for-agent

- [ ] `SimulationRequest` and the Streamlit sidebar include a **Solvent System** section allowing users to choose `aqueous`, `semi_aqueous`, or `non_aqueous`, select an organic co-solvent, and adjust `water_content_wt_pct` (with regime and water content staying mutually consistent).
- [ ] Selecting `non_aqueous` (`water_content_wt_pct < 1%`) renders a prominent warning banner in Single Metal and Galvanic views stating that aqueous pH, V vs. SHE, and aqueous Pourbaix assumptions are not strictly valid in this phase.
- [ ] Cleaning Recipe comparison supports comparing recipes across different solvent regimes / water contents side by side.
- [ ] Exported HTML, PDF, JSON, and CSV reports include `solvent_regime`, `solvent_subclass`, `organic_solvent_type`, and `water_content_wt_pct` metadata, verified by controller and reporting tests.
