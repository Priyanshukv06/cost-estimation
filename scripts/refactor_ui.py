import re

def refactor_streamlit_app():
    with open('frontend/streamlit_app.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Replace tabs with sidebar navigation
    content = content.replace(
        '    tab1, tab2 = st.tabs(["Patient Evaluation", "Model Monitoring"])',
        '    app_mode = st.sidebar.radio("Navigation", ["Patient Evaluation", "Model Monitoring"])\n    if app_mode == "Patient Evaluation":'
    )
    content = content.replace('    with tab1:', '        pass # Used to be tab1')
    content = content.replace('    with tab2:', '    elif app_mode == "Model Monitoring":')
    
    # 2. Hide sidebar contents when in Model Monitoring
    # We need to wrap the sidebar contents inside the 'if app_mode == "Patient Evaluation":'
    # Find the sidebar block
    sidebar_start = content.find('    with st.sidebar:')
    guard_start = content.find('    # ── Guard: field options required')
    
    if sidebar_start != -1 and guard_start != -1:
        sidebar_content = content[sidebar_start:guard_start]
        # We want to keep 'with st.sidebar:' but inside it put app_mode and wrap the rest.
        # But wait, app_mode should be the first thing in the sidebar.
        
        # We will manually replace the sidebar logic.
        new_sidebar = """    with st.sidebar:
        app_mode = st.radio("Navigation", ["Patient Evaluation", "Model Monitoring"])
        
        if app_mode == "Patient Evaluation":
            st.markdown("### ⚙️ Settings")

            # Risk filter selector
            filter_label = st.selectbox(
                "Risk Filter Level",
                options=list(FILTER_LEVELS.keys()),
                index=2,  # Default: Balanced
                help="Controls how strictly the model flags risky predictions. Lenient = fewer flags, Aggressive = more flags.",
            )
            risk_level = FILTER_LEVELS[filter_label]

            st.divider()

            # Randomize button
            st.markdown("### 🎲 Sample Data")
            st.caption("Load a random patient from the test dataset")
            if st.button("🔀 Randomize Patient", use_container_width=True, type="primary"):
                random_patient = fetch_random_patient()
                if random_patient:
                    st.session_state["patient_data"] = random_patient
                    st.session_state["has_actuals"] = True

                    # Directly SET each widget key to the new patient's value.
                    for field_name in FIELD_LABELS:
                        wkey = f"field_{field_name}"
                        new_val = str(random_patient.get(field_name, ""))
                        opts = field_options.get(field_name, [])
                        if opts and new_val in opts:
                            st.session_state[wkey] = new_val
                        elif opts:
                            st.session_state[wkey] = opts[0]

                    # Auto-predict after randomize
                    st.session_state["auto_predict"] = True
                    st.rerun()

            st.divider()

            # Backend status
            st.markdown("### 📡 Backend Status")
            try:
                health = httpx.get(f"{API_BASE}/health", timeout=10).json()
                st.success(f"Connected — {health.get('cost_models_count', 0) + health.get('charge_models_count', 0)} models loaded")
            except Exception:
                st.error("Backend unreachable (may be cold-starting, wait ~30s)")

            st.divider()

            # Portfolio / GitHub links
            st.markdown("### 📂 Project")
            st.markdown(
                "[![GitHub](https://img.shields.io/badge/GitHub-Source_Code-181717?logo=github&style=for-the-badge)]"
                "(https://github.com/Priyanshukv06/cost-estimation)"
            )
            st.caption("Built with FastAPI + scikit-learn")
            st.caption(f"Backend: `{API_BASE}`")
        else:
            # Provide dummy risk_level just in case, though it's not used in Monitoring tab
            risk_level = "balanced"
"""
        content = content[:sidebar_start] + new_sidebar + content[guard_start:]
        
    # We must also remove the `app_mode = st.sidebar.radio` from where we injected it above.
    content = content.replace(
        '    app_mode = st.sidebar.radio("Navigation", ["Patient Evaluation", "Model Monitoring"])\n    if app_mode == "Patient Evaluation":',
        '    if app_mode == "Patient Evaluation":'
    )

    # 3. Update Model Monitoring Tab content
    
    # We will replace the whole tab2 section
    tab2_start = content.find('    elif app_mode == "Model Monitoring":')
    if tab2_start != -1:
        # Keep everything up to tab2_start, but rewrite tab2
        
        new_tab2 = """    elif app_mode == "Model Monitoring":
        st.markdown('<div class="section-title">🔍 Model Monitoring & Test Statistics</div>', unsafe_allow_html=True)
        st.caption("Insights based on the full external test dataset.")
        
        # Load test stats
        @st.cache_data(ttl=300)
        def fetch_test_stats():
            try:
                r = httpx.get(f"{API_BASE}/api/v1/stats/test_stats", timeout=REQUEST_TIMEOUT)
                r.raise_for_status()
                return r.json()
            except Exception as e:
                st.error(f"Failed to fetch test stats: {e}")
                return {}
                
        test_stats = fetch_test_stats()
        
        if not test_stats:
            st.warning("Test stats not available. Ensure the backend has loaded them.")
        else:
            model_toggle = st.radio("Select Model", ["Cost Estimation", "Charge Estimation"], horizontal=True)
            mt = "cost" if "Cost" in model_toggle else "charge"
            stats = test_stats.get(mt, {})
            
            if stats:
                st.markdown("### 📊 Dual Risk Impact Analysis")
                
                # Format threshold labels
                thresh_stats = stats.get("threshold_stats", {})
                
                # Render table with all presets
                import pandas as pd
                
                table_rows = []
                for level, data in thresh_stats.items():
                    table_rows.append({
                        "Preset": level.capitalize(),
                        "U-Thresh": f"{data['threshold_u']:.2f}",
                        "O-Thresh": f"{data['threshold_o']:.2f}",
                        "Filtered %": f"{data['filtering_pct']:.1f}%",
                        "Overall MAE": f"${data['overall_mae']:,.0f}",
                        "Under-Pred %": f"{data['under_pct']:.1f}%",
                        "Under MAE": f"${data['under_mae']:,.0f}",
                        "Over-Pred %": f"{data['over_pct']:.1f}%",
                        "Over MAE": f"${data['over_mae']:,.0f}"
                    })
                    
                df_thresh = pd.DataFrame(table_rows)
                # Sort putting No Filter first, then Aggressive, Cautious, Balanced, Moderate, Lenient.
                # Actually, threshold_u increases, so let's just sort by U-Thresh
                df_thresh = df_thresh.sort_values(by="U-Thresh", ascending=False)
                st.dataframe(df_thresh, use_container_width=True, hide_index=True)
                
                st.markdown("---")
                st.markdown("### 🛡️ Buffer Strategy Analysis (Under-Prediction Risk Only)")
                
                thresh_options = []
                for level, data in thresh_stats.items():
                    label = f"{level.capitalize()} (U: {data['threshold_u']:.2f}, O: {data['threshold_o']:.2f})"
                    thresh_options.append((level, label))
                    
                selected_label = st.selectbox(
                    "Select Risk Threshold Preset for Buffer Analysis", 
                    [lbl for _, lbl in thresh_options],
                    index=3 # Default to Balanced assuming No Filter is now 0
                )
                selected_level = next(lvl for lvl, lbl in thresh_options if lbl == selected_label)
                
                st.caption(f"Applies a financial buffer scaled by the under-prediction probability to claims that cleared the {selected_level.capitalize()} thresholds.")
                
                buf_stats = stats.get("buffer_stats", {}).get(selected_level, [])
                if buf_stats:
                    buf_df = pd.DataFrame(buf_stats)
                    # Format DataFrame
                    display_df = buf_df.copy()
                    display_df['buffer'] = display_df['buffer'].apply(lambda x: f"${x:,}")
                    display_df['overall_mae'] = display_df['overall_mae'].apply(lambda x: f"${x:,.0f}")
                    display_df['under_pct'] = display_df['under_pct'].apply(lambda x: f"{x:.1f}%")
                    display_df['under_mae'] = display_df['under_mae'].apply(lambda x: f"${x:,.0f}")
                    display_df['over_pct'] = display_df['over_pct'].apply(lambda x: f"{x:.1f}%")
                    display_df['over_mae'] = display_df['over_mae'].apply(lambda x: f"${x:,.0f}")
                    
                    display_df.columns = [
                        "Buffer Max", "Overall MAE", 
                        "Under-Pred %", "Avg Deficit (Under MAE)", 
                        "Over-Pred %", "Avg Excess (Over MAE)",
                        "Flipped Cases (Under → Over)"
                    ]
                    
                    st.dataframe(display_df, use_container_width=True, hide_index=True)
                
                st.markdown("---")
                st.markdown("### 🎯 Precision-Recall Curves")
                pr_cols = st.columns(2)
                
                pr_under = stats.get("pr_curves", {}).get("under", {})
                pr_over = stats.get("pr_curves", {}).get("over", {})
                
                with pr_cols[0]:
                    st.markdown("**Under-Prediction Risk Classifier**")
                    if pr_under:
                        df_pu = pd.DataFrame({"Precision": pr_under["precision"]}, index=pr_under["recall"]).sort_index()
                        df_pu["Recall"] = df_pu.index
                        df_pu["F1 Score"] = 2 * (df_pu["Precision"] * df_pu["Recall"]) / (df_pu["Precision"] + df_pu["Recall"] + 1e-9)
                        df_pu.drop(columns=["Recall"], inplace=True)
                        st.line_chart(df_pu, x_label="Recall", y_label="Score")
                        st.caption(f"AUC: {pr_under.get('auc', 0):.3f}")
                
                with pr_cols[1]:
                    st.markdown("**Over-Prediction Risk Classifier**")
                    if pr_over:
                        df_po = pd.DataFrame({"Precision": pr_over["precision"]}, index=pr_over["recall"]).sort_index()
                        df_po["Recall"] = df_po.index
                        df_po["F1 Score"] = 2 * (df_po["Precision"] * df_po["Recall"]) / (df_po["Precision"] + df_po["Recall"] + 1e-9)
                        df_po.drop(columns=["Recall"], inplace=True)
                        st.line_chart(df_po, x_label="Recall", y_label="Score")
                        st.caption(f"AUC: {pr_over.get('auc', 0):.3f}")
                        
                st.markdown("---")
                st.markdown("### 🔑 Feature Importance (Top Drivers)")
                st.caption("Derived via permutation importance from the dual risk classifiers.")
                
                fi_under = stats.get("feature_importance", {}).get("under", {})
                fi_over = stats.get("feature_importance", {}).get("over", {})
                
                fi_cols = st.columns(2)
                with fi_cols[0]:
                    st.markdown("**Drivers of Under-Prediction Risk**")
                    if fi_under:
                        df_fi_u = pd.DataFrame(list(fi_under.items()), columns=["Feature", "Importance"]).sort_values("Importance", ascending=True)
                        st.bar_chart(df_fi_u.set_index("Feature"), horizontal=True, color="#ff7675")
                with fi_cols[1]:
                    st.markdown("**Drivers of Over-Prediction Risk**")
                    if fi_over:
                        df_fi_o = pd.DataFrame(list(fi_over.items()), columns=["Feature", "Importance"]).sort_values("Importance", ascending=True)
                        st.bar_chart(df_fi_o.set_index("Feature"), horizontal=True, color="#00b894")

"""
        content = content[:tab2_start] + new_tab2
        
    # Put __main__ at the end correctly
    if 'if __name__ == "__main__":\n    main()' not in content:
        content += '\nif __name__ == "__main__":\n    main()\n'
        
    # Write back
    with open('frontend/streamlit_app.py', 'w', encoding='utf-8') as f:
        f.write(content)
        
if __name__ == "__main__":
    refactor_streamlit_app()
