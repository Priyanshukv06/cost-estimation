import os

CODE = '''
    with tab2:
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
                thresh_options = []
                for level, data in thresh_stats.items():
                    label = f"{level.capitalize()} (U: {data['threshold_u']:.2f}, O: {data['threshold_o']:.2f})"
                    thresh_options.append((level, label))
                    
                selected_label = st.selectbox(
                    "Select Risk Threshold Preset", 
                    [lbl for _, lbl in thresh_options],
                    index=2 # Default to Balanced
                )
                selected_level = next(lvl for lvl, lbl in thresh_options if lbl == selected_label)
                
                # Show threshold stats
                ts = thresh_stats[selected_level]
                
                t_cols = st.columns(4)
                with t_cols[0]:
                    render_metric_card("Filtered (Rejected)", f"{ts['filtering_pct']:.1f}%", "risky")
                with t_cols[1]:
                    render_metric_card("Overall MAE", f"${ts['overall_mae']:,.0f}", "cost" if mt == "cost" else "charge")
                with t_cols[2]:
                    render_metric_card("Under-Predictions", f"{ts['under_pct']:.1f}%", "risky")
                with t_cols[3]:
                    render_metric_card("Over-Predictions", f"{ts['over_pct']:.1f}%", "safe")
                    
                st.markdown(f"**Average Deficit (Under MAE):** ${ts['under_mae']:,.0f} &nbsp;&nbsp;|&nbsp;&nbsp; **Average Excess (Over MAE):** ${ts['over_mae']:,.0f}")
                
                st.markdown("---")
                st.markdown("### 🛡️ Buffer Strategy Analysis (Under-Prediction Risk Only)")
                st.caption(f"Applies a financial buffer scaled by the under-prediction probability to claims that cleared the {selected_level.capitalize()} thresholds.")
                
                import pandas as pd
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
                        df_pu = pd.DataFrame({"Precision": pr_under["precision"]}, index=pr_under["recall"])
                        st.line_chart(df_pu, x_label="Recall", y_label="Precision", color="#ff7675")
                        st.caption(f"AUC: {pr_under.get('auc', 0):.3f}")
                
                with pr_cols[1]:
                    st.markdown("**Over-Prediction Risk Classifier**")
                    if pr_over:
                        df_po = pd.DataFrame({"Precision": pr_over["precision"]}, index=pr_over["recall"])
                        st.line_chart(df_po, x_label="Recall", y_label="Precision", color="#00b894")
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
'''

def main():
    filepath = 'frontend/streamlit_app.py'
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(CODE)
    print("Appended tab2 successfully!")

if __name__ == '__main__':
    main()
