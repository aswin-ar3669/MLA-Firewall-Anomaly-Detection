# python -m streamlit run .\main.py

import streamlit as st
import subprocess
import tempfile
import os
import shutil
import json
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Anomaly Firewall Detection", layout="wide")

st.title("Anomaly Firewall Detection — Train · Test · Predict")

# Utility functions
def save_uploaded_file(uploaded, target_path: Path):
    with open(target_path, "wb") as f:
        f.write(uploaded.getbuffer())
    return target_path

def run_script(cmd, cwd=None):
    """Run a subprocess command and return (returncode, stdout, stderr)."""
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=cwd)
        return proc.returncode, proc.stdout, proc.stderr
    except Exception as e:
        return 1, "", str(e)


# Sidebar for global options
st.sidebar.header("Global settings")
workdir_base = st.sidebar.text_input("Workspace base folder", value="./firewall_ui_workspace")
Path(workdir_base).mkdir(parents=True, exist_ok=True)

st.sidebar.markdown("---")
# Tabs for Train / Test / Predict
tab = st.tabs(["Test / Evaluate data from Models", "Prediction the Firewall Anomaly", "Compare Supervised Models", "Compare Unsupervised Models"])

# -------------------- TEST TAB --------------------
with tab[0]:
    st.header("Test / Evaluate models")
    test_csv_u = st.file_uploader("Upload test CSV", type=["csv"], key="test_csv")
    test_outdir = st.text_input("Test output directory (relative to workspace)", value="test_out", key="test_outdir")
    run_test = st.button("Run Test")

    if run_test:
        missing = []
        if test_csv_u is None:
            missing.append("test CSV")
        if missing:
            st.error("Please upload: " + ", ".join(missing))
        else:
            with st.spinner("Running test script..."):
                tmp = Path(workdir_base) / "test_tmp"
                if tmp.exists():
                    shutil.rmtree(tmp)
                tmp.mkdir(parents=True)

                test_csv_path = save_uploaded_file(test_csv_u, tmp / "test.csv")
                iso_path = Path("models/model_isolation_forest.joblib")
                clf_path = Path("models/model_gb_classifier.joblib")
                outdir_path = tmp / test_outdir
                outdir_path.mkdir(parents=True, exist_ok=True)

                # run test_model.py
                script_src = Path("test_model.py")
                if not script_src.exists():
                    st.error("test_model.py was not found in the Streamlit working directory. Please place test_model.py next to this app.")
                else:
                    cmd = ["python", "test_model.py", "--test_csv", str(test_csv_path), "--iso_model", str(iso_path), "--clf_model", str(clf_path), "--outdir", str(outdir_path)]
                    rc, out, err = run_script(cmd)

                    st.subheader("Console output")
                    st.text_area("stdout", out, height=200)

                    if rc == 0:
                        st.success("Testing completed — outputs saved in test output directory")
                        # List files in outdir
                        files = list(outdir_path.glob("**/*"))
                        if files:
                            st.write("Files produced:")

                            image_files = [f for f in files if f.suffix.lower() in [".png", ".jpg", ".jpeg"]]
                            other_files = [f for f in files if f.suffix.lower() in [".csv", ".json"]]

                            # Display download buttons for CSV/JSON
                            if other_files:
                                cols = st.columns(4)
                                for id, f in enumerate(other_files):
                                    rel = f.relative_to(outdir_path)
                                    # If it's a CSV file, show as DataFrame
                                    if f.suffix.lower() == ".csv":
                                        st.subheader(f"📄 {rel.name}")
                                        try:
                                            df_out = pd.read_csv(f)
                                            st.dataframe(df_out.head(50))
                                        except Exception as e:
                                            st.warning(f"Could not read {rel.name}: {e}")

                                    with cols[id % 2]:
                                        st.download_button(
                                            f"Download {rel}",
                                            data=open(f, 'rb'),
                                            file_name=rel.name,
                                            mime="application/octet-stream"
                                        )

                            # Display images in two columns
                            if image_files:
                                cols = st.columns(2)
                                for idx, f in enumerate(image_files):
                                    rel = f.relative_to(outdir_path)
                                    with cols[idx % 2]:
                                        st.image(str(f), caption=str(rel), use_container_width=True)


                    else:
                        st.error(f"Testing failed (rc={rc}). See stderr for details.")

# -------------------- PREDICT TAB --------------------
with tab[1]:
    st.header("Run predictions")
    input_csv_u = st.file_uploader("Upload input CSV for prediction", type=["csv"], key="input_csv")
    out_csv_name = st.text_input("Output CSV filename (relative to workspace)", value="predictions.csv", key="pred_outname")
    run_predict = st.button("Predict")

    if run_predict:
        missing = []
        if input_csv_u is None:
            missing.append("input CSV")
        if missing:
            st.error("Please upload: " + ", ".join(missing))
        else:
            with st.spinner("Running prediction script..."):
                tmp = Path(workdir_base) / "predict_tmp"
                if tmp.exists():
                    shutil.rmtree(tmp)
                tmp.mkdir(parents=True)

                input_csv_path = save_uploaded_file(input_csv_u, tmp / "input.csv")
                iso_path = Path("models/model_isolation_forest.joblib")
                clf_path = Path("models/model_gb_classifier.joblib")
                out_path = tmp / out_csv_name

                script_src = Path("predict.py")
                if not script_src.exists():
                    st.error("predict.py was not found in the Streamlit working directory. Please place predict.py next to this app.")
                else:
                    cmd = ["python", "predict.py", "--input_csv", str(input_csv_path), "--iso_model", str(iso_path), "--clf_model", str(clf_path), "--out", str(out_path)]
                    rc, out, err = run_script(cmd)

                    st.subheader("Console output")
                    st.text_area("stdout", out, height=100)

                    if rc == 0 and out_path.exists():
                        st.success("Prediction completed — download results below")
                        df_out = pd.read_csv(out_path)
                        st.dataframe(df_out.head(50))
                        st.download_button("Download predictions CSV", data=open(out_path,'rb'), file_name=out_path.name)
                    else:
                        st.error("Prediction failed — check stderr for details.")

# -------------------- COMPARE SUPERVISED TAB --------------------
with tab[2]:
    st.header("Compare Supervised Models")
    st.write("Compare multiple supervised learning models for firewall anomaly detection.")

    sup_csv = st.file_uploader("Upload training CSV for supervised comparison", type=["csv"], key="sup_compare_csv")

    if st.button("Run Supervised Comparison", key="run_sup_compare"):
        if sup_csv is None:
            st.error("Please upload a CSV file first.")
        else:
            with st.spinner("Running supervised model comparison..."):
                # Create unique temp directory
                sup_tmp = Path(workdir_base) / f"supervised_tmp"
                sup_tmp.mkdir(parents=True, exist_ok=True)

                try:
                    # Save uploaded file
                    csv_path = sup_tmp / "train.csv"
                    save_uploaded_file(sup_csv, csv_path)

                    # Output directory
                    sup_outdir = sup_tmp / "results"

                    # Run comparison script
                    cmd = [
                        "python", "compare_supervised.py",
                        "--csv", str(csv_path),
                        "--outdir", str(sup_outdir)
                    ]

                    code, stdout, stderr = run_script(cmd)

                    if code == 0:
                        st.success("✅ Supervised comparison completed!")
                        result_csv = sup_outdir / "supervised_results.csv"

                        if result_csv.exists():
                            df_results = pd.read_csv(result_csv)
                            st.dataframe(df_results, use_container_width=True)

                            # Create visualizations
                            st.subheader("📈 Performance Metrics Visualization")

                            # Top metrics display
                            col1, col2, col3, col4 = st.columns(4)

                            with col1:
                                st.metric("Best Accuracy", f"{df_results['Accuracy'].max():.4f}")
                                best_acc_model = df_results.loc[df_results['Accuracy'].idxmax(), 'Model']
                                st.caption(f"Model: {best_acc_model}")

                            with col2:
                                st.metric("Best Precision", f"{df_results['Precision'].max():.4f}")
                                best_prec_model = df_results.loc[df_results['Precision'].idxmax(), 'Model']
                                st.caption(f"Model: {best_prec_model}")

                            with col3:
                                st.metric("Best Recall", f"{df_results['Recall'].max():.4f}")
                                best_rec_model = df_results.loc[df_results['Recall'].idxmax(), 'Model']
                                st.caption(f"Model: {best_rec_model}")

                            with col4:
                                st.metric("Best F1-Score", f"{df_results['F1'].max():.4f}")
                                best_f1_model = df_results.loc[df_results['F1'].idxmax(), 'Model']
                                st.caption(f"Model: {best_f1_model}")

                            # Download button
                            csv_data = df_results.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                "Download Results CSV",
                                csv_data,
                                "supervised_comparison_results.csv",
                                "text/csv"
                            )
                        else:
                            st.warning("Results file not found")

                        if stdout:
                            with st.expander("📋 Output"):
                                st.code(stdout)
                    else:
                        st.error(f"❌ Script failed with code {code}")
                        if stderr:
                            st.error("Error details:")
                            st.code(stderr)
                        if stdout:
                            st.info("Output:")
                            st.code(stdout)

                except Exception as e:
                    st.error(f"Error: {str(e)}")

# -------------------- COMPARE UNSUPERVISED TAB --------------------
with tab[3]:
    st.header("Compare Unsupervised Models")
    st.write(
        "Compare multiple unsupervised anomaly detection models (Isolation Forest, One-Class SVM, Local Outlier Factor).")

    unsup_csv = st.file_uploader("Upload training CSV for unsupervised comparison", type=["csv"],
                                 key="unsup_compare_csv")
    if st.button("Run Unsupervised Comparison", key="run_unsup_compare"):
        if unsup_csv is None:
            st.error("❌ Please upload a CSV file first.")
        else:
            with st.spinner("🔄 Running unsupervised model comparison... This may take a few minutes."):
                # Create unique temp directory with timestamp
                unsup_tmp = Path(workdir_base) / f"unsupervised_tmp"
                unsup_tmp.mkdir(parents=True, exist_ok=True)

                try:
                    # Save uploaded file
                    csv_path = unsup_tmp / "train.csv"
                    save_uploaded_file(unsup_csv, csv_path)

                    # Output directory
                    unsup_outdir = unsup_tmp / "results"

                    # Run comparison script
                    cmd = [
                        "python", "compare_unsupervised.py",
                        "--csv", str(csv_path),
                        "--outdir", str(unsup_outdir),
                    ]

                    code, stdout, stderr = run_script(cmd)

                    if code == 0:
                        st.success("✅ Unsupervised comparison completed!")
                        result_csv = unsup_outdir / "unsupervised_results.csv"

                        if result_csv.exists():
                            # Read file content first to avoid permission issues
                            df_results = pd.read_csv(result_csv)

                            # Display results
                            st.subheader("📊 Model Comparison Results")
                            st.dataframe(df_results, use_container_width=True)

                            # Create visualizations
                            st.subheader("📈 Performance Metrics Visualization")

                            col1, col2, col3 = st.columns(3)

                            with col1:
                                st.metric("Best AUROC", f"{df_results['AUROC'].max():.4f}")
                            with col2:
                                st.metric("Best AP", f"{df_results['AP'].max():.4f}")
                            with col3:
                                st.metric("Best F1", f"{df_results['BestF1'].max():.4f}")

                            # Bar chart comparison
                            st.bar_chart(df_results.set_index('Model'))

                            # Download button - read content first
                            csv_data = df_results.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="📥 Download Results CSV",
                                data=csv_data,
                                file_name=f"unsupervised_comparison.csv",
                                mime="text/csv",
                                key="download_unsup_results"
                            )
                        else:
                            st.warning("⚠️ Results file not found")

                        # Show script output
                        if stdout:
                            with st.expander("📋 Script Output"):
                                st.code(stdout)
                    else:
                        st.error(f"❌ Script failed with exit code {code}")
                        if stderr:
                            st.error("**Error Details:**")
                            st.code(stderr)
                        if stdout:
                            st.info("**Output:**")
                            st.code(stdout)

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

# Footer
st.markdown("---")
st.caption("Place train_model.py, test_model.py and predict.py next to this app for it to work.")
