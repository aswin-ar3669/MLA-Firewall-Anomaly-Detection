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
tab = st.tabs(["Test / Evaluate data from Models", "Prediction the Firewall Anomaly"])

# -------------------- TEST TAB --------------------
with tab[0]:
    st.header("Test / Evaluate models")
    test_csv_u = st.file_uploader("Upload test CSV", type=["csv"], key="test_csv")
    iso_model_u = st.file_uploader("Upload IsolationForest model (.joblib)", type=["joblib"], key="iso_model")
    clf_model_u = st.file_uploader("Upload GB classifier model (.joblib)", type=["joblib"], key="clf_model")
    test_outdir = st.text_input("Test output directory (relative to workspace)", value="test_out", key="test_outdir")
    run_test = st.button("Run Test")

    if run_test:
        missing = []
        if test_csv_u is None:
            missing.append("test CSV")
        if iso_model_u is None:
            missing.append("IsolationForest model")
        if clf_model_u is None:
            missing.append("GB classifier model")
        if missing:
            st.error("Please upload: " + ", ".join(missing))
        else:
            with st.spinner("Running test script..."):
                tmp = Path(workdir_base) / "test_tmp"
                if tmp.exists():
                    shutil.rmtree(tmp)
                tmp.mkdir(parents=True)

                test_csv_path = save_uploaded_file(test_csv_u, tmp / "test.csv")
                iso_path = save_uploaded_file(iso_model_u, tmp / iso_model_u.name)
                clf_path = save_uploaded_file(clf_model_u, tmp / clf_model_u.name)
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
    iso_model_u2 = st.file_uploader("Upload IsolationForest model (.joblib)", type=["joblib"], key="iso_model2")
    clf_model_u2 = st.file_uploader("Upload GB classifier model (.joblib)", type=["joblib"], key="clf_model2")
    out_csv_name = st.text_input("Output CSV filename (relative to workspace)", value="predictions.csv", key="pred_outname")
    run_predict = st.button("Predict")

    if run_predict:
        missing = []
        if input_csv_u is None:
            missing.append("input CSV")
        if iso_model_u2 is None:
            missing.append("IsolationForest model")
        if clf_model_u2 is None:
            missing.append("GB classifier model")
        if missing:
            st.error("Please upload: " + ", ".join(missing))
        else:
            with st.spinner("Running prediction script..."):
                tmp = Path(workdir_base) / "predict_tmp"
                if tmp.exists():
                    shutil.rmtree(tmp)
                tmp.mkdir(parents=True)

                input_csv_path = save_uploaded_file(input_csv_u, tmp / "input.csv")
                iso_path = save_uploaded_file(iso_model_u2, tmp / iso_model_u2.name)
                clf_path = save_uploaded_file(clf_model_u2, tmp / clf_model_u2.name)
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
                        st.dataframe(df_out.head(100))
                        st.download_button("Download predictions CSV", data=open(out_path,'rb'), file_name=out_path.name)
                    else:
                        st.error("Prediction failed — check stderr for details.")


# Footer
st.markdown("---")
st.caption("Place train_model.py, test_model.py and predict.py next to this app for it to work.")
