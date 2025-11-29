import gradio as gr
import pandas as pd
from src.cleaner import llm_clean_csv, version_history, rollback_version

# --- helpers ---

def get_version_timestamps():
    return [v["timestamp"] for v in reversed(version_history)]


def extract_changes(df_before, df_after):
    changes = []
    for col in df_after.columns:
        if col.startswith("*"):
            continue
        for idx in df_after.index:
            val_before = df_before.at[idx, col] if idx in df_before.index and col in df_before.columns else None
            val_after = df_after.at[idx, col]

            if pd.isna(val_before):
                val_before = None

            if val_before != val_after:
                reason_cols = [
                    c for c in df_after.columns
                    if c.startswith(f"*{col}_") or "_imputed" in c or "_reason" in c or "_confidence" in c
                ]
                reason_data = {c: df_after.at[idx, c] for c in reason_cols if c in df_after.columns}

                changes.append({
                    "row": idx,
                    "column": col,
                    "before": val_before,
                    "after": val_after,
                    **reason_data
                })

    return pd.DataFrame(changes) if changes else pd.DataFrame(columns=["row", "column", "before", "after"])


def process_file(uploaded_file, data_type):
    df = pd.read_csv(uploaded_file.name)
    df_clean, _ = llm_clean_csv(df, data_type)

    versions = get_version_timestamps()
    changes_df = extract_changes(df, df_clean)
    flagged_rows = df_clean[df_clean.filter(like="_flag_review").any(axis=1)]

    return df, df_clean, changes_df, flagged_rows, versions


def show_version(version_timestamp):
    for v in version_history:
        if v["timestamp"] == version_timestamp:
            df_before = v["df_before"]
            df_after = v["df"]

            changes_df = extract_changes(df_before, df_after)
            flagged_rows = df_after[df_after.filter(like="_flag_review").any(axis=1)]

            return df_before, df_after, changes_df, flagged_rows

    empty_df = pd.DataFrame()
    return empty_df, empty_df, empty_df, empty_df


# --- Gradio UI ---

with gr.Blocks() as demo:

    gr.HTML("""
    <style>
        .gr-block {
            background-color: rgba(255,255,255,0.85);
            border-radius: 12px;
            padding: 15px;
            margin-bottom: 20px;
        }
        .top-banner {
            max-width: 100%;
            max-height: 460px;
            object-fit: cover;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        h1, h2, h3 { 
            text-align: center; 
            font-family: 'Inter', sans-serif; 
        }
        table.dataframe td, table.dataframe th { 
            min-width: 180px !important; 
            padding: 6px 12px !important; 
        }
        .dataframe { 
            overflow-x: auto !important; 
            max-height: 450px; 
        }
    </style>
    """)

    gr.Image("image.png", elem_classes="top-banner")
    gr.Markdown("## Data Detox - AI-Powered Data Quality Made Simple")

    with gr.Row():
        file_input = gr.File(label="Upload CSV")
        data_type_radio = gr.Radio(["Performance", "Attendance"], label="File Type", value="Performance")
        clean_btn = gr.Button("Clean CSV")

    with gr.Row():
        output_before = gr.DataFrame(label="Before Cleaning", wrap=True)
        output_after = gr.DataFrame(label="After Cleaning", wrap=True)

    changes_table = gr.DataFrame(label="Changes / Imputations")
    flagged_table = gr.DataFrame(label="Flagged Rows for Review")

    version_dropdown = gr.Dropdown(
        label="Select Version",
        choices=get_version_timestamps(),
        allow_custom_value=True
    )
    rollback_btn = gr.Button("Rollback to Selected Version")

    version_output_before = gr.DataFrame(label="Before Selected Version")
    version_output_after = gr.DataFrame(label="After Selected Version")
    version_changes = gr.DataFrame(label="Changes / Imputations (Selected Version)")
    version_flagged = gr.DataFrame(label="Flagged Rows (Selected Version)")

    # --- event bindings ---
    clean_btn.click(
        process_file,
        inputs=[file_input, data_type_radio],
        outputs=[output_before, output_after, changes_table, flagged_table, version_dropdown]
    )

    rollback_btn.click(
        show_version,
        inputs=version_dropdown,
        outputs=[version_output_before, version_output_after, version_changes, version_flagged]
    )

    version_dropdown.change(
        show_version,
        inputs=version_dropdown,
        outputs=[version_output_before, version_output_after, version_changes, version_flagged]
    )


if __name__ == "__main__":
    demo.launch()
