import pandas as pd

def export_csv(df, filename="cleaned_data.csv"):
    """Export cleaned dataframe to CSV"""
    df.to_csv(filename, index=False)
    return f"CSV exported as {filename}"

def show_diff(df_before, df_after):
    """Return differences between two dataframes"""
    diff_rows = df_before.ne(df_after)
    changes = []
    for row_idx in diff_rows.index:
        for col in diff_rows.columns:
            if diff_rows.at[row_idx, col]:
                changes.append({
                    "row": row_idx,
                    "column": col,
                    "before": df_before.at[row_idx, col],
                    "after": df_after.at[row_idx, col]
                })
    return pd.DataFrame(changes)
