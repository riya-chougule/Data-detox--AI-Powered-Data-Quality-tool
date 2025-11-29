import pandas as pd
import os
from datetime import datetime
import re
import numpy as np

version_history = []  # each entry: {timestamp, df_before, df, file, changes, summary}

# --- grading helpers ---

GRADE_MAP = [
    (97, 100, "A+"), (90, 96, "A"), (85, 89, "A-"),
    (80, 84, "B+"), (75, 79, "B"), (70, 74, "B-"),
    (65, 69, "C+"), (60, 64, "C"), (50, 59, "D"), (0, 49, "F"),
]

def score_to_grade(score):
    try:
        s = float(score)
    except Exception:
        return None
    for lo, hi, g in GRADE_MAP:
        if lo <= s <= hi:
            return g
    return None

def grade_to_score_estimate(grade):
    for lo, hi, g in GRADE_MAP:
        if g == grade:
            return (lo + hi) / 2.0
    return None

# --- duplicate detection ---

def detect_duplicates(df, kind="auto"):
    if 'student_id' in df.columns:
        student_key = 'student_id'
    else:
        student_key = 'student_name' if 'student_name' in df.columns else None

    keys = []
    if kind == "attendance" or ("attendance_status" in df.columns and kind == "auto"):
        keys = [k for k in ([student_key, 'class', 'date'] if student_key else ['class', 'date']) if k in df.columns]
    elif kind == "performance" or ("subject" in df.columns and kind == "auto"):
        keys = [k for k in ([student_key, 'subject', 'test_date'] if student_key else ['subject', 'test_date']) if k in df.columns]
    else:
        keys = [k for k in ([student_key, 'date'] if student_key else ['date']) if k in df.columns]

    if not keys:
        return pd.DataFrame(), []

    dup_mask = df.duplicated(subset=keys, keep=False)
    duplicates = df[dup_mask].copy()
    return duplicates, keys


def dedupe_with_strategy(df, keys, strategy="keep_first"):
    if not keys:
        return df.copy(), []

    if strategy == "keep_first":
        deduped = df.drop_duplicates(subset=keys, keep='first').copy()
        removed = df[df.duplicated(subset=keys, keep='first')].copy()
        return deduped, list(removed.index)

    elif strategy == "keep_most_complete":
        completeness = df.notna().sum(axis=1)
        df['_completeness'] = completeness
        df_sorted = df.sort_values(by=keys + ['_completeness'],
                                   ascending=[True]*len(keys) + [False])
        deduped = df_sorted.drop_duplicates(subset=keys, keep='first').drop(columns=['_completeness']).copy()
        kept_idx = set(deduped.index)
        removed_idx = [i for i in df.index if i not in kept_idx]
        return deduped, removed_idx

    else:
        return df.copy(), []

# --- validation helpers ---

def try_parse_date(x):
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y", "%Y/%m/%d"):
        try:
            return pd.to_datetime(x, format=fmt)
        except Exception:
            continue
    try:
        return pd.to_datetime(x, errors='coerce')
    except Exception:
        return pd.NaT


def validate_dates(df, date_cols):
    bad_dates = {}
    for col in date_cols:
        if col in df.columns:
            parsed = pd.to_datetime(df[col], errors='coerce')
            bad = df[parsed.isna()]
            bad_dates[col] = bad.index.tolist()
            df[col] = parsed.dt.strftime('%Y-%m-%d')
    return bad_dates


def validate_scores(df):
    issues = {}
    if 'score' in df.columns:
        numeric = pd.to_numeric(df['score'], errors='coerce')
        non_numeric_idx = numeric[numeric.isna() & df['score'].notna()].index.tolist()
        issues['non_numeric_scores'] = non_numeric_idx

        out_of_range_idx = df[(numeric < 0) | (numeric > 100)].index.tolist()
        issues['out_of_range'] = out_of_range_idx
    return issues


def detect_outliers_scores(df, z_thresh=3.5):
    if 'score' not in df.columns:
        return []
    numeric = pd.to_numeric(df['score'], errors='coerce')
    median = numeric.median()
    mad = (np.abs(numeric - median)).median()
    if mad == 0 or pd.isna(mad):
        return []
    modified_z = 0.6745 * (numeric - median) / mad
    outlier_idx = df.index[modified_z.abs() > z_thresh].tolist()
    return outlier_idx

# --- attendance cleaning ---

def standardize_attendance(df):
    df_clean = df.copy()

    df_clean["_attendance_imputed"] = False
    df_clean["_attendance_reason"] = ""
    df_clean["_attendance_confidence"] = 1.0
    df_clean["_attendance_flag_review"] = False

    norm_map = {
        "presnt": "Present", "present": "Present", "absent": "Absent",
        "excused": "Excused", "late": "Late", "p": "Present", "a": "Absent"
    }

    for idx in df_clean.index:
        status = df_clean.at[idx, "attendance_status"] if "attendance_status" in df_clean.columns else None
        notes = df_clean.at[idx, "notes"] if "notes" in df_clean.columns else ""
        status_str = str(status).strip() if pd.notna(status) else ""

        if status_str:
            key = status_str.strip().lower()
            if key in norm_map:
                new_status = norm_map[key]
                if new_status != status:
                    df_clean.at[idx, "attendance_status"] = new_status
                    df_clean.at[idx, "_attendance_confidence"] = 0.95
                    df_clean.at[idx, "_attendance_reason"] = "normalized_status"

        if (pd.isna(status) or status_str == "" or status_str.lower() in ["nan", "none"]):
            note = "" if pd.isna(notes) else str(notes).lower()
            inferred = None
            confidence = 0.0
            reason = ""

            if "excused" in note:
                inferred = "Excused"; confidence = 0.95; reason = "notes:excused"
            elif re.search(r"\bsick\b|\bsick leave\b|\bfever\b", note):
                inferred = "Absent"; confidence = 0.9; reason = "notes:sick"
            elif "family emergency" in note or "family" in note:
                inferred = "Absent"; confidence = 0.9; reason = "notes:family_emergency"
            elif "late arrival" in note or "late" in note:
                inferred = "Absent"; confidence = 0.85; reason = "notes:late_as_absent"
            elif "absent" in note or "no show" in note or "truancy" in note:
                inferred = "Absent"; confidence = 0.9; reason = "notes:absent"
            else:
                inferred = "Unknown"; confidence = 0.4; reason = "no evidence in notes"

            df_clean.at[idx, "attendance_status"] = inferred
            df_clean.at[idx, "_attendance_imputed"] = True
            df_clean.at[idx, "_attendance_reason"] = reason
            df_clean.at[idx, "_attendance_confidence"] = confidence

            if confidence < 0.6:
                df_clean.at[idx, "_attendance_flag_review"] = True

    if "notes" in df_clean.columns:
        df_clean["notes"] = df_clean["notes"].fillna("")
    if "class" in df_clean.columns:
        df_clean["class"] = df_clean["class"].astype(str).str.strip().str.title()

    return df_clean, None

# --- performance cleaning ---

def standardize_performance(df):
    df_clean = df.copy()

    # metadata
    df_clean["_score_imputed"] = False
    df_clean["_score_reason"] = ""
    df_clean["_score_confidence"] = 1.0

    df_clean["_grade_imputed"] = False
    df_clean["_grade_reason"] = ""
    df_clean["_grade_confidence"] = 1.0

    df_clean["_performance_flag_review"] = False

    if "subject" in df_clean.columns:
        df_clean["subject"] = df_clean["subject"].astype(str).str.strip().str.title()
    if "remarks" in df_clean.columns:
        df_clean["remarks"] = df_clean["remarks"].fillna("No remarks")
    else:
        df_clean["remarks"] = "No remarks"

    if "score" in df_clean.columns:
        df_clean["score"] = pd.to_numeric(df_clean["score"], errors="coerce")

    subject_medians = {}
    if "subject" in df_clean.columns and "score" in df_clean.columns:
        grouped = df_clean.dropna(subset=["score"]).groupby("subject")["score"]
        for subj, series in grouped:
            if len(series) > 0:
                subject_medians[subj] = float(series.median())

    # main loop
    for idx in df_clean.index:
        score = df_clean.at[idx, "score"] if "score" in df_clean.columns else None
        grade = df_clean.at[idx, "grade"] if "grade" in df_clean.columns else None
        subj = df_clean.at[idx, "subject"] if "subject" in df_clean.columns else None

        score_present = pd.notna(score)
        grade_present = pd.notna(grade) and str(grade).strip() != ""

        if score_present and not grade_present:
            derived = score_to_grade(score)
            if derived:
                df_clean.at[idx, "grade"] = derived
                df_clean.at[idx, "_grade_imputed"] = True
                df_clean.at[idx, "_grade_reason"] = "derived_from_score"
                df_clean.at[idx, "_grade_confidence"] = 0.98

        elif (not score_present) and grade_present:
            est = grade_to_score_estimate(str(grade).strip())
            if est is not None:
                df_clean.at[idx, "score"] = float(est)
                df_clean.at[idx, "_score_imputed"] = True
                df_clean.at[idx, "_score_reason"] = "estimated_from_grade_band"
                df_clean.at[idx, "_score_confidence"] = 0.6
                df_clean.at[idx, "_performance_flag_review"] = True

        elif (not score_present) and (not grade_present):
            median_val = subject_medians.get(subj) if subj else None
            if median_val is not None and not np.isnan(median_val):
                df_clean.at[idx, "score"] = float(median_val)
                df_clean.at[idx, "_score_imputed"] = True
                df_clean.at[idx, "_score_reason"] = "median_subject"
                df_clean.at[idx, "_score_confidence"] = 0.75

                derived = score_to_grade(median_val)
                if derived:
                    df_clean.at[idx, "grade"] = derived
                    df_clean.at[idx, "_grade_imputed"] = True
                    df_clean.at[idx, "_grade_reason"] = "derived_from_imputed_score"
                    df_clean.at[idx, "_grade_confidence"] = 0.85

                df_clean.at[idx, "_performance_flag_review"] = True
            else:
                df_clean.at[idx, "_performance_flag_review"] = True
                df_clean.at[idx, "_score_confidence"] = 0.0
                df_clean.at[idx, "_score_reason"] = "no_evidence_for_imputation"
                df_clean.at[idx, "_grade_reason"] = "no_evidence_for_imputation"

    # auto remarks helper
    def generate_performance_remark(score, grade):
        if pd.notna(score):
            s = float(score)
            if s >= 90: return "Excellent performance. Keep it up."
            elif s >= 80: return "Very good performance with strong understanding."
            elif s >= 70: return "Good performance. There is room for improvement."
            elif s >= 60: return "Average performance. Needs consistent practice."
            elif s >= 50: return "Below average. Improvement required."
            else: return "Poor performance. Immediate attention needed."

        if pd.notna(grade):
            g = str(grade).upper()
            if g in ["A+", "A"]: return "Excellent performance. Keep it up."
            elif g in ["A-", "B+"]: return "Very good performance with strong understanding."
            elif g in ["B", "B-"]: return "Good performance. There is room for improvement."
            elif g in ["C+", "C"]: return "Average performance. Needs consistent practice."
            elif g == "D": return "Below average. Improvement required."
            else: return "Poor performance. Immediate attention needed."

        return "No remarks"

    for idx in df_clean.index:
        score = df_clean.at[idx, "score"]
        grade = df_clean.at[idx, "grade"]
        correct_remark = generate_performance_remark(score, grade)
        df_clean.at[idx, "remarks"] = correct_remark

    df_clean["remarks"] = df_clean["remarks"].astype(str)
    return df_clean, None

# --- summary ---

def generate_quality_summary(df_before, df_after, removed_indices, duplicates_count, validation_issues, outliers):
    summary = {
        "total_rows_before": len(df_before),
        "total_rows_after": len(df_after),
        "duplicates_found": duplicates_count,
        "duplicates_removed": len(removed_indices),
        "validation_issues": {
            k: (len(v) if isinstance(v, list) else v)
            for k, v in validation_issues.items()
        } if isinstance(validation_issues, dict) else validation_issues,
        "outliers_count": len(outliers) if outliers else 0,
        "imputations": {
            "attendance_imputed": int(df_after.get("_attendance_imputed", pd.Series()).sum())
                if "_attendance_imputed" in df_after.columns else 0,
            "grade_imputed": int(df_after.get("_grade_imputed", pd.Series()).sum())
                if "_grade_imputed" in df_after.columns else 0,
            "score_imputed": int(df_after.get("_score_imputed", pd.Series()).sum())
                if "_score_imputed" in df_after.columns else 0
        }
    }
    return summary

# --- main pipeline ---

def llm_clean_csv(df, data_type="Performance", dedupe_strategy="keep_first", detect_dups=True):
    df_before = df.copy()

    validation_issues = {}

    date_cols = [c for c in df.columns if c.lower() in ("date", "test_date")]
    bad_dates = validate_dates(df_before, date_cols) if date_cols else {}
    validation_issues['bad_dates'] = bad_dates

    duplicates, keys = detect_duplicates(
        df_before,
        kind=("attendance" if data_type == "Attendance" else "performance")
    )
    duplicates_count = len(duplicates)

    removed_indices = []
    df_work = df_before.copy()

    if detect_dups and duplicates_count > 0:
        df_work, removed_indices = dedupe_with_strategy(df_work, keys, strategy=dedupe_strategy)

    if data_type == "Performance":
        df_clean, _ = standardize_performance(df_work)
    elif data_type == "Attendance":
        df_clean, _ = standardize_attendance(df_work)
    else:
        df_clean = df_work.copy()

    score_issues = {}
    if 'score' in df_clean.columns:
        score_issues_list = validate_scores(df_clean)
        outliers = detect_outliers_scores(df_clean)
        score_issues['score_issues'] = score_issues_list
        score_issues['outliers'] = outliers
    else:
        outliers = []

    validation_issues.update(score_issues)

    if not os.path.exists("cleaned_versions"):
        os.makedirs("cleaned_versions")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_filename = f"cleaned_versions/cleaned_{timestamp}.csv"
    df_clean.to_csv(version_filename, index=False)

    summary = generate_quality_summary(
        df_before,
        df_clean,
        removed_indices,
        duplicates_count,
        validation_issues,
        outliers
    )

    changes = {
        "duplicates_removed_idx": removed_indices,
        "duplicates_keys": keys,
        "bad_dates": bad_dates,
        "outliers": outliers,
    }

    version_history.append({
        "timestamp": timestamp,
        "df_before": df_before.copy(),
        "df": df_clean.copy(),
        "file": version_filename,
        "changes": changes,
        "summary": summary
    })

    return df_clean, version_filename


def rollback_version(version_timestamp):
    for v in version_history:
        if v['timestamp'] == version_timestamp:
            return v['df']
    return pd.DataFrame()
