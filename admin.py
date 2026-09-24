import streamlit as st
import pandas as pd
import json
import csv
import io
from datetime import datetime, timedelta
from collections import Counter
from data_store import get_all, clear_all

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Admin Panel · Medicine AI", page_icon="🛡️", layout="wide")

# ── ADMIN CREDENTIALS (change these!) ────────────────────────────────────────
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "medai@2024"

# ── STYLES ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=Plus+Jakarta+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }

.admin-badge { display:inline-block; background:#EEEDFE; color:#534AB7; font-size:11px; font-weight:600; letter-spacing:.08em; text-transform:uppercase; padding:5px 14px; border-radius:20px; margin-bottom:.75rem; border:.5px solid #CECBF6; }
.page-title  { font-family:'DM Serif Display',serif; font-size:36px; font-weight:400; line-height:1.15; margin-bottom:.25rem; }
.page-title em { font-style:italic; color:#7F77DD; }
.subtitle    { font-size:14px; color:#6B7280; margin-bottom:2rem; }

.kpi { background:#F9FAFB; border:.5px solid #E5E7EB; border-radius:14px; padding:1.2rem 1.4rem; text-align:center; }
.kpi-val   { font-family:'DM Serif Display',serif; font-size:36px; line-height:1; color:#111827; margin-bottom:4px; }
.kpi-label { font-size:12px; color:#6B7280; font-weight:500; }
.kpi-delta { font-size:11px; margin-top:4px; }
.delta-up   { color:#1D9E75; }
.delta-down { color:#D85A30; }

.sec-header { font-size:11px; font-weight:600; letter-spacing:.1em; text-transform:uppercase; color:#6B7280; margin:1.5rem 0 .75rem; border-bottom:.5px solid #E5E7EB; padding-bottom:.4rem; }

.log-row { background:#F9FAFB; border:.5px solid #E5E7EB; border-radius:12px; padding:.9rem 1.1rem; margin-bottom:.5rem; }
.log-drug { font-size:14px; font-weight:600; color:#111827; }
.log-meta  { font-size:12px; color:#6B7280; margin-top:2px; }

.sev-pill { padding:2px 10px; border-radius:20px; font-size:11px; font-weight:600; }
.sev-H { background:#FAECE7; color:#993C1D; }
.sev-M { background:#FAEEDA; color:#854F0B; }
.sev-L { background:#E1F5EE; color:#0F6E56; }

.insight-card { background:#EEEDFE; border:.5px solid #CECBF6; border-radius:14px; padding:1rem 1.25rem; margin-bottom:.6rem; }
.insight-card p { font-size:13px; color:#3C3489; line-height:1.6; margin:0; }

.login-wrap { max-width:420px; margin:6rem auto 0; background:#fff; border:.5px solid #E5E7EB; border-radius:20px; padding:2.5rem; }
.login-title { font-family:'DM Serif Display',serif; font-size:28px; text-align:center; margin-bottom:.25rem; }
.login-sub   { font-size:13px; color:#6B7280; text-align:center; margin-bottom:1.75rem; }

.danger-btn button { background:#FAECE7 !important; color:#993C1D !important; border:.5px solid #F5C4B3 !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# AUTH
# ══════════════════════════════════════════════════════════════════════════════
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

def do_login(username, password):
    if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
        st.session_state.authenticated = True
        st.rerun()
    else:
        st.error("Invalid username or password.")

if not st.session_state.authenticated:
    st.markdown("""
    <div class="login-wrap">
        <div class="login-title">🛡️ Admin Panel</div>
        <div class="login-sub">Medicine AI · Restricted Access</div>
    </div>
    """, unsafe_allow_html=True)

    with st.form("login_form"):
        st.markdown('<p style="font-size:12px;font-weight:600;color:#6B7280;margin-bottom:4px">USERNAME</p>', unsafe_allow_html=True)
        uname = st.text_input("Username", placeholder="admin", label_visibility="collapsed")
        st.markdown('<p style="font-size:12px;font-weight:600;color:#6B7280;margin-bottom:4px;margin-top:12px">PASSWORD</p>', unsafe_allow_html=True)
        pwd   = st.text_input("Password", type="password", placeholder="••••••••", label_visibility="collapsed")
        submitted = st.form_submit_button("Sign in →", use_container_width=True, type="primary")
        if submitted:
            do_login(uname, pwd)
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# LOGGED IN — LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
records = get_all()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛡️ Admin Panel")
    st.caption(f"Logged in as **{ADMIN_USERNAME}**")
    st.markdown("---")
    page = st.radio("Navigate", ["📊 Dashboard", "📋 Prediction Log", "💊 Medicine Insights", "⚙️ Settings"], label_visibility="collapsed")
    st.markdown("---")
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def records_to_df(records):
    rows = []
    for r in records:
        rows.append({
            "ID":              r["id"],
            "Timestamp":       r["timestamp"],
            "Patient":         r["patient"]["name"],
            "Age":             r["patient"]["age"],
            "Weight (kg)":     r["patient"]["weight"],
            "Drug":            r["medicine"]["drug"],
            "Therapeutic":     r["medicine"]["therapeutic_class"],
            "Chemical":        r["medicine"]["chemical_class"],
            "Use":             r["medicine"]["primary_use"],
            "Habit Forming":   r["medicine"]["habit_forming"],
            "Severity Score":  r["result"]["severity_score"],
            "Effects Found":   r["result"]["effects_found"],
            "Total Known":     r["result"]["total_effects"],
            "Summary":         r["result"]["summary"],
        })
    return pd.DataFrame(rows)

def severity_label(score):
    if score >= 8: return "High"
    if score >= 5: return "Medium"
    return "Low"


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Dashboard":
    st.markdown('<div class="admin-badge">Admin · Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<p class="page-title">Analytics <em>Overview</em></p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Real-time usage statistics across all predictions.</p>', unsafe_allow_html=True)

    if not records:
        st.info("No predictions yet. Run the app and make a prediction first.")
        st.stop()

    df = records_to_df(records)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df["Date"]      = df["Timestamp"].dt.date

    total        = len(df)
    avg_sev      = round(df["Severity Score"].mean(), 1)
    avg_effects  = round(df["Effects Found"].mean(), 1)
    habit_count  = df[df["Habit Forming"] == "Yes"].shape[0]

    # Today vs yesterday
    today     = datetime.now().date()
    yesterday = today - timedelta(days=1)
    today_ct  = df[df["Date"] == today].shape[0]
    yest_ct   = df[df["Date"] == yesterday].shape[0]
    delta_txt = f"▲ {today_ct - yest_ct} vs yesterday" if today_ct >= yest_ct else f"▼ {yest_ct - today_ct} vs yesterday"
    delta_cls = "delta-up" if today_ct >= yest_ct else "delta-down"

    # KPI row
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.markdown(f'<div class="kpi"><div class="kpi-val">{total}</div><div class="kpi-label">Total predictions</div><div class="kpi-delta {delta_cls}">{delta_txt}</div></div>', unsafe_allow_html=True)
    k2.markdown(f'<div class="kpi"><div class="kpi-val">{today_ct}</div><div class="kpi-label">Today</div></div>', unsafe_allow_html=True)
    k3.markdown(f'<div class="kpi"><div class="kpi-val">{avg_sev}</div><div class="kpi-label">Avg severity score</div></div>', unsafe_allow_html=True)
    k4.markdown(f'<div class="kpi"><div class="kpi-val">{avg_effects}</div><div class="kpi-label">Avg effects found</div></div>', unsafe_allow_html=True)
    k5.markdown(f'<div class="kpi"><div class="kpi-val">{habit_count}</div><div class="kpi-label">Habit-forming queries</div></div>', unsafe_allow_html=True)

    st.markdown("")

    # Charts row
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<p class="sec-header">Predictions over time</p>', unsafe_allow_html=True)
        daily = df.groupby("Date").size().reset_index(name="Count")
        st.bar_chart(daily.set_index("Date")["Count"], use_container_width=True, height=220)

    with col_b:
        st.markdown('<p class="sec-header">Severity distribution</p>', unsafe_allow_html=True)
        df["Sev Label"] = df["Severity Score"].apply(severity_label)
        sev_dist = df["Sev Label"].value_counts().reset_index()
        sev_dist.columns = ["Severity", "Count"]
        st.bar_chart(sev_dist.set_index("Severity")["Count"], use_container_width=True, height=220)

    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown('<p class="sec-header">Top 8 most searched drugs</p>', unsafe_allow_html=True)
        top_drugs = df["Drug"].value_counts().head(8).reset_index()
        top_drugs.columns = ["Drug", "Count"]
        st.bar_chart(top_drugs.set_index("Drug")["Count"], use_container_width=True, height=220)

    with col_d:
        st.markdown('<p class="sec-header">Top therapeutic classes</p>', unsafe_allow_html=True)
        top_thera = df["Therapeutic"].value_counts().head(8).reset_index()
        top_thera.columns = ["Therapeutic Class", "Count"]
        st.bar_chart(top_thera.set_index("Therapeutic Class")["Count"], use_container_width=True, height=220)

    # Recent predictions table
    st.markdown('<p class="sec-header">Recent predictions</p>', unsafe_allow_html=True)
    recent = df.sort_values("Timestamp", ascending=False).head(10)[
        ["ID","Timestamp","Patient","Drug","Severity Score","Effects Found"]
    ]
    st.dataframe(recent, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICTION LOG
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Prediction Log":
    st.markdown('<div class="admin-badge">Admin · Prediction Log</div>', unsafe_allow_html=True)
    st.markdown('<p class="page-title">Full <em>Prediction History</em></p>', unsafe_allow_html=True)

    if not records:
        st.info("No predictions logged yet.")
        st.stop()

    df = records_to_df(records)

    # Filters
    f1, f2, f3 = st.columns(3)
    with f1:
        drug_filter = st.selectbox("Filter by drug", ["All"] + sorted(df["Drug"].unique().tolist()))
    with f2:
        sev_filter = st.selectbox("Filter by severity", ["All", "High (8-10)", "Medium (5-7)", "Low (1-4)"])
    with f3:
        habit_filter = st.selectbox("Habit forming", ["All", "Yes", "No"])

    filtered = df.copy()
    if drug_filter  != "All": filtered = filtered[filtered["Drug"] == drug_filter]
    if habit_filter != "All": filtered = filtered[filtered["Habit Forming"] == habit_filter]
    if sev_filter   == "High (8-10)":   filtered = filtered[filtered["Severity Score"] >= 8]
    elif sev_filter == "Medium (5-7)":  filtered = filtered[(filtered["Severity Score"] >= 5) & (filtered["Severity Score"] < 8)]
    elif sev_filter == "Low (1-4)":     filtered = filtered[filtered["Severity Score"] < 5]

    st.caption(f"Showing {len(filtered)} of {len(df)} records")
    st.markdown("")

    # Detailed log cards
    for _, row in filtered.sort_values("ID", ascending=False).iterrows():
        sev   = severity_label(row["Severity Score"])
        s_cls = {"High": "sev-H", "Medium": "sev-M", "Low": "sev-L"}[sev]
        ts    = pd.to_datetime(row["Timestamp"]).strftime("%d %b %Y, %I:%M %p")
        st.markdown(f"""
        <div class="log-row">
            <div style="display:flex;justify-content:space-between;align-items:center">
                <div>
                    <span class="log-drug">#{int(row['ID'])} — {row['Drug']}</span>
                    <span class="log-meta"> · {row['Therapeutic']} · {row['Chemical']}</span>
                </div>
                <span class="sev-pill {s_cls}">{sev} risk · {row['Severity Score']}/10</span>
            </div>
            <div class="log-meta" style="margin-top:6px">
                👤 {row['Patient']} · Age {row['Age']} · {row['Weight (kg)']} kg &nbsp;|&nbsp;
                💊 {row['Effects Found']} effects found &nbsp;|&nbsp;
                🕒 {ts}
            </div>
            <div style="font-size:12px;color:#4B5563;margin-top:6px;line-height:1.5">{row['Summary']}</div>
        </div>
        """, unsafe_allow_html=True)

    # CSV Export
    st.markdown('<p class="sec-header">Export data</p>', unsafe_allow_html=True)
    csv_buf = io.StringIO()
    filtered.to_csv(csv_buf, index=False)
    st.download_button(
        label="⬇️ Download filtered data as CSV",
        data=csv_buf.getvalue(),
        file_name=f"predictions_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MEDICINE INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💊 Medicine Insights":
    st.markdown('<div class="admin-badge">Admin · Insights</div>', unsafe_allow_html=True)
    st.markdown('<p class="page-title">Medicine <em>Insights</em></p>', unsafe_allow_html=True)

    if not records:
        st.info("No predictions logged yet.")
        st.stop()

    df = records_to_df(records)

    # Most dangerous drugs (avg severity)
    st.markdown('<p class="sec-header">Drugs by average severity score</p>', unsafe_allow_html=True)
    drug_sev = df.groupby("Drug")["Severity Score"].mean().round(1).sort_values(ascending=False).reset_index()
    drug_sev.columns = ["Drug", "Avg Severity Score"]
    st.dataframe(drug_sev, use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="sec-header">Most queried therapeutic classes</p>', unsafe_allow_html=True)
        tc = df["Therapeutic"].value_counts().reset_index()
        tc.columns = ["Class", "Queries"]
        st.dataframe(tc, use_container_width=True, hide_index=True)

    with col2:
        st.markdown('<p class="sec-header">Habit-forming vs non habit-forming</p>', unsafe_allow_html=True)
        hf = df["Habit Forming"].value_counts().reset_index()
        hf.columns = ["Habit Forming", "Count"]
        st.bar_chart(hf.set_index("Habit Forming")["Count"], use_container_width=True, height=200)

    # Age group analysis
    st.markdown('<p class="sec-header">Age group breakdown</p>', unsafe_allow_html=True)
    df["Age Group"] = pd.cut(df["Age"], bins=[0,12,17,30,45,60,120],
                             labels=["Child (0-12)","Teen (13-17)","Young Adult (18-30)",
                                     "Adult (31-45)","Middle-aged (46-60)","Senior (60+)"])
    age_grp = df["Age Group"].value_counts().reset_index()
    age_grp.columns = ["Age Group", "Count"]
    st.bar_chart(age_grp.set_index("Age Group")["Count"], use_container_width=True, height=220)

    # Auto-generated insights
    st.markdown('<p class="sec-header">Auto-generated insights</p>', unsafe_allow_html=True)
    top_drug    = df["Drug"].value_counts().idxmax() if not df.empty else "N/A"
    top_thera   = df["Therapeutic"].value_counts().idxmax() if not df.empty else "N/A"
    high_sev_pct= round(len(df[df["Severity Score"] >= 8]) / len(df) * 100) if not df.empty else 0
    avg_age     = round(df["Age"].mean(), 1) if not df.empty else "N/A"

    insights = [
        f"💊 <strong>{top_drug}</strong> is the most frequently searched medicine ({df['Drug'].value_counts().max()} queries).",
        f"🏥 <strong>{top_thera}</strong> is the most queried therapeutic class, suggesting this is the dominant patient concern.",
        f"⚠️ <strong>{high_sev_pct}%</strong> of all predictions returned a High severity score (8+), indicating serious risk profiles are common.",
        f"👤 The average patient age querying this system is <strong>{avg_age} years</strong>.",
        f"🔴 <strong>{df[df['Habit Forming']=='Yes'].shape[0]}</strong> queries involved habit-forming medicines — consider flagging these in the patient UI.",
    ]
    for ins in insights:
        st.markdown(f'<div class="insight-card"><p>{ins}</p></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SETTINGS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚙️ Settings":
    st.markdown('<div class="admin-badge">Admin · Settings</div>', unsafe_allow_html=True)
    st.markdown('<p class="page-title">System <em>Settings</em></p>', unsafe_allow_html=True)

    st.markdown('<p class="sec-header">Credentials</p>', unsafe_allow_html=True)
    st.info(f"Current admin username: **{ADMIN_USERNAME}**  \nTo change credentials, edit `ADMIN_USERNAME` and `ADMIN_PASSWORD` at the top of `admin.py`.")

    st.markdown('<p class="sec-header">Data management</p>', unsafe_allow_html=True)
    total = len(records)
    st.metric("Total records stored", total)

    col1, col2 = st.columns(2)
    with col1:
        # Full export
        if records:
            df  = records_to_df(records)
            buf = io.StringIO()
            df.to_csv(buf, index=False)
            st.download_button(
                "⬇️ Export all data (CSV)",
                data=buf.getvalue(),
                file_name=f"all_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.button("⬇️ Export all data (CSV)", disabled=True, use_container_width=True)

    with col2:
        # JSON export
        if records:
            st.download_button(
                "⬇️ Export all data (JSON)",
                data=json.dumps(records, indent=2),
                file_name=f"all_predictions_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json",
                use_container_width=True,
            )

    st.markdown('<p class="sec-header">Danger zone</p>', unsafe_allow_html=True)
    st.warning("⚠️ Clearing all records is permanent and cannot be undone.")
    confirm = st.checkbox("I understand this will permanently delete all prediction logs")
    if confirm:
        st.markdown('<div class="danger-btn">', unsafe_allow_html=True)
        if st.button("🗑️ Clear all records", use_container_width=True):
            clear_all()
            st.success("All records cleared.")
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<p class="sec-header">System info</p>', unsafe_allow_html=True)
    st.markdown(f"""
    | Property | Value |
    |---|---|
    | Log file | `predictions_log.json` |
    | Admin username | `{ADMIN_USERNAME}` |
    | Total records | `{total}` |
    | App link | http://localhost:8501 |
    | Admin link | http://localhost:8502 |
    """)

st.markdown("---")
st.caption("Medicine AI · Admin Panel · Built with Streamlit")
