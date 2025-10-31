# app.py — Olist Decision Dashboard (Single Page)
# -----------------------------------------------
# Requires:
#   - olist_clean_with_features.csv
#   - segments_summary.csv
#   - impact_models.csv
# Optional:
#   - ffd_logs.csv
#   - incidents.csv
# Run: streamlit run app.py

import os
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from datetime import timedelta

st.set_page_config(page_title="Olist Decision Dashboard", layout="wide")

# ---------- Helpers ----------
def read_csv_safe(path, **kw):
    if os.path.exists(path):
        return pd.read_csv(path, **kw)
    return None

@st.cache_data
def load_data():
    clean = read_csv_safe("olist_clean_with_features.csv", parse_dates=[
        "order_purchase_timestamp","order_delivered_customer_date","order_estimated_delivery_date"
    ])
    seg = read_csv_safe("segments_summary.csv")
    impact = read_csv_safe("impact_models.csv")
    ffd = read_csv_safe("ffd_logs.csv", parse_dates=["timestamp"])
    inc = read_csv_safe("incidents.csv", parse_dates=["opened_at"])

    # demo fallbacks
    if clean is None:
        st.warning("olist_clean_with_features.csv not found — using demo frame")
        dates = pd.date_range("2017-01-01", periods=400, freq="D")
        clean = pd.DataFrame({
            "order_id": np.arange(1,5001),
            "order_purchase_timestamp": np.random.choice(dates, 5000),
            "on_time": np.random.choice([1,1,1,0], 5000, p=[0.93,0.03,0.02,0.02]),
            "delivery_time_days": np.random.gamma(4.5, 2.2, 5000),
            "review_score_mean": np.random.choice([5,4,3,2,1],[5000],p=[0.5,0.25,0.15,0.07,0.03]),
            "gross_revenue": np.random.gamma(3, 50, 5000),
        })
        clean["order_estimated_delivery_date"] = clean["order_purchase_timestamp"] + pd.to_timedelta(np.random.randint(5,12,5000), unit="D")

    if seg is None:
        st.warning("segments_summary.csv not found — using demo segments")
        seg = pd.DataFrame({
            "سگمنت":["S1 مسیر سریع شهری / Urban Fast-Track","S2 مسیرهای پرریسک / Long-tail Risk States","S3 مشتریان تکراری / Repeat Loyalists","S4 مشتریان تازه‌وارد / Newcomers","S5 اقلام سنگین‌وزن / Heavy-Freight"],
            "قاعده (قابل تفسیر)":["state in {SP,RJ} & Top10","low OTD states & Top10","repeat_flag_90d=1","orders_count_90d=1","freight top 10%"],
            "اندازه":[2300,1800,1200,2600,900],
            "OTD":[95.2,92.0,94.8,94.2,95.6],
            "زمان تحویل (p90)":[11.8,12.3,14.0,14.8,20.1],
            "تکرار خرید ۹۰روزه":[22.0,14.0,41.0,0.0,9.0],
            "GMV/90d":[2_300_000,1_700_000,1_200_000,2_500_000,1_000_000],
            "بازی پیشنهادی (Play)":["Express if p90>7d","ETA+Suppress backorder","Coupon next order","First-order assurance","Carrier switch / weight cap"]
        })

    if impact is None:
        st.warning("impact_models.csv not found — using demo impact (base scenario)")
        impact = pd.DataFrame({
            "سگمنت": seg["سگمنت"].repeat(3).values,
            "سناریو": ["امیدی","پایه","بدبینانه"]*len(seg),
            "GMV پایه (۹۰ روز)": seg["GMV/90d"].repeat(3).values,
            "سفارش‌ها (۹۰ روز)": seg["اندازه"].repeat(3).values,
            "AOV": (seg["GMV/90d"]/seg["اندازه"]).repeat(3).round(2).values,
            "uplift_gmv_%":[6,3,0.5]*len(seg),
            "cost_per_order":[0.8,0.6,0.4]*len(seg),
            "take_rate":[0.12,0.12,0.12]*len(seg),
            "افزایش GMV (۹۰ روز)":0,
            "سهم پلتفرم از افزایش GMV":0,
            "هزینه کل":0,
            "خالص اثر بر پلتفرم":0
        })
        # compute base numbers
        def recalc(row):
            incr = row["GMV پایه (۹۰ روز)"] * (row["uplift_gmv_%"]/100)
            plat = incr * row["take_rate"]
            cost = row["سفارش‌ها (۹۰ روز)"] * row["cost_per_order"]
            net  = plat - cost
            return pd.Series({"افزایش GMV (۹۰ روز)":incr,"سهم پلتفرم از افزایش GMV":plat,"هزینه کل":cost,"خالص اثر بر پلتفرم":net})
        impact[["افزایش GMV (۹۰ روز)","سهم پلتفرم از افزایش GMV","هزینه کل","خالص اثر بر پلتفرم"]] = impact.apply(recalc, axis=1)

    if ffd is None:
        # demo 7-day logs
        st.info("ffd_logs.csv not found — generating demo logs")
        now = clean["order_purchase_timestamp"].max()
        ts = pd.date_range(now - pd.Timedelta(days=6), now, freq="H")
        nodes = ["Node1_high_risk","Node2_segment_geo","Node3_stable_hold"]
        ffd = pd.DataFrame({
            "timestamp": np.random.choice(ts, 600),
            "node": np.random.choice(nodes, 600, p=[0.35,0.45,0.20]),
            "guardrail_fired": np.random.choice([0,1], 600, p=[0.85,0.15])
        })

    if inc is None:
        # demo incidents
        inc = pd.DataFrame({
            "incident_id":[101,102],
            "opened_at":[clean["order_purchase_timestamp"].max()-pd.Timedelta(days=3),
                         clean["order_purchase_timestamp"].max()-pd.Timedelta(days=1)],
            "severity":["high","medium"],
            "title":["Carrier outage in PR","API throttle on checkout"],
            "status":["open","open"]
        })
    return clean, seg, impact, ffd, inc

clean, seg, impact, ffd, inc = load_data()

# ---------- Time window & baseline ----------
end_date = clean["order_purchase_timestamp"].max()
start_90 = end_date - pd.Timedelta(days=90)
df90 = clean[clean["order_purchase_timestamp"].between(start_90, end_date)].copy()

weekly = (df90
          .groupby(df90["order_purchase_timestamp"].dt.to_period("W").apply(lambda r: r.start_time))
          .agg(orders=("order_id","nunique"), otd=("on_time","mean"),
               review=("review_score_mean","mean"))
          .reset_index())
weekly["otd_pct"] = weekly["otd"]*100
weekly["review_rolling"] = weekly["review"].rolling(4, min_periods=1).mean()

baseline_otd = weekly["otd_pct"].mean()
baseline_p90 = np.nanpercentile(df90["delivery_time_days"].dropna(), 90)

# ---------- KPI Cards ----------
st.markdown("### کارت‌های KPI")
col1, col2, col3, col4 = st.columns(4)

# NSM: On-time delivered orders count (90d)
nsm_value = int(df90[df90["on_time"]==1]["order_id"].nunique())
col1.metric("North Star (On-time Orders, 90d)", f"{nsm_value:,}")

# OTD
col2.metric("OTD (90d)", f"{(df90['on_time'].mean()*100):.1f}%")

# p90
p90 = np.nanpercentile(df90["delivery_time_days"].dropna(), 90) if df90["delivery_time_days"].notna().any() else np.nan
col3.metric("Delivery Time p90 (days)", f"{p90:.1f}")

# Net GMV growth (base scenario)
impact_base = impact[impact["سناریو"]=="پایه"]
net_gain = impact_base["خالص اثر بر پلتفرم"].sum()
col4.metric("Net GMV Growth (90d, base)", f"${net_gain:,.0f}")

st.divider()

# ---------- Instability Panel ----------
st.markdown("### پنل ناپایداری (Instability panel)")
c1, c2 = st.columns([2,1])

with c1:
    st.caption("نوسان هفتگی OTD")
    chart_otd = alt.Chart(weekly).mark_line(point=True).encode(
        x=alt.X("order_purchase_timestamp:T", title="Week"),
        y=alt.Y("otd_pct:Q", title="OTD (%)"),
        tooltip=["order_purchase_timestamp","orders","otd_pct"]
    ).properties(height=240)
    st.altair_chart(chart_otd, use_container_width=True)

with c2:
    st.caption("افت امتیاز ریویو (قناری)")
    chart_rev = alt.Chart(weekly).mark_line(point=True).encode(
        x=alt.X("order_purchase_timestamp:T", title="Week"),
        y=alt.Y("review_rolling:Q", title="Review (4w rolling)"),
        tooltip=["order_purchase_timestamp","review_rolling"]
    ).properties(height=240)
    st.altair_chart(chart_rev, use_container_width=True)

st.caption("رویدادهای باز (incidents)")
open_inc = inc[inc["status"].str.lower()=="open"].copy()
open_inc = open_inc.sort_values("opened_at", ascending=False)
st.dataframe(open_inc, use_container_width=True, hide_index=True)

st.divider()

# ---------- Segments Table ----------
st.markdown("### جدول سگمنت‌ها (این فصل)")
# Join expected impact (base) per segment
base_impact = (impact_base[["سگمنت","خالص اثر بر پلتفرم"]]
               .groupby("سگمنت", as_index=False).sum()
               .rename(columns={"خالص اثر بر پلتفرم":"اثر ۹۰ روزه (خالص/پایه)"}))
seg_view = seg.merge(base_impact, on="سگمنت", how="left")

seg_view = seg_view.rename(columns={
    "اندازه":"اندازه (Orders)",
    "OTD":"OTD (%)",
    "تکرار خرید ۹۰روزه":"Repeat (%)",
    "بازی پیشنهادی (Play)":"اقدام برنامه‌ریزی‌شده",
    "اثر ۹۰ روزه (خالص/پایه)":"اثر ۹۰ روزهٔ مورد انتظار"
})
seg_cols = ["سگمنت","اندازه (Orders)","OTD (%)","Repeat (%)","اقدام برنامه‌ریزی‌شده","اثر ۹۰ روزهٔ مورد انتظار"]
st.dataframe(seg_view[seg_cols], use_container_width=True, hide_index=True)

st.divider()

# ---------- Policy Status ----------
st.markdown("### وضعیت سیاست (FFD Policy Status – last 7 days)")
last7 = ffd[ffd["timestamp"] >= (ffd["timestamp"].max() - pd.Timedelta(days=7))].copy()
node_counts = last7.groupby("node")["timestamp"].count().reset_index().rename(columns={"timestamp":"fires"})
gr_counts = last7.groupby("guardrail_fired")["timestamp"].count().reset_index().rename(columns={"timestamp":"count"})

cc1, cc2 = st.columns(2)
with cc1:
    st.caption("Fires by Node (7d)")
    st.dataframe(node_counts.sort_values("fires", ascending=False), use_container_width=True, hide_index=True)
with cc2:
    st.caption("Guardrails Tripped (7d)")
    gr_counts["label"] = gr_counts["guardrail_fired"].map({0:"No",1:"Yes"})
    st.dataframe(gr_counts[["label","count"]], use_container_width=True, hide_index=True)

st.info("راهنما: حباب بزرگ‌تر در سگمنت‌ها یعنی سهم GMV بیشتر؛ OTD بالاتر بهتر است؛ p90 پایین‌تر نشانهٔ عملیات چابک‌تر است.")
