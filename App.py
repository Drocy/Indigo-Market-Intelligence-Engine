# ============================================================
# INDIGO MARKET INTELLIGENCE
# Streamlit Application — V1
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from intelligence_engine import (
    ASSET_METADATA,
    ASSETS,
    run_indigo_engine
)


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Indigo Market Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. APPLICATION HEADER
# ============================================================

st.title("Indigo Market Intelligence")

st.markdown(
    """
    **Institutional Market & Portfolio Intelligence**

    A decision-support engine for detecting anomalies, monitoring
    market regimes, analysing portfolio risk and identifying
    conditions requiring institutional attention.
    """
)

st.divider()


# ============================================================
# 3. SIDEBAR
# ============================================================

st.sidebar.header("Analysis Controls")

period = st.sidebar.selectbox(
    "Historical Analysis Period",
    options=[
        "1y",
        "2y",
        "3y",
        "5y"
    ],
    index=1
)

st.sidebar.subheader("Asset Universe")

selected_assets = st.sidebar.multiselect(
    "Assets to Analyse",
    options=ASSETS,
    default=ASSETS
)

st.sidebar.subheader("Portfolio")

default_portfolio = [
    "MSFT",
    "AAPL",
    "NVDA",
    "AMZN",
    "GOOGL",
    "TSLA",
    "BTC-USD",
    "ETH-USD"
]

selected_portfolio_assets = st.sidebar.multiselect(
    "Portfolio Assets",
    options=[
        asset
        for asset in selected_assets
    ],
    default=[
        asset
        for asset in default_portfolio
        if asset in selected_assets
    ]
)


# ============================================================
# 4. PORTFOLIO WEIGHTS
# ============================================================

if selected_portfolio_assets:

    st.sidebar.subheader(
        "Portfolio Weights"
    )

    equal_weight = (
        1 /
        len(selected_portfolio_assets)
    )

    portfolio = {}

    for ticker in selected_portfolio_assets:

        portfolio[ticker] = st.sidebar.number_input(
            f"{ticker} Weight",
            min_value=0.0,
            max_value=1.0,
            value=float(
                round(
                    equal_weight,
                    4
                )
            ),
            step=0.01
        )

else:

    portfolio = {}


# ============================================================
# 5. VALIDATE PORTFOLIO
# ============================================================

portfolio_weight_sum = (
    sum(portfolio.values())
    if portfolio
    else 0
)

if portfolio:

    st.sidebar.write(
        f"**Total weight:** "
        f"{portfolio_weight_sum:.2%}"
    )

    if not np.isclose(
        portfolio_weight_sum,
        1.0,
        atol=0.001
    ):

        st.sidebar.warning(
            "Portfolio weights must sum to 100%."
        )


# ============================================================
# 6. RUN ENGINE
# ============================================================

run_analysis = st.sidebar.button(
    "Run Market Intelligence",
    type="primary",
    use_container_width=True
)


# ============================================================
# 7. INITIAL STATE
# ============================================================

if (
    "engine_results"
    not in st.session_state
):

    st.session_state[
        "engine_results"
    ] = None


# ============================================================
# 8. ENGINE EXECUTION
# ============================================================

if run_analysis:

    if not selected_assets:

        st.error(
            "Select at least one asset."
        )

        st.stop()

    if not portfolio:

        st.error(
            "Select at least one portfolio asset."
        )

        st.stop()

    if not np.isclose(
        portfolio_weight_sum,
        1.0,
        atol=0.001
    ):

        st.error(
            "Portfolio weights must sum to 100%."
        )

        st.stop()

    with st.spinner(
        "Running Indigo Market Intelligence Engine..."
    ):

        try:

            results = run_indigo_engine(
                assets=selected_assets,
                period=period,
                portfolio=portfolio
            )

            st.session_state[
                "engine_results"
            ] = results

            st.success(
                "Market intelligence analysis completed."
            )

        except Exception as e:

            st.error(
                "The analysis could not be completed."
            )

            st.exception(e)

            st.stop()


# ============================================================
# 9. CHECK ENGINE STATE
# ============================================================

results = st.session_state[
    "engine_results"
]

if results is None:

    st.info(
        "Configure the analysis using the sidebar "
        "and click **Run Market Intelligence**."
    )

    st.stop()


# ============================================================
# 10. EXTRACT RESULTS
# ============================================================

validated_analysis = (
    results[
        "validated_analysis"
    ]
)

portfolio_weights = (
    results[
        "portfolio_weights"
    ]
)

portfolio_returns = (
    results[
        "portfolio_returns"
    ]
)

performance = (
    results[
        "performance"
    ]
)

risk_contribution = (
    results[
        "risk_contribution"
    ]
)

correlation = (
    results[
        "correlation"
    ]
)

current_liquidity = (
    results[
        "current_liquidity"
    ]
)

portfolio_liquidity = (
    results[
        "portfolio_liquidity"
    ]
)

alerts = (
    results[
        "alerts"
    ]
)

executive = (
    results[
        "executive"
    ]
)


# ============================================================
# 11. EXECUTIVE STATUS
# ============================================================

st.header(
    "Executive Overview"
)

status = executive[
    "portfolio_alert_status"
]


if status == "CRITICAL ATTENTION REQUIRED":

    status_icon = "🔴"

elif status == "HIGH ATTENTION":

    status_icon = "🟠"

elif status == "MONITOR":

    status_icon = "🟡"

else:

    status_icon = "🟢"


st.markdown(
    f"""
    ### {status_icon} {status}

    **Analysis period:** {period}

    **Assets analysed:** {len(validated_analysis)}
    """
)


# ============================================================
# 12. KEY PERFORMANCE METRICS
# ============================================================

metric_1, metric_2, metric_3, metric_4 = (
    st.columns(4)
)

metric_1.metric(
    "Portfolio Return",
    f"{performance['total_return']:.2%}"
)

metric_2.metric(
    "Annualised Volatility",
    f"{performance['annualized_volatility']:.2%}"
)

metric_3.metric(
    "Maximum Drawdown",
    f"{performance['maximum_drawdown']:.2%}"
)

metric_4.metric(
    "Liquidity Stress",
    f"{executive['liquidity_score']:.1f}/100"
)


# ============================================================
# 13. ALERT METRICS
# ============================================================

st.subheader(
    "Intelligence Alerts"
)

alert_col_1, alert_col_2, alert_col_3 = (
    st.columns(3)
)

alert_col_1.metric(
    "Active Alerts",
    executive[
        "active_alerts"
    ]
)

alert_col_2.metric(
    "High Alerts",
    executive[
        "high_alerts"
    ]
)

alert_col_3.metric(
    "Critical Alerts",
    executive[
        "critical_alerts"
    ]
)


# ============================================================
# 14. PORTFOLIO PERFORMANCE CHART
# ============================================================

st.header(
    "Portfolio Performance"
)

performance_chart = pd.DataFrame({

    "Cumulative Return":
        performance[
            "cumulative"
        ],

    "Drawdown":
        performance[
            "drawdown"
        ]

})

tab_perf_1, tab_perf_2 = st.tabs(
    [
        "Cumulative Performance",
        "Drawdown"
    ]
)

with tab_perf_1:

    st.line_chart(
        performance[
            "cumulative"
        ],
        use_container_width=True
    )

with tab_perf_2:

    st.line_chart(
        performance[
            "drawdown"
        ],
        use_container_width=True
    )


# ============================================================
# 15. RISK CONTRIBUTION
# ============================================================

st.header(
    "Portfolio Risk Contribution"
)

risk_display = (
    risk_contribution
    .copy()
)

risk_display[
    "Weight"
] = (
    risk_display[
        "Weight"
    ].map(
        lambda x:
        f"{x:.2%}"
    )
)

risk_display[
    "Risk_Contribution"
] = (
    risk_contribution[
        "Risk_Contribution"
    ].map(
        lambda x:
        f"{x:.2%}"
    )
)

risk_display[
    "Risk_Multiple"
] = (
    risk_contribution[
        "Risk_Multiple"
    ].map(
        lambda x:
        f"{x:.2f}x"
    )
)

risk_display[
    "Risk_Overweight"
] = (
    risk_contribution[
        "Risk_Overweight"
    ].map(
        lambda x:
        f"{x:.2%}"
    )
)

st.dataframe(
    risk_display,
    use_container_width=True
)


# ============================================================
# 16. RISK CONTRIBUTION CHART
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 5)
)

ax.bar(
    risk_contribution.index,
    risk_contribution[
        "Risk_Contribution"
    ]
)

ax.set_title(
    "Portfolio Risk Contribution"
)

ax.set_ylabel(
    "Risk Contribution"
)

ax.tick_params(
    axis="x",
    rotation=45
)

plt.tight_layout()

st.pyplot(
    fig
)

plt.close(fig)


# ============================================================
# 17. ASSET INTELLIGENCE
# ============================================================

st.header(
    "Asset Intelligence"
)

asset_rows = []

for ticker, df in (
    validated_analysis.items()
):

    latest = df.iloc[-1]

    asset_rows.append({

        "Ticker":
            ticker,

        "Asset":
            ASSET_METADATA[
                ticker
            ][
                "name"
            ],

        "Class":
            ASSET_METADATA[
                ticker
            ][
                "class"
            ],

        "Price":
            latest[
                "Close"
            ],

        "Return":
            latest[
                "Return"
            ],

        "Volatility":
            latest[
                "Rolling_Volatility_20D"
            ],

        "Drawdown":
            latest[
                "Drawdown"
            ],

        "Market Regime":
            latest[
                "Market_Regime"
            ],

        "Evidence Score":
            latest[
                "Final_Evidence_Score"
            ],

        "Intelligence Level":
            latest[
                "Intelligence_Level"
            ]
    })

asset_intelligence = pd.DataFrame(
    asset_rows
)

st.dataframe(
    asset_intelligence,
    use_container_width=True
)


# ============================================================
# 18. ASSET DETAIL
# ============================================================

st.subheader(
    "Asset Detail"
)

asset_selector = st.selectbox(
    "Select Asset",
    options=list(
        validated_analysis.keys()
    )
)

selected_df = (
    validated_analysis[
        asset_selector
    ]
)


detail_col_1, detail_col_2, detail_col_3 = (
    st.columns(3)
)

latest = selected_df.iloc[-1]

detail_col_1.metric(
    "Current Price",
    f"{latest['Close']:,.2f}"
)

detail_col_2.metric(
    "Evidence Score",
    f"{latest['Final_Evidence_Score']:.0f}"
)

detail_col_3.metric(
    "Intelligence Level",
    latest[
        "Intelligence_Level"
    ]
)


st.line_chart(
    selected_df.set_index(
        "Date"
    )[
        "Close"
    ],
    use_container_width=True
)


# ============================================================
# 19. ANOMALY MONITOR
# ============================================================

st.header(
    "Anomaly Monitor"
)

anomaly_columns = [

    "Return_Anomaly",

    "Volume_Anomaly",

    "Statistical_Anomaly",

    "GARCH_Anomaly",

    "Multivariate_Anomaly",

    "Relative_Anomaly",

    "Correlation_Break"
]

available_anomaly_columns = [
    col
    for col in anomaly_columns
    if col in asset_intelligence.columns
]


anomaly_rows = []

for ticker, df in (
    validated_analysis.items()
):

    latest = df.iloc[-1]

    row = {
        "Ticker": ticker
    }

    for column in anomaly_columns:

        if column in latest.index:

            row[column] = int(
                latest[column]
                if pd.notna(
                    latest[column]
                )
                else 0
            )

    anomaly_rows.append(row)


anomaly_table = pd.DataFrame(
    anomaly_rows
)

st.dataframe(
    anomaly_table,
    use_container_width=True
)


# ============================================================
# 20. CORRELATION ANALYSIS
# ============================================================

st.header(
    "Cross-Asset Intelligence"
)

st.subheader(
    "Correlation Matrix"
)

st.dataframe(
    correlation[
        "matrix"
    ].round(3),
    use_container_width=True
)


st.subheader(
    "Highest Correlation Relationships"
)

high_corr = (
    correlation[
        "pairs"
    ]
    .head(10)
    .copy()
)

st.dataframe(
    high_corr,
    use_container_width=True
)


# ============================================================
# 21. LIQUIDITY INTELLIGENCE
# ============================================================

st.header(
    "Liquidity & Stress Intelligence"
)

liquidity_display = (
    current_liquidity
    .copy()
)

st.dataframe(
    liquidity_display,
    use_container_width=True
)


fig, ax = plt.subplots(
    figsize=(10, 5)
)

ax.bar(
    current_liquidity[
        "Ticker"
    ],
    current_liquidity[
        "Liquidity_Stress"
    ]
)

ax.axhline(
    50,
    linestyle="--"
)

ax.set_title(
    "Current Liquidity Stress by Asset"
)

ax.set_ylabel(
    "Stress Score"
)

ax.tick_params(
    axis="x",
    rotation=45
)

plt.tight_layout()

st.pyplot(
    fig
)

plt.close(fig)


# ============================================================
# 22. PORTFOLIO LIQUIDITY
# ============================================================

st.subheader(
    "Portfolio Liquidity Priorities"
)

st.metric(
    "Portfolio Liquidity Stress",
    f"{portfolio_liquidity['score']:.1f}/100"
)

st.dataframe(
    portfolio_liquidity[
        "priorities"
    ],
    use_container_width=True
)


# ============================================================
# 23. ALERT CENTRE
# ============================================================

st.header(
    "Institutional Alert Centre"
)

active_alerts = (
    alerts[
        "active_alerts"
    ]
)

if active_alerts.empty:

    st.success(
        "No active market intelligence alerts."
    )

else:

    severity_filter = st.multiselect(
        "Filter by Alert Severity",
        options=[
            "CRITICAL",
            "HIGH",
            "MEDIUM",
            "LOW"
        ],
        default=[
            "CRITICAL",
            "HIGH",
            "MEDIUM"
        ]
    )

    filtered_alerts = (
        active_alerts[
            active_alerts[
                "Alert_Severity"
            ].isin(
                severity_filter
            )
        ]
    )

    for _, row in (
        filtered_alerts.iterrows()
    ):

        if row[
            "Alert_Severity"
        ] == "CRITICAL":

            st.error(
                row[
                    "Alert_Message"
                ]
            )

        elif row[
            "Alert_Severity"
        ] == "HIGH":

            st.warning(
                row[
                    "Alert_Message"
                ]
            )

        else:

            st.info(
                row[
                    "Alert_Message"
                ]
            )


# ============================================================
# 24. EXECUTIVE RISK SUMMARY
# ============================================================

st.header(
    "Executive Intelligence"
)

st.subheader(
    "Principal Risks"
)

for risk in (
    executive[
        "principal_risks"
    ]
):

    st.markdown(
        f"- {risk}"
    )


st.subheader(
    "Decision Attention Queue"
)

for action in (
    executive[
        "decision_actions"
    ]
):

    st.markdown(
        f"- {action}"
    )


# ============================================================
# 25. TOP RISK CONTRIBUTORS
# ============================================================

st.subheader(
    "Top Risk Contributors"
)

top_risks = (
    executive[
        "top_risk_contributors"
    ]
    .copy()
)

st.dataframe(
    top_risks,
    use_container_width=True
)


# ============================================================
# 26. INDIGO INSTITUTIONAL EXECUTIVE REPORT
# ============================================================

st.header("Indigo Institutional Intelligence Report")


# ------------------------------------------------------------
# 26.1 Helper functions
# ------------------------------------------------------------

def fmt_pct(value):
    """Format decimal as percentage."""
    if pd.isna(value):
        return "N/A"
    return f"{value:.2%}"


def fmt_score(value):
    """Format score."""
    if pd.isna(value):
        return "N/A"
    return f"{value:.1f}/100"


def classify_drawdown(value):
    """Classify drawdown severity."""
    value = abs(float(value))

    if value >= 0.30:
        return "Severe"
    elif value >= 0.20:
        return "Material"
    elif value >= 0.10:
        return "Moderate"
    else:
        return "Contained"


def classify_volatility(value):
    """Classify annualised volatility."""
    value = float(value)

    if value >= 0.40:
        return "Very High"
    elif value >= 0.25:
        return "High"
    elif value >= 0.15:
        return "Moderate"
    else:
        return "Low"


def classify_concentration(value):
    """Classify top-five risk concentration."""
    value = float(value)

    if value >= 0.80:
        return "Very High"
    elif value >= 0.70:
        return "High"
    elif value >= 0.50:
        return "Moderate"
    else:
        return "Contained"


def classify_liquidity(value):
    """Classify liquidity stress."""
    value = float(value)

    if value >= 75:
        return "Severe"
    elif value >= 50:
        return "Elevated"
    elif value >= 25:
        return "Moderate"
    else:
        return "Contained"


# ------------------------------------------------------------
# 26.2 Extract core metrics safely
# ------------------------------------------------------------

total_return = performance.get(
    "total_return",
    np.nan
)

annualized_return = performance.get(
    "annualized_return",
    np.nan
)

annualized_volatility = performance.get(
    "annualized_volatility",
    np.nan
)

maximum_drawdown = performance.get(
    "maximum_drawdown",
    np.nan
)

current_drawdown = performance.get(
    "current_drawdown",
    np.nan
)

liquidity_score = executive.get(
    "liquidity_score",
    np.nan
)

top_5_risk_share = executive.get(
    "top_5_risk_share",
    np.nan
)

average_absolute_correlation = executive.get(
    "average_absolute_correlation",
    np.nan
)

active_alert_count = executive.get(
    "active_alerts",
    0
)

critical_alert_count = executive.get(
    "critical_alerts",
    0
)

high_alert_count = executive.get(
    "high_alerts",
    0
)

portfolio_status = executive.get(
    "portfolio_alert_status",
    "MONITOR"
)


# ------------------------------------------------------------
# 26.3 Derived assessments
# ------------------------------------------------------------

drawdown_assessment = classify_drawdown(
    maximum_drawdown
)

volatility_assessment = classify_volatility(
    annualized_volatility
)

concentration_assessment = classify_concentration(
    top_5_risk_share
)

liquidity_assessment = classify_liquidity(
    liquidity_score
)


# ------------------------------------------------------------
# 26.4 Determine principal structural concern
# ------------------------------------------------------------

risk_factors = {
    "Portfolio concentration":
        top_5_risk_share
        if pd.notna(top_5_risk_share)
        else 0,

    "Maximum drawdown":
        abs(maximum_drawdown)
        if pd.notna(maximum_drawdown)
        else 0,

    "Annualised volatility":
        annualized_volatility
        if pd.notna(annualized_volatility)
        else 0,

    "Liquidity stress":
        liquidity_score / 100
        if pd.notna(liquidity_score)
        else 0
}

principal_risk = max(
    risk_factors,
    key=risk_factors.get
)


# ------------------------------------------------------------
# 26.5 Executive Assessment
# ------------------------------------------------------------

st.subheader(
    "1. Executive Assessment"
)

if portfolio_status == "CRITICAL ATTENTION REQUIRED":

    assessment_icon = "🔴"

elif portfolio_status == "HIGH ATTENTION":

    assessment_icon = "🟠"

elif portfolio_status == "MONITOR":

    assessment_icon = "🟡"

else:

    assessment_icon = "🟢"


st.markdown(
    f"""
### {assessment_icon} Portfolio Status: {portfolio_status}
"""
)


assessment_text = (
    f"The portfolio generated a total return of "
    f"**{fmt_pct(total_return)}** over the analysis period, "
    f"with annualised volatility of "
    f"**{fmt_pct(annualized_volatility)}**, classified as "
    f"**{volatility_assessment.lower()}**. "
)

if pd.notna(maximum_drawdown):

    assessment_text += (
        f"Maximum drawdown reached "
        f"**{fmt_pct(maximum_drawdown)}**, representing a "
        f"**{drawdown_assessment.lower()}** level of downside exposure. "
    )

if pd.notna(top_5_risk_share):

    assessment_text += (
        f"The five largest risk contributors account for "
        f"**{fmt_pct(top_5_risk_share)}** of portfolio risk, "
        f"indicating **{concentration_assessment.lower()} "
        f"risk concentration**. "
    )

if pd.notna(liquidity_score):

    assessment_text += (
        f"Portfolio liquidity stress is assessed at "
        f"**{fmt_score(liquidity_score)}**, indicating "
        f"**{liquidity_assessment.lower()} liquidity pressure**. "
    )


st.write(
    assessment_text
)


# ------------------------------------------------------------
# 26.6 Core Risk Dashboard
# ------------------------------------------------------------

st.subheader(
    "2. Performance & Risk Assessment"
)

report_metric_1, report_metric_2, report_metric_3, report_metric_4 = (
    st.columns(4)
)

report_metric_1.metric(
    "Total Return",
    fmt_pct(total_return)
)

report_metric_2.metric(
    "Annualised Return",
    fmt_pct(annualized_return)
)

report_metric_3.metric(
    "Annualised Volatility",
    fmt_pct(annualized_volatility)
)

report_metric_4.metric(
    "Maximum Drawdown",
    fmt_pct(maximum_drawdown)
)


risk_assessment_table = pd.DataFrame({

    "Risk Dimension": [

        "Volatility",

        "Maximum Drawdown",

        "Risk Concentration",

        "Liquidity Stress",

        "Average Absolute Correlation"
    ],

    "Observed Value": [

        fmt_pct(annualized_volatility),

        fmt_pct(maximum_drawdown),

        fmt_pct(top_5_risk_share),

        fmt_score(liquidity_score),

        fmt_pct(
            average_absolute_correlation
        )
    ],

    "Assessment": [

        volatility_assessment,

        drawdown_assessment,

        concentration_assessment,

        liquidity_assessment,

        (
            "Elevated"
            if average_absolute_correlation >= 0.50
            else
            "Moderate"
            if average_absolute_correlation >= 0.30
            else
            "Contained"
        )
    ]
})


st.dataframe(
    risk_assessment_table,
    use_container_width=True,
    hide_index=True
)


# ------------------------------------------------------------
# 26.7 Structural Risk
# ------------------------------------------------------------

st.subheader(
    "3. Structural Risk Assessment"
)

st.markdown(
    f"""
**Principal structural concern: {principal_risk}**
"""
)


if concentration_assessment in [
    "High",
    "Very High"
]:

    st.warning(
        f"Portfolio risk is materially concentrated. "
        f"The five largest contributors represent "
        f"{fmt_pct(top_5_risk_share)} of total portfolio risk. "
        f"This creates sensitivity to a relatively small "
        f"number of underlying positions."
    )

else:

    st.success(
        "Portfolio risk concentration is currently "
        "within a relatively contained range."
    )


# ------------------------------------------------------------
# 26.8 Top Risk Contributors
# ------------------------------------------------------------

st.markdown(
    "**Largest Contributors to Portfolio Risk**"
)

top_risk_table = (
    risk_contribution
    .sort_values(
        "Risk_Contribution",
        ascending=False
    )
    .head(10)
    .copy()
)

if not top_risk_table.empty:

    display_top_risk = top_risk_table.copy()

    if "Weight" in display_top_risk.columns:

        display_top_risk[
            "Weight"
        ] = display_top_risk[
            "Weight"
        ].map(
            lambda x:
            f"{x:.2%}"
        )

    if "Risk_Contribution" in display_top_risk.columns:

        display_top_risk[
            "Risk_Contribution"
        ] = display_top_risk[
            "Risk_Contribution"
        ].map(
            lambda x:
            f"{x:.2%}"
        )

    if "Risk_Multiple" in display_top_risk.columns:

        display_top_risk[
            "Risk_Multiple"
        ] = display_top_risk[
            "Risk_Multiple"
        ].map(
            lambda x:
            f"{x:.2f}x"
        )

    st.dataframe(
        display_top_risk,
        use_container_width=True
    )


# ------------------------------------------------------------
# 26.9 Liquidity Assessment
# ------------------------------------------------------------

st.subheader(
    "4. Liquidity & Market Stress"
)

liquidity_text = (
    f"Portfolio liquidity stress is currently assessed "
    f"at **{fmt_score(liquidity_score)}**. "
)

if liquidity_assessment == "Contained":

    liquidity_text += (
        "Current liquidity conditions do not represent "
        "a primary portfolio-level concern."
    )

elif liquidity_assessment == "Moderate":

    liquidity_text += (
        "Liquidity conditions warrant monitoring, "
        "particularly during periods of elevated market stress."
    )

elif liquidity_assessment == "Elevated":

    liquidity_text += (
        "Liquidity conditions represent a meaningful "
        "risk consideration for portfolio management."
    )

else:

    liquidity_text += (
        "Liquidity conditions represent a significant "
        "portfolio-level risk requiring close attention."
    )


st.write(
    liquidity_text
)


# ------------------------------------------------------------
# 26.10 Liquidity Priorities
# ------------------------------------------------------------

if (
    "priorities"
    in portfolio_liquidity
):

    liquidity_priorities = (
        portfolio_liquidity[
            "priorities"
        ]
    )

    if isinstance(
        liquidity_priorities,
        pd.DataFrame
    ) and not liquidity_priorities.empty:

        st.markdown(
            "**Assets requiring the greatest liquidity attention**"
        )

        st.dataframe(
            liquidity_priorities,
            use_container_width=True
        )


# ------------------------------------------------------------
# 26.11 Market Intelligence Alerts
# ------------------------------------------------------------

st.subheader(
    "5. Intelligence Alerts"
)

if active_alert_count == 0:

    st.success(
        "No active intelligence alerts were identified "
        "by the current monitoring framework."
    )

else:

    st.warning(
        f"{active_alert_count} active intelligence "
        f"alert(s) currently require monitoring."
    )

    if (
        isinstance(
            active_alerts,
            pd.DataFrame
        )
        and not active_alerts.empty
    ):

        alert_export = (
            active_alerts
            .copy()
        )

        st.dataframe(
            alert_export,
            use_container_width=True
        )


# ------------------------------------------------------------
# 26.12 Principal Risks
# ------------------------------------------------------------

st.subheader(
    "6. Principal Risks"

)

principal_risks = executive.get(
    "principal_risks",
    []
)

if principal_risks:

    for risk in principal_risks:

        st.markdown(
            f"- {risk}"
        )

else:

    st.write(
        "No principal risks were returned by the engine."
    )


# ------------------------------------------------------------
# 26.13 Decision Attention Queue
# ------------------------------------------------------------

st.subheader(
    "7. Decision-Attention Queue"
)

decision_actions = executive.get(
    "decision_actions",
    []
)

if decision_actions:

    for action in decision_actions:

        st.markdown(
            f"- {action}"
        )

else:

    st.write(
        "No immediate decision-attention items were generated."
    )


# ------------------------------------------------------------
# 26.14 Institutional Conclusion
# ------------------------------------------------------------

st.subheader(
    "8. Institutional Conclusion"
)

if principal_risk == "Portfolio concentration":

    conclusion = (
        "The principal issue identified by the current "
        "intelligence framework is portfolio concentration. "
        "The portfolio's overall risk profile is being driven "
        "disproportionately by a limited number of positions. "
        "Institutional attention should therefore focus on "
        "risk concentration, diversification and the "
        "behaviour of the dominant risk contributors."
    )

elif principal_risk == "Maximum drawdown":

    conclusion = (
        "The principal issue identified by the current "
        "intelligence framework is downside exposure. "
        "Historical drawdown indicates that the portfolio "
        "has experienced material capital impairment during "
        "the analysis period. Monitoring should therefore "
        "focus on drawdown persistence, recovery conditions "
        "and the assets responsible for downside risk."
    )

elif principal_risk == "Annualised volatility":

    conclusion = (
        "The principal issue identified by the current "
        "intelligence framework is volatility. "
        "The portfolio is experiencing a relatively elevated "
        "level of return dispersion, increasing uncertainty "
        "around portfolio outcomes and risk-adjusted performance."
    )

else:

    conclusion = (
        "The principal issue identified by the current "
        "intelligence framework is liquidity stress. "
        "Institutional attention should focus on the assets "
        "exhibiting the greatest liquidity pressure and the "
        "potential impact of deteriorating market conditions."
    )


st.info(
    conclusion
)


# ============================================================
# 26.15 MACHINE-READABLE EXECUTIVE EXPORT
# ============================================================

st.subheader(
    "Executive Report Export"
)


report_rows = [

    {
        "Section": "Executive Status",
        "Metric": "Portfolio Status",
        "Value": portfolio_status,
        "Assessment": portfolio_status
    },

    {
        "Section": "Performance",
        "Metric": "Total Return",
        "Value": fmt_pct(total_return),
        "Assessment": "Observed"
    },

    {
        "Section": "Performance",
        "Metric": "Annualised Return",
        "Value": fmt_pct(annualized_return),
        "Assessment": "Observed"
    },

    {
        "Section": "Risk",
        "Metric": "Annualised Volatility",
        "Value": fmt_pct(annualized_volatility),
        "Assessment": volatility_assessment
    },

    {
        "Section": "Risk",
        "Metric": "Maximum Drawdown",
        "Value": fmt_pct(maximum_drawdown),
        "Assessment": drawdown_assessment
    },

    {
        "Section": "Risk",
        "Metric": "Current Drawdown",
        "Value": fmt_pct(current_drawdown),
        "Assessment": "Current"
    },

    {
        "Section": "Structural Risk",
        "Metric": "Top-5 Risk Concentration",
        "Value": fmt_pct(top_5_risk_share),
        "Assessment": concentration_assessment
    },

    {
        "Section": "Liquidity",
        "Metric": "Portfolio Liquidity Stress",
        "Value": fmt_score(liquidity_score),
        "Assessment": liquidity_assessment
    },

    {
        "Section": "Market Structure",
        "Metric": "Average Absolute Correlation",
        "Value": fmt_pct(
            average_absolute_correlation
        ),
        "Assessment": (
            "Elevated"
            if average_absolute_correlation >= 0.50
            else
            "Moderate"
            if average_absolute_correlation >= 0.30
            else
            "Contained"
        )
    },

    {
        "Section": "Alerts",
        "Metric": "Active Alerts",
        "Value": active_alert_count,
        "Assessment": "Monitor"
    },

    {
        "Section": "Alerts",
        "Metric": "High Alerts",
        "Value": high_alert_count,
        "Assessment": "Priority"
    },

    {
        "Section": "Alerts",
        "Metric": "Critical Alerts",
        "Value": critical_alert_count,
        "Assessment": "Priority"
    },

    {
        "Section": "Executive Intelligence",
        "Metric": "Principal Risk",
        "Value": principal_risk,
        "Assessment": "Primary Attention Area"
    }
]


institutional_report = pd.DataFrame(
    report_rows
)


st.dataframe(
    institutional_report,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# 26.16 CSV DOWNLOAD
# ============================================================

csv_report = (
    institutional_report
    .to_csv(
        index=False
    )
)


st.download_button(
    label="Download Institutional Intelligence Report",
    data=csv_report,
    file_name=(
        "indigo_institutional_market_intelligence_report.csv"
    ),
    mime="text/csv",
    use_container_width=True
)


# ============================================================
# 26.17 FULL REPORT DOWNLOAD
# ============================================================

full_report_sections = []

full_report_sections.append(
    "INDIGO INSTITUTIONAL MARKET INTELLIGENCE REPORT"
)

full_report_sections.append(
    "=" * 60
)

full_report_sections.append(
    f"Portfolio Status: {portfolio_status}"
)

full_report_sections.append(
    ""
)

full_report_sections.append(
    "EXECUTIVE ASSESSMENT"
)

full_report_sections.append(
    assessment_text
)

full_report_sections.append(
    ""
)

full_report_sections.append(
    "PERFORMANCE & RISK"
)

full_report_sections.append(
    f"Total Return: {fmt_pct(total_return)}"
)

full_report_sections.append(
    f"Annualised Return: {fmt_pct(annualized_return)}"
)

full_report_sections.append(
    f"Annualised Volatility: {fmt_pct(annualized_volatility)}"
)

full_report_sections.append(
    f"Maximum Drawdown: {fmt_pct(maximum_drawdown)}"
)

full_report_sections.append(
    f"Current Drawdown: {fmt_pct(current_drawdown)}"
)

full_report_sections.append(
    ""
)

full_report_sections.append(
    "STRUCTURAL RISK"
)

full_report_sections.append(
    f"Principal Risk: {principal_risk}"
)

full_report_sections.append(
    f"Top-5 Risk Concentration: "
    f"{fmt_pct(top_5_risk_share)}"
)

full_report_sections.append(
    f"Average Absolute Correlation: "
    f"{fmt_pct(average_absolute_correlation)}"
)

full_report_sections.append(
    ""
)

full_report_sections.append(
    "LIQUIDITY"
)

full_report_sections.append(
    f"Portfolio Liquidity Stress: "
    f"{fmt_score(liquidity_score)}"
)

full_report_sections.append(
    f"Liquidity Assessment: "
    f"{liquidity_assessment}"
)

full_report_sections.append(
    ""
)

full_report_sections.append(
    "ALERTS"
)

full_report_sections.append(
    f"Active Alerts: {active_alert_count}"
)

full_report_sections.append(
    f"High Alerts: {high_alert_count}"
)

full_report_sections.append(
    f"Critical Alerts: {critical_alert_count}"
)

full_report_sections.append(
    ""
)

full_report_sections.append(
    "PRINCIPAL RISKS"
)

for risk in principal_risks:

    full_report_sections.append(
        f"- {risk}"
    )

full_report_sections.append(
    ""
)

full_report_sections.append(
    "DECISION-ATTENTION QUEUE"
)

for action in decision_actions:

    full_report_sections.append(
        f"- {action}"
    )

full_report_sections.append(
    ""
)

full_report_sections.append(
    "INSTITUTIONAL CONCLUSION"
)

full_report_sections.append(
    conclusion
)

full_report_sections.append(
    ""
)

full_report_sections.append(
    "Generated by Indigo Market Intelligence V1."
)

full_report_sections.append(
    "For analytical and decision-support purposes only."
)


full_report = "\n".join(
    full_report_sections
)


st.download_button(
    label="Download Executive Intelligence Brief",
    data=full_report,
    file_name=(
        "indigo_executive_intelligence_brief.txt"
    ),
    mime="text/plain",
    use_container_width=True
)


# ============================================================
# 27. FOOTER
# ============================================================

st.divider()

st.caption(
    """
    Indigo Market Intelligence V1

    Decision-support analytics for institutional
    market monitoring and risk intelligence.

    This system provides analytical intelligence and
    does not constitute investment advice.
    """
)
