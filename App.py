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
# 26. REPORT EXPORT
# ============================================================

st.header(
    "Report Export"
)

executive_report = pd.DataFrame({

    "Metric": [

        "Portfolio Total Return",

        "Annualized Return",

        "Annualized Volatility",

        "Maximum Drawdown",

        "Current Drawdown",

        "Portfolio Liquidity Stress",

        "Top 5 Risk Concentration",

        "Average Absolute Correlation",

        "Active Alerts",

        "Critical Alerts",

        "High Alerts",

        "Portfolio Alert Status"
    ],

    "Value": [

        performance[
            "total_return"
        ],

        performance[
            "annualized_return"
        ],

        performance[
            "annualized_volatility"
        ],

        performance[
            "maximum_drawdown"
        ],

        performance[
            "current_drawdown"
        ],

        executive[
            "liquidity_score"
        ],

        executive[
            "top_5_risk_share"
        ],

        executive[
            "average_absolute_correlation"
        ],

        executive[
            "active_alerts"
        ],

        executive[
            "critical_alerts"
        ],

        executive[
            "high_alerts"
        ],

        executive[
            "portfolio_alert_status"
        ]
    ]
})


csv_report = (
    executive_report
    .to_csv(
        index=False
    )
)

st.download_button(
    label="Download Executive Report",
    data=csv_report,
    file_name=(
        "indigo_market_intelligence_report.csv"
    ),
    mime="text/csv",
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
