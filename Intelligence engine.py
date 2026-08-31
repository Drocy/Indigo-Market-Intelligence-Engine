# ============================================================
# INDIGO MARKET INTELLIGENCE ENGINE — V1
# ============================================================
#
# Institutional Market & Portfolio Intelligence Engine
#
# Designed for:
#   - Public equities
#   - Crypto assets
#   - Portfolio risk analysis
#   - Statistical anomaly detection
#   - Volatility intelligence
#   - Market regime detection
#   - Cross-asset analysis
#   - Liquidity stress
#   - Institutional alert prioritisation
#
# Data source:
#   Yahoo Finance
#
# V1 philosophy:
#   Detect -> Contextualise -> Prioritise -> Report
#
# ============================================================


# ============================================================
# 1. IMPORTS
# ============================================================

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from scipy import stats
from scipy.spatial.distance import mahalanobis

from statsmodels.tsa.stattools import acf

from arch import arch_model


# ============================================================
# 2. ASSET UNIVERSE
# ============================================================

ASSET_METADATA = {

    # Equities
    "AAPL": {
        "name": "Apple",
        "class": "Equity",
        "benchmark": "^GSPC"
    },

    "MSFT": {
        "name": "Microsoft",
        "class": "Equity",
        "benchmark": "^GSPC"
    },

    "NVDA": {
        "name": "NVIDIA",
        "class": "Equity",
        "benchmark": "^GSPC"
    },

    "AMZN": {
        "name": "Amazon",
        "class": "Equity",
        "benchmark": "^GSPC"
    },

    "GOOGL": {
        "name": "Alphabet",
        "class": "Equity",
        "benchmark": "^GSPC"
    },

    "TSLA": {
        "name": "Tesla",
        "class": "Equity",
        "benchmark": "^GSPC"
    },

    # Crypto
    "BTC-USD": {
        "name": "Bitcoin",
        "class": "Crypto",
        "benchmark": "BTC-USD"
    },

    "ETH-USD": {
        "name": "Ethereum",
        "class": "Crypto",
        "benchmark": "BTC-USD"
    },

    "SOL-USD": {
        "name": "Solana",
        "class": "Crypto",
        "benchmark": "BTC-USD"
    },

    "BNB-USD": {
        "name": "BNB",
        "class": "Crypto",
        "benchmark": "BTC-USD"
    },

    "XRP-USD": {
        "name": "XRP",
        "class": "Crypto",
        "benchmark": "BTC-USD"
    }
}


ASSETS = list(ASSET_METADATA.keys())


# ============================================================
# 3. DATA DOWNLOAD
# ============================================================

def download_market_data(
    ticker,
    period="2y",
    interval="1d"
):

    data = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False
    )

    if data.empty:
        raise ValueError(
            f"No market data retrieved for {ticker}."
        )

    # Handle MultiIndex returned by yfinance
    if isinstance(data.columns, pd.MultiIndex):

        try:
            data.columns = data.columns.get_level_values(0)
        except Exception:
            data.columns = [
                col[0]
                for col in data.columns
            ]

    data = data.reset_index()

    # Standardise date column
    if "Date" not in data.columns:

        if "Datetime" in data.columns:
            data = data.rename(
                columns={
                    "Datetime": "Date"
                }
            )

    data["Date"] = pd.to_datetime(
        data["Date"]
    )

    data = data.sort_values(
        "Date"
    )

    data = data.drop_duplicates(
        subset="Date"
    )

    return data.reset_index(
        drop=True
    )


# ============================================================
# 4. BASIC RETURN FEATURES
# ============================================================

def add_return_features(df):

    df = df.copy()

    df["Return"] = (
        df["Close"]
        .pct_change()
    )

    df["Log_Return"] = np.log(
        df["Close"]
        /
        df["Close"].shift(1)
    )

    return df


# ============================================================
# 5. VOLATILITY FEATURES
# ============================================================

def add_volatility_features(df):

    df = df.copy()

    df["Rolling_Volatility_20D"] = (
        df["Return"]
        .rolling(20)
        .std()
        *
        np.sqrt(252)
    )

    df["Rolling_Volatility_60D"] = (
        df["Return"]
        .rolling(60)
        .std()
        *
        np.sqrt(252)
    )

    return df


# ============================================================
# 6. VOLUME FEATURES
# ============================================================

def add_volume_features(df):

    df = df.copy()

    df["Volume_Change"] = (
        df["Volume"]
        .pct_change()
    )

    df["Volume_MA_20"] = (
        df["Volume"]
        .rolling(20)
        .mean()
    )

    df["Volume_Ratio"] = (
        df["Volume"]
        /
        df["Volume_MA_20"]
    )

    return df


# ============================================================
# 7. DRAWdown & TREND FEATURES
# ============================================================

def add_price_structure_features(df):

    df = df.copy()

    df["Running_Max"] = (
        df["Close"]
        .cummax()
    )

    df["Drawdown"] = (
        df["Close"]
        /
        df["Running_Max"]
        - 1
    )

    df["MA_20"] = (
        df["Close"]
        .rolling(20)
        .mean()
    )

    df["MA_60"] = (
        df["Close"]
        .rolling(60)
        .mean()
    )

    df["Trend_Strength"] = (
        (
            df["MA_20"]
            /
            df["MA_60"]
        ) - 1
    )

    return df


# ============================================================
# 8. ROBUST Z-SCORE
# ============================================================

def robust_zscore(
    series,
    window=60
):

    rolling_median = (
        series
        .rolling(window)
        .median()
    )

    rolling_mad = (
        series
        .rolling(window)
        .apply(
            lambda x:
            np.median(
                np.abs(
                    x - np.median(x)
                )
            ),
            raw=True
        )
    )

    denominator = (
        1.4826
        *
        rolling_mad
    )

    z_score = (
        series
        - rolling_median
    ) / denominator.replace(
        0,
        np.nan
    )

    return z_score


def add_robust_anomaly_features(df):

    df = df.copy()

    df["Return_Robust_Z"] = (
        robust_zscore(
            df["Return"]
        )
    )

    df["Volume_Robust_Z"] = (
        robust_zscore(
            df["Volume_Change"]
        )
    )

    df["Return_Anomaly"] = (
        df["Return_Robust_Z"]
        .abs()
        >= 3
    ).astype(int)

    df["Volume_Anomaly"] = (
        df["Volume_Robust_Z"]
        .abs()
        >= 3
    ).astype(int)

    df["Statistical_Anomaly"] = (
        (
            df["Return_Anomaly"]
            +
            df["Volume_Anomaly"]
        )
        >= 1
    ).astype(int)

    return df


# ============================================================
# 9. GARCH VOLATILITY MODEL
# ============================================================

def add_garch_features(df):

    df = df.copy()

    returns = (
        df["Return"]
        .dropna()
        * 100
    )

    df["GARCH_Volatility"] = np.nan
    df["GARCH_Standardized_Return"] = np.nan
    df["GARCH_Anomaly"] = 0

    if len(returns) < 100:
        return df

    try:

        model = arch_model(
            returns,
            vol="Garch",
            p=1,
            q=1,
            mean="Constant",
            dist="normal"
        )

        fitted = model.fit(
            disp="off"
        )

        conditional_volatility = (
            fitted.conditional_volatility
        )

        standardized_returns = (
            returns
            /
            conditional_volatility
        )

        df.loc[
            standardized_returns.index,
            "GARCH_Volatility"
        ] = conditional_volatility

        df.loc[
            standardized_returns.index,
            "GARCH_Standardized_Return"
        ] = standardized_returns

        df.loc[
            standardized_returns.index,
            "GARCH_Anomaly"
        ] = (
            standardized_returns
            .abs()
            >= 3
        ).astype(int)

    except Exception:

        pass

    return df


# ============================================================
# 10. MULTIVARIATE ANOMALY DETECTION
# ============================================================

def add_multivariate_anomaly_features(df):

    df = df.copy()

    features = [
        "Return",
        "Volume_Change",
        "Rolling_Volatility_20D",
        "Drawdown"
    ]

    available = [
        col
        for col in features
        if col in df.columns
    ]

    clean = (
        df[available]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .dropna()
    )

    df["Mahalanobis_Distance"] = np.nan
    df["Multivariate_Anomaly"] = 0

    if len(clean) < 50:
        return df

    try:

        mean_vector = (
            clean.mean()
            .values
        )

        covariance = (
            np.cov(
                clean.values,
                rowvar=False
            )
        )

        covariance += (
            np.eye(
                covariance.shape[0]
            )
            * 1e-6
        )

        inverse_covariance = np.linalg.inv(
            covariance
        )

        distances = []

        for observation in clean.values:

            distance = np.sqrt(
                mahalanobis(
                    observation,
                    mean_vector,
                    inverse_covariance
                ) ** 2
            )

            distances.append(
                distance
            )

        distances = pd.Series(
            distances,
            index=clean.index
        )

        df.loc[
            distances.index,
            "Mahalanobis_Distance"
        ] = distances

        threshold = (
            distances.quantile(
                0.99
            )
        )

        df.loc[
            distances.index,
            "Multivariate_Anomaly"
        ] = (
            distances >= threshold
        ).astype(int)

    except Exception:

        pass

    return df


# ============================================================
# 11. MARKET REGIME DETECTION
# ============================================================

def add_regime_features(df):

    df = df.copy()

    # Trend regime
    conditions = [
        df["Trend_Strength"] > 0.05,
        df["Trend_Strength"] < -0.05
    ]

    choices = [
        "Bullish",
        "Bearish"
    ]

    df["Trend_Regime"] = np.select(
        conditions,
        choices,
        default="Neutral"
    )

    # Volatility regime
    volatility_median = (
        df["Rolling_Volatility_60D"]
        .rolling(120)
        .median()
    )

    df["Volatility_Regime"] = np.where(
        df["Rolling_Volatility_20D"]
        > volatility_median * 1.25,
        "High Volatility",
        np.where(
            df["Rolling_Volatility_20D"]
            < volatility_median * 0.75,
            "Low Volatility",
            "Normal Volatility"
        )
    )

    # Combined market regime
    df["Market_Regime"] = (
        df["Trend_Regime"]
        + " / "
        + df["Volatility_Regime"]
    )

    # Regime changes
    df["Regime_Change"] = (
        df["Market_Regime"]
        !=
        df["Market_Regime"].shift(1)
    ).astype(int)

    df["Regime_Duration"] = (
        df["Market_Regime"]
        .groupby(
            (
                df["Market_Regime"]
                !=
                df["Market_Regime"].shift()
            ).cumsum()
        )
        .cumcount()
        + 1
    )

    df["Regime_Change_Event"] = (
        df["Regime_Change"]
        == 1
    ).astype(int)

    return df


# ============================================================
# 12. BENCHMARK / CROSS-ASSET ANALYSIS
# ============================================================

def add_cross_asset_analysis(
    df,
    benchmark_df
):

    df = df.copy()

    benchmark = benchmark_df.copy()

    benchmark["Benchmark_Return"] = (
        benchmark["Close"]
        .pct_change()
    )

    benchmark_returns = (
        benchmark[
            [
                "Date",
                "Benchmark_Return"
            ]
        ]
        .dropna()
    )

    df = df.merge(
        benchmark_returns,
        on="Date",
        how="left"
    )

    df["Rolling_Correlation"] = (
        df["Return"]
        .rolling(60)
        .corr(
            df["Benchmark_Return"]
        )
    )

    benchmark_variance = (
        df["Benchmark_Return"]
        .rolling(60)
        .var()
    )

    covariance = (
        df["Return"]
        .rolling(60)
        .cov(
            df["Benchmark_Return"]
        )
    )

    df["Rolling_Beta"] = (
        covariance
        /
        benchmark_variance
    )

    df["Relative_Return"] = (
        df["Return"]
        -
        df["Benchmark_Return"]
    )

    df["Relative_Return_Robust_Z"] = (
        robust_zscore(
            df["Relative_Return"]
        )
    )

    df["Relative_Anomaly"] = (
        df["Relative_Return_Robust_Z"]
        .abs()
        >= 3
    ).astype(int)

    df["Correlation_Z"] = (
        robust_zscore(
            df["Rolling_Correlation"]
        )
    )

    df["Correlation_Break"] = (
        df["Correlation_Z"]
        .abs()
        >= 3
    ).astype(int)

    return df


# ============================================================
# 13. EVIDENCE / INTELLIGENCE SCORE
# ============================================================

def add_evidence_score(df):

    df = df.copy()

    evidence_components = [

        "Statistical_Anomaly",

        "GARCH_Anomaly",

        "Multivariate_Anomaly",

        "Regime_Change_Event",

        "Relative_Anomaly",

        "Correlation_Break"
    ]

    available = [
        col
        for col in evidence_components
        if col in df.columns
    ]

    df["Final_Evidence_Score"] = (
        df[available]
        .fillna(0)
        .sum(axis=1)
    )

    df["Intelligence_Level"] = np.select(

        [
            df["Final_Evidence_Score"] >= 4,
            df["Final_Evidence_Score"] >= 2,
            df["Final_Evidence_Score"] >= 1
        ],

        [
            "Critical",
            "High",
            "Monitor"
        ],

        default="Normal"
    )

    return df


# ============================================================
# 14. FORWARD RETURN FEATURES
# ============================================================

def add_forward_returns(df):

    df = df.copy()

    df["Forward_Return_5D"] = (
        df["Close"]
        .shift(-5)
        /
        df["Close"]
        - 1
    )

    df["Forward_Return_20D"] = (
        df["Close"]
        .shift(-20)
        /
        df["Close"]
        - 1
    )

    return df


# ============================================================
# 15. COMPLETE ASSET ANALYSIS PIPELINE
# ============================================================

def analyze_asset(
    ticker,
    benchmark_ticker=None,
    period="2y"
):

    if benchmark_ticker is None:

        benchmark_ticker = (
            ASSET_METADATA[ticker]
            ["benchmark"]
        )

    # Download asset
    df = download_market_data(
        ticker,
        period=period
    )

    # Basic features
    df = add_return_features(df)

    df = add_volatility_features(df)

    df = add_volume_features(df)

    df = add_price_structure_features(df)

    # Anomaly analysis
    df = add_robust_anomaly_features(df)

    df = add_garch_features(df)

    df = add_multivariate_anomaly_features(df)

    # Regimes
    df = add_regime_features(df)

    # Benchmark analysis
    if benchmark_ticker != ticker:

        benchmark_df = (
            download_market_data(
                benchmark_ticker,
                period=period
            )
        )

        df = add_cross_asset_analysis(
            df,
            benchmark_df
        )

    else:

        df["Rolling_Correlation"] = 1.0
        df["Rolling_Beta"] = 1.0
        df["Relative_Return"] = 0.0
        df["Relative_Return_Robust_Z"] = 0.0
        df["Relative_Anomaly"] = 0
        df["Correlation_Z"] = 0.0
        df["Correlation_Break"] = 0

    # Evidence
    df = add_evidence_score(df)

    # Forward-looking diagnostic
    df = add_forward_returns(df)

    return df


# ============================================================
# 16. RUN COMPLETE MARKET UNIVERSE
# ============================================================

def build_market_universe(
    assets=None,
    period="2y"
):

    if assets is None:
        assets = ASSETS

    all_analysis = {}

    for ticker in assets:

        try:

            print(
                f"Analysing {ticker}..."
            )

            all_analysis[ticker] = (
                analyze_asset(
                    ticker,
                    period=period
                )
            )

            print(
                f"✓ {ticker}"
            )

        except Exception as e:

            print(
                f"✗ {ticker}: {e}"
            )

    return all_analysis


# ============================================================
# 17. VALIDATION
# ============================================================

def validate_analysis(
    all_analysis
):

    validated_analysis = {}

    required_columns = [

        "Date",
        "Close",
        "Return",
        "Log_Return",
        "Rolling_Volatility_20D",
        "Rolling_Volatility_60D",
        "Volume_Ratio",
        "Drawdown",
        "Return_Robust_Z",
        "Volume_Robust_Z",
        "Statistical_Anomaly",
        "GARCH_Anomaly",
        "Mahalanobis_Distance",
        "Multivariate_Anomaly",
        "Market_Regime",
        "Regime_Change",
        "Rolling_Correlation",
        "Rolling_Beta",
        "Relative_Return",
        "Relative_Anomaly",
        "Correlation_Break",
        "Final_Evidence_Score",
        "Intelligence_Level"
    ]

    for ticker, df in all_analysis.items():

        missing = [
            col
            for col in required_columns
            if col not in df.columns
        ]

        if missing:

            print(
                f"⚠ {ticker} missing:"
            )

            print(missing)

        else:

            validated_analysis[ticker] = df

    return validated_analysis


# ============================================================
# 18. PORTFOLIO CONFIGURATION
# ============================================================

PORTFOLIO = {

    "MSFT": 0.20,
    "AAPL": 0.15,
    "NVDA": 0.15,
    "AMZN": 0.10,
    "GOOGL": 0.10,
    "TSLA": 0.10,
    "BTC-USD": 0.10,
    "ETH-USD": 0.10
}


def create_portfolio_weights(
    portfolio=None
):

    if portfolio is None:
        portfolio = PORTFOLIO

    weights = pd.Series(
        portfolio,
        dtype=float
    )

    if not np.isclose(
        weights.sum(),
        1.0
    ):

        raise ValueError(
            "Portfolio weights must sum to 100%."
        )

    return weights


# ============================================================
# 19. PORTFOLIO RETURN MATRIX
# ============================================================

def build_portfolio_returns(
    validated_analysis,
    portfolio_weights
):

    available_assets = [
        ticker
        for ticker in portfolio_weights.index
        if ticker in validated_analysis
    ]

    missing_assets = [
        ticker
        for ticker in portfolio_weights.index
        if ticker not in validated_analysis
    ]

    if missing_assets:

        raise ValueError(
            "Missing portfolio assets: "
            + ", ".join(missing_assets)
        )

    return_series = {}

    for ticker in available_assets:

        df = validated_analysis[ticker].copy()

        temp = df[
            [
                "Date",
                "Return"
            ]
        ].copy()

        temp["Date"] = pd.to_datetime(
            temp["Date"]
        )

        temp = (
            temp
            .set_index("Date")
        )

        return_series[ticker] = (
            temp["Return"]
        )

    portfolio_returns = pd.concat(
        return_series.values(),
        axis=1,
        join="inner"
    )

    portfolio_returns.columns = (
        available_assets
    )

    portfolio_returns = (
        portfolio_returns
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .dropna()
    )

    aligned_weights = (
        portfolio_weights
        .reindex(
            portfolio_returns.columns
        )
    )

    portfolio_returns["Portfolio"] = (
        portfolio_returns
        .mul(
            aligned_weights,
            axis=1
        )
        .sum(axis=1)
    )

    return portfolio_returns


# ============================================================
# 20. PORTFOLIO PERFORMANCE
# ============================================================

def calculate_portfolio_performance(
    portfolio_returns
):

    portfolio_series = (
        portfolio_returns[
            "Portfolio"
        ]
    )

    cumulative = (
        1 + portfolio_series
    ).cumprod()

    total_return = (
        cumulative.iloc[-1]
        - 1
    )

    annualized_return = (
        (1 + total_return)
        ** (
            252
            /
            len(portfolio_series)
        )
        - 1
    )

    annualized_volatility = (
        portfolio_series.std()
        *
        np.sqrt(252)
    )

    running_peak = (
        cumulative.cummax()
    )

    drawdown = (
        cumulative
        /
        running_peak
        - 1
    )

    maximum_drawdown = (
        drawdown.min()
    )

    current_drawdown = (
        drawdown.iloc[-1]
    )

    return {

        "cumulative": cumulative,

        "drawdown": drawdown,

        "total_return": total_return,

        "annualized_return":
            annualized_return,

        "annualized_volatility":
            annualized_volatility,

        "maximum_drawdown":
            maximum_drawdown,

        "current_drawdown":
            current_drawdown
    }


# ============================================================
# 21. RISK CONTRIBUTION
# ============================================================

def calculate_risk_contribution(
    portfolio_returns,
    portfolio_weights
):

    asset_returns = portfolio_returns[
        portfolio_weights.index
    ].copy()

    covariance_matrix = (
        asset_returns.cov()
        * 252
    )

    weights = (
        portfolio_weights
        .reindex(
            asset_returns.columns
        )
        .values
    )

    portfolio_variance = (
        weights
        @ covariance_matrix.values
        @ weights
    )

    portfolio_volatility = np.sqrt(
        portfolio_variance
    )

    marginal_contribution = (
        covariance_matrix.values
        @ weights
    ) / portfolio_volatility

    component_contribution = (
        weights
        *
        marginal_contribution
    )

    risk_contribution_pct = (
        component_contribution
        /
        portfolio_volatility
    )

    risk_contribution = pd.DataFrame({

        "Weight": weights,

        "Risk_Contribution":
            risk_contribution_pct

    }, index=asset_returns.columns)

    risk_contribution[
        "Risk_Multiple"
    ] = (
        risk_contribution[
            "Risk_Contribution"
        ]
        /
        risk_contribution[
            "Weight"
        ]
    )

    risk_contribution[
        "Risk_Overweight"
    ] = (
        risk_contribution[
            "Risk_Contribution"
        ]
        -
        risk_contribution[
            "Weight"
        ]
    )

    return (
        risk_contribution
        .sort_values(
            "Risk_Contribution",
            ascending=False
        )
    )


# ============================================================
# 22. CORRELATION ANALYSIS
# ============================================================

def calculate_correlation_analysis(
    portfolio_returns,
    portfolio_weights
):

    asset_returns = (
        portfolio_returns[
            portfolio_weights.index
        ]
    )

    correlation_matrix = (
        asset_returns.corr()
    )

    correlation_pairs = []

    assets = (
        correlation_matrix.columns
    )

    for i in range(
        len(assets)
    ):

        for j in range(
            i + 1,
            len(assets)
        ):

            asset_1 = assets[i]
            asset_2 = assets[j]

            correlation = (
                correlation_matrix
                .loc[
                    asset_1,
                    asset_2
                ]
            )

            correlation_pairs.append({

                "Asset_1": asset_1,

                "Asset_2": asset_2,

                "Correlation":
                    correlation
            })

    correlation_pairs = pd.DataFrame(
        correlation_pairs
    )

    correlation_pairs = (
        correlation_pairs
        .sort_values(
            "Correlation",
            ascending=False
        )
    )

    high_correlation_pairs = (
        correlation_pairs[
            correlation_pairs[
                "Correlation"
            ] >= 0.70
        ]
    )

    average_pairwise_correlation = (
        correlation_pairs[
            "Correlation"
        ].mean()
    )

    average_absolute_correlation = (
        correlation_pairs[
            "Correlation"
        ].abs().mean()
    )

    correlation_exposure = {}

    for asset in assets:

        correlations = (
            correlation_matrix[asset]
            .drop(asset)
            .abs()
        )

        correlation_exposure[asset] = (
            correlations >= 0.70
        ).sum()

    correlation_exposure = pd.Series(
        correlation_exposure,
        name="High_Correlation_Count"
    )

    return {

        "matrix":
            correlation_matrix,

        "pairs":
            correlation_pairs,

        "high_correlation_pairs":
            high_correlation_pairs,

        "average_pairwise_correlation":
            average_pairwise_correlation,

        "average_absolute_correlation":
            average_absolute_correlation,

        "correlation_exposure":
            correlation_exposure
    }


# ============================================================
# 23. LIQUIDITY STRESS
# ============================================================

def calculate_liquidity_stress(
    df
):

    df = df.copy()

    volume_stress = (
        df["Volume_Robust_Z"]
        .abs()
        .clip(0, 4)
        / 4
    )

    volatility_baseline = (
        df[
            "Rolling_Volatility_20D"
        ]
        .rolling(60)
        .median()
    )

    volatility_stress = (
        df[
            "Rolling_Volatility_20D"
        ]
        /
        volatility_baseline
    )

    volatility_stress = (
        (volatility_stress - 1)
        .clip(
            lower=0,
            upper=3
        )
        / 3
    )

    return_stress = (
        df["Return_Robust_Z"]
        .abs()
        .clip(0, 4)
        / 4
    )

    drawdown_stress = (
        df["Drawdown"]
        .abs()
        .clip(
            0,
            0.50
        )
        / 0.50
    )

    df["Liquidity_Stress_Score"] = (
        100
        *
        (
            0.30 * volume_stress
            +
            0.30 * volatility_stress
            +
            0.20 * return_stress
            +
            0.20 * drawdown_stress
        )
    )

    return df


def build_liquidity_analysis(
    validated_analysis
):

    liquidity_analysis = {}

    for ticker, df in validated_analysis.items():

        liquidity_df = df[
            [
                "Date",
                "Close",
                "Volume",
                "Volume_Ratio",
                "Rolling_Volatility_20D",
                "Return_Robust_Z",
                "Volume_Robust_Z",
                "Drawdown"
            ]
        ].copy()

        liquidity_df = (
            liquidity_df
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
        )

        liquidity_df = (
            calculate_liquidity_stress(
                liquidity_df
            )
        )

        liquidity_analysis[ticker] = (
            liquidity_df
        )

    return liquidity_analysis


# ============================================================
# 24. CURRENT LIQUIDITY CONDITIONS
# ============================================================

def calculate_current_liquidity(
    liquidity_analysis
):

    current_liquidity = []

    for ticker, df in (
        liquidity_analysis.items()
    ):

        latest = df.iloc[-1]

        current_liquidity.append({

            "Ticker": ticker,

            "Volume_Ratio":
                latest["Volume_Ratio"],

            "Volatility":
                latest[
                    "Rolling_Volatility_20D"
                ],

            "Drawdown":
                latest["Drawdown"],

            "Liquidity_Stress":
                latest[
                    "Liquidity_Stress_Score"
                ]
        })

    current_liquidity = pd.DataFrame(
        current_liquidity
    )

    def classify(score):

        if score >= 70:
            return "SEVERE"

        elif score >= 50:
            return "HIGH"

        elif score >= 30:
            return "MODERATE"

        return "NORMAL"

    current_liquidity[
        "Liquidity_Status"
    ] = (
        current_liquidity[
            "Liquidity_Stress"
        ]
        .apply(classify)
    )

    return (
        current_liquidity
        .sort_values(
            "Liquidity_Stress",
            ascending=False
        )
    )


# ============================================================
# 25. PORTFOLIO LIQUIDITY
# ============================================================

def calculate_portfolio_liquidity(
    current_liquidity,
    portfolio_weights
):

    portfolio_liquidity = (
        current_liquidity
        .merge(
            portfolio_weights.rename(
                "Weight"
            ),
            left_on="Ticker",
            right_index=True,
            how="left"
        )
    )

    portfolio_liquidity[
        "Weighted_Liquidity_Stress"
    ] = (
        portfolio_liquidity[
            "Liquidity_Stress"
        ]
        *
        portfolio_liquidity[
            "Weight"
        ]
    )

    portfolio_liquidity_score = (
        portfolio_liquidity[
            "Weighted_Liquidity_Stress"
        ].sum()
    )

    portfolio_liquidity[
        "Liquidity_Attention"
    ] = (
        portfolio_liquidity[
            "Weight"
        ]
        *
        portfolio_liquidity[
            "Liquidity_Stress"
        ]
    )

    return {

        "table":
            portfolio_liquidity,

        "score":
            portfolio_liquidity_score,

        "priorities":
            portfolio_liquidity
            .sort_values(
                "Liquidity_Attention",
                ascending=False
            )
    }


# ============================================================
# 26. ALERT ENGINE
# ============================================================

def build_alert_monitor(
    validated_analysis,
    portfolio_weights,
    risk_contribution,
    current_liquidity
):

    current_intelligence = []

    for ticker in portfolio_weights.index:

        df = validated_analysis[
            ticker
        ].copy()

        latest = df.iloc[-1]

        current_intelligence.append({

            "Ticker": ticker,

            "Intelligence_Score":
                latest.get(
                    "Final_Evidence_Score",
                    np.nan
                ),

            "Intelligence_Level":
                latest.get(
                    "Intelligence_Level",
                    "Unknown"
                ),

            "Market_Regime":
                latest.get(
                    "Market_Regime",
                    "Unknown"
                ),

            "Statistical_Anomaly":
                latest.get(
                    "Statistical_Anomaly",
                    0
                ),

            "GARCH_Anomaly":
                latest.get(
                    "GARCH_Anomaly",
                    0
                ),

            "Multivariate_Anomaly":
                latest.get(
                    "Multivariate_Anomaly",
                    0
                ),

            "Regime_Change":
                latest.get(
                    "Regime_Change",
                    0
                ),

            "Correlation_Break":
                latest.get(
                    "Correlation_Break",
                    0
                )
        })

    current_intelligence = pd.DataFrame(
        current_intelligence
    )

    def generate_alerts(row):

        alerts = []

        if row[
            "Statistical_Anomaly"
        ] == 1:

            alerts.append(
                "Statistical anomaly"
            )

        if row[
            "GARCH_Anomaly"
        ] == 1:

            alerts.append(
                "Volatility anomaly"
            )

        if row[
            "Multivariate_Anomaly"
        ] == 1:

            alerts.append(
                "Multivariate anomaly"
            )

        if row[
            "Regime_Change"
        ] == 1:

            alerts.append(
                "Market regime change"
            )

        if row[
            "Correlation_Break"
        ] == 1:

            alerts.append(
                "Correlation break"
            )

        return alerts

    current_intelligence[
        "Alerts"
    ] = current_intelligence.apply(
        generate_alerts,
        axis=1
    )

    current_intelligence[
        "Alert_Count"
    ] = (
        current_intelligence[
            "Alerts"
        ]
        .apply(len)
    )

    alert_monitor = (
        current_intelligence
        .merge(
            risk_contribution[
                [
                    "Weight",
                    "Risk_Contribution",
                    "Risk_Multiple"
                ]
            ],
            left_on="Ticker",
            right_index=True,
            how="left"
        )
    )

    alert_monitor = (
        alert_monitor
        .merge(
            current_liquidity[
                [
                    "Ticker",
                    "Liquidity_Stress",
                    "Liquidity_Status"
                ]
            ],
            on="Ticker",
            how="left"
        )
    )

    alert_monitor[
        "Attention_Priority"
    ] = (

        alert_monitor[
            "Intelligence_Score"
        ].fillna(0)

        *

        (
            1
            +
            alert_monitor[
                "Risk_Contribution"
            ].fillna(0)
        )

        *

        (
            1
            +
            alert_monitor[
                "Liquidity_Stress"
            ].fillna(0)
            / 100
        )
    )

    def classify_severity(row):

        score = row[
            "Attention_Priority"
        ]

        alerts = row[
            "Alert_Count"
        ]

        if (
            score >= 100
            or alerts >= 4
        ):

            return "CRITICAL"

        elif (
            score >= 60
            or alerts >= 3
        ):

            return "HIGH"

        elif (
            score >= 30
            or alerts >= 1
        ):

            return "MEDIUM"

        return "LOW"

    alert_monitor[
        "Alert_Severity"
    ] = alert_monitor.apply(
        classify_severity,
        axis=1
    )

    alert_monitor = (
        alert_monitor
        .sort_values(
            "Attention_Priority",
            ascending=False
        )
    )

    active_alerts = (
        alert_monitor[
            alert_monitor[
                "Alert_Count"
            ] > 0
        ]
        .copy()
    )

    def create_message(row):

        alerts = ", ".join(
            row["Alerts"]
        )

        return (
            f"[{row['Alert_Severity']}] "
            f"{row['Ticker']}: "
            f"{alerts}. "
            f"Intelligence score = "
            f"{row['Intelligence_Score']:.1f}; "
            f"Risk contribution = "
            f"{row['Risk_Contribution']:.1%}; "
            f"Liquidity stress = "
            f"{row['Liquidity_Stress']:.1f}/100."
        )

    active_alerts[
        "Alert_Message"
    ] = active_alerts.apply(
        create_message,
        axis=1
    )

    return {

        "monitor":
            alert_monitor,

        "active_alerts":
            active_alerts
    }


# ============================================================
# 27. EXECUTIVE INTELLIGENCE
# ============================================================

def generate_executive_intelligence(
    performance,
    risk_contribution,
    correlation_analysis,
    portfolio_liquidity,
    alert_engine
):

    top_5_risk_share = (
        risk_contribution
        .head(5)[
            "Risk_Contribution"
        ]
        .sum()
    )

    average_absolute_correlation = (
        correlation_analysis[
            "average_absolute_correlation"
        ]
    )

    liquidity_score = (
        portfolio_liquidity[
            "score"
        ]
    )

    active_alerts = (
        alert_engine[
            "active_alerts"
        ]
    )

    critical_alerts = (
        active_alerts[
            active_alerts[
                "Alert_Severity"
            ] == "CRITICAL"
        ]
    )

    high_alerts = (
        active_alerts[
            active_alerts[
                "Alert_Severity"
            ] == "HIGH"
        ]
    )

    principal_risks = []

    if top_5_risk_share >= 0.75:

        principal_risks.append(
            "Portfolio risk is highly concentrated among a small number of assets."
        )

    elif top_5_risk_share >= 0.60:

        principal_risks.append(
            "Portfolio exhibits moderate risk concentration."
        )

    if average_absolute_correlation >= 0.60:

        principal_risks.append(
            "High cross-asset correlation may reduce effective diversification."
        )

    if liquidity_score >= 50:

        principal_risks.append(
            "Elevated liquidity stress is present across the portfolio."
        )

    if len(critical_alerts) > 0:

        principal_risks.append(
            f"{len(critical_alerts)} critical market intelligence alert(s) detected."
        )

    elif len(high_alerts) > 0:

        principal_risks.append(
            f"{len(high_alerts)} high-priority market intelligence alert(s) detected."
        )

    if not principal_risks:

        principal_risks.append(
            "No major portfolio-level risk concentration identified by the current diagnostic framework."
        )

    if len(critical_alerts) > 0:

        portfolio_alert_status = (
            "CRITICAL ATTENTION REQUIRED"
        )

    elif len(high_alerts) > 0:

        portfolio_alert_status = (
            "HIGH ATTENTION"
        )

    elif len(active_alerts) > 0:

        portfolio_alert_status = (
            "MONITOR"
        )

    else:

        portfolio_alert_status = (
            "NO MATERIAL ALERTS"
        )

    decision_actions = []

    if len(critical_alerts) > 0:

        decision_actions.append(
            "Review critical alerts immediately and assess underlying market drivers."
        )

    if len(high_alerts) > 0:

        decision_actions.append(
            "Review high-priority assets for changes in volatility, regime, or anomaly behaviour."
        )

    if top_5_risk_share >= 0.75:

        decision_actions.append(
            "Review portfolio concentration and dependence on the largest risk contributors."
        )

    if average_absolute_correlation >= 0.60:

        decision_actions.append(
            "Investigate whether correlated positions represent hidden common-factor exposure."
        )

    if liquidity_score >= 50:

        decision_actions.append(
            "Review liquidity conditions and potential exit-risk for stressed positions."
        )

    if not decision_actions:

        decision_actions.append(
            "Continue routine monitoring; no material escalation identified."
        )

    return {

        "top_5_risk_share":
            top_5_risk_share,

        "average_absolute_correlation":
            average_absolute_correlation,

        "liquidity_score":
            liquidity_score,

        "active_alerts":
            len(active_alerts),

        "critical_alerts":
            len(critical_alerts),

        "high_alerts":
            len(high_alerts),

        "portfolio_alert_status":
            portfolio_alert_status,

        "principal_risks":
            principal_risks,

        "decision_actions":
            decision_actions,

        "top_risk_contributors":
            risk_contribution.head(3)
    }


# ============================================================
# 28. COMPLETE ENGINE
# ============================================================

def run_indigo_engine(
    assets=None,
    period="2y",
    portfolio=None
):

    # ----------------------------------------
    # Market universe
    # ----------------------------------------

    all_analysis = (
        build_market_universe(
            assets=assets,
            period=period
        )
    )

    # ----------------------------------------
    # Validation
    # ----------------------------------------

    validated_analysis = (
        validate_analysis(
            all_analysis
        )
    )

    # ----------------------------------------
    # Portfolio
    # ----------------------------------------

    portfolio_weights = (
        create_portfolio_weights(
            portfolio
        )
    )

    # Ensure portfolio coverage
    missing_assets = [
        ticker
        for ticker in portfolio_weights.index
        if ticker not in validated_analysis
    ]

    if missing_assets:

        raise ValueError(
            "Portfolio contains assets not "
            "available in validated analysis: "
            +
            ", ".join(missing_assets)
        )

    # ----------------------------------------
    # Portfolio returns
    # ----------------------------------------

    portfolio_returns = (
        build_portfolio_returns(
            validated_analysis,
            portfolio_weights
        )
    )

    # ----------------------------------------
    # Performance
    # ----------------------------------------

    performance = (
        calculate_portfolio_performance(
            portfolio_returns
        )
    )

    # ----------------------------------------
    # Risk
    # ----------------------------------------

    risk_contribution = (
        calculate_risk_contribution(
            portfolio_returns,
            portfolio_weights
        )
    )

    # ----------------------------------------
    # Correlation
    # ----------------------------------------

    correlation_analysis = (
        calculate_correlation_analysis(
            portfolio_returns,
            portfolio_weights
        )
    )

    # ----------------------------------------
    # Liquidity
    # ----------------------------------------

    liquidity_analysis = (
        build_liquidity_analysis(
            validated_analysis
        )
    )

    current_liquidity = (
        calculate_current_liquidity(
            liquidity_analysis
        )
    )

    portfolio_liquidity = (
        calculate_portfolio_liquidity(
            current_liquidity,
            portfolio_weights
        )
    )

    # ----------------------------------------
    # Alerts
    # ----------------------------------------

    alert_engine = (
        build_alert_monitor(
            validated_analysis,
            portfolio_weights,
            risk_contribution,
            current_liquidity
        )
    )

    # ----------------------------------------
    # Executive intelligence
    # ----------------------------------------

    executive_intelligence = (
        generate_executive_intelligence(
            performance,
            risk_contribution,
            correlation_analysis,
            portfolio_liquidity,
            alert_engine
        )
    )

    # ----------------------------------------
    # Return complete engine state
    # ----------------------------------------

    return {

        "asset_metadata":
            ASSET_METADATA,

        "all_analysis":
            all_analysis,

        "validated_analysis":
            validated_analysis,

        "portfolio_weights":
            portfolio_weights,

        "portfolio_returns":
            portfolio_returns,

        "performance":
            performance,

        "risk_contribution":
            risk_contribution,

        "correlation":
            correlation_analysis,

        "liquidity":
            liquidity_analysis,

        "current_liquidity":
            current_liquidity,

        "portfolio_liquidity":
            portfolio_liquidity,

        "alerts":
            alert_engine,

        "executive":
            executive_intelligence
    }


# ============================================================
# 29. DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":

    results = run_indigo_engine()

    print()
    print("=" * 70)
    print(
        "INDIGO MARKET INTELLIGENCE ENGINE"
    )
    print("=" * 70)

    print(
        f"\nAssets analysed: "
        f"{len(results['validated_analysis'])}"
    )

    print(
        f"Portfolio return: "
        f"{results['performance']['total_return']:.2%}"
    )

    print(
        f"Annualised volatility: "
        f"{results['performance']['annualized_volatility']:.2%}"
    )

    print(
        f"Maximum drawdown: "
        f"{results['performance']['maximum_drawdown']:.2%}"
    )

    print(
        f"Portfolio liquidity stress: "
        f"{results['portfolio_liquidity']['score']:.2f}/100"
    )

    print(
        f"Active alerts: "
        f"{results['executive']['active_alerts']}"
    )

    print(
        f"Portfolio alert status: "
        f"{results['executive']['portfolio_alert_status']}"
    )

    print(
        "\nENGINE EXECUTION COMPLETE."
    )
