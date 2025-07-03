"""
VaporIQ Galaxy Dashboard (Dash + Plotly)
---------------------------------------
• Full-screen galaxy background (background.png beside app.py)
• Four tabs, each with 10 Plotly charts (no seaborn)
"""

from __future__ import annotations

import base64
import os
from pathlib import Path

import numpy as np
import pandas as pd
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_curve,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

# ───────────────────────────────────
# CONFIG
# ───────────────────────────────────
DATA_PATH = Path(__file__).parent
BACKGROUND_FILE = DATA_PATH / "background.png"
ENC_BG = base64.b64encode(BACKGROUND_FILE.read_bytes()).decode()

COLOR_BG_ALPHA = "rgba(0,0,0,0.6)"
FONT_FAMILY = "Helvetica, Arial, sans-serif"

# ───────────────────────────────────
# DATA LOAD
# ───────────────────────────────────
def load_data():
    users = pd.read_csv(DATA_PATH / "users_synthetic.csv")
    trends = pd.read_csv(DATA_PATH / "flavor_trends.csv", parse_dates=["Date"])
    return users, trends


# ───────────────────────────────────
# PLOT BUILDERS
# ───────────────────────────────────
def data_viz_figs(users: pd.DataFrame, trends: pd.DataFrame) -> list:
    counts = users["FlavourFamilies"].value_counts().reset_index()
    counts.columns = ["Flavour", "Count"]
    return [
        px.histogram(users, x="Age", nbins=20, title="Age Distribution"),
        px.scatter(users, x="Age", y="PodsPerWeek", color="Gender", title="Pods vs Age"),
        px.bar(counts, x="Flavour", y="Count", title="Flavour Popularity"),
        px.box(users, x="PurchaseChannel", y="PodsPerWeek", title="Pods by Channel"),
        px.violin(users, x="Gender", y="PodsPerWeek", box=True, title="Pods by Gender"),
        px.pie(users, names="SubscribeIntent", title="Subscribe Intent"),
        px.density_heatmap(users, x="SweetLike", y="MentholLike", title="Sweet vs Menthol"),
        px.treemap(users, path=["Gender", "PurchaseChannel"], values="PodsPerWeek", title="Pods Treemap"),
        px.line(trends, x="Date", y=trends.columns[1:4], title="Top-3 Flavour Trends"),
        px.area(trends, x="Date", y="Custard Kunafa", title="Custard Kunafa Trend"),
    ]


def classify(users: pd.DataFrame) -> dict:
    X = users[["Age", "SweetLike", "MentholLike", "PodsPerWeek"]]
    y = users["SubscribeIntent"]
    Xs = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.2, random_state=42)
    models = {
        "KNN": KNeighborsClassifier(),
        "DT": DecisionTreeClassifier(random_state=42),
        "RF": RandomForestClassifier(random_state=42),
        "GB": GradientBoostingClassifier(random_state=42),
    }
    results = {}
    for name, mdl in models.items():
        mdl.fit(X_train, y_train)
        y_pred = mdl.predict(X_test)
        proba = mdl.predict_proba(X_test)[:, 1] if hasattr(mdl, "predict_proba") else np.zeros_like(y_pred)
        results[name] = {
            "model": mdl,
            "acc": mdl.score(X_test, y_test),
            "f1": f1_score(y_test, y_pred),
            "y_true": y_test,
            "y_pred": y_pred,
            "prob": proba,
        }
    return results


def cluster(users: pd.DataFrame):
    X = MinMaxScaler().fit_transform(users[["Age", "SweetLike", "MentholLike", "PodsPerWeek"]])
    sil = []
    for k in range(2, 11):
        km = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(X)
        sil.append(silhouette_score(X, km.labels_))
    best_k = sil.index(max(sil)) + 2
    users["Cluster"] = KMeans(n_clusters=best_k, random_state=42, n_init="auto").fit_predict(X)
    return users, sil


def tastedna_figs(results: dict, users: pd.DataFrame) -> list:
    names = list(results.keys())
    accs = [results[n]["acc"] for n in names]
    f1s = [results[n]["f1"] for n in names]
    best = max(results, key=lambda n: results[n]["f1"])

    cm = confusion_matrix(results[best]["y_true"], results[best]["y_pred"])
    fpr, tpr, _ = roc_curve(results[best]["y_true"], results[best]["prob"])
    prec, rec, _ = precision_recall_curve(results[best]["y_true"], results[best]["prob"])
    rf_imp = results["RF"]["model"].feature_importances_

    sil_scores = cluster(users)[1]

    figs = [
        px.bar(x=names, y=accs, title="Accuracy by Model"),
        px.bar(x=names, y=f1s, title="F1 Score by Model"),
        px.imshow(cm, text_auto=True, title=f"{best} Confusion Matrix"),
        go.Figure(data=go.Scatter(x=fpr, y=tpr, mode="lines")).update_layout(title=f"{best} ROC Curve"),
        go.Figure(data=go.Scatter(x=rec, y=prec, mode="lines")).update_layout(title=f"{best} Precision-Recall"),
        px.histogram(results[best]["prob"], nbins=20, title="Predicted Probabilities"),
        px.bar(x=["Age", "Sweet", "Menthol", "Pods"], y=rf_imp, title="RF Feature Importance"),
        px.box(users, x="Cluster", y="PodsPerWeek", title="Pods per Cluster"),
        px.pie(users, names="Cluster", title="Cluster Distribution"),
        px.line(x=np.arange(2, 11), y=sil_scores, title="Silhouette Score by k"),
    ]
    return figs


def forecasting_figs(trends: pd.DataFrame) -> list:
    slopes = {c: np.polyfit(np.arange(len(trends)), trends[c], 1)[0] for c in trends.columns[1:]}
    top = max(slopes, key=slopes.get)
    split = int(len(trends) * 0.8)

    lr = LinearRegression().fit(np.arange(split).reshape(-1, 1), trends[top][:split])
    preds = lr.predict(np.arange(split, len(trends)).reshape(-1, 1))
    resid = trends[top][split:] - preds

    norm = (np.array(list(slopes.values())) - min(slopes.values())) / (
        max(slopes.values()) - min(slopes.values()) + 1e-9
    )

    figs = [
        px.line(trends, x="Date", y=top, title=f"{top} Trend"),
        px.area(trends.assign(Cumulative=trends[top].cumsum()), x="Date", y="Cumulative", title="Cumulative Mentions"),
        px.scatter(x=trends["Date"][split:], y=trends[top][split:], title="Actual vs Predicted")
          .add_scatter(x=trends["Date"][split:], y=preds, mode="lines", name="Predicted"),
        px.bar(x=list(slopes.keys()), y=list(slopes.values()), title="Slope Comparison"),
        px.box(trends, y=top, title=f"{top} Distribution"),
        px.histogram(resid, nbins=20, title="Residuals Distribution"),
        px.imshow(trends.iloc[:, 1:].corr(), text_auto=True, title="Correlation Heatmap"),
        go.Figure(data=go.Indicator(mode="gauge+number", value=slopes[top], title={"text": "Slope Gauge"})),
        go.Figure(data=go.Scatterpolar(r=norm, theta=list(slopes.keys()), fill="toself")).update_layout(title="Normalized Slopes Radar"),
        px.pie(names=list(slopes.keys()), values=np.abs(list(slopes.values())), title="Slope Share"),
    ]
    return figs


def apriori_figs(users: pd.DataFrame) -> list:
    basket = pd.concat(
        [
            users["FlavourFamilies"].str.get_dummies(sep=","),
            pd.get_dummies(users["PurchaseChannel"], prefix="Channel"),
        ],
        axis=1,
    ).astype(bool)

    freq = apriori(basket, min_support=0.05, use_colnames=True)
    rules = association_rules(freq, metric="confidence", min_threshold=0.6)
    best = rules.nlargest(1, "lift").iloc[0]

    freq_sorted = freq.sort_values("support", ascending=False).head(20)
    freq_sorted["cum"] = freq_sorted["support"].cumsum()

    figs = [
        px.bar(freq.nlargest(10, "support"), x="itemsets", y="support", title="Top Itemsets"),
        px.scatter(rules, x="support", y="confidence", size="lift", title="Rule Metrics"),
        px.pie(rules.nlargest(5, "lift"), names="antecedents", values="lift", title="Top Antecedents"),
        px.sunburst(rules, path=["antecedents", "consequents"], values="lift", title="Rules Sunburst"),
        px.histogram(rules, x="support", title="Support Distribution"),
        px.imshow(rules[["support", "confidence", "lift"]], title="Metrics Heatmap"),
        go.Figure(
            data=go.Scatterpolar(
                r=[best["support"], best["confidence"], best["lift"]],
                theta=["Support", "Confidence", "Lift"],
                fill="toself",
            )
        ).update_layout(title="Best Rule Radar"),
        go.Figure(go.Indicator(mode="gauge+number", value=best["lift"], title={"text": "Lift Gauge"})),
        px.box(rules, y="lift", title="Lift Distribution"),
        px.area(freq_sorted, x=freq_sorted.index, y="cum", title="Cumulative Support"),
    ]
    return figs


# ───────────────────────────────────
# DASH APP
# ───────────────────────────────────
users_df, trends_df = load_data()
class_results = classify(users_df)
users_df, _ = cluster(users_df)

app = Dash(__name__)
app.layout = html.Div(
    style={
        "minHeight": "100vh",
        "backgroundImage": f"url('data:image/png;base64,{ENC_BG}')",
        "backgroundSize": "cover",
        "backgroundPosition": "center",
        "padding": "20px",
        "fontFamily": FONT_FAMILY,
        "color": "#FFFFFF",
    },
    children=[
        html.H1("VaporIQ Galaxy Dashboard", style={"textAlign": "center", "marginBottom": "25px"}),
        dcc.Tabs(id="tabs", value="tab-1", children=[
            dcc.Tab(label="Data Viz", value="tab-1"),
            dcc.Tab(label="TasteDNA", value="tab-2"),
            dcc.Tab(label="Forecasting", value="tab-3"),
            dcc.Tab(label="Apriori", value="tab-4"),
        ]),
        html.Div(id="content"),
    ],
)


@app.callback(Output("content", "children"), Input("tabs", "value"))
def render(tab):
    if tab == "tab-1":
        figs = data_viz_figs(users_df, trends_df)
    elif tab == "tab-2":
        figs = tastedna_figs(class_results, users_df)
    elif tab == "tab-3":
        figs = forecasting_figs(trends_df)
    else:
        figs = apriori_figs(users_df)

    return html.Div(
        style={"backgroundColor": COLOR_BG_ALPHA, "padding": "20px", "borderRadius": "12px", "marginTop": "20px"},
        children=[dcc.Graph(figure=f) for f in figs],
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run_server(host="0.0.0.0", port=port)
