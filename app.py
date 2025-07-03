"""
VaporIQ Galaxy Dashboard (Dash + Plotly)
---------------------------------------
• Full-screen galaxy background (background.png in same folder)
• All charts use Plotly (no seaborn)
• Four tabs, each with 10 unique Plotly charts
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
        px.scatter(users, x="Age", y="PodsPerWeek", color="Gender", title="Pods per Week vs Age"),
        px.bar(counts, x="Flavour", y="Count", title="Flavour Popularity"),
        px.box(users, x="PurchaseChannel", y="PodsPerWeek", title="Pods per Channel"),
        px.violin(users, x="Gender", y="PodsPerWeek", box=True, title="Pods per Gender"),
        px.pie(users, names="SubscribeIntent", title="Subscribe Intent Split"),
        px.density_heatmap(users, x="SweetLike", y="MentholLike", title="Sweet vs Menthol"),
        px.treemap(users, path=["Gender", "PurchaseChannel"], values="PodsPerWeek", title="Pods Treemap"),
        px.line(trends, x="Date", y=trends.columns[1:4], title="Top 3 Flavour Trends"),
        px.area(trends, x="Date", y="Custard Kunafa", title="Custard Kunafa Trend"),
    ]


def classify(users: pd.DataFrame) -> dict:
    X = users[["Age", "SweetLike", "MentholLike", "PodsPerWeek"]]
    y = users["SubscribeIntent"]
    X_scaled = MinMaxScaler().fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    models = {
        "KNN": KNeighborsClassifier(),
        "DT": DecisionTreeClassifier(random_state=42),
        "RF": RandomForestClassifier(random_state=42),
        "GB": GradientBoostingClassifier(random_state=42),
    }
    res = {}
    for name, mdl in models.items():
        mdl.fit(X_tr, y_tr)
        y_pred = mdl.predict(X_te)
        prob = mdl.predict_proba(X_te)[:, 1] if hasattr(mdl, "predict_proba") else np.zeros_like(y_pred)
        res[name] = {
            "model": mdl,
            "acc": mdl.score(X_te, y_te),
            "f1": f1_score(y_te, y_pred),
            "y_true": y_te,
            "y_pred": y_pred,
            "prob": prob,
        }
    return res


def cluster(users: pd.DataFrame):
    X = MinMaxScaler().fit_transform(users[["Age", "SweetLike", "MentholLike", "PodsPerWeek"]])
    sils = []
    for k in range(2, 11):
        km = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(X)
        sils.append(silhouette_score(X, km.labels_))
    best_k = sils.index(max(sils)) + 2
    users["Cluster"] = KMeans(n_clusters=best_k, random_state=42, n_init="auto").fit_predict(X)
    return users, sils


def tastedna_figs(res: dict, users: pd.DataFrame) -> list:
    names = list(res.keys())
    acc = [res[n]["acc"] for n in names]
    f1 = [res[n]["f1"] for n in names]
    best = max(res, key=lambda n: res[n]["f1"])
    cm = confusion_matrix(res[best]["y_true"], res[best]["y_pred"])
    fpr, tpr, _ = roc_curve(res[best]["y_true"], res[best]["prob"])
    prec, rec, _ = precision_recall_curve(res[best]["y_true"], res[best]["prob"])
    rf_imp = res["RF"]["model"].feature_importances_
    return [
        px.bar(x=names, y=acc, title="Model Accuracy"),
        px.bar(x=names, y=f1, title="Model F1 Score"),
        px.imshow(cm, text_auto=True, title=f"{best} Confusion Matrix"),
        go.Figure(go.Scatter(x=fpr, y=tpr, mode="lines")).update_layout(title=f"{best} ROC Curve"),
        go.Figure(go.Scatter(x=rec, y=prec, mode="lines")).update_layout(title=f"{best} Precision-Recall"),
        px.histogram(res[best]["prob"], nbins=20, title="Predicted Probabilities"),
        px.bar(x=["Age", "Sweet", "Menthol", "Pods"], y=rf_imp, title="RF Feature Importance"),
        px.box(users, x="Cluster", y="PodsPerWeek", title="Pods per Cluster"),
        px.pie(users, names="Cluster", title="Cluster Distribution"),
        px.line(x=np.arange(2, 11), y=cluster(users)[1], title="Silhouette Score by k"),
    ]


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
    return [
        px.line(trends, x="Date", y=top, title=f"{top} Trend"),
        px.area(trends.assign(Cumulative=trends[top].cumsum()), x="Date", y="Cumulative", title="Cumulative Mentions"),
        px.scatter(
            x=trends["Date"][split:], y=trends[top][split:], title="Actual vs Predicted"
        ).add_scatter(x=trends["Date"][split:], y=preds, mode="lines", name="Predicted"),
        px.bar(x=list(slopes.keys()), y=list(slopes.values()), title="Slope Comparison"),
        px.box(trends, y=top, title=f"{top} Distribution"),
        px.histogram(resid, nbins=20, title="Residuals Distribution"),
        px.imshow(trends.iloc[:, 1:].corr(), text_auto=True, title="Correlation Heatmap"),
        go.Figure(go.Indicator(mode="gauge+number", value=slopes[top], title={"text": "Slope Gauge"})),
        go.Figure(go.Scatterpolar(r=norm, theta=list(slopes.keys()), fill="toself")).update_layout(title="Normalized Slopes Radar"),
        px.pie(names=list(slopes.keys()), values=np.abs(list(slopes.values())), title="Slope Share"),
    ]


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
    return [
        px.bar(freq.nlargest(10, "support"), x="itemsets", y="support", title="Top Itemsets"),
        px.scatter(rules, x="support", y="confidence", size="lift", title="Rule Metrics"),
        px.pie(
            rules.nlargest(5, "lift"),
            names="antecedents",
            values="lift",
            title="Top Antecedents",
        ),
        px.sunburst(rules, path=["antecedents", "consequents"], values="lift", title="Rules Sunburst"),
        px.histogram(rules, x="support", title="Support Distribution"),
        px.imshow(rules[["support", "confidence", "lift"]], title="Metrics Heatmap"),
        go.Figure(
            go.Scatterpolar(
                r=[best["support"], best["confidence"], best["lift"]],
                theta=["Support", "Confidence", "Lift"],
                fill="toself",
            )
