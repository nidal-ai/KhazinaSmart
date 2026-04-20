"""
KhazBot AI engine for KhazinaSmart.
Answers natural language questions about inventory using Claude API or rule-based fallback.
Bilingual: detects French / English and answers in the same language.
"""
import pandas as pd
import re

KHAZBOT_SYSTEM_PROMPT = """You are KhazBot, the AI inventory assistant for KhazinaSmart — an intelligent inventory management platform.
You help business owners understand their stock health, KPIs, and charts.
You have access to real inventory data provided as context.
ALWAYS reference specific numbers, stores, departments, or products from the data.
Be concise, professional, and actionable.
Format numbers clearly (e.g., "1,250 units", "45,000 MAD").
CRITICAL: Detect the language of the user's question (French or English) and ALWAYS answer in the SAME language.
Never mix languages in one answer.
Never say you don't have data — analyze what's provided and give your best answer."""


FRENCH_HINTS = [
    "explique", "expliquer", "quoi", "que ", "qu'est", "qu’est", "quel", "quelle", "quels", "quelles",
    "pourquoi", "comment", "combien", "où", "ou ", "veut dire", "signifie", "moi", "moi-",
    "le ", "la ", "les ", "un ", "une ", "des ", "du ", "au ", "aux ", "ce ", "cette ", "ces ",
    "mon ", "ma ", "mes ", "ton ", "ta ", "tes ", "son ", "sa ", "ses ",
    "graphique", "tableau", "courbe", "tendance", "vente", "ventes", "magasin", "magasins",
    "catégorie", "catégories", "produit", "produits", "stock", "surstock", "rupture",
    "budget", "argent", "revenu", "revenus", "santé", "alerte", "alertes", "kpi", "kpis",
    "afficher", "montrer", "donner", "dire", "résumé", "résumer", "analyser",
    "trop", "commander", "acheter", "meilleur", "meilleure", "haut", "risque",
]


def detect_language(text: str) -> str:
    """Return 'fr' if French, else 'en'. Uses keyword + accent heuristic."""
    if not text:
        return "en"
    t = " " + text.lower().strip() + " "
    if re.search(r"[àâäéèêëïîôöùûüÿçœæ]", t):
        return "fr"
    score = sum(1 for w in FRENCH_HINTS if w in t)
    return "fr" if score >= 2 else "en"


def format_inventory_context(alerts_df: pd.DataFrame) -> str:
    """Convert top rows of alerts_df to a clean string summary for LLM context."""
    top = alerts_df.head(50)
    total = len(alerts_df)
    overstock_count = (alerts_df["status"] == "Overstock").sum()
    stockout_count = (alerts_df["status"] == "Stockout Risk").sum()
    healthy_count = (alerts_df["status"] == "Healthy").sum()

    top_overstock = alerts_df[alerts_df["status"] == "Overstock"].head(5)
    top_stockout = alerts_df[alerts_df["status"] == "Stockout Risk"].head(5)

    total_risk = (alerts_df["risk_score"].sum() * 50).round(0)

    lines = [
        f"=== INVENTORY SUMMARY ===",
        f"Total items monitored: {total:,}",
        f"Overstock: {overstock_count} | Stockout Risk: {stockout_count} | Healthy: {healthy_count}",
        f"Total capital at risk: {total_risk:,.0f} MAD",
        "",
        "TOP 5 OVERSTOCK ITEMS:",
    ]
    for _, row in top_overstock.iterrows():
        lines.append(
            f"  Store {row['Store']}, Dept {row['Dept']} — Risk: {row['risk_score']:.1f} — {row['action_needed']}"
        )

    lines.append("\nTOP 5 STOCKOUT RISK ITEMS:")
    for _, row in top_stockout.iterrows():
        lines.append(
            f"  Store {row['Store']}, Dept {row['Dept']} — Risk: {row['risk_score']:.1f} — {row['action_needed']}"
        )

    return "\n".join(lines)


def answer_inventory_question(question: str, alerts_df: pd.DataFrame, api_key: str = None) -> str:
    """Answer a natural language inventory question using Claude API or rule-based fallback."""
    context = format_inventory_context(alerts_df)
    lang = detect_language(question)

    if api_key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            lang_instruction = (
                "The user wrote in FRENCH. You MUST answer in FRENCH only."
                if lang == "fr"
                else "The user wrote in ENGLISH. You MUST answer in ENGLISH only."
            )
            user_message = f"{lang_instruction}\n\nInventory data:\n{context}\n\nQuestion: {question}"
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=512,
                system=KHAZBOT_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
            return response.content[0].text
        except Exception:
            pass

    return _rule_based_answer(question, alerts_df, lang)


def _rule_based_answer(question: str, alerts_df: pd.DataFrame, lang: str) -> str:
    q = question.lower()
    overstock = alerts_df[alerts_df["status"] == "Overstock"]
    stockout = alerts_df[alerts_df["status"] == "Stockout Risk"]
    healthy = alerts_df[alerts_df["status"] == "Healthy"]
    total = len(alerts_df)
    total_risk = (alerts_df["risk_score"].sum() * 50)
    overstock_cost = (overstock["risk_score"].sum() * 50)
    stockout_cost = (stockout["risk_score"].sum() * 50)

    is_kpi = any(k in q for k in ["kpi", "kpis", "indicateur", "indicateurs", "metric", "metrics"])
    is_explain = any(k in q for k in ["explain", "explique", "expliquer", "what does", "que veut", "signifie", "veut dire", "comprendre", "meaning"])
    is_chart_word = any(k in q for k in ["chart", "charts", "graph", "graphs", "graphique", "graphiques", "courbe", "tableau", "trend", "tendance", "diagramme"])

    is_budget = any(k in q for k in ["budget", "cost", "coût", "capital", "mad", "argent", "financier", "money", "dirham"])
    is_order = any(k in q for k in ["order", "acheter", "commander", "rupture", "manque", "restock", "réapprov", "stockout"])
    is_overstock = any(k in q for k in ["overstock", "surstock", "surplus", "trop de stock", "excès"])
    is_best = any(k in q for k in ["best", "meilleur", "meilleure", "top store", "top magasin", "performe", "performance", "highest", "plus haut"])

    # KPI EXPLANATION
    if is_kpi:
        if lang == "fr":
            return (
                f"📊 **Explication des KPI de KhazinaSmart :**\n\n"
                f"**1. Revenu Total** — Somme des ventes de tous les magasins sur la période. "
                f"C'est votre chiffre d'affaires global, l'indicateur principal de performance.\n\n"
                f"**2. Magasins Actifs** — Nombre de magasins présents dans vos données. "
                f"Permet de comparer les performances entre points de vente.\n\n"
                f"**3. Catégories** — Nombre de groupes de produits (ex: Boissons, Boulangerie). "
                f"Plus il y en a, plus votre gamme est diversifiée.\n\n"
                f"**4. Alertes Surstock** — Produits avec trop de stock par rapport à la demande prévue. "
                f"Vous avez actuellement **{len(overstock)} alertes** = capital bloqué : **{overstock_cost:,.0f} MAD**.\n\n"
                f"**5. Risques de Rupture** — Produits qui risquent de manquer. "
                f"Actuellement : **{len(stockout)} produits** = ventes potentielles perdues : **{stockout_cost:,.0f} MAD**.\n\n"
                f"💡 **Recommandation :** Concentrez-vous sur les **{len(overstock) + len(stockout)} produits à risque** pour protéger votre trésorerie."
            )
        return (
            f"📊 **KhazinaSmart KPI Explained:**\n\n"
            f"**1. Total Revenue** — Sum of all sales across all stores in the period. "
            f"This is your global turnover, the main performance indicator.\n\n"
            f"**2. Active Stores** — Number of stores in your dataset. "
            f"Used to compare performance between locations.\n\n"
            f"**3. Categories** — Number of product groups (e.g. Beverages, Bakery). "
            f"More categories = more diversified range.\n\n"
            f"**4. Overstock Alerts** — Products with too much stock vs forecasted demand. "
            f"You currently have **{len(overstock)} alerts** = capital locked: **{overstock_cost:,.0f} MAD**.\n\n"
            f"**5. Stockout Risks** — Products that may run out. "
            f"Currently: **{len(stockout)} items** = potential lost sales: **{stockout_cost:,.0f} MAD**.\n\n"
            f"💡 **Recommendation:** Focus on the **{len(overstock) + len(stockout)} at-risk items** to protect your cash flow."
        )

    # CHART EXPLANATION (only when an explicit chart/graph word is present)
    if is_chart_word and not (is_budget or is_order or is_overstock or is_best):
        if lang == "fr":
            return (
                f"📈 **Explication des graphiques :**\n\n"
                f"**Tendance Hebdomadaire** — La courbe violette montre l'évolution du revenu semaine par semaine. "
                f"Les points rouges représentent les semaines de promotion. "
                f"Cherchez les pics : ils correspondent souvent à des fêtes ou des promos.\n\n"
                f"**Revenu par Catégorie** — Barres horizontales triées par chiffre d'affaires. "
                f"Les catégories en haut sont vos meilleures ventes.\n\n"
                f"**Revenu par Magasin** — Performance comparée de chaque point de vente. "
                f"Le magasin le plus haut est votre champion.\n\n"
                f"💡 **À retenir :** Utilisez ces graphiques pour identifier vos produits stars, "
                f"vos magasins performants, et l'impact de vos promotions."
            )
        return (
            f"📈 **Charts Explained:**\n\n"
            f"**Weekly Revenue Trend** — The purple line shows revenue evolution week by week. "
            f"Red dots = promotional weeks. Look for peaks: they often match holidays or promos.\n\n"
            f"**Revenue by Category** — Horizontal bars sorted by sales. "
            f"Top categories = your best sellers.\n\n"
            f"**Revenue by Store** — Compared performance per location. "
            f"The highest bar is your champion store.\n\n"
            f"💡 **Takeaway:** Use these charts to spot star products, "
            f"top-performing stores, and the impact of promotions."
        )

    # OVERSTOCK
    if is_overstock or "trop" in q:
        if overstock.empty:
            return "✅ Aucune situation de surstock détectée." if lang == "fr" else "✅ No overstock situations detected."
        if lang == "fr":
            lines = [f"🔴 **{len(overstock)} situations de surstock détectées :**\n"]
            for _, row in overstock.head(5).iterrows():
                lines.append(f"- **Magasin {row['Store']}, Rayon {row['Dept']}** — Score de risque : {row['risk_score']:.1f} — {row['action_needed']}")
        else:
            lines = [f"🔴 **{len(overstock)} overstock situations detected:**\n"]
            for _, row in overstock.head(5).iterrows():
                lines.append(f"- **Store {row['Store']}, Dept {row['Dept']}** — Risk score: {row['risk_score']:.1f} — {row['action_needed']}")
        return "\n".join(lines)

    # STOCKOUT / ORDER
    if is_order:
        if stockout.empty:
            return "✅ Aucun risque de rupture immédiat. Vos niveaux de stock sont sains." if lang == "fr" else "✅ No immediate stockout risks. Your inventory levels are healthy."
        if lang == "fr":
            lines = [f"📦 **{len(stockout)} produits à réapprovisionner cette semaine :**\n"]
            for _, row in stockout.head(5).iterrows():
                lines.append(f"- **Magasin {row['Store']}, Rayon {row['Dept']}** — {row['action_needed']} — Risque : {row['risk_score']:.1f}")
        else:
            lines = [f"📦 **{len(stockout)} items need restocking this week:**\n"]
            for _, row in stockout.head(5).iterrows():
                lines.append(f"- **Store {row['Store']}, Dept {row['Dept']}** — {row['action_needed']} — Risk: {row['risk_score']:.1f}")
        return "\n".join(lines)

    # BUDGET / FINANCIAL
    if is_budget:
        if lang == "fr":
            return (
                f"💰 **Résumé du risque financier :**\n\n"
                f"- Capital total à risque : **{total_risk:,.0f} MAD**\n"
                f"- Exposition surstock : **{overstock_cost:,.0f} MAD** ({len(overstock)} produits)\n"
                f"- Ventes perdues (rupture) : **{stockout_cost:,.0f} MAD** ({len(stockout)} produits)\n"
                f"- Produits à risque : **{len(overstock) + len(stockout)}** sur {len(alerts_df)}"
            )
        return (
            f"💰 **Financial Risk Summary:**\n\n"
            f"- Total capital at risk: **{total_risk:,.0f} MAD**\n"
            f"- Overstock exposure: **{overstock_cost:,.0f} MAD** ({len(overstock)} items)\n"
            f"- Stockout lost sales: **{stockout_cost:,.0f} MAD** ({len(stockout)} items)\n"
            f"- Items at risk: **{len(overstock) + len(stockout)}** out of {len(alerts_df)}"
        )

    # BEST STORE
    if is_best or "store" in q or "magasin" in q:
        if "predicted_demand" in alerts_df.columns:
            store_sales = alerts_df.groupby("Store")["predicted_demand"].mean().sort_values(ascending=False)
            top_store = store_sales.index[0]
            if lang == "fr":
                return (
                    f"🏪 **Magasin le plus performant : Magasin {top_store}**\n\n"
                    f"Ventes hebdomadaires moyennes prévues : **{store_sales.iloc[0]:,.0f} unités**\n\n"
                    f"Top 3 magasins par demande prévue :\n"
                    + "\n".join([f"- Magasin {s} : {v:,.0f}" for s, v in store_sales.head(3).items()])
                )
            return (
                f"🏪 **Best performing store: Store {top_store}**\n\n"
                f"Average predicted weekly sales: **{store_sales.iloc[0]:,.0f} units**\n\n"
                f"Top 3 stores by predicted demand:\n"
                + "\n".join([f"- Store {s}: {v:,.0f}" for s, v in store_sales.head(3).items()])
            )

    # DEFAULT SUMMARY
    if lang == "fr":
        return (
            f"📊 **Résumé de la santé de votre inventaire :**\n\n"
            f"- Produits surveillés : **{total:,}**\n"
            f"- 🔴 Alertes surstock : **{len(overstock)}** ({len(overstock)/total*100:.1f}%)\n"
            f"- 🟡 Risques de rupture : **{len(stockout)}** ({len(stockout)/total*100:.1f}%)\n"
            f"- 🟢 Produits sains : **{len(healthy)}** ({len(healthy)/total*100:.1f}%)\n\n"
            f"Votre priorité : agir sur les **{len(overstock)+len(stockout)} produits à risque** "
            f"pour protéger votre trésorerie ({total_risk:,.0f} MAD en jeu).\n\n"
            f"💡 Demandez-moi : *\"explique-moi les KPI\"*, *\"explique les graphiques\"*, "
            f"*\"que dois-je commander ?\"* ou *\"quel est le budget à risque ?\"*"
        )
    return (
        f"📊 **KhazinaSmart Inventory Health Summary:**\n\n"
        f"- Total items monitored: **{total:,}**\n"
        f"- 🔴 Overstock alerts: **{len(overstock)}** ({len(overstock)/total*100:.1f}%)\n"
        f"- 🟡 Stockout risks: **{len(stockout)}** ({len(stockout)/total*100:.1f}%)\n"
        f"- 🟢 Healthy items: **{len(healthy)}** ({len(healthy)/total*100:.1f}%)\n\n"
        f"Top priority: address the **{len(overstock)+len(stockout)} at-risk items** "
        f"to protect cash flow ({total_risk:,.0f} MAD at stake).\n\n"
        f"💡 Try asking: *\"explain the KPIs\"*, *\"explain the charts\"*, "
        f"*\"what should I order?\"* or *\"what is the budget at risk?\"*"
    )


def get_starter_questions() -> list:
    """Return list of suggested starter questions for KhazBot."""
    return [
        "Explain the KPIs / Explique-moi les KPI 📊",
        "Explain the charts / Explique les graphiques 📈",
        "Which products are most overstocked? 🔴",
        "What should I order this week? 📦",
        "What is the total budget at risk? 💰",
    ]
