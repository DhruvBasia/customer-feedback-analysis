"""
Step 2 - Exploratory data analysis (EDA).

Creates six charts in outputs/figures/ and a text summary in outputs/eda_summary.txt.

Run:  python src/eda.py
"""
import pandas as pd

from common import BLUE, FIG_DIR, OUT_DIR, SENT_COLORS, SENTIMENTS, UNCLEAR_THEME, load_data, set_chart_style

plt, sns = set_chart_style()


def save(fig, name):
    fig.savefig(FIG_DIR / name)
    plt.close(fig)
    print("saved", name)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    neg = df[df["sentiment"] == "negative"]
    themed = neg[neg["complaint_theme"] != UNCLEAR_THEME]

    # 01 - how many messages of each sentiment
    counts = df["sentiment"].value_counts().reindex(SENTIMENTS)
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(counts.index, counts.values, color=[SENT_COLORS[s] for s in counts.index])
    ax.bar_label(bars, labels=[f"{v:,}\n({v / len(df):.0%})" for v in counts.values], padding=3)
    ax.set(title="Customer messages by sentiment", ylabel="Messages", ylim=(0, counts.max() * 1.2))
    save(fig, "01_sentiment_distribution.png")

    # 02 - sentiment share for each airline
    share = pd.crosstab(df["airline"], df["sentiment"], normalize="index")[SENTIMENTS]
    share = share.sort_values("negative")
    fig, ax = plt.subplots(figsize=(8, 4))
    left = pd.Series(0.0, index=share.index)
    for s in SENTIMENTS:
        ax.barh(share.index, share[s], left=left, color=SENT_COLORS[s], label=s)
        for y, (v, l) in enumerate(zip(share[s], left)):
            ax.text(l + v / 2, y, f"{v:.0%}", ha="center", va="center", color="white", fontsize=9)
        left += share[s]
    ax.set(title="Sentiment share by airline", xlim=(0, 1), xlabel="Share of the airline's messages")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.32), frameon=False)
    ax.grid(False)
    save(fig, "02_sentiment_by_airline.png")

    # 03 - daily trend (the first day has only a handful of messages, so it is left out)
    daily = pd.crosstab(df["date"], df["sentiment"])[SENTIMENTS]
    daily = daily[daily.sum(axis=1) >= 100]
    fig, ax = plt.subplots(figsize=(8, 4))
    for s in SENTIMENTS:
        ax.plot(daily.index, daily[s], marker="o", color=SENT_COLORS[s], label=s)
    ax.set(title="Messages per day by sentiment", ylabel="Messages")
    ax.set_xticks(daily.index, [d.strftime("%d %b") for d in daily.index])
    ax.legend(frameon=False)
    save(fig, "03_daily_sentiment_trend.png")

    # 04 - complaint themes
    themes = themed["complaint_theme"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(themes.index, themes.values, color=BLUE)
    ax.bar_label(bars, labels=[f"{v:,} ({v / len(themed):.0%})" for v in themes.values], padding=3)
    ax.set(title="Complaint themes in negative messages", xlabel="Messages", xlim=(0, themes.max() * 1.2))
    save(fig, "04_complaint_themes.png")

    # 05 - which theme hurts which airline
    heat = pd.crosstab(themed["airline"], themed["complaint_theme"], normalize="index")
    heat = heat[themes.index[::-1]]
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(heat, annot=True, fmt=".0%", cmap="Blues", cbar=False, linewidths=2, ax=ax)
    ax.set(title="Share of each airline's complaints by theme", xlabel="", ylabel="")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    save(fig, "05_theme_by_airline_heatmap.png")

    # 06 - message length by sentiment
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x="sentiment", y="word_count", order=SENTIMENTS, hue="sentiment",
                palette=SENT_COLORS, legend=False, ax=ax)
    ax.set(title="Message length by sentiment", xlabel="", ylabel="Words per message")
    save(fig, "06_message_length.png")

    # ---------------- text summary ----------------
    neg_rate = (df["sentiment"] == "negative").groupby(df["airline"]).mean().sort_values(ascending=False)
    lines = [
        "EDA SUMMARY - Customer Feedback Analysis",
        "=" * 44,
        f"Messages analysed      : {len(df):,}",
        f"Period                 : {df['created_at'].min():%d %b %Y} to {df['created_at'].max():%d %b %Y}",
        f"Airlines               : {df['airline'].nunique()}",
        "",
        "Sentiment split",
        *[f"  {s:<9}: {counts[s]:>6,}  ({counts[s] / len(df):.1%})" for s in SENTIMENTS],
        "",
        "Negative share by airline",
        *[f"  {a:<15}: {v:.1%}" for a, v in neg_rate.items()],
        "",
        f"Complaint themes ({len(themed):,} negative messages with a clear theme; "
        f"{len(neg) - len(themed):,} more were labelled \"{UNCLEAR_THEME}\")",
        *[f"  {t:<22}: {v:>5,}  ({v / len(themed):.1%})" for t, v in themes[::-1].items()],
        "",
        "Median words per message",
        *[f"  {s:<9}: {df.loc[df['sentiment'] == s, 'word_count'].median():.0f}" for s in SENTIMENTS],
    ]
    (OUT_DIR / "eda_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
