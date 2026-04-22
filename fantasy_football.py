import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Fantasy Football Draft Analysis", layout="wide")

@st.cache_data
def load_data():
    draft = pd.read_csv("draft_results.csv")
    rosters = pd.read_csv("weekly_team_rosters.csv")
    matchups = pd.read_csv("sleeper_matchups.csv")
    return draft, rosters, matchups

def anonymize_users(matchups_df, rosters_df):
    unique_users = sorted(matchups_df["display_name"].dropna().unique())
    user_map = {name: f"User {i+1}" for i, name in enumerate(unique_users)}
    matchups_df = matchups_df.copy()
    rosters_df = rosters_df.copy()
    matchups_df["display_name"] = matchups_df["display_name"].map(user_map)
    rosters_df["display_name"] = rosters_df["display_name"].map(user_map)
    return matchups_df, rosters_df

def add_win_column(matchups_df):
    matchups_df = matchups_df.copy()
    matchups_df["result"] = matchups_df.groupby(["week", "matchup_id"])["points"].transform(
        lambda x: ["W" if p == x.max() else "L" for p in x]
    )
    matchups_df["win"] = (matchups_df["result"] == "W").astype(int)
    return matchups_df

@st.cache_data
def build_elite_rb_wr_analysis(draft, rosters, matchups):
    elite = draft[(draft["round"] <= 2) & (draft["position"].isin(["RB", "WR"]))].copy()

    elite_perf = rosters.merge(elite[["player_name"]], on="player_name", how="inner")
    elite_perf = elite_perf[elite_perf["is_starter"] == True].copy()
    elite_perf = elite_perf.merge(
        matchups[["week", "roster_id", "win", "points"]],
        on=["week", "roster_id"],
        how="left"
    )

    eligible_players = (
        elite_perf.groupby("player_name")["week"]
        .count()
        .reset_index(name="starter_weeks")
    )
    eligible_players = eligible_players[eligible_players["starter_weeks"] > 12]

    elite_perf_filtered = elite_perf.merge(
        eligible_players[["player_name"]],
        on="player_name",
        how="inner"
    )

    player_stats = elite_perf_filtered.groupby(["player_name", "position"]).agg(
        starter_weeks=("week", "count"),
        win_rate=("win", "mean"),
        avg_weekly_points=("points", "mean"),
        std_weekly_points=("points", "std")
    ).reset_index()

    player_stats["win_rate"] = player_stats["win_rate"].round(3)
    player_stats["avg_weekly_points"] = player_stats["avg_weekly_points"].round(2)
    player_stats["std_weekly_points"] = player_stats["std_weekly_points"].round(2)
    player_stats = player_stats.sort_values(["avg_weekly_points", "win_rate"], ascending=[False, False])

    rb_wr_summary = elite_perf_filtered.groupby("position").agg(
        total_starts=("week", "count"),
        avg_win_rate=("win", "mean"),
        avg_points=("points", "mean"),
        std_points=("points", "std")
    ).reset_index()

    rb_wr_summary["avg_win_rate"] = rb_wr_summary["avg_win_rate"].round(3)
    rb_wr_summary["avg_points"] = rb_wr_summary["avg_points"].round(2)
    rb_wr_summary["std_points"] = rb_wr_summary["std_points"].round(2)

    team_elite = (
        elite_perf_filtered.groupby(["display_name", "position"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    team_results = matchups.groupby("display_name")["win"].mean().reset_index(name="win_rate")
    team_analysis = team_results.merge(team_elite, on="display_name", how="left")

    for col in ["RB", "WR"]:
        if col not in team_analysis.columns:
            team_analysis[col] = 0

    team_analysis[["RB", "WR"]] = team_analysis[["RB", "WR"]].fillna(0).astype(int)
    team_analysis["win_rate"] = team_analysis["win_rate"].round(3)
    team_analysis = team_analysis.sort_values("win_rate", ascending=False)

    return player_stats, rb_wr_summary, team_analysis

@st.cache_data
def build_qb_moneyball(draft, rosters, matchups):
    qb_draft = draft[draft["position"] == "QB"].copy()

    qb_perf = rosters.merge(
        qb_draft[["player_name", "round", "pick_no"]],
        on="player_name",
        how="inner"
    )

    qb_perf = qb_perf[qb_perf["is_starter"] == True].copy()
    qb_perf = qb_perf.merge(
        matchups[["week", "roster_id", "points"]],
        on=["week", "roster_id"],
        how="left"
    )

    team_avg = matchups.groupby("roster_id")["points"].mean().reset_index(name="team_avg")
    qb_perf = qb_perf.merge(team_avg, on="roster_id", how="left")
    qb_perf["qb_impact"] = qb_perf["points"] - qb_perf["team_avg"]

    qb_perf = qb_perf.merge(
        matchups[["week", "roster_id", "win"]],
        on=["week", "roster_id"],
        how="left"
    )

    qb_stats = qb_perf.groupby("player_name").agg(
        avg_impact=("qb_impact", "mean"),
        win_rate=("win", "mean"),
        starts=("week", "count"),
        pick_no=("pick_no", "first"),
        round=("round", "first")
    ).reset_index()

    qb_stats["cost"] = qb_stats["pick_no"] ** 0.5
    qb_stats["value_score"] = (qb_stats["avg_impact"] * qb_stats["win_rate"]) / qb_stats["cost"]

    qb_stats["avg_impact"] = qb_stats["avg_impact"].round(2)
    qb_stats["win_rate"] = qb_stats["win_rate"].round(3)
    qb_stats["value_score"] = qb_stats["value_score"].round(4)
    qb_stats = qb_stats.sort_values("value_score", ascending=False)

    def qb_tier(round_num):
        if round_num <= 3:
            return "Early"
        elif round_num <= 8:
            return "Mid"
        return "Late"

    qb_stats["tier"] = qb_stats["round"].apply(qb_tier)

    tier_summary = qb_stats.groupby("tier").agg(
        avg_impact=("avg_impact", "mean"),
        avg_value=("value_score", "mean"),
        count=("player_name", "count")
    ).reset_index()

    tier_order = pd.CategoricalDtype(categories=["Early", "Mid", "Late"], ordered=True)
    tier_summary["tier"] = tier_summary["tier"].astype(tier_order)
    tier_summary = tier_summary.sort_values("tier")

    return qb_stats, tier_summary

@st.cache_data
def build_mid_round_analysis(draft, rosters, matchups):
    mid_round = draft[(draft["round"] >= 4) & (draft["round"] <= 10)].copy()

    mid_perf = rosters.merge(mid_round[["player_name"]], on="player_name", how="inner")
    mid_perf = mid_perf[mid_perf["is_starter"] == True].copy()
    mid_perf = mid_perf.merge(
        matchups[["week", "roster_id", "points", "win"]],
        on=["week", "roster_id"],
        how="left"
    )

    team_mid_stats = mid_perf.groupby("display_name").agg(
        mid_avg_points=("points", "mean"),
        mid_std_points=("points", "std"),
        mid_usage=("week", "count")
    ).reset_index()

    team_results = matchups.groupby("display_name")["win"].mean().reset_index(name="win_rate")
    team_mid_stats = team_mid_stats.merge(team_results, on="display_name", how="left")

    team_mid_stats["mid_avg_points"] = team_mid_stats["mid_avg_points"].round(2)
    team_mid_stats["mid_std_points"] = team_mid_stats["mid_std_points"].round(2)
    team_mid_stats["win_rate"] = team_mid_stats["win_rate"].round(3)
    team_mid_stats = team_mid_stats.sort_values("win_rate", ascending=False)

    corr = team_mid_stats["mid_std_points"].corr(team_mid_stats["win_rate"])
    return team_mid_stats, corr

def make_bar_chart(df, x_col, y_col, title, xlabel, ylabel, rotate=False):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(df[x_col], df[y_col])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if rotate:
        plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig

def make_scatter_with_labels(df, x_col, y_col, label_col, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df[x_col], df[y_col])
    for _, row in df.iterrows():
        ax.text(row[x_col], row[y_col], str(row[label_col]), fontsize=8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    return fig

st.title("Fantasy Football Draft Analysis Dashboard")
st.caption("Built from Sleeper league data with anonymized team names.")

draft, rosters, matchups = load_data()
matchups, rosters = anonymize_users(matchups, rosters)
matchups = add_win_column(matchups)

player_stats, rb_wr_summary, team_analysis = build_elite_rb_wr_analysis(draft, rosters, matchups)
qb_stats, tier_summary = build_qb_moneyball(draft, rosters, matchups)
team_mid_stats, mid_corr = build_mid_round_analysis(draft, rosters, matchups)

tab1, tab2, tab3, tab4 = st.tabs([
    "Overview",
    "Question 1: Elite RB vs WR",
    "Question 2: QB Moneyball",
    "Question 3: Mid-Round Variability"
])

with tab1:
    c1, c2, c3 = st.columns(3)
    wr_row = rb_wr_summary[rb_wr_summary["position"] == "WR"].iloc[0]
    rb_row = rb_wr_summary[rb_wr_summary["position"] == "RB"].iloc[0]

    c1.metric("Elite WR avg win rate", f"{wr_row['avg_win_rate']:.3f}")
    c2.metric("Elite RB avg win rate", f"{rb_row['avg_win_rate']:.3f}")
    c3.metric("Mid-round variance correlation", f"{mid_corr:.3f}")

    st.subheader("Top-level takeaway")
    st.write(
        "This dashboard combines three questions: whether elite WRs or RBs offered better value, "
        "which QBs produced the best value relative to draft cost, and how mid-round variability related to win rate."
    )

    st.subheader("Anonymized team snapshot")
    st.dataframe(team_analysis, use_container_width=True)

    st.subheader("Quick rankings")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Top elite players by average weekly points")
        st.dataframe(player_stats.head(10), use_container_width=True)
    with col2:
        st.write("Top QB values")
        st.dataframe(qb_stats.head(10), use_container_width=True)

with tab2:
    st.subheader("Elite RB vs WR summary")
    st.dataframe(rb_wr_summary, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = make_bar_chart(
            player_stats.head(12),
            "player_name",
            "avg_weekly_points",
            "Elite RB/WRs: Average Weekly Team Points",
            "Player",
            "Average Weekly Team Points",
            rotate=True
        )
        st.pyplot(fig)
    with col2:
        fig = make_scatter_with_labels(
            player_stats,
            "std_weekly_points",
            "win_rate",
            "player_name",
            "Elite RB/WR Variability vs Win Rate",
            "Weekly Points Variability (Std Dev)",
            "Win Rate"
        )
        st.pyplot(fig)

    st.subheader("Player table")
    st.dataframe(player_stats, use_container_width=True)

with tab3:
    st.subheader("QB Moneyball results")
    st.dataframe(qb_stats, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = make_bar_chart(
            qb_stats.head(12),
            "player_name",
            "value_score",
            "QB Moneyball: Performance Relative to Draft Cost",
            "QB",
            "Value Score",
            rotate=True
        )
        st.pyplot(fig)
    with col2:
        fig = make_scatter_with_labels(
            qb_stats,
            "pick_no",
            "avg_impact",
            "player_name",
            "QB Draft Cost vs Impact",
            "Draft Pick Number",
            "Average Impact Above Team Baseline"
        )
        st.pyplot(fig)

    st.subheader("QB tier summary")
    st.dataframe(tier_summary, use_container_width=True)
    fig = make_bar_chart(
        tier_summary,
        "tier",
        "avg_value",
        "QB Moneyball Value by Draft Tier",
        "QB Draft Tier",
        "Average Value Score"
    )
    st.pyplot(fig)

with tab4:
    st.subheader("Mid-round team analysis")
    st.write(f"Correlation between mid-round variability and win rate: **{mid_corr:.3f}**")
    st.dataframe(team_mid_stats, use_container_width=True)

    fig = make_scatter_with_labels(
        team_mid_stats,
        "mid_std_points",
        "win_rate",
        "display_name",
        "Mid-Round Variability vs Win Rate",
        "Mid-Round Variability (Std Dev)",
        "Win Rate"
    )
    st.pyplot(fig)

st.sidebar.header("How to run")
st.sidebar.code("streamlit run app.py", language="bash")
st.sidebar.write("Put app.py in the same folder as draft_results.csv, weekly_team_rosters.csv, and sleeper_matchups.csv.")
