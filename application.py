# -----------------------------------
# WIN RATE STATS
# -----------------------------------

@st.cache_data(ttl=0)
def compute_win_rates():
    matches = pd.read_csv("matches.csv")

    valid_teams = set(team_data.keys())

    # Fix old IPL team names
    name_map = {
        "Delhi Daredevils": "Delhi Capitals",
        "Kings XI Punjab": "Punjab Kings",
        "Royal Challengers Bengaluru": "Royal Challengers Bangalore",
        "Rising Pune Supergiants": "Rising Pune Supergiant"
    }

    matches["team1"] = matches["team1"].replace(name_map)
    matches["team2"] = matches["team2"].replace(name_map)
    matches["winner"] = matches["winner"].replace(name_map)

    stats = {}

    for team in valid_teams:

        # Matches played by team
        played = matches[
            (matches["team1"] == team) |
            (matches["team2"] == team)
        ]

        # Only normal matches
        played = played[played["result"] == "normal"]

        # Wins count
        wins = played[played["winner"] == team].shape[0]

        # Total matches
        total = played.shape[0]

        # Win rate calculation
        win_rate = round((wins / total) * 100, 1) if total > 0 else 0

        stats[team] = {
            "wins": wins,
            "total": total,
            "rate": win_rate
        }

    return stats


# Load stats
win_stats = compute_win_rates()


# Refresh Button
if st.button("Refresh Team Stats"):
    st.cache_data.clear()
    st.rerun()