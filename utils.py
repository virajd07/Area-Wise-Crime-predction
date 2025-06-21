import plotly.express as px

def crime_trend_plot(df, state):
    state_df = df[df["STATE/UT"] == state].groupby("Year").sum(numeric_only=True).reset_index()
    crimes = [col for col in state_df.columns if col != "Year"]
    fig = px.line(state_df, x="Year", y=crimes, title=f"Crime Trends in {state}")
    fig.update_layout(
        template="plotly_dark",
        height=500,
        title_font_size=22,
        xaxis_title="Year",
        yaxis_title="Crime Count"
    )
    return fig
