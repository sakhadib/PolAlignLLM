import plotly.graph_objects as go
import pandas as pd

# Data from the provided JSON
data = [
    {"model_name": "xai_grok_4", "score": 0.413233},
    {"model_name": "ibm_granite_3_3_8b_instruct", "score": -0.556900},
    {"model_name": "openrouter_sonoma_dusk_alpha", "score": -0.592000},
    {"model_name": "cohere_command_r_08_2024", "score": -0.601867},
    {"model_name": "minimax_minimax_01", "score": -0.674333},
    {"model_name": "mistralai_mistral_medium_3", "score": -0.775100},
    {"model_name": "meituan_longcat_flash_chat", "score": -0.804467},
    {"model_name": "microsoft_phi_3_5_mini_128k_instruct", "score": -0.834800},
    {"model_name": "moonshotai_kimi_k2_instruct", "score": -0.849867},
    {"model_name": "meta_llama_3_70b_instruct", "score": -0.857200},
    {"model_name": "meta_llama_llama_4_scout", "score": -0.857267},
    {"model_name": "google_gemma_3_27b_it", "score": -0.892867},
    {"model_name": "openai_gpt_4_1_nano", "score": -0.896767},
    {"model_name": "openai_gpt_4o", "score": -0.906967},
    {"model_name": "meta_llama_3_8b_instruct", "score": -0.934567},
    {"model_name": "anthropic_claude_4_sonnet", "score": -0.963133},
    {"model_name": "openai_gpt_5_nano", "score": -0.963533},
    {"model_name": "deepseekAi_deepseek_v3", "score": -0.975700},
    {"model_name": "qwen_qwen3_235b_a22b_instruct_2507", "score": -1.016367},
    {"model_name": "meta_llama_4_maverick_instruct", "score": -1.051567},
    {"model_name": "openai_gpt_oss_120b", "score": -1.062467},
    {"model_name": "openai_gpt_4_1", "score": -1.121533},
    {"model_name": "openai_gpt_5", "score": -1.210467},
    {"model_name": "openai_gpt_4_1_mini", "score": -1.238333},
    {"model_name": "google_gemini_2_5_flash", "score": -1.295567},
    {"model_name": "google_gemini_2_5_pro", "score": -1.371400},
    {"model_name": "openai_gpt_5_mini", "score": -1.408467}
]

# Create DataFrame and sort by score (highest to lowest)
df = pd.DataFrame(data)
df = df.sort_values('score', ascending=False).reset_index(drop=True)

# Create the lollipop chart
fig = go.Figure()

# Add lines from 0 to each score
for i, row in df.iterrows():
    color = '#DB4545' if row['score'] >= 0 else '#1FB8CD'  # Red for positive, blue for negative
    
    # Add line
    fig.add_trace(go.Scatter(
        x=[0, row['score']], 
        y=[i, i],
        mode='lines',
        line=dict(color=color, width=3),
        showlegend=False,
        hoverinfo='skip'
    ))

# Add dots at the end of each line
positive_scores = df[df['score'] >= 0]
negative_scores = df[df['score'] < 0]

if not positive_scores.empty:
    fig.add_trace(go.Scatter(
        x=positive_scores['score'],
        y=positive_scores.index,
        mode='markers',
        marker=dict(color='#DB4545', size=10, line=dict(width=1, color='darkred')),
        name='Right-leaning (Positive)',
        hovertemplate='%{text}<br>Score: %{x:.3f}<extra></extra>',
        text=positive_scores['model_name']
    ))

if not negative_scores.empty:
    fig.add_trace(go.Scatter(
        x=negative_scores['score'],
        y=negative_scores.index,
        mode='markers',
        marker=dict(color='#1FB8CD', size=10, line=dict(width=1, color='darkblue')),
        name='Left-leaning (Negative)',
        hovertemplate='%{text}<br>Score: %{x:.3f}<extra></extra>',
        text=negative_scores['model_name']
    ))

# Add vertical reference line at x=0
fig.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1)

# Update layout
fig.update_layout(
    title=dict(
        text='Model Political Bias Scores',
        font=dict(size=24, family='Arial Black'),
        x=0.5,
        xanchor='center'
    ),
    xaxis_title=dict(
        text='Political Bias Score',
        font=dict(size=16, family='Arial')
    ),
    yaxis_title=dict(
        text='Models',
        font=dict(size=16, family='Arial')
    ),
    yaxis=dict(
        tickmode='array',
        tickvals=list(range(len(df))),
        ticktext=df['model_name'].tolist(),
        autorange='reversed',  # Highest scores at top
        tickfont=dict(size=10, family='Arial'),
        side='left'
    ),
    showlegend=True,
    legend=dict(
        orientation='h', 
        yanchor='bottom', 
        y=1.02, 
        xanchor='center', 
        x=0.5,
        font=dict(size=14, family='Arial')
    ),
    xaxis=dict(
        showgrid=True, 
        gridwidth=1, 
        gridcolor='lightgray', 
        range=[-3, 3],
        tickfont=dict(size=12, family='Arial'),
        dtick=0.5,
        zeroline=True,
        zerolinecolor='black',
        zerolinewidth=2
    ),
    plot_bgcolor='white',
    paper_bgcolor='white',
    margin=dict(l=300, r=50, t=80, b=60),  # Increased left margin for full model names
    height=max(400, len(df) * 25 + 150),  # Dynamic height based on number of models
    font=dict(family='Arial')
)

# Update traces to clip on axis
fig.update_traces(cliponaxis=False)

# Save the chart
fig.write_image("model_scores.png", width=1400, height=800, scale=2)
fig.write_image("model_scores.svg", format="svg", width=1400, height=800)

print("Chart created and saved successfully!")
print(f"Number of models: {len(df)}")
print(f"Positive scores: {len(positive_scores)}")
print(f"Negative scores: {len(negative_scores)}")