import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np

# Load the data from the CSV file
file_path = r"D:\Research\Political_Bias_of_LLMs\PoliticalCompass\7_summerize\pc_scores_summary_by_model.csv"
df = pd.read_csv(file_path)

# Extract provider from model name
def extract_provider(model_name):
    """Extract provider from model name, handling special cases"""
    if model_name.startswith('deepseek_ai_'):
        return 'deepseek_ai'
    elif model_name.startswith('meta_meta_'):
        return 'meta'
    elif model_name.startswith('meta_llama_'):
        return 'meta'
    else:
        return model_name.split('_')[0]

df['provider'] = df['model_name'].apply(extract_provider)

# Create abbreviated model names (max 15 chars)
def abbreviate_model_name(name):
    # Remove provider prefixes and shorten
    prefixes_to_remove = ['anthropic_', 'openai_', 'meta_', 'ibm_', 'moonshotai_', 'deepseek_ai_', 
                         'cohere_', 'google_', 'meituan_', 'qwen_', 'together_']
    for prefix in prefixes_to_remove:
        name = name.replace(prefix, '')
    
    # Specific model name abbreviations
    abbreviations = {
        'claude_4_sonnet': 'claude4',
        'granite_granite_3_3_8b_instruct': 'granite3',
        'llama_4_maverick_instruct': 'llama4_mav',
        'meta_llama_3_70b_instruct': 'llama3_70b',
        'meta_llama_3_8b_instruct': 'llama3_8b',
        'llama_llama_4_scout': 'llama4_scout',
        'kimi_k2_instruct': 'kimi_k2',
        'gpt_4_1_mini': 'gpt4.1_mini',
        'gpt_4_1_nano': 'gpt4.1_nano',
        'gpt_4_1': 'gpt4.1',
        'gpt_4o': 'gpt4o',
        'gpt_5_mini': 'gpt5_mini',
        'gpt_5_nano': 'gpt5_nano',
        'gpt_5': 'gpt5',
        'gpt_oss_120b': 'gpt_oss',
        'deepseek_v3': 'deepseek3',
        'command_r_08_2024': 'command_r',
        'gemini_2_5_flash': 'gemini2.5',
        'gemma_3_27b_it': 'gemma3_27b',
        'longcat_flash_chat': 'longcat'
    }
    
    for original, abbreviated in abbreviations.items():
        name = name.replace(original, abbreviated)
    
    return name[:15]

df['short_name'] = df['model_name'].apply(abbreviate_model_name)

# Automatically assign colors to providers
unique_providers = sorted(df['provider'].unique())
# Use a high-quality color palette - combination of distinct colors for data visualization
color_palette = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', 
    '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78',
    '#98df8a', '#ff9896', '#c5b0d5', '#c49c94', '#f7b6d3', '#c7c7c7',
    '#393b79', '#5254a3', '#6b6ecf', '#9c9ede', '#637939', '#8ca252',
    '#b5cf6b', '#cedb9c', '#8c6d31', '#bd9e39', '#e7ba52', '#e7cb94',
    '#843c39', '#ad494a', '#d6616b', '#e7969c', '#7b4173', '#a55194',
    '#ce6dbd', '#de9ed6'
]
provider_colors = {provider: color_palette[i % len(color_palette)] 
                  for i, provider in enumerate(unique_providers)}

# Print color assignments for reference
print("Auto-assigned provider colors:")
for provider, color in provider_colors.items():
    print(f"  {provider}: {color}")

# Define the colors for each quadrant
colors = {
    "Top Left": "#FF8686",  # Red (Authoritarian-Left)
    "Top Right": "#6868E3",  # Blue (Authoritarian-Right)
    "Bottom Left": "#ADEEAD",  # Green (Libertarian-Left)
    "Bottom Right": "#7C4C7C"  # Purple (Libertarian-Right)
}

# Create figure
fig = go.Figure()

# Add quadrant backgrounds
# Bottom Left (Libertarian-Left)
fig.add_shape(
    type="rect",
    x0=-10, y0=-10, x1=0, y1=0,
    fillcolor=colors["Bottom Left"],
    opacity=0.3,
    line=dict(color="black", width=2)
)

# Top Left (Authoritarian-Left)
fig.add_shape(
    type="rect",
    x0=-10, y0=0, x1=0, y1=10,
    fillcolor=colors["Top Left"],
    opacity=0.3,
    line=dict(color="black", width=2)
)

# Bottom Right (Libertarian-Right)
fig.add_shape(
    type="rect",
    x0=0, y0=-10, x1=10, y1=0,
    fillcolor=colors["Bottom Right"],
    opacity=0.3,
    line=dict(color="black", width=2)
)

# Top Right (Authoritarian-Right)
fig.add_shape(
    type="rect",
    x0=0, y0=0, x1=10, y1=10,
    fillcolor=colors["Top Right"],
    opacity=0.3,
    line=dict(color="black", width=2)
)

# Add center lines
fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=3, opacity=0.6)
fig.add_vline(x=0, line_dash="solid", line_color="black", line_width=3, opacity=0.6)

# Add scatter points for each provider
for provider in df['provider'].unique():
    provider_data = df[df['provider'] == provider]
    
    fig.add_trace(go.Scatter(
        x=provider_data['econ_score_mean'],
        y=provider_data['soc_score_mean'],
        mode='markers+text',
        name=provider.capitalize(),
        marker=dict(
            color=provider_colors[provider],
            size=16,
            line=dict(width=2, color='white'),
            opacity=0.8
        ),
        text=provider_data['short_name'],
        textposition='middle right',
        textfont=dict(size=10, color='black'),
        hovertemplate='<b>%{text}</b><br>' +
                     'Econ: %{x:.2f}<br>' +
                     'Social: %{y:.2f}<br>' +
                     'Provider: ' + provider.capitalize() + '<br>' +
                     '<extra></extra>'
    ))

# Update layout
fig.update_layout(
    title=dict(
        text='LLM Political Compass',
        x=0.5,
        font=dict(size=20)
    ),
    xaxis=dict(
        title='Economic Score (Left ← → Right)',
        range=[-10, 10],
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        zeroline=True,
        zerolinewidth=3,
        zerolinecolor='black',
        tickmode='linear',
        tick0=-10,
        dtick=2
    ),
    yaxis=dict(
        title='Social Score (Libertarian ← → Authoritarian)',
        range=[-10, 10],
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        zeroline=True,
        zerolinewidth=3,
        zerolinecolor='black',
        tickmode='linear',
        tick0=-10,
        dtick=2
    ),
    width=900,
    height=900,
    plot_bgcolor='white',
    paper_bgcolor='white',
    legend=dict(
        title="Provider",
        orientation="v",
        yanchor="top",
        y=0.98,
        xanchor="left",
        x=1.02,
        font=dict(size=10),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="gray",
        borderwidth=1
    )
)

# Update traces
fig.update_traces(cliponaxis=False)

# Show the plot
fig.show()
