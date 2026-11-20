import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy.stats import chi2

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
    '#98df8a', '#ff9896', '#c5b0d5', '#c49c94', '#f7b6d3', '#c7c7c7'
]
provider_colors = {provider: color_palette[i % len(color_palette)] 
                  for i, provider in enumerate(unique_providers)}

# Print color assignments for reference
print("Auto-assigned provider colors:")
for provider, color in provider_colors.items():
    print(f"  {provider}: {color}")

# Define the colors for each quadrant
quadrant_colors = {
    "Top Left": "#FF8686",  # Red (Authoritarian-Left)
    "Top Right": "#6868E3",  # Blue (Authoritarian-Right)
    "Bottom Left": "#ADEEAD",  # Green (Libertarian-Left)
    "Bottom Right": "#7C4C7C"  # Purple (Libertarian-Right)
}

def create_ellipse(mean_x, mean_y, std_x, std_y, confidence_level=0.95, n_points=100):
    """
    Create ellipse coordinates for given confidence level
    """
    # Chi-square critical value for 2 degrees of freedom
    if confidence_level == 0.68:  # ~65% (1 standard deviation)
        chi2_val = 2.279  # chi2.ppf(0.68, 2)
    elif confidence_level == 0.95:  # ~95% (2 standard deviations)
        chi2_val = 5.991  # chi2.ppf(0.95, 2)
    else:
        chi2_val = chi2.ppf(confidence_level, 2)
    
    # Create ellipse
    theta = np.linspace(0, 2*np.pi, n_points)
    
    # Scale by standard deviations and chi-square value
    scale_x = std_x * np.sqrt(chi2_val)
    scale_y = std_y * np.sqrt(chi2_val)
    
    x = mean_x + scale_x * np.cos(theta)
    y = mean_y + scale_y * np.sin(theta)
    
    return x, y

# Create figure
fig = go.Figure()

# Add quadrant backgrounds
# Bottom Left (Libertarian-Left)
fig.add_shape(
    type="rect",
    x0=-10, y0=-10, x1=0, y1=0,
    fillcolor=quadrant_colors["Bottom Left"],
    opacity=0.3,
    line=dict(color="black", width=2)
)

# Top Left (Authoritarian-Left)
fig.add_shape(
    type="rect",
    x0=-10, y0=0, x1=0, y1=10,
    fillcolor=quadrant_colors["Top Left"],
    opacity=0.3,
    line=dict(color="black", width=2)
)

# Bottom Right (Libertarian-Right)
fig.add_shape(
    type="rect",
    x0=0, y0=-10, x1=10, y1=0,
    fillcolor=quadrant_colors["Bottom Right"],
    opacity=0.3,
    line=dict(color="black", width=2)
)

# Top Right (Authoritarian-Right)
fig.add_shape(
    type="rect",
    x0=0, y0=0, x1=10, y1=10,
    fillcolor=quadrant_colors["Top Right"],
    opacity=0.3,
    line=dict(color="black", width=2)
)

# Add center lines
fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=3, opacity=0.6)
fig.add_vline(x=0, line_dash="solid", line_color="black", line_width=3, opacity=0.6)

# Add ellipses and mean points for each model
for idx, (i, row) in enumerate(df.iterrows()):
    provider = row['provider']
    color = provider_colors[provider]
    
    # Create 68% confidence ellipse (1 std dev)
    ellipse_68_x, ellipse_68_y = create_ellipse(
        row['econ_score_mean'], row['soc_score_mean'],
        row['econ_score_std'], row['soc_score_std'],
        confidence_level=0.68
    )
    
    # Create 95% confidence ellipse (2 std dev)
    ellipse_95_x, ellipse_95_y = create_ellipse(
        row['econ_score_mean'], row['soc_score_mean'],
        row['econ_score_std'], row['soc_score_std'],
        confidence_level=0.95
    )
    
    # Add 95% ellipse (outer, lighter)
    fig.add_trace(go.Scatter(
        x=ellipse_95_x,
        y=ellipse_95_y,
        mode='lines',
        line=dict(color=color, width=1, dash='dash'),
        fill='tonext' if idx > 0 else None,
        fillcolor=color,
        opacity=0.1,
        showlegend=False,
        hoverinfo='skip',
        name=f'{row["short_name"]} 95%'
    ))
    
    # Add 68% ellipse (inner, darker)
    fig.add_trace(go.Scatter(
        x=ellipse_68_x,
        y=ellipse_68_y,
        mode='lines',
        line=dict(color=color, width=2),
        fill='toself',
        fillcolor=color,
        opacity=0.2,
        showlegend=False,
        hoverinfo='skip',
        name=f'{row["short_name"]} 68%'
    ))

# Add mean points for each provider (grouped by provider for legend)
added_providers = set()
for i, row in df.iterrows():
    provider = row['provider']
    color = provider_colors[provider]
    
    # Only show legend for first occurrence of each provider
    show_legend = provider not in added_providers
    if show_legend:
        added_providers.add(provider)
    
    # Add mean point
    fig.add_trace(go.Scatter(
        x=[row['econ_score_mean']],
        y=[row['soc_score_mean']],
        mode='markers+text',
        name=provider.capitalize() if show_legend else None,
        legendgroup=provider,
        showlegend=show_legend,
        marker=dict(
            color=color,
            size=12,
            line=dict(width=2, color='white'),
            opacity=1.0
        ),
        text=[row['short_name']],
        textposition='middle right',
        textfont=dict(size=9, color='black'),
        hovertemplate='<b>%{text}</b><br>' +
                     'Mean Econ: %{x:.2f} ± ' + f'{row["econ_score_std"]:.2f}<br>' +
                     'Mean Social: %{y:.2f} ± ' + f'{row["soc_score_std"]:.2f}<br>' +
                     'Provider: ' + provider.capitalize() + '<br>' +
                     '<extra></extra>'
    ))

# Update layout
fig.update_layout(
    title=dict(
        text='LLM Political Compass with Standard Deviation Ellipses',
        x=0.5,
        font=dict(size=18)
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
    width=1000,
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
    ),
    annotations=[
        dict(
            x=0.02, y=0.98,
            xref='paper', yref='paper',
            text='Solid ellipses: 68% confidence (±1σ)<br>Dashed ellipses: 95% confidence (±2σ)',
            showarrow=False,
            font=dict(size=10),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="gray",
            borderwidth=1
        )
    ]
)

# Update traces
fig.update_traces(cliponaxis=False)

# Show the plot
fig.show()
