import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np

# Load the data from the CSV file
file_path = r"D:\Research\Political_Bias_of_LLMs\supplyValuesTest\6_summerize\sapply_scores_summary_by_model.csv"
df = pd.read_csv(file_path)

# Extract provider from model name
df['provider'] = df['model_name'].apply(lambda x: x.split('_')[0])

# Create abbreviated model names (max 15 chars)
def abbreviate_model_name(name):
    # Remove provider prefix and shorten
    name = name.replace('anthropic_', '').replace('openai_', '').replace('meta_', '').replace('ibm_', '').replace('moonshotai_', '')
    name = name.replace('claude_4_sonnet', 'claude4').replace('granite_granite_3_3_8b_instruct', 'granite3')
    name = name.replace('llama_4_maverick_instruct', 'llama4_mav').replace('meta_llama_3_70b_instruct', 'llama3_70b')
    name = name.replace('meta_llama_3_8b_instruct', 'llama3_8b').replace('kimi_k2_instruct', 'kimi_k2')
    name = name.replace('gpt_4_1_mini', 'gpt4.1_mini').replace('gpt_4_1_nano', 'gpt4.1_nano')
    name = name.replace('gpt_4_1', 'gpt4.1').replace('gpt_4o', 'gpt4o')
    name = name.replace('gpt_5_mini', 'gpt5_mini').replace('gpt_5_nano', 'gpt5_nano')
    name = name.replace('gpt_5', 'gpt5').replace('gpt_oss_120b', 'gpt_oss')
    return name[:15]

df['short_name'] = df['model_name'].apply(abbreviate_model_name)

# Define colors for each provider (using the specified color palette)
provider_colors = {
    'anthropic': "#115861",
    'ibm': '#DB4545', 
    'meta': '#2E8B57',
    'moonshotai': '#5D878F',
    'openai': '#D2BA4C',
    'deepseek': "#0EB3FF",
}

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
        x=provider_data['right_score_mean'],
        y=provider_data['auth_score_mean'],
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
        text='Sapply Value Political Compass',
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
