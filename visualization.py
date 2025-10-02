import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from sklearn.cluster import KMeans
import umap

def create_stylish_pie_charts(df, figsize=(16, 8), out_file=None):
    """
    Create stylish pie charts for topic and format distributions.
    """
    # Color palette for up to 24 categories
    colors_24 = [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', 
        '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9', '#F8C471', '#82E0AA',
        '#F1948A', '#85929E', '#F39C12', '#8E44AD', '#3498DB', '#2ECC71',
        '#E74C3C', '#9B59B6', '#1ABC9C', '#F39C12', '#34495E', '#E67E22'
    ]
    
    topic_counts = df['topic_name'].value_counts()
    format_counts = df['format_name'].value_counts()
    
    # Assign colors
    topic_colors = colors_24[:len(topic_counts)]
    format_colors = colors_24[:len(format_counts)]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle('Document Distribution by Topic and Format', fontsize=16, fontweight='bold', y=0.95)
    
    # Helper for conditional autopct (show only if > 3%)
    def autopct_func(pct):
        return f'{pct:.1f}%' if pct > 3 else ''
    
    # Topic pie
    wedges1, texts1, autotexts1 = ax1.pie(
        topic_counts.values,
        labels=topic_counts.index,
        colors=topic_colors,
        autopct=autopct_func,
        startangle=90,
        wedgeprops=dict(width=0.7, edgecolor='white', linewidth=1.5),
        textprops={'fontsize': 8, 'fontweight': 'bold'}
    )
    
    ax1.set_title('Topics Distribution', fontsize=14, fontweight='bold', pad=20)
    
    # Format pie
    wedges2, texts2, autotexts2 = ax2.pie(
        format_counts.values,
        labels=format_counts.index,
        colors=format_colors,
        autopct=autopct_func,
        startangle=90,
        wedgeprops=dict(width=0.7, edgecolor='white', linewidth=1.5),
        textprops={'fontsize': 8, 'fontweight': 'bold'}
    )
    
    ax2.set_title('Formats Distribution', fontsize=14, fontweight='bold', pad=20)
    
    # Style percentage texts: contrast color
    for autotext in autotexts1 + autotexts2:
        # Choose white or black depending on background brightness
        r, g, b, a = to_rgba(autotext.get_color())
        autotext.set_color('white')
        autotext.set_fontsize(9)
        autotext.set_fontweight('bold')
    
    # Adjust layout to avoid overlaps
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save figure
    if out_file:
        plt.savefig(out_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"✓ Saved stylish pie charts to: {out_file}")
    
    plt.show()
    return fig


def create_advanced_pie_charts(df, figsize=(18, 10), out_file=None):
    """
    Create advanced pie charts for topic and format distributions with legends.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'topic_name' and 'format_name' columns
    figsize : tuple
        Figure size (width, height)
    out_file : str, optional
        Path to save the figure (.pdf, .png, .jpg, .svg). Displays plot if None.
    """
    
    # Helper: generate gradient color palette
    def create_gradient_palette(n, base_colors=None):
        if base_colors is None:
            base_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
        colors = []
        for i in range(n):
            base_idx = i % len(base_colors)
            base_color = base_colors[base_idx]
            # Convert hex to RGB
            base_rgb = tuple(int(base_color[j:j+2], 16) for j in (1, 3, 5))
            # Adjust brightness
            variation = 0.8 + 0.4 * (i // len(base_colors)) / max(1, (n // len(base_colors)))
            new_rgb = tuple(min(255, int(c * variation)) for c in base_rgb)
            colors.append(f"#{new_rgb[0]:02x}{new_rgb[1]:02x}{new_rgb[2]:02x}")
        return colors

    # Count topics and formats
    topic_counts = df['topic_name'].value_counts()
    format_counts = df['format_name'].value_counts()
    
    topic_colors = create_gradient_palette(len(topic_counts))
    format_colors = create_gradient_palette(len(format_counts),
                                            base_colors=['#E74C3C', '#9B59B6', '#3498DB', 
                                                         '#2ECC71', '#F39C12', '#1ABC9C'])
    
    # Create figure with grid spec
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 4, hspace=0.4, wspace=0.4)
    ax1 = fig.add_subplot(gs[0, :2])  # Topic pie
    ax2 = fig.add_subplot(gs[0, 2:])  # Format pie
    ax3 = fig.add_subplot(gs[1, :2])  # Topic legend
    ax4 = fig.add_subplot(gs[1, 2:])  # Format legend
    
    # Topic pie chart (no labels on pie)
    wedges1, _ = ax1.pie(
        topic_counts.values,
        colors=topic_colors,
        startangle=90,
        wedgeprops=dict(width=0.7, edgecolor='white', linewidth=2)
    )
    ax1.set_title('Topics Distribution', fontsize=16, fontweight='bold', pad=20)
    
    # Format pie chart
    wedges2, _ = ax2.pie(
        format_counts.values,
        colors=format_colors,
        startangle=90,
        wedgeprops=dict(width=0.7, edgecolor='white', linewidth=2)
    )
    ax2.set_title('Formats Distribution', fontsize=16, fontweight='bold', pad=20)
    
    # Legends with percentages
    ax3.axis('off')
    ax4.axis('off')
    
    topic_legend_elements = [
        plt.Rectangle((0,0),1,1, fc=topic_colors[i],
                      label=f"{topic} ({100*count/len(df):.1f}%)")
        for i, (topic, count) in enumerate(topic_counts.items())
    ]
    format_legend_elements = [
        plt.Rectangle((0,0),1,1, fc=format_colors[i],
                      label=f"{format_name} ({100*count/len(df):.1f}%)")
        for i, (format_name, count) in enumerate(format_counts.items())
    ]
    
    ax3.legend(handles=topic_legend_elements, loc='center', fontsize=10, frameon=False, ncol=2)
    ax3.set_title('Topics Legend', fontsize=12, fontweight='bold')
    
    ax4.legend(handles=format_legend_elements, loc='center', fontsize=10, frameon=False, ncol=2)
    ax4.set_title('Formats Legend', fontsize=12, fontweight='bold')
    
    plt.suptitle(f'Document Distribution Analysis (Total: {len(df):,} documents)',
                 fontsize=18, fontweight='bold')
    
    # Save or show
    if out_file:
        plt.savefig(out_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"✓ Saved advanced pie charts to: {out_file}")
    
    plt.show()
    print('ha')
    return fig



def visualize_embeddings_2d(embeddings_list, df_meta, color_by='topic', n_clusters=None, figsize=(10, 8), out_file=None):
    """
    Cluster and visualize embeddings in 2D using UMAP, colored by topic or format.
    
    Parameters
    ----------
    embeddings_list : list of np.ndarray
        List of embedding arrays. Shape of each array: (n_docs_in_file, embedding_dim)
    df_meta : pd.DataFrame
        DataFrame containing metadata columns: 'topic_name' and 'format_name'.
        Must align with the embeddings order when concatenated.
    color_by : str
        'topic' or 'format' — determines which column to use for coloring.
    n_clusters : int, optional
        If provided, performs KMeans clustering with this many clusters for visualization (not required if coloring by metadata).
    figsize : tuple
        Figure size.
    out_file : str, optional
        Path to save figure.
    """
    
    # --- Stack embeddings into a single array ---
    embeddings = np.vstack(embeddings_list)  # shape: (total_docs, embedding_dim)
    
    # --- Dimensionality reduction ---
    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding_2d = reducer.fit_transform(embeddings)
    
    # --- Prepare colors ---
    if color_by == 'topic':
        labels = df_meta['topic_name'].values
        unique_labels = df_meta['topic_name'].unique()
        base_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    elif color_by == 'format':
        labels = df_meta['format_name'].values
        unique_labels = df_meta['format_name'].unique()
        base_colors = ['#E74C3C', '#9B59B6', '#3498DB', '#2ECC71', '#F39C12', '#1ABC9C']
    else:
        raise ValueError("color_by must be 'topic' or 'format'")
    
    # Generate a gradient colormap if more labels than base colors
    if len(unique_labels) > len(base_colors):
        cmap = LinearSegmentedColormap.from_list("gradient_colors", base_colors, N=len(unique_labels))
        color_map = {label: cmap(i/len(unique_labels)) for i, label in enumerate(unique_labels)}
    else:
        color_map = {label: base_colors[i] for i, label in enumerate(unique_labels)}
    
    # Map labels to colors
    point_colors = [color_map[label] for label in labels]
    
    # --- Optional clustering overlay (KMeans) ---
    if n_clusters:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
    else:
        cluster_labels = None
    
    # --- Plot ---
    plt.figure(figsize=figsize)
    plt.scatter(
        embedding_2d[:, 0], embedding_2d[:, 1],
        c=point_colors, s=50, alpha=0.8, edgecolors='w', linewidth=0.5
    )
    
    plt.xlabel("UMAP Dimension 1", fontsize=12)
    plt.ylabel("UMAP Dimension 2", fontsize=12)
    plt.title(f"2D Visualization of Embeddings colored by {color_by.capitalize()}", fontsize=16, fontweight='bold')
    
    # Legend
    for label, color in color_map.items():
        plt.scatter([], [], c=color, label=label, s=50)
    plt.legend(title=color_by.capitalize(), bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # Save or show
    if out_file:
        plt.savefig(out_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"✓ Saved scatterplot to {out_file}")
    
    plt.show()
    return embedding_2d