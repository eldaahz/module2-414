import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import wget

# STEP 1: DOWNLOAD DATA
print("Downloading data...")
url = 'http://snap.stanford.edu/data/soc-redditHyperlinks-body.tsv'
wget.download(url, 'reddit_hyperlinks.tsv')
print("\nData downloaded")

# STEP 2: LOAD DATA
print("\nLoading data...")
df = pd.read_csv('reddit_hyperlinks.tsv', sep='\t')

print(f"Loaded {len(df):,} hyperlinks")
print(f"Between {df['SOURCE_SUBREDDIT'].nunique():,} subreddits")

# STEP 3: BUILD NETWORK
print("\nBuilding network...")
G = nx.DiGraph()

for _, row in df.iterrows():
    G.add_edge(
        row['SOURCE_SUBREDDIT'],
        row['TARGET_SUBREDDIT'],
        sentiment=row['LINK_SENTIMENT']
    )

print(f"Network: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

# STEP 4: CALCULATE IMPORTANCE METRICS
print("\nCalculating importance metrics...")

betweenness = nx.betweenness_centrality(G, normalized=True)
pagerank = nx.pagerank(G)
in_degree = dict(G.in_degree())

print("Metrics calculated")

# STEP 5: CREATE RESULTS DATAFRAME
results = pd.DataFrame({
    'subreddit': list(G.nodes()),
    'betweenness': [betweenness[s] for s in G.nodes()],
    'pagerank': [pagerank[s] for s in G.nodes()],
    'in_degree': [in_degree[s] for s in G.nodes()]
})

# Normalize to 0-100
for col in ['betweenness', 'pagerank', 'in_degree']:
    results[f'{col}_norm'] = (results[col] / results[col].max()) * 100

# Composite importance score
results['importance'] = (
    results['betweenness_norm'] * 0.4 +
    results['pagerank_norm'] * 0.3 +
    results['in_degree_norm'] * 0.3
)

# STEP 6: IDENTIFY TOP NODES
top_20 = results.nlargest(20, 'importance')

role_map = {
    'askreddit': 'Universal Hub',
    'iama': 'Celebrity Authority',
    'subredditdrama': 'Drama Bridge',
    'outoftheloop': 'Context Connector',
    'writingprompts': 'Creative Hub',
    'pics': 'Visual Authority',
    'leagueoflegends': 'Gaming Bridge',
    'videos': 'Media Gateway',
    'gaming': 'Gaming Hub',
    'mhoc': 'Political Sim Bridge',
    'funny': 'General Hub',
    'todayilearned': 'Knowledge Authority',
    'explainlikeimfive': 'Explainer Hub',
    'worldnews': 'News Authority',
    'pcmasterrace': 'Tech Community Hub',
    'legaladvice': 'Advice Authority',
    'conspiracy': 'Fringe Bridge',
    'bitcoin': 'Crypto Hub',
    'news': 'News Gateway',
    'games': 'Gaming Authority'
}

top_20 = top_20.copy()
top_20['role'] = top_20['subreddit'].map(role_map).fillna('Community Hub')

# Save results (after role column is assigned)
results.to_csv('subreddit_importance_scores.csv', index=False)
top_20.to_csv('top_20_subreddits.csv', index=False)
print("\nResults saved to CSV files")

print("\n" + "="*80)
print("TOP 20 MOST IMPORTANT SUBREDDITS")
print("="*80)
print(f"{'Rank':<6}{'Subreddit':<25}{'Importance':<12}{'Betweenness':<15}{'PageRank':<12}{'In-Degree':<12}{'Role'}")
print("-"*95)

for i, row in enumerate(top_20.itertuples(), 1):
    print(f"{i:<6}{row.subreddit:<25}{row.importance:<12.2f}{row.betweenness:<15.6f}{row.pagerank:<12.6f}{row.in_degree:<12}{row.role}")

# STEP 7: VISUALIZATIONS
print("\nCreating visualizations...")

# Figure 1: Bar charts comparing metrics
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

top_20.nlargest(15, 'betweenness').plot(
    kind='barh', x='subreddit', y='betweenness',
    ax=axes[0], color='#FF6B6B', legend=False
)
axes[0].set_title('Top 15: Betweenness Centrality', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Betweenness Score')

top_20.nlargest(15, 'pagerank').plot(
    kind='barh', x='subreddit', y='pagerank',
    ax=axes[1], color='#4ECDC4', legend=False
)
axes[1].set_title('Top 15: PageRank', fontsize=14, fontweight='bold')
axes[1].set_xlabel('PageRank Score')

top_20.nlargest(15, 'in_degree').plot(
    kind='barh', x='subreddit', y='in_degree',
    ax=axes[2], color='#95E1D3', legend=False
)
axes[2].set_title('Top 15: In-Degree', fontsize=14, fontweight='bold')
axes[2].set_xlabel('Number of Incoming Links')

plt.tight_layout()
plt.savefig('importance_metrics_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: importance_metrics_comparison.png")

# Figure 2: Network graph of top 30
top_30_subs = results.nlargest(30, 'importance')['subreddit'].tolist()
subgraph = G.subgraph(top_30_subs)

plt.figure(figsize=(16, 16))
pos = nx.spring_layout(subgraph, k=2, iterations=50, seed=42)

node_sizes = [results[results['subreddit'] == node]['importance'].values[0] * 50
              for node in subgraph.nodes()]

nx.draw_networkx_nodes(subgraph, pos, node_size=node_sizes,
                       node_color='lightblue', alpha=0.8,
                       edgecolors='black', linewidths=1.5)

nx.draw_networkx_edges(subgraph, pos, edge_color='gray',
                       alpha=0.3, arrows=True, arrowsize=10)

nx.draw_networkx_labels(subgraph, pos, font_size=9, font_weight='bold')

plt.title('Reddit Network: Top 30 Subreddits\n(Node size = Importance)',
          fontsize=16, fontweight='bold')
plt.axis('off')
plt.savefig('network_visualization.png', dpi=300, bbox_inches='tight')
print("Saved: network_visualization.png")

# STEP 8: VALIDATION CHECKS
print("\n" + "="*80)
print("VALIDATION CHECKS")
print("="*80)

# Check 1: PageRank sums to 1
pagerank_sum = sum(pagerank.values())
print(f"PageRank sum: {pagerank_sum:.6f} (should be ~1.0)")

# Check 2: Network is connected
is_connected = nx.is_weakly_connected(G)
print(f"Network is weakly connected: {is_connected}")

# Check 3: No negative values
has_negative = any(results['betweenness'] < 0) or any(results['pagerank'] < 0)
print(f"No negative values: {not has_negative}")

# Check 4: Self-loops
self_loops = list(nx.selfloop_edges(G))
print(f"Self-loops: {len(self_loops)} ({len(self_loops)/G.number_of_edges()*100:.3f}%)")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print("\nGenerated files:")
print("  1. subreddit_importance_scores.csv")
print("  2. top_20_subreddits.csv")
print("  3. importance_metrics_comparison.png")
print("  4. network_visualization.png")
