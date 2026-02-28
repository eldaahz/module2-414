cd /Users/eldaahzelalem/INST414
cat > README.md << 'EOF'
# Reddit Network Analysis

Network analysis of Reddit hyperlink data to identify influential subreddit hubs.

## Dataset
Stanford SNAP Reddit Hyperlink Network (2014-2017)
- 286,561 hyperlinks
- 27,863 subreddits

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
python3 module2.py
```

## Output Files
- `subreddit_importance_scores.csv` - All subreddit rankings
- `top_20_subreddits.csv` - Top 20 summary
- `importance_metrics_comparison.png` - Visualization
- `network_visualization.png` - Network graph

## Methods
Combines three centrality metrics:
- Betweenness Centrality (40%)
- PageRank (30%)
- In-Degree (30%)
