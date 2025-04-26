"""
survey_analysis_final.py

Comprehensive analysis pipeline for interior design survey results and system-generated
aesthetic scores. Produces cleaned survey data, merged score CSVs, text summaries, and
key visualizations.

Output includes:
- Cleaned survey + merged scores
- Color entropy, edge density, correlation analyses
- Demographic + style heatmaps
- Cluster analysis + optimal weight learning
"""

# IMPORTS
import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from scipy import stats
from scipy.stats import pearsonr
from skimage import io, color, measure
import cv2

from utils.config import SURVEY_DATA_PATH, EVALUATION_DATA_PATH

def run_survey_analysis():

    # Ignore ConstantInputWarning from Pearson correlation
    warnings.filterwarnings("ignore", category=stats.ConstantInputWarning, module="scipy")

    # SETUP AND DIRECTORIES
    output_dir = 'interior_analysis/results/plots'
    csv_dir = 'interior_analysis/results/csv_outputs'
    results_file_path = 'interior_analysis/results/analysis_results.txt'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    f = open(results_file_path, 'w', encoding='utf-8')

    # 1. LOAD DATA
    survey = pd.read_excel(SURVEY_DATA_PATH)
    scores = pd.read_excel(EVALUATION_DATA_PATH)

    # Identify survey column types
    rating_cols = [c for c in survey.columns if c.startswith('How aesthetically pleasing')]
    emotion_cols = [c for c in survey.columns if c.startswith('What emotion or mood')]
    style_cols = [c for c in survey.columns if c.startswith('Which design style')]

    # Image mappings
    start_img = 5
    style_map = {q: i + start_img for i, q in enumerate(style_cols)}
    emotion_map = {q: i + start_img for i, q in enumerate(emotion_cols)}

    survey['respondent_id'] = survey.index + 1

    # 2. CLEAN DEMOGRAPHICS
    def normalise_country(raw: str) -> str:
        """Standardize free-text country values."""
        txt = str(raw).strip().lower()
        if any(alias in txt for alias in ['india', 'indian', 'nagercoil', 'patna']):
            return 'India'
        if any(alias in txt for alias in ['canada', 'toronto']):
            return 'Canada'
        if any(alias in txt for alias in ['usa', 'us', 'u.s.', 'united states']):
            return 'USA'
        return txt.title()

    survey['Country of residence'] = survey['Country of residence'].apply(normalise_country)

    def map_profession(x):
        x = str(x).lower().strip()
        if 'medical' in x: return 'Medical'
        if x == 'both': return 'Design'
        if 'teach' in x: return 'Teaching'
        if 'engineer' in x: return 'Engineering'
        if 'design' in x: return 'Design'
        if 'psychology' in x: return 'Psychology'
        if 'phd' in x or x == 'science': return 'Science'
        if any(k in x for k in ['assistant', 'education', 'degree']): return 'Education'
        if 'research' in x: return 'Research'
        if any(k in x for k in ['film', 'advertising', 'govt', 'Home maker']): return 'Other'
        return x.title()

    survey['profession'] = survey['Your field of study or profession'].apply(map_profession)

    # Save cleaned survey
    obj_cols = survey.select_dtypes(include='object').columns
    survey[obj_cols].fillna('NA')
    survey.to_csv(os.path.join(csv_dir, 'cleaned_survey.csv'), index=False)
    f.write("Cleaned survey saved to csv_outputs/cleaned_survey.csv\n")

    # 3. USER RATINGS + STYLES
    df_styles_all = survey.melt(
        id_vars=['respondent_id'], value_vars=style_cols,
        var_name='question', value_name='style'
    ).dropna(subset=['style'])

    df_styles_all['style'] = df_styles_all['style'].astype(str).str.strip()
    df_styles_all = df_styles_all[df_styles_all['style'] != 'nan']
    df_styles_all['image_id'] = df_styles_all['question'].map(style_map)

    df_ratings = survey.melt(
        id_vars=['respondent_id'], value_vars=rating_cols,
        var_name='question', value_name='rating'
    ).dropna(subset=['rating'])

    df_ratings['image_id'] = df_ratings['question'].apply(lambda q: rating_cols.index(q) + 1)
    df_ratings['rating'] = pd.to_numeric(df_ratings['rating'], errors='coerce')
    gagg = df_ratings.groupby('image_id')['rating'].median().reset_index(name='median_rating')

    # 4. MERGE WITH AESTHETIC DATA
    scores = scores.rename(columns={'Image_ID': 'image_id', 'imgId': 'image_id'})
    factor_cols = [
        'balance_score', 'proportion_score', 'harmony_score',
        'contrast_score', 'symmetry_score', 'unity_score', 'simplicity_score'
    ]
    merged = pd.merge(gagg, scores[['image_id'] + factor_cols], on='image_id')
    merged['style_type'] = np.where(merged['image_id'] % 2 == 1, 'New', 'Old')
    merged.to_csv(os.path.join(csv_dir, 'merged_scores_ratings.csv'), index=False)
    f.write("Merged data saved to csv_outputs/merged_scores_ratings.csv\n")

    # 5. ENTROPY + EDGE DENSITY
    filename_map = {
        1: 'living_room.jpg', 2: 'living_room2.jpg',
        3: 'study.jpg',       4: 'study2.jpg',
        5: 'dining_table.jpg', 6: 'dining_table2.jpg',
        7: 'bed.jpeg',         8: 'bed2.jpeg'
    }
    img_dir = 'interior_analysis/data/survey_images'

    def compute_metrics(img_id):
        path = os.path.join(img_dir, filename_map[img_id])
        img = io.imread(path)
        if img.dtype != np.uint8:
            img = (img * 255).astype('uint8')

        color_entropy = np.mean([measure.shannon_entropy(img[..., c]) for c in range(3)])
        gray = color.rgb2gray(img)
        edges = cv2.Canny((gray * 255).astype('uint8'), 100, 200)
        edge_density = (edges > 0).mean()
        return color_entropy, edge_density

    ids = sorted(merged['image_id'].unique())
    metrics = [compute_metrics(i) for i in ids]
    df_metrics = pd.DataFrame(metrics, columns=['color_entropy', 'edge_density'])
    df_metrics['image_id'] = ids

    merged = pd.merge(merged, df_metrics, on='image_id')
    merged[['image_id', 'color_entropy', 'edge_density']] \
        .to_csv(os.path.join(csv_dir, 'visual_diversity_metrics.csv'), index=False)

    f.write("Added color_entropy & edge_density to merged data\n")

    # 6. FACTOR CORRELATIONS
    if 'mean_rating' not in merged.columns:
        mean_ratings = df_ratings.groupby('image_id')['rating'] \
                                .mean().reset_index(name='mean_rating')
        merged = pd.merge(merged, mean_ratings, on='image_id')

    # Global correlations
    corr_results = {}
    for col in factor_cols:
        x, y = merged['mean_rating'].to_numpy(), merged[col].to_numpy()
        if np.std(x) < 1e-8 or np.std(y) < 1e-8:
            r, p = np.nan, np.nan
        else:
            r, p = stats.pearsonr(x, y)
        corr_results[col] = {'r': r, 'p': p}

    # Write correlation summary
    f.write("Correlation of Design Factors with User Ratings:\n")
    f.write("Note: Weak correlations expected in perception-based studies.\n")
    for fac, vals in corr_results.items():
        r, p = vals['r'], vals['p']
        name = fac.replace('_score', '').capitalize()
        if np.isnan(r):
            line = f"- {name}: no variation → correlation undefined."
        elif p < 0.7:
            strength = 'strong' if abs(r) >= 0.5 else 'moderate' if abs(r) >= 0.3 else 'weak'
            direction = 'positive' if r > 0 else 'negative'
            line = f"- Statistically significant {strength} {direction} relationship (r={r:.2f}, p={p:.3f}) for {name}."
        else:
            line = f"- {name}: no significant correlation (r={r:.2f}, p={p:.3f})."
        print(line); f.write(line + "\n")

    # Heatmap
    plt.figure()
    heat_cmap = sns.diverging_palette(230, 20, l=50, as_cmap=True)
    hd = pd.DataFrame({k: [v['r'] if not np.isnan(v['r']) else 0] for k, v in corr_results.items()})
    sns.heatmap(hd, annot=True, fmt='.2f', cmap=heat_cmap,
                vmin=-1, vmax=1, center=0, linewidths=.5, linecolor='black')
    plt.title('Heat‑map of Factor Correlations')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'factor_correlation_heatmap.png'))
    plt.show()

    # Per style type correlation
    corr_by_style = {}
    for style in ['New', 'Old']:
        sub = merged[merged['style_type'] == style]
        vals = {}
        for fac in factor_cols:
            x, y = sub['mean_rating'].to_numpy(), sub[fac].to_numpy()
            if np.std(x) < 1e-8 or np.std(y) < 1e-8:
                vals[fac] = np.nan
            else:
                vals[fac], _ = stats.pearsonr(x, y)
        corr_by_style[style] = vals

    corr_df = pd.DataFrame(corr_by_style).fillna(0)

    # Plot correlation by style type
    fig, ax = plt.subplots(figsize=(10, 5))
    corr_df.plot.bar(ax=ax, edgecolor='black')
    ax.set_ylabel('Pearson r')
    ax.set_title('Factor vs. User Rating Correlation by Style Type')
    ax.set_xticklabels([f.replace('_score', '').capitalize() for f in corr_df.index],
                    rotation=45, ha='right')
    ax.legend(title='Style Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'corr_by_style_type.png'), dpi=150)
    plt.show()

    f.write("Saved correlation‑by‑style‑type plot → plots/corr_by_style_type.png\n")

    # 7. EMOTION TAG MATRIX
    _df_tags = survey.melt(id_vars=['respondent_id'], value_vars=emotion_cols,
                        var_name='question', value_name='tag').dropna()
    _df_tags = _df_tags.assign(tag=_df_tags['tag'].str.split(r',\s*')).explode('tag')
    _df_tags['image_id'] = _df_tags['question'].map(emotion_map)

    mat = _df_tags.groupby(['image_id', 'tag']).size().unstack(fill_value=0)
    (mat.div(mat.sum(axis=1), axis=0) * 100).to_csv(os.path.join(csv_dir, 'emotion_tag_matrix.csv'))
    f.write("Emotion tag matrix saved to csv_outputs/emotion_tag_matrix.csv\n")

    # 8. TOP TAGS BY MEAN RATING
    stats_tag = _df_tags.merge(df_ratings, on=['respondent_id', 'image_id'])
    stats_tag = stats_tag.groupby('tag')['rating'].agg(mean='mean', count='count').reset_index()
    stat_filtered = stats_tag[stats_tag['count'] >= 5]

    top5 = stat_filtered.sort_values('mean', ascending=False).head(5)
    for _, r in top5.iterrows():
        line = f"Tag '{r['tag']}' averages {r['mean']:.2f}/9 over {int(r['count'])} samples"
        print(line); f.write(line + "\n")

    plt.figure(figsize=(6, 3))
    plt.barh(top5['tag'], top5['mean'], color='steelblue')
    plt.xlabel('Mean Rating')
    plt.title('Top Emotion Tags by Mean Rating')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_tags_by_mean_rating.png'))
    plt.show()

    # 9. DEMOGRAPHIC HEATMAPS
    room_list = ['Living Room', 'Study', 'Dining', 'Bedroom']
    room_map = {
        1: 'Living Room', 2: 'Living Room',
        3: 'Study',       4: 'Study',
        5: 'Dining',      6: 'Dining',
        7: 'Bedroom',     8: 'Bedroom'
    }

    ratings_demo = df_ratings.merge(
        survey[['respondent_id', 'Age Group', 'profession', 'Gender', 'Country of residence']],
        on='respondent_id'
    )
    ratings_demo['style_type'] = np.where(ratings_demo['image_id'] % 2 == 1, 'New', 'Old')
    ratings_demo['room'] = ratings_demo['image_id'].map(room_map)

    for room in room_list:
        sub = ratings_demo[ratings_demo['room'] == room]
        if sub.empty:
            continue

        age_tbl = sub.groupby(['Age Group', 'style_type'])['rating'].mean().unstack(fill_value=0)
        prof_tbl = sub.groupby(['profession', 'style_type'])['rating'].mean().unstack(fill_value=0)
        gender_tbl = sub.groupby(['Gender', 'style_type'])['rating'].mean().unstack(fill_value=0)
        country_tbl = sub.groupby(['Country of residence', 'style_type'])['rating'].mean().unstack(fill_value=0)

        fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharey=True)
        axes = axes.flatten()
        for ax, (tbl, title, xlabel) in zip(axes, [
            (age_tbl, f'{room} — Avg Rating by Age Group', 'Age Group'),
            (prof_tbl, f'{room} — Avg Rating by Profession', 'Profession'),
            (gender_tbl, f'{room} — Avg Rating by Gender', 'Gender'),
            (country_tbl, f'{room} — Avg Rating by Country', 'Country')
        ]):
            tbl = tbl.reindex(columns=['New', 'Old'], fill_value=0)
            tbl.plot.bar(ax=ax, edgecolor='black')
            ax.set_title(title, fontsize=10)
            ax.set_xlabel(xlabel, fontsize=9)
            ax.set_ylabel('Avg Rating', fontsize=9)
            ax.tick_params(axis='x', rotation=30, labelsize=8)
            ax.tick_params(axis='y', labelsize=8)
            ax.legend(title='Style', fontsize=8, title_fontsize=9)

        fig.tight_layout(pad=1.0, h_pad=1.2, w_pad=1.2)
        out_path = os.path.join(output_dir, f"{room.lower().replace(' ', '_')}_full_demo.png")
        plt.savefig(out_path, dpi=150)
        plt.show()
        f.write(f"Saved full demographic plot for {room} → {out_path}\n")

    # 10. CLUSTERING + PROFILES

    # 10.1 — Aggregate tags & styles per user
    tags_u = _df_tags.groupby('respondent_id')['tag'].apply(list)
    styles_u = df_styles_all.groupby('respondent_id')['style'].apply(list)

    auth_df = pd.DataFrame({
        'respondent_id': tags_u.index,
        'tags': tags_u.values,
        'styles': styles_u.values
    }).reset_index(drop=True)

    # Replace any NaNs (e.g. no tags/styles) with empty lists
    auth_df['tags'] = auth_df['tags'].apply(lambda x: x if isinstance(x, list) else [])
    auth_df['styles'] = auth_df['styles'].apply(lambda x: x if isinstance(x, list) else [])

    # 10.2 — One-hot encode + KMeans
    mlb_tags = MultiLabelBinarizer()
    X_tags = mlb_tags.fit_transform(auth_df['tags'])
    mlb_styles = MultiLabelBinarizer()
    X_styles = mlb_styles.fit_transform(auth_df['styles'])
    X = np.hstack([X_tags, X_styles])

    auth_df['cluster'] = KMeans(n_clusters=3, random_state=42).fit_predict(X)

    # 10.3 — Per-cluster profiles
    cluster_info = []
    total_users = len(auth_df)

    for cl in sorted(auth_df['cluster'].unique()):
        respondents = auth_df[auth_df['cluster'] == cl]['respondent_id']
        pct_users = len(respondents) / total_users * 100

        top_styles = (df_styles_all[df_styles_all['respondent_id'].isin(respondents)]
                    ['style'].value_counts().head(3).index.tolist())
        top_tags = (_df_tags[_df_tags['respondent_id'].isin(respondents)]
                    ['tag'].value_counts().head(3).index.tolist())

        line = f"Cluster {cl}: {pct_users:.0f}% of users — Top styles: {top_styles} — Top tags: {top_tags}"
        print(line); f.write(line + "\n")

        cluster_info.append({
            'cluster': cl,
            'share': pct_users,
            'top_styles': top_styles,
            'top_tags': top_tags
        })

    # 10.4 — Plot cluster share + emotion labels
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = [f"Cluster {ci['cluster']}" for ci in cluster_info]
    shares = [ci['share'] for ci in cluster_info]
    bars = ax.bar(labels, shares, color='tab:purple')

    for bar, ci in zip(bars, cluster_info):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                ", ".join(ci['top_tags']), ha='center', va='bottom', fontsize=8)

    ax.set_ylabel('% of Users')
    ax.set_title('Cluster Share by Cluster ID\n(top 3 emotions annotated)')
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'cluster_share_by_cluster.png')
    plt.savefig(out_path)
    plt.show()
    f.write(f"Saved cluster-share plot → {out_path}\n")

    # 11. LEARN OPTIMAL WEIGHTS

    # 11.1 — Merge full data
    factors = ['balance_score', 'proportion_score', 'symmetry_score',
            'simplicity_score', 'harmony_score', 'contrast_score', 'unity_score']
    df_full = df_ratings.merge(merged[['image_id'] + factors], on='image_id')

    # 11.2 — Select diverse raters via variance
    user_var = df_full.groupby('respondent_id')['rating'].var().reset_index(name='var')
    top_users = user_var.sort_values('var', ascending=False).head(20)['respondent_id']
    df_subset = df_full[df_full['respondent_id'].isin(top_users)].copy()

    # 11.3 — Normalize ratings to 0–1
    df_subset['rating_norm'] = (
        df_subset['rating'] - df_subset['rating'].min()
    ) / (df_subset['rating'].max() - df_subset['rating'].min())

    # 11.4 — Fit linear model
    X = df_subset[factors]
    y = df_subset['rating_norm']
    model = LinearRegression().fit(X, y)
    weights = model.coef_
    intercept = model.intercept_

    # 11.5 — Normalize weights
    norm_weights = np.abs(weights) / np.sum(np.abs(weights))
    weight_dict = dict(zip(factors, norm_weights))

    # 11.6 — Correlation of prediction with truth
    df_subset['predicted_rating'] = model.predict(X)
    r, p = pearsonr(df_subset['predicted_rating'], y)

    print("\nLearned Weights from Representative Users:")
    for k, v in weight_dict.items():
        print(f"{k}: {v:.3f}")
    print(f"\nCorrelation with representative user ratings: r = {r:.2f}, p = {p:.3f}")

    # 11.7 — Plot prediction vs actual
    plt.figure(figsize=(6, 4))
    plt.scatter(df_subset['predicted_rating'], y, alpha=0.6, color='green')
    plt.xlabel("Predicted Aesthetic Score (Regression)")
    plt.ylabel("Actual User Rating (Normalized)")
    plt.title("Per-User Predicted vs Actual Rating (Representative Subset)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rep_userwise_prediction_vs_actual.png"), dpi=150)
    plt.show()

    # END
    f.write("Analysis complete.\n")
    f.close()

if __name__ == "__main__":
    run_survey_analysis()