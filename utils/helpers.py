def generate_story(df, correlation_matrix, target_col=None):
    """Simple rule-based text generation to tell a 'story' about the data"""
    story = []
    
    # Shape story
    story.append(f"The dataset contains **{df.shape[0]}** records and **{df.shape[1]}** features.")
    
    # Missing value story
    missing = df.isnull().sum().sum()
    if missing > 0:
        story.append(f"⚠️ Data quality alert: There are {missing} missing values that need handling.")
    else:
        story.append("✅ Data quality is robust with no missing values.")
    
    # Correlation story
    if target_col and target_col in correlation_matrix.columns:
        top_corr = correlation_matrix[target_col].drop(target_col).abs().sort_values(ascending=False).head(3)
        story.append(f"The strongest predictors for **{target_col}** appear to be: " + ", ".join([f"**{idx}** ({val:.2f})" for idx, val in top_corr.items()]))
        
    return " ".join(story)
