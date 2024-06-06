import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

file = 'cleaned_dataset_part1.csv'
df = pd.read_csv(file)
sentiment = SentimentIntensityAnalyzer()
for row in df['review_text']:
    df['Sentiment_Compound_Score'] = [sentiment.polarity_scores(row)['compound']]
    df['Positive_Sentiment_Score'] = [sentiment.polarity_scores(row)['pos']]
    df['Neutral_Sentiment_Score'] = [sentiment.polarity_scores(row)['neu']]
    df['Negative_Sentiment_Score'] = [sentiment.polarity_scores(row)['neg']]


print(df['Sentiment_Compound_Score'].astype(str) + '  ' + df['review_text'])
save_file = file.split('.')[0]+'_with_All_Sentiment_Scores.csv'
df.to_csv(save_file, index= False)