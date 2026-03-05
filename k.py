import pandas as pd
def standardize(messy):
    print(f"Loading raw data from {messy}...")
    df = pd.read_csv(messy)
    column_mapping ={
        "ProductId": "product_id",
        "Text":"review_text",
        "Score":"rating(out of 5)",
        "Time":"review_date"
    }
    #"Standardizing column names..."
    df.rename(columns=column_mapping, inplace=True)
    mandatory_columns = ['product_id', 'review_text', 'rating(out of 5)', 'review_date']
    df=df[mandatory_columns]
    df=df.dropna(subset=mandatory_columns)
    df['review_date'] = pd.to_datetime(df['review_date'], unit='s').dt.strftime('%Y-%m-%d')
    #"Data standardized"
    return df
Clean_dataset=standardize(r"C:\Users\DELL\OneDrive\Desktop\DATA\Reviews.csv")
Clean_dataset.to_csv(r"C:\Users\DELL\OneDrive\Desktop\DATA\Clean_reviews.csv", index=False)
print("Cleaned data saved to Clean_reviews.csv") 
