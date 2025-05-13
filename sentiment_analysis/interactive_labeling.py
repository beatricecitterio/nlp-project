import pandas as pd
import os

df = pd.read_csv("next_batch_to_label.csv").reset_index(drop=True)

tone_labels = [
    "Neutral / Informational",
    "Supportive / Affirmative / Celebratory",
    "Critical / Angry",
    "Call to Action / Propaganda",
]

def label_tweets(df):
    print("\n=== Tweet Tone Labeling ===\n")
    for i, row in df.iterrows():
        print(f"\nTweet {i+1}: {row['Content']}")
        print("Select tone label:")
        for idx, label in enumerate(tone_labels, 1):
            print(f"  {idx}. {label}")
        
        while True:
            try:
                choice = int(input("Enter choice number (1-4): ").strip())
                if 1 <= choice <= len(tone_labels):
                    df.at[i, 'tone_label'] = tone_labels[choice - 1]
                    break
                else:
                    print("Invalid choice. Please enter a number between 1 and 4.")
            except ValueError:
                print("Invalid input. Please enter a number.")
    return df

def save_labels(df):
    labeled_path = "labeled_tweets_with_tone.csv"
    if os.path.exists(labeled_path):
        existing = pd.read_csv(labeled_path)
        combined = pd.concat([existing, df], ignore_index=True).drop_duplicates(subset=["Content"])
    else:
        combined = df

    combined.to_csv(labeled_path, index=False)
    print(f"\n✅ Appended {len(df)} tweets. Total labeled: {len(combined)}")

df = label_tweets(df)
save_labels(df)