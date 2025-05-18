import json
import os
import pandas as pd
import random
import re
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def inject_keyword(tweet, processed_tweet, keywords):
    
    # Check if the tweet contains a keyword
    has_keyword = any(keyword in processed_tweet for keyword in keywords)
    if has_keyword:
        return tweet
    
    # Inject a random keyword
    keyword = random.choice(keywords)
    words = tweet.split()
    insert_position = int(random.gauss(len(words) // 2, len(words) // 2))
    insert_position = max(0, min(len(words), insert_position))
    words.insert(insert_position, keyword)
    return ' '.join(words)


def build_prompt(tweet, party, topic, sentiment):

    OPENINGS = [
    "Scrivi un tweet come se fossi un politico italiano con queste caratteristiche:\n",
    "Genera un probabile tweet di un politico con:\n",
    ]
    
    PARTY_KEYWORDS = [
        "Ideologia",
        "Ideologia politica"
        "Partito",
        "Partito politico",
        "Partito di appartenenza",
    ]

    TOPIC_KEYWORDS = [
        "Argomento",
        "Contenuto",
        "Tema",
    ]

    SENTIMENT_KEYWORDS = [
        "Accezione",
        "Tono",
    ]

    if party is None or topic is None:
        raise ValueError("Missing information (party or topic) to build the prompt")
    prompt_json = {}

    # Build the prompt
    prompt = random.choice(OPENINGS)
    prompt += f"{random.choice(PARTY_KEYWORDS)}: {party}\n"
    prompt += f"{random.choice(TOPIC_KEYWORDS)}: {topic}\n"
    prompt += f"{random.choice(SENTIMENT_KEYWORDS)}: {sentiment}\n"
    prompt_json['prompt'] = prompt

    if tweet is None:
        raise ValueError("Missing tweet to build the prompt")
    prompt_json['tweet'] = tweet

    return prompt_json


def process_tweets(tweet):

    # Remove all URLs and mentions
    tweet = re.sub(r'https?://(?:t\.co/\w+|www\.\w+\.\w+)', '', tweet, flags=re.IGNORECASE)
    tweet = re.sub(r'@\w+', '', tweet, flags=re.IGNORECASE)
    # Remove extra whitespaces
    tweet = re.sub(r'\s+', ' ', tweet)

    return tweet


def combine_tweets_and_topics(data_dir, save_path):

    tweet_df_names = ["tweets_left.csv", "tweets_right.csv"]
    topic_df_names = ["topics_left.csv", "topics_right.csv"]

    dfs = []
    for i in range(len(tweet_df_names)):
        topic_df = pd.read_csv(os.path.join(data_dir, topic_df_names[i]))

        topics_dict = {}
        for _, row in topic_df.iterrows():
            topic_number = row['TopicNumber']
            topic_words = row['TopicName']
            topics_dict[topic_number] = topic_words

        topics_to_topics = {}
        for _, row in topic_df.iterrows():
            topic_number = row['TopicNumber']
            macro_topic = row['MacroTopic']
            topics_to_topics[topic_number] = macro_topic

        topics_to_words = {}
        for _, row in topic_df.iterrows():
            topic_number = row['TopicNumber']
            topic_words = row['Keywords']
            topics_to_words[topic_number] = topic_words
    
        tweets_df = pd.read_csv(os.path.join(data_dir, tweet_df_names[i]))
        tweets_df = tweets_df.drop_duplicates(subset=['content'])
        tweets_df['party'] = "Sinistra" if i == 0 else "Destra"
        tweets_df['macro_topic'] = tweets_df['dominant_topic'].map(topics_to_topics)
        tweets_df['topic_words'] = tweets_df['dominant_topic'].map(topics_to_words)
        tweets_df['dominant_topic'] = tweets_df['dominant_topic'].map(topics_dict)

        dfs.append(tweets_df)
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.to_csv(save_path, index=False)


def combine_tweets_and_sentiment(data_path, sentiment_path):

    SENTIMENT_MAP = {
        '0.0' : 'Esortativo / Propaganda',
        '1.0' : 'Critico / Negativo',
        '2.0' : 'Neutrale / Informativo',
        '3.0' : 'Supporto / Positivo',
        'unknown' : 'Generico'
    }
    tweets_df = pd.read_csv(data_path)
    sentiment_df = pd.read_csv(sentiment_path)

    # Merge sentiment to tweets
    merged_df = tweets_df.merge(
        sentiment_df[['ID', 'label_id']],
        left_on='ID',
        right_on='ID',
        how='left'
    )
    merged_df['label_id'] = merged_df['label_id'].astype(str).map(SENTIMENT_MAP)
    merged_df.to_csv(data_path, index=False)



def generate_jsonl_data(data_path, save_dir, split_ratio=(0.7, 0.1)):
    

    df = pd.read_csv(data_path)

    # Stratified split based on 'dominant_topic'
    train_ratio, eval_ratio = split_ratio
    test_ratio = 1 - train_ratio - eval_ratio

    # First split off test set
    train_eval_df, test_df = train_test_split(
        df, test_size=test_ratio, stratify=df['dominant_topic'], random_state=42
    )
    # Then split train/eval
    eval_size = eval_ratio / (train_ratio + eval_ratio)
    train_df, eval_df = train_test_split(
        train_eval_df, test_size=eval_size, stratify=train_eval_df['dominant_topic'], random_state=42
    )

    splits = {
        "train": train_df,
        "eval": eval_df,
        "test": test_df
    }

    for portion, split_df in splits.items():
        bar = tqdm(total=len(split_df), desc=f"Processing {portion} tweets", unit="tweet")
        filename = os.path.join(save_dir, f"prompts_{portion}.jsonl")
        with open(filename, "w") as f:
            for _, row in split_df.iterrows():
                tweet = row['content']
                tweet = process_tweets(tweet)
                processed_tweet = row['processed_tweet_LDA']
                party = row['party']
                topic = row['dominant_topic']
                sentiment = row['label_id']
                keywords = row['topic_words']

                injected_tweet = inject_keyword(tweet, processed_tweet, keywords)
                prompt_json = build_prompt(injected_tweet, party, topic, sentiment)
                f.write(json.dumps(prompt_json) + "\n")
                bar.update(1)
        bar.close()


if __name__ == "__main__":
    cwd = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(cwd, "data")
    data_path = os.path.join(cwd, "data", "finetune_data.csv")
    combine_tweets_and_topics(data_dir, data_path)

    sentiment_path = os.path.join(cwd, "data", "final_tone_labeled_tweets.csv")
    combine_tweets_and_sentiment(data_path, sentiment_path)

    save_path = os.path.join(cwd, "data")
    generate_jsonl_data(data_path, save_path)