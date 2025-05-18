import os
import pandas as pd



def postprocess_tweets(df):

    df[["party", "topic", "tone"]] = df["prompt"].apply(
        lambda x: pd.Series(extract_prompt_features(x))
    )
    df.drop(columns=["prompt"], inplace=True)
    return df


def extract_prompt_features(prompt):

    prompt_text = prompt.split("[INST]")[1].split("[/INST]")[0].strip()
    lines = prompt_text.split("\n")
    
    party = lines[1].split(":", 1)[1].strip()
    topic = lines[2].split(":", 1)[1].strip()
    sentiment = lines[3].split(":", 1)[1].strip()
    return party, topic, sentiment


if __name__ == "__main__":

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "generated_")
    for df_name in ["350m_baseline"]:
        df_path = data_path+f"{df_name}.csv"
        save_path = data_path+f"{df_name}_postprocessed.csv"
        df = pd.read_csv(df_path)
        df = postprocess_tweets(df)
        df.to_csv(save_path, index=False)