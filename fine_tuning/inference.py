from transformers import pipeline

pipe = pipeline("text-generation", model="final-tweet-model", tokenizer="final-tweet-model")

prompt = "Topic: lavoro\nParty: left\nSentiment: positive\nTweet:"
print(pipe(prompt, max_new_tokens=50)[0]["generated_text"])
