from tweety import TwitterAsync
import asyncio
import csv
import pandas as pd
import nest_asyncio


def get_last_date(file):
    try:
        data = pd.read_csv(file)
        data['Date'] = pd.to_datetime(data['Date'], infer_datetime_format=True)
        data = data.sort_values(by='Date', ascending=True)
        last_date = data['Date'].iloc[0]
        return last_date.strftime('%Y-%m-%d')
    except Exception as e:
        print(f"Error reading previous CSV: {e}")
        return "2022-12-31"


async def main(final_date="2022-12-31", i=0):
    client = TwitterAsync("session_name")
    
    print("Starting interactive login...")
    await client.start()
    
    if client.me:
        print(f"Successfully logged in as: {client.me.username}")
    else:
        print("Login failed or session not established properly")
        return
    date_query = f"from:GiorgiaMeloni since:2021-01-01 until:{final_date}"
    print(f"Searching with query: {date_query}")
    
    csv_filename = f"GiorgiaMeloni_tweets_2021_2022_full{i}.csv"
    with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Date', 'ID', 'URL', 'Content', 'Likes', 'Retweets'])
        
        tweet_count = 0
        page_num = 0
        max_tweets = 100  
        
        try:
            
            max_pages = 5
            
            async for search_object, tweets in client.iter_search(date_query, pages=max_pages):
                page_num += 1
                
                print(f"Page {page_num}: Found {len(tweets)} tweets | Total so far: {tweet_count + len(tweets)}")
                
                if hasattr(search_object, 'cursor') and search_object.cursor:
                    print(f"Cursor: {search_object.cursor[:20]}...")
                
                for tweet in tweets:
                    date = getattr(tweet, 'created_on', 'Unknown')
                    tweet_id = getattr(tweet, 'id', 'Unknown')
                    content = getattr(tweet, 'text', getattr(tweet, 'content', 'No text'))
                    likes = getattr(tweet, 'likes', getattr(tweet, 'favorite_count', 0))
                    retweets = getattr(tweet, 'retweet_count', 0)
                    
                    writer.writerow([
                        date,
                        tweet_id,
                        f"https://twitter.com/GiorgiaMeloni/status/{tweet_id}" if tweet_id != 'Unknown' else 'Unknown',
                        content,
                        likes,
                        retweets
                    ])
                    
                    tweet_count += 1
                    
                    if tweet_count >= max_tweets:
                        print(f"Reached {max_tweets} tweets, stopping.")
                        break
                
                if tweet_count >= max_tweets:
                    break
                
                if page_num < max_pages:
                    delay = 2 
                    print(f"Waiting {delay} seconds before fetching next page...")
                    await asyncio.sleep(delay)
            
            print(f"\nTotal tweets collected: {tweet_count}")
            print(f"Results saved to: {csv_filename}")
    
            if tweet_count > 0:
                return get_last_date(csv_filename)
            else:
                return None
            
        except Exception as e:
            print(f"Error during search: {e}")
            print(f"Tweets collected before error: {tweet_count}")
            print(f"Results saved to: {csv_filename}")
            return None


async def run_iterations(start_date="2022-03-26", start_index=12, max_iterations=24):
    current_date = start_date
    
    for i in range(max_iterations):
        iteration_index = start_index + i
        print(f"Starting iteration {i+1}/{max_iterations} (file index: {iteration_index})")
        print(f"Using upper date bound: {current_date}")
        
        new_date = await main(final_date=current_date, i=iteration_index)
        
        if new_date:
            current_date = new_date
            print(f"New upper date bound for next iteration: {current_date}")
            
            if current_date <= "2021-01-01":
                print(f"Reached the target start date: {current_date}. Stopping iterations.")
                break
        else:
            print("No new date found or error occurred. Stopping iterations.")
            break
        
        print(f"Waiting 5 seconds before starting next iteration...")
        await asyncio.sleep(5)
    
    print("All iterations completed.")


nest_asyncio.apply()


asyncio.run(run_iterations(
    start_date="2021-10-15",  
    start_index=17,           
    max_iterations=24        
))