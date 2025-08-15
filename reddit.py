import time
import httpx
import pandas as pd
from httpx import ReadTimeout

base_url = 'https://www.reddit.com'
endpoint = '/r/Positive'
category = '/hot'
url = base_url + endpoint + category + ".json"

after_post_id = None
dataset = []

headers = {'User-Agent': 'MyRedditApp/0.0.1'}
max_retries = 3
total_posts = 0
target_posts = 10000  # Number of threads you want to extract

while total_posts < target_posts:
    params = {
        'limit': 100,
        't': 'year',
        'after': after_post_id
    }

    for attempt in range(max_retries):
        try:
            response = httpx.get(url, headers=headers, params=params, timeout=30)
            print(f'Fetching "{response.url}"...')
            if response.status_code == 200:
                break
        except ReadTimeout:
            print(f"Timeout occurred, retrying... (Attempt {attempt + 1}/{max_retries})")
            time.sleep(2)
    else:
        raise Exception("Failed to fetch data after multiple retries")

    json_data = response.json()
    posts = [rec['data'] for rec in json_data['data']['children']]
    dataset.extend(posts)
    total_posts += len(posts)
    after_post_id = json_data['data']['after']
    print(f"Fetched {len(posts)} posts. Total collected: {total_posts}")

    # Stop if there are no more posts to fetch
    if not after_post_id:
        print("No more posts available to fetch.")
        break

    time.sleep(1)  # Adjust the sleep time if needed to avoid rate limiting

# Convert to DataFrame and save to CSV
df = pd.DataFrame(dataset)
df.to_csv('reddit_positive_2.csv', index=False)
print(f"Saved {len(df)} posts to 'reddit_positive_2.csv'")
