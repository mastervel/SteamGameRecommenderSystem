from steam_api_pkg.steam_user import SteamUser
import random
import os
import csv
import collections
from tqdm import tqdm
import time
import pandas as pd
import warnings
import requests

def getRandomSampleOfFriends(api_key: str, steam_id: str, k: int):
    data = SteamUser(api_key,steam_id).getFriendsList()
    friends_list = [friend['steamid'] for friend in data]
    if not friends_list:
        return None
    else:
        sample = random.choices(friends_list, k = k)
    return sample

def sample_to_dataframe(user_list, user_games_list):
    # Prepare data by combining user IDs with their respective games
    data = []
    for user, games in zip(user_list, user_games_list):
        for game in games:
            game['user_id'] = user  # Add user_id to each game record
            data.append(game)

    # Convert data to DataFrame
    df = pd.DataFrame(data)
    return df

def get_sampled_users(file_name: str = "agg_data.csv"):
    sampled_users = set()  # Initialize the set to store sampled users

    try:
        # Check if the file exists before attempting to open it
        if not os.path.exists(file_name):
            raise FileNotFoundError(f"The file '{file_name}' was not found in the directory.")
        
        # Read the CSV if the file exists
        with open(file_name, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)  # Reads CSV as a dictionary
            sampled_users = set([row['user_id'] for row in reader])

    except FileNotFoundError as e:
        print(e)
        # Return an empty set if the file is not found
        sampled_users = set()  # Ensure it's empty

    return sampled_users


def getSampleByDOS(api_key: str, seed: str = "76561198059056914", 
                   max_depth: int = 6, k: int = 10, sample_size: int = 100, filename: str = "agg_data.csv"):

    sampled_users = get_sampled_users(filename)
    
    # Objects
    queue = collections.deque([(seed, 0)])
    visited = set([seed]) # Used to prevent repeated samples of the same user friends list
    
    # Initialize progress bar
    progress_bar = tqdm(total=sample_size - len(sampled_users), desc="Sampling Users", unit="user", dynamic_ncols=True)
    
    start_time = time.time()

    # Storing Owned Games data
    user_list = []
    user_games_list = []
    
    # Loop
    while queue and len(sampled_users) < sample_size:
        steam_id, depth = queue.popleft()
        
        if depth > max_depth:
            break # if max depth is achieved break the loop
    
        friends = getRandomSampleOfFriends(api_key, steam_id, k = k)
        if not friends:
            continue # if the friends is null we more to the next in queue

        prev_count = len(sampled_users) # for progress bar tracking users sampled
        
        new_friends = [friend for friend in friends if friend not in visited] # filter out already visited users
        for friend in new_friends:
            queue.append((friend, depth + 1)) # add to queue first coz already checked for visited
            
            player = SteamUser(api_key,friend)
            
            try:
                data = player.getOwnedGames()
            except requests.exceptions.JSONDecodeError:
                visited.add(friend)
                continue
                
            if not data or 'games' not in data:
                visited.add(friend)
                continue # skipping users without a games data available
    
            games_list = data['games']
            if all(game['playtime_forever'] == 0 for game in games_list):
                visited.add(friend)
                continue # skipping users without playtime enabled for public access
                    
            if friend not in sampled_users:
                user_list.append(friend)
                user_games_list.append(games_list)
                sampled_users.add(friend)
                visited.add(friend)

        
        # Update progress bar
        added_count = len(sampled_users) - prev_count
        if added_count > 0:
            progress_bar.update(added_count)
    
    progress_bar.close()
    
    # End timer
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Output runtime
    print(f"Script finished in {elapsed_time:.4f} seconds")
    #print(f"Total API calls made: {api_calls}")
    return user_list, user_games_list

def save_to_agg_data(df: pd.DataFrame, file_name: str = "agg_data.csv"):
    # Check if the file already exists
    file_exists = os.path.exists(file_name)
    if file_exists:
        agg_df = pd.read_csv("agg_data.csv")
        agg_df = pd.concat([agg_df, df])
        agg_df.to_csv("agg_data.csv",index=False)
    else: df.to_csv("agg_data.csv",index=False)

def getTimeCreated(api_key: str, batch_size: int, filename: str = "agg_data.csv"):
    # Warn if batch_size exceeds 100 and cap it
    if batch_size > 100:
        warnings.warn("batch_size exceeded 100. It has been set to the maximum limit of 100.", UserWarning)
        batch_size = 100

    # Previously Sampled Users
    sampled_users = collections.deque(get_sampled_users())

    return_dict = {}

    while sampled_users:
        # Limit batch size to the smaller of batch_size or remaining sampled_users
        user_ids = [sampled_users.popleft() for _ in range(min(batch_size, len(sampled_users)))]
        user_ids_str = ",".join(map(str, user_ids))
        
        url = "http://api.steampowered.com/ISteamUser/GetPlayerSummaries/v0002/"
        params = {
            'key': api_key,
            'steamids': user_ids_str
        }
        response = requests.get(url, params=params)
        
        # Check for errors
        if response.status_code != 200:
            raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
        
        # Parse JSON response
        data = response.json()
        
        # Extract player summaries
        data = data.get("response", {}).get("players", [])
        time_created_dict = {user['steamid']: user.get("timecreated") for user in data}

        return_dict.update(time_created_dict)

    return return_dict
