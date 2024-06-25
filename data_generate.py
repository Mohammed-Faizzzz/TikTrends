import pandas as pd
import numpy as np
import random

# Seed for reproducibility
np.random.seed(42)

hashtags_pool = [
    '#tiktok', '#fun', '#music', '#love', '#dance', '#comedy', '#food', '#fashion', '#travel', '#fitness',
    '#tech', '#news', '#art', '#diy', '#pets', '#insideout', '#thalapathy', '#vijay', '#thalapathyvijay', 
    '#technology', '#mbappe', '#aasakooda', '#dennycaknan', '#babiesoftiktok', '#gracieabrams', '#chinesegirl', 
    '#students', '#plane', '#satire', '#thlapathyfan', '#explore', '#tel4', '#sharingiscaring', '#scam', 
    '#hairtutorial', '#text', '#nusantarahouse', '#mina', '#steamboat', '#copaamerica', '#21', '#pokkiri', 
    '#souvenir', '#glow', '#voiceover', '#hok', '#deodorant', '#groceryshopping', '#nikah', '#playstation', 
    '#lovequotes', '#housing', '#hermes', '#nintendoswitch', '#disgust', '#clip', '#wildanimals', '#babytok', 
    '#homelander', '#thomsoneastcoastline', '#oatside', '#faye', '#esports'
]

# Weights for the hashtags based on their popularity (higher values for more popular hashtags)
hashtag_weights = [
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 10, 5, 5, 4, 5, 5, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
]

captions_pool = [
    "Just another day in paradise 😊", "Who else loves #throwback Thursdays?", "Can't stop the feeling! 🎵",
    "Dream big, eat bigger 🍔", "When you remember tomorrow is Friday 🕺", "Life is better with friends 👯‍♀️",
    "Check out this amazing sunset! 🌅", "Do what you love, love what you do 💖", "Trying the new place in town #foodie",
    "Workout done, feel great! 💪 #fitness", "Explore the world 🌍 #travel", "My pet being adorable #pets",
    "DIY home decor #diy", "Latest tech gadgets reviewed #tech", "Breaking news and insights #news",
    "Watch till the end!", "This is not a drill", "Tag a friend who needs to see this", "Who can relate?",
    "This is so me", "When you're feeling", "The perfect day looks like this", "This is everything",
    "I can't even", "This is a mood", "This is a vibe", "This is a look", "This is a dream", "This is a nightmare",
    "This cannot be real", "I’m in shock", "I’m shooketh", "Not sure how I feel about this", "Life hack", "This is your sign",
    "This is a game changer", "This is a life saver", "This is a disaster", "This is a masterpiece", "This is a mess",
    "The moment you’ve been waiting for", "Look what just dropped", "I love this trend so much",
    "Typical", "Spill the tea", "SMH", "The assignment was fully understood", "I’m not crying, you’re crying", "You need to try this",
    "Your wish is our command", "Just trying to help", "Thanks but no thanks", "Run don’t walk", "I’m obsessed", "I’m addicted",
    "IYKYK", "You asked, we listened", "I got you, besties", "It’s serving", "It’s a serve", "It’s a look", "It’s a vibe", 
    "It’s a mood", "It's giving", "THIS IS NOT A DRILL", "Take all my money", "I see it, I like it, I want it, I got it",
    "Slayyy 😍", "I’m living for this", "I’m here for it", "I’m so here for this", "I’m so here for it", "I’m so here for that"
]

# Weights for captions
caption_weights = np.random.randint(1, 5, len(captions_pool))

locations_pool = [
    'New York, USA', 'Tokyo, Japan', 'London, UK', 'Paris, France', 'Sydney, Australia', 'Los Angeles, USA',
    'Berlin, Germany', 'Seoul, South Korea', 'Moscow, Russia', 'Dubai, UAE', 'San Francisco, USA', 'Beijing, China',
    'Bangkok, Thailand', 'Cape Town, South Africa', 'Rio de Janeiro, Brazil'
]

# Generate a more specific caption for popular hashtags
def generate_specific_caption(hashtag):
    if hashtag == '#insideout':
        return "Reliving the best moments with #insideout 😊"
    elif hashtag == '#thalapathy':
        return "All hail #thalapathy! 🙌"
    elif hashtag == '#vijay':
        return "Vijay's best performances! #vijay"
    elif hashtag == '#mbappe':
        return "Watch Mbappe's latest goals! #mbappe ⚽️"
    else:
        return random.choices(captions_pool, weights=caption_weights, k=1)[0]

# Generate synthetic TikTok data
data = {
    'Hashtag': random.choices(hashtags_pool, weights=hashtag_weights, k=3000),
    'Location': [random.choice(locations_pool) for _ in range(3000)]
}
data['Caption'] = [generate_specific_caption(hashtag) for hashtag in data['Hashtag']]

df = pd.DataFrame(data)

# Function to generate engagement metrics based on views
def generate_engagement(row):
    # Base views
    views = np.random.randint(5000, 5000000)
    
    # Engagement metrics as fractions of views
    likes = int(views * np.random.uniform(0.01, 0.1))
    comments = int(views * np.random.uniform(0.001, 0.01))
    shares = int(views * np.random.uniform(0.0005, 0.005))
    saves = int(views * np.random.uniform(0.0001, 0.002))
    
    # Influence from specific hashtags, captions, and location combinations
    if '#love' in row['Hashtag'] or 'life' in row['Caption'].lower():
        likes *= 1.5
        comments *= 1.3
        shares *= 1.2
    if '#food' in row['Hashtag'] or 'eat' in row['Caption'].lower() or 'Paris' in row['Location']:
        saves *= 1.5  # More saves for food-related or culturally rich content
    if '#fitness' in row['Hashtag'] or 'workout' in row['Caption'].lower():
        shares *= 1.5  # Fitness routines are more likely to be shared
    if 'New York' in row['Location'] or '#trend' in row['Hashtag']:
        views *= 1.2  # Boost for trend-setting locations and topics
    
    return min(5000000, int(views)), min(100000, int(likes)), min(50000, int(comments)), min(25000, int(shares)), min(10000, int(saves))

# Apply enhanced function to DataFrame
df['Views'], df['Likes'], df['Comments'], df['Shares'], df['Saves'] = zip(*df.apply(generate_engagement, axis=1))

# Display first few rows and save
print(df.head())
df.to_csv('synthetic_tiktok_data.csv', index=False)
