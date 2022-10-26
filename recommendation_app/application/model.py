import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
import re

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import json
import pandas as pd
def generate_playlist_feature(URL,complete_feature_set, playlist_df):
    '''
    Summarize a user's playlist into a single vector
    ---
    Input: 
    complete_feature_set (pandas dataframe): Dataframe which includes all of the features for the spotify songs
    playlist_df (pandas dataframe): playlist dataframe
        
    Output: 
    complete_feature_set_playlist_final (pandas series): single vector feature that summarizes the playlist
    complete_feature_set_nonplaylist (pandas dataframe): 
    '''
    
    # Find song features in the playlist
    complete_feature_set_playlist = complete_feature_set[complete_feature_set['id'].isin(playlist_df['id'].values)]
    # Find all non-playlist song features
        
    if complete_feature_set_playlist.empty==True:
        client_id = "5356afb958c84e71a2c37c43e2a2cbf2" 
        client_secret = "83e531491e9c458ba658ac30c4c56bc0"

        #use the clint secret and id details
        client_credentials_manager = SpotifyClientCredentials(client_id=client_id,client_secret=client_secret)
        sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
        playlist_id = URL.split("/")[4].split("?")[0]
        playlist_tracks_data = sp.playlist_tracks(playlist_id)

        playlist_tracks_id = []
        for track in playlist_tracks_data['items']:
            playlist_tracks_id.append(track['track']['id'])
        features = sp.audio_features(playlist_tracks_id)
        complete_feature_set_playlist = pd.DataFrame(data=features, columns=features[0].keys())
        complete_feature_set_playlist=complete_feature_set_playlist[['danceability','energy','loudness','speechiness','acousticness','instrumentalness','liveness','valence','tempo','key','mode','duration_ms','id']]
    complete_feature_set_nonplaylist = complete_feature_set[~complete_feature_set['id'].isin(playlist_df['id'].values)]
    complete_feature_set_playlist_final = complete_feature_set_playlist.drop(columns = "id")    
    return complete_feature_set_playlist_final.sum(axis = 0), complete_feature_set_nonplaylist


def generate_playlist_recos(df, features, nonplaylist_features):
    '''
    Generated recommendation based on songs in aspecific playlist.
    ---
    Input: 
    df (pandas dataframe): spotify dataframe
    features (pandas series): summarized playlist feature (single vector)
    nonplaylist_features (pandas dataframe): feature set of songs that are not in the selected playlist
        
    Output: 
    non_playlist_df_top_40: Top 40 recommendations for that playlist
    '''
    
    non_playlist_df = df[df['id'].isin(nonplaylist_features['id'].values)]
    # Find cosine similarity between the playlist and the complete song set
    non_playlist_df['sim'] = cosine_similarity(nonplaylist_features.drop('id', axis = 1).values, features.values.reshape(1, -1))[:,0]
    non_playlist_df_top_40 = non_playlist_df.sort_values('sim',ascending = False).head(50)
    non_playlist_df_top_40=non_playlist_df_top_40[non_playlist_df_top_40.pos>90]
    return non_playlist_df_top_40


def recommend_from_playlist(URL,songDF,complete_feature_set,playlistDF_test):

    # Find feature
    complete_feature_set_playlist_vector, complete_feature_set_nonplaylist = generate_playlist_feature(URL,complete_feature_set, playlistDF_test)
    
    # Generate recommendation
    top40 = generate_playlist_recos(songDF, complete_feature_set_playlist_vector, complete_feature_set_nonplaylist)

    return top40
