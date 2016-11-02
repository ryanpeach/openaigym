import gym
import argparse

parser = argparse.ArgumentParser(description='Upload path for Open AI gym.')
parser.add_argument('folder', type=str, help='Folder name in ./output/ to upload.')
parser.add_argument('API', type=str, help='Open AI Gym API Key.')

args = parser.parse_args()
API = args.API
folder_name = args.folder
gym.upload('./output/'+folder_name+'/', api_key=API)