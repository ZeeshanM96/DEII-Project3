import requests
import pandas as pd
from time import sleep
from tqdm import tqdm
from dotenv import load_dotenv
import os

load_dotenv()

GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
GITHUB_API_URL = 'https://api.github.com/search/repositories'
REPO_API_URL = 'https://api.github.com/repos'
HEADERS = {'Authorization': f'token {GITHUB_TOKEN}'}

# Function to get repositories data
def get_repositories(query, max_repos=1000):
    repos = []
    page = 1

    while len(repos) < max_repos:
        params = {
            'q': query,
            'sort': 'stars',
            'order': 'desc',
            'per_page': 100,
            'page': page
        }

        response = requests.get(GITHUB_API_URL, headers=HEADERS, params=params)
        if response.status_code != 200:
            print(f"Failed to fetch repositories: {response.status_code}")
            break

        data = response.json()
        repos.extend(data['items'])

        if len(data['items']) < 100:
            break

        page += 1
        sleep(1)  # Sleep to avoid hitting the rate limit

    return repos[:max_repos]

# Function to get detailed repository information
def get_repo_details(full_name):
    url = f'{REPO_API_URL}/{full_name}'
    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        print(f"Failed to fetch repository details for {full_name}: {response.status_code}")
        return None
    return response.json()

# Function to extract relevant fields from repository data
def extract_repo_info(repo):
    details = get_repo_details(repo['full_name'])
    if details is None:
        return None
    return {
        'name': repo['name'],
        'full_name': repo['full_name'],
        'html_url': repo['html_url'],
        'description': repo['description'],
        'created_at': repo['created_at'],
        'updated_at': repo['updated_at'],
        'pushed_at': repo['pushed_at'],
        'stargazers_count': repo['stargazers_count'],
        'forks_count': repo['forks_count'],
        'watchers_count': repo['watchers_count'],
        'open_issues_count': repo['open_issues_count'],
        'language': repo['language'],
        'default_branch': repo['default_branch'],
        'commits_count': details.get('commits_count'),
        'network_count': details.get('network_count'),
        'subscribers_count': details.get('subscribers_count')
    }

# Collect data
query = 'stars:>=50'
repositories = get_repositories(query)
repo_data = []

for repo in tqdm(repositories, desc="Fetching detailed repository data"):
    repo_info = extract_repo_info(repo)
    if repo_info:
        repo_data.append(repo_info)
    sleep(1)  # Sleep to avoid hitting the rate limit

# Save data to CSV
df = pd.DataFrame(repo_data)
df.to_csv('github_repositories_detailed.csv', index=False)

print("Data collection completed. Saved to github_repositories_detailed.csv.")

