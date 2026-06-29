import json, os, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

if os.path.exists('results.json'):
    with open('results.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f'Total POIs: {len(data)}')
    for i, p in enumerate(data[:15]):
        print(f'{i+1}. {p["name"]}')
else:
    print('No results.json yet')

if os.path.exists('checkpoint.json'):
    with open('checkpoint.json', 'r', encoding='utf-8') as f:
        cp = json.load(f)
    visited = cp.get('visited', [])
    queue = cp.get('queue', [])
    print(f'\nCheckpoint: {len(visited)} visited, {len(queue)} queued')
