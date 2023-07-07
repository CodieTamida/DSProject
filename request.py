import requests

url = 'http://localhost:5000/api/'

payload = {
    'fa': 0,
    'va': 0,
    'ca': 0,
    'rs':0,
    'chlorides':0,
    'fsd':0,
    'tsd':0,
    'density': 0,
    'ph': 0,
    'sulphates':0,
    'alcohol': 0

}
#r = requests.post(url, jsonify={'exp':1.8})
r = requests.post(url, jsonify={'fa':1.8, 'va':0, 'ca':0})

print(r.json())
