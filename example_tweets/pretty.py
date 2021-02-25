import json

with open("tweet.json", "r") as fobj:
    parsed = json.loads(fobj.read())

with open("tweet.json", "w") as fobj:
    print(json.dumps(parsed, indent=4), file=fobj)
