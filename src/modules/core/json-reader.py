from urllib.request import urlopen
import json

class JSonReader:
    def readJson(self):
        url = "https://api.github.com"
        response = urlopen(url)
        data_json = json.loads(response.read())
        print(data_json)

JSonReader().readJson()