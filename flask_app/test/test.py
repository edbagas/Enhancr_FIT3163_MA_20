import requests
import os
import glob

image_paths = glob.glob('C:\\Users\\Lenovo\\Desktop\\flask_app\\test\\images\\*')
resp = requests.post('http://localhost:5000/enhance',files={'file': image_paths})

print(resp)