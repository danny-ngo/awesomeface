import cognitive_face as CF
import matplotlib.pyplot as plt
import json
from PIL import Image
from io import BytesIO
import requests

LIST_NUMBER = 10

KEY = "efb1c590aab74d90875f4c3814436885"  # Replace with a valid subscription key (keeping the quotes in place).
CF.Key.set(KEY)

BASE_URL = 'https://westcentralus.api.cognitive.microsoft.com/face/v1.0'  # Replace with your regional Base URL
CF.BaseUrl.set(BASE_URL)

# You can use this example JPG or replace the URL below with your own URL to a JPEG image.
img_url = "https://amp.businessinsider.com/images/5bd09e10d2e64856c555267a-750-563.jpg"
star = CF.face.detect(img_url)

'''
CF.face_list.create(LIST_NUMBER)

with open("BigList.txt") as f:
    content = f.readlines()
    for line in content:
        try:
            line = line[:(len(line) - 1)]
            CF.face_list.add_face(line, LIST_NUMBER, line + "", None)
        except:
            x = 1

'''

similarity = CF.face.find_similars(star[0]['faceId'], LIST_NUMBER, None, None, 1, 'matchFace')
print(similarity)
masterList = CF.face_list.get(LIST_NUMBER)
print(masterList)

masterList = masterList["persistedFaces"]
foundURL = ""
for entry in masterList:
    if entry["persistedFaceId"] == similarity[0]['persistedFaceId']:
        foundURL = entry['userData']

image = Image.open(BytesIO(requests.get(foundURL).content))
plt.imshow(image)
print()
plt.axis("off")
_ = plt.title(similarity[0]['confidence'], size="x-large", y=-0.1)
plt.show()

print(similarity)
