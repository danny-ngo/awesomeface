import cognitive_face as CF
import matplotlib.pyplot as plt
import json
from PIL import Image
from io import BytesIO
import requests

LIST_NUMBER = 1

KEY = "efb1c590aab74d90875f4c3814436885"  # Replace with a valid subscription key (keeping the quotes in place).
CF.Key.set(KEY)

BASE_URL = 'https://westcentralus.api.cognitive.microsoft.com/face/v1.0'  # Replace with your regional Base URL
CF.BaseUrl.set(BASE_URL)

# You can use this example JPG or replace the URL below with your own URL to a JPEG image.
img_url = "https://fm.cnbc.com/applications/cnbc.com/resources/img/editorial/2013/02/26/100496736-steve-jobs-march-2011-getty.1910x1000.jpg"
star = CF.face.detect(img_url)

'''
with open("BigList.txt") as f:
    content = f.readlines()
    for line in content:
        try:
            line = line[:(len(line) - 1)]
            CF.face_list.add_face(line,LIST_NUMBER , line)
        except:
            x = 1
'''

similarity = CF.face.find_similars(star[0]['faceId'], LIST_NUMBER, None, None, 1, 'matchFace')

masterList = CF.face_list.get(1)
print(masterList)

image = Image.open(BytesIO(requests.get(img_url).content))
plt.imshow(image)
print()
plt.axis("off")
_ = plt.title(similarity[0]['confidence'], size="x-large", y=-0.1)
plt.show()

print(similarity)
