import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO


def load_image_from_url(url):
    response=requests.get(url)
    return Image.open(BytesIO(response.content))

elephant_url="https://cdn.britannica.com/02/152302-050-1A984FCB/African-savanna-elephant.jpg"
elephant=load_image_from_url(elephant_url)

plt.figure(figsize=(6,4))
plt.imshow(elephant)
plt.title('elephant')
plt.axis("off")
plt.show()


elephant_np=np.array(elephant)
print("Elephant image shape:",elephant_np.shape)


elephant_gray=elephant.convert("L")



plt.figure(figsize=(6,4))
plt.imshow(elephant_gray,cmap="berlin")
plt.title("Elephant (Grayscale)")
plt.axis("off")
plt.show()