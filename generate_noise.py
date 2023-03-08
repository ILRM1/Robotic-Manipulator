import matplotlib.pyplot as plt
from perlin_noise import PerlinNoise
import random
import os

if not os.path.exists('noise'):
    os.makedirs('noise')

cmaps = plt.colormaps()
xpix, ypix = 369, 369

num=0
for _ in range(50):
    for cmap in cmaps:
        pic = []
        noise1 = PerlinNoise(octaves=random.randint(1,10))
        noise2 = PerlinNoise(octaves=random.randint(1,20))
        noise3 = PerlinNoise(octaves=random.randint(1,30))
        noise4 = PerlinNoise(octaves=random.randint(1,40))
        for i in range(xpix):
            row = []
            for j in range(ypix):
                noise_val = noise1([i/xpix, j/ypix])
                noise_val += random.random() * noise2([i/xpix, j/ypix])
                noise_val += random.random() * noise3([i/xpix, j/ypix])
                noise_val += random.random() * noise4([i/xpix, j/ypix])

                row.append(noise_val)
            pic.append(row)

        plt.imshow(pic, cmap=cmap)
        plt.axis('off')
        plt.savefig('/home/user/PycharmProjects/iros_code/noise/'+str(num)+'.png', bbox_inches="tight", pad_inches=0)
        num+=1