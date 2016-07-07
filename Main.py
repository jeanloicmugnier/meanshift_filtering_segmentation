from PIL import Image, ImageFilter
import numpy as np
import Color_Space as cs
from MSF import MSF

# img = Image.open()

msf = MSF("images/input.png")
img = msf.run_mean_shift(0.1,10,10,10)



'''
test luv OCNVERSION WRONG
'''
# pt = cs.Color_Space.convert_rgb_to_luv(np.array([10, 10, 10]))
# print(pt)
# pt = cs.Color_Space.convert_rgb_to_luv(np.array([100, 100, 100]))
# print(pt)
# pt = cs.Color_Space.convert_rgb_to_luv(np.array([1, 1, 1]))
# print(pt)

'''
TODO IMPLEMENT THE DISTANCE THING

h
K vizinhos mais próximas não deixa de ser força bruta? NÃO ARVORE BINARIA


'''
