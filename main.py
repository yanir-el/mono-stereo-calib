import numpy as np
import polyscope as ps
import re
import os
import sys
from skimage import transform
'''
Load a PFM file into a Numpy array. Note that it will have
a shape of H x W, not W x H. Returns a tuple containing the
loaded image and the scale factor from the file.
'''
def load_pfm(file):
  color = None
  width = None
  height = None
  scale = None
  endian = None

  header = header = file.readline().decode('utf-8').rstrip()
  if header == 'PF':
    color = True
  elif header == 'Pf':
    color = False
  else:
    raise Exception('Not a PFM file.')

  dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
  if dim_match:
    width, height = map(int, dim_match.groups())
  else:
    raise Exception('Malformed PFM header.')

  scale = float(file.readline().rstrip())
  if scale < 0: # little-endian
    endian = '<'
    scale = -scale
  else:
    endian = '>' # big-endian

  data = np.fromfile(file, endian + 'f')
  shape = (height, width, 3) if color else (height, width)
  return np.reshape(data, shape), scale



def pointcloud(depth, fov):
#  fy = fx = 0.5 / np.tan(fov * 0.5)  # assume aspectRatio is one.
  (fx,fy)= fov
  height = depth.shape[0]
  width = depth.shape[1]

  mask = np.where(depth > 0)

  x = mask[1]
  y = mask[0]

  normalized_x = (x.astype(np.float32) - width * 0.5) / width
  normalized_y = (y.astype(np.float32) - height * 0.5) / height

  world_x = normalized_x * depth[y, x] / fx
  world_y = normalized_y * depth[y, x] / fy
  world_z = 0.25*depth[y, x]
  ones = np.ones(world_z.shape[0], dtype=np.float32)

  return np.vstack((world_x, world_y, world_z, ones)).T

if __name__ == '__main__':

  directory = r'C:\Users\valen\PycharmProjects\pfm_to_numpy'
#  for filename in os.listdir(directory):
 #   if filename.endswith(".pfm"):
  #    with open(filename, "rb") as f:
   #       depth_map,scale=load_pfm(f)
    #      fov = 0.5418937007   #iphone_XR fov
     #     cloud = pointcloud(depth_map,fov)
      #    path=filename.rsplit(".")[0]
       #   np.save(path,cloud)

  with open("13 - right up.pfm", "rb") as f:
    depth_map, scale = load_pfm(f)
    fov = (3,4)

    cloud = pointcloud(depth_map, fov)
    path="test10.npy"
    np.save(path,cloud)







