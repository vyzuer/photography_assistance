import numpy as np
import cv2
from skimage import transform as tf

shape = (1, 10, 2) # Needs to be a 3D array
source = np.random.randint(0, 100, shape).astype(np.int)
target = source + np.array([1, 0]).astype(np.int)
transformation = cv2.estimateRigidTransform(source, target, False)
print transformation

t_form = tf.estimate_transform('affine', source[0], target[0])

print t_form.params
print t_form.scale, t_form.shear, t_form.rotation, t_form.translation
