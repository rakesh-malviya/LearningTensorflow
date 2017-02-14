# import matplotlib.image as mpimg
# # First, load the image
# filename = "../data/lesson3/MarshOrchid.jpg"
# image = mpimg.imread(filename)
# 
# # Print out its shape
# print(image.shape)
# 
# import matplotlib.pyplot as plt
# plt.imshow(image)
# plt.show()

import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

filename = "../data/lesson3/MarshOrchid.jpg"
image = mpimg.imread(filename)

# Create a TensorFlow Variable
x = tf.Variable(image, name='x')

model = tf.global_variables_initializer()

with tf.Session() as sess:
  x = tf.transpose(x, [1,0,2])
  sess.run(model)
  result = sess.run(x)
  
plt.imshow(result)
plt.show()
