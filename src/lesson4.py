import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

fileName = "../data/lesson3/MarshOrchid.jpg"
raw_image_data = mpimg.imread(fileName)


image = tf.placeholder("uint8",[None,None,3])
slice = tf.slice(image,  [1000, 0, 0], [3000, -1, -1])

# with tf.Session() as session:
#   result = session.run(slice,feed_dict={image:raw_image_data})
#   print(result.shape)
#   
# plt.imshow(result)
# plt.show()

data = [[[1,2,3],[4,5,6]],[[4,5,6],[7,8,9]],[[7,8,9],[10,11,12]]]
data = np.array(data)
print(data.shape)
print(data[0][0][0])
print(data[1][0][0])
print(data[2][0][0])
squares = tf.foldl(lambda a, x: a + x, data)

with tf.Session() as session:
#   result = session.run(slice,feed_dict={image:raw_image_data})
  resultsSq = session.run(squares)
  print(resultsSq)
  print(resultsSq.shape)

# plt.imshow(result)
# plt.show()