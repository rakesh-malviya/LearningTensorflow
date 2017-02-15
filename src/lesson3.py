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
print(image.shape)
# height, width, depth = image.shape

# Create a TensorFlow Variable
x = tf.Variable(image, name='x')
shapeX = tf.shape(x)
model = tf.global_variables_initializer()

with tf.Session() as sess:
  #  We take each column 
  
  height, width, depth = sess.run(shapeX)
  tempX = x  
  x = tf.reverse_sequence(x, [height//2] * width, seq_dim=0, batch_dim=1)
  
  
  x = tf.transpose(x, [1,0,2])
  sess.run(model)
  result = sess.run(x)
  
  tempX1 = tf.slice(tempX, [0,0,0], [height,width//2+1,depth])
  tempX2 = tf.slice(tempX, [0,0,0], [height,width//2,depth])
  tempX2shape = tf.shape(tempX2)
  height2, width2, depth2 = tf.gather(tempX2shape,0),tf.gather(tempX2shape,1),tf.gather(tempX2shape,2)
  width2 = tf.reshape(width2, [-1])
  height2 = tf.reshape(height2, [-1])
  input1 = tf.tile(width2, height2)
  tempX2 = tf.reverse_sequence(tempX2, input1, seq_dim=1, batch_dim=0)  
  tempX = tf.concat(1,[tempX1,tempX2])
  resultMirror = sess.run(tempX)
  h2 = sess.run(input1)
    

print(result.shape)
plt.imshow(result)
plt.show()

# print h2

print(resultMirror.shape)
plt.imshow(resultMirror)
plt.show()