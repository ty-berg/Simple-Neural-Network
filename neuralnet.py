import math
import numpy as np
import matplotlib.pyplot as plt
import random as random

def trainBayes(a):
  data = np.array([[1,-1,-1,0],[1,0,1,0],[0,0,-1,0],[0,-1,1,0],[1,1,1,1],[0,1,0,1],[0,0,1,1],[-1,0,1,1]])
  py = .5
  xparams = np.array([[(0+a)/(4+a*3),(2+a)/(4+a*3),(2+a)/(4+a*3)],[(2+a)/(4+a*3),(2+a)/(4+a*3),(0+a)/(4+a*3)],[(2+a)/(4+a*3),(0+a)/(4+a*3),(2+a)/(4+a*3)],
                      [(1+a)/(4+a*3),(2+a)/(4+a*3),(1+a)/(4+a*3)],[(0+a)/(4+a*3),(2+a)/(4+a*3),(2+a)/(4+a*3)],[(0+a)/(4+a*3),(1+a)/(4+a*3),(3+a)/(4+a*3)]])
  y1 = np.zeros(8)
  y2 = np.zeros(8)
  x = 0
  samples = [1,2,3,4,5,6,7,8]
  for d in data:
      if a==0:
          y1[x] = py * xparams[0,d[0]+1]*xparams[1,d[1]+1]*xparams[2,d[2]+1]
          y2[x] = py * xparams[3,d[0]+1]*xparams[4,d[1]+1]*xparams[5,d[2]+1]
      if a==1:
        y1[x] = xparams[0,d[0]+1]*xparams[1,d[1]+1]*xparams[2,d[2]+1]
        y2[x] = xparams[3,d[0]+1]*xparams[4,d[1]+1]*xparams[5,d[2]+1]
      x = x+1
  for i in range(len(y1)):
    sum = y1[i]+y2[i]
    y1[i] = y1[i]/(sum)
    y2[i] = y2[i]/(sum)
  plt.plot(samples,y1,label = "Pr(Y=0)")
  plt.plot(samples,y2, label = "Pr(Y=1)")
  plt.legend()
  plt.show()

def linearClass():
  fig = plt.figure()
  ax = plt.axes(projection='3d')
  x1z = np.array([1,1,0,0])
  x2z = np.array([-1,0,0,-1]) 
  x3z = np.array([-1,1,-1,1])
  x1o = np.array([1,0,0,-1])
  x2o = np.array([1,1,0,0])
  x3o = np.array([1,0,1,1])
  ax.scatter(x1z,x2z,x3z,marker = "o")
  ax.scatter(x1o,x2o,x3o,marker = "^")

  ax.set_xlabel('x1')
  ax.set_ylabel('x2')
  ax.set_zlabel('x3')

  ax.view_init(-140, 60)

def perceptron1(a):
  w = [0,0,0,0]
  w1 = np.array([0])
  w2 = np.array([0])
  w3 = np.array([0])
  w4 = np.array([0])
  conv = False
  data = np.array([[1,-1,-1,0],[1,0,1,0],[0,0,-1,0],[0,-1,1,0],[1,1,1,1],[0,1,0,1],[0,0,1,1],[-1,0,1,1]])
  count = 0
  x = np.zeros(4)
  while not conv:
    for d in data:
      x[0] = a
      for i in range(1,4):
        x[i] = d[i-1]
      act = np.dot(w,x)
      if act < 0:
        if d[3] == 0:
          count = count + 1
        else:
          w = w + x
      if act > 0:
        if d[3] == 1:
          count = count + 1
        else:
          w = w - x
      if act == 0:
        if d[3] == 0:
          count = count + 1
        else:
          w = w + x
    w1 = np.append(w1,w[0])
    w2 = np.append(w2,w[1])
    w3 = np.append(w3,w[2])
    w4 = np.append(w4,w[3])
    if count == 8:
      conv = True
    else:
      count = 0

  size = len(w1)
  xax = np.zeros(size)
  for i in range(size):
    xax[i] = i
  plt.plot(xax,w1, label = "w1")
  plt.plot(xax,w2, label = "w2")
  plt.plot(xax,w3, label = "w3")
  plt.plot(xax,w4, label = "w4")
  plt.legend()
  plt.show()

  plt.figure()
  sig = np.zeros(8)
  samples = [1,2,3,4,5,6,7,8]
  index = 0
  for d in data:
    x[0] = a
    for i in range(1,4):
      x[i] = d[i-1]
    sig[index] = 1/(1+ math.exp(np.dot(-w,x)))
    index = index + 1
  print(sig)
  plt.plot(samples,sig) 
  plt.show()

def perceptron2(a):
  w = [0,0,0,0]
  w1 = np.array([0])
  w2 = np.array([0])
  w3 = np.array([0])
  w4 = np.array([0])
  conv = False
  data = np.array([[1,-1,-1,0],[1,0,1,0],[0,0,-1,0],[0,-1,1,0],[1,1,1,1],[0,1,0,1],[0,0,1,1],[-1,0,1,1]])
  count = 0
  x = np.zeros(4)
  while not conv:
    for d in data:
      x[0] = a
      for i in range(1,4):
        x[i] = d[i-1]
      act = np.dot(w,x)
      if act < 0:
        if d[3] == 0:
          count = count + 1
        else:
          w = w + x
      if act > 0:
        if d[3] == 1:
          count = count + 1
        else:
          w = w - x
      if act == 0:
        if d[3] == 1:
          count = count + 1
        else:
          w = w - x
    w1 = np.append(w1,w[0])
    w2 = np.append(w2,w[1])
    w3 = np.append(w3,w[2])
    w4 = np.append(w4,w[3])
    if count == 8:
      conv = True
    else:
      count = 0  
  
  size = len(w1)
  xax = np.zeros(size)
  for i in range(size):
    xax[i] = i
  plt.plot(xax,w1, label = "w1")
  plt.plot(xax,w2, label = "w2")
  plt.plot(xax,w3, label = "w3")
  plt.plot(xax,w4, label = "w4")
  plt.legend()
  plt.show()
  
  plt.figure()
  sig = np.zeros(8)
  samples = [1,2,3,4,5,6,7,8]
  index = 0
  for d in data:
    x[0] = a
    for i in range(1,4):
      x[i] = d[i-1]
    sig[index] = 1/(1+ math.exp(np.dot(-w,x)))
    index = index + 1
  print(sig)
  plt.plot(samples,sig) 
  plt.show()

def backprop(a):
  w1 = np.array([[random.random(),random.random()],[random.random(),random.random()],[random.random(),random.random()],[random.random(),random.random()]])
  w2 = np.array([[random.random()],[random.random()],[random.random()]])
  data = np.array([[1,-1,-1,0],[1,0,1,0],[0,0,-1,0],[0,-1,1,0],[1,1,1,1],[0,1,0,1],[0,0,1,1],[-1,0,1,1]])
  conv = 0
  max = 100
  loss = np.zeros((8,max))
  while conv < max:
    index = 0
    for d in data:
      g1 = 1/(1 + math.exp(-(w1[1,0]*d[0] + w1[2,0]*d[1] + w1[3,0]*d[2] + w1[0,0])))
      g2 = 1/(1 + math.exp(-(w1[1,1]*d[0] + w1[2,1]*d[1] + w1[3,1]*d[2] + w1[0,1])))
      g3 = 1/(1 + math.exp(-(w2[1,0]*g1 + w2[2,0]*g2 + w2[0,0])))
      l = .5*(d[3]-g3)**2
      loss[index,conv] = l
      index = index + 1
      dg3 = (1/(1 + math.exp(-(w2[1,0]*g1 + w2[2,0]*g2 + w2[0,0])))) * (1-1/(1 + math.exp(-(w2[1,0]*g1 + w2[2,0]*g2 + w2[0,0]))))
      dw201 = -(d[3]-g3) * dg3
      dw221 = -(d[3]-g3) * dg3 * g2
      dw211 = -(d[3]-g3) * dg3 * g1
      dg1 = (1/(1 + math.exp(-(w1[1,0]*d[0] + w1[2,0]*d[1] + w1[3,0]*d[2] + w1[0,0]))))*(1-1/(1 + math.exp(-(w1[1,0]*d[0] + w1[2,0]*d[1] + w1[3,0]*d[2] + w1[0,0]))))
      dg2 = (1/(1 + math.exp(-(w1[1,1]*d[0] + w1[2,1]*d[1] + w1[3,1]*d[2] + w1[0,1]))))*(1-1/(1 + math.exp(-(w1[1,1]*d[0] + w1[2,1]*d[1] + w1[3,1]*d[2] + w1[0,1]))))
      dw101 = -(d[3]-g3) * dg3 * dg1
      dw102 = -(d[3]-g3) * dg3 * dg2
      dw111 = -(d[3]-g3) * dg3 * dg1 *d[0]
      dw121 = -(d[3]-g3) * dg3 * dg1 *d[1]
      dw131 = -(d[3]-g3) * dg3 * dg1 *d[2]
      dw112 = -(d[3]-g3) * dg3 * dg2 *d[0]
      dw122 = -(d[3]-g3) * dg3 * dg2 *d[1]
      dw132 = -(d[3]-g3) * dg3 * dg2 *d[2]
      grad1 = np.array([[dw101,dw102],[dw111,dw112],[dw121,dw122],[dw131,dw132]])
      grad2 = np.array([[dw201],[dw211],[dw221]])
      w1 = w1 - a*grad1
      w2 = w2 - a*grad2
    conv = conv + 1
  plt.figure()
  xaxis = list(range(1,max + 1))
  plt.plot(xaxis,loss[0,:], label = "Sample 1")
  plt.plot(xaxis,loss[1,:], label = "Sample 2")
  plt.plot(xaxis,loss[2,:], label = "Sample 3")
  plt.plot(xaxis,loss[3,:], label = "Sample 4")
  plt.plot(xaxis,loss[4,:], label = "Sample 5")
  plt.plot(xaxis,loss[5,:], label = "Sample 6")
  plt.plot(xaxis,loss[6,:], label = "Sample 7")
  plt.plot(xaxis,loss[7,:], label = "Sample 8")
  plt.legend()
  plt.show()
  print(w1)
  print(w2)

def backprop1(a):
  w1 = np.array([[random.random(),random.random()],[random.random(),random.random()],[random.random(),random.random()],[random.random(),random.random()]])
  w2 = np.array([[random.random()],[random.random()],[random.random()]])
  data = np.array([[1,-1,-1,0],[1,0,1,0],[0,0,-1,0],[0,-1,1,0],[1,1,1,1],[0,1,0,1],[0,0,1,1],[-1,0,1,1]])
  conv = 0
  max = 100
  loss = np.zeros((8,max))
  while conv < max:
    index = 0
    for d in data:
      g1 = np.tanh(w1[1,0]*d[0] + w1[2,0]*d[1] + w1[3,0]*d[2] + w1[0,0])
      g2 = np.tanh(w1[1,1]*d[0] + w1[2,1]*d[1] + w1[3,1]*d[2] + w1[0,1])
      g3 = np.tanh(w2[1,0]*g1 + w2[2,0]*g2 + w2[0,0])
      l = .5*(d[3]-g3)**2
      loss[index,conv] = l
      index = index + 1
      #derivative of tanh found at: https://socratic.org/questions/what-is-the-derivative-of-tanh-x
      dg3 = 1-np.tanh(w2[1,0]*g1 + w2[2,0]*g2 + w2[0,0])**2
      dw201 = -(d[3]-g3) * dg3
      dw221 = -(d[3]-g3) * dg3 * g2
      dw211 = -(d[3]-g3) * dg3 * g1
      dg1 = 1-np.tanh(w1[1,0]*d[0] + w1[2,0]*d[1] + w1[3,0]*d[2] + w1[0,0])**2
      dg2 = 1-np.tanh(w1[1,1]*d[0] + w1[2,1]*d[1] + w1[3,1]*d[2] + w1[0,1])**2
      dw101 = -(d[3]-g3) * dg3 * dg1
      dw102 = -(d[3]-g3) * dg3 * dg2
      dw111 = -(d[3]-g3) * dg3 * dg1 *d[0]
      dw121 = -(d[3]-g3) * dg3 * dg1 *d[1]
      dw131 = -(d[3]-g3) * dg3 * dg1 *d[2]
      dw112 = -(d[3]-g3) * dg3 * dg2 *d[0]
      dw122 = -(d[3]-g3) * dg3 * dg2 *d[1]
      dw132 = -(d[3]-g3) * dg3 * dg2 *d[2]
      grad1 = np.array([[dw101,dw102],[dw111,dw112],[dw121,dw122],[dw131,dw132]])
      grad2 = np.array([[dw201],[dw211],[dw221]])
      w1 = w1 - a*grad1
      w2 = w2 - a*grad2
    conv = conv + 1
  plt.figure()
  xaxis = list(range(1,max + 1))
  plt.plot(xaxis,loss[0,:], label = "Sample 1")
  plt.plot(xaxis,loss[1,:], label = "Sample 2")
  plt.plot(xaxis,loss[2,:], label = "Sample 3")
  plt.plot(xaxis,loss[3,:], label = "Sample 4")
  plt.plot(xaxis,loss[4,:], label = "Sample 5")
  plt.plot(xaxis,loss[5,:], label = "Sample 6")
  plt.plot(xaxis,loss[6,:], label = "Sample 7")
  plt.plot(xaxis,loss[7,:], label = "Sample 8")
  plt.legend()
  plt.show()

def backprop2(a):
  w1 = np.array([[random.random(),random.random()],[random.random(),random.random()],[random.random(),random.random()],[random.random(),random.random()]])
  w2 = np.array([[random.random()],[random.random()],[random.random()]])
  data = np.array([[1,-1,-1,0],[1,0,1,0],[0,0,-1,0],[0,-1,1,0],[1,1,1,1],[0,1,0,1],[0,0,1,1],[-1,0,1,1]])
  conv = 0
  max = 100
  loss = np.zeros((8,max))
  while conv < max:
    index = 0
    for d in data:
      if w1[1,0]*d[0] + w1[2,0]*d[1] + w1[3,0]*d[2] + w1[0,0] > 0:
        g1 = w1[1,0]*d[0] + w1[2,0]*d[1] + w1[3,0]*d[2] + w1[0,0]
      else:
        g1 = 0
      if w1[1,1]*d[0] + w1[2,1]*d[1] + w1[3,1]*d[2] + w1[0,1] > 0:
        g2 = w1[1,1]*d[0] + w1[2,1]*d[1] + w1[3,1]*d[2] + w1[0,1]
      else:
        g2 = 0
      if w2[1,0]*g1 + w2[2,0]*g2 + w2[0,0] > 0:
        g3 = w2[1,0]*g1 + w2[2,0]*g2 + w2[0,0]
      else:
        g3 = 0
      l = .5*(d[3]-g3)**2
      loss[index,conv] = l
      index = index + 1
      if w2[1,0]*g1 + w2[2,0]*g2 + w2[0,0]>0:
        dg3 = 1
      else:
        dg3 = 0
      dw201 = -(d[3]-g3) * dg3
      dw221 = -(d[3]-g3) * dg3 * g2
      dw211 = -(d[3]-g3) * dg3 * g1
      if w1[1,0]*d[0] + w1[2,0]*d[1] + w1[3,0]*d[2] + w1[0,0]>0:
        dg1 = 1
      else:
        dg1 = 0
      if w1[1,1]*d[0] + w1[2,1]*d[1] + w1[3,1]*d[2] + w1[0,1]>0:
        dg2 = 1
      else:
        dg2 = 0
      dw101 = -(d[3]-g3) * dg3 * dg1
      dw102 = -(d[3]-g3) * dg3 * dg2
      dw111 = -(d[3]-g3) * dg3 * dg1 *d[0]
      dw121 = -(d[3]-g3) * dg3 * dg1 *d[1]
      dw131 = -(d[3]-g3) * dg3 * dg1 *d[2]
      dw112 = -(d[3]-g3) * dg3 * dg2 *d[0]
      dw122 = -(d[3]-g3) * dg3 * dg2 *d[1]
      dw132 = -(d[3]-g3) * dg3 * dg2 *d[2]
      grad1 = np.array([[dw101,dw102],[dw111,dw112],[dw121,dw122],[dw131,dw132]])
      grad2 = np.array([[dw201],[dw211],[dw221]])
      w1 = w1 - a*grad1
      w2 = w2 - a*grad2
    conv = conv + 1
  plt.figure()
  xaxis = list(range(1,max + 1))
  plt.plot(xaxis,loss[0,:], label = "Sample 1")
  plt.plot(xaxis,loss[1,:], label = "Sample 2")
  plt.plot(xaxis,loss[2,:], label = "Sample 3")
  plt.plot(xaxis,loss[3,:], label = "Sample 4")
  plt.plot(xaxis,loss[4,:], label = "Sample 5")
  plt.plot(xaxis,loss[5,:], label = "Sample 6")
  plt.plot(xaxis,loss[6,:], label = "Sample 7")
  plt.plot(xaxis,loss[7,:], label = "Sample 8")
  plt.legend()
  plt.show()

def backprop3(a):
  w1 = np.array([[random.random(),random.random()],[random.random(),random.random()],[random.random(),random.random()],[random.random(),random.random()]])
  w2 = np.array([[random.random()],[random.random()],[random.random()]])
  data = np.array([[1,-1,-1,0],[1,0,1,0],[0,0,-1,0],[0,-1,1,0],[1,1,1,1],[0,1,0,1],[0,0,1,1],[-1,0,1,1]])
  conv = 0
  max = 100
  loss = np.zeros((8,max))
  while conv < max:
    index = 0
    for d in data:
      g1 = np.log(1 + math.exp(w1[1,0]*d[0] + w1[2,0]*d[1] + w1[3,0]*d[2] + w1[0,0]))
      g2 = np.log(1 + math.exp(w1[1,1]*d[0] + w1[2,1]*d[1] + w1[3,1]*d[2] + w1[0,1]))
      g3 = np.log(1 + math.exp(w2[1,0]*g1 + w2[2,0]*g2 + w2[0,0]))
      l = .5*(d[3]-g3)**2
      loss[index,conv] = l
      index = index + 1
      #derivative of softplus found at: https://medium.com/@abhinavr8/activation-functions-neural-networks-66220238e1ff#:~:text=Softplus%20function%3A%20f(x),also%20called%20the%20logistic%20function.
      dg3 = math.exp(w2[1,0]*g1 + w2[2,0]*g2 + w2[0,0])/((1 + math.exp(w2[1,0]*g1 + w2[2,0]*g2 + w2[0,0])))
      dw201 = -(d[3]-g3) * dg3
      dw221 = -(d[3]-g3) * dg3 * g2
      dw211 = -(d[3]-g3) * dg3 * g1
      dg1 = math.exp(w1[1,1]*d[0] + w1[2,1]*d[1] + w1[3,1]*d[2] + w1[0,1])/((1 + math.exp(w1[1,1]*d[0] + w1[2,1]*d[1] + w1[3,1]*d[2] + w1[0,1])))
      dg2 = math.exp(w1[1,1]*d[0] + w1[2,1]*d[1] + w1[3,1]*d[2] + w1[0,1])/((1 + math.exp(w1[1,1]*d[0] + w1[2,1]*d[1] + w1[3,1]*d[2] + w1[0,1])))
      dw101 = -(d[3]-g3) * dg3 * dg1
      dw102 = -(d[3]-g3) * dg3 * dg2
      dw111 = -(d[3]-g3) * dg3 * dg1 *d[0]
      dw121 = -(d[3]-g3) * dg3 * dg1 *d[1]
      dw131 = -(d[3]-g3) * dg3 * dg1 *d[2]
      dw112 = -(d[3]-g3) * dg3 * dg2 *d[0]
      dw122 = -(d[3]-g3) * dg3 * dg2 *d[1]
      dw132 = -(d[3]-g3) * dg3 * dg2 *d[2]
      grad1 = np.array([[dw101,dw102],[dw111,dw112],[dw121,dw122],[dw131,dw132]])
      grad2 = np.array([[dw201],[dw211],[dw221]])
      w1 = w1 - a*grad1
      w2 = w2 - a*grad2
    conv = conv + 1
  plt.figure()
  xaxis = list(range(1,max + 1))
  plt.plot(xaxis,loss[0,:], label = "Sample 1")
  plt.plot(xaxis,loss[1,:], label = "Sample 2")
  plt.plot(xaxis,loss[2,:], label = "Sample 3")
  plt.plot(xaxis,loss[3,:], label = "Sample 4")
  plt.plot(xaxis,loss[4,:], label = "Sample 5")
  plt.plot(xaxis,loss[5,:], label = "Sample 6")
  plt.plot(xaxis,loss[6,:], label = "Sample 7")
  plt.plot(xaxis,loss[7,:], label = "Sample 8")
  plt.legend()
  plt.show()



trainBayes(0)
trainBayes(1)
perceptron1(1)
perceptron2(1)
linearClass()
backprop(1)
print("tanh")
backprop1(.5)
print("relu")
backprop2(.005)
print("softplus")
backprop3(.005)
