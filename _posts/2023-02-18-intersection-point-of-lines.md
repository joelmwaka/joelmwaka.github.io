---
layout: post
title:  "Intersection Point of Many Lines"
date:   2023-02-18 10:00:00 +0000
categories: Python PyTorch
---

In this post, we will find an optimal intersection point of a set of lines that do not intersect at a point. We will phrase this problem as a least squares minimization problem. We can also give weights to each line based on our confidence on that line. 

Given a set of lines as shown below, how to find a point that represents the intersection of these lines?
<img src="https://user-images.githubusercontent.com/36071915/219940779-48913e67-7619-44ad-b319-706ac9c71ff8.png" width="500" >

We define the intersection point as the point that minimizes the distance to all lines. This distance is the perpindicular line between point and any line as shown below.

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c2/Distance_from_a_point_to_a_line.svg/2560px-Distance_from_a_point_to_a_line.svg.png" width="500" >

The distance for any point can be calculated as the rejection of point **p** from line point **a**. See [here](https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#:~:text=In%20Euclidean%20geometry%2C%20the%20'distance,nearest%20point%20on%20the%20line.) for details

Then our task is simply to minimize this distance for a single point and all lines. If we have any weights for each line, simply multiply by the weight to give less importance to specific lines. The distance to any line from test point is given as:

![dist](https://latex.codecogs.com/svg.image?\inline%20\large%20d_i%20=%20\left%20\|%20(\vec{p}%20-%20\vec{a}_i)%20-%20((\vec{p}%20-%20\vec{a_i})%20\cdot%20\hat{n}_i)\hat{n}_i%20\right%20\|)

**Demo**

For this demo, I will use the autograd library of Pytorch. If your lines are in 3D, simply add a third dimension to all relevant variables.

```python
import torch

#points on that define the start of the line 
p = torch.tensor([[1., 1], [-1, 2], [1.3, 1], [2, 3]])
#vectors that define each line
n = torch.tensor([[1., 1], [ 2, 0], [-1., 1], [1, 3]])

#initial guess of the intersection point. Change this to any other point if you have a good initial guess
x = torch.tensor([[0.,0]])

#weights for each line.
w = torch.tensor([1, 1., 1., 1.]).view(-1, 1)
nhat = n / torch.linalg.norm(n, dim = 1, keepdim = True) #convert to unit vectors
x.requires_grad = True

optim = torch.optim.Adam([x], lr= 0.1) #this likely will improve stability. But if you want to manually update weights, just uncomment the lines below.
for ep in range(100):
    #x.grad = None
    optim.zero_grad() #reset gradients
    x_to_p = x - p # (a - p) vector in the equation above
    d = torch.sum(x_to_p * nhat, dim = 1, keepdim = True) * nhat - x_to_p #this is calculating the distance over all lines as a batch.
    loss = torch.sum(w * d**2, dim = 1) #Notice the weights multiplied here
    loss = torch.mean(loss)  #total loss that needs to be minimized
    loss.backward()
    
    #lr = 0.1
    #x.data -= lr * x.grad
    optim.step() #remove this line if manually updating
print(loss, x)
```

Running the above code produces the first figure at the top. If you also want to see the figure, here is the code:

```python
import matplotlib.pyplot as plt
plt.style.use('ggplot')
#%matplotlib inline #use this if in jupyter notebook.

def show(p, n, s = None, extend = 4):
    #plt.scatter(p[:,0], p[:,1])
    
    left = p + extend * -n
    right = p + extend * n
    
    for l,r in zip(left, right):
        plt.plot((l[0], r[0]), (l[1], r[1]), c = (0,0,1), ls = '--')
    
    if s is not None:
        print(s)
        plt.scatter([s[0]], [s[1]], c = [(1,0,0)], label = 'Minimized intersection point')
        plt.legend()

#call this after the training is finished.
show(p.numpy(), nhat.numpy(), x.detach().squeeze().numpy())
```
