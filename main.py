# coding=utf-8
# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import requests
from bs4 import BeautifulSoup
from urllib.parse import urlencode
from urllib.request import urlopen, Request
import json
import ssl
import numpy as np
import torch
import torch as nn
import torch.optim as  optim
from torchvision import datasets, transforms



def python_start(name):
    N, D_in, H, D_out = 64, 1000, 100, 10
    x = torch.randn(N, D_in)
    y = torch.randn(N, D_out)

    #w1 = torch.randn(D_in, H, requires_grad = True)
    #w2 = torch.randn(H, D_out,requires_grad = True)

    model = torch.nn.Sequential(
        torch.nn.Linear(D_in,H),
        torch.nn.ReLU(),
        torch.nn.Linear(H,D_out),
    )

    class TwoLayerNet(torch.nn.Module):
            def __init__(self,D_in,H,D_out):
                super(TwoLayerNet,self).__init__() 
                self.linear1 = torch.nn.Linear(D_in,H,bias = False)
                self.linear2 = torch.nn.Linear(H,D_out,bias = False)
            def forward(self,x):
                y_pred = self.linear2(self.linear1(x).clamp(min=0))
                return y_pred
    model = TwoLayerNet(D_in,H,D_out)
            

    #torch.nn.init.normal(model[0].weight)
    #torch.nn.init.normal(model[2].weight)

    loss_fn = torch.nn.MSELoss(reduction='sum')
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for it in range(500):
        y_pred = model(x)
        #h = x.mm(w1)
        #h_relu = h.clamp(min = 0) #np.maximum(h,0)
        #y_pred = h_relu.mm(w2)

        #loss = np.square(y_pred - y).sum()
        #loss = (y_pred - y).pow(2).sum()
        loss = loss_fn(y_pred,y)
        print(it, loss.item())

        #model.zero_grad()
       # grad_y_pred = 2.0 * (y_pred - y)
        #grad_w2 = h_relu.t().mm(grad_y_pred)
        #grad_h_relu = grad_y_pred.mm(w2.t())
        #grad_h = grad_h_relu.clone()
        #grad_h[h<0] = 0
        #grad_w1 = x.t().mm(grad_h)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #with torch.no_grad():
            
            #w1 -= learning_rate * w1.grad
            #w2 -= learning_rate * w2.grad
            #w1.grad.zero_()
            #w2.grad.zero_()

            #for param in model.parameters():
            #    param -= learning_rate * param.grad


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    python_start('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
