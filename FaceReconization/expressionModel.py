"""
inputs(48*48*1) ->
conv(24*24*64) -> conv(12*12*128) -> conv(6*6*256) ->
Dropout -> fc(4096) -> Dropout -> fc(1024) ->
outputs(7)
"""

import torch.nn as nn
import torch

def gaussian_weights_init(m):
    
    '''
    Generate numbers with gaussian distribution
    '''
    
    classname = m.__class__.__name__
    # 字符串查找find，找不到返回-1，不等-1即字符串中含有该字符
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.04)

class ExpressionCNN(nn.Module):
    
    '''
    Model for expression recognition
    '''
    
    def __init__(self):
        
        super(ExpressionCNN, self).__init__()
        
        # layer1(conv + relu + pool)
        # input:(batch_size, 1, 48, 48), output(batch_size, 64, 24, 24)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.BatchNorm2d(num_features=64),
            nn.RReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # layer2(conv + relu + pool)
        # input:(batch_size, 64, 24, 24), output(batch_size, 128, 12, 12)
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(num_features=128),
            nn.RReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # layer3(conv + relu + pool)
        # input: (batch_size, 128, 12, 12), output: (batch_size, 256, 6, 6)
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(num_features=256),
            nn.RReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Initialization
        self.conv1.apply(gaussian_weights_init)
        self.conv2.apply(gaussian_weights_init)
        self.conv3.apply(gaussian_weights_init)
        
        self.fullc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(256*6*6, 4096),
            nn.RReLU(inplace=True),

            nn.Dropout(p=0.5),
            nn.Linear(4096, 1024),
            nn.RReLU(inplace=True),

            nn.Linear(1024, 256),
            nn.RReLU(inplace=True),
            nn.Linear(256, 7)
        )
        
    def forward(self, x):
        
        '''
        Forward propagation
        '''
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.shape[0], -1)  # 数据扁平化
        y = self.fullc(x)

        return y
    
if __name__ == "__main__":
    
    test_tensor = None