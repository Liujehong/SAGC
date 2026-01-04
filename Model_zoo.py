
import numpy as np
import torch
from sklearn.preprocessing import minmax_scale

from torch.utils.data import Dataset
from Preprocessing import Processor
import torch.nn.functional as F
from sklearn.decomposition import PCA
import torch.nn.functional as F


class ldrEncoder(torch.nn.Module):
    def __init__(self, channels,P):
        super(ldrEncoder, self).__init__()
     
        self.conv1 = torch.nn.Conv2d(channels, 16, kernel_size=3, stride=1, padding='same')  
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding='same') 
        self.conv3 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding='same') 
        self.conv4 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding='same') 
        self.conv5 = torch.nn.Conv2d(64, P, kernel_size=3, stride=1, padding='same')
        
        
        self.bn1 = torch.nn.BatchNorm2d(16) 
        self.bn2 = torch.nn.BatchNorm2d(32) 
        self.bn3 = torch.nn.BatchNorm2d(64) 
        self.bn4 = torch.nn.BatchNorm2d(64) 
        self.bn5 = torch.nn.BatchNorm2d(P) 

        
        self.rl = torch.nn.ReLU()


    def forward(self, x):
        
        ldr = self.conv1(x)
        x = self.bn1(ldr)
        x = self.rl(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.rl(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.rl(x)
        
        
        x = self.conv4(x) 
        x = self.bn4(x)
        x = self.rl(x)
        
        x = self.conv5(x)
        x = self.bn5(x)

        #x as a
        return x,ldr
    
class Encoder(torch.nn.Module):
    def __init__(self, channels,P):
        super(Encoder, self).__init__()
     
        self.conv1 = torch.nn.Conv2d(channels, 16, kernel_size=3, stride=1, padding='same')  
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding='same') 
        self.conv3 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding='same') 
        self.conv4 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding='same') 
        self.conv5 = torch.nn.Conv2d(64, P, kernel_size=3, stride=1, padding='same')
        
        
        self.bn1 = torch.nn.BatchNorm2d(16) 
        self.bn2 = torch.nn.BatchNorm2d(32) 
        self.bn3 = torch.nn.BatchNorm2d(64) 
        self.bn4 = torch.nn.BatchNorm2d(64) 
        self.bn5 = torch.nn.BatchNorm2d(P) 

        
        self.rl = torch.nn.ReLU()
#==========================================================================================#       
      


    def forward(self, x):
        
        hsi = self.conv1(x)
        x = self.bn1(hsi)
        x = self.rl(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.rl(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.rl(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.rl(x)
        
        x = self.conv5(x)
        x = self.bn5(x)

        #x as abu
        return x,hsi

class Decoder(torch.nn.Module):
    def __init__(self, P,L):
        super(Decoder, self).__init__()
        self.conv1 = torch.nn.Conv2d(P, L, kernel_size=1, stride=1, bias=False)
        self.rl = torch.nn.ReLU()

        

    def forward(self, x):
        x = self.conv1(x)
        x = self.rl(x)
        return x





class netw(torch.nn.Module):
    def __init__(self, hsiinchannels,ldrinchannels,P,device,decoder):
        super(netw, self).__init__()
        self.device = device
        self.decoder = decoder
        self.headlayer_hsi  = torch.nn.Conv2d(hsiinchannels, 16, kernel_size=3, stride=1, padding='same')
        self.encoder_hsi = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding='same'),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding='same'),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding='same'),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, P, kernel_size=3, stride=1, padding='same'),
            torch.nn.BatchNorm2d(P),
        )
        self.headlayer_ldr=  torch.nn.Conv2d(ldrinchannels, 16, kernel_size=3, stride=1, padding='same')
        self.encoder_ldr = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding='same'),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding='same'),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding='same'),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, P, kernel_size=3, stride=1, padding='same'),
            torch.nn.BatchNorm2d(P),
        )
        self.softmax = torch.nn.Softmax(dim=1)
        self.initialize_weights()
        
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                # torch.nn.init.xavier_normal_(m.weight.data)
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
                    #torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.ConvTranspose2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, torch.nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()
        #完成对decoder的VCA初始化


    def forward(self, x_ls):
        hsi = self.headlayer_hsi(x_ls[0])
        ldr = self.headlayer_ldr(x_ls[1])
        abu_hsi = self.encoder_hsi(hsi)
        abu_ldr = self.encoder_ldr(ldr)
        abu_hsi = self.softmax(abu_hsi+abu_ldr)
        x_hsi_hat = self.decoder(abu_hsi)
        return abu_hsi,x_hsi_hat

    
    def fm(self, v,targetv):
        return F.mse_loss(v, targetv,reduction="mean")
    

    
    


class CustomDataset(Dataset):
    def __init__(self, data_list):
        """
        初始化自定义数据集
        data_list: 包含两个元素的列表，第一个元素形状为（71500，64），
                   第二个元素形状为（71500，2）
        """
        self.data1 = data_list[0].reshape((1, data_list[0].shape[0], data_list[0].shape[1], data_list[0].shape[2]))
        self.data2 = data_list[1].reshape((1, data_list[1].shape[0], data_list[1].shape[1], data_list[1].shape[2]))
        self.length = self.data1.shape[0]
  
    def __len__(self):
        """
        返回数据集的总长度
        """
        return self.length

    def __getitem__(self, idx):
        """
        根据索引获取一个样本
        """
        # 取出第idx个样本
        sample1 = self.data1[idx]
        sample2 = self.data2[idx]
        
        # 返回一个字典或者元组，包含两个样本
        return (torch.tensor(sample1, dtype=torch.float32),
                torch.tensor(sample2, dtype=torch.float32))
        


def load_multimodal_data(gt_path, *src_path, is_labeled=True, nb_comps, device,seed):
    p = Processor()
    n_modality = len(src_path)
    modality_list = []
    in_channels = []
    x_nopca_list= []
    
    for i in range(n_modality):
        img, gt = p.prepare_data(src_path[i], gt_path)
        print(f'modality %s shape: %s' % (i, img.shape))
        img_scale = minmax_scale(img.reshape(-1, img.shape[-1]).astype(np.float64), feature_range=(0, 1)).reshape(img.shape)
        # img, gt = img[:, :100, :], gt[:, :100 ]
        n_row, n_column, n_band = img.shape
        x_nopca_list.append(img)
        modality_list.append(img_scale)
        if i == 1:
            nonorm_img = img
        
    n_row, n_column, n_band = modality_list[0].shape    
    pca = PCA(n_components=nb_comps,random_state=seed)
    modality_list[0] = pca.fit_transform(modality_list[0].reshape(n_row*n_column, n_band)).reshape((n_row, n_column, nb_comps))
    print('pca shape: %s, percentage: %s' % (modality_list[0].shape, np.sum(pca.explained_variance_ratio_)))    
    return modality_list, gt, x_nopca_list,nonorm_img