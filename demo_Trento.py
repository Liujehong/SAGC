import sys
import time
import warnings
import os
from matplotlib.colors import ListedColormap
# from fast_pytorch_kmeans import KMeans
from Preprocessing import Processor, CLASS_MAP_COLOR_16, CLASS_MAP_COLOR_B, CLASS_MAP_COLOR_8
from sklearn.cluster import KMeans,k_means
warnings.filterwarnings("ignore")
from mytuils import record_reuslt, _get_anchor_, train, _generate_anchors_align_
sys.path.append('./')
sys.path.append('/root/python_codes/')
import os
import seaborn as sns
import numpy as np
import torch
import scipy.io as sio
import torch.nn.functional as F
import argparse
from utils import yaml_config_hook, metric, initialization_utils, load_multimodal_data2
from Model_zoo import netw, CustomDataset, load_multimodal_data, Encoder, Decoder, ldrEncoder
from fast_pytorch_kmeans import KMeans as K
# from utils.superpixel_anchors import generate_anchors
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
from datetime import datetime
from tqdm import tqdm
# from send_email import myemail
from hysime import *
# from torch.utils.tensorboard import SummaryWriter
from vca import vca
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")
print(f'using {DEVICE}')

import matplotlib.pyplot as plt



def cosine_similarity(a, b):
    """计算两个向量的余弦相似度"""
    return torch.dot(F.normalize(a, dim=0), F.normalize(b, dim=0)).item()

def greedy_anchor_selection(candidates, num_anchor):
    N = candidates.size(0)
    selected_indices = []

    init_idx = torch.randint(0, N, (1,)).item()
    selected_indices.append(init_idx)
    remaining_indices = set(range(N))
    remaining_indices.remove(init_idx)

    for _ in range(1, num_anchor):
        min_total_sim = float('inf')
        best_idx = -1

        for idx in remaining_indices:
            candidate = candidates[idx]
            total_sim = 0.0
            for sel_idx in selected_indices:
                sel = candidates[sel_idx]
                sim = cosine_similarity(candidate, sel)
                total_sim += sim

            if total_sim < min_total_sim:
                min_total_sim = total_sim
                best_idx = idx

        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)
    return candidates[torch.tensor(selected_indices, device=candidates.device)]


def select_anchor_from_vca_kmeans_union(centroids_modality1, centroids_vca, num_anchor, device='cpu'):
    """
    从 VCA 与 K-means 的端元联合集合中，选择 num_anchor 个最不相关锚点
    输入：
        centroids_modality1: numpy array [R, C]，K-means 聚类中心
        centroids_vca: torch.Tensor [R, C]，VCA 端元
        num_anchor: int，要选择的最终锚点数量
        device: str，'cuda' 或 'cpu'
    返回：
        torch.Tensor [num_anchor, C]，最终锚点
    """
    centroids_kmeans = torch.from_numpy(centroids_modality1).float().to(device)
    centroids_vca = centroids_vca.to(device)
    candidates = torch.cat([centroids_vca, centroids_kmeans], dim=0)  # [2R, C]
    selected_anchors = greedy_anchor_selection(candidates, num_anchor)

    return selected_anchors  # [num_anchor, C]



#
class NonZeroClipper(object):
    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            w.clamp_(1e-6,1.0)



class SumToOneLoss(torch.nn.Module):
    def __init__(self):
        super(SumToOneLoss, self).__init__()
        self.register_buffer('one', torch.tensor(1, dtype=torch.float))
        self.loss = torch.nn.L1Loss(size_average=False)

    def get_target_tensor(self, input):
        target_tensor = self.one
        return target_tensor.expand_as(input)

    def __call__(self, input, gamma_reg=0.1):
        input = torch.sum(input, 1)
        target_tensor = self.get_target_tensor(input)
        loss = self.loss(input, target_tensor.to(input.device))
        return gamma_reg*loss



#---------------------------------------------------#
def predict(anchor_graph, n_cluster):
    Kmeans = K(n_clusters=n_cluster,mode='cosine')
    labels = Kmeans.fit_predict(anchor_graph.T)
    return labels.cpu().numpy()



#---------------------------------------------------#

if __name__ == "__main__":
    start_time_point = datetime.now()
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='DSAMVSC Model Training')
    parser.add_argument('--n_anchor', type=int, default=8, help='Number of anchors per class')
    parser.add_argument('--nb_comps', type=int, default=12, help='Number of components')
    parser.add_argument('--gamma', type=float, default=0.3,help='Gamma value for CoFM')
    parser.add_argument('--beta', type=float, default=0.7,help='Gamma value for rectified flow')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate for the optimizer')
    # config = yaml_config_hook("config.yaml")
    # for k, v in config.items():
    #     parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    print("本次超参:")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    
    #===========Houston============================#
    # the_dataset =  "Houston"
    # dataset_root =  './datasets/Houston/Houston'
    # model_path =   "save/Houston"
    # n_anchor = args.n_anchor   # 锚点与类别数之间的倍数关系
    # nb_comps = args.nb_comps
    # gamma = args.gamma
    # beta = args.beta
    # LEARNING_RATE = args.learning_rate 
    # epoch = 100
    # seed =  42

    # ===========MUUFL============================#
    # the_dataset =  "MUUFL"
    # dataset_root =  './datasets/MUUFL/MUUFLGfport'
    # model_path =   "save/MUUFL-Gulfport"
    # model_path =   "save/Trento"
    # n_anchor = args.n_anchor
    # nb_comps = args.nb_comps
    # gamma = args.gamma
    # beta = args.beta
    # LEARNING_RATE = args.learning_rate
    # epoch = 200
    # seed =  111

    # # #===========Trento============================#
    the_dataset = "Trento"
    dataset_root =  './datasets/Trento/'
    model_path =   "save/Trento"
    n_anchor = args.n_anchor
    nb_comps = args.nb_comps
    gamma = args.gamma
    beta = args.beta
    LEARNING_RATE = args.learning_rate
    epoch = 200
    seed = 42  
    
    is_labeled_pixel = False
    load_model =  False
    save_model =  False
    verbose =  False
    email =  False
    save_fig =  True
    root = dataset_root

    initialization_utils.set_global_random_seed(seed=seed)
    # prepare data
    if the_dataset == "Houston":
        im_1, im_2 = 'HSI', 'Lidar'
        gt_ = 'GT'
        img_path = (root + im_1 + '.mat', root + im_2 + '.mat')
        data_name = (im_1, im_2)
    elif the_dataset == "Trento":
        im_1, im_2 = 'Trento-HSI', 'Trento-Lidar'
        gt_ = 'Trento-GT'
        img_path = (root + im_1 + '.mat', root + im_2 + '.mat')
        # img_path = (root + im_2 + '.mat', )
        data_name = (im_1, im_2)
    elif the_dataset == "MUUFL":
        im_1, im_2 = 'HSI', 'LiDAR_data_first_return'  # 'LiDAR_data_first_last_return'
        gt_ = 'GT'
        img_path = (root + im_1 + '.mat', root + im_2 + '.mat')
        data_name = (im_1, im_2)
    else:
        raise NotImplementedError
    gt_path = root + gt_ + '.mat'

    for i, p_ in enumerate(img_path):
        print(f'modality #{i + 1}: {p_}')
    
    #x_pca is a list ,0 denote hsi, 1 denote LiDAR
    x_pca, gt, x_nopca,pca = load_multimodal_data(gt_path, *img_path, is_labeled=is_labeled_pixel, nb_comps=nb_comps,seed=seed, device=DEVICE)
    #run hysime
    anchor_num = runhysime(x_nopca[0].reshape(-1,x_nopca[0].shape[-1]))
    print("anchor_num:",anchor_num)
    y = gt.reshape((-1,))
    dataset = CustomDataset(x_pca)
    BATCH_SIZE = 1
    dataloader_train = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    if is_labeled_pixel:
        class_num = len(np.unique(y))
        y_labeled = np.copy(y)
    else:
        indx_labeled = np.nonzero(y)[0]
        y_labeled = y[indx_labeled]
        class_num = len(np.unique(y)) - 1
    data_size = x_pca[0].shape[0] *x_pca[0].shape[1]
    spatial_size = gt.shape
    print('# classes:', class_num)
    anchor_num = anchor_num*n_anchor
    start_time = time.time()
    record = record_reuslt()
    decoder = Decoder(anchor_num,nb_comps + x_pca[1].shape[2]) #第二个nb_comps表示伪波段数 即PCA后的伪端元维度
    model = netw(nb_comps,x_pca[1].shape[2],anchor_num, DEVICE, decoder).to(DEVICE)
    path = 'models/state_dict_model.pth'
    if os.path.exists(path) and load_model:
        checkpoint = torch.load(path)
        model_dict = model.state_dict()
        filtered_dict = {k: v for k, v in checkpoint.items() if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict)
    data = np.concatenate((x_pca[0].reshape(-1,x_pca[0].shape[-1]),x_pca[1].reshape(-1,x_pca[1].shape[-1])),axis=1)  
    data = torch.from_numpy(data.T).float().clone().detach().to(DEVICE)
    _,indices,_ = vca(x_nopca[0].reshape(-1,x_nopca[0].shape[-1]),anchor_num)
    modality1 = x_pca[0].reshape(-1, x_pca[0].shape[-1])
    centorid_from_vca = torch.from_numpy(modality1[indices]).float().clone().detach().to(DEVICE)
    Kmeans = K(n_clusters=anchor_num, mode='cosine',minibatch=10000)
    cluster_labels = Kmeans.fit_predict(
        torch.from_numpy(modality1).float().to(DEVICE)
    )
    centroids_K_means = Kmeans.centroids.cpu().numpy()
    centroids_modality1 = select_anchor_from_vca_kmeans_union(centroids_K_means,centorid_from_vca,anchor_num)
    modality2 = x_pca[1].reshape(-1, x_pca[1].shape[-1])
    centroids_modality2 = []
    for i in range(anchor_num):
        indices = np.where(cluster_labels.cpu().numpy() == i)[0]
        cluster_feat_modality2 = modality2[indices]
        centroid_m2 = cluster_feat_modality2.mean(axis=0)
        centroids_modality2.append(centroid_m2)
        
    centroids_modality2 = np.stack(centroids_modality2)
    anchor_centers = np.concatenate((centroids_modality1, centroids_modality2), axis=1)
    cluster_centers = torch.from_numpy(anchor_centers).float().to(DEVICE).T
    E_init = cluster_centers.unsqueeze(2).unsqueeze(3).float().to(DEVICE)
    #装载锚点
    model_dict = model.state_dict()
    model_dict['decoder.conv1.weight'] =  E_init
    model.load_state_dict(model_dict)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    ema_model = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.99))
    loss_func1 = torch.nn.MSELoss(size_average=True,reduce=True,reduction='mean')
    criterionSumToOne = SumToOneLoss()
    

    model.train()
    sc = False
    if verbose:
        progress_bar = tqdm(total=epoch, desc="Training")
    for e in range(epoch):
        for step, (x0, x1) in enumerate(dataloader_train):
            x0 = x0.clone().detach().to(DEVICE)
            x1 = x1.clone().detach().to(DEVICE)

            x0 = x0.permute(0, 3, 1, 2)
            x1 = x1.permute(0, 3, 1, 2)
            tl = [x0, x1]
            
            optimizer.zero_grad()
            abu_hsi,x_hat_hsi = model(tl)
            loss1 = loss_func1(data,x_hat_hsi.reshape(data.shape))
            loss2 = criterionSumToOne(abu_hsi)
            loss3 = torch.norm(abu_hsi, p=1) 
            loss = loss1 + loss2 + beta*loss3
            loss.backward()
            optimizer.step()
        if verbose:
            progress_bar.update(1)
    
        abu_hsi = abu_hsi.detach().reshape(anchor_num,-1)
        with torch.no_grad():
            y_pred = predict(abu_hsi, class_num)

        if not is_labeled_pixel:
            y_pred_labeled = y_pred[indx_labeled]
            y_pred_2D = y_pred.reshape(gt.shape)
        else:
            y_pred_labeled = y_pred
        y_pred_2D,acc, kappa, nmi, ari, pur, bcubed_F, ca= metric.cluster_accuracy(y_labeled, y_pred_labeled,y_pred_2D, return_aligned=True)
        
        if verbose:
            print('OA = {:.8f} Kappa = {:.8f} NMI = {:.8f} ARI = {:.8f} Purity = {:.8f}  BCubed F = {:.8f}'.format(acc, kappa,
                                                                                                                nmi, ari,
                                                                                                                pur,bcubed_F))

        new_results = {
            'OA':float(round(acc, 4)),
            'Kappa':float(round(kappa, 4)),
            'NMI':float(round(nmi, 4)),
            'ARI':float(round(ari, 4)),
            'Purity':float(round(pur, 4)),
            'BCubed F':float(round(bcubed_F, 4)),
            
        }                                                                                               
            
        for i, ca_ in enumerate(ca):
            new_results[f'class #{i} ACC'] = float(round(ca_, 4))
            # print(f'class #{i} ACC: {ca_:.4f}')
        new_results['Z'] = abu_hsi.cpu().numpy()
        new_results['y_pred_2D'] = y_pred_2D
        running_time = time.time() - start_time
        new_results['running time'] = f'{float(round(running_time, 3))} s'
        new_results['anchor_num'] = n_anchor
        new_results['nb_comps'] = nb_comps
        new_results['epoch'] = e
        new_results['gamma'] = gamma
        new_results['beta'] = beta
        new_results['learning rate'] = LEARNING_RATE

        

        record.add_history(new_results)
    if verbose:    
        progress_bar.close()
    
        
        
        
    max_oa_metrics = record.best_result('OA')
    
    
    
    
   


    # ## ====================================
    # # show classification map
    # # ======================================
    if save_fig:
        p = Processor()
        save_name = f'Figures/classmap-{the_dataset}-{max_oa_metrics["OA"] * 10000:.0f}-{max_oa_metrics["learning rate"]}.pdf'
        gt_color = p.colorize_map(max_oa_metrics['y_pred_2D'], colors=CLASS_MAP_COLOR_16, background_color=None)
        sio.savemat(f'q_{the_dataset}_OA_{max_oa_metrics["OA"] * 10000:.0f}.mat', {'y_pred':max_oa_metrics['y_pred_2D']})    
        fig, ax = plt.subplots()
        ax.imshow(gt_color)
        plt.axis('off')
        plt.tight_layout()
        print(save_name)
        fig.savefig(save_name, format='pdf', bbox_inches='tight', pad_inches=0, dpi=300)
        # plt.show()
        plt.close()


    print('------------------------------------------------------------------------------')
    for key, value in max_oa_metrics.items():
        if key != 'Z' and key != 'y_pred_2D':
            print(f'{key} : {value}')
    print('------------------------------------------------------------------------------')
    if save_model:
        torch.save(model.state_dict(), 'models/state_dict_model.pth')

    end_time_point = datetime.now()
    
