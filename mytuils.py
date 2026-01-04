import torch
from torchvision import transforms


class record_reuslt():
    def __init__(self):
        self.results_history = []
    
    def add_history(self, result_history: dict):
        self.results_history.append(result_history)
        
    def best_result(self, Metrics: str):
        the_best = max(self.results_history, key=lambda x: x[Metrics])
        return the_best
    
class set_grad():
    def __init__(self) -> None:
        self.set_contents = {}
    # @staticmethod
    def update(self, *args):
        for param in args:
            param.requires_grad = True

            
    # @staticmethod
    def fix(self, *args):
        for param in args:
            param.requires_grad = False


def train(model, x, optimizer, sub_iter, *update_param):
            """
            :param:
            model: The network
            x : The data
            optimizer: The network's optimizer
            sub_iter: Number of iterations
            *update_param: want update params
            
            """
            for param in model.parameters():
                param.requires_grad = False   
            for _ in range(sub_iter):
                for param in update_param:
                    param.requires_grad=True
                optimizer.zero_grad()
                loss = model(x)
                loss.backward()
                optimizer.step()
            # print(loss.item())

def generate_anchors_align(self, x_list):
        """
        generate anchors by concatenating two view along feature dim
        :param x_list:
        :return:
        """
        x = torch.concat(x_list, dim=0)
        self.n_anchor = self.n_cluster * self.anchor4_kmeans
        k_means = KMeans(n_clusters=self.n_anchor, verbose=False, minibatch=10000)
        _ = k_means.fit_predict(x.t())
        anchors = []
        start = 0
        for i in range(len(x_list)):
            end = start + x_list[i].shape[0]
            anchors.append(k_means.centroids.t()[start:end, :])
            start = start + x_list[i].shape[0]
        return anchors
    
def normalization(x):
            frobenius_norm = torch.norm(x, p='fro')  # 计算 Frobenius 范数
            x = x / frobenius_norm  # 归一化
            return x
        
class GadGet():
    def __init__(self):
        pass
    
    
    def to_reshape(self, shape, *x):
        pass
            
            
    
    
def apply_augmentation(dataset, data_augmentation):
    augmented_data = []
    for img in dataset:
        img = transforms.ToPILImage()(img)             # 转换为 PIL 图像
        augmented_img = data_augmentation(img)         # 应用数据增强
        augmented_data.append(augmented_img)
    return torch.stack(augmented_data)                 # 将增强的图像转换为 PyTorch 张量


def _get_anchor_(anchor_num, x, device):
    kmeans = KMeans(anchor_num)
    anchor = []
    for x_ in x:
        # x_ = torch.tensor(x_).to(device)
        # x_ = x_.flatten(1)
        kmeans.fit_predict(x_.t())
        anchor.append(kmeans.centroids.t().to(device))
        
    return anchor


def _generate_anchors_align_(x_list, anchor_num, device):
        # l = []
        # for x_ in x_list:
        #     x_ = torch.tensor(x_).to(device)
        #     l.append(x_)
            
        x = torch.concat(x_list, dim=0)
        k_means = KMeans(n_clusters=anchor_num, verbose=False, minibatch=10000)
        _ = k_means.fit_predict(x.t())
        anchors = []
        start = 0
        for i in range(len(x_list)):
            end = start + x_list[i].shape[0]
            anchors.append(k_means.centroids.t()[start:end, :])
            start = start + x_list[i].shape[0]
        return anchors