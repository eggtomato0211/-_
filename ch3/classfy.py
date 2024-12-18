from load_data import load_binary_data
from gmm_model import GMM
import matplotlib.pyplot as plt

def run_gmm_experiment(class1_data, class2_data, num_components_list):
    for num_components in num_components_list:
        gmm = GMM(num_components=num_components)
        log_likelihoods = gmm.fit(class1_data)  # 学習データとしてclass1_dataを使用
        gmm.plot_log_likelihood(log_likelihoods)  # 対数尤度のプロット
        gmm.plot_means()  # 平均ベクトルの可視化

# データの読み込み
class1_data = load_binary_data('class1.dat')
class2_data = load_binary_data('class2.dat')

# 混合数を1から4まで変えて実験
num_components_list = [1, 2, 3, 4]
run_gmm_experiment(class1_data, class2_data, num_components_list)
