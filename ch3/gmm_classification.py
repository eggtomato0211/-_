import numpy as np
import matplotlib.pyplot as plt
from load_data import load_binary_data
from gmm_model import GMM

def run_gmm_experiment(class1_data, class2_data, num_components_list):
    """
    GMMの実験を行い、混合数ごとに対数尤度とクラスごとの分布を表示
    """
    for num_components in num_components_list:
        print(f"\n【実験: 混合数 = {num_components}】")
        
        # GMMの学習
        gmm1 = GMM(num_components=num_components)
        gmm2 = GMM(num_components=num_components)
        gmm1.fit(class1_data)
        gmm2.fit(class2_data)

        # 対数尤度のプロットと保存
        gmm1.plot_log_likelihood(gmm1.fit(class1_data), num_components)

        # クラス1とクラス2の散布図 + 平均ベクトルの可視化と保存
        plot_class_means(class1_data, class2_data, gmm1, gmm2, num_components)


def plot_class_means(class1_data, class2_data, gmm1, gmm2, num_components):
    """
    クラス1とクラス2のデータをプロットし、GMMの成分の平均ベクトルを可視化
    """
    plt.figure(figsize=(8, 6))
    
    # クラス1とクラス2のデータをプロット
    plt.scatter(class1_data[:, 0], class1_data[:, 1], label='Class 1', color='blue', alpha=0.6)
    plt.scatter(class2_data[:, 0], class2_data[:, 1], label='Class 2', color='orange', alpha=0.6)
    
    # GMMの平均ベクトルを可視化
    for m in range(gmm1.num_components):
        plt.scatter(gmm1.means[m][0], gmm1.means[m][1], color='red', marker='x', s=100, label=f'Class 1 Mean {m+1}')
    for m in range(gmm2.num_components):
        plt.scatter(gmm2.means[m][0], gmm2.means[m][1], color='green', marker='x', s=100, label=f'Class 2 Mean {m+1}')
    
    plt.title(f'Mean Positions (num_components={num_components})')
    plt.xlabel('Temperature')
    plt.ylabel('Humidity')
    plt.legend()
    plt.savefig(f'means_{num_components}.png')
    plt.close()


def classify_and_calculate_accuracy(class1_data, class2_data, num_components_list):
    """
    各混合数でクラス1とクラス2の識別率を計算し、表示する
    """
    results = []

    for num_components in num_components_list:
        print(f"\n【識別実験: 混合数 = {num_components}】")
        
        # GMMの学習
        gmm1 = GMM(num_components=num_components)
        gmm2 = GMM(num_components=num_components)
        gmm1.fit(class1_data)
        gmm2.fit(class2_data)

        # クラス1とクラス2の識別を行う
        predictions_class1 = classify_data(gmm1, gmm2, class1_data)
        predictions_class2 = classify_data(gmm1, gmm2, class2_data)

        # 正しいラベルを作成 (クラス1は1、クラス2は2)
        true_labels_class1 = np.ones(len(class1_data))
        true_labels_class2 = np.ones(len(class2_data)) * 2

        # 識別率を計算
        accuracy1 = calculate_accuracy(predictions_class1, true_labels_class1)
        accuracy2 = calculate_accuracy(predictions_class2, true_labels_class2)

        print(f"クラス1の識別率: {accuracy1:.2f}%")
        print(f"クラス2の識別率: {accuracy2:.2f}%")

        results.append((num_components, accuracy1, accuracy2))

    return results


def classify_data(gmm1, gmm2, data):
    """
    データをクラス1またはクラス2に分類する
    """
    N = data.shape[0]
    predictions = np.zeros(N)
    
    for i in range(N):
        log_prob1 = gmm1.log_likelihood(data[i].reshape(1, -1))
        log_prob2 = gmm2.log_likelihood(data[i].reshape(1, -1))
        predictions[i] = 1 if log_prob1 > log_prob2 else 2  # 識別結果 (クラス1 or クラス2)
    
    return predictions


def calculate_accuracy(predictions, true_labels):
    """
    識別率を計算する
    """
    correct = np.sum(predictions == true_labels)
    total = len(true_labels)
    return (correct / total) * 100

def run_gmm_with_initial_conditions(data, initial_conditions, num_components):
    """
    異なる初期条件でGMMを学習し、対数尤度の変化をプロットする。
    """
    for i, means in enumerate(initial_conditions):
        print(f"【初期条件セット {i + 1}】")
        gmm = GMM(num_components=num_components)
        gmm.initialize_parameters(data, custom_means=means)  # 初期平均ベクトルを設定
        log_likelihoods = gmm.fit(data)  # 学習
        gmm.plot_log_likelihood(log_likelihoods, num_components=num_components)


if __name__ == "__main__":
    # データの読み込み
    class1_data = load_binary_data('class1.dat')
    class2_data = load_binary_data('class2.dat')

    # 初期条件の設定（例: 2成分の場合）
    initial_conditions = [
        np.array([[10, 10], [20, 20]]),
        np.array([[15, 15], [25, 25]]),
        np.array([[5, 5], [30, 30]])
    ]

    # 初期条件を変更した実験を実施
    run_gmm_with_initial_conditions(class1_data, initial_conditions, num_components=2)
        
    
    
