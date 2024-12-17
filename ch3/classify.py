#インポート
import numpy as np
from load_data import load_binary_data
from gmm_model import GMM

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


if __name__ == "__main__":
    # クラス1とクラス2のデータを読み込む
    class1_data = load_binary_data('class1.dat')
    class2_data = load_binary_data('class2.dat')

    # GMMの学習
    gmm1 = GMM(num_components=2)
    gmm2 = GMM(num_components=2)
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