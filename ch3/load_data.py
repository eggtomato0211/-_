#必要ライブラリのインポート
import numpy as np
import matplotlib.pyplot as plt

#データの可視化
def load_binary_data(filename, num_sumples=300, dimensions=2):
    #バイナリーデータを読み込む
    data = np.fromfile(filename, dtype= np.float32)
    return data.reshape((num_sumples, dimensions))


def visualize_data(class1_data, class2_data):
    """
    クラス1とクラス2のデータを散布図で表示する関数
    引数:
        class1_data (np.ndarray): クラス1のデータ
        class2_data (np.ndarray): クラス2のデータ
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(class1_data[:, 0], class1_data[:, 1], label='Class 1 (Kobe)', color='blue', alpha=0.6)
    plt.scatter(class2_data[:, 0], class2_data[:, 1], label='Class 2 (Naha)', color='orange', alpha=0.6)
    plt.xlabel('Temperature')
    plt.ylabel('Humidity')
    plt.title('Scatter Plot of Class 1 (Kobe) and Class 2 (Naha)')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # データの読み込み
    class1_data = load_binary_data('class1.dat')
    class2_data = load_binary_data('class2.dat')

    # データの可視化
    visualize_data(class1_data, class2_data)