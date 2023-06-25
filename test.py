# 各種インポート
import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
import tempfile
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import os

# 正規化
def act_minxmax_scaler(data):
    data = np.array(data).reshape(-1, 1)
    scaler = MinMaxScaler()
    scaler.fit(data)
    scaled_data = scaler.transform(data)
    return scaled_data

# 部分時系列の作成
def split_part_recurrent_data(data_list, window):
    data_vec = []
    for i in range(len(data_list)-window+1):
        data_vec.append(data_list[i:i+window])

    return data_vec

# 部分窓の定義
window = 7636 

# メインプログラム
def main():
    st.title('Autoencoder Comparison App')

    # サイドバーにmse_thresholdの数値入力ボックスを追加（千分の一の位置まで）
    mse_threshold = st.sidebar.number_input(
        "MSE Threshold x 1000",
        min_value=0,  # 下限値
        max_value=1000,  # 上限値
        value=0,  # 初期値
        step=1,  # 加減値
    )

    mse_threshold /= 1000  # 実際の値に戻す

    # サイドバーにthresholdの数値入力ボックスを追加
    warning_threshold = st.sidebar.number_input(
        "Warning Threshold",
        min_value=0,  # 下限値
        max_value=1000,  # 上限値
        value=10,  # 初期値
        step=1  # 加減値
    )

    # サイドバーにattention_thresholdの数値入力ボックスを追加
    # ここで、上で定義したthresholdをattention_thresholdのmax_valueとして用いる
    attention_threshold = st.sidebar.number_input(
        "Attention Threshold",
        min_value=0,  # 下限値
        max_value=warning_threshold,  # 上限値をthresholdに設定
        value=min(5, warning_threshold),  # 初期値がthresholdを超えないように設定
        step=1  # 加減値
    )

    # 値を表示（デバッグ用）
    st.write(f"MSE Threshold: {mse_threshold:.3f}")
    st.write(f"Warning Threshold: {warning_threshold}")
    st.write(f"Attention Threshold: {attention_threshold}")
    

    # モデルファイルとサンプルデータのアップロード待機
    model_file = st.file_uploader("Upload your Autoencoder model", type=["h5"])
    data_file = st.file_uploader("Upload your sample data", type=["csv"])
    if model_file is not None and data_file is not None:

        tfile_model = tempfile.NamedTemporaryFile(delete=False) 
        tfile_model.write(model_file.read())
        tfile_model.close()

        with st.spinner('Loading model...'):
            model = load_model(tfile_model.name)
        st.success('Model loaded successfully!')
        
        os.remove(tfile_model.name)  # モデルファイルの削除

        with st.spinner('Loading data...'):
            data = pd.read_csv(data_file)
        st.success('Data loaded successfully!')

        # 正規化
        data = data['Torque']
        scaled_data = act_minxmax_scaler(data)
        # 一次元リスト
        test_data = scaled_data.flatten()
        # 部分時系列作成
        test_vec2 = split_part_recurrent_data(test_data, window)
        # np変換
        test_vec2 = np.array(test_vec2)
        # モデルをテストデータに適用
        pred_vec = model.predict(test_vec2)
        # dfを1次元配列に変換します。
        pred_data = pred_vec.flatten()

        # カラムの作成
        col1, col2 = st.columns(2)
        # col1, col2, col3, col4 = st.columns(4)
        # scaled_dataのグラフを描画
        with col1:
            #st.header('Scaled Data')
            st.markdown("<h2 style='font-size: 20px; text-align: center; color: black;'>Scaled Data</h2>", unsafe_allow_html=True)
            fig = plt.figure(figsize=(5, 5))
            plt.plot(scaled_data, color='blue')
            st.pyplot(fig)

        # pred_dataのグラフを描画
        with col2:
            #st.header('Predicted Data')
            st.markdown("<h2 style='font-size: 20px; text-align: center; color: black;'>Predicted Data</h2>", unsafe_allow_html=True)
            fig = plt.figure(figsize=(5, 5))
            plt.plot(pred_data, color='red')
            st.pyplot(fig)

        # ヒストグラムのカラムを作成
        col3, col4 = st.columns(2)

        # scaled_dataのヒストグラムを描画
        with col3:
            #st.header('Histogram of Scaled Data')
            st.markdown("<h2 style='font-size: 20px; text-align: center; color: black;'>Histogram of Scaled Data</h2>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.hist(scaled_data, bins=30, alpha=0.7, color='b')
            st.pyplot(fig)

        # pred_dataのヒストグラムを描画
        with col4:
            #st.header('Histogram of Predicted Data')
            st.markdown("<h2 style='font-size: 20px; text-align: center; color: black;'>Histogram of Predicted Data</h2>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.hist(pred_data, bins=30, alpha=0.7, color='r')
            st.pyplot(fig)

        
        # 各データの要素同士の二乗差分算出
        squared_error = np.square(test_data - pred_data)

        # 二乗誤差が閾値（mse_threshold）未満の場合はゼロとみなします。
        squared_error = np.where(squared_error < mse_threshold, 0, squared_error)

        # カラムの作成
        col5, col6 = st.columns(2)

        # squared_errorのグラフを描画
        with col5:
            #st.header('Squared Error')
            st.markdown("<h2 style='font-size: 20px; text-align: center; color: black;'>Squared Error</h2>", unsafe_allow_html=True)
            fig = plt.figure(figsize=(5, 5))
            plt.plot(squared_error, color='green')
            st.pyplot(fig)

        # squared_errorのヒストグラムを描画
        with col6:
            #st.header('Histogram of Squared Error')
            st.markdown("<h2 style='font-size: 20px; text-align: center; color: black;'>Histogram of Squared Error</h2>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.hist(squared_error, bins=30, alpha=0.7, color='g')
            st.pyplot(fig)

        # 最大値/最小値/平均値/中央値/累計差分値の算出
        max_error = np.max(squared_error)
        min_error = np.min(squared_error)
        mean_error = np.mean(squared_error)
        median_error = np.median(squared_error)
        total_squared_error = np.sum(squared_error)

        # データフレームの作成
        df_errors = pd.DataFrame({
            'Value': [max_error, min_error, mean_error, median_error, total_squared_error]
        }, index=['Max Error', 'Min Error', 'Mean Error', 'Median Error', 'Total Squared Error'])
        
        # データフレームをHTML形式に変換
        df_errors_html = df_errors.to_html(float_format="{:0.20f}".format)

        # HTMLにスタイルを適用
        df_errors_html = df_errors_html.replace('<table>', '<table style="width:80%; font-size:40px; margin-left: auto; margin-right: auto;">')

        # HTMLを表示
        st.markdown(df_errors_html, unsafe_allow_html=True)

        # 結果に応じた色とメッセージを設定
        if total_squared_error < attention_threshold:
            color1 = 'green'
            message1 = 'NORMAL'
            color2 = 'black'
            message2 = "The current torque data from the motor is within the normal range. It's safe to continue operation."
            result_html = f'''
            <div style="font-size:150px; color:{color1}; text-align:center;">{message1}</div>
            <div style="font-size:30px; color:{color2}; text-align:center;">{message2}</div>
            '''
        elif total_squared_error < warning_threshold:
            color1 = 'orange'
            message1 = 'ATTENTION'
            color2 = 'black'
            message2 = "The torque data from the motor exceeds the normal range. We recommend a detailed inspection."
            result_html = f'''
            <div style="font-size:150px; color:{color1}; text-align:center;">{message1}</div>
            <div style="font-size:30px; color:{color2}; text-align:center;">{message2}</div>
            '''
        else:
            color1 = 'red'
            message1 = 'WARNING'
            color2 = 'black'
            message2 = "An anomaly has been detected in the motor's torque data. Immediate action is required."
            result_html = f'''
            <div style="font-size:150px; color:{color1}; text-align:center;">{message1}</div>
            <div style="font-size:30px; color:{color2}; text-align:center;">{message2}</div>
            '''

        st.markdown(result_html, unsafe_allow_html=True)


       
        




if __name__ == "__main__":
    main()
