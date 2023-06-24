import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
import tempfile
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt

window = 7636 # 部分窓
threshold = 15 # 異常データとみなす閾値

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

# 異常発生時の赤字表示用
def color_red_if_over_threshold(val):
    #Thresholdを超える場合、文字を赤色で太字にする
    color = 'red' if val > threshold else 'black'
    weight = 'bold' if val > threshold else 'normal'
    return f'color: {color}; font-weight: {weight}'


# メインプログラム
def main():
    st.title('Autoencoder Comparison App')

    # モデルファイルのアップロード
    model_file = st.file_uploader("Upload your Autoencoder model", type=["h5"])
    if model_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(model_file.read())

        with st.spinner('Loading model...'):
            model = load_model(tfile.name)
        st.success('Model loaded successfully!')

    # サンプルデータのアップロード
    data_file = st.file_uploader("Upload your sample data", type=["csv"])
    if data_file is not None:
        with st.spinner('Loading data...'):
            data = pd.read_csv(data_file)
        st.success('Data loaded successfully!')
        # st.write(data.head())

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

        
        # オプション１：両配列の長さを比較し、短い配列を長い配列の長さに合わせて拡張します。
        #if len(test_data) != len(pred_data):
        #    if len(test_data) < len(pred_data):
        #        diff = len(pred_data) - len(test_data)
        #        arr1 = np.pad(test_data, (0, diff), 'edge')
        #    else:
        #        diff = len(test_data) - len(pred_data)
        #        pred_data = np.pad(pred_data, (0, diff), 'edge')

        # 各データの要素同士の二乗差分算出
        squared_error = np.square(test_data - pred_data)

        # オプション２：二乗誤差が0.001未満の場合はゼロとみなします。
        # threshold = 0.001
        # squared_error = np.where(squared_error < threshold, 0, squared_error)

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
        
        
        # データフレームの表示
        #st.write(df_errors)


        # Total Squared Errorの列だけにスタイリングを適用
        # html = df_errors.style.apply(lambda s: [color_red_if_over_threshold(v) for v in s], subset=['Total Squared Error']).set_properties(**{'font-size': '20pt', 'text-align': 'center'}).render()

        # HTMLにスタイルを適用
        # html = html.replace('<table>', '<table style="width:50%; margin-left: auto; margin-right: auto;">')

        # HTMLを表示
        # st.markdown(html, unsafe_allow_html=True)

        データフレームをHTML形式に変換
        #html = df_errors.to_html(float_format="{:0.3f}".format)

        HTMLにスタイルを適用
        #html = html.replace('<table>', '<table style="width:150%; font-size:20px; margin-left: auto; margin-right: auto;">')

        HTMLを表示
        #st.markdown(html, unsafe_allow_html=True)
       
        




if __name__ == "__main__":
    main()
