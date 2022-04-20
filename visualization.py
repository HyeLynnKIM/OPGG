import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import os
from yellowbrick.cluster import SilhouetteVisualizer

path = os.getcwd()

def loc_graph(df, position='J', team='blue'):
    """
    라이너 별 움직임의 연속성을 확인하기 위한 그래프 시각화
    """
    
    plt.figure(figsize=(16,12))

    ax1 = plt.subplot(2, 1, 1)
    df.plot(x='time_stamp2', y=f'player.{team}.{position}.coordinate.x', ax=ax1, color='r')
    df.plot.scatter(x='time_stamp2', y=f'player.{team}.{position}.coordinate.x', ax=ax1, color='r')
    
    plt.legend(['real_time', 'mean_time'], loc='upper left')
    plt.ylim(-0.1, 288)
    plt.ylabel('X_loc')
    plt.xticks(visible=False)
    ax1.set_title('X_location', fontsize=20)

    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    df.plot(x='time_stamp2', y=f'player.{team}.{position}.coordinate.y', ax=ax2, color='royalblue') 
    df.plot.scatter(x='time_stamp2', y=f'player.{team}.{position}.coordinate.y', ax=ax2, color='royalblue')

    plt.legend(['real_time', 'mean_time'], loc='upper left')
    plt.ylim(-0.1, 288)
    plt.xlabel('time_stamp')
    plt.ylabel('Y_loc')
    ax2.set_title('Y_location', fontsize=20)

    plt.show()


def loc_minimap(df, position='J'):
    """
    좌표를 미니맵에 시각화
    연한색이 출발지점 / 진한색이 도착지점
    """

    minimap = Image.open(path+'/data/image/empty_minimap_no_fog.png')
    minimap = minimap.transpose(Image.FLIP_TOP_BOTTOM)

    ax1 = plt.subplot(1, 2, 1)
    plt.rcParams["figure.figsize"] = (16,16)
    df.plot.scatter(x=f'player.blue.{position}.coordinate.x', y=f'player.blue.{position}.coordinate.y',
                               s=30, c=df.time_stamp2, cmap=plt.cm.Blues, vmin=360, vmax=540, ax=ax1, colorbar=False)
    plt.imshow(minimap, extent=[0, 288, 0, 288])
    plt.gca().invert_yaxis()
    plt.axis('off')
    ax1.set_title(f'Blue_position : {position}', fontsize=20)
    
    ax2 = plt.subplot(1, 2, 2)
    df.plot.scatter(x=f'player.red.{position}.coordinate.x', y=f'player.red.{position}.coordinate.y',
                               s=30, c=df.time_stamp2, cmap=plt.cm.Reds, vmin=360, vmax=540, ax=ax2, colorbar=False)
    plt.imshow(minimap, extent=[0, 288, 0, 288])
    plt.gca().invert_yaxis()
    plt.axis('off')
    ax2.set_title(f'Red_position : {position}', fontsize=20)

    plt.show()
    
def piechart(df):
    """
    각 군집별 pie_chart 를 시각화
    """
    
    cluster_list = [cluster for cluster in df.columns.tolist() if "cluster" in cluster]
    
    k = 1
    
    for cluster in cluster_list:
        plt.subplot(3,4,k)
        df[cluster].value_counts().plot.pie(y=cluster)
        plt.title(cluster)
        k += 1

    plt.show()

# 급격히 떨어지는 구간을 군집 개수로 선택
def elbow_visualization(k_range, inertia_arr):  
    plt.figure(figsize=(6, 6))
    plt.plot(k_range, inertia_arr)
    plt.title('Elbow method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.show()
    
# 실루엣 계수가 가장 높은 군집 개수로 선택
def silhouette_visualization(k_range, silhouette_arr):
    plt.figure(figsize=(6, 6))
    plt.plot(k_range, silhouette_arr)
    plt.title('Silhouette coeff')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette_score')
    plt.show()
    
# 클러스터 종합 평가
def cluster_visualization(k_range, inertia_arr, silhouette_arr):
    plt.figure(figsize=(12,8))

    ax1 = plt.subplot(2, 1, 1)
    plt.plot(k_range, inertia_arr)
    plt.ylabel('Inertia')
    plt.xticks(visible=False)
    ax1.set_title('Elbow method', fontsize=20)

    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    plt.plot(k_range, silhouette_arr)
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette_score')
    ax2.set_title('Sihouette_score', fontsize=20)

    plt.show()