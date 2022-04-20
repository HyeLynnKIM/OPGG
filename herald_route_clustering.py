from data_preprocessing import *
from feature_engineering import *
from visualization import *
from yellowbrick.cluster import SilhouetteVisualizer
from sklearn.decomposition import PCA
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset

from tslearn.clustering import silhouette_score
from sklearn.cluster import KMeans


# 전체 클러스터링 프로세스를 정리한 함수입니다.
def herald_route_cluster(df):
    dfs = []
    for i in df['gameclass'].unique():
        dfs.append(scailing(df[df['gameclass']==i].reset_index(drop=True)))
        
    df = pd.concat(dfs, axis=0, ignore_index=True)
    df = loc_category(df)
    df = fill_zero_route(df)
    
    pos = input('select position : ')
    t = input('select team : ')
    
    evaluating_ts_cluster(df, position=pos, team=t)
    
    n = input('select your num of clusters : ')
    
    df = merge_category_ts_cluster(df)
    
    return df

# 시계열 군집화 성능 평가 함수입니다.
def evaluating_ts_cluster(df, position, team, metric="dtw"):
    df_ = df.copy()
    df_ = df_[['gameclass', f'player.{team}.{position}.route']]
    
    route_list = []
    game_list = df_['gameclass'].unique().tolist()

    for game in game_list:
        routes = df_[df_['gameclass']==game][f'player.{team}.{position}.route'].tolist()
        length = len(routes)
        frame_10 = range(0, length, 10)
        routes_list = []
        for i in frame_10:
            routes_list.append(routes[i])
        route_list.append(routes_list)
    
    df_ = pd.DataFrame({'gameclass':game_list, f'player.{team}.{position}.route':route_list})
    
    inertia_arr = []
    silhouettes = []
    k_range = range(2,11)
    data = to_time_series_dataset(df_[f'player.{team}.{position}.route'])
    
    for k in k_range:
        
        model = TimeSeriesKMeans(n_clusters=k, metric=metric, random_state=0)
        model.fit(data)
        inertia = model.inertia_
        
        yhat = model.predict(data)
        silhouette = silhouette_score(data, yhat)
                
        inertia_arr.append(inertia)
        silhouettes.append(silhouette)
        
    inertia_arr = np.array(inertia_arr)
    silhouette_arr = np.array(silhouettes)
    
#     elbow_visualization(k_range, inertia_arr)
#     silhouette_visualization(k_range, silhouette_arr)
    cluster_visualization(k_range, inertia_arr, silhouette_arr)
    
# 시계열 군집화를 진행하는 함수입니다.
def category_ts_clustering(df, position, team, n_cluster, metric="dtw"):
    df_ = df.copy()
    df_ = df_[['gameclass', f'player.{team}.{position}.route']]
    
    route_list = []
    game_list = df_['gameclass'].unique().tolist()

    for game in game_list:
        routes = df_[df_['gameclass']==game][f'player.{team}.{position}.route'].tolist()
        length = len(routes)
        frame_10 = range(0, length, 10)
        routes_list = []
        for i in frame_10:
            routes_list.append(routes[i])
        route_list.append(routes_list)
    
    df_ = pd.DataFrame({'gameclass':game_list, f'player.{team}.{position}.route':route_list})

    data = to_time_series_dataset(df_[f'player.{team}.{position}.route'])
    model = TimeSeriesKMeans(n_clusters=n_cluster, metric=metric, random_state=0)
    cluster = model.fit_predict(data)
    cluster_df = pd.DataFrame(cluster, columns=[f'player.{team}.{position}.cluster'])
    
#     print('----------------------------------------------------')
#     print(f"n_cluster '{n_cluster}' Silhouette_score :", silhouette_score(data, cluster))
#     print('----------------------------------------------------')
    
    df_ = pd.concat([df_[['gameclass']], cluster_df], axis=1)
    
    return df_
    
# 범주형으로 군집화된 데이터프레임을 통합 데이터프레임에 merge하는 함수입니다.
def merge_category_ts_cluster(df, n):
    df = df.copy()
    position_list = ['T', 'J', 'M', 'A', 'S']
    team_list = ['blue', 'red']
    for team in team_list:
        for position in position_list:
            cluster = category_ts_clustering(df, position=position, team=team, n_cluster=n)
            df = pd.merge(df, cluster, how='left', on='gameclass')
            
    return df



"""
아래 함수들은 분석에 진행했지만 성능이 좋지않아
최종적으로 사용하지 않은 함수들입니다.
===================================================================================================
"""

# 해당 포지션, 팀의 동선을 카테고리화 하여 군집화하는 함수입니다.
def category_clustering(df, position, team, n_cluster):
    df_ = df.copy()
    df_ = df_[['gameclass', 'time_stamp2', f'player.{team}.{position}.route']]
    
    game_list = df_['gameclass'].unique().tolist()
    time_list = list(range(360, 541))
    df_time_list = df_['time_stamp2'].tolist()
    
    for time in time_list:
        df_[f'player.{team}.{position}.{time}.route'] = 0
        
    for game in game_list:
        for time in time_list:
            try:
                df_.loc[df_['gameclass']==game, f'player.{team}.{position}.{time}.route'] = \
                df_.loc[(df_['time_stamp2']==time) & (df_['gameclass']==game), f'player.{team}.{position}.route'].values[0]

            except IndexError: # 해당 시간대에 값이 없어서 생기는 오류
                continue
            
    df_ = df_.drop(columns=[f'player.{team}.{position}.route', 'time_stamp2'])
    time_routes = [time_route for time_route in df_.columns.tolist() if ".route" in time_route]
    
    for time in time_routes:
        df_ = pd.get_dummies(df_, columns=[time])
    
    dfs_ = []
    
    for game in game_list:
        dfs_.append(df_[df_['gameclass']==game].head(1))
        
    df_ = pd.concat(dfs_, axis=0, ignore_index=True)
    df_class = df_[['gameclass']]
    
    data = df_.iloc[:,1:].copy()
    model = KMeans(n_clusters=n_cluster)
    cluster = model.fit_predict(data)
    cluster_df = pd.DataFrame(cluster, columns=[f'player.{team}.{position}.cluster'])
    
    print('----------------------------------------------------')
    print(f"n_cluster '{n_cluster}' Silhouette_score :", silhouette_score(data, cluster))
    print('----------------------------------------------------')
    
    visualization_silhouette(model, data)
    
    df_ = pd.concat([df_[['gameclass']], cluster_df], axis=1)
    
    return df_

# 군집화된 데이터프레임을 통합 데이터프레임에 merge하는 함수입니다.
def merge_cluster(df):
    df = df.copy()
    position_list = ['T', 'J', 'M', 'A', 'S']
    team_list = ['blue', 'red']
    for team in team_list:
        for position in position_list:
            cluster = clustering(df, position=position, team=team)
            df = pd.merge(df, cluster, how='left', on='gameclass')
            
    return df

# 해당 포지션, 팀의 동선을 군집화하는 함수입니다.
def clustering(df, position, team, n_cluster=10, metric="dtw"):
    df_ = df.copy()
    df_ = df_[['gameclass', 'time_stamp2', f'player.{team}.{position}.coordinate.x', f'player.{team}.{position}.coordinate.y']]
    
    x_mean = np.mean(df_[f'player.{team}.{position}.coordinate.x'])
    x_dev = np.std(df_[f'player.{team}.{position}.coordinate.x'])
    y_mean = np.mean(df_[f'player.{team}.{position}.coordinate.y'])
    y_dev = np.std(df_[f'player.{team}.{position}.coordinate.y'])
    
    df_[f'player.{team}.{position}.coordinate.x'] = (df_[f'player.{team}.{position}.coordinate.x'] - x_mean)/x_dev
    df_[f'player.{team}.{position}.coordinate.y'] = (df_[f'player.{team}.{position}.coordinate.y'] - y_mean)/y_dev
    
    pca = PCA(n_components=1)
    principalComponents = pca.fit_transform(df_.iloc[:,-2:])
    principalDf = pd.DataFrame(data=principalComponents, columns=['PCA_xy'])
    df_ = pd.concat([df_['gameclass'], principalDf], axis=1)
    
    pca_list = []
    game_list = df_['gameclass'].unique().tolist()

    for game in game_list:
        pca_list.append(df_[df_['gameclass']==game]['PCA_xy'].tolist())
    
    df_ = pd.DataFrame({'gameclass':game_list, 'PCA_xy':pca_list})

    data = to_time_series_dataset(df_['PCA_xy'])
    model = TimeSeriesKMeans(n_clusters=n_cluster, metric=metric, random_state=0)
    cluster = model.fit_predict(data)
    cluster_df = pd.DataFrame(cluster, columns=[f'player.{team}.{position}.cluster'])
    
    print('----------------------------------------------------')
    print(f"n_cluster '{n_cluster}' Silhouette_score :", silhouette_score(data, cluster))
    print('----------------------------------------------------')
    
    visualization_silhouette(model, data)
    
    df_ = pd.concat([df_[['gameclass']], cluster_df], axis=1)
    
    return df_