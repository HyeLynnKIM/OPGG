import pandas as pd
import numpy as np
from visualization import *
from feature_engineering import *

loc_list = ['player.blue.T.coordinate.x', 'player.blue.T.coordinate.y', 'player.blue.J.coordinate.x',
 'player.blue.J.coordinate.y', 'player.blue.M.coordinate.x', 'player.blue.M.coordinate.y', 'player.blue.A.coordinate.x',
 'player.blue.A.coordinate.y', 'player.blue.S.coordinate.x', 'player.blue.S.coordinate.y', 'player.red.T.coordinate.x',
 'player.red.T.coordinate.y', 'player.red.J.coordinate.x', 'player.red.J.coordinate.y', 'player.red.M.coordinate.x',
 'player.red.M.coordinate.y', 'player.red.A.coordinate.x', 'player.red.A.coordinate.y', 'player.red.S.coordinate.x',
 'player.red.S.coordinate.y']

loc_x_list = [x for x in loc_list if ".x" in x]
loc_y_list = [y for y in loc_list if ".y" in y]

blue_loc_list = [loc for loc in loc_list if "blue" in loc]
blue_x_loc_list = [x for x in blue_loc_list if "coordinate.x" in x]
blue_y_loc_list = [y for y in blue_loc_list if "coordinate.y" in y]
red_loc_list = [loc for loc in loc_list if "red" in loc]
red_x_loc_list = [x for x in red_loc_list if "coordinate.x" in x]
red_y_loc_list = [y for y in red_loc_list if "coordinate.y" in y]

# 좌표 스케일을 조정하는 함수입니다.
def scailing(df):
    blue_home_x = df[blue_x_loc_list].min().min()
    blue_home_y = df[blue_y_loc_list].max().max()
    
    red_home_x = df[red_x_loc_list].max().max()
    red_home_y = df[red_y_loc_list].min().min()
    
    df = df[df['replay']==False]
    df = df.copy()
    
    # scaling (미니맵 사이즈가 288 이상으로 잘못 크롭된 경우)
    if blue_home_x >= (0+20):
        w = blue_home_x - 20
        df[blue_x_loc_list] = df[blue_x_loc_list] - w
    if blue_home_x < (0+20):
        w = 20 - blue_home_x
        df[blue_x_loc_list] = df[blue_x_loc_list] + w
        
    if blue_home_y >= (288-20):
        w = blue_home_y - (288-20)
        df[blue_y_loc_list] = df[blue_y_loc_list] - w
    if blue_home_y < (288-20):
        w = (288-20) - blue_home_y 
        df[blue_y_loc_list] = df[blue_y_loc_list] + w
        
    if red_home_y >= (0+20):
        w = red_home_y - 20
        df[red_y_loc_list] = df[red_y_loc_list] - w
    if red_home_y < (0+20):
        w = 20 - red_home_y
        df[red_y_loc_list] = df[red_y_loc_list] + w
        
    if red_home_x >= (288-20):
        w = red_home_x - (288-20)
        df[red_x_loc_list] = df[red_x_loc_list] - w
    if red_home_x < (288-20):
        w = (288-20) - red_home_x 
        df[red_x_loc_list] = df[red_x_loc_list] + w
    
    return df

def fill_zero_route(dfs):
    dfs_ = dfs.copy()
    position_list = ['T', 'J', 'M', 'A', 'S']
    team_list = ['blue', 'red']
    gameclass = dfs_['gameclass'].unique().tolist()
    df_list = []
    for gc in gameclass:
        df_ = dfs_[dfs_['gameclass']==gc]
        for team in team_list:
            for pos in position_list:
                for idx in df_.index:
                    if df_[f'player.{team}.{pos}.route'][idx]==0:
                        prev = 0
                        prev_idx = 1
                        while(prev == 0):
                            #print(gc, idx ,team, pos, prev_idx)
                            if ((idx - prev_idx) >= df_.head(1).index):
                                if df_[f'player.{team}.{pos}.route'][idx - prev_idx] > 0:
                                    prev=df_.loc[idx - prev_idx, f'player.{team}.{pos}.route']
                                    break
                            elif ((idx + prev_idx) < df_.tail(1).index):
                                if df_[f'player.{team}.{pos}.route'][idx + prev_idx] > 0:
                                    prev = df_.loc[idx + prev_idx, f'player.{team}.{pos}.route']
                                    break
                            prev_idx += 1
                        df_.loc[idx, f'player.{team}.{pos}.route'] = prev
        df_list.append(df_)
    dff = pd.concat(df_list, axis=0, ignore_index=True)
    return dff