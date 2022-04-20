import pandas as pd
import numpy as np
from visualization import *

"""
total_final.csv 에 바로 적용X
total_final.csv 를 로드 후
각 게임 별로 feature를 만들고,
다시 concat 하는것이 좋습니다.
"""

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

# 챔피언별 우물로부터의 거리 / 팀별 거리 스코어를 반환합니다.
def make_distance(df):
    
    df = df.copy()
    blue_home_x = df[blue_x_loc_list].min().min()
    blue_home_y = df[blue_y_loc_list].max().max()
    
    red_home_x = df[red_x_loc_list].max().max()
    red_home_y = df[red_y_loc_list].min().min()
    
    blue_home = (blue_home_x, blue_home_y)
    red_home = (red_home_x, red_home_y)
    
    for i in range(5):
        df[f"{blue_x_loc_list[i][:-13]}.distance"] = \
        np.linalg.norm(df.loc[:, [blue_x_loc_list[i], blue_y_loc_list[i]]] - blue_home, axis=1)
        df[f"{red_x_loc_list[i][:-13]}.distance"] = \
        np.linalg.norm(df.loc[:, [red_x_loc_list[i], red_y_loc_list[i]]] - red_home, axis=1)
        
    dist_list = [dist for dist in df.columns.tolist() if "distance" in dist]
    blue_dist_list = [blue for blue in dist_list if "blue" in blue]
    red_dist_list = [red for red in dist_list if "red" in red]
    
    df['team.blue.dist.score'] = 0
    df['team.red.dist.score'] = 0
    
    for idx in df.index:
        df.loc[idx, 'team.blue.dist.score'] = sum(df.loc[idx, blue_dist_list].sort_values(ascending=False)[:3])/3
        df.loc[idx, 'team.red.dist.score'] = sum(df.loc[idx, red_dist_list].sort_values(ascending=False)[:3])/3
    
    df = df.drop(columns=dist_list)
    
    blue_dist_min = df['team.blue.dist.score'].min()
    blue_dist_max = df['team.blue.dist.score'].max()
    red_dist_min = df['team.red.dist.score'].min()
    red_dist_max = df['team.red.dist.score'].max()
    
    df['team.blue.dist.score'] = (df['team.blue.dist.score'] - blue_dist_min)/(blue_dist_max - blue_dist_min)
    df['team.red.dist.score'] = (df['team.red.dist.score'] - red_dist_min)/(red_dist_max - red_dist_min)
    
    return df

# make_kill_score에 쓰일 함수입니다.
def KDA_split(x):
    k, d, a = [int(i) for i in x.split("/")]
    return pd.Series([k, d, a], index= ['k', 'd', 'a'])

# Team별 kill_score를 반환하는 함수입니다.
def make_kill_score(df):
    df = df.copy()
    kda_list = [loc for loc in df.columns.tolist() if "KDA" in loc]
    kda_df = df[kda_list]
    blue_kda_list = [loc for loc in kda_df.columns.tolist() if "blue" in loc]
    red_kda_list = [loc for loc in kda_df.columns.tolist() if "red" in loc]
    df['team.blue.kill_score'] = ""
    df['team.red.kill_score'] = ""
    for f in kda_df.index:
        kda_sum=0
        for b in blue_kda_list:
            kda_sum+=KDA_split(kda_df[b][f])[0]
        df['team.blue.kill_score'][f] = kda_sum
        kda_sum=0
        for r in red_kda_list:
            kda_sum+=KDA_split(kda_df[r][f])[0]
        df['team.red.kill_score'][f] = kda_sum
    return df

# make_kda_score에 쓰일 함수입니다.
def kdasplit(x):
    if type(x)!=str:
        pass
    else:
        k, d, a = [int(i) for i in x.split("/")]
        if d == 0:
            kda_score = k+a
        else:
            kda_score = (k+a)/d
        return kda_score

# player별 KDA를 반환하는 함수입니다.
def make_kda_score(df):
    df = df.copy()
    kda_list = [loc for loc in df.columns.tolist() if "KDA" in loc]
    blue_kda_list = [loc for loc in kda_list if "blue" in loc]
    red_kda_list = [loc for loc in kda_list if "red" in loc]
    
    for kda in kda_list:
        df[f'{kda}.score'] = df[kda].apply(lambda x: kdasplit(x))
        
    return df

# Team별 total_cs를 반환하는 함수입니다.
def make_total_cs(df):
    df = df.copy()
    cs_list = [loc for loc in df.columns.tolist() if "CS" in loc]
    cs_df = df[cs_list]
    blue_cs_list = [loc for loc in cs_df.columns.tolist() if "blue" in loc]
    red_cs_list = [loc for loc in cs_df.columns.tolist() if "red" in loc]
    df['team.blue.total_cs'] = ""
    df['team.red.total_cs'] = ""
    for d in cs_df.index:
        cs_sum=0
        for b in blue_cs_list:
            cs_sum+=int(cs_df[b][d])
        df['team.blue.total_cs'][d]=cs_sum
        cs_sum=0
        for r in red_cs_list:
            cs_sum+=int(cs_df[r][d])
        df['team.red.total_cs'][d]=cs_sum
    return df

# 개인 별 이전 프레임 대비 현재 cs 변화를 반환하는 함수입니다.
def make_cs_change(df):
    df = df.copy()
    cs_list = [loc for loc in df.columns.tolist() if "CS" in loc]
    for c in cs_list:
        cs_var_df = pd.DataFrame()
        cs_var_df[c] = df[c]
        cs_var_df[c + '.change'] = ''
        for f in cs_var_df.index:
            if f==0: pass
            #elif (check_empty(cs_var_df, f) == True) | (check_empty(cs_var_df, f-1) == True): pass
            else:
                cs_var_df[c + '.change'][f] = int(cs_var_df[c][f]) - int(cs_var_df[c][f-1])
        for f in cs_var_df.index:
            if cs_var_df[c + '.change'][f]== '':
                cs_var_df[c + '.change'][f] = 0
        df[c + '.change'] = cs_var_df[c + '.change']
    return df

# 모든 process를 아우르는 함수입니다.
def make_features(df):
    df = df.copy()
    df = make_distance(df)
    df = make_kill_score(df)
    df = make_kda_score(df)
    df = make_cs_change(df)
    
    return df


# category_clustering함수에 쓰일 루트를 카테고리화 시키는 함수입니다.
def loc_category(dfs):
    df = dfs.copy()
    df['player.blue.T.route'] = 0
    df['player.blue.J.route'] = 0
    df['player.blue.M.route'] = 0
    df['player.blue.A.route'] = 0
    df['player.blue.S.route'] = 0
    df['player.red.T.route'] = 0
    df['player.red.J.route'] = 0
    df['player.red.M.route'] = 0
    df['player.red.A.route'] = 0
    df['player.red.S.route'] = 0
    route_list = [route for route in df.columns if "route" in route]
    
    for i in range(10):
        blue_home_cond = (df[loc_x_list[i]] >= 15) & (df[loc_x_list[i]] <= 100) & (df[loc_y_list[i]] >= 190) & (df[loc_y_list[i]] <= 280)
        blue_top_cond = (df[loc_x_list[i]] >= 15) & (df[loc_x_list[i]] <= 40) & (df[loc_y_list[i]] >= 40) & (df[loc_y_list[i]] <= 170)
        blue_bottom_cond = (df[loc_x_list[i]] >= 100) & (df[loc_x_list[i]] <= 240) & (df[loc_y_list[i]] >= 240) & (df[loc_y_list[i]] <= 280)
        blue_dukk_cond = (df[loc_x_list[i]] >= 45) & (df[loc_x_list[i]] <= 60) & (df[loc_y_list[i]] >= 115) & (df[loc_y_list[i]] <= 135)
        blue_bluebuff_cond = (df[loc_x_list[i]] >= 70) & (df[loc_x_list[i]] <= 85) & (df[loc_y_list[i]] >= 120) & (df[loc_y_list[i]] <= 140)
        blue_wolf_cond = (df[loc_x_list[i]] >= 70) & (df[loc_x_list[i]] <= 90) & (df[loc_y_list[i]] >= 150) & (df[loc_y_list[i]] <= 170)
        blue_kalbu_cond = (df[loc_x_list[i]] >= 125) & (df[loc_x_list[i]] <= 150) & (df[loc_y_list[i]] >= 175) & (df[loc_y_list[i]] <= 195)
        blue_redbuff_cond = (df[loc_x_list[i]] >= 135) & (df[loc_x_list[i]] <= 160) & (df[loc_y_list[i]] >= 200) & (df[loc_y_list[i]] <= 215)
        blue_golem_cond = (df[loc_x_list[i]] >= 150) & (df[loc_x_list[i]] <= 175) & (df[loc_y_list[i]] >= 225) & (df[loc_y_list[i]] <= 240)
        blue_mid_cond1 = (df[loc_x_list[i]] >= 115) & (df[loc_x_list[i]] <= 130) & (df[loc_y_list[i]] >= 155) & (df[loc_y_list[i]] <= 170)
        blue_mid_cond2 = (df[loc_x_list[i]] >= 100) & (df[loc_x_list[i]] <= 115) & (df[loc_y_list[i]] >= 170) & (df[loc_y_list[i]] <= 190)
        red_home_cond = (df[loc_x_list[i]] >= 188) & (df[loc_x_list[i]] <= 273) & (df[loc_y_list[i]] >= 8) & (df[loc_y_list[i]] <= 98)
        red_top_cond = (df[loc_x_list[i]] >= 48) & (df[loc_x_list[i]] <= 188) & (df[loc_y_list[i]] >= 8) & (df[loc_y_list[i]] <= 48)
        red_bottom_cond = (df[loc_x_list[i]] >= 248) & (df[loc_x_list[i]] <= 273) & (df[loc_y_list[i]] >= 118) & (df[loc_y_list[i]] <= 248)
        red_dukk_cond = (df[loc_x_list[i]] >= 228) & (df[loc_x_list[i]] <= 243) & (df[loc_y_list[i]] >= 153) & (df[loc_y_list[i]] <= 173)
        red_bluebuff_cond = (df[loc_x_list[i]] >= 195) & (df[loc_x_list[i]] <= 218) & (df[loc_y_list[i]] >= 148) & (df[loc_y_list[i]] <= 168)
        red_wolf_cond = (df[loc_x_list[i]] >= 198) & (df[loc_x_list[i]] <= 218) & (df[loc_y_list[i]] >= 118) & (df[loc_y_list[i]] <= 138)
        red_kalbu_cond = (df[loc_x_list[i]] >= 138) & (df[loc_x_list[i]] <= 163) & (df[loc_y_list[i]] >= 93) & (df[loc_y_list[i]] <= 113)
        red_redbuff_cond = (df[loc_x_list[i]] >= 128) & (df[loc_x_list[i]] <= 153) & (df[loc_y_list[i]] >= 873) & (df[loc_y_list[i]] <= 88)
        red_golem_cond = (df[loc_x_list[i]] >= 113) & (df[loc_x_list[i]] <= 138) & (df[loc_y_list[i]] >= 48) & (df[loc_y_list[i]] <= 63)
        red_mid_cond1 = (df[loc_x_list[i]] >= 158) & (df[loc_x_list[i]] <= 173) & (df[loc_y_list[i]] >= 118) & (df[loc_y_list[i]] <= 133)
        red_mid_cond2 = (df[loc_x_list[i]] >= 173) & (df[loc_x_list[i]] <= 188) & (df[loc_y_list[i]] >= 98) & (df[loc_y_list[i]] <= 118)
        common_top = (df[loc_x_list[i]] >= 48) & (df[loc_x_list[i]] <= 80) & (df[loc_y_list[i]] >= 48) & (df[loc_y_list[i]] <= 75)
        common_mid = (df[loc_x_list[i]] >= 130) & (df[loc_x_list[i]] <= 155) & (df[loc_y_list[i]] >= 133) & (df[loc_y_list[i]] <= 155)
        common_bottom = (df[loc_x_list[i]] >= 210) & (df[loc_x_list[i]] <= 240) & (df[loc_y_list[i]] >= 210) & (df[loc_y_list[i]] <= 240)
        common_herald = (df[loc_x_list[i]] >= 80) & (df[loc_x_list[i]] <= 75) & (df[loc_y_list[i]] >= 75) & (df[loc_y_list[i]] <= 115)
        common_dragon = (df[loc_x_list[i]] >= 175) & (df[loc_x_list[i]] <= 210) & (df[loc_y_list[i]] >= 175) & (df[loc_y_list[i]] <= 210)

        df.loc[blue_home_cond, route_list[i]] = 1
        df.loc[red_home_cond, route_list[i]] = 2
                         
        df.loc[blue_top_cond, route_list[i]] = 3
        df.loc[red_top_cond, route_list[i]] = 4
                         
        df.loc[blue_bottom_cond, route_list[i]] = 5
        df.loc[red_bottom_cond, route_list[i]] = 6
                         
        df.loc[blue_dukk_cond, route_list[i]] = 7
        df.loc[red_dukk_cond, route_list[i]] = 8
                         
        df.loc[blue_bluebuff_cond, route_list[i]] = 9
        df.loc[red_bluebuff_cond, route_list[i]] = 10
                         
        df.loc[blue_wolf_cond, route_list[i]] = 11
        df.loc[red_wolf_cond, route_list[i]] = 12
                         
        df.loc[blue_kalbu_cond, route_list[i]] = 13
        df.loc[red_kalbu_cond, route_list[i]] = 14
                         
        df.loc[blue_redbuff_cond, route_list[i]] = 15
        df.loc[red_redbuff_cond, route_list[i]] = 16
                         
        df.loc[blue_golem_cond, route_list[i]] = 17
        df.loc[red_golem_cond, route_list[i]] = 18
                         
        df.loc[(blue_mid_cond1 | blue_mid_cond2), route_list[i]] = 19
        df.loc[(red_mid_cond1 | red_mid_cond2), route_list[i]] = 20
                         
        df.loc[common_top, route_list[i]] = 21
        df.loc[common_bottom, route_list[i]] = 22
        df.loc[common_mid, route_list[i]] = 23
        df.loc[common_herald, route_list[i]] = 24
        df.loc[common_dragon, route_list[i]] = 25
        
    return df