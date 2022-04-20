# json 데이터 추출하기
import os
from analysis.remerge import Remerge

directory = os.getcwd() + '/data/rawdata'
file_list = os.listdir(directory)
file_list = [file for file in file_list if "Raw" in file]

for file in file_list:
    try:

        print('-----------------------------')
        file = file[:-18]
        
        if os.path.isfile(f'{os.getcwd()}/data/postprocessed/{file+".json"}'):
            print(f'{file} 존재')
            continue
            
        remerge = Remerge(file)
        remerge.to_json_data()
        print(f"{file} 까지 완료되었습니다.")
        
        
    except UnboundLocalError:
        print(f"{file} 에서 UnboundLocalError 에러")
        pass
    
    except KeyError:
        print(f"{file} 에서 KeyError 에러")
        pass
    
    except TypeError:
        print(f"{file} 에서 TypeError 에러")
        pass
    
    except ValueError:
        print(f"{file} 에서 ValueError 에러")
        pass
    
    
    
# 전령 시간대 데이터프레임 생성
import os
import pandas as pd
import numpy as np
from analysis.remerge import Remerge

directory = os.getcwd() + '/data/rawdata'
file_list = os.listdir(directory)
file_list = [file[:-18] for file in file_list if "Raw" in file]

for file in file_list:
    
    if os.path.isfile(f'{os.getcwd()}/data/Herald_time_csv/{file+".csv"}'):
        print(f'{file} 존재')
        continue
    
    remerge = Remerge(file)
    data_w = remerge.load_json(form='wide', replay=False)
    cond1 = data_w.time_stamp2 >= 360
    cond2 = data_w.time_stamp2 <= 540
    data_w = data_w[cond1 & cond2]
    data_w.to_csv(f'{os.getcwd()}/data/Herald_time_csv/{file+".csv"}', index=False)
    print(f"{file} 까지 완료되었습니다.")
    
    
# "통합" 전령 시간대 데이터프레임 생성
import os
import pandas as pd
import numpy as np
from analysis.remerge import Remerge

directory = os.getcwd() + '/data/rawdata'
file_list = os.listdir(directory)
file_list = [file[:-18] for file in file_list if "Raw" in file]
csv_list = [file+'.csv' for file in file_list]

df_list = []

for csv in csv_list:
    
    test = pd.read_csv(f'{os.getcwd()}/data/Herald_time_csv/{csv}')
    df_list.append(test)
    
df = pd.concat(df_list, ignore_index=True)
df.to_csv(f'{os.getcwd()}/data/Herald_time_csv/total2.csv', index=False)


# 행 길이 체크
directory = os.getcwd() + '/data/rawdata'
file_list = os.listdir(directory)
file_list = [file[:-18] for file in file_list if "Raw" in file]

for file in file_list:
    t = pd.read_csv(f'{os.getcwd()}/data/Herald_time_csv/{file+".csv"}')
    print(file, ':', len(t))
    
# feature_engineering.py 모듈 적용
import feature_engineering as fe

for i in range(len(dfs)):
    dfs[i] = fe.make_features(dfs[i])