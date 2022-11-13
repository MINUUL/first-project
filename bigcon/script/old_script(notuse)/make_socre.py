import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob

def to_geo(df):
    
    def made(df,i):
        
        x = df['x'].iloc[i]
        y = df['y'].iloc[i]
        
        return f'POINT ({x} {y})'
    
    lst = []
    
    for i in range(len(df)):     
        lst.append(made(df,i))
    lst = gpd.GeoSeries.from_wkt(lst)
    
    
    
    df = gpd.GeoDataFrame(df,geometry = lst, crs="EPSG:4326")
    
    return df


path = 'C:/bigcon/'
#park_table = pd.read_excel(path + 'input/additional_dataset/parkigLotList (1).xlsx')
#park_table[['x','y']] = park_table[['경도','위도']]
#park_table = park_table[['주차장명','x','y','주차구획수']]

score_table = to_geo(park_table)
score_table

score_space = gpd.GeoDataFrame(score_table, geometry = score_table.buffer(0.01))

score_space.plot()
plt.show()

# 앱실행 횟수
app_data = []

for i in sorted(glob(path + 'input/basic_dataset/*.csv')):
    app_data.append(pd.read_csv(i))
    
df_res = app_data[0] ## 주거자 데이터
df_act = app_data[1] ## 용인시에서 활동한 활동지 기준 데이터
df_res_act = app_data[2] ## 용인시 거주자가 활동한 지역 데이터

df_res_act.drop(['mega_cd','mega_nm'], axis = 1, inplace = True)
df_res_act = df_res_act[df_res_act['ccw_cd'] == 4146]

all_df = pd.concat([df_res,df_act,df_res_act])
all_df.rename(columns={'cell_xcrd':'x','cell_ycrd':'y'}, inplace = True)
all_df = all_df[['x','y','count_cust']]
all_df = to_geo(all_df)

# 전기차 보유대수 둘 다 행정동 기준이라 어떤 특정 좁은 장소에 대한 수요량을 구하기는 어려움
# 일단 근처 앱실행 수를 기준으로 산정해보았습니다.
# 해야할 것: 가중치 정하고, 후보지 선정하고 OD매트릭스에 넣을 요소 정하기
# OD매트릭스에 넣을 수 있는건 앱실행수 세대수 주차가능수 주요건물
# 인구수

tmp = gpd.sjoin(score_space,all_df).groupby(['주차장명'])['count_cust'].sum()

#tmp.to_csv(path + 'input/additional_dataset/tmp.csv',encoding = 'CP949')
