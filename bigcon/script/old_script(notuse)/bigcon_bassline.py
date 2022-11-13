#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
from glob import glob
import seaborn as sns
import matplotlib.pyplot as plt
import geopandas as gpd
from sklearn.cluster import KMeans





import warnings
warnings.filterwarnings(action='ignore')


# 한글 폰트 깨짐 방지
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

# 마이너스 부호 깨짐 현상 방지
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False



path = 'C:/2022빅콘테스트_데이터분석리그_데이터분석분야_챔피언부문_데이터셋_220908/'
app_data = []
lst = sorted(glob(path + '*.csv'))

for i in lst:
    app_data.append(pd.read_csv(i))
    
df_res = app_data[0] ## 주거자 데이터
df_act = app_data[1] ## 용인시에서 활동한 활동지 기준 데이터
df_res_act = app_data[2] ## 용인시 거주자가 활동한 지역 데이터


df_res.head()



## x좌표 127.074356 127.160350 
## Y좌표 37.251510 37.334835
df_res_act.describe()



for df in app_data:
    print('총 수요량:',df['count_cust'].sum())




np.unique(df_res['app_web'])


def mk_cnt(df,col): ## 앱실행 횟수 모음. 출퇴근 인구는 어떻게 처리할지?
    
    res = df.groupby(col)['count_cust'].sum()
    
    return res

groups = ['dow', 'time_zone','adng_nm']
df_grpby = []

for col in groups:
    df_grpby.append(mk_cnt(df_res, col) + mk_cnt(df_act,col))




sns.set_style('darkgrid')
sns.lineplot(data = df_grpby[0]) # 수요일이 제일 많고, 목,금요일 순으로 많음



sns.set_style('darkgrid')
sns.lineplot(data = df_grpby[1]) #time zone 4 압도적으로 많음


# 상위 10개 동 수요량. 유의미한 차이는 없는듯?
df_top10 = df_grpby[2].sort_values(ascending = False).head(10)
df_top10 = df_top10.reset_index()
sns.barplot(x = 'adng_nm', y = 'count_cust', data = df_top10)




df_top10 = df_grpby[2].sort_values(ascending = False).tail(10)
df_top10 = df_top10.reset_index()
sns.barplot(x = 'adng_nm', y = 'count_cust', data = df_top10)



map_all = gpd.read_file(path + '행정_읍면동/HangJeongDong_ver20210401.geojson') #출처:https://github.com/vuski/admdongkor



map_all.head()


map_all['adm_cd2'] = map_all['adm_cd2'].astype('float64')/100

youngin_EMD = map_all[map_all['sgg'].str.contains('^4146', na = False)]



youngin_EMD.head()


# # 시각화


def mk_yg(df): ## 앱실행 횟수 모음. 출퇴근 인구는 어떻게 처리할지?
    
    res = df.groupby(['adng_cd','adng_nm'])['count_cust'].sum()
    
    return res

youngin_grpsum = mk_yg(df_res) + mk_yg(df_act)
youngin_grpsum = youngin_grpsum.to_frame().reset_index('adng_nm')

sudo_grpsum = mk_yg(df_res_act)
sudo_grpsum = sudo_grpsum.to_frame().reset_index('adng_nm')
sudo_grpsum.head()
youngin_grpsum.head()


mapped = youngin_EMD.set_index('adm_cd2').join(youngin_grpsum).reset_index()
mapped

mapped_all = map_all.set_index('adm_cd2').join(sudo_grpsum).reset_index()
mapped_all



# 일렉트로닉 쇼크
## 결측값이 존재하는데, 동 구분 기준이 약간 다른듯.
## 밑의 지도와 비교해서 emd_cd 직접 수정 필요.
maps = mapped

maps['coords'] = maps['geometry'].apply(lambda x: x.representative_point().coords[:])
maps['coords'] = [coords[0] for coords in maps['coords']]

to_be_mapped = 'count_cust'
vmin, vmax = 0,max(maps['count_cust'])
fig, ax = plt.subplots(1, figsize=(30,30))


maps.plot(column=to_be_mapped, cmap='Blues', linewidth=0.8, ax=ax, edgecolors='0.8')
ax.set_title('electronic shock counts', fontdict={'fontsize':30})
ax.set_axis_off()

#for idx, row in maps.iterrows():
#    plt.annotate(s=row['adng_nm'], xy=row['coords'], horizontalalignment='center', color= 'k')
    
#sns.scatterplot(data = df_res_act,x = 'cell_xcrd', y = 'cell_ycrd',alpha = 0.5,size = 1)

sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm._A = []

cbar = fig.colorbar(sm, orientation='horizontal')

#전기차 충전소 위치 법정동에서 행정동으로 변환

# ev_cnt 변수는 다른 스크립트에서 전치리한 전기차 충전소 변수

mapped['coords'] = mapped['geometry'].apply(lambda x: x.representative_point().coords[:])
mapped['coords'] = [coords[0] for coords in mapped['coords']]

tmp = mapped.set_index('adng_nm').join(ev_cnt)

# 클러스트링을 위한 데이터로 변환

tmp = tmp[['count_cust', 'count']]
tmp = pd.concat([tmp,ap_cnt],axis = 1).fillna(0)

# tmp['x'] = list(map(lambda x: re.sub('[(,),,,]', '', x).split()[0],  tmp['coords'].astype('str')   ))
# tmp['y'] = list(map(lambda x: re.sub('[(,),,,]', '', x).split()[1],  tmp['coords'].astype('str')   ))

# tmp = tmp[['x','y','count']]





#mapped.sort_values(by = 'count_cust', ascending = False).head(5)



#sns.scatterplot(x = 'cell_xcrd',y = "cell_ycrd",data = df_res)


col_nm = tmp.columns

sample = np.array(tmp.copy()).astype(float)

def elbow(n_iter,df):
    sse = []
    for i in range(1,n_iter + 1):
        k_means = KMeans(init="k-means++", n_clusters=i, n_init=20)
        k_means.fit(df)
        sse.append(k_means.inertia_)
        
        
    plt.plot(range(1,n_iter + 1), sse, marker = 'o')
    plt.xlabel('클러스터 갯수')
    plt.ylabel('SSE')
    plt.show
    
elbow(20,sample)





k_means = KMeans(init="k-means++", n_clusters=4, n_init=20)
k_means.fit(sample)

### 다양한 기법을 사용하고 교집합 시킬건지, 다양한 변수를 각각 클러스트링 하고 교집합 시킬건지 고민해봐야됨
pd.options.display.float_format = '{:.5f}'.format
pd.reset_option('display.float_format')

k_means_labels = k_means.labels_

k_means_labels

k_means_cluster_centers = pd.DataFrame(k_means.cluster_centers_).copy()

k_means_cluster_centers.columns = ['앱 실행 수', '기존 전기차 충전소 수', '아파트 총 세대수']

emds = mapped['adng_nm']

groups = []

for i in range(4):
    s = emds[k_means_labels == i]
    groups.append(s)

np.array(groups)

    
    
    
#### 필요 없음...

fig = plt.figure(figsize=(6, 4))

# 레이블 수에 따라 색상 배열 생성, 고유한 색상을 얻기 위해 set(k_means_labels) 설정
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))

# plot 생성
to_be_mapped = sample[:,2]
ax = fig.add_subplot(1, 1,1)
mapped.plot(column=to_be_mapped, cmap='Blues', linewidth=0.8, ax=ax, edgecolors='0.8')



for k, col in zip(range(3), colors):
    my_members = (k_means_labels == k)

    # 중심 정의
    cluster_center = k_means_cluster_centers[k]

    # 중심 그리기
    ax.scatter(sample[my_members, 0], sample[my_members, 1], c = col)
    #ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=5)

ax.set_title('K-Means')
plt.show()



k_means.cluster_centers_


