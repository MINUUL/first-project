#!/usr/bin/env python
# coding: utf-8

# ### 전기차 충전소 데이터 가져오기
# 
# ev monitor 사이트에서 받아온 전기차 충전기 데이터.
# 운영기관	충전소	충전기ID	충전기타입	지역	시군구	주소	이용가능시간	이용자 제한	충전용량
# 
# 용인시의 충전소 데이터를 추출한 다음 동별로 groupby해서 동별 충전소 지수를 반영할 예정

# In[2]:

from geopy.geocoders import Nominatim
import pandas as pd
import numpy as np
import re

# In[3]:
    
#** 주소를 좌표로 바꿔주는 함수 **
# 정확하지 않은 주소들이 있어서 정규표현식으로 처리를 하던가 해야 할 것 같습니다..
# 시 + 도 + 도로명으로 일관되게 바꾸었는데 단순화하니까 중복되는 좌표가 너무 많습니다.

def geocoding(address):
    address = address.split('(')[0]
    address = address.split(',')[0]
    geolocoder = Nominatim(user_agent = 'South Korea', timeout=None)
    geo = geolocoder.geocode(address)
    try:
        crd = {"lat": str(geo.latitude), "lng": str(geo.longitude)}
    except :
        print(address)
        return np.nan
    return crd


# In[4]:
geocoding('수지구 만현로 48')

path = 'C:/2022빅콘테스트_데이터분석리그_데이터분석분야_챔피언부문_데이터셋_220908/자료/'


# In[ ]:

#ev_cg.to_csv(path + '용인시 충전소.csv',encoding= 'EUC-KR')


# In[ ]:
## 도로명 뽑아오는 헬프함수
## 도로명 덩어리 슬라이스

def doro_slice(s):
    s = s.split('로')[0] + '로'
    return s


def help_func(lst):

    
    if len(lst) == 0: return np.nan
    
    if "로" in lst[0]:
        return doro_slice(lst[0])
    
    else : 
        lst = lst[1:]
        return help_func(lst)
 
#주소 정리 함수
def adj_add(add):
    add = add.split()
    cnt = 1
    z = 0
    for i in range(len(add)):
        
        if '로' in i or '길' in add[i] :
            add = add[0:i + 1]
            break
        cnt += 1
    
    # 멀쩡한 부분
    tmp3 = add[-3]
    # 도로명 포함된 부분
    tmp1 = add[-2]
    # 도로명 이후 부분
    tmp2 = add[-1]
    
    tmp1 = tmp1.split['로'][0] + '로 ' + tmp1.split['로'][1] 
    tmp1 = re.sup(' ','',tmp1)

    if '(' in tmp2 or ')' in tmp2:
        tmp2 = tmp2.split('[(,)]')[0]
        
    tmp2 = re.sup('[가-힣]','',tmp2[0:5])
        

    
    res = ''
    for j in range(len(tmp3)):
        res = res + tmp3[i]
    
    if '길' in tmp2:
        tmp1 = re.sup(' ','', tmp1 + tmp2)
        z = 1
        res = res + ' ' + tmp1
    else:
        res = res + tmp1 + tmp2
    
    return res


 
## 도로명, 법정동 테이블
table = pd.read_table(path + 'rnaddrkor_gyunggi.txt', header = None, encoding = 'CP949',sep = '|')
table = table[table[0].str.contains('^4146')]
table = table[[4,10]].drop_duplicates([4,10])

table[10] = list(map(doro_slice,table[10]))

table = table[[4,10]].drop_duplicates([4,10])
table.columns = ['읍면동', '도로명']

 
## 법정동, 행정동 테이블

emd_table = pd.read_csv(path + '/공공데이터/jscode20210401/KIKmix.20210401.csv')
emd_table['행정동코드'] = emd_table['행정동코드'].astype(str)
emd_table = emd_table[emd_table['행정동코드'].str.contains('^4146')][['읍면동명','동리명']].dropna()

 
# 기존 충전소 전처리
ev_cg = pd.read_excel(path + '충전소 리스트.xlsx')
ev_cg.head()


cols = ev_cg.iloc[1,:]
ev_cg = ev_cg[2:]

ev_cg.columns = cols
ev_cg.head()

ev_cg = ev_cg[ev_cg['시군구'].str.contains('^용인시',na = False)]

address = np.array(ev_cg['주소'])

def final_f(df, address_col = '주소' ,new_col = '도로명', mod = 0):
    global table
    address = np.array(df[address_col])
    
    doro = list(map(lambda x: x.split(),address))
    
    doroname = list(map(help_func,doro))
    
    if mod == 1:
        return doroname
    
    df[new_col] = doroname
    
    df = df.set_index(new_col).join(table.set_index('도로명')).reset_index()

    count = df.groupby(['읍면동']).size()
    
    count.name = 'count'
    
    return count

ev_cnt = final_f(df = ev_cg)

ev_cnt = emd_table.set_index('동리명').join(ev_cnt).dropna()
ev_cnt = ev_cnt.groupby(['읍면동명'])['count'].sum()

 
## 동별 아파트 세대 수
ap = pd.read_excel(path + '/공공데이터/20220918_아파트정보목록.xlsx')
ap['동'] = list(map(lambda x: x.split()[-1],ap['동']))
ap = ap.groupby(['동'])['세대수'].sum()
ap_cnt = emd_table.set_index('동리명').join(ap).dropna().groupby(['읍면동명']).sum()







 
## 주차장 수
## 다음에
# park = pd.read_csv(path + '/공공데이터/용인도시공사_주차장 정보_20220621.csv',encoding = 'CP949',thousands = ',')
# park['위 치'] = list(map(lambda x: re.sub('[(,)]',' ',x),park['위 치']))

# park_lst = park['위 치'][park['위 치'].str.contains('동')]
# park_lst.drop([29,30,31],inplace = True)
# park_lst = list(map(lambda x: pd.DataFrame(x.split())[pd.Series(x.split()).str.contains('동')][0] , park_lst))

# park['위 치'] = final_f(df = park,address_col = '위 치',mod = 1)
# park['위 치'][park['위 치'].isnull()] = park_lst
# park['주차면수(면)'] = park['주차면수(면)'].astype(float)

# park = park[['위 치', '주차면수(면)']].groupby(['위 치']).sum()

# park.join(table.set_index('도로명'))

# park_cnt.groupby(['읍면동']).sum()







 
