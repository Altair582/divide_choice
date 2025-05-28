import math
import os
import random
from satellite_settings import *
from start_stk import Start_STK

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import matplotlib.pyplot as plt
import time

startTime = time.time()
import networkx as nx
from tqdm import tqdm
import pandas as pd
import numpy as np
from comtypes.gen import STKObjects, STKUtil, AgStkGatorLib
from comtypes.client import CreateObject, GetActiveObject, GetEvents, CoGetObject, ShowEvents
from ctypes import *
import comtypes.gen._00020430_0000_0000_C000_000000000046_0_2_0
from comtypes import GUID
from comtypes import helpstring
from comtypes import COMMETHOD
from comtypes import dispid
from ctypes.wintypes import VARIANT_BOOL
from ctypes import HRESULT
from comtypes import BSTR
from comtypes.automation import VARIANT
from comtypes.automation import _midlSAFEARRAY
from comtypes import CoClass
from comtypes import IUnknown
import comtypes.gen._00DD7BD4_53D5_4870_996B_8ADB8AF904FA_0_1_0
import comtypes.gen._8B49F426_4BF0_49F7_A59B_93961D83CB5D_0_1_0
from comtypes.automation import IDispatch
import comtypes.gen._42D2781B_8A06_4DB2_9969_72D6ABF01A72_0_1_0
from comtypes import DISPMETHOD, DISPPROPERTY, helpstring
import functions
"""
SET TO TRUE TO USE ENGINE, FALSE TO USE GUI
"""
def initialize_stk_scenario():
    useStkEngine = True
    Read_Scenario = False
    stkRoot = Start_STK(useStkEngine, Read_Scenario)
    stkRoot.UnitPreferences.SetCurrentUnit("DateFormat", "EpSec")
    
    if not Read_Scenario:
        stkRoot.NewScenario('StarLink')
    scenario = stkRoot.CurrentScenario
    return stkRoot, scenario
scenario=initialize_stk_scenario()[1]
scenario2 = scenario.QueryInterface(STKObjects.IAgScenario)
# scenario2.StartTime = '24 Sep 2020 16:00:00.00'
# scenario2.StopTime = '25 Sep 2020 16:00:00.00'


totalTime = time.time() - startTime
print("--- Scenario creation: {a:4.3f} sec\t\tTotal time: {b:4.3f} sec ---".format(a=totalTime, b=totalTime))
Time_Range = 3600*12  # Seconds
Time_Step = 3600  # Seconds
orbitNum = 24
satsNum = 40

    
G_sat = nx.Graph()
Target_Area = [['beijing', 39.92, 116.46]]
core_network_matrix = [['beijing', 39.92, 116.46], ['shanghai', 31.22, 121.48], ['wulumuqi', 43.47, 87.41], ['lasa', 29.6, 91], ['hainan', 20.02, 110.35]]

Gateway_Load = np.zeros([50, 50])
#f = open('population.txt')
#lines = f.readlines()
with open('population.txt') as f:
    lines = f.readlines()
if len(lines) < 50:
    raise ValueError("文件行数不足 50 行，无法填充 Gateway_Load 数组。")
target_load_row = 0
for line in lines:
    list = line.strip('\n').split(' ')
    Gateway_Load[target_load_row, :] = list[0:50]
    target_load_row += 1


if not Read_Scenario:
    Creat_satellite(scenario, numOrbitPlanes=orbitNum, numSatsPerPlane=satsNum, hight=620, Inclination=90)  # Starlink
    sat_list = stkRoot.CurrentScenario.Children.GetElements(STKObjects.eSatellite)
    Add_transmitter_receiver(sat_list)


def CreateNetworkX(G_Matrix):
    G_networkx = nx.Graph()
    for i in range(len(G_Matrix)):
        for j in range(len(G_Matrix)):
            if G_Matrix[i][j] > 0:
                G_networkx.add_edge(i, j, weight=1000 / (G_Matrix[i][j] + random.random()/10))
            # if G_Matrix[i][j] == 1:
            #     G_networkx.add_edge(i, j)
    return G_networkx

def Get_Satellite_Network(current_time):
    global Propagation_Delay_forward
    global Propagation_Delay_backward
    Adjacency_Matrix = np.zeros([len(sat_list), len(sat_list)])
    Delay_Matrix = np.zeros([len(sat_list), len(sat_list)])

    for sat_num, sat in enumerate(sat_list):
        now_sat_name = sat.InstanceName
        now_sat_transmitter = sat.Children.GetElements(STKObjects.eTransmitter)[0]  # 找到该卫星的发射机
        Set_Transmitter_Parameter(now_sat_transmitter, frequency=12, EIRP=20, DataRate=14)
        now_plane_num = int(now_sat_name.split('_')[0][3:])
        now_sat_num = int(now_sat_name.split('_')[1])
        access_backward = now_sat_transmitter.GetAccessToObject(
            Get_sat_receiver(sat_dic['Sat' + str(now_plane_num) + '_' + str((now_sat_num + 1) % satsNum)]))
        access_forward = now_sat_transmitter.GetAccessToObject(
            Get_sat_receiver(sat_dic['Sat' + str(now_plane_num) + '_' + str((now_sat_num - 1) % satsNum)]))

        if current_time==0:
            access_backward = now_sat_transmitter.GetAccessToObject(
                Get_sat_receiver(sat_dic['Sat' + str(now_plane_num) + '_' + str((now_sat_num + 1) % satsNum)]))
            access_forward = now_sat_transmitter.GetAccessToObject(
                Get_sat_receiver(sat_dic['Sat' + str(now_plane_num) + '_' + str((now_sat_num - 1) % satsNum)]))
            Propagation_Delay_forward = Compute_Propagation_Delay(access_forward, current_time)
            Propagation_Delay_backward = Compute_Propagation_Delay(access_backward, current_time)

        if current_time == scenario2.StartTime:
            global EbN0_Min_forward_and_backward
            EbN0_Min_forward_and_backward = Compute_Min_EbN0(scenario2, access_forward, n=0)
        # print('EbN0_Min_forward_and_backward   ', EbN0_Min_forward_and_backward)
        EbN0_Min_forward_and_backward = Compute_Min_EbN0(scenario2, access_forward, n=0)

        if now_sat_num == 0:
            Adjacency_Matrix[sat_num][now_plane_num * satsNum + now_sat_num + 1] = 1
            Adjacency_Matrix[sat_num][now_plane_num * satsNum + satsNum - 1] = 1
            Delay_Matrix[sat_num][now_plane_num * satsNum + now_sat_num + 1] = Propagation_Delay_forward
            Delay_Matrix[sat_num][now_plane_num * satsNum + satsNum - 1] = Propagation_Delay_backward


        elif now_sat_num == satsNum - 1:
            Adjacency_Matrix[sat_num][now_plane_num * satsNum + now_sat_num - 1] = 1
            Adjacency_Matrix[sat_num][now_plane_num * satsNum + 0] = 1
            Delay_Matrix[sat_num][now_plane_num * satsNum + now_sat_num - 1] = Propagation_Delay_backward
            Delay_Matrix[sat_num][now_plane_num * satsNum + 0] = Propagation_Delay_forward

        else:
            Adjacency_Matrix[sat_num][now_plane_num * satsNum + now_sat_num - 1] = 1
            Adjacency_Matrix[sat_num][now_plane_num * satsNum + now_sat_num + 1] = 1
            Delay_Matrix[sat_num][now_plane_num * satsNum + now_sat_num - 1] = Propagation_Delay_backward
            Delay_Matrix[sat_num][now_plane_num * satsNum + now_sat_num + 1] = Propagation_Delay_forward
        # print(Adjacency_Matrix[sat_num])
        Propagation_Delay_left = 0
        Propagation_Delay_right = 0
        # 创建空列表存放卫星每个时刻
        X_List = []
        Y_List = []
        Z_List = []
        Time_List = []
        # 获得每个每个卫星的纬度
        Lat = sat_position[sat_num][1]
        # print(now_sat_name, "   纬度:  ", Lat)
        if float(Lat) > 75 or float(Lat) < -75:
            # print("卫星纬度大于 75，无星间链路")
            pass
        else:
            # 计算左侧链路信息
            if now_plane_num != 0:
                access_left = now_sat_transmitter.GetAccessToObject(
                    Get_sat_receiver(sat_dic['Sat' + str((now_plane_num - 1) % orbitNum) + '_' + str(now_sat_num)]))
                Propagation_Delay_left = Compute_Propagation_Delay(access_left, current_time)
                Adjacency_Matrix[sat_num][(now_plane_num - 1) * satsNum + now_sat_num] = 1
                Delay_Matrix[sat_num][(now_plane_num - 1) * satsNum + now_sat_num] = Propagation_Delay_left



            if now_plane_num != orbitNum - 1:
                # 计算右侧链路信息
                access_right = now_sat_transmitter.GetAccessToObject(
                    Get_sat_receiver(sat_dic['Sat' + str((now_plane_num + 1) % orbitNum) + '_' + str(now_sat_num)]))
                Propagation_Delay_right = Compute_Propagation_Delay(access_right, current_time)
                Adjacency_Matrix[sat_num][(now_plane_num + 1) * satsNum + now_sat_num] = 1
                Delay_Matrix[sat_num][(now_plane_num + 1) * satsNum + now_sat_num] = Propagation_Delay_right

    #
    np.savetxt("./data/" + str(n) + "Delay_Matrix_origin.txt", Delay_Matrix)
    #G_hop = nx.Graph()
    G_hop = CreateNetworkX(Delay_Matrix)
    hop = np.zeros((960, 960))
    for i in range(960):
        for j in range(960):
            hop[i][j] = nx.shortest_path_length(G_hop, i, j)
    np.savetxt('./data/' + str(n) + 'hop.txt', hop, fmt='%d')

    G = CreateNetworkX(Delay_Matrix)
    for i in range(Delay_Matrix.shape[0]):
        for j in range(Delay_Matrix.shape[0]):
            Delay_Matrix[i][j] = nx.dijkstra_path_length(G, i, j)


    np.savetxt("./data/"+str(n) + "Adjacency_Matrix.txt", Adjacency_Matrix, fmt="%d")
    np.savetxt("./data/"+str(n) + "Delay_Matrix.txt", Delay_Matrix)
    print("  is ok ")
    return G,Adjacency_Matrix,Delay_Matrix


if __name__ == '__main__':
    gateway_poseiton_and_load = functions.Gateway_Position_And_Load_Matrix(Gateway_Load)
    n = 0

    ground_dic = functions.Create_Ground_and_Target_Area(gateway_poseiton_and_load,scenario,stkRoot,core_network_matrix)

    sat_dic = functions.Create_Sat_Dic(sat_list, orbitNum, satsNum)
    
    stkRoot, scenario = initialize_stk_scenario()
    
    import csv
    terrestrial_links = [
        {'Source': 'GS1', 'Target': 'GS2', 'Delay_ms': 5, 'Bandwidth_Mbps': 1000, 'LossRate': 0.001},
        {'Source': 'GS1', 'Target': 'GS3', 'Delay_ms': 8, 'Bandwidth_Mbps': 500, 'LossRate': 0.002},
        # 添加更多地面链路...
    ]
    with open('terrestrial_links.csv', 'w', newline='') as csvfile:
        fieldnames = ['Source', 'Target', 'Delay_ms', 'Bandwidth_Mbps', 'LossRate']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(terrestrial_links)
    
    # ... existing code ...
    gateway_poseiton_and_load = functions.Gateway_Position_And_Load_Matrix(Gateway_Load)
    n = 0

    # 定义仿真参数
    start_time = '01 Jan 2023 00:00:00.000'
    end_time = '01 Jan 2023 01:00:00.000'  # 1小时仿真
    time_step = 60  # 60秒时间步长
    
    # 定义地面站和卫星列表
    ground_stations = ['GS1', 'GS2', 'GS3']  # 替换为实际地面站名称
    satellites = ['Sat0_0', 'Sat0_1', 'Sat1_0']  # 替换为实际卫星名称
    
    # 输出文件路径
    output_file = 'dynamic_satellite_links.csv'
    
    # 调用函数计算链路参数
    from functions import get_dynamic_link_data
    get_dynamic_link_data(scenario, stkRoot, start_time, end_time, 
                         time_step, ground_stations, satellites, output_file)
    
    print(f"链路参数已保存到 {output_file}")
    tmp = []
    while(n <= Time_Range):
        current_time = scenario2.StartTime + n * Time_Step
        if current_time > scenario2.StartTime + Time_Range:
            break
        # 计算卫星位置
        print("calculate satellite position")
        sat_position = []
        for sat in tqdm(sat_list):
            s_lat, s_lon, s_alt = functions.Compute_Satellite_Position(current_time, sat)
            sat_position.append([sat.InstanceName, s_lat, s_lon, s_alt])
        print("calculate satellite position ok")
        G_sat,Adjacency_Matrix,Delay_Matrix = Get_Satellite_Network(current_time)
        # nx.draw(G_sat, node_color="orange", edge_color="grey", with_labels=True, pos=nx.kamada_kawai_layout(G_sat))
        # plt.show()
        # 计算卫星负载
        Sat_load = functions.Get_Satellite_Load_Matrix(gateway_poseiton_and_load, n, sat_position, scenario2, stkRoot,
                                                       orbitNum, satsNum)
        Sat_load = Sat_load.T
        np.savetxt("./sat_load/Sat_load"+str(n)+".txt" , Sat_load, fmt = "%d")

        n = n + 1
    print(tmp)