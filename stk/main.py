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
from comtypes import _midlSAFEARRAY
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
useStkEngine = True
Read_Scenario = False
stkRoot = Start_STK(useStkEngine, Read_Scenario)
stkRoot.UnitPreferences.SetCurrentUnit("DateFormat", "EpSec")

print("Creating scenario...")
if not Read_Scenario:
    stkRoot.NewScenario('StarLink')
scenario = stkRoot.CurrentScenario
scenario2 = scenario.QueryInterface(STKObjects.IAgScenario)
# scenario2.StartTime = '24 Sep 2020 16:00:00.00'
# scenario2.StopTime = '25 Sep 2020 16:00:00.00'


totalTime = time.time() - startTime
print("--- Scenario creation: {a:4.3f} sec\t\tTotal time: {b:4.3f} sec ---".format(a=totalTime, b=totalTime))
Time_Range = 3600 * 12  # Seconds
Time_Step = 3600  # Seconds
orbitNum = 24
satsNum = 40


G_sat = nx.Graph()
Target_Area = [['beijing', 39.92, 116.46]]
core_network_matrix = [['beijing', 39.92, 116.46], ['shanghai', 31.22, 121.48],
                       ['wulumuqi', 43.47, 87.41], ['lasa', 29.6, 91], ['hainan', 20.02, 110.35]]

Gateway_Load = np.zeros([50, 50])
# f = open('population.txt')
# lines = f.readlines()
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
    Creat_satellite(scenario, numOrbitPlanes=orbitNum, numSatsPerPlane=satsNum,
                    hight=620, Inclination=90)  # Starlink
    sat_list = stkRoot.CurrentScenario.Children.GetElements(STKObjects.eSatellite)
    Add_transmitter_receiver(sat_list)


def CreateNetworkX(G_Matrix):
    G_networkx = nx.Graph()
    for i in range(len(G_Matrix)):
        for j in range(len(G_Matrix)):
            if G_Matrix[i][j] > 0:
                G_networkx.add_edge(i, j, weight=1000 / (G_Matrix[i][j] + random.random() / 10))
            # if G_Matrix[i][j] == 1:
            #     G_networkx.add_edge(i, j)
    return G_networkx



def Get_Satellite_Network(current_time,slot):
    global Propagation_Delay_forward
    global Propagation_Delay_backward
    Adjacency_Matrix = np.zeros([len(sat_list), len(sat_list)])
    Delay_Matrix = np.zeros([len(sat_list), len(sat_list)])
    LoadSat_slot = np.zeros([len(sat_list),1]) # 用于存储卫星负载

    for sat_num, sat in enumerate(sat_list):
        now_sat_name = sat.InstanceName
        now_sat_transmitter = sat.Children.GetElements(
            STKObjects.eTransmitter)[0]  # 找到该卫星的发射机
        Set_Transmitter_Parameter(now_sat_transmitter, frequency=12, EIRP=20,
                                  DataRate=14)
        now_plane_num = int(now_sat_name.split('_')[0][3:])
        now_sat_num = int(now_sat_name.split('_')[1])
        access_backward = now_sat_transmitter.GetAccessToObject(
            Get_sat_receiver(
                sat_dic['Sat' + str(now_plane_num) + '_' + str(
                    (now_sat_num + 1) % satsNum)]))
        access_forward = now_sat_transmitter.GetAccessToObject(
            Get_sat_receiver(
                sat_dic['Sat' + str(now_plane_num) + '_' + str(
                    (now_sat_num - 1) % satsNum)]))

        if current_time == 0:
            access_backward = now_sat_transmitter.GetAccessToObject(
                Get_sat_receiver(
                    sat_dic['Sat' + str(now_plane_num) + '_' + str(
                        (now_sat_num + 1) % satsNum)]))
            access_forward = now_sat_transmitter.GetAccessToObject(
                Get_sat_receiver(
                    sat_dic['Sat' + str(now_plane_num) + '_' + str(
                        (now_sat_num - 1) % satsNum)]))
            Propagation_Delay_forward = Compute_Propagation_Delay(
                access_forward, current_time)
            Propagation_Delay_backward = Compute_Propagation_Delay(
                access_backward, current_time)

        if current_time == scenario2.StartTime:
            global EbN0_Min_forward_and_backward
            EbN0_Min_forward_and_backward = Compute_Min_EbN0(scenario2,
                                                               access_forward,
                                                               n=0)
        # print('EbN0_Min_forward_and_backward   ', EbN0_Min_forward_and_backward)
        EbN0_Min_forward_and_backward = Compute_Min_EbN0(scenario2,
                                                           access_forward, n=0)

        if now_sat_num == 0:
            Adjacency_Matrix[sat_num][
                now_plane_num * satsNum + now_sat_num + 1] = 1
            Adjacency_Matrix[sat_num][
                now_plane_num * satsNum + satsNum - 1] = 1
            Delay_Matrix[sat_num][
                now_plane_num * satsNum + now_sat_num + 1] = Propagation_Delay_forward
            Delay_Matrix[sat_num][
                now_plane_num * satsNum + satsNum - 1] = Propagation_Delay_backward


        elif now_sat_num == satsNum - 1:
            Adjacency_Matrix[sat_num][
                now_plane_num * satsNum + now_sat_num - 1] = 1
            Adjacency_Matrix[sat_num][now_plane_num * satsNum + 0] = 1
            Delay_Matrix[sat_num][
                now_plane_num * satsNum + now_sat_num - 1] = Propagation_Delay_backward
            Delay_Matrix[sat_num][
                now_plane_num * satsNum + 0] = Propagation_Delay_forward

        else:
            Adjacency_Matrix[sat_num][
                now_plane_num * satsNum + now_sat_num - 1] = 1
            Adjacency_Matrix[sat_num][
                now_plane_num * satsNum + now_sat_num + 1] = 1
            Delay_Matrix[sat_num][
                now_plane_num * satsNum + now_sat_num - 1] = Propagation_Delay_backward
            Delay_Matrix[sat_num][
                now_plane_num * satsNum + now_sat_num + 1] = Propagation_Delay_forward
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
                    Get_sat_receiver(
                        sat_dic['Sat' + str((now_plane_num - 1) % orbitNum) + '_' + str(
                            now_sat_num)]))
                Propagation_Delay_left = Compute_Propagation_Delay(
                    access_left, current_time)
                Adjacency_Matrix[sat_num][
                    (now_plane_num - 1) * satsNum + now_sat_num] = 1
                Delay_Matrix[sat_num][
                    (now_plane_num - 1) * satsNum + now_sat_num] = Propagation_Delay_left



            if now_plane_num != orbitNum - 1:
                # 计算右侧链路信息
                access_right = now_sat_transmitter.GetAccessToObject(
                    Get_sat_receiver(
                        sat_dic['Sat' + str((now_plane_num + 1) % orbitNum) + '_' + str(
                            now_sat_num)]))
                Propagation_Delay_right = Compute_Propagation_Delay(
                    access_right, current_time)
                Adjacency_Matrix[sat_num][
                    (now_plane_num + 1) * satsNum + now_sat_num] = 1
                Delay_Matrix[sat_num][
                    (now_plane_num + 1) * satsNum + now_sat_num] = Propagation_Delay_right
        LoadSat_slot[sat_num][0] = Sat_load[sat_num] # 将卫星负载存入 LoadSat_slot
    #
    np.savetxt("./data/" + str(n) + "Delay_Matrix_origin.txt", Delay_Matrix)
    # G_hop = nx.Graph()
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

    np.savetxt("./data/" + str(n) + "Adjacency_Matrix.txt", Adjacency_Matrix,
               fmt="%d")
    np.savetxt("./data/" + str(n) + "Delay_Matrix.txt", Delay_Matrix)
    print("  is ok ")
    return G, Adjacency_Matrix, Delay_Matrix,LoadSat_slot


def get_ground_link_delay(gateway_name, target_name, current_time):
    """
    计算地面站和目标区域之间的链路延迟。

    Args:
        gateway_name (str): 地面站的名称.
        target_name (str): 目标区域的名称.
        current_time (float): 当前时间 (STK Epoch Seconds).

    Returns:
        float: 地面链路延迟 (秒).
    """
    # 获取地面站对象
    gateway = stkRoot.GetObjectFromPath(f'Target/{gateway_name}')
    target = stkRoot.GetObjectFromPath(f'Target/{target_name}')

    # 确保地面站和目标区域对象都存在
    if gateway is None or target is None:
        print(f"Error: 地面站或目标区域不存在. Gateway: {gateway_name}, Target: {target_name}")
        return 0  # 返回默认值

    # 创建Access对象
    access = gateway.GetAccessToObject(target)
    access.ComputeAccess()

    # 获取Access数据提供器
    accessDP = access.DataProviders.Item('Link Information')
    accessDP2 = accessDP.QueryInterface(STKObjects.IAgDataPrvTimeVar)

    try:
        # 执行计算并获取传播延迟
        result = accessDP2.ExecSingleElements(current_time, ElementNames=["Propagation Delay"])
        propagation_delay = result.DataSets.GetDataSetByName('Propagation Delay').GetValues()[0]
        return propagation_delay
    except Exception as e:
        print(f"Error computing ground link delay: {e}")
        return 0  # 返回默认值


def get_ground_link_bandwidth(gateway_name, target_name):
    """
    获取地面站和目标区域之间的链路带宽。

    Args:
        gateway_name (str): 地面站的名称.
        target_name (str): 目标区域的名称.

    Returns:
        float: 地面链路带宽 (Mbps).
    """
    # 这里需要替换为实际的带宽获取逻辑
    # 由于STK本身不直接提供地面链路带宽的计算，这部分需要根据你的网络配置和模型来确定
    # 以下是一些可能的方案：
    # 1. 如果你在STK中定义了地面站之间的链路，并且设置了链路容量，你可以尝试通过STK对象模型获取。
    # 2. 如果地面链路带宽是固定的，你可以直接返回一个常量值。
    # 3. 如果你需要考虑地面网络的动态变化，你可能需要通过外部数据源（例如，网络监控工具）来获取带宽信息，
    #    并将其导入到你的Python脚本中。

    # 示例 (假设带宽是固定的):
    #  bandwidth = 1000  # Mbps
    #  return bandwidth

    # 示例 (从外部文件读取):
    try:
        df = pd.read_csv('ground_link_bandwidth.csv') # 假设地面链路带宽信息存储在 CSV 文件中
        bandwidth = df.loc[df['gateway'] == gateway_name, target_name].values[0]
        return bandwidth
    except Exception as e:
        print(f"Error getting ground link bandwidth: {e}")
        return 1000 # 返回默认值

if __name__ == '__main__':
    gateway_poseiton_and_load = functions.Gateway_Position_And_Load_Matrix(
        Gateway_Load)
    n = 0

    ground_dic = functions.Create_Ground_and_Target_Area(
        gateway_poseiton_and_load, scenario, stkRoot, core_network_matrix)

    sat_dic = functions.Create_Sat_Dic(sat_list, orbitNum, satsNum)

    tmp = []
    while (n <= Time_Range):
        current_time = scenario2.StartTime + n * Time_Step
        if current_time > scenario2.StartTime + Time_Range:
            break
        # 计算卫星位置
        print("calculate satellite position")
        sat_position = []
        for sat in tqdm(sat_list):
            s_lat, s_lon, s_alt = functions.Compute_Satellite_Position(
                current_time, sat)
            sat_position.append([sat.InstanceName, s_lat, s_lon, s_alt])
        print("calculate satellite position ok")
        G_sat, Adjacency_Matrix, Delay_Matrix, LoadSat_slot = Get_Satellite_Network(current_time,n) # 获取卫星网络拓扑和负载
        # nx.draw(G_sat, node_color="orange", edge_color="grey", with_labels=True, pos=nx.kamada_kawai_layout(G_sat))
        # plt.show()
        # 计算卫星负载
        Sat_load = functions.Get_Satellite_Load_Matrix(
            gateway_poseiton_and_load, n, sat_position, scenario2, stkRoot,
            orbitNum, satsNum)
        Sat_load = Sat_load.T
        np.savetxt("./sat_load/Sat_load" + str(n) + ".txt", Sat_load,
                   fmt="%d")
        # 地面链路状态
        # ground_delay = 0  # 假设的地面链路延迟获取函数  需要替换为实际的地面链路延迟获取方式
        # ground_bandwidth = 0  # 假设的地面链路带宽获取函数 需要替换为实际的地面链路带宽获取方式
        # link_available = True  # 假设的卫星-地面链路切换标记  需要替换为实际的链路可用性判断
        # 假设地面站和目标区域的名称
        gateway_name = 'Gateway_0_0'  # 替换为你的实际地面站名称
        target_name = 'CoreNetwork_0'  # 替换为你的实际目标区域名称

        ground_delay = get_ground_link_delay(gateway_name, target_name, current_time)
        ground_bandwidth = get_ground_link_bandwidth(gateway_name, target_name)
        link_available = True  # 假设地面链路始终可用，需要根据实际情况修改


        network_state = {
            "sat": {
                "delay": Delay_Matrix,  # 卫星链路延迟矩阵
                "load": LoadSat_slot,  # 卫星负载
                "available": True  # 假设卫星链路始终可用，需要根据实际情况修改
            },
            "ground": {
                "delay": ground_delay,  # 地面链路延迟
                "bandwidth": ground_bandwidth,  # 地面链路带宽
                "available": link_available  # 地面链路可用性
            }
        }
        print(network_state) # 打印网络状态信息
        n = n + 1
    print(tmp)
