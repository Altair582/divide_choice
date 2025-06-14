import time
import random
import networkx as nx
import numpy as np
import nsga2
import matplotlib.pyplot as plt
import P_T_time
import linkcost_compute
import near_core_sat
import NF
# import draw_result
def CAPITAL(COST_AMF, COST_SMF, COST_UPF, AMF_NUMBER, SMF_NUMBER, UPF_NUMBER):
    return COST_AMF * AMF_NUMBER + COST_SMF * SMF_NUMBER + COST_UPF * UPF_NUMBER


def F(AMF_NUMBER, SMF_NUMBER, UPF_NUMBER, amf_location, smf_location,
      upf_location):
    result_24 = []
    result_24_time = []
    result_24_linkcost = []
    # result_24_rel = []
    linkcostXn, linkcostN2, linkcostSr, linkcostSession = [],[],[],[]
    capital = CAPITAL(COST_AMF, COST_SMF, COST_UPF, AMF_NUMBER, SMF_NUMBER, UPF_NUMBER)
    for slot in range(12):
        aaa = time.time()
        if ground_CN:
            amf_location = core_location_sat_slot[slot]
            smf_location = amf_location
            upf_location = smf_location
        sat_link_index,sat_link_index_for_ns3  = P_T_time.compute_sat_link_index(delay_slot[slot], amf_location, smf_location, upf_location,core_location_sat_slot[slot])
        np.savetxt("./order_Nsga/"+str(slot)+"order.txt", sat_link_index_for_ns3,fmt= "%d")
        tptime = P_T_time.TransTime_PTime(LoadSat_slot[slot], delay_slot[slot], AMF_NUMBER, SMF_NUMBER, UPF_NUMBER, amf_location, smf_location,
                                 upf_location,
                                 resource_sat,core_location_sat_slot[slot],sat_link_index)
        linkcost, reliability,linkcostXn_tmp,linkcostN2_tmp,linkcostSr_tmp,linkcostSession_tmp = linkcost_compute.LinkCost(LoadSat_slot[slot], delay_slot[slot], hop_slot[slot], LINK_COST_3, LINK_COST_4, LINK_COST_5,
                                         amf_location, smf_location, upf_location,core_location_sat_slot[slot],sat_link_index)

        opl = _mu * capital + _nu * tptime + _lambda * linkcost
        result_24.append(opl)
        result_24_time.append(tptime)
        result_24_linkcost.append(linkcost)
        # result_24_rel.append(reliability)
        linkcostXn.append(linkcostXn_tmp)
        linkcostN2.append(linkcostN2_tmp)
        linkcostSr.append(linkcostSr_tmp)
        linkcostSession.append(linkcostSession_tmp)
        # print('compute fitness ok', time.time() - aaa)

    result = sum(result_24)/12
    result_24_time_mean = sum(result_24_time)/12
    result_24_linkcost_mean = sum(result_24_linkcost) / 12
    # result_24_rel_mean = sum(result_24_rel) / 24
    # print(result_24_time_mean, result_24_linkcost_mean, result_24_rel_mean,result)
    #
    # print('capital', capital, 'time', tptime, 'linkcost', linkcost,'reliability ',reliability,'\nopl', opl)
    # qqq.append(result_24_time_mean)
    # qq2.append(result_24_linkcost_mean)
    # print(min(qqq), max(qqq), min(qq2), max(qq2))
    print("linkcostXn",linkcostXn)
    print("linkcostN2",linkcostN2)
    print("linkcostSr",linkcostSr)
    print("linkcostSession",linkcostSession)
    # print( _mu * capital, _nu * time + _lambda * (linkcost - zeta * reliability))
    # return _mu * capital, _nu * result_24_time_mean + _lambda * result_24_linkcost_mean - zeta * result_24_rel_mean, result, capital, result_24_time, result_24_linkcost, result_24_rel
    return _mu * capital, _nu * result_24_time_mean + _lambda * result_24_linkcost_mean, result, capital, result_24_time, result_24_linkcost

def translateDNA(l_amf, l_smf, l_upf):
    amf = np.zeros((len(l_amf), 960))
    smf = np.zeros((len(l_smf), 960))
    upf = np.zeros((len(l_upf), 960))
    for i in range(len(l_amf)):
        amf[i][int(l_amf[i])] = 1
    for i in range(len(l_smf)):
        smf[i][int(l_smf[i])] = 1
    for i in range(len(l_upf)):
        upf[i][int(l_upf[i])] = 1
    return amf, smf, upf
def calculate_link_cost(deployment, deployment_type, max_amf=54, max_smf=54, max_upf=54):
    """Calculate link cost and other metrics for a deployment."""
    amf_list = list(map(int, [x for x in deployment[:max_amf] if x != 999]))
    smf_list = list(map(int, [x for x in deployment[max_amf:max_amf + max_smf] if x != 999]))
    upf_list = list(map(int, [x for x in deployment[max_amf + max_smf:] if x != 999]))

    amf, smf, upf = translateDNA(amf_list, smf_list, upf_list)
    capital, performance, result, cap, tptime, linkcost = F(len(amf_list), len(smf_list), len(upf_list), amf, smf, upf)

    print(f"{deployment_type} Deployment:")
    print(f"AMF: {amf_list}")
    print(f"SMF: {smf_list}")
    print(f"UPF: {upf_list}")
    print(f"Link Cost: {sum(linkcost) / 12}")
    print(f"Capital: {capital}")
    print(f"Performance: {performance}")
    print(f"Result: {result}")
    print("------------")

    return {
        "deployment_type": deployment_type,
        "amf_list": amf_list,
        "smf_list": smf_list,
        "upf_list": upf_list,
        "link_cost": sum(linkcost) / 12,
        "capital": capital,
        "performance": performance,
        "result": result
    }
def get_nf_count(deployment, max_amf=54, max_smf=54, max_upf=54):
    amf_count = sum(1 for x in deployment[:max_amf] if x != 999)
    smf_count = sum(1 for x in deployment[max_amf:max_amf + max_smf] if x != 999)
    upf_count = sum(1 for x in deployment[max_amf + max_smf:] if x != 999)
    return amf_count + smf_count + upf_count
qqq = []
qq2 = []
qq1 = []
# 业务3 用户XN切换 amf smf upf 0.7
LINK_COST_3 = [160, 554 , 142, 73, 294, 139]
# 业务4 用户N2切换 amf 0.2
LINK_COST_4 = [1630, 374, 191, 1786, 678, 381, 142, 73, 167, 658, 118, 352, 156, 73, 100, 102, 126]
# 业务5 业务请求 amf 0.2
LINK_COST_5 = [166, 365, 142, 73, 167, 1262, 118]
# 会话建立
LINK_COST_6 = [238,811,308,140,235,169,112,1057,125,270,118,472,142,73,167]
COST_AMF = 300
COST_SMF = 200
COST_UPF = 400
_mu = 1
_nu = 104
_lambda = 1
compute_nsga2 = False
Test_nsga2 = True
ground_CN = False
youzhuangtai = False
P_S = False
C_S = True

resource_sat = np.loadtxt('./resource.txt')


amf_process_time = []
smf_process_time = []
upf_process_time = []
for i in range(960):
    amf_process_time.append(NF.AMF_ProcessTime(resource_sat[i]))
    smf_process_time.append(NF.SMF_ProcessTime(resource_sat[i]))
    upf_process_time.append(NF.UPF_ProcessTime(resource_sat[i]))

delayMean = np.loadtxt('./delayMean.txt')
LoadSat_slot = []
delay_slot = []
hop_slot = []
core_location_sat_slot = []
for i in range(12):
    LoadSat = np.loadtxt('../stk/sat_load/Sat_load' + str(i) + '.txt').T.reshape(1, -1)[0]
    LoadSat = [int(max(x / 100000, 1)) for x in LoadSat]
    a = sum(LoadSat)
    # print(a)
    LoadSat = [x / a for x in LoadSat]
    LoadSat_slot.append(LoadSat)
    delay_slot.append(np.loadtxt('../stk/data/' + str(i) + 'Delay_Matrix.txt'))
    hop_slot.append(np.loadtxt('../stk/data/' + str(i) + 'hop.txt'))
    tmp = near_core_sat.near_core_sat[i]
    core_location_sat = np.zeros((len(tmp), 960))
    for i in range(len(tmp)):
        core_location_sat[i][int(tmp[i])] = 1
    core_location_sat_slot.append(core_location_sat)



if Test_nsga2:

    tptime = []
    linkcost = []
    reliability = []
    capital = []
    opl = np.loadtxt('./opl.txt')
    print(len(opl))
    amf = opl[len(opl)-10][0:54]
    smf = opl[len(opl)-10][54:108]
    upf = opl[len(opl)-10][108:162]

    amf = list(map(int, list(filter(lambda x: x != 999, amf))))
    smf = list(map(int, list(filter(lambda x: x != 999, smf))))
    upf = list(map(int, list(filter(lambda x: x != 999, upf))))
    nf_counts = [get_nf_count(deployment) for deployment in opl]
    print(amf, len(amf))
    print( smf, len(smf))
    print( upf, len(upf))
    # 找到 NF 实例数量最多和最少的方案的索引
    ps_index = np.argmax(nf_counts)  # P-S 部署（最大 NF 数量）
    cs_index = np.argmin(nf_counts)  # C-S 部署（最小 NF 数量）
    ps_deployment = opl[ps_index]
    cs_deployment = opl[cs_index]
    if P_S:
        ps_result = calculate_link_cost(ps_deployment, "P-S")
        with open('linkcost of P-S.txt', 'w') as f:
            f.write(f"P-S 部署 (NF 实例数量: {nf_counts[ps_index]}):\n")
            f.write(f"AMF: {ps_result['amf_list']}\n")
            f.write(f"SMF: {ps_result['smf_list']}\n")
            f.write(f"UPF: {ps_result['upf_list']}\n")
            f.write(f"链路成本: {ps_result['link_cost']}\n")
            f.write(f"资本成本: {ps_result['capital']}\n")
            f.write(f"性能: {ps_result['performance']}\n")
            f.write(f"结果: {ps_result['result']}\n")
        print(f"P-S 部署 NF 数量: {nf_counts[ps_index]}")
    if C_S:
        cs_result = calculate_link_cost(cs_deployment, "C-S")
        with open('linkcost of C-S.txt', 'w') as f:
            f.write(f"C-S 部署 (NF 实例数量: {nf_counts[cs_index]}):\n")
            f.write(f"AMF: {cs_result['amf_list']}\n")
            f.write(f"SMF: {cs_result['smf_list']}\n")
            f.write(f"UPF: {cs_result['upf_list']}\n")
            f.write(f"链路成本: {cs_result['link_cost']}\n")
            f.write(f"资本成本: {cs_result['capital']}\n")
            f.write(f"性能: {cs_result['performance']}\n")
            f.write(f"结果: {cs_result['result']}\n")
        print(f"C-S 部署 NF 数量: {nf_counts[cs_index]}")    
    if Test_nsga2 and not P_S and not C_S:
        ns3_process_time_nfs = np.zeros([3,len(amf)])
        for i in range(len(amf)):
            ns3_process_time_nfs[0][i] = amf_process_time[int(amf[i])]
        for i in range(len(smf)):
            ns3_process_time_nfs[1][i] = smf_process_time[int(smf[i])]
        for i in range(len(upf)):
            ns3_process_time_nfs[2][i] = upf_process_time[int(upf[i])]
        np.savetxt('./ns3_process_time_nfs.txt',ns3_process_time_nfs)
    amf, smf, upf = translateDNA(amf, smf, upf)
    _, _, _, c, t, l = F(len(amf), len(smf), len(upf), amf, smf, upf)


    print(c)
    # print(t)
    # print(l)
    # print(r)


if __name__ == '__main__':
    if compute_nsga2:
        MAX_AMF_NUMBER = 54
        MAX_SMF_NUMBER = 54
        MAX_UPF_NUMBER = 54
        start_time = time.time()

        nsga2.suanfa(MAX_AMF_NUMBER, MAX_SMF_NUMBER, MAX_UPF_NUMBER)
        stop_time = time.time()
        print('time = ', stop_time - start_time)
