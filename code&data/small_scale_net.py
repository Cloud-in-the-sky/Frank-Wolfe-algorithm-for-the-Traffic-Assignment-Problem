# -*- coding: utf-8 -*-
"""

@author: Shi Lewei, Liao Peng
"""

import pandas as pd
import numpy as np
import time
import copy


def read_data(path):
    # 读数据
    data = pd.read_excel(path)
    return data


def get_shortestpath(inode, tnode_list, network, index_table):
    # Dijkstra算法求一个起点到一系列终点的最短路
    S = []
    S_non = list(set(network['init_node']))
    # S_non1=copy.deepcopy(S_non)
    K = len(S_non)
    K1 = max(S_non)
    M = 1000000
    dist = dict.fromkeys(S_non, M)
    dist1 = dict.fromkeys(S_non, M)
    pred = dict.fromkeys(S_non, 0)
    dist[inode] = 0
    dist1[inode] = 0
    # s = time.process_time()
    while len(S) < K:
        min_node = min(dist1, key=lambda k: dist1[k])
        S.append(min_node)
        # 找到所有需要的最短路就停止运行，节省时间，因为S中的节点一定是最短路
        if set(tnode_list) <= set(S):
            break
        else:
            dist1.pop(min_node)
            S_non.remove(min_node)  # 更新两个集合
            s_i = index_table[min_node]
            if min_node < K1:
                e_i = index_table[min_node+1]
            else:
                e_i = len(network["init_node"])
            next_min_nodelist = network['term_node'][s_i:e_i]
            for index, node in enumerate(next_min_nodelist):  # 更新节点的d
                c = network['flow_time'][s_i+index]
                if dist[node] > dist[min_node] + c:
                    dist[node] = dist[min_node] + c
                    dist1[node] = dist[min_node] + c
                    pred[node] = min_node
    # 返回格式：[起点，终点，最短路，成本]
    result = []
    for tnode in tnode_list:
        shortest_path = []
        iinode = tnode
        while iinode != inode:
            shortest_path.append(iinode)
            iinode = pred[iinode]
        shortest_path.append(inode)
        shortest_path.reverse()
        result.append([inode, tnode, shortest_path, dist[tnode]])
    return result


def add_three_col(network, od):
    # 给network新增flow、auxiliary_flow、flow_time三列
    values = np.zeros(len(network["init_node"]))
    network["flow"] = copy.deepcopy(values)  # 第4列
    network["auxiliary_flow"] = copy.deepcopy(values)  # 第5列
    values_free_flow_time = copy.deepcopy(network["free_flow_time"])
    network["flow_time"] = values_free_flow_time  # 第6列
    # 给od新增一列min_cost表示OD间最小成本
    values = np.zeros(len(od["init_node"]))
    od["min_cost"] = copy.deepcopy(values)


def construct_index_table(network):
    # 原有的network和OD数据是前向星型表示法
    # 增加索引表减少在最短路中寻找其上的边的时间
    init_node_list = list(set(network["init_node"]))
    init_node_list.insert(0, 0)
    for i in range(1, len(init_node_list)):
        for j in range(len(network["init_node"])):
            if network["init_node"][j] == i:
                init_node_list[i] = j
                break
    return init_node_list


def fun(x, *args):
    # 作为minimize_scalar中的函数形式传入
    sum = 0
    for i in range(len(args[0])):
        c = args[3][i]*args[3][i]*args[3][i]*args[3][i]
        d = (args[1][i]+x*(args[2][i]-args[1][i]))*(args[1][i]+x*(args[2][i]-args[1][i]))*(args[1][i]+x*(args[2][i]-args[1][i])) \
            *(args[1][i]+x*(args[2][i]-args[1][i]))*(args[1][i]+x*(args[2][i]-args[1][i]))
        item = args[0][i]*(args[1][i]+x*(args[2][i]-args[1][i])+0.03/c*d)
        sum += item
    return sum


def line_search(network):
    # 求解步长alpha
    from scipy.optimize import minimize_scalar
    root_info = minimize_scalar(fun, bounds=(0, 1), method="bounded",
                                args=(network["free_flow_time"],
                                      network["flow"],
                                      network["auxiliary_flow"],
                                      network["capacity"]))
    return root_info.x


def initialization(network, od, index_table, start_term_rel):
    # 初始化得到一个最初的结果，将边上的flow直接设为auxiliary_flow
    # 传入的index_table是index_table_network
    # 小网络中没有用到多进程求解最短路
    all_shorest_path = []
    for item in start_term_rel:
        result = get_shortestpath(item["start_node"],
                                  item["term_node_list"], network, index_table)
        all_shorest_path += result
    for i in range(len(od["init_node"])):

        shortest_path = all_shorest_path[i][2]
        min_cost = all_shorest_path[i][3]

        od["min_cost"][i] = min_cost
        # 更新这条最短路上所有边的流量并计算新的通行时间
        for k in range(len(shortest_path)-1):
            index_s = index_table[int(shortest_path[k])]
            if int(shortest_path[k]) < start_term_rel[-1]["start_node"]:
                index_e = index_table[int(shortest_path[k])+1]
            else:
                index_e = len(network["init_node"])
            for j in range(index_s, index_e):
                # 如果终点相同，则找到了所需的边
                if network["term_node"][j] == int(shortest_path[k+1]):
                    # 给最短路上的边分配流量
                    network["auxiliary_flow"][j] += od["demand"][i]
                    break

    network["flow"] = copy.deepcopy(network["auxiliary_flow"])
    # 根据更新后的流量更新通行时间
    import math
    for i in range(len(network["init_node"])):
        network["flow_time"][i] = network["free_flow_time"][i] * \
        (1 + 0.15*math.pow(network["flow"][i]/network["capacity"][i], 4))
    print("initialization done")


def FW_algorithm(network, od):
    # FW算法
    import time
    add_three_col(network, od)
    index_table_network = construct_index_table(network)
    index_table_od = construct_index_table(od)
    delta = 100
    delta_list = []
    alpha_list = []
    # 获得起终点关系的列表
    start_term_rel = []
    start_node_list = list(set(od['init_node']))
    for start in start_node_list:
        s_i = index_table_od[start_node_list.index(start)+1]
        if start < max(start_node_list):
            e_i = index_table_od[start_node_list.index(start)+2]
        else:
            e_i = len(od["init_node"])
        term_node_list = od['term_node'][s_i:e_i]
        dictionary = dict(start_node=start, term_node_list=term_node_list)
        start_term_rel.append(dictionary)
    # 初始化
    initialization(network, od, index_table_network, start_term_rel)
    len_od = len(od["init_node"])
    while abs(delta) > 0.0001:
        # 每次迭代auxiliary_flow初始为0
        s = time.process_time()
        values_init_flow = np.zeros(len(network["init_node"]))
        network["auxiliary_flow"] = copy.deepcopy(values_init_flow)
        all_shortest_path = []
        for item in start_term_rel:
            result = get_shortestpath(item["start_node"],
                                      item["term_node_list"],
                                      network, index_table_network)
            all_shortest_path += result
        for i in range(len_od):
            shortest_path = all_shortest_path[i][2]
            min_cost = all_shortest_path[i][3]
            od["min_cost"][i] = min_cost
            # 更新这条最短路上所有边的流量并计算新的通行时间
            for k in range(len(shortest_path)-1):
                index_s = index_table_network[int(shortest_path[k])]
                if int(shortest_path[k]) < len(index_table_network)-1:
                    index_e = index_table_network[int(shortest_path[k])+1]
                else:
                    index_e = len(network["init_node"])
                for j in range(index_s, index_e):
                    # 如果终点相同，则找到了所需的边
                    if network["term_node"][j] == int(shortest_path[k+1]):
                        # 给最短路上的边分配流量
                        network["auxiliary_flow"][j] += od["demand"][i]
                        break
        # 线搜索
        alpha = line_search(network)
        alpha_list.append(alpha)
        print(alpha)
        # 更新每条边的流量
        ak = np.array(network["flow"])
        ak += alpha*(np.array(network["auxiliary_flow"]) -
                     np.array(network["flow"]))
        network["flow"] = list(ak)
        # 根据更新后的流量更新通行时间
        import math
        for i in range(len(network["init_node"])):
            network["flow_time"][i] = network["free_flow_time"][i] * \
            (1 + 0.15*math.pow(network["flow"][i]/network["capacity"][i], 4))

        # 计算评价指标delta
        delta = (sum(np.array(od["min_cost"]) * np.array(od["demand"])) -
                 sum(np.array(network["flow"]) * np.array(network["flow_time"]))) / \
            sum(np.array(network["flow"])*np.array(network["flow_time"]))
        delta_list.append(float(delta))
        e = time.process_time()
        print(delta)
        print(e-s)
    return delta_list, alpha_list


def save_data(path, network, delta_list, alpha_list):
    # 保存结果数据
    writer = pd.ExcelWriter(path)
    pd.DataFrame(network).to_excel(writer, sheet_name='result')
    pd.DataFrame(delta_list).to_excel(writer, sheet_name='delta')
    pd.DataFrame(alpha_list).to_excel(writer, sheet_name='alpha')
    writer.save()


if __name__ == '__main__':
    # 程序入口
    network_1 = read_data('network1.xlsx')
    od_1 = read_data('od_data1.xlsx')
    # 将demand为0的OD对过滤掉
    od_1_new = od_1[od_1["demand"] > 0]
    # 将DataFrame转化为dictionary格式
    network_dict_1 = network_1.to_dict('list')
    od_dict_1 = od_1_new.to_dict('list')

    start = time.process_time()
    delta_list_1, alpha_list_1 = FW_algorithm(network_dict_1, od_dict_1)
    end = time.process_time()
    save_data("result1-1.xlsx",
              network_dict_1, delta_list_1, alpha_list_1)
    # 记录总占用CPU时间
    print("总占用CPU时间为：")
    print(end - start)
    print("Done")
