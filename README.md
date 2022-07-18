# Frank-Wolfe-algorithm-for-Traffic-Assignment-Problem

## 中文介绍

作者：师乐为、廖鹏

本项目使用Python 3.9.7实现了Frank-Wolfe算法，用于求解要求用户均衡（User Equilibrium）的交通流分配问题（Traffic Assignment Problem）。并使用两个具体网络的网络结构和OD需求数据进行了求解，两个网络的相关数据来自以下的链接，项目中的Excel文件是预处理后的结果。

- SiouxFalls network [网络结构](https://github.com/bstabler/TransportationNetworks/blob/master/SiouxFalls/SiouxFalls_net.tntp)
- SiouxFalls network [OD需求](https://github.com/bstabler/TransportationNetworks/blob/master/SiouxFalls/SiouxFalls_trips.tntp)
- Chicago Sketch network [网络结构](https://github.com/bstabler/TransportationNetworks/blob/master/Chicago-Sketch/ChicagoSketch_net.tntp)
- Chicago Sketch network [OD需求](https://raw.githubusercontent.com/bstabler/TransportationNetworks/master/Chicago-Sketch/ChicagoSketch_trips.tntp)

本项目有以下几个要点：使用字典作为数据结构、最短路的求解使用Dijkstra算法、在大网络的求解中结合了多进程。

如果你对本项目有任何问题，欢迎联系我们。(leweishi@foxmail.com)

## English Introduction

Authors: Shi Lewei, Liao Peng

 The project implements Frank-Wolfe algorithm by Python 3.9.7. The algorithm aims to solve the Traffic Assignment Problem requiring user equilibrium. The network structure and OD demand data of two specific networks are used to apply the algorithm. The relevant data of the two networks are from the following links. The excel files in the project are the results of preprocessing.

- SiouxFalls network [structure](https://github.com/bstabler/TransportationNetworks/blob/master/SiouxFalls/SiouxFalls_net.tntp)
- SiouxFalls network [OD demand](https://github.com/bstabler/TransportationNetworks/blob/master/SiouxFalls/SiouxFalls_trips.tntp)
- Chicago Sketch network [structure](https://github.com/bstabler/TransportationNetworks/blob/master/Chicago-Sketch/ChicagoSketch_net.tntp)
- Chicago Sketch network [OD demand](https://raw.githubusercontent.com/bstabler/TransportationNetworks/master/Chicago-Sketch/ChicagoSketch_trips.tntp)

Some points in the project are listed below: using dictionary as the data structure, using Dijkstra algorithm to solve the shortest path problem, using the multiprocessing in solving the large network.

If you have any question about the project, welcome to connect us. (leweishi@foxmail.com)
