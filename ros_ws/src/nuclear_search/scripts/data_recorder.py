#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import csv
import numpy as np
from std_msgs.msg import Float64
from nav_msgs.msg import Odometry

class DataRecorder:
    def __init__(self):
        rospy.init_node('data_recorder')
        
        # 真实放射源位置 (用于计算误差)
        self.true_source_pos = np.array([10.0, 12.0])
        
        # 数据存储
        self.log_file = open('search_data.csv', 'w', encoding='utf-8')
        self.writer = csv.writer(self.log_file)
        # 写入表头：迭代步数, 时间, 香农熵, 定位误差
        self.writer.writerow(['Iteration', 'Time', 'Shannon_Entropy', 'Loc_Error'])
        
        self.iteration = 0
        self.current_entropy = 0.0
        
        # 订阅话题
        # 假设您的算法节点发布了以下话题
        rospy.Subscriber('/shannon_entropy', Float64, self.entropy_cb)
        rospy.Subscriber('/robot_1/odom', Odometry, self.odom_cb) # 以机器人1为例
        
        rospy.loginfo("数据记录器已启动，正在保存至 search_data.csv...")

    def entropy_cb(self, msg):
        self.current_entropy = msg.data

    def odom_cb(self, msg):
        # 简单演示：计算机器人当前位置与源的距离作为实时误差
        # 在实际SCI论文中，这里应使用粒子群的加权均值(Centroid)来计算误差
        curr_pos = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        error = np.linalg.norm(curr_pos - self.true_source_pos)
        
        # 每当收到位置更新时记录一次
        self.writer.writerow([self.iteration, rospy.get_time(), self.current_entropy, error])
        self.iteration += 1

    def run(self):
        rospy.spin()
        self.log_file.close()

if __name__ == '__main__':
    DataRecorder().run()
