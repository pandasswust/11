#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import tf
from tf.transformations import euler_from_quaternion
from scipy.spatial.distance import cdist
from scipy.spatial import Voronoi
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import SpawnModel, DeleteModel
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Twist, Pose
from std_msgs.msg import ColorRGBA
import csv
import os
import datetime
import subprocess

class MultiAgentNuclearSearch:
    def __init__(self):
        rospy.init_node('multi_agent_aoa_voronoi')
        
        # ==========================================
        # 1. 未知核环境生成 (随机数量、随机位置、随机活度)
        # ==========================================
        self.num_actual_sources = np.random.randint(3, 6) 
        self.actual_sources = []
        for _ in range(self.num_actual_sources):
            pos = np.random.uniform(2.0, 18.0, size=2)
            act = 10 ** np.random.uniform(3.0, 7.0)
            self.actual_sources.append({
                'pos': pos, 
                'activity': act, 
                'label': f"{act:.1E} Bq",
                'physically_shielded': False 
            })
            
        self.bg_radiation = 100.0 
        self.discovered_positions = [] 
        
        # ==========================================
        # 2. 粒子滤波与底层参数
        # ==========================================
        self.num_particles = 10000
        self.particles = np.random.uniform(low=[0, 0], high=[20, 20], size=(self.num_particles, 2))
        self.weights = np.ones(self.num_particles) / self.num_particles
        self.particle_assignments = np.zeros(self.num_particles, dtype=int)
        
        self.sigma_I = 200.0      
        self.sigma_AOA = 0.5      
        
        self.num_robots = 3
        self.robot_poses = np.zeros((self.num_robots, 2)) 
        self.robot_yaws = np.zeros(self.num_robots)
        self.robot_quats = [[0, 0, 0, 1] for _ in range(self.num_robots)]
        
        self.update_dist_threshold = 0.1 
        self.last_update_poses = None 
        
        self.shot_init, self.shot_collapse, self.shot_encircle = False, False, False
        self.shot_count = 0
        
        # ==========================================
        # 3. CSV 记录器
        # ==========================================
        self.data_dir = os.path.expanduser('~/nuclear_ws/data')
        if not os.path.exists(self.data_dir): os.makedirs(self.data_dir)
        self.run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_filename = os.path.join(self.data_dir, f'radiation_search_{self.run_timestamp}.csv')
        
        with open(self.csv_filename, mode='w', newline='') as f:
            csv.writer(f).writerow(['Iteration', 'Spatial_Entropy', 'Discovered_Count', 
                             'Robot1_X', 'Robot1_Y', 'Robot2_X', 'Robot2_Y', 'Robot3_X', 'Robot3_Y'])
        
        # 4. ROS 通信
        rospy.Subscriber('/gazebo/model_states', ModelStates, self.gazebo_states_cb)
        self.cmd_pubs = [rospy.Publisher(f'/robot_{i+1}/cmd_vel', Twist, queue_size=10) for i in range(self.num_robots)]
        self.particle_pub = rospy.Publisher('/visualization/particles', Marker, queue_size=10)
        self.source_pub = rospy.Publisher('/visualization/nuclear_sources', MarkerArray, queue_size=10)
        self.voronoi_pub = rospy.Publisher('/visualization/voronoi_lines', Marker, queue_size=10)
        self.tf_broadcaster = tf.TransformBroadcaster()
        
        self.spawn_gazebo_radioactive_barrels()
        self.rate = rospy.Rate(10) 

    def take_screenshot(self, stage_name):
        filename = os.path.join(self.data_dir, f"rviz_{self.run_timestamp}_{stage_name}.png")
        try:
            subprocess.Popen(['scrot', filename])
            rospy.loginfo(f"📸 自动截图已保存: {filename}")
        except Exception: pass

    def spawn_gazebo_radioactive_barrels(self):
        try:
            rospy.wait_for_service('/gazebo/spawn_sdf_model', timeout=3.0)
            spawn_model = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
            for idx, s in enumerate(self.actual_sources):
                sdf_xml = f"""
                <sdf version="1.6"><model name="radioactive_barrel_{idx}"><static>true</static><link name="link"><visual name="visual"><pose>0 0 0.5 0 0 0</pose><geometry><cylinder><radius>0.3</radius><length>1.0</length></cylinder></geometry><material><ambient>0.0 0.8 0.0 1</ambient><diffuse>0.0 0.8 0.0 1</diffuse><emissive>0.0 1.0 0.0 1</emissive></material></visual><collision name="collision"><pose>0 0 0.5 0 0 0</pose><geometry><cylinder><radius>0.3</radius><length>1.0</length></cylinder></geometry></collision></link></model></sdf>
                """
                p = Pose()
                p.position.x, p.position.y = s['pos'][0], s['pos'][1]
                spawn_model(f"radioactive_barrel_{idx}", sdf_xml, "", p, "world")
        except Exception: pass

    def gazebo_states_cb(self, msg):
        for i in range(self.num_robots):
            name = f"robot_{i+1}"
            if name in msg.name:
                idx = msg.name.index(name)
                self.robot_poses[i] = [msg.pose[idx].position.x, msg.pose[idx].position.y]
                q = msg.pose[idx].orientation
                self.robot_quats[i] = [q.x, q.y, q.z, q.w]
                self.robot_yaws[i] = euler_from_quaternion(self.robot_quats[i])[2]

    def simulate_radiation_detector(self, robot_pos):
        total_I, dominant_aoa, max_c = self.bg_radiation, 0.0, 0.0
        for s in self.actual_sources:
            if s['physically_shielded']: continue 
            d_sq = max(np.sum((robot_pos - s['pos'])**2), 0.1) 
            c = s['activity'] / d_sq
            total_I += c
            if c > max_c:
                max_c, dominant_aoa = c, np.arctan2(s['pos'][1] - robot_pos[1], s['pos'][0] - robot_pos[0])
        return np.random.normal(total_I, self.sigma_I), np.random.normal(dominant_aoa, self.sigma_AOA)

    def voronoi_aoa_update(self):
        if np.all(self.robot_poses == 0): return
        if self.last_update_poses is None:
            self.last_update_poses = np.copy(self.robot_poses)
            return 
            
        dist_matrix = cdist(self.particles, self.robot_poses)
        self.particle_assignments = np.argmin(dist_matrix, axis=1)
        any_update = False 
        
        shared_readings = []
        for j in range(self.num_robots):
            _, meas_AOA = self.simulate_radiation_detector(self.robot_poses[j])
            shared_readings.append((self.robot_poses[j], meas_AOA))

        for i in range(self.num_robots):
            mask = (self.particle_assignments == i)
            if not np.any(mask): continue 
            
            r_pos = self.robot_poses[i]
            if np.linalg.norm(r_pos - self.last_update_poses[i]) < self.update_dist_threshold: continue 
            
            prob_fused = np.ones(np.sum(mask))
            for j in range(self.num_robots):
                peer_pos, peer_aoa = shared_readings[j]
                dx, dy = self.particles[mask, 0] - peer_pos[0], self.particles[mask, 1] - peer_pos[1]
                p_aoa = np.arctan2(dy, dx)
                angle_diff = np.minimum(np.abs(peer_aoa - p_aoa), 2*np.pi - np.abs(peer_aoa - p_aoa))
                prob_fused *= np.exp(-(angle_diff**2) / (2 * self.sigma_AOA**2))
            
            self.weights[mask] *= prob_fused
            self.last_update_poses[i] = np.copy(r_pos)
            any_update = True

        if any_update:
            self.weights += 1e-300
            self.weights /= np.sum(self.weights)

    def evaluate_info_gain(self, robot_idx, candidate_pose):
        mask = (self.particle_assignments == robot_idx)
        if not np.any(mask): return 0
        cell_particles = self.particles[mask]
        dx, dy = cell_particles[:, 0] - candidate_pose[0], cell_particles[:, 1] - candidate_pose[1]
        return np.var(np.arctan2(dy, dx))

    def auto_drive_to_info_gradient(self):
        if np.all(self.robot_poses == 0): return
        
        for i in range(self.num_robots):
            mask = (self.particle_assignments == i)
            twist = Twist() 
            if np.any(mask):
                centroid = np.average(self.particles[mask], axis=0, weights=self.weights[mask])
                rx, ry, ryaw = self.robot_poses[i, 0], self.robot_poses[i, 1], self.robot_yaws[i]
                
                best_gain, best_target = -1, centroid
                base_angle = np.arctan2(centroid[1]-ry, centroid[0]-rx)
                for offset in [-0.8, -0.4, 0, 0.4, 0.8]:
                    sample_angle = base_angle + offset
                    tx, ty = rx + 2.0*np.cos(sample_angle), ry + 2.0*np.sin(sample_angle)
                    gain = self.evaluate_info_gain(i, [tx, ty])
                    if gain > best_gain:
                        best_gain, best_target = gain, [tx, ty]
                
                dx, dy = best_target[0] - rx, best_target[1] - ry
                
                # APF 斥力叠加
                safe_dist = 1.5; k_wall = 2.5
                rep_x, rep_y = 0.0, 0.0
                if rx < safe_dist: rep_x += k_wall * (safe_dist - rx)
                if rx > 20.0 - safe_dist: rep_x -= k_wall * (rx - (20.0 - safe_dist))
                if ry < safe_dist: rep_y += k_wall * (safe_dist - ry)
                if ry > 20.0 - safe_dist: rep_y -= k_wall * (ry - (20.0 - safe_dist))
                
                k_rob = 3.0; safe_rob = 1.2
                for j in range(self.num_robots):
                    if i == j: continue
                    d_rob = np.linalg.norm(self.robot_poses[i] - self.robot_poses[j])
                    if d_rob < safe_rob and d_rob > 0.01:
                        rep_x += k_rob * (safe_rob - d_rob) * (rx - self.robot_poses[j, 0]) / d_rob
                        rep_y += k_rob * (safe_rob - d_rob) * (ry - self.robot_poses[j, 1]) / d_rob
                
                final_dx = dx + rep_x
                final_dy = dy + rep_y
                dist = np.hypot(final_dx, final_dy)
                target_yaw = np.arctan2(final_dy, final_dx)
                err = (target_yaw - ryaw + np.pi) % (2 * np.pi) - np.pi
                
                if np.hypot(centroid[0]-rx, centroid[1]-ry) > 0.1:
                    twist.angular.z = max(-2.0, min(2.0, 3.0 * err))
                    if abs(err) > 0.5: twist.linear.x = 0.05
                    else: twist.linear.x = max(0.15, min(1.0, 0.8 * dist)) 
            self.cmd_pubs[i].publish(twist)

    def _clip_segment(self, p1, p2):
        def compute_outcode(x, y):
            code = 0
            if x < 0: code |= 1
            elif x > 20: code |= 2
            if y < 0: code |= 4
            elif y > 20: code |= 8
            return code
        x0, y0, x1, y1 = p1[0], p1[1], p2[0], p2[1]
        outcode0, outcode1 = compute_outcode(x0, y0), compute_outcode(x1, y1)
        accept = False
        while True:
            if not (outcode0 | outcode1): accept = True; break
            elif outcode0 & outcode1: break
            else:
                outcode_out = outcode1 if outcode1 > outcode0 else outcode0
                x, y = 0.0, 0.0
                if outcode_out & 8:   x, y = x0 + (x1 - x0) * (20.0 - y0) / (y1 - y0), 20.0
                elif outcode_out & 4: x, y = x0 + (x1 - x0) * (0.0 - y0) / (y1 - y0), 0.0
                elif outcode_out & 2: y, x = y0 + (y1 - y0) * (20.0 - x0) / (x1 - x0), 20.0
                elif outcode_out & 1: y, x = y0 + (y1 - y0) * (0.0 - x0) / (x1 - x0), 0.0
                if outcode_out == outcode0: x0, y0 = x, y; outcode0 = compute_outcode(x0, y0)
                else: x1, y1 = x, y; outcode1 = compute_outcode(x1, y1)
        if accept: return np.array([x0, y0]), np.array([x1, y1])
        return None, None

    def publish_rviz_markers(self):
        for i in range(self.num_robots):
            if not np.all(self.robot_poses[i] == 0):
                self.tf_broadcaster.sendTransform((self.robot_poses[i,0], self.robot_poses[i,1], 0.1),
                    self.robot_quats[i], rospy.Time.now(), f"robot_{i+1}/base_link", "world")

        boundary = Marker()
        boundary.header.frame_id = "world"
        boundary.ns, boundary.id, boundary.type = "boundary", 3, Marker.LINE_STRIP
        boundary.action = Marker.ADD
        boundary.scale.x = 0.08 
        boundary.color.r = boundary.color.g = boundary.color.b = 1.0; boundary.color.a = 0.8
        for c in [(0,0), (20,0), (20,20), (0,20), (0,0)]: boundary.points.append(Point(x=c[0], y=c[1], z=0.01))
        self.voronoi_pub.publish(boundary)

        marker_array = MarkerArray()
        for idx, s in enumerate(self.actual_sources):
            core = Marker()
            core.header.frame_id = "world"
            core.ns, core.id, core.type, core.action = "source_core", idx, Marker.SPHERE, Marker.ADD
            core.pose.position.x, core.pose.position.y, core.pose.position.z = s['pos'][0], s['pos'][1], 0.2
            core.scale.x = core.scale.y = core.scale.z = 0.5; core.color.a = 1.0
            
            if s['physically_shielded']: 
                # 扁平化处理，不再凸起阻碍视线
                core.scale.z = 0.01 
                core.color.r = core.color.g = core.color.b = 0.3
            else: 
                core.color.r, core.color.g, core.color.b = 1.0, 0.2, 0.2
            marker_array.markers.append(core)

            if not s['physically_shielded']:
                halo = Marker()
                halo.header.frame_id = "world"
                halo.ns, halo.id, halo.type, halo.action = "source_halo", idx, Marker.CYLINDER, Marker.ADD
                halo.pose.position.x, halo.pose.position.y, halo.pose.position.z = s['pos'][0], s['pos'][1], 0.05
                halo.scale.x = halo.scale.y = max(2.0, np.log10(s['activity']))
                halo.scale.z = 0.02
                halo.color.r, halo.color.g, halo.color.b, halo.color.a = 1.0, 0.5, 0.0, 0.2 
                marker_array.markers.append(halo)

            text = Marker()
            text.header.frame_id = "world"
            text.ns, text.id, text.type, text.action = "source_text", idx, Marker.TEXT_VIEW_FACING, Marker.ADD
            text.pose.position.x, text.pose.position.y, text.pose.position.z = s['pos'][0], s['pos'][1], 1.5
            text.scale.z = 0.5; text.color.a = 1.0
            if s['physically_shielded']:
                text.text = "[SHIELDED]"; text.color.r = text.color.g = text.color.b = 0.5
            else:
                text.text = s['label']; text.color.r = text.color.g = text.color.b = 1.0
            marker_array.markers.append(text)
            
        self.source_pub.publish(marker_array)

        p_marker = Marker()
        p_marker.header.frame_id = "world"
        p_marker.ns, p_marker.id, p_marker.type, p_marker.action = "particles", 1, Marker.POINTS, Marker.ADD
        p_marker.scale.x = p_marker.scale.y = 0.05 
        colors = [(1.0, 0.2, 0.2), (0.2, 1.0, 0.2), (0.2, 0.5, 1.0)]
        d_step = max(1, self.num_particles // 3000)
        for idx in range(0, self.num_particles, d_step):
            p = self.particles[idx]
            assign_idx = self.particle_assignments[idx]
            p_marker.points.append(Point(x=p[0], y=p[1], z=0.02))
            c = ColorRGBA()
            c.r, c.g, c.b, c.a = colors[assign_idx][0], colors[assign_idx][1], colors[assign_idx][2], 0.4
            p_marker.colors.append(c)
        self.particle_pub.publish(p_marker)

        v_marker = Marker()
        v_marker.header.frame_id = "world"
        v_marker.ns, v_marker.id, v_marker.type, v_marker.action = "voronoi", 2, Marker.LINE_LIST, Marker.ADD
        v_marker.scale.x = 0.15 
        v_marker.color.r, v_marker.color.g, v_marker.color.b, v_marker.color.a = 1.0, 1.0, 0.0, 1.0
        c_poses = np.copy(self.robot_poses)
        if len(c_poses) >= 3 and not np.all(c_poses == 0):
            c_poses += np.random.normal(0, 1e-4, c_poses.shape)
            try:
                vor = Voronoi(c_poses)
                center = vor.points.mean(axis=0)
                for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
                    simplex = np.asarray(simplex)
                    if np.all(simplex >= 0): p0, p1 = vor.vertices[simplex]
                    else:
                        i = simplex[simplex >= 0][0]
                        t = vor.points[pointidx[1]] - vor.points[pointidx[0]]
                        t /= np.linalg.norm(t)
                        n = np.array([-t[1], t[0]])
                        midpoint = vor.points[pointidx].mean(axis=0)
                        direction = np.sign(np.dot(midpoint - center, n)) * n
                        if direction[0] == 0 and direction[1] == 0: direction = n
                        p0 = vor.vertices[i]; p1 = p0 + direction * 40.0 
                    cp0, cp1 = self._clip_segment(p0, p1)
                    if cp0 is not None and cp1 is not None:
                        v_marker.points.append(Point(x=cp0[0], y=cp0[1], z=0.05))
                        v_marker.points.append(Point(x=cp1[0], y=cp1[1], z=0.05))
                if len(v_marker.points) > 0: self.voronoi_pub.publish(v_marker)
            except Exception: pass

    def run(self):
        # ====================================================================
        # 新增拦截验证：在开始执行主循环前，必须确认 Gazebo 数据正常发布
        # ====================================================================
        rospy.loginfo("⏳ 等待 Gazebo 物理引擎和机器人模型加载...")
        try:
            rospy.wait_for_message('/gazebo/model_states', ModelStates, timeout=20.0)
        except rospy.ROSException:
            rospy.logerr("❌ 超时：未收到 Gazebo 状态数据，请检查物理引擎是否崩溃 (Exit Code 255)！节点退出。")
            rospy.signal_shutdown("Gazebo crashed or models failed to load.")
            return
            
        # 进一步确认坐标不再是全 0 (即机器人已经真正在环境里生成完毕)
        while np.all(self.robot_poses == 0) and not rospy.is_shutdown():
            rospy.sleep(0.5)
            
        rospy.loginfo(f"✅ Gazebo 状态确认！")
        rospy.loginfo(f"🎬 搭载[定向核辐射探测器]的多机集群已启动，场地内存在 {self.num_actual_sources} 个未知放射源！")
        # ====================================================================

        iteration = 0
        
        while not rospy.is_shutdown():
            self.voronoi_aoa_update()
            self.auto_drive_to_info_gradient() 
            
            hist, _, _ = np.histogram2d(self.particles[:, 0], self.particles[:, 1], bins=40, range=[[0, 20], [0, 20]])
            p_grid = hist / np.sum(hist)
            p_grid = p_grid[p_grid > 0]
            spatial_entropy = -np.sum(p_grid * np.log(p_grid))
            
            all_clean = True
            if iteration > 10:
                for j in range(self.num_robots):
                    I, _ = self.simulate_radiation_detector(self.robot_poses[j])
                    if I > self.bg_radiation + 300.0: all_clean = False
            
            if all_clean and spatial_entropy > 6.0:
                rospy.loginfo_throttle(5.0, f"🎉 探测器阵列全量程降至本底辐射，共肃清 {len(self.discovered_positions)} 个源，核设施区域安全！")
                for pub in self.cmd_pubs: pub.publish(Twist()) 
                self.publish_rviz_markers() 
                iteration += 1
                self.rate.sleep()
                continue
            
            if iteration == 1 and not self.shot_init:
                self.take_screenshot("1_Initialization")
                self.shot_init = True
            if spatial_entropy < 4.5 and not self.shot_collapse:
                self.take_screenshot(f"2_AOA_Collapse_{self.shot_count}")
                self.shot_collapse = True
                
            est_pos = np.average(self.particles, axis=0, weights=self.weights)
            robot_distances = np.linalg.norm(self.robot_poses - est_pos, axis=1)
            min_robot_dist = np.min(robot_distances) if len(robot_distances) > 0 else float('inf')

            if spatial_entropy < 3.5 and 2.0 < min_robot_dist < 4.0 and not self.shot_encircle:
                self.take_screenshot(f"3_Info_Encirclement_{self.shot_count}")
                self.shot_encircle = True
            
            rospy.loginfo("Iter: %03d | Entropy: %.4f | Discovered: %d", 
                          iteration, spatial_entropy, len(self.discovered_positions))
            
            with open(self.csv_filename, mode='a', newline='') as f:
                csv.writer(f).writerow([
                    iteration, round(spatial_entropy, 4), len(self.discovered_positions),
                    round(self.robot_poses[0, 0], 3), round(self.robot_poses[0, 1], 3),
                    round(self.robot_poses[1, 0], 3), round(self.robot_poses[1, 1], 3),
                    round(self.robot_poses[2, 0], 3), round(self.robot_poses[2, 1], 3)
                ])
                
            self.publish_rviz_markers()
            
            if spatial_entropy < 3.0 and min_robot_dist < 1.5:
                rospy.logwarn(f"\n🎯 探测器阵列交叉锁定热点并抵近确认！(估算坐标: {est_pos[0]:.2f}, {est_pos[1]:.2f})")
                
                self.discovered_positions.append(np.copy(est_pos))
                
                shielded_any_actual = False
                
                # ========================================================
                # 物理清除核心逻辑：调取底层服务瞬间消除 Box (Gazebo 模型)
                # ========================================================
                try:
                    delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
                except Exception:
                    delete_model = None

                for idx, s in enumerate(self.actual_sources):
                    if not s['physically_shielded'] and np.linalg.norm(est_pos - s['pos']) < 2.5:
                        s['physically_shielded'] = True
                        shielded_any_actual = True
                        rospy.logwarn(f"🛡️ 物理消除：已彻底屏蔽 {s['label']} 的源，清除物理障碍！")
                        
                        # 真正地从 Gazebo 中把这个桶删掉，绝不让它再挡路
                        if delete_model:
                            try:
                                delete_model(f"radioactive_barrel_{idx}")
                            except Exception: pass
                
                if shielded_any_actual:
                    self.publish_rviz_markers() 
                    rospy.sleep(0.5)
                    self.take_screenshot(f"4_Shielding_Done_{self.shot_count}")
                    self.shot_count += 1
                    
                    self.num_particles = 10000
                    self.particles = np.random.uniform(low=[0, 0], high=[20, 20], size=(self.num_particles, 2))
                    self.weights = np.ones(self.num_particles) / self.num_particles
                    self.particle_assignments = np.zeros(self.num_particles, dtype=int)
                    self.last_update_poses = None 
                    
                    self.shot_collapse = False
                    self.shot_encircle = False
                    rospy.sleep(1.0) 
                else:
                    rospy.logwarn("⚠️ 确证失败！此处无显著高能射线溢出，系统重新评估...")
                    self.particles += np.random.normal(0, 5.0, size=self.particles.shape)
                    self.particles[:, 0] = np.clip(self.particles[:, 0], 0.0, 20.0)
                    self.particles[:, 1] = np.clip(self.particles[:, 1], 0.0, 20.0)
                continue 
            
            n_eff = 1.0 / np.sum(self.weights**2)
            if n_eff < self.num_particles / 2.0:
                indices = np.random.choice(self.num_particles, 10000, p=self.weights)
                self.particles = self.particles[indices]
                self.weights = np.ones(self.num_particles) / self.num_particles
                self.particle_assignments = self.particle_assignments[indices]
                
                self.particles += np.random.normal(0, 0.2, size=self.particles.shape)
                self.particles[:, 0] = np.clip(self.particles[:, 0], 0.0, 20.0)
                self.particles[:, 1] = np.clip(self.particles[:, 1], 0.0, 20.0)
            
            iteration += 1
            self.rate.sleep()

if __name__ == '__main__':
    try:
        MultiAgentNuclearSearch().run()
    except rospy.ROSInterruptException: pass
