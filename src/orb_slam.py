"""
Pythonic Implementation of the ORB SLAM Algorithm
based on the paper:
"ORBSLAM-based Endoscope Tracking and 3D Reconstruction" by Nader Mahmoud, et al.

Date: 2024-01-04
Author: Mitterrand Ekole
"""

import numpy as np
import cv2
import g2o


class ORBSLAM:
    def __init__(self, map_points, orb_threshold=10):
        self.map_points = map_points
        self.orb_threshold = orb_threshold
        self.orb=cv2.ORB_create()
        self.last_frame=None
        self.current_pose=np.eye(3,4)
        self.optimizer=g2o.SparseOptimizer()

    def get_orb_descripor(self, frame):
        keypoints, descriptors = self.orb.detectAndCompute(frame, None)
        return keypoints, descriptors
    
    def match_features(self, frame1,frame2):
        bf=cv2.BFMatcher(cv2.NORM_HAMMING)
        matches=bf.match(frame1,frame2)
        matches.sort(key=lambda x:x.distance)

        return matches


    def opmitize_map(self):
        #add vertices for camera pose
        for i in range(len(self.key_frames)):
            vertex=g2o.VertexSE3Expmap()
            vertex.set_id(i)
            vertex.set_estimate(self.key_frames[i]['pose'])

            if i==0:
                vertex.set_fixed(True) #fixing first keyframe when camera is static
            self.optimizer.add_vertex(vertex)

        #add vertices for map points
            
        for i in range (len(self.map_points)):
            point=g2o.VertexSBAPointXYZ()
            point.set_id(i +len(self.key_frame))
            point.set_estimate(self.map_points[i]['position'])
            self.optimizer.add_vertex(point)

        #add edges for reprojection error
            
        for i, keyframe in enumerate(self.key_frames):
            for j , match in enumerate(keyframe['matches']):
                edge=g2o.EdgeProjectXYZ2UV()
                edge.set_vertex(0,self.optimizer.vertex(i))
                edge.set_vertex(1,self.optimizer.vertex(j+len(self.key_frames)))
                edge.set_measurement(match['position'])
                edge.set_information(np.eye(2))
                edge.set_robust_kernel(g2o.RobustKernelHuber())
                self.optimizer.add_edge(edge)
        #optimize
                
        self.optimizer.initialize_optimization()
        self.optimizer.optimize(20)  #optimisation iterations

        #update map points and keyframe poses

        for i, keyframe in enumerate(self.key_frames):
            self.key_frames[i]['pose']=self.optimizer.vertex(i).estimate()
        for i, map_points in enumerate(self.map_points):
            self.map_points[i]['position']=self.optimizer.vertex(i+len(self.key_frames)).estimate()
        

    def track_frame(self, current_frame):
        if self.last_frame is None:
            self.last_frame = current_frame
            return self.current_pose
        last_keypoints, last_descriptors = self.get_orb_descriptors(self.last_frame)
        current_keypoints, current_descriptors = self.get_orb_descriptors(current_frame)

        matches = self.match_features(last_descriptors, current_descriptors)

        good_matches = [m for m in matches if m.distance < self.orb_threshold]

        if len(good_matches) > 20:
            last_points = np.float32([last_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            current_points = np.float32([current_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            essential_matrix, _ = cv2.findEssentialMat(last_points, current_points)
            _, rotation, translation, mask = cv2.recoverPose(essential_matrix, last_points, current_points)

            current_pose = np.hstack((rotation, translation))

            # Save keyframe information
            keyframe = {'pose': current_pose.copy(), 'matches': []}

            for m in good_matches:
                match = {
                    'point_index': m.trainIdx,
                    'keypoint': current_keypoints[m.trainIdx].pt
                }
                keyframe['matches'].append(match)

            self.key_frames.append(keyframe)

            #perform bundle adjustement at lower frequency

            if len(self.key_frames)%5==0:
                self.optimize_map()

            self.current_pose = np.dot(self.current_pose, np.hstack((rotation, translation)))
            self.last_frame = current_frame

            return self.current_pose
        else:
            return None
        

#testing the algorithm
        
if __name__=='__main__':
    map_points=np.random.rand(100,3)
    initial_frame=cv2.imread('') #path to image
    orb_slam=ORBSLAM(map_points)


    #for video stream
    for frame_path in ["path to video","path to video"]:
        current_frame=cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        pose=orb_slam.track_frame(current_frame)

        if pose is not None:
            print("Estimated Pose is: ", pose)

        else:
            print(" Tracking Failed for frame", frame_path)



           