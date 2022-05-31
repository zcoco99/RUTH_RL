# Determine finger configurations from contact points - assume contact points form isosceles triangle
import numpy as np
import math
from scipy.optimize import fsolve

class ruthModel:
    def __init__(self):
        pass

    def compute_ruth_pose(self, CP1a, CP2a, CP3a):
        # Step 1a - Define contact points (CP1 is middle one, CP3 left, CP2 right)
        # CP1a = [50;60;50]; CP2a = [70;40;50]; CP3a = [30;40;50];
        sep12 = np.linalg.norm(CP2a-CP1a)
        sep13 = np.linalg.norm(CP3a-CP1a)
        sep23 = np.linalg.norm(CP3a-CP2a)
        if np.abs(sep12-sep13) < np.abs(sep12-sep23) and np.abs(sep12-sep13) < np.abs(sep13-sep23): # if sides 12 and 13 are closest to equal
            CP1 = CP1a
            CP2 = CP2a
            CP3 = CP3a
            up_vec = np.cross((CP2-CP3)/np.linalg.norm(CP2-CP3), (CP1-CP3)/np.linalg.norm(CP1-CP3))
            if up_vec[2]>0:
                CP2, CP3 = CP3, CP2
            CP1_ori = CP1
            CP2_ori = CP2
            CP3_ori = CP3
        elif np.abs(sep12-sep23) < np.abs(sep13-sep23) and np.abs(sep12-sep23) < np.abs(sep13-sep23): # if sides 12 and 23 are closest to equal
            CP1 = CP2a
            CP2 = CP1a
            CP3 = CP3a
            up_vec = np.cross((CP2-CP3)/np.linalg.norm(CP2-CP3), (CP1-CP3)/np.linalg.norm(CP1-CP3))
            if up_vec[2]>0:
                CP2, CP3 = CP3, CP2
            CP1_ori = CP1
            CP2_ori = CP2
            CP3_ori = CP3
        else:
            CP1 = CP3a
            CP2 = CP1a
            CP3 = CP2a
            up_vec = np.cross((CP2-CP3)/np.linalg.norm(CP2-CP3), (CP1-CP3)/np.linalg.norm(CP1-CP3))
            if up_vec[2]>0:
                CP2, CP3 = CP3, CP2
            CP1_ori = CP1
            CP2_ori = CP2
            CP3_ori = CP3

        x_ax = (CP2-CP3)/np.linalg.norm(CP2-CP3)
        z_ax = np.cross(x_ax,CP1-CP3)/np.linalg.norm(np.cross(x_ax,CP1-CP3))
        y_ax = np.cross(z_ax,x_ax)
        R = np.column_stack((x_ax, y_ax, z_ax))
        RT = np.matrix.transpose(R)
        CP2 = np.matmul(RT,(CP2-CP3))
        CP1 = np.matmul(RT,(CP1-CP3))
        CP3 = np.zeros(3) 
        # CP2 = CP2-CP1; CP3 = CP3-CP1; rot_ang = 45; Ry = [cosd(rot_ang),0,sin(rot_ang);0,1,0;-sin(rot_ang),0,cos(rot_ang)];
        # CP2 = Ry*CP2+CP1; CP3 = Ry*CP3+CP1;
        

        # Step 2 - Find centre point
        C_i = (CP2+CP3)/2
        C_f = CP1
        min_ang_dif = math.inf
        min_t = 0
        min_dis = math.inf
        object_max_angle = np.arctan((CP1[1]-CP3[1])/(CP1[0]-CP3[0]))
        object_min_angle = 0
        l1 = 28.5
        l2 = 70
        P1 = np.array([0,0])
        P5 = np.array([70,0])
        d15 = np.linalg.norm(P5-P1)
        fingerLength = 100  # Length of finger when fully extedned
        P2 = P1 + l1*np.array([np.cos(0),np.sin(0)])
        P4 = P5 + l1*np.array([np.cos(np.pi),np.sin(np.pi)])
        d24 = np.sqrt((P4[0]-P2[0])**2+(P4[1]-P2[1])**2)
        A_243 = np.sqrt((d24**2+l2**2+l2**2)**2-2*(d24**4+l2**4+l2**4))/4
        Z_243 = (1/(2*d24**2))*np.array([[d24**2+l2**2-l2**2, -4*A_243], [4*A_243, d24**2+l2**2-l2**2]])
        p24 = P4-P2
        p23 = np.matmul(Z_243,p24)
        P3 = P2 + p23
        C2 = (P2+P3+P4)/3 # Compute centre point of joints
        fivebar_max_angle = np.arctan((C2[1]-P2[1])/(C2[0]-P2[0]))
        P2 = P1 + l1*np.array([np.cos(np.pi),np.sin(np.pi)])
        P4 = P5 + l1*np.array([np.cos(0),np.sin(0)])
        d24 = np.sqrt((P4[0]-P2[0])**2+(P4[1]-P2[1])**2)
        A_243 = np.sqrt((d24**2+l2**2+l2**2)**2-2*(d24**4+l2**4+l2**4))/4
        Z_243 = (1/(2*d24**2))*np.array([[d24**2+l2**2-l2**2, -4*A_243], [4*A_243, d24**2+l2**2-l2**2]])
        p24 = P4-P2
        p23 = np.matmul(Z_243,p24)
        P3 = P2 + p23
        C2 = (P2+P3+P4)/3  # Compute centre point of joints
        fivebar_min_angle = np.arctan((C2[1]-P2[1])/(C2[0]-P2[0]))

        min_angle = max(object_min_angle,fivebar_min_angle)
        max_angle = min(object_max_angle,fivebar_max_angle)
        step_angle = (max_angle-min_angle)/100
        opt_angle = min_angle  # initalise optimal angle randomly


        for angle in np.arange(min_angle, max_angle, step_angle):

            #  C = C_i + t*(C_f-C_i); % Centre of the object
            C = CP3[0:2] + np.array([np.linalg.norm(CP2[0:2]-CP3[0:2])/2,np.tan(angle)*np.linalg.norm(CP2[0:2]-CP3[0:2])/2])  # Centre of the object


            # Step3 - Compute finger base positions
            da = np.linalg.norm(CP1-CP2)
            db = np.linalg.norm(CP3[0:2]-CP2[0:2])  # distance between contact points
            d24 = db*l2/da  # Extrapolate P2-P4 distance from contact point seperation, and 5-bar link length 
            d12x = d15/2 - d24/2  # x distance from P1 tp P2
            #  theta1 = asind(d12x/l1); theta2 = -theta1; % Copute necessary motor angles
            v = C[0:2]-CP3[0:2]
            #  theta1 = 180-atan2d(v(2),v(1)); theta2 = 180-theta1;
            #  C2 = P2 + [abs(P3(1)-P2(1));tand(angle)*norm(CP2-CP3)]; % Centre of the object

            min_ang_dif = math.inf
            for t in np.arange(0,np.pi+np.pi/180,np.pi/180):
                theta1 = t
                theta2 = np.pi-t
                P2 = P1 + l1*np.array([np.cos(theta1),np.sin(theta1)])
                P4 = P5 + l1*np.array([np.cos(theta2),np.sin(theta2)])
                d24 = np.sqrt((P4[0]-P2[0])**2+(P4[1]-P2[1])**2)
                A_243 = np.sqrt((d24**2+l2**2+l2**2)**2-2*(d24**4+l2**4+l2**4))/4
                Z_243 = (1/(2*d24**2))*np.array([[d24**2+l2**2-l2**2, -4*A_243], [4*A_243, d24**2+l2**2-l2**2]])
                p24 = P4-P2
                p23 = np.matmul(Z_243, p24)
                P3 = P2 + p23
                C2 = (P2+P3+P4)/3  # Compute centre point of joints
                grip_ang = np.arctan((C2[1]-P2[1])/(C2[0]-P2[0]))
                if np.abs(grip_ang-angle)<min_ang_dif:  # Difference between object angle and gripper angle
                    min_ang_dif = np.abs(grip_ang-angle)
                    min_t = t

            theta1 = min_t
            theta2 = np.pi-min_t
            P2 = P1 + l1*np.array([np.cos(theta1),np.sin(theta1)])
            P4 = P5 + l1*np.array([np.cos(theta2),np.sin(theta2)])
            d24 = np.sqrt((P4[0]-P2[0])**2+(P4[1]-P2[1])**2)
            A_243 = np.sqrt((d24**2+l2**2+l2**2)**2-2*(d24**4+l2**4+l2**4))/4
            Z_243 = (1/(2*d24**2))*np.array([[d24**2+l2**2-l2**2, -4*A_243], [4*A_243, d24**2+l2**2-l2**2]])
            p24 = P4-P2
            p23 = np.matmul(Z_243, p24)
            P3 = P2 + p23
            C2 = (P2+P3+P4)/3  # Compute centre point of joints
            grip_ang = np.arctan((C2[1]-P2[1])/(C2[0]-P2[0]))
            P1 = P1-C2+C[0:2]
            P2 = P2-C2+C[0:2]
            P3 = P3-C2+C[0:2]
            P4 = P4-C2+C[0:2]
            P5 = P5-C2+C[0:2]  # Move joints so they are centred around centre of contact points

            dis = np.linalg.norm(P2-CP3[0:2]) - np.linalg.norm(P3-CP1[0:2])
            if np.abs(dis)<np.abs(min_dis):
                min_dis = dis
                opt_angle = angle
                opt_t = min_t
            #  atand((C(2)-P2(2))/(C(1)-P2(1)));
            #  atand((C(2)-CP3(2))/(C(1)-CP3(1)));

        C = CP3[0:2] + np.array([np.linalg.norm(CP2[0:2]-CP3[0:2])/2, np.tan(opt_angle)*np.linalg.norm(CP2[0:2]-CP3[0:2])/2])  # Centre of the object
        theta1 = opt_t
        theta2 = np.pi-opt_t
        P2 = P1 + l1*np.array([np.cos(theta1),np.sin(theta1)])
        P4 = P5 + l1*np.array([np.cos(theta2),np.sin(theta2)])
        d24 = np.sqrt((P4[0]-P2[0])**2+(P4[1]-P2[1])**2)
        A_243 = np.sqrt((d24**2+l2**2+l2**2)**2-2*(d24**4+l2**4+l2**4))/4
        Z_243 = (1/(2*d24**2))*np.array([[d24**2+l2**2-l2**2, -4*A_243], [4*A_243, d24**2+l2**2-l2**2]])
        p24 = P4-P2
        p23 = np.matmul(Z_243,p24)
        P3 = P2 + p23
        C2 = (P2+P3+P4)/3  # Compute centre point of joints
        grip_ang = np.arctan((C2[1]-P2[1])/(C2[0]-P2[0]))
        P1 = P1-C2+C[0:2]
        P2 = P2-C2+C[0:2]
        P3 = P3-C2+C[0:2]
        P4 = P4-C2+C[0:2]
        P5 = P5-C2+C[0:2]  # Move joints so they are centred around centre of contact points


        # Compute angle for required contact point
        def sym_fing_fip(theta1s, val):  # val=0 for x, 1 for y
            Rtheta1s = np.array([[np.cos(theta1s[0]),-np.sin(theta1s[0])], [np.sin(theta1s[0]), np.cos(theta1s[0])]])
            C1 = np.array([0,10.85])
            C1C2s = np.matmul(Rtheta1s,np.array([0,44.7]))
            Rtheta1s_sq = np.matmul(Rtheta1s,Rtheta1s)
            C2P9s = np.matmul(Rtheta1s_sq,np.array([-7,35]))
            P9s = C1 + C1C2s + C2P9s
            return P9s[val]+hori_bend*(1-val)

        hori_bend = max([np.linalg.norm(P2-CP3[0:2]), np.linalg.norm(P3-CP1[0:2]), np.linalg.norm(P4-CP2[0:2])])
        theta1_calc = fsolve(sym_fing_fip, 0, 0)  # Make init guess equal zero
#        theta1_calc = 0
        vert_bend = sym_fing_fip(theta1_calc,1)
#        vert_bend = 0        
         
        vert_offset = np.array([0, 0, -vert_bend])
        P1 = np.append(P1,0) + vert_offset
        P2 = np.append(P2,0) + vert_offset
        P3 = np.append(P3,0) + vert_offset
        P4 = np.append(P4,0) + vert_offset
        P5 = np.append(P5,0) + vert_offset
        C = np.append(C,0)

        R_rot = np.copy(R)
#        if R_rot[2, 2] > 0:
#            R_rot[:,0] = -R_rot[:,0]
#            R_rot[:,2] = -R_rot[:,2]

#        TCP_Point = np.matmul(R,(P1+P5)/2 - np.array([0, 0, 50])-CP3)+CP3
        TCP_Point = np.matmul(R_rot,(P1+P5)/2 - np.array([0, 0, 50]))+CP3_ori


        return TCP_Point, theta1_calc, R, RT



