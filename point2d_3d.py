
import numpy as np
import cv2

# checkerboard corner point position (ul, vl, ur, vr)
# l for group_5_19_left.jpg, r for group_5_19_right.jpg
# The 3D distance of the following two points are 20mm (ground truth)
points = [
[598.563,	436.873,	525.596,	380.127], 
[631.825,	427.795,	557.187,	383.776],
]

# intrinsic matrix of the left and right cameras (got from around 10 images)
intrinsic_left = np.array([[948.209,0,617.72],[0,960.912,374.665],[0,0,1]])
intrinsic_right = np.array([[960.613,0,653.568],[0,968.804,369.677],[0,0,1]])

distortion_left = np.array([0.0487857,-0.279364,0.00329971,-0.0073942,0.367614])
distortion_right = np.array([0.0400987,-0.218146,0.00793717,0.00552612,0.133228])

# rotation and translation matrix of camera1&2 related to checkerboard (group_5_18)
Rl = cv2.Rodrigues(np.array([-0.73609308,
 0.52918454,
 -1.22768851]))[0]
Tl = [[ 4.76239558],[-3.58845201],[30.37025909]]

Rr = cv2.Rodrigues(np.array([-0.79458121,
 -0.38134605,
 0.00356695]))[0]
Tr = [[-2.77118624],
 [ 0.77588674],
 [23.01224711]]

def point2d_3d(ul, vl, ur, vr):

    # Zc_left * [[ul],[vl],[1]] = Pl * [[X],[Y],[Z],[1]]
    Pl = np.dot(intrinsic_left, np.hstack((Rl, Tl)))
    Pr = np.dot(intrinsic_right, np.hstack((Rr, Tr)))

    # solve AX = B
    A_eq = [[ul*Pl[2][0]-Pl[0][0], ul*Pl[2][1]-Pl[0][1], ul*Pl[2][2]-Pl[0][2]],\
        [vl*Pl[2][0]-Pl[1][0], vl*Pl[2][1]-Pl[1][1], vl*Pl[2][2]-Pl[1][2]],\
        [ur*Pr[2][0]-Pr[0][0], ur*Pr[2][1]-Pr[0][1], ur*Pr[2][2]-Pr[0][2]],\
        [vr*Pr[2][0]-Pr[1][0], vr*Pr[2][1]-Pr[1][1], vr*Pr[2][2]-Pr[1][2]]] 
    B_eq = [Pl[0][3]-ul*Pl[2][3], Pl[1][3]-vl*Pl[2][3], Pr[0][3]-ur*Pr[2][3], Pr[1][3]-vr*Pr[2][3]]

    answer = np.linalg.lstsq(A_eq, B_eq, rcond=-1)
    X = 20*answer[0][0]
    Y = 20*answer[0][1]
    Z = 20*answer[0][2]
#     print(X,Y,Z,end='\n')
#     print(np.dot(A_eq, [[X],[Y],[Z]]))

    return X, Y, Z

def undistort_image(img, intrinsic, distortion):
    h, w = img.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(intrinsic,distortion,(w,h),0,(w,h))
    dst = cv2.undistort(img, intrinsic, distortion, None, newcameramtx)
    # crop the image
    x,y,w,h = roi
    dst = dst[y:y+h, x:x+w]
    return dst

def distance_3D(point0_2d, point1_2d):
    X0, Y0, Z0 = point2d_3d(point0_2d[0],point0_2d[1],point0_2d[2],point0_2d[3])
    X1, Y1, Z1 = point2d_3d(point1_2d[0],point1_2d[1],point1_2d[2],point1_2d[3])
    
    distance = np.sqrt((X0 - X1)*(X0 - X1) + (Y0 - Y1)*(Y0 - Y1) + (Z0 - Z1)*(Z0 - Z1))
    print(distance) 

def main():
    
    print("Distance in 3D (mm)") 
    for point_2d in points[1:]:
        distance_3D(points[0], point_2d)


if __name__ == '__main__':
    main()
    