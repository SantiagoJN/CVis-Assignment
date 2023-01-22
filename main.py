#####################################################################################
#
# MRGCV Unizar - Computer vision - Assignment
# Date: 2022-2023 Course
# Author: Santiago Jiménez (7809609)
#
#####################################################################################

import numpy as np
import cv2
import utils
import matplotlib.pyplot as plt
import scipy.linalg
import math
import scipy.optimize as scOptim

np.set_printoptions(precision=4, linewidth=1024, suppress=True)

PRINT_TASK1 = True  # Switch whether to print the plots of task 1 or not
print1_level = 15     # Number of prints available {0..9}
PRINT_TASK2 = True

image1 = "new2"
image2 = "new3"
image3 = "new2"
image4 = "old2_1980"
matches_path = f"{image1}_{image2}_matches.npz"
matches_path2 = f"{image3}_{image4}_matches.npz"
matches_path3 = f"{image4}_{image3}_matches.npz"
img1_path = f"../Cuartel Palafox/{image1}.jpeg"
img2_path = f"../Cuartel Palafox/{image2}.jpeg"
img3_path = f"../Cuartel Palafox/{image3}.jpeg"
img4_path = f"../Cuartel Palafox/{image4}.jpg"
Kc_path = "../calibration/K_mobile_2.txt"

#######################################################################
####################             TASK 1               #################
#################### 3D SCENE ESTIMATION FROM MATCHES #################
#######################################################################

matches_file = np.load(matches_path)
matches_mask = (matches_file.f.matches > 0.0)

x1 = matches_file.f.keypoints0 * np.reshape(matches_mask, (matches_mask.size, 1))
x1 = ((x1[x1 != 0]).reshape((-1, 2))).T  # Get rid of the zeros and reshape it
x1 = np.vstack((x1, np.ones(x1.shape[1])))  # Converting into homogeneous coordinates

x2 = np.zeros((x1.shape[1], 2))
points_count = 0
for match in matches_file.f.matches:
    if match != -1:  # If it is actually a match
        x2[points_count] = matches_file.f.keypoints1[match]
        points_count += 1

x2 = x2.T  # Get rid of the zeros and reshape it
x2 = np.vstack((x2, np.ones(x2.shape[1])))  # Converting into homogeneous coordinates

canon = np.eye(3, 4)  # Canonical Perspective Projection

img1 = cv2.cvtColor(cv2.imread(img1_path), cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(cv2.imread(img2_path), cv2.COLOR_BGR2RGB)

if PRINT_TASK1 and print1_level > 0:
    utils.print_matches(img1, img2, x1, x2, image1, image2)  # TODO: 0

# ~~~~~~~~~~ Estimating F with both 8-point algorithm and the RANSAC algorithm ~~~~~~~~~~
F_matches = utils.compute_fundamental(x2, x1)
if PRINT_TASK1 and print1_level > 1:
    utils.draw_epipolar_lines(F_matches, "F by matches", img1, img2)  # Use the provided F to draw the lines # TODO: 1
print(f'F computed with the matches: \n{F_matches}')

do_plot = (PRINT_TASK1 and print1_level > 2)
F, MaxVotes, x1, x2 = utils.ransac_fundamental(x1, x2, 200, 1, img1, img2, plot_result=do_plot)  # TODO: 2
print(f'There were {points_count - MaxVotes} outliers out of the {points_count} points.')
print(f'F computed with RANSAC: \n{F}')

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## ~~~~~~~~~~~Structure From Motion to estimate the 3D poses~~~~~~~~~~~
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
K_c = np.loadtxt(Kc_path)
points_groups = np.zeros((4, 4, x1.shape[1]))  # Where we'll save all the points to display later

best_proy, Ps = utils.scene_estimation_SFM(x1, x2, K_c, F_matches, points_groups)
P21 = Ps[best_proy]
if PRINT_TASK1 and print1_level > 3:
    utils.draw_multiple_3D_points(points_groups, "Points proyected using camera 1")  # 3D Plot of the 4 solutions  # TODO: 3

T_c1_w = np.eye(4, 4)  # Hemos asumido que la cámara 1 está en el centro del mundo
T_c2_c1 = np.linalg.inv(K_c) @ P21  # Sacar T_c2_c1 a partir de P21 (T.3, dia.24)
T_c2_c1 = np.vstack((T_c2_c1, np.array([0, 0, 0, 1])))
T_c2_w = T_c2_c1 @ T_c1_w
print(f'T_c2_w: \n{T_c2_w}')

t21 = T_c2_c1[:3, -1]
scale = np.linalg.norm(t21)
print(f'Scale: {scale}')  # TODO: hacer algo con esto?

points3d = T_c2_w @ points_groups[best_proy]
#if PRINT_TASK1 and print1_level > 4:
#    utils.draw_multiple_3D_points(points_groups2, "Points proyected using camera 2")  # 3D Plot of the 4 solutions  # TODO: 4
print(f'Position of C2 with respect to C1: {T_c2_c1[:,3]}')

if PRINT_TASK1 and print1_level > 5:
    utils.draw_3D_points(points3d*20, T_c1_w, T_c2_w)  # TODO: 5

reprojection_c1 = K_c @ canon @ points_groups[best_proy]
reprojection_c1 = reprojection_c1 / reprojection_c1[2]
error_c1 = x1 - reprojection_c1

reprojection_c2 = K_c @ canon @ T_c2_w @ points_groups[best_proy]
reprojection_c2 = reprojection_c2 / reprojection_c2[2]
error_c2 = x2 - reprojection_c2

print(f'Mean reprojection error in c1 = {error_c1.mean()}, and in c2 = {error_c2.mean()}')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~ESTIMATION OF THE 3rd CAMERA POSE~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# DLT: L.6, slide 13

matches_file2 = np.load(matches_path2)
matches_mask2 = (matches_file2.f.matches > 0.0)

# ~~~~~~Getting the 2D points~~~~~~
x3 = matches_file2.f.keypoints0 * np.reshape(matches_mask2, (matches_mask2.size, 1))
x3 = ((x3[x3 != 0]).reshape((-1, 2))).T  # Get rid of the zeros and reshape it
x3 = np.vstack((x3, np.ones(x3.shape[1])))  # Converting into homogeneous coordinates

x4 = np.zeros((x3.shape[1], 2))
points_count = 0
for match in matches_file2.f.matches:
    if match != -1:  # If it is actually a match
        x4[points_count] = matches_file2.f.keypoints1[match]
        points_count += 1

x4 = x4.T  # Get rid of the zeros and reshape it
x4 = np.vstack((x4, np.ones(x4.shape[1])))  # Converting into homogeneous coordinates

canon = np.eye(3, 4)  # Canonical Perspective Projection

img3 = cv2.cvtColor(cv2.imread(img3_path), cv2.COLOR_BGR2RGB)
img4 = cv2.cvtColor(cv2.imread(img4_path), cv2.COLOR_BGR2RGB)

if PRINT_TASK1 and print1_level > 6:
    utils.print_matches(img3, img4, x3, x4, image3, image4)  # TODO: 6

# ~~~~~~Get the 3D points~~~~~~
# Brute force search for matches that are close enough
threshold = 2
index_pairs = []
for point1 in range(x1.shape[1]):  # Iterate over all points of the first matches
    for point2 in range(x3.shape[1]):  # Iterate over all points of the second matches
        dist = abs(x1[0, point1] - x3[0, point2]) + abs(x1[1, point1] - x3[1, point2])
        if dist < threshold:  # It seems to be the same point
            index_pairs.append([point1, point2])  # Save the paired indices

index_pairs = np.array(index_pairs)  # Convert it into a np array
paired_x1 = np.array([x1[:, x] for x in index_pairs[:, 0]]).T  # Get the paired points
paired_x3 = np.array([x3[:, x] for x in index_pairs[:, 1]]).T
print(f'{x1.shape[1]} points in the first matches, {x3.shape[1]} in the second => {index_pairs.shape[0]} coincide.')
if PRINT_TASK1 and print1_level > 7:
    utils.print_matched_matches(img3, paired_x1, paired_x3)  # TODO: 7

x4_filtered = np.array([x4[:, x] for x in index_pairs[:, 1]]).T  # The 2D points from the old picture (come from x4)
x4_3D_filtered = np.array([points_groups[best_proy, :, x] for x in index_pairs[:, 0]]).T

# Check if the 2D - 3D points make sense
if PRINT_TASK1 and print1_level > 8:
    utils.print_used_matches(img4, x4, index_pairs[:,1], "Matches of the old picture")  # TODO: 8
    utils.draw_3D_points(x4_3D_filtered, canon, canon)

T_c3_w, P31 = utils.DLT(x4_filtered, x4_3D_filtered)
print(f'T_c3_w: \n{T_c3_w}')
print(f'Position of C3 with respect to C1: {T_c3_w[:,3]}')

scaled_3dpoints = points_groups[best_proy] * 20
if PRINT_TASK1 and print1_level > 9:
    utils.draw_3D_points_3Cams(scaled_3dpoints, T_c1_w, T_c2_w, T_c3_w)  # TODO: 9


#######################################################################
####################             TASK 2               #################
################### DETERMINING CHANGES IN THE IMAGES #################
#######################################################################

# Changed points in the NEW image
matches_file2 = np.load(matches_path2)
changes_mask1 = ((matches_file2.f.match_confidence < 0.05) * (matches_file2.f.match_confidence > 0.0))

changed1 = matches_file2.f.keypoints0 * np.reshape(changes_mask1, (changes_mask1.size, 1))
changed1 = ((changed1[changed1 != 0]).reshape((-1, 2))).T  # Get rid of the zeros and reshape it
changed1 = np.vstack((changed1, np.ones(changed1.shape[1])))  # Converting into homogeneous coordinates

# Changed points in the OLD image
matches_file3 = np.load(matches_path3)
changes_mask2 = ((matches_file3.f.match_confidence < 0.05) * (matches_file3.f.match_confidence > 0.0))

changed2 = matches_file3.f.keypoints0 * np.reshape(changes_mask2, (changes_mask2.size, 1))
changed2 = ((changed2[changed2 != 0]).reshape((-1, 2))).T  # Get rid of the zeros and reshape it
changed2 = np.vstack((changed2, np.ones(changed2.shape[1])))  # Converting into homogeneous coordinates

# We have the unchanged points from the matches computed in previous steps!
unchanged1 = x3
unchanged2 = x4

if PRINT_TASK2:
    utils.print_changes(img3, img4, changed1, changed2, unchanged1, unchanged2)

