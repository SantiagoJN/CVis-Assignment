#####################################################################################
#
# MRGCV Unizar - Computer vision - Assignment
# Date: 2022-2023 Course
# File to gather the extra functions used during the assignment
# Author: Santiago Jiménez (7809609)
#
#####################################################################################

import matplotlib.pyplot as plt
import numpy as np
import random
import other_functions as of
import cv2
import scipy.linalg

figure_1_id = 1
figure_2_id = 2


def get_random_color():
    R = random.uniform(0, 1)
    G = random.uniform(0, 1)
    B = random.uniform(0, 1)
    return R, G, B


def wait_keyboard():
    keyboardClick = False
    while not keyboardClick:
        keyboardClick = plt.waitforbuttonpress()


# Source: https://stackoverflow.com/questions/14720331/how-to-generate-random-colors-in-matplotlib
def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


def print_matches(img1, img2, x1, x2, image1="Image 1", image2="Image 2"):
    nPoints = x1.shape[1]
    colors = [get_random_color() for i in range(nPoints)]

    plt.figure(figure_1_id)
    plt.imshow(img1, cmap='gray', vmin=0, vmax=255)
    for p in range(nPoints):
        plt.plot(x1[0, p], x1[1, p], '.', color=colors[p], markersize=7)
    plt.title(image1)
    plt.draw()

    plt.figure(figure_2_id)
    plt.imshow(img2, cmap='gray', vmin=0, vmax=255)
    for p in range(nPoints):
        plt.plot(x2[0, p], x2[1, p], '.', color=colors[p], markersize=7)
    plt.title(image2)
    plt.draw()

    plt.show()


def print_matched_matches(img, x1, x2):
    # Print in blue one group of points, and in red the other one
    plt.figure(figure_1_id)
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.plot(x1[0, :], x1[1, :], '+', color='r', markersize=7)
    plt.plot(x2[0, :], x2[1, :], 'x', color='b', markersize=7)
    legend_elements = [plt.Line2D([0], [0], marker='+', color='r', label='First matches', markerfacecolor='k', markersize=15),
                       plt.Line2D([0], [0], marker='x', color='b', label='Second matches', markerfacecolor='k', markersize=15)]
    plt.legend(handles=legend_elements)
    plt.title("Matched matches")
    plt.draw()

    plt.show()


def print_single_matches(img, x1, title="Image 1"):

    plt.figure(figure_1_id)
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.plot(x1[0, :], x1[1, :], '.', color='r', markersize=7)
    plt.title(title)
    plt.draw()

    plt.show()


def print_used_matches(img, x, used_indexes, title="Image"):

    plt.figure(figure_1_id)
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    for p in range(x.shape[1]):
        c = 'lime' if p in used_indexes else 'r'  # Print in green if it's used. Red if it's not
        plt.plot(x[0, p], x[1, p], '.', color=c, markersize=7)
    legend_elements = [
        plt.Line2D([0], [0], marker='.', color='lime', label='Used matches', markerfacecolor='k', markersize=15),
        plt.Line2D([0], [0], marker='.', color='r', label='Unused matches', markerfacecolor='k', markersize=15)]
    plt.legend(handles=legend_elements)
    plt.title(title)
    plt.draw()

    plt.show()


def print_changes(img1, img2, changed1, changed2, unchanged1, unchanged2):

    msize = 5
    plt.figure(figure_1_id)
    plt.imshow(img1, cmap='gray', vmin=0, vmax=255)
    plt.plot(changed1[0, :], changed1[1, :], '.', color='r', markersize=msize)
    plt.plot(unchanged1[0, :], unchanged1[1, :], '.', color='lime', markersize=msize)
    legend_elements = [
        plt.Line2D([0], [0], marker='.', color='lime', label='Unchanged areas', markerfacecolor='k', markersize=15),
        plt.Line2D([0], [0], marker='.', color='r', label='Changed areas', markerfacecolor='k', markersize=15)]
    plt.legend(handles=legend_elements)
    plt.title("Changes in the new image")
    plt.draw()

    plt.figure(figure_2_id)
    plt.imshow(img2, cmap='gray', vmin=0, vmax=255)
    plt.plot(changed2[0, :], changed2[1, :], '.', color='r', markersize=msize)
    plt.plot(unchanged2[0, :], unchanged2[1, :], '.', color='lime', markersize=msize)
    legend_elements = [
        plt.Line2D([0], [0], marker='.', color='lime', label='Unchanged areas', markerfacecolor='k', markersize=15),
        plt.Line2D([0], [0], marker='.', color='r', label='Changed areas', markerfacecolor='k', markersize=15)]
    plt.legend(handles=legend_elements)
    plt.title("Changes in the old image")
    plt.draw()

    plt.show()


def compute_epipole(F):
    _, _, V = np.linalg.svd(F)
    e = V[-1]  # Get the solution
    e = e / e[2]  # Normalize
    return e


def draw_epipolar_lines(F, info, img1, img2):

    plt.figure(figure_1_id)
    plt.imshow(img1)
    plt.title('Click a point in the image! (middle button to exit)')

    plt.figure(figure_2_id)
    plt.xlim([-3000, 3000])
    plt.imshow(img2)
    plt.title('Epipolar lines. (Click to continue...)')
    print(f'Epipole: {compute_epipole(F.T)}')
    while True:  # Infinite loop taking clicked points in image 1

        plt.figure(figure_2_id)
        plt.title(f'Epipolar lines drawn with {info}. (Click to continue...)')

        plt.figure(figure_1_id)
        coord_clicked_point = plt.ginput(1, show_clicks=False)  # Get clicked coordinates

        if not coord_clicked_point:  # Right click pressed -> exit loop
            plt.close(figure_1_id)
            plt.close(figure_2_id)
            break

        p_clicked = np.array([coord_clicked_point[0][0], coord_clicked_point[0][1]])

        random_color = get_random_color()  # Get random color

        # Draw clicked point's epipolar line in image 2
        plt.plot(p_clicked[0], p_clicked[1], '+', color=random_color, markersize=15)
        plt.text(p_clicked[0], p_clicked[1], "({0:.2f}, {1:.2f})".format(p_clicked[0], p_clicked[1]), fontsize=10,
                 color=random_color)
        plt.draw()  # Drawing figure 1

        plt.figure(figure_2_id)
        point = np.array([p_clicked[0], p_clicked[1], 1.])
        # line equation  => a*x + b*y + c = 0
        line = F @ point  # Compute the epipolar line
        #print("Line equation: {0:.2f}*x + {1:.2f}*y + {2:.2f} = 0.".format(line[0], line[1], line[2]))

        p_l_y = np.array([0, -line[2] / line[1]])  # Intersection of the line with the axis Y (x=0)
        p_l_x = np.array([-line[2] / line[0], 0])  # Intersection point of the line with the axis X (y=0)

        # Draw the line segment p_l_x to  p_l_y
        plt.axline(p_l_y, p_l_x, color=random_color)

        e = compute_epipole(F.T)
        plt.plot(e[0], e[1], '+', color='r', markersize=15)
        plt.text(e[0], e[1], "Epipole ({0:.2f}, {1:.2f})".format(e[0], e[1]), fontsize=10, color='r')
        plt.draw()  # Drawing figure 2

        plt.waitforbuttonpress()


def compute_fundamental(x1, x2):
    """    Computes the fundamental matrix from corresponding points
        (x1,x2 3*n arrays) using the 8 point algorithm.
        Each row in the A matrix below is constructed as
        [x'*x, x'*y, x', y'*x, y'*y, y', x, y, 1] """

    n = x1.shape[1]
    if x2.shape[1] != n:
        raise ValueError("Number of points don't match.")

    # build matrix for equations
    A = np.zeros((n, 9))
    for i in range(n):
        A[i] = [x1[0, i] * x2[0, i], x1[0, i] * x2[1, i], x1[0, i] * x2[2, i],
                x1[1, i] * x2[0, i], x1[1, i] * x2[1, i], x1[1, i] * x2[2, i],
                x1[2, i] * x2[0, i], x1[2, i] * x2[1, i], x1[2, i] * x2[2, i]]

    # compute linear least square solution
    U, S, V = np.linalg.svd(A)
    F = V[-1].reshape(3, 3)

    # constrain F
    # make rank 2 by zeroing out last singular value
    U, S, V = np.linalg.svd(F)
    S[2] = 0
    F = np.dot(U, np.dot(np.diag(S), V))

    return F / F[2, 2]


def triangulate_v2(x1, x2, P1, P2_c2_w):
    pos3D = np.zeros((4, 1))  # Reserve memory
    for sample in range(x1.shape[1]):  # Apply SVD to every point of the images
        # Build the equation system
        A_0 = (P1[2][:] * x1[0][sample]) - P1[0][:]
        A_1 = (P1[2][:] * x1[1][sample]) - P1[1][:]
        A_2 = (P2_c2_w[2][:] * x2[0][sample]) - P2_c2_w[0][:]
        A_3 = (P2_c2_w[2][:] * x2[1][sample]) - P2_c2_w[1][:]
        A = np.array((A_0, A_1, A_2, A_3))

        u, s, vh = np.linalg.svd(A)  # Apply SVD to get the solution to the homogeneous system
        X1s = vh[-1, :]  # Normalize the result
        X1s = X1s / X1s[3]
        pos3D = np.column_stack((pos3D, np.array(X1s).T))  # Append the data

    pos3D = np.delete(pos3D, 0, 1)  # Get rid of the first column
    return pos3D


def draw_3D_points(points, T_w_c1, T_w_c2, c='r'):
    # Plot the 3D cameras and the 3D points
    fig3D = plt.figure(3)

    ax = plt.axes(projection='3d', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    of.drawRefSystem(ax, np.eye(4, 4), '-', 'W')
#    of.drawRefSystem(ax, T_w_c1, '-', 'C1')
    of.drawRefSystem(ax, T_w_c2, '-', 'C2')

    ax.scatter(points[0, :], points[1, :], points[2, :], marker='.', color=c)
    # plotNumbered3DPoints(ax, points, 'r', (0.1, 0.1, 0.1))  # For plotting with numbers

    # Matplotlib does not correctly manage the axis('equal')
    xFakeBoundingBox = np.linspace(-1, 1, 2)
    yFakeBoundingBox = np.linspace(-1, 1, 2)
    zFakeBoundingBox = np.linspace(-1, 1, 2)
    plt.plot(xFakeBoundingBox, yFakeBoundingBox, zFakeBoundingBox, 'w.')
    #  print('Close the figure to continue. Left button for orbit, right button for zoom.')
    plt.show()


def draw_3D_points_3Cams(points, T_w_c1, T_w_c2, T_w_c3, c='r'):
    # Plot the 3D cameras and the 3D points
    fig3D = plt.figure(3)

    ax = plt.axes(projection='3d', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

#    of.drawRefSystem(ax, np.eye(4, 4), '-', 'W')
    of.drawRefSystem(ax, T_w_c1, '-', 'C1')
    of.drawRefSystem(ax, T_w_c2, '-', 'C2')
    of.drawRefSystem(ax, T_w_c3, '-', 'C3')

    ax.scatter(points[0, :], points[1, :], points[2, :], marker='.', color=c)
    # plotNumbered3DPoints(ax, points, 'r', (0.1, 0.1, 0.1))  # For plotting with numbers

    # Matplotlib does not correctly manage the axis('equal')
    xFakeBoundingBox = np.linspace(-1, 1, 2)
    yFakeBoundingBox = np.linspace(-1, 1, 2)
    zFakeBoundingBox = np.linspace(-1, 1, 2)
    plt.plot(xFakeBoundingBox, yFakeBoundingBox, zFakeBoundingBox, 'w.')
    #  print('Close the figure to continue. Left button for orbit, right button for zoom.')
    plt.show()


def draw_multiple_3D_points(points, title):
    # Plot the 3D cameras and the 3D points
    fig3D = plt.figure(4)

    ax = plt.axes(projection='3d', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Create cubic bounding box to simulate equal aspect ratio
    points_aux = points.reshape(-1, points.shape[-1])
    X = points_aux[0]
    Y = points_aux[1]
    Z = points_aux[2]
    max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max()
    Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (X.max() + X.min())
    Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (Y.max() + Y.min())
    Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (Z.max() + Z.min())
    # Comment or uncomment following both lines to test the fake bounding box:
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], 'w')

    of.drawRefSystem(ax, np.eye(4, 4), '-', 'W')
#####    of.drawRefSystem(ax, T_w_c1, '-', 'C1')
#####    of.drawRefSystem(ax, T_w_c2, '-', 'C2')

#    ax.scatter(points[0, 0, :], points[0, 1, :], points[0, 2, :], marker='.', color='r', label=f'Ground Truth')

    colors = ['b', 'g', 'y', 'k']
    for p in range(4):
        ax.scatter(points[p, 0, :], points[p, 1, :], points[p, 2, :], marker='.', color=colors[p], label=f'Solution {p+1}')
    # plotNumbered3DPoints(ax, points, 'r', (0.1, 0.1, 0.1))  # For plotting with numbers

    # Matplotlib does not correctly manage the axis('equal')
    xFakeBoundingBox = np.linspace(0, 1, 1)
    yFakeBoundingBox = np.linspace(0, 1, 1)
    zFakeBoundingBox = np.linspace(0, 1, 1)
    plt.plot(xFakeBoundingBox, yFakeBoundingBox, zFakeBoundingBox, 'w.')
    #  print('Close the figure to continue. Left button for orbit, right button for zoom.')
    plt.legend()
    plt.title(title)
    #accuracy = mean_distance_3d(points[0], points[1])
    #print("Mean distance between the points and the GT = {:.3f}.".format(accuracy))

    plt.show()


def scene_estimation_SFM(x1, x2, K_c, F_matches, points_groups):
    # We obtain E by the formula
    E = K_c.T @ F_matches @ K_c
    # We apply SVD to get the multiple R possible solutions
    U, S, V = np.linalg.svd(E)

    W = np.array(([0, -1, 0], [1, 0, 0], [0, 0, 1]))

    R1 = U @ W @ V  # np.linalg.det(R1) = 1 (so we know it'll be +R1)
    R2 = U @ W.T @ V  # np.linalg.det(R2) = 1 (so we know it'll be +R2)
    t1 = U[:3, -1]  # norm = 1
    t2 = -U[:3, -1]  # norm = 1

    # P1 remains constant for all 4 solutions
    P1 = K_c @ np.array(([1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]))
    # 4 solutions of the P matrix: (R1, t1), (R2, t2), (-R1, t1), (-R2, t2)
    P2_1 = K_c @ np.column_stack((R1, t1))
    P2_2 = K_c @ np.column_stack((R2, t2))
    P2_3 = K_c @ np.column_stack((-R1, t1))
    P2_4 = K_c @ np.column_stack((-R2, t2))

    # We triangulate for each possible solution
    P_list = [P2_1, P2_2, P2_3, P2_4]
    votes_list = np.zeros(4)
    R_list = [R1, R2, -R1, -R2]
    t_list = [t1, t2, t1, t2]

    best_proy = 0
    most_votes = 0

    colors = ['r', 'g', 'b', 'y']
    for proy in range(4):
        tr = triangulate_v2(x1, x2, P1, P_list[proy])

        points_groups[proy] = tr

        # Comprobation of points in front of camera 2
        values_2 = R_list[proy][2] @ (tr[:3].T - t_list[proy]).T
        front_c2 = np.all([values_2 > 0], axis=0)
        votes_c2 = np.sum(front_c2)

        # Comprobation of points in front of camera 1
        values_1 = np.array([0, 0, 1]) @ tr[:3]
        front_c1 = np.all([values_1 > 0], axis=0)
        votes_c1 = np.sum(front_c1)

        # We compute the number of points in front of both cameras
        valid = np.all([front_c2, front_c1], axis=0)  # Points in the image
        votes_valid = np.sum(valid)  # Sum the votes of all points

        # Print the results
        print(f'For proy {proy + 1}, valid points = {votes_valid}; (c1: {votes_c1}, c2: {votes_c2})')

        # Save results if necessary
        if votes_valid > most_votes:
            best_proy = proy

    return best_proy, P_list


def plotResidualsdiff(gt, res, title, img):
    # Plot the 2D points
    plt.figure()
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    of.plotResidual(gt, res, 'k-')
    plt.plot(res[0, :], res[1, :], 'bo')
    plt.plot(gt[0, :], gt[1, :], 'rx')
    of.plotNumberedImagePoints(gt[0:2, :], 'r', 4)
    plt.title(title)
    plt.draw()



def ransac_fundamental(x1, x2, nAttempts, inliersSigma, img1, img2, plot_result=False):

    RANSACThreshold = 3 * inliersSigma
    nVotesMax = 0

    finalH = None

    for kAttempt in range(nAttempts):

        # Find 8 random points to compute the fundamental
        r = []
        for i in range(8):
            r.append(random.randrange(x1.shape[1]))

        xSubSel1 = np.array([x1[:, r[0]], x1[:, r[1]], x1[:, r[2]], x1[:, r[3]], x1[:, r[4]], x1[:, r[5]], x1[:, r[6]], x1[:, r[7]]])
        xSubSel2 = np.array([x2[:, r[0]], x2[:, r[1]], x2[:, r[2]], x2[:, r[3]], x2[:, r[4]], x2[:, r[5]], x2[:, r[6]], x2[:, r[7]]])

        # Compute Homography using our 8 random selections
        F = compute_fundamental(xSubSel2.T, xSubSel1.T)

        # Computing the distance from the points to the GT
        res2 = []
        for i in range(x1.shape[1]):
            line = F @ x1[:, i]  # Compute the epipolar line
            p_l_y = np.array([0, -line[2] / line[1]])  # Intersection of the line with the axis Y (x=0)
            p_l_x = np.array([-line[2] / line[0], 0])  # Intersection point of the line with the axis X (y=0)
            res = np.linalg.norm(np.cross(p_l_y - p_l_x,  p_l_x - x2[0:2, i])) / np.linalg.norm(p_l_y - p_l_x)
            res2.append(res)

        votes = np.abs(res2) < RANSACThreshold  # votes
        nVotes = np.sum(votes)  # Number of votes

        if nVotes > nVotesMax:  # Si supera al anterior máximo, actualizamos
            nVotesMax = nVotes
            votesMax = votes
            finalF = F
            best_x1_inliers = xSubSel1
            best_x2_inliers = xSubSel2

    if plot_result == True:
        draw_epipolar_lines(finalF, "RANSAC-estimated F", img1, img2)
    #display_best_inliers(best_x1_inliers, best_x2_inliers)

    x1_inliers = x1 * votesMax
    x1_inliers = x1_inliers[x1_inliers != 0]  # Get rid of the zeros
    x1_inliers = x1_inliers.reshape(3, -1)

    x2_inliers = x2 * votesMax
    x2_inliers = x2_inliers[x2_inliers != 0]  # Get rid of the zeros
    x2_inliers = x2_inliers.reshape(3, -1)

    return finalF, nVotesMax, x1_inliers, x2_inliers


def DLT(points2d, points3d):
    #points2d = imagePoints
    #points3d = np.ascontiguousarray(points_lm[0:3, :].T).reshape((points_lm.shape[1], 1, 3))
    nPoints = points2d.shape[1]
    A = np.zeros((nPoints*2, 12))
    for point in range(nPoints):  # Add two equations for each point: Lesson 6, slide 14
        xi, yi = points2d[:2, point]
        Xi, Yi, Zi = points3d[:3, point]
        Wi = 1

        A[point * 2] = np.array([-Xi, -Yi, -Zi, -Wi, 0, 0, 0, 0, xi*Xi, xi*Yi, xi*Zi, xi*Wi])
        A[(point * 2) + 1] = np.array([0, 0, 0, 0, -Xi, -Yi, -Zi, -Wi, yi*Xi, yi*Yi, yi*Zi, yi*Wi])

    U, S, V = np.linalg.svd(A)  # Apply SVD to the equation system to get P31
    P31 = V[-1].reshape(3, 4)

    [K, R31, t31] = cv2.decomposeProjectionMatrix(P31)[:3]  # Limit the output values to 3, because it returns more
    t31 = (t31[:3] / t31[3]).reshape((3,))
    # https://stackoverflow.com/questions/62686618/opencv-decompose-projection-matrix
    #t31 = -t31  # For some reason, the process changes the sign of t;
    T31_DLT = of.ensamble_T(R31, t31)
    return T31_DLT, P31

#################################
##### For bundle Adjustment #####
#################################

def crossMatrixInv(M):
    x = [M[2, 1], M[0, 2], M[1, 0]]
    return x


def crossMatrix(x):
    M = np.array([[0, -x[2], x[1]],
                  [x[2], 0, -x[0]],
                  [-x[1], x[0], 0]], dtype="object")
    return M

def resBundleProjection6dof(Op, x1Data, x2Data, x3Data, K_c, nPoints):
    R_ini1 = np.array([Op[0], Op[1], Op[2]])
    R_recover1 = scipy.linalg.expm(crossMatrix(R_ini1))
    _theta1 = Op[3]
    _phi1 = Op[4]
    t_ini1 = np.array([np.sin(_phi1) * np.cos(_theta1), np.sin(_theta1) * np.sin(_phi1), np.cos(_phi1)])

    R_ini2 = np.array([Op[5], Op[6], Op[7]])
    R_recover2 = scipy.linalg.expm(crossMatrix(R_ini2))
    t_ini2 = np.array([Op[8], Op[9], Op[10]])

    X3Dpoints = np.reshape(Op[11:], (3, nPoints))
    X3Dpoints = np.vstack((X3Dpoints, np.ones(nPoints)))

    canon = np.eye(3, 4)  # Canonical Perspective Projection
    T21 = of.ensamble_T(R_recover1, t_ini1)
    T31 = of.ensamble_T(R_recover2, t_ini2)

    X_proy_1 = K_c @ canon @ X3Dpoints
    X_proy_2 = K_c @ canon @ T21 @ X3Dpoints
    X_proy_3 = K_c @ canon @ T31 @ X3Dpoints

    X_proy_1 = X_proy_1 / X_proy_1[2]
    X_proy_2 = X_proy_2 / X_proy_2[2]
    X_proy_3 = X_proy_3 / X_proy_3[2]

    X_err_1 = x1Data - X_proy_1
    X_err_2 = x2Data - X_proy_2
    X_err_3 = x3Data - X_proy_3

    _res = np.hstack((X_err_1, X_err_2, X_err_3))
    _res = np.delete(_res, 2, 0)

    return _res.flatten()

def resBundleProjection6dof_generic(Op, x1Data, x2Data, xCamsData, K_c, nPoints):
    R_ini1 = np.array([Op[0], Op[1], Op[2]])
    R_recover1 = scipy.linalg.expm(crossMatrix(R_ini1))
    _theta1 = Op[3]
    _phi1 = Op[4]
    t_ini1 = np.array([np.sin(_phi1) * np.cos(_theta1), np.sin(_theta1) * np.sin(_phi1), np.cos(_phi1)])

    nCameras = (Op.size - 5 - nPoints * 3) / 6 # Number of extra cameras
    if nCameras - int(nCameras): # If the number of cameras isn't an integer
        print("Warning: the introduced cameras may be wrong")
    nCameras = int(nCameras) # Convert to int
    cameras_Rs = np.zeros((nCameras, 3, 3)) # Each camera has a 3x3 R matrix
    cameras_ts = np.zeros((nCameras, 3)) # Analogous with ts
    for cam in range(nCameras):
        R_aux = np.array([Op[cam + 5], Op[cam + 6], Op[cam + 7]])
        R_aux = scipy.linalg.expm(crossMatrix(R_aux))
        cameras_Rs[cam, :, :] = R_aux # Save the matrix
        t_aux = np.array([Op[cam + 8], Op[cam + 9], Op[cam + 10]])
        cameras_ts[cam] = t_aux

    X3Dpoints = np.reshape(Op[-nPoints*3:], (3, nPoints)) # Recover the points here
    X3Dpoints = np.vstack((X3Dpoints, np.ones(nPoints)))

    canon = np.eye(3, 4)  # Canonical Perspective Projection
    T21 = of.ensamble_T(R_recover1, t_ini1)

    cameras_Ts = np.zeros((nCameras, 4, 4))  # Each camera has a 3x4 T matrix
    for cam in range(nCameras): # Recover all the cameras
        cameras_Ts[cam] = of.ensamble_T(cameras_Rs[cam], cameras_ts[cam])

    X_proy_1 = K_c @ canon @ X3Dpoints
    X_proy_2 = K_c @ canon @ T21 @ X3Dpoints
    X_proy_cams = np.zeros((nCameras, 3, nPoints))
    for cam in range(nCameras):
        X_proy_cams[cam] = K_c @ canon @ cameras_Ts[cam] @ X3Dpoints

    X_proy_1 = X_proy_1 / X_proy_1[2]
    X_proy_2 = X_proy_2 / X_proy_2[2]
    for cam in range(nCameras):
        X_proy_cams[cam] = X_proy_cams[cam] / X_proy_cams[cam][2]

    X_err_1 = x1Data - X_proy_1
    X_err_2 = x2Data - X_proy_2
    X_err_cams = np.zeros((nCameras, 3, nPoints))
    for cam in range(nCameras):
        X_err_cams[cam] = xCamsData[cam] - X_proy_cams[cam]

    _res = np.hstack((X_err_1, X_err_2))
    for cam in range(nCameras):
        _res = np.hstack((_res, X_err_cams[cam]))
    _res = np.delete(_res, 2, 0)

    return _res.flatten()

def plotResidualsdiff(gt, res, title, img):
    # Plot the 2D points
    plt.figure()
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    of.plotResidual(gt, res, 'k-')
    plt.plot(res[0, :], res[1, :], 'bo')
    plt.plot(gt[0, :], gt[1, :], 'rx')
    of.plotNumberedImagePoints(gt[0:2, :], 'r', 4)
    plt.title(title)
    plt.draw()

