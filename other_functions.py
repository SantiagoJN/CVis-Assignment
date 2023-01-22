#####################################################################################
#
# MRGCV Unizar - Computer vision - Laboratory 2
# Functions from the previous lab
# Date: October 2022
# Authors: Juan Zacarías (757279), Santiago Jiménez (7809609)
#
#####################################################################################
import numpy as np
import matplotlib.pyplot as plt


def ensamble_T(R_w_c, t_w_c) -> np.array:
    # Ensamble the a SE(3) matrix with the rotation matrix and translation vector.
    T_w_c = np.zeros((4, 4), dtype=np.float32)
    T_w_c[0:3, 0:3] = R_w_c
    T_w_c[0:3, 3] = t_w_c
    T_w_c[3, 3] = 1
    return T_w_c


def plotLabeledImagePoints(x, labels, strColor, offset):
    for k in range(x.shape[1]):
        plt.text(x[0, k] + offset[0], x[1, k] + offset[1], labels[k], color=strColor)


def plotNumberedImagePoints(x, strColor, offset):
    for k in range(x.shape[1]):
        plt.text(x[0, k] + offset[0], x[1, k] + offset[1], str(k), color=strColor)


def plotLabelled3DPoints(ax, X, labels, strColor, offset):
    for k in range(X.shape[1]):
        ax.text(X[0, k] + offset[0], X[1, k] + offset[1], X[2, k] + offset[2], labels[k], color=strColor)


def plotNumbered3DPoints(ax, X, strColor, offset):
    for k in range(X.shape[1]):
        ax.text(X[0, k] + offset[0], X[1, k] + offset[1], X[2, k] + offset[2], str(k), color=strColor)

def drawRefSystem(ax, T_w_c, strStyle, nameStr):
    """
        Draw a reference system in a 3D plot: Red for X axis, Green for Y axis, and Blue for Z axis
    -input:
        ax: axis handle
        T_w_c: (4x4 matrix) Reference system C seen from W.
        strStyle: lines style.
        nameStr: Name of the reference system.
    """
    draw3DLine(ax, T_w_c[0:3, 3:4], T_w_c[0:3, 3:4] + T_w_c[0:3, 0:1], strStyle, 'r', 1)
    draw3DLine(ax, T_w_c[0:3, 3:4], T_w_c[0:3, 3:4] + T_w_c[0:3, 1:2], strStyle, 'g', 1)
    draw3DLine(ax, T_w_c[0:3, 3:4], T_w_c[0:3, 3:4] + T_w_c[0:3, 2:3], strStyle, 'b', 1)
    ax.text(np.squeeze(T_w_c[0, 3] + 0.1), np.squeeze(T_w_c[1, 3] + 0.1), np.squeeze(T_w_c[2, 3] + 0.1), nameStr)

def draw3DLine(ax, xIni, xEnd, strStyle, lColor, lWidth):
    """
    Draw a segment in a 3D plot
    -input:
        ax: axis handle
        xIni: Initial 3D point.
        xEnd: Final 3D point.
        strStyle: Line style.
        lColor: Line color.
        lWidth: Line width.
    """
    ax.plot([np.squeeze(xIni[0]), np.squeeze(xEnd[0])], [np.squeeze(xIni[1]), np.squeeze(xEnd[1])],
            [np.squeeze(xIni[2]), np.squeeze(xEnd[2])],
            strStyle, color=lColor, linewidth=lWidth)


def plotResidual(x,xProjected,strStyle):
    """
        Plot the residual between an image point and an estimation based on a projection model.
         -input:
             x: Image points.
             xProjected: Projected points.
             strStyle: Line style.
         -output: None
         """

    for k in range(x.shape[1]):
        plt.plot([x[0, k], xProjected[0, k]], [x[1, k], xProjected[1, k]], strStyle)


def plotNumberedImagePoints(x,strColor,offset):
    """
        Plot indexes of points on a 2D image.
         -input:
             x: Points coordinates.
             strColor: Color of the text.
             offset: Offset from the point to the text.
         -output: None
         """
    for k in range(x.shape[1]):
        plt.text(x[0, k]+offset, x[1, k]+offset, str(k), color=strColor)

