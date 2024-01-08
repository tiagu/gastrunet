import numpy as np
import itertools
from skimage.morphology import skeletonize, thin
from scipy.spatial import cKDTree
from fil_finder import FilFinder2D
import astropy.units as u
import pandas as pd
import cv2 as cv2
from skimage import morphology, graph
from skan import Skeleton
from skan.draw import overlay_skeleton_2d
import itertools
from fastai.vision.all import *
from scipy.spatial.distance import euclidean
from PIL import Image

def skeleton_endpoints(skel):
    # Make our input nice, possibly necessary.
    skel = skel.copy()
    skel[skel!=0] = 1
    skel = np.uint8(skel)

    # Apply the convolution.
    kernel = np.uint8([[1,  1, 1],
                       [1, 10, 1],
                       [1,  1, 1]])
    src_depth = -1
    filtered = cv2.filter2D(skel,src_depth,kernel)

    #out = np.zeros_like(skel)
    #out[np.where(filtered==11)] = 1
    return np.where(filtered==11)


def nearest_neighbor(current_point, points):
    distances = [np.linalg.norm(np.array(current_point) - np.array(p)) for p in points]
    min_distance_index = np.argmin(distances)
    return points[min_distance_index]




def find_biggest_distance(coordinates):
    max_distance = 0
    max_distance_points = None

    for pair in itertools.combinations(coordinates, 2):
        distance = euclidean(pair[0], pair[1])
        if distance > max_distance:
            max_distance = distance
            max_distance_points = pair

    return max_distance


def circular_max(image, center, radius):
    # Get the coordinates of the circular neighborhood
    y, x = np.ogrid[-center[1]:image.shape[0]-center[1], -center[0]:image.shape[1]-center[0]]
    circular_mask = x**2 + y**2 <= radius**2

    # Apply the circular mask and compute the average
    values = image[circular_mask]
    average_value = np.max(values)

    return average_value



def fit_line(coordinates):
    x, y = zip(*coordinates)
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return m, c

def rotate_coords(coordinates,theta,midpoint):
    
    tempX = coordinates[0] - midpoint[0]
    tempY = coordinates[1] - midpoint[1]
    
    rotatedX = tempX*np.cos(theta) - tempY*np.sin(theta)
    rotatedY = tempX*np.sin(theta) + tempY*np.cos(theta)
    
    X = rotatedX + midpoint[0]
    Y = rotatedY + midpoint[1]
    
    return [X,Y]

def get_endpoints(coordinates, slope, intercept):
    x_values = [x for x, y in coordinates]
    y_values = [y for x, y in coordinates]

    x_min = min(x_values)
    x_max = max(x_values)

    y_min = slope * x_min + intercept
    y_max = slope * x_max + intercept

    endpoint1 = (x_min, y_min)
    endpoint2 = (x_max, y_max)

    return endpoint1, endpoint2


def inscribe_rectangle(line_coordinates, length):
    m, c = fit_line(line_coordinates)

    # Calculate the angle of the fitted line in radians
    angle_rad = np.arctan(m)
    angle_deg = np.arctan(m)*(180/np.pi)
    
    #get endpoints 
    endpoint1, endpoint2 = get_endpoints(line_coordinates, m, c)
    midpoint = (((endpoint1[0] + endpoint2[0]) / 2),((endpoint1[1] + endpoint2[1]) / 2))
    
    #points of rect top
    
    m_perpendicular = -1 / m
    
    x2 = endpoint1[0] + length / np.sqrt(1 + m_perpendicular**2)
    y2 = endpoint1[1] + m_perpendicular * length / np.sqrt(1 + m_perpendicular**2)
    top_left = (round(x2),round(y2) )

    width_line = euclidean(endpoint1, endpoint2)
    
    x2 = top_left[0] + width_line / np.sqrt(1 + m**2)
    y2 = top_left[1] + m * width_line / np.sqrt(1 + m**2)
    top_right = (round(x2),round(y2)) 
    
    #bottom part 
    x2 = endpoint1[0] - length / np.sqrt(1 + m_perpendicular**2)
    y2 = endpoint1[1] - m_perpendicular * length / np.sqrt(1 + m_perpendicular**2)
    bottom_left = (round(x2),round(y2))
    
    x2 = bottom_left[0] + width_line / np.sqrt(1 + m**2)
    y2 = bottom_left[1] + m * width_line / np.sqrt(1 + m**2)
    bottom_right = (round(x2),round(y2))

    return m,c,[top_left,top_right,bottom_right,bottom_left,top_left],angle_rad,[midpoint],[endpoint1, endpoint2],width_line  # Closing the rectangle




def get_coordinates_inside_rectangle(rectangle_corners):
    rounded_corners = [[int(x), int(y)] for x, y in rectangle_corners]

    x_values, y_values = zip(*rounded_corners)

    min_x, min_y = min(x_values), min(y_values)
    max_x, max_y = max(x_values), max(y_values)
   
    coordinates_inside = [(x, y) for x in range(min_x, max_x + 1) for y in range(min_y, max_y + 1)]

    return coordinates_inside



def labeller(x,path):
        return path/f'masks/{x.stem}_mask_{x.suffix}'

def labeller2(x,path):
        return path/f'masks/{x.stem}_length.png'
    
def labeller3(x,path):
        return path/f'only_GFP/{x.stem}{x.suffix}'
    

class pred_tensor:
    
    def __init__(self):
        print('')
        
    def __new__(self, path, path_img, pred_1, pred_2):
        pred_arx = pred_1.argmax(dim=0)
        pred_arx = pred_arx.numpy()
        rescaled = (255.0 / pred_arx.max() * (pred_arx - pred_arx.min())).astype(np.uint8)
        im = Image.fromarray(rescaled)

        im.save(labeller(path_img,path))

        px_ON=np.sum(rescaled > 0)
        px_ON_perc=np.sum(rescaled > 0)*100 /(im.shape[0]*im.shape[0]) 

        contours,_ = cv2.findContours(np.asarray(im), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        areas = [cv2.contourArea(c) for c in contours]
        combined_lists = list(zip(areas, contours))
        sorted_combined = sorted(combined_lists, key=lambda x: x[0], reverse=True)
        contours = [x[1] for x in sorted_combined]
        areas = [cv2.contourArea(c) for c in contours]
    
        possibilities = np.where( np.asarray(areas)>1000 )[0] # reasonably sized cell aggregate

        if len(possibilities) > 1:
            pxmax=[]
            for i in possibilities:
                cnt = contours[i]
                pxmax.append(len(contours[i]))
            possibilities = np.delete(possibilities, np.where(pxmax != np.max(pxmax))[0])


        if len(possibilities)>=1:

            cnt=contours[possibilities[0]]
            edges_choosen= cv2.drawContours(np.zeros(im.shape),  cnt, -1, (255,255,255),-1)
            mask_choosen = np.zeros(im.shape, dtype=np.uint8)
            cv2.drawContours(mask_choosen, [cnt], -1, (255, 255, 255), thickness=cv2.FILLED)

            #plt.imshow(mask_choosen)

            area = cv2.contourArea(cnt)

            perimeter = cv2.arcLength(cnt,True)

            policomplex = perimeter/area

            moments = cv2.moments(cnt)
            if moments['m00'] != 0.0:
                cx = moments['m10']/moments['m00']
                cy = moments['m01']/moments['m00']
                centroid = (cx,cy)
            else:
                centroid = 'Region has zero area'

            bounding_box=cv2.boundingRect(cnt)
            (bx,by,bw,bh) = bounding_box
            aspect_ratio = bw/float(bh)

            convex_hull = cv2.convexHull(cnt)

            convex_area = cv2.contourArea(convex_hull)

            solidity = area/float(convex_area)

            ellipse = cv2.fitEllipse(cnt)

            (center,axes,orientation) = ellipse

            # length of MAJOR and minor axis
            majoraxis_length = max(axes)
            minoraxis_length = min(axes)

            # eccentricity = sqrt( 1 - (ma/MA)^2) --- ma= minor axis --- MA= major axis
            eccentricity = np.sqrt(1-(minoraxis_length/majoraxis_length)**2)


            # make image to export seg and length

            img_rgb = cv2.imread(str(path_img), cv2.IMREAD_COLOR)
            im_thin = thin(mask_choosen)
            im_thin =  np.uint8( im_thin.copy() )

            edges = cv2.Canny(np.asarray(mask_choosen),10,20)


            # Kernels for hitmiss from https://stackoverflow.com/questions/26586123/filling-gaps-in-shape-edges/72502985#72502985
            k1 = np.array(([0, 0, 0], [-1, 1, -1], [-1, -1, -1]), dtype="int")
            k2 = np.array(([0, -1, -1], [0, 1, -1], [0, -1, -1]), dtype="int")
            k3 = np.array(([-1, -1, 0],  [-1, 1, 0], [-1, -1, 0]), dtype="int")
            k4 = np.array(([-1, -1, -1], [-1, 1, -1], [0, 0, 0]), dtype="int")

            k5 = np.array(([-1, -1, -1], [-1, 1, -1], [0, -1, -1]), dtype="int")
            k6 = np.array(([-1, -1, -1], [-1, 1, -1], [-1, -1, 0]), dtype="int")
            k7 = np.array(([-1, -1, 0], [-1, 1, -1], [-1, -1, -1]), dtype="int")
            k8 = np.array(([0, -1, -1], [-1, 1, -1], [-1, -1, -1]), dtype="int")               

            # hit-or-miss transform
            o1 = cv2.morphologyEx(im_thin, cv2.MORPH_HITMISS, k1)
            o2 = cv2.morphologyEx(im_thin, cv2.MORPH_HITMISS, k2)
            o3 = cv2.morphologyEx(im_thin, cv2.MORPH_HITMISS, k3)
            o4 = cv2.morphologyEx(im_thin, cv2.MORPH_HITMISS, k4)
            out1 = o1 + o2 + o3 + o4

            # store the loose end points
            pts = np.argwhere(out1 == 1)
            for pt in pts:
                loose_ends = cv2.circle(img_rgb, (pt[1], pt[0]), 10, (255,0,255), -1)

            #extend loose ends to next edge point
            im_thin_ext =  np.uint8( im_thin.copy() )
            edge_coords = np.where(edges==255)
            edge_array = np.column_stack((edge_coords[0],edge_coords[1]))
            kdtree = cKDTree(edge_array)

            for i in pts:
                query_point = i
                closest_point_index = kdtree.query(query_point)[1]
                closest_point = edge_array[closest_point_index]
                cv2.line(np.asarray(im_thin_ext),tuple([query_point[1],query_point[0]]), 
                         tuple([closest_point[1],closest_point[0]]), 1, 1)

            #longest path
            fil = FilFinder2D(im_thin_ext, mask=im_thin_ext)
            fil.create_mask(border_masking=True, verbose=False, use_existing_mask=True)
            fil.medskel(verbose=False)
            fil.analyze_skeletons(branch_thresh=40* u.pix, skel_thresh=10 * u.pix, prune_criteria='length')

            max_length = fil.lengths()
            fil.exec_rht()
            fil_ori = fil.orientation
            fil_curvature = fil.curvature


            pred_1 = pred_2 #lazy
            pred_arx = pred_1.argmax(dim=0)
            pred_arx = pred_arx.numpy()
            rescaled = (255.0 / pred_arx.max() * (pred_arx - pred_arx.min())).astype(np.uint8)
            im_gfp = Image.fromarray(rescaled)

            gfp_in = cv2.bitwise_and(mask_choosen, (np.uint8(im_gfp)), mask=(np.uint8(im_gfp)))

            px_ON_gfp=np.sum(gfp_in > 0)
            px_ON_perc_gfp=np.sum(gfp_in > 0)*100 /(im.shape[0]*im.shape[0])
            px_ON_perc_gfp_organoid= round(np.sum(gfp_in > 0)*100/ np.sum(mask_choosen > 0),1)

            contours2 ,_ = cv2.findContours(np.asarray(gfp_in), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            areas_gfp = [cv2.contourArea(c) for c in contours2]
            #print(len(areas_gfp))

            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
            ax.imshow(img_rgb)
            ax.contour(im, [0.5], colors='w')
            ax.contour(im_thin, [0.5], colors='b')
            ax.contour(mask_choosen, [0.5], colors='magenta',linewidths = 0.5)
            ax.contour(gfp_in, [1], colors='green',linewidths = 1)
            ax.contour(fil.skeleton_longpath, [0.5], colors='r')
            ax.axis('off')
            fig.savefig(labeller2(path_img,path), bbox_inches='tight', pad_inches=0)
            plt.close()
            

#             gfp_file = str(labeller3(chunk[p]))
#             gfp_file = gfp_file.replace('disp', 'GFP')
#             img_gfp = cv2.imread(gfp_file, cv2.IMREAD_GRAYSCALE)

#             dist = cv2.distanceTransform(mask_choosen, distanceType=cv2.DIST_L2, maskSize=5).astype(np.float32)


#             end_pts = skeleton_endpoints(np.uint8(fil.skeleton_longpath>0))

#             x_values = np.where(fil.skeleton_longpath>0)[0]
#             y_values = np.where(fil.skeleton_longpath>0)[1]
#             curve_points = list(zip(x_values, y_values))

#             start_point = (end_pts[0][0],end_pts[1][0])

#             current_point = start_point
#             visited_points = [start_point]

#             while len(curve_points)>=1:
#                 nearest_neighbor_point = nearest_neighbor(current_point, curve_points)    
#                 current_point = nearest_neighbor_point
#                 curve_points.remove(current_point)
#                 visited_points.append(nearest_neighbor_point)


#             k=10 # every k points in long path

#             gfp_path_means=[]
#             gfp_path_max=[]
#             gfp_path_std=[]
#             gfp_path_rect_area=[]
#             width_fitline_rectangles =[]


#             for i in range(0, len(visited_points), k):
#                 points_rect=[]
#                 points_rect_dist_edge=[]
                
#                 end_index = min(i + k, len(visited_points))
#                 for pt in range(i,end_index):
#                     points_rect.append( (visited_points[pt][1], visited_points[pt][0]) )
#                     points_rect_dist_edge.append(round(circular_max(dist,(visited_points[pt][1], visited_points[pt][0] ) , 30 ) ))

#                 coordinates =  points_rect
#                 length = round(np.mean(points_rect_dist_edge))
#                 #width = 5#round(find_biggest_distance(coordinates))

#                 rectangle_corners = inscribe_rectangle(coordinates, length)
#                 width_fitline_rectangles.append( euclidean(rectangle_corners[5][0], rectangle_corners[5][1]) )
                
#                 coords = get_coordinates_inside_rectangle(rectangle_corners[2])
                
#                 coords = [(y, x) for x, y in coords if 0 <= x < img_gfp.shape[1] and 0 <= y < img_gfp.shape[0]]


#                 gfp_path_means.append(np.mean([img_gfp[x] for x in coords]))
#                 gfp_path_max.append(np.max([img_gfp[x] for x in coords]))
#                 gfp_path_std.append( np.std([img_gfp[x] for x in coords]) )
#                 gfp_path_rect_area.append( len(coords) )

            return [ path_img ,px_ON, px_ON_perc, area, perimeter, policomplex, centroid, bounding_box, aspect_ratio, convex_area, solidity, eccentricity, max_length, fil_ori, fil_curvature, px_ON_gfp, px_ON_perc_gfp, px_ON_perc_gfp_organoid, areas_gfp ]
        else:
            return [path_img, px_ON, px_ON_perc, np.nan,  np.nan, np.nan,np.nan, 
                        np.nan, np.nan, np.nan,np.nan,np.nan,
                        np.nan, np.nan, np.nan,
                       np.nan, np.nan, np.nan,np.nan]
