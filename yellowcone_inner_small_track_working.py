from turtle import distance
from eufs_msgs.msg import WaypointArrayStamped, Waypoint, ConeArrayWithCovariance, CarState, FullState
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from rclpy.node import Node
import rclpy
import math

import numpy as np
from scipy import interpolate
import scipy.special as sps  # For computing Bernstein polynomials

class Planner(Node):
    def __init__(self, name):
        super().__init__(name)
        # Declare ROS parameters
        self.threshold = self.declare_parameter("threshold", 6.0).value

        # Create subscribers
        self.cones_sub = self.create_subscription(ConeArrayWithCovariance, "/cones", self.cones_callback, 1)

        # Create publishers
        self.track_line_pub = self.create_publisher(WaypointArrayStamped, "/trajectory", 1)
        self.visualization_pub = self.create_publisher(Marker, "/planner/viz", 1)
        self.car_pose_pub = self.create_publisher(Marker, "/planner/CarPose", 1)
        self.test_pose_pub = self.create_publisher(Marker, "/planner/TestPose", 1)
        self.line_strip_pub = self.create_publisher(Marker, "/planner/LineStrip", 1)
        self.line_list_pub = self.create_publisher(Marker, "/planner/LineList", 1)
        
    def cones_callback(self, msg):
        blue_cones = self.convert(msg.blue_cones)
        yellow_cones = self.convert(msg.yellow_cones)
        orange_cones = self.convert(msg.orange_cones)
        big_orange_cones = self.convert(msg.big_orange_cones)
        uncolored_cones = self.convert(msg.unknown_color_cones)

        # Add uncolored lidar cones to the appropriate sides
        close_uncolored = uncolored_cones[np.abs(uncolored_cones) < 4]
        close_orange_cones = orange_cones[np.abs(orange_cones) < 10]
        close_big_orange_cones = big_orange_cones[np.abs(big_orange_cones) < 10]
        
        blue_cones = self.to_2d_list(np.concatenate((blue_cones, 
                                                     close_uncolored[close_uncolored.imag > 0],
                                                     close_orange_cones[close_orange_cones.imag > 0],
                                                     close_big_orange_cones[close_big_orange_cones.imag > 0])))
        yellow_cones = self.to_2d_list(np.concatenate((yellow_cones, 
                                                       close_uncolored[close_uncolored.imag < 0],
                                                       close_orange_cones[close_orange_cones.imag < 0],
                                                       close_big_orange_cones[close_big_orange_cones.imag < 0])))

        midpoints = self.find_midpoints(blue_cones, yellow_cones, orange_cones, big_orange_cones)
        midpoints = self.sort_midpoints(midpoints)
        
        # Convert midpoints to a numpy array of complex numbers
        midpoints_c = np.array([])
        for m in midpoints:
            midpoints_c = np.append(midpoints_c, m[0] + 1j * m[1])

        if len(midpoints_c) == 0:
            return

        # Compute a smooth path using a Bezier curve based on the midpoints as control points.
        try:
            midpoints_c = self.bezier_curve(midpoints_c, num_points=50)
        except Exception as e:
            self.get_logger().info("Failed to compute bezier curve: " + str(e))
        
        self.publish_path(midpoints_c)
        self.publish_visualisation(midpoints_c)
        self.publish_line_stip(midpoints_c)

    def bezier_curve(self, control_points, num_points=50):
        """
        Compute a Bezier curve from a set of control points.
        :param control_points: numpy array of complex numbers.
        :param num_points: number of sample points to generate along the curve.
        :return: numpy array of complex numbers representing the Bezier curve.
        """
        n = len(control_points) - 1
        t = np.linspace(0, 1, num_points)
        curve = np.zeros(num_points, dtype=complex)
        for i, point in enumerate(control_points):
            bernstein = sps.comb(n, i) * (t**i) * ((1 - t)**(n - i))
            curve += point * bernstein
        return curve

    def infer_middlepoints_blue_cone(self, blue_cones):
        self.get_logger().info("#####################---------- ONLY BLUE CONES :----------------->")
        num_cones = len(blue_cones)
        midpoints = []
        for index in range(num_cones - 1):
            x3 = (blue_cones[index][0] + blue_cones[index + 1][0]) / 2
            y3 = (blue_cones[index][1] + blue_cones[index + 1][1]) / 2

            B = np.array(blue_cones[index][1] - blue_cones[index + 1][1],
                         blue_cones[index][0] - blue_cones[index + 1][0])
            c = B / np.linalg.norm(B)
            midpoints.append((x3, y3) + (-2 * c))
        return midpoints

    def infer_middlepoints_yellow_cone(self, yellow_cones):
        self.get_logger().info("#####################---------- ONLY YELLOW CONES :----------------->")
        num_cones = len(yellow_cones)
        midpoints = []
        for index in range(num_cones - 1):
            x3 = (yellow_cones[index][0] + yellow_cones[index + 1][0]) / 2
            y3 = (yellow_cones[index][1] + yellow_cones[index + 1][1]) / 2

            B = np.array(yellow_cones[index][1] - yellow_cones[index + 1][1],
                         yellow_cones[index][0] - yellow_cones[index + 1][0])
            c = B / np.linalg.norm(B)
            midpoints.append((x3, y3) + (2 * c))
        return midpoints
   
    def find_midpoints(self, blue_cones, yellow_cones, orange_cones, big_orange_cones):
        """
        Find the midpoints along the track with an offset that biases the path away from the inner (yellow) cones.
        """
        if len(blue_cones) <= len(yellow_cones):
            num_cones = len(blue_cones)
        else:
            num_cones = len(yellow_cones)

        blue_cones = self.sort_list(blue_cones)
        yellow_cones = self.sort_list(yellow_cones)

        len_b = len(blue_cones)
        len_y = len(yellow_cones)
        midpoints = []
        line_list = [] 

        # Offset factor to bias the midpoint away from the yellow cones.
        offset_factor = 0.1  

        if (len_b > 0 and len_y > 0):
            for cone_pos in range(0, num_cones):
                # Compute the difference vector from yellow to blue cone.
                dx = blue_cones[cone_pos][0] - yellow_cones[cone_pos][0]
                dy = blue_cones[cone_pos][1] - yellow_cones[cone_pos][1]
                # Compute the biased midpoint.
                mid_x = (blue_cones[cone_pos][0] + yellow_cones[cone_pos][0]) / 2 + offset_factor * dx
                mid_y = (blue_cones[cone_pos][1] + yellow_cones[cone_pos][1]) / 2 + offset_factor * dy
                midpoints.append([mid_x, mid_y])
                
                # For visualization purposes: add the cone positions as complex numbers.
                bluecones_c = blue_cones[cone_pos][0] + 1j * blue_cones[cone_pos][1]
                yellowcones_c = yellow_cones[cone_pos][0] + 1j * yellow_cones[cone_pos][1]
                line_list.append(bluecones_c)
                line_list.append(yellowcones_c)

                if cone_pos < num_cones - 1:
                    # For the second midpoint, use the next blue cone with the current yellow cone.
                    dx2 = blue_cones[cone_pos + 1][0] - yellow_cones[cone_pos][0]
                    dy2 = blue_cones[cone_pos + 1][1] - yellow_cones[cone_pos][1]
                    mid_x2 = (blue_cones[cone_pos + 1][0] + yellow_cones[cone_pos][0]) / 2 + offset_factor * dx2
                    mid_y2 = (blue_cones[cone_pos + 1][1] + yellow_cones[cone_pos][1]) / 2 + offset_factor * dy2
                    midpoints.append([mid_x2, mid_y2])
                    nextbluecones_c = blue_cones[cone_pos + 1][0] + 1j * blue_cones[cone_pos + 1][1]
                    line_list.append(yellowcones_c)
                    line_list.append(nextbluecones_c)
            self.publish_line_list(line_list)
        elif (len_b > 0):
            midpoints = self.infer_middlepoints_blue_cone(blue_cones)
        elif (len_y > 0):
            midpoints = self.infer_middlepoints_yellow_cone(yellow_cones)
        return midpoints

    def sort_midpoints(self, midpoints):
        """
        Sort the midpoints so that each consecutive midpoint is further from the car.
        """
        cone_dict = {}
        for cones in range(0, len(midpoints)):
            cone_dict.update({math.sqrt(((midpoints[cones][0] - 0) ** 2) + ((midpoints[cones][1] - 0) ** 2)): midpoints[cones]})
        sorted_dict = {k: cone_dict[k] for k in sorted(cone_dict)}
        sorted_midpoints = list(sorted_dict.values())
        return sorted_midpoints

    def publish_path(self, midpoints):
        waypoint_array = WaypointArrayStamped()
        waypoint_array.header.frame_id = "base_footprint"
        waypoint_array.header.stamp = self.get_clock().now().to_msg()
        for p in midpoints:
            point = Point(x=p.real, y=p.imag)
            waypoint = Waypoint(position=point)
            waypoint_array.waypoints.append(waypoint)
        self.track_line_pub.publish(waypoint_array)

    def publish_visualisation(self, midpoints):
        marker = Marker()
        marker.header.frame_id = "base_footprint"
        marker.action = Marker.ADD
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.type = Marker.POINTS
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.id = 0
        marker.scale.x = 0.35
        marker.scale.y = 0.35
        marker.ns = "midpoints"
        for midpoint in midpoints:
            marker.points.append(Point(x=midpoint.real, y=midpoint.imag))
        self.visualization_pub.publish(marker)
    
    def publish_car_pose(self, car_pose):
        marker = Marker()
        marker.header.frame_id = "base_footprint"
        marker.action = Marker.ADD
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.type = Marker.POINTS
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.id = 1
        marker.scale.x = 0.35
        marker.scale.y = 0.35
        marker.ns = "car_pose"
        marker.points.append(Point(x=car_pose.real, y=car_pose.imag))
        self.car_pose_pub.publish(marker)

    def publish_test_pose(self, test_pose):
        marker = Marker()
        marker.header.frame_id = "base_footprint"
        marker.action = Marker.ADD
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.type = Marker.POINTS
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.id = 2
        marker.scale.x = 0.35
        marker.scale.y = 0.35
        marker.ns = "test_pose"
        marker.points.append(Point(x=test_pose.real, y=test_pose.imag))
        self.test_pose_pub.publish(marker)

    def publish_line_stip(self, line_stip):
        marker = Marker()
        marker.header.frame_id = "base_footprint"
        marker.action = Marker.ADD
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.type = Marker.LINE_STRIP
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.id = 3
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.ns = "line_stip"
        for line_stip in line_stip:
            marker.points.append(Point(x=line_stip.real, y=line_stip.imag))
        self.line_strip_pub.publish(marker)

    def publish_line_list(self, line_list):
        marker = Marker()
        marker.header.frame_id = "base_footprint"
        marker.action = Marker.ADD
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.type = Marker.LINE_LIST
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.id = 4
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.ns = "line_list"
        for line_list in line_list:
            marker.points.append(Point(x=line_list.real, y=line_list.imag))
        self.line_list_pub.publish(marker)
    
    def sort_list(self, points):
        cone_dict = {}
        for cones in range(0, len(points)):
            cone_dict.update({math.sqrt(((points[cones][0] - 0) ** 2) + ((points[cones][1] - 0) ** 2)): points[cones]})
        sorted_dict = {k: cone_dict[k] for k in sorted(cone_dict)}
        sorted_points = list(sorted_dict.values())
        return sorted_points

    def convert(self, cones):
        """
        Converts a cone array message into a np array of complex numbers.
        """
        return np.array([c.point.x + 1j * c.point.y for c in cones])

    def to_2d_list(self, arr):
        """
        Convert an array of complex numbers to a 2D list of the form [[x1,y1], ...].
        """
        return [[x.real, x.imag] for x in arr]

def main():
    rclpy.init(args=None)
    node = Planner("local_planner")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
