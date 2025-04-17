# 🏎️ Local Path Planner for Autonomous Racing – UH Racing Autonomous

This ROS 2 package implements a local path planner for autonomous racecars, designed for use in competitions such as Formula Student AI (FS-AI). It processes cone positions around the car to generate a smooth and reliable path for the vehicle to follow, using Python and Bezier curve fitting.

---

## 📌 Overview

The planner node listens to cone positions received from perception modules and calculates a centerline path between the cones. It then smooths this path using a Bezier curve and publishes it as waypoints for the control system to follow. This helps the autonomous car stay within the track boundaries and navigate various track shapes, including curves and sharp turns.

---

## 📂 File Structure

- `planner.py` – Main ROS 2 node that calculates and publishes the path.
- `main()` – Entry point that initializes the ROS node.

---

## ⚙️ How It Works

### 1. **Cone Input Handling**

The node subscribes to the `/cones` topic, which provides cone positions via `ConeArrayWithCovariance`. The cones are divided into:
- Blue cones (left side)
- Yellow cones (right side)
- Orange and big orange cones (start/finish)
- Uncolored cones (from LIDAR, needing classification)

Nearby uncolored/orange cones are assigned to blue or yellow sides based on their position relative to the car.

### 2. **Midpoint Generation**

The planner calculates **midpoints** between pairs of blue and yellow cones:
- If both blue and yellow cones are available, it computes a midpoint biased slightly toward the blue cone side.
- If only one color is visible (e.g. in sparse data situations), it infers a path based on only those cones.

### 3. **Bezier Curve Smoothing**

To create a smooth trajectory, the midpoint path is passed into a Bezier curve generator using **Bernstein polynomials** (via `scipy.special`). This ensures the car follows a continuous, smooth path instead of jerky turns between straight segments.

### 4. **Visualization**

Multiple `Marker` topics are published to visualize:
- Midpoints
- Path lines
- Car pose
- Test pose
- Cone pairings (for debugging)
These can be viewed in **RViz** for validation and tuning.

---

## 🧠 Memory Handling

In cases where not enough cones are in the field of view (e.g., sharp turns), the car "remembers" the previous path direction to continue moving safely until enough cones appear to re-calculate midpoints.

---

## 🚀 ROS 2 Topics

| Type | Topic | Description |
|------|-------|-------------|
| Subscriber | `/cones` | Raw cone positions from perception |
| Publisher | `/trajectory` | Path as `WaypointArrayStamped` |
| Publisher | `/planner/viz` | Midpoint markers |
| Publisher | `/planner/CarPose` | Current car pose visualization |
| Publisher | `/planner/LineList`, `/LineStrip`, `/TestPose` | Additional debug visualizations |

---

## 🧪 Dependencies

- Python 3
- ROS 2 Galactic or later
- `rclpy`, `geometry_msgs`, `visualization_msgs`, `eufs_msgs`
- `numpy`, `scipy`, `math`

Install Python dependencies using:

```bash
pip install numpy scipy
