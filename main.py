from robot import *

class RobotPlotter:
    def __init__(self):
        # Initialize figure and axis
        plt.ion()  # Turn on interactive mode
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.lines = []  # Store lines objects to update them later
        
    def plot_links(self, robot, tool, colors=['r', 'g', 'b', 'y', 'c', 'k'], init=False):
        links = [robot[i, :] for i in range(robot.shape[0])]
        tool_off, tool_x, tool_y, tool_z = tool[0], tool[1], tool[2], tool[3]

        def plot_line(i, start, end, color, init):
            if init:
                line, = self.ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color=color)
                self.lines.append(line)
                return
            self.lines[i].set_data([start[0], end[0]], [start[1], end[1]])
            self.lines[i].set_3d_properties([start[2], end[2]])
            self.lines[i].set_color(color)

        line_id = 0
        plot_line(line_id, jnp.array([0, 0, 0]), links[0], colors[0], init)
        for i, (link, color) in enumerate(list(zip(links, colors))[1:]):
            start, end = links[i], link
            line_id += 1
            plot_line(line_id, start, end, color, init)

        plot_line(line_id+1, tool_off, tool_x, 'r', init)
        plot_line(line_id+2, tool_off, tool_y, 'g', init)
        plot_line(line_id+3, tool_off, tool_z, 'b', init)

        # Updating labels and limits
        if init:
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            self.ax.set_zlabel('Z')

        min_val = jnp.min(jnp.array([*links, tool_off, tool_x, tool_y, tool_z]))
        max_val = jnp.max(jnp.array([*links, tool_off, tool_x, tool_y, tool_z]))

        self.ax.set_xlim([min_val, max_val])
        self.ax.set_ylim([min_val, max_val])
        self.ax.set_zlim([min_val, max_val])

        plt.draw()
        plt.pause(0.01)

# Example usage

if __name__ == "__main__":

    program = [
                (jnp.array([0.15, 0.1, 0.1]), jnp.array([0, -45, 45])),
               (jnp.array([0.16, 0.1, 0.1]), jnp.array([0, -45, 45])),
               (jnp.array([0.17, 0.1, 0.1]), jnp.array([0, -45, 45])),
               (jnp.array([0.18, 0.1, 0.1]), jnp.array([0, -45, 45])),
               (jnp.array([0.19, 0.1, 0.1]), jnp.array([0, -45, 45])),
               (jnp.array([0.20, 0.1, 0.1]), jnp.array([0, -45, 45]))
            ]

    plotter = RobotPlotter()
    angles = jnp.array([0, 0, 0, 0, 0, 0], dtype=jnp.float32)
    robot, tool, _ = forward_kinematics(angles)
    plotter.plot_links(robot, tool, init=True)
    while True:
        for pos, rot in program:
            angles = find_angles(pos, rot, angles)
            robot, tool, _ = forward_kinematics(angles)
            plotter.plot_links(robot, tool)

# Use in a loop to continuously update plot
# while True:
#    plotter.plot_links(robot, tool, colors)
