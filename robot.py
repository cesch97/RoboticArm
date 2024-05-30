import jax
import jax.numpy as jnp
from jax import grad, jit
import optax
import matplotlib.pyplot as plt

# CONSTANTS #
REF_LEN = 0.05


def get_rotation_matrix(rot):
    rot_rad = jnp.radians(rot)
    x, y, z = rot_rad[0], rot_rad[1], rot_rad[2]
    Rx = jnp.array([
        [1, 0, 0],
        [0, jnp.cos(x), -jnp.sin(x)],
        [0, jnp.sin(x), jnp.cos(x)]
    ])
    Ry = jnp.array([
        [jnp.cos(y), 0, jnp.sin(y)],
        [0, 1, 0],
        [-jnp.sin(y), 0, jnp.cos(y)]
    ])
    Rz = jnp.array([
        [jnp.cos(z), -jnp.sin(z), 0],
        [jnp.sin(z), jnp.cos(z), 0],
        [0, 0, 1]
    ])
    return Rz @ Ry @ Rx

def translate_link(link, tra):
    return link + tra

def rotate_link(link, rot):
    R = get_rotation_matrix(rot)
    return R @ link

def combine_rotations(rot1, rot2):
    R1 = get_rotation_matrix(rot1)
    R2 = get_rotation_matrix(rot2)
    R = R2 @ R1
    R = R / jnp.linalg.norm(R, axis=1).reshape(3, 1)
    sy = jnp.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    
    def true_fun(_):
        x = jnp.arctan2(-R[1,2], R[1,1])
        y = jnp.arctan2(-R[2,0], sy)
        z = 0.
        return jnp.degrees(jnp.array([x, y, z]))

    def false_fun(_):
        x = jnp.arctan2(R[2,1] , R[2,2])
        y = jnp.arctan2(-R[2,0], sy)
        z = jnp.arctan2(R[1,0], R[0,0])
        return jnp.degrees(jnp.array([x, y, z]))

    return jax.lax.cond(singular, true_fun, false_fun, None)

@jit
def forward_kinematics(angles):
    # defining links
    link_1 = jnp.array([0.03, 0, 0.095])
    link_2 = jnp.array([-0.12, 0, 0])
    link_3 = jnp.array([-0.02, 0, 0.025])
    link_4 = jnp.array([0.145, 0, 0])
    link_5 = jnp.array([0.03, 0, 0])
    link_6 = jnp.array([0.01, 0, 0])
    # defining tool reference from 6 axis
    tool_off = jnp.array([0, 0, 0])
    tool_rot = jnp.array([0, 0, 45])

    # Calculating links positions
    # 1
    rot_1 = jnp.array([0, 0, angles[0]])
    end_1 = rotate_link(link_1, rot_1)
    # 2
    rot_2 = jnp.array([0, angles[1], 0])
    rot_2 = combine_rotations(rot_2, rot_1)
    end_2 = translate_link(rotate_link(link_2, rot_2), end_1)
    # 3
    rot_3 = jnp.array([0, angles[2] - rot_2[1], 0])
    rot_3 = combine_rotations(rot_3, rot_2)
    end_3 = translate_link(rotate_link(link_3, rot_3), end_2)
    # 4
    rot_4 = jnp.array([angles[3], 0, 0])
    rot_4 = combine_rotations(rot_4, rot_3)
    end_4 = translate_link(rotate_link(link_4, rot_4), end_3)
    # 5
    rot_5 = jnp.array([0, angles[4], 0])
    rot_5 = combine_rotations(rot_5, rot_4)
    end_5 = translate_link(rotate_link(link_5, rot_5), end_4)
    # 6
    rot_6 = jnp.array([angles[5], 0, 0])
    rot_6 = combine_rotations(rot_6, rot_5)
    end_6 = translate_link(rotate_link(link_6, rot_6), end_5)

    # to express the orientation of the tool we add 3 perpendicular lines
    # starting from the position of the end tool of length "ref_len"
    rot_t = combine_rotations(tool_rot, rot_6)
    end_t = translate_link(rotate_link(tool_off, rot_t), end_6)
    tool_x = translate_link(rotate_link(jnp.array([REF_LEN, 0, 0]), rot_t), end_t)
    tool_y = translate_link(rotate_link(jnp.array([0, REF_LEN, 0]), rot_t), end_t)
    tool_z = translate_link(rotate_link(jnp.array([0, 0, REF_LEN]), rot_t), end_t)

    # Packing the results
    robot = jnp.array([end_1, end_2,end_3, end_4, end_5, end_6])
    tool_pos = jnp.array([end_t, tool_x, tool_y, tool_z])
    tool_rot = rot_t
    return robot, tool_pos, tool_rot

def loss_fn(angles, t_tool):
    _, tool, _ = forward_kinematics(angles)
    return jnp.sum((t_tool - tool)**2)

@jit
def check_tolerance(angles, pos, rot, pos_tol, ang_tol):
    _, tool_pos, tool_rot = forward_kinematics(angles)
    pos = jnp.array(pos)
    rot = jnp.array(rot)
    pos_err = jnp.linalg.norm(pos[0] - tool_pos[0])
    ang_err = jnp.max(jnp.abs(rot - tool_rot))
    return jnp.logical_and(pos_err <= pos_tol, ang_err <= ang_tol)

def inverse_kinematics(tool_pos, tool_rot, i_angles, pos_toll=0.001, ang_toll=0.1, check_step=10, max_steps=1000, lr=1e1):
    dloss_fn = jit(grad(loss_fn))
    # Initialize optimizer and optimizer state
    optimizer = optax.adam(lr)
    params = {'angles': i_angles}
    opt_state = optimizer.init(params)
    # Main loop
    for i in range(max_steps):
        optimizer = optax.adam(lr)  # Update the optimizer with the new learning rate
        # Compute gradients
        grads = dloss_fn(params['angles'], tool_pos)
        # Update parameters
        updates, opt_state = optimizer.update({'angles': grads}, opt_state)
        params = optax.apply_updates(params, updates)
        # Checking tolerance
        if i % check_step == 0:
            # print(f"Step {i}, Loss: {loss_fn(params['angles'], tool_pos)}, LR: {lr}")
            if check_tolerance(params['angles'], tool_pos, tool_rot, pos_toll, ang_toll):
                return params['angles']
            
    # if not check_tolerance(params['angles'], tool_pos, tool_rot, pos_toll, ang_toll):
    #     print("Impossible to reach the desired point!")    
    return params['angles']
    

def find_angles(pos, rot, init_angs=jnp.zeros(6)):
    t_x = translate_link(rotate_link(jnp.array([REF_LEN, 0, 0]), rot), pos)
    t_y = translate_link(rotate_link(jnp.array([0, REF_LEN, 0]), rot), pos)
    t_z = translate_link(rotate_link(jnp.array([0, 0, REF_LEN]), rot), pos)
    tool = jnp.array([pos, t_x, t_y, t_z])
    angles = inverse_kinematics(tool, rot, init_angs)
    return angles

def plot_links(robot, tool, colors):
    links = [robot[i, :] for i in range(robot.shape[0])]
    tool_off, tool_x, tool_y, tool_z = tool[0], tool[1], tool[2], tool[3]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    min_val, max_val = jnp.inf, -jnp.inf

    def plot_line(start, end, color, min_val=jnp.inf, max_val=-jnp.inf):
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color=color)

        min_val, max_val = min(min_val, start[0]), max(max_val, start[0])
        min_val, max_val = min(min_val, end[0]), max(max_val, end[0])
        min_val, max_val = min(min_val, start[1]), max(max_val, start[1])
        min_val, max_val = min(min_val, end[1]), max(max_val, end[1])
        min_val, max_val = min(min_val, start[2]), max(max_val, start[2])
        min_val, max_val = min(min_val, end[2]), max(max_val, end[2])
        return min_val, max_val

    min_val, max_val = plot_line(jnp.array([0, 0, 0]), links[0], colors[0], min_val, max_val)
    for i, (link, color) in enumerate(list(zip(links, colors))[1:]):
        start, end = links[i], link
        min_val, max_val = plot_line(start, end, color, min_val, max_val)

    min_val, max_val = plot_line(tool_off, tool_x, 'r', min_val, max_val)
    min_val, max_val = plot_line(tool_off, tool_y, 'g', min_val, max_val)
    min_val, max_val = plot_line(tool_off, tool_z, 'b', min_val, max_val)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_xlim([min_val, max_val])
    ax.set_ylim([min_val, max_val])
    ax.set_zlim([min_val, max_val])

    plt.show()

def apply_angles(angles):
    robot, tool_pos, _ = forward_kinematics(angles)
    plot_links(robot, tool_pos, ['r', 'g', 'b', 'y', 'c', 'k'])


if __name__ == "__main__":

    i_angles = jnp.array([0, 0, 0, 0, 0, 0], dtype=jnp.float32)
    o_angles = find_angles(jnp.array([0.15, 0.1, 0.1]), jnp.array([0, -45, 45]), i_angles)
    apply_angles(o_angles)
