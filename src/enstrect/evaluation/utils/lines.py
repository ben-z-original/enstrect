import ezdxf
import numpy as np
import open3d as o3d
import networkx as nx
from enstrect.evaluation.utils.linemesh import LineMesh


def lineset_to_linemesh(lineset, color=[1, 0, 0], radius=0.0005):
    """Converts a open3d LineSet to a LineMesh for controllable line widths."""
    colors = [color] * len(lineset.lines)
    lineset.colors = o3d.utility.Vector3dVector(colors)
    linemesh = LineMesh(lineset.points, lineset.lines, lineset.colors, radius=radius)
    line_mesh_geoms = linemesh.cylinder_segments
    return line_mesh_geoms


def G_to_linemesh_o3d(G, color=[1, 0, 0], radius=0.0005):
    line_sets = G_to_lineset_o3d(G)

    line_geoms = []
    for line_set in line_sets:
        line_geoms.extend(lineset_to_linemesh(line_set, color=color, radius=radius))

    return line_geoms


def G_to_lineset_o3d(G):
    # connected subgraphs
    S = [nx.convert_node_labels_to_integers(G.subgraph(c), first_label=0) for c in nx.connected_components(G)]

    line_sets = []
    for SG in S:
        positions = np.array([SG.nodes[node_id]['pos'] for node_id in SG.nodes])
        degrees = np.array([val for (node, val) in SG.degree()])

        # is polygon (boundary)
        if np.all(1 < degrees):
            cycle_edges = nx.find_cycle(SG)
            lines = np.array(cycle_edges)
        # is polyline (centerline)
        else:
            src, trg = np.where(degrees == 1)[0]
            path = nx.shortest_path(SG, source=list(SG.nodes)[src], target=list(SG.nodes)[trg])
            path = np.int64(np.array(path)) + 1
            lines = [[path[i] - 1, path[i + 1] - 1] for i in range(len(path) - 1)]

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(positions)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_sets.append(line_set)

    return line_sets


def obj_to_G(obj_path):
    with open(str(obj_path), 'r') as f:
        obj = f.readlines()

    vertices = np.array([line.split()[1:] for line in obj if line.startswith("v ")], dtype=np.float64)
    lines = [np.array(line.split()[1:], dtype=np.int64) - 1 for line in obj if line.startswith("l ")]

    # line = lines[0]
    edges = [(line[i], line[i + 1]) for line in lines for i, _ in enumerate(line[:-1])]

    G = nx.Graph()
    G.add_edges_from(edges)
    nx.set_node_attributes(G, {i: v for i, v in enumerate(vertices)}, "pos")
    return G


def G_to_obj(G, polyline_path):
    if len(G.nodes) == 0: return None

    line_sets = G_to_lineset_o3d(G)

    vertices_obj, lines_obj = "", ""
    count = 1  # obj starts counting at 1

    for line_set in line_sets:
        vertices_obj += "".join([f"v {x} {y} {z}\n" for x, y, z in np.array(line_set.points)])
        lines = np.array(line_set.lines).flatten()
        lines = np.append(lines[[0]], lines[1:][np.diff(lines) != 0]) + count
        lines_obj += "l " + " ".join(map(str, lines)) + "\n"
        count += len(line_set.points)

    with open(str(polyline_path), "w") as f:
        f.write(vertices_obj + lines_obj)


def save_lines_as_obj(lines, out_dir, category):
    close_polygon = False if category == "crack" else True

    all_vertices = [f"v {vert[0]} {vert[1]} {vert[2]}\n" for vert in np.vstack(lines)]
    all_lines, counter = [], 1

    for verts in lines:
        line = list(range(counter, counter + len(verts)))
        if close_polygon:
            line.extend([counter])
        all_lines.append(f"l {' '.join([str(elem) for elem in line])}\n")
        counter += len(verts)

    with open(str(out_dir / "annotations" / f"{category}.obj"), "w") as fi:
        fi.writelines(all_vertices)
        fi.writelines(all_lines)


def load_polylines_dxf(annotation_dir, translation, rotation, scale):
    files = list(annotation_dir.glob("**/*.dxf"))
    classes = set([f.parts[-2] for f in files])
    lines = {key: [] for key in classes}

    # collect vertices
    for f in files:
        msp = ezdxf.readfile(str(f)).modelspace()
        polyline = msp.query("POLYLINE").first
        vertices = np.array([vert.format() for vert in polyline.vertices])
        vertices -= translation.cpu().numpy()
        vertices @= rotation.cpu().numpy()
        vertices *= scale
        lines[f.parts[-2]].append(vertices)

    return classes, lines


def compute_line_length(vertices):
    """Compute the length of a polyline."""
    return np.sum(np.linalg.norm(np.diff(vertices, axis=0), axis=1))


def interpolate_G(G, gap=0.01):
    S = [nx.convert_node_labels_to_integers(G.subgraph(c), first_label=0) for c in nx.connected_components(G)]
    H = nx.Graph()

    for SG in S:
        if len(SG) == 0:
            continue
        positions = np.array([SG.nodes[node_id]['pos'] for node_id in SG.nodes])
        degrees = np.array([val for (node, val) in SG.degree()])
        if np.all(degrees < 1) or np.all(2 < degrees):  # TODO: for boundaries different
            raise RuntimeError("Graph contains cycles")
        elif np.all(2 == degrees):
            # remove arbitrary edge
            SG.remove_edge(*list(G.edges())[0])
            degrees = np.array([val for (node, val) in SG.degree()])
        try:
            src, trg = np.where(degrees == 1)[0]
        except:
            raise RuntimeError("Found unprocessable subgraph")
            continue
        path = np.array(nx.shortest_path(SG, source=list(SG.nodes)[src], target=list(SG.nodes)[trg]))

        line_pos = positions[path]
        line_inter = interpolate_line_points(line_pos, gap=gap)

        size_prev = len(H)
        nx.add_path(H, np.arange(len(line_inter)) + size_prev)
        nx.set_node_attributes(H, {i + size_prev: pos for i, pos in enumerate(line_inter)}, "pos")

    return H


def interpolate_line_points(pts, gap=0.01):
    """Interpolates the vertices of a line given a specific sampling interval."""
    diff = np.sqrt(np.sum(np.power(np.diff(pts, axis=0), 2), axis=1))
    xp = np.append(0, np.cumsum(diff))
    x = np.arange(0 + xp[-1] % gap / 2, xp[-1], gap)
    x = np.hstack((0, x, xp[-1]))

    # interpolate points
    points = np.stack([np.interp(x, xp, pts[:, 0]),
                       np.interp(x, xp, pts[:, 1]),
                       np.interp(x, xp, pts[:, 2])], axis=1)

    return points
