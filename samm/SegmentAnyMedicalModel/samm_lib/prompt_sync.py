import slicer
import vtk

from .slice_payload import array_position_from_ras, slice_spec


def sync_2d_prompts_to_slice(volumeNode, viewName, positiveNode, negativeNode, boxNode):
    volumeShape = slicer.util.arrayFromVolume(volumeNode).shape
    spec = slice_spec(volumeNode, volumeShape, viewName)
    sync_point_node(volumeNode, spec, positiveNode)
    sync_point_node(volumeNode, spec, negativeNode)
    sync_plane_node(volumeNode, spec, boxNode)


def sync_point_node(volumeNode, sliceSpec, node):
    ras = vtk.vtkVector3d(0.0, 0.0, 0.0)
    for index in range(node.GetNumberOfControlPoints()):
        node.GetNthControlPointPosition(index, ras)
        moved = ras_on_slice(volumeNode, [ras[0], ras[1], ras[2], 1.0], sliceSpec)
        if ras_changed(ras, moved):
            node.SetNthControlPointPosition(index, moved[0], moved[1], moved[2])


def sync_plane_node(volumeNode, sliceSpec, node):
    if node.GetNumberOfControlPoints() == 0:
        return
    origin = [0.0] * 3
    node.GetOrigin(origin)
    moved = ras_on_slice(volumeNode, [*origin, 1.0], sliceSpec)
    if ras_changed(origin, moved):
        node.SetOrigin(moved)


def ras_on_slice(volumeNode, ras, sliceSpec):
    point = array_position_from_ras(volumeNode, ras)
    point[sliceSpec["axis"]] = sliceSpec["index"]
    return ras_from_array_position(volumeNode, point)[:3]


def ras_from_array_position(volumeNode, point):
    matrix = vtk.vtkMatrix4x4()
    volumeNode.GetIJKToRASMatrix(matrix)
    return matrix.MultiplyPoint([point[2], point[1], point[0], 1.0])


def ras_changed(current, moved):
    return any(abs(current[axis] - moved[axis]) > 1e-6 for axis in range(3))
