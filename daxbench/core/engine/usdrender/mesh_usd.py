import os
import shutil

from pxr import Usd, UsdGeom, Gf

my_path = os.path.dirname(os.path.abspath(__file__))


def create_usd_cloth_scene(vertices, indices, output_file: str):
    # Create a new stage
    # stage = Usd.Stage.CreateNew(output_file)
    shutil.copyfile(f"{my_path}/scene/cloth.usda", output_file)
    stage = Usd.Stage.Open(output_file)

    # Get the root prim
    root = stage.GetPrimAtPath("/World")

    # Set the timecode for the stage
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(len(vertices))

    # Define a Mesh
    cloth_path = root.GetPath().AppendChild("mesh")
    # mesh = UsdGeom.Mesh.Define(stage, root.GetPath().AppendChild("mesh"))
    # mesh_xform = UsdGeom.Xformable(bowl.GetPrim())
    mesh = UsdGeom.Mesh.Get(stage, cloth_path)

    # Set mesh attributes
    for t in range(vertices.shape[0]):
        mesh.GetPointsAttr().Set(vertices[t].tolist(), time=t)

    mesh.GetFaceVertexCountsAttr().Set([3] * indices.shape[0])
    mesh.GetFaceVertexIndicesAttr().Set(indices.flatten().tolist())

    """ camera """
    # check if camera exists, if not add it with the above parameters
    camera = stage.GetPrimAtPath("/World/camera")
    if not camera:
        camera = UsdGeom.Camera.Define(stage, '/World/camera')
        camera_xform = UsdGeom.Xformable(camera.GetPrim())
        camera_xform.AddTranslateOp().Set(Gf.Vec3d(-1.5, 1.5, -1.5))
        camera_xform.AddRotateXYZOp().Set(Gf.Vec3f(8, -128, -25))
        camera.GetPrim().GetAttribute('focalLength').Set(60)

    # Save the USD file with a new file path
    stage.SetDefaultPrim(root.GetPrim())
    stage.Save()
