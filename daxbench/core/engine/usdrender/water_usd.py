import os
import shutil

import numpy as np
from pxr import Usd, UsdGeom, Gf, Sdf

my_path = os.path.dirname(os.path.abspath(__file__))


def create_usd_liquid_scene(states, output_file: str):
    # Create a new stage
    # stage = Usd.Stage.CreateNew(output_file)
    shutil.copyfile(f"{my_path}/scene/water_bowl.usda", output_file)
    stage = Usd.Stage.Open(output_file)

    # Get the root prim
    root = stage.GetPrimAtPath("/World")

    # Set the timecode for the stage
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(len(states))
    for t, _ in enumerate(states):
        # create particle system at time t
        # Assuming 'stage' is an existing Usd.Stage object
        particle_system_path = f"/World/ParticleSystem{t}"
        particle_system_prim = stage.DefinePrim(particle_system_path, "PhysxParticleSystem")

        # Set apiSchemas metadata
        api_schemas = Sdf.TokenListOp()
        api_schemas.prependedItems = ["PhysxParticleIsosurfaceAPI"]
        particle_system_prim.SetMetadata("apiSchemas", api_schemas)

        # Set attributes and relationships
        particle_system_prim.CreateRelationship("material:binding").AddTarget("/World/Looks/Water_Ocean_Green_Reef")
        # api_schemas = Sdf.TokenListOp()
        # api_schemas.prependedItems = ["weakerThanDescendants"]
        # particle_system_prim.SetMetadata("bindMaterialAs", api_schemas)
        particle_system_prim.CreateAttribute("particleContactOffset", Sdf.ValueTypeNames.Float).Set(0.005)
        particle_system_prim.CreateAttribute("particleSystemEnabled", Sdf.ValueTypeNames.Bool).Set(False)
        particle_system_prim.CreateAttribute("visibility", Sdf.ValueTypeNames.Token).Set("inherited")

    for t, state in enumerate(states):

        """ liquid """
        # Check if the ParticleSet exists in the stage
        particle_set_path = root.GetPath().AppendChild(f"particleSet{t}")
        particles = UsdGeom.Points.Get(stage, particle_set_path)

        if not particles:
            # Create a new ParticleSet with type Points if it doesn't exist
            particles = UsdGeom.Points.Define(stage, particle_set_path)
        particles.GetPointsAttr().Set(np.array(state.x[0]).tolist())
        particles.GetVelocitiesAttr().Set(np.array(state.v[0]).tolist())

        # Set values for the particle set
        api_schemas = Sdf.TokenListOp()
        api_schemas.prependedItems = ["PhysxParticleSetAPI", "PhysicsMassAPI"]
        particles.GetPrim().SetMetadata("apiSchemas", api_schemas)
        particles.GetPrim().CreateAttribute("physics:density", Sdf.ValueTypeNames.Float).Set(0)
        particles.GetPrim().CreateAttribute("physics:mass", Sdf.ValueTypeNames.Float).Set(0)
        particles.GetPrim().CreateAttribute("physxParticle:fluid", Sdf.ValueTypeNames.Bool).Set(True)
        particles.GetPrim().CreateAttribute("physxParticle:particleEnabled", Sdf.ValueTypeNames.Bool).Set(False)
        particles.GetPrim().CreateAttribute("physxParticle:particleGroup", Sdf.ValueTypeNames.Int).Set(0)
        particles.GetPrim().CreateRelationship("physxParticle:particleSystem").AddTarget(f'/World/ParticleSystem{t}')
        particles.GetPrim().CreateAttribute("physxParticle:selfCollision", Sdf.ValueTypeNames.Bool).Set(False)
        particles.GetVisibilityAttr().Set("invisible")

        for t2, _ in enumerate(states):
            ps = stage.GetPrimAtPath(f"/World/ParticleSystem{t2}")
            ps = UsdGeom.Xformable(ps.GetPrim())
            xform_attr = ps.GetXformOpOrderAttr()

            xform_ops = xform_attr.Get()
            xform_ops = [] if xform_ops is None else xform_ops
            translate_op_name = 'xformOp:translate'
            translate_op_exists = any([op == translate_op_name for op in xform_ops])
            position = [0, 0, 0] if t2 == t else [0, -100, 0]

            if translate_op_exists:
                # Update the existing translate operation.
                translate_op = UsdGeom.XformOp(ps.GetPrim().GetAttribute(translate_op_name))
                translate_op.Set(Gf.Vec3d(position[0], position[1], position[2]), time=t)
            else:
                # Add a new translate operation.
                ps.AddTranslateOp().Set(Gf.Vec3d(position[0], position[1], position[2]))

        """ bowl """
        # TODO address the path issue
        for i in range(0, 2):
            bowl = stage.GetPrimAtPath(f"/World/bowl{i + 1}")
            bowl_pos = np.array(state.primitives[i].position[0, 0])
            if i == 1: bowl_pos[1] += 0.03
            bowl_pos = bowl_pos.tolist()

            bowl_xform = UsdGeom.Xformable(bowl.GetPrim())
            xform_attr = bowl_xform.GetXformOpOrderAttr()

            xform_ops = xform_attr.Get()
            translate_op_name = 'xformOp:translate'
            translate_op_exists = any([op == translate_op_name for op in xform_ops])
            if translate_op_exists:
                # Update the existing translate operation.
                translate_op = UsdGeom.XformOp(bowl_xform.GetPrim().GetAttribute(translate_op_name))
                translate_op.Set(Gf.Vec3d(bowl_pos[0], bowl_pos[1], bowl_pos[2]), time=t)
            else:
                # Add a new translate operation.
                bowl_xform.AddTranslateOp().Set(Gf.Vec3d(bowl_pos[0], bowl_pos[1], bowl_pos[2]))

            # check if scale attribute exists, if not add it to 0.008125, 0.007625, 0.008125
            scale_op_name = 'xformOp:scale'
            scale_op_exists = any([op == scale_op_name for op in xform_ops])
            if scale_op_exists:
                # Update the existing translate operation.
                scale_op = UsdGeom.XformOp(bowl_xform.GetPrim().GetAttribute(scale_op_name))
                scale_op.Set(Gf.Vec3d(0.009, 0.008, 0.009))
            else:
                # Add a new translate operation.
                bowl_xform.AddScaleOp().Set(Gf.Vec3d(0.009, 0.008, 0.009))

            # check if rotate attribute exists, if not add it
            bowl_rot = np.array(state.primitives[i].rotation[0, 0]).tolist()  # quaternion
            bowl_rot = Gf.Quatf(bowl_rot[0], bowl_rot[1], bowl_rot[2], bowl_rot[3])

            rotate_op_name = 'xformOp:orient'
            rotate_op_exists = any([op == rotate_op_name for op in xform_ops])
            if rotate_op_exists:
                # Update the existing translate operation.
                rotate_op = UsdGeom.XformOp(bowl_xform.GetPrim().GetAttribute(rotate_op_name))
                rotate_op.Set(bowl_rot, time=t)
            else:
                # Add a new translate operation.
                bowl_xform.AddOrientOp().Set(bowl_rot)

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


if __name__ == "__main__":
    create_usd_liquid_scene(None, "scene/water_bowl2.usda")
