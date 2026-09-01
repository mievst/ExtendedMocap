import logging
import os

import bpy
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")


class BlenderAnimationProcessor:
    def __init__(self, model_name, collection_name):
        logging.info(
            "Initializing BlenderAnimationProcessor with model: %s, collection: %s",
            model_name,
            collection_name,
        )
        self.model = bpy.context.scene.objects.get(model_name)
        self.collection = bpy.data.collections[collection_name]
        self.setup_scene()

    def setup_scene(self):
        logging.info("Setting up the scene")
        for obj in bpy.context.scene.objects:
            obj.select_set(obj == self.model)
        bpy.context.view_layer.objects.active = self.model
        self.model.select_set(True)

    def clamp_location(self, location):
        """Clamp the location coordinates to be within the unit cube bounds."""
        clamped_location = tuple(max(min(coord, 0.5), -0.5) for coord in location)
        return clamped_location

    def process_animations(self):
        logging.info("Processing animations")
        for obj in bpy.context.scene.objects:
            if obj != self.model:
                obj.select_set(False)
        bpy.context.view_layer.objects.active = self.model
        self.model.select_set(True)
        for obj in self.collection.objects:
            armature_name = obj.name
            logging.info("Processing armature: %s", armature_name)
            armature = bpy.context.scene.objects.get(armature_name)
            bpy.context.view_layer.objects.active = armature
            bpy.ops.object.make_links_data(type="ANIMATION")
            frames = int(bpy.context.object.animation_data.action.frame_range[-1])

            file_name = os.path.basename(bpy.data.filepath)
            export_name = os.path.join(
                os.path.dirname(bpy.data.filepath),
                "data",
                "mocap",
                "renders",
                f"{file_name}_{armature_name}.mp4",
            )
            bpy.context.scene.frame_set(0)
            initial_location = armature.pose.bones["CC_Base_Hip"].location
            offset = tuple(-coord for coord in initial_location)
            for frame in tqdm(range(frames)):
                bpy.context.scene.frame_set(frame)
                bone = self.model.pose.bones["CC_Base_Hip"]
                new_location = tuple(
                    coord + offset[i]
                    for i, coord in enumerate(armature.pose.bones["CC_Base_Hip"].location)
                )
                bone.location = self.clamp_location(new_location)
                bpy.context.view_layer.update()
                bone.keyframe_insert(data_path="location", frame=bpy.context.scene.frame_current)

            self.render_animation(frames, export_name)
            armature.select_set(False)

    def render_animation(self, frames, export_name):
        logging.info("Rendering animation to %s", export_name)
        bpy.context.scene.render.engine = "BLENDER_EEVEE"
        bpy.context.scene.eevee.taa_render_samples = 8
        bpy.context.scene.eevee.sss_samples = 1
        bpy.context.scene.eevee.use_volumetric_lights = False
        bpy.context.scene.frame_start = 0
        bpy.context.scene.frame_end = frames
        bpy.context.scene.frame_step = 1
        bpy.context.scene.render.use_sequencer = False
        bpy.context.scene.render.use_compositing = False
        bpy.context.scene.render.fps = 60
        bpy.context.scene.frame_step = 1
        bpy.context.scene.render.image_settings.file_format = "FFMPEG"
        bpy.context.scene.render.ffmpeg.format = "MPEG4"
        bpy.context.scene.render.ffmpeg.codec = "MPEG4"
        bpy.context.scene.render.ffmpeg.video_bitrate = 10000
        bpy.context.scene.render.ffmpeg.audio_codec = "AAC"
        bpy.context.scene.render.ffmpeg.audio_bitrate = 192
        bpy.context.scene.render.filepath = export_name
        bpy.ops.render.render(animation=True)


# Usage
# processor = BlenderAnimationProcessor("party-m-0001", "AccuRig")
# processor.process_animations()
