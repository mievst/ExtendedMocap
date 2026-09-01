"""Blender integration modules.

These modules import ``bpy`` and must only be imported from inside Blender's
Python environment. They are intentionally NOT imported by
``extended_mocap.__init__`` so the core pipeline works without Blender.
"""
