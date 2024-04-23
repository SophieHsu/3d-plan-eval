import os
import numpy as np
from types import SimpleNamespace
from igibson import ig_dataset_path


OBJECT_KEYS = SimpleNamespace(
    COUNTER='counter',
    TABLE='table',
    STOVE='stove',
    BOWL='bowl',
    COLOR_BOWL='color_bowl',
    LARGE_BOWL='large_bowl',
    STEAK_BOWL='steak_bowl',
    PAN='pan',
    SINK='sink',
    FRIDGE='fridge',
    VIDALIA_ONION='vidalia_onion',
    BROCCOLI='broccoli',
    GREEN_ONION='green_onion',
    STEAK='steak',
    TRAY='tray',
    APPLE='apple',
    PLATE='plate',
    SCRUB_BRUSH='scrub_brush',
    CHOPPING_BOARD='shopping_board',
    KNIFE='knife',
    ONION='onion'
)

CONF_KEYS = SimpleNamespace(
    FILENAME='filename',
    DENSITY='density',
    SCALE='scale',
    SCALE_FACTOR='scale_factor',
    MODEL_PATH='model_path',
    CATEGORY='category',
    FIXED_BASE='fixed_base',
)
