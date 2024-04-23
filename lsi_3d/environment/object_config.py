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

OBJECT_CONFIG = SimpleNamespace(**{
    OBJECT_KEYS.COUNTER: {
        CONF_KEYS.FILENAME: os.path.join(ig_dataset_path, 'objects/bottom_cabinet/46452/46452.urdf'),
        CONF_KEYS.DENSITY: 10000,
        CONF_KEYS.SCALE: np.array([1.04, 0.97, 0.95]) / 1.15,
        CONF_KEYS.MODEL_PATH: os.path.dirname(os.path.join(ig_dataset_path, 'objects/bottom_cabinet/46452/46452.urdf')),
        CONF_KEYS.CATEGORY: OBJECT_KEYS.COUNTER,
        CONF_KEYS.FIXED_BASE: True,
    },
    OBJECT_KEYS.VIDALIA_ONION: {
        CONF_KEYS.FILENAME: os.path.join(ig_dataset_path, 'objects/vidalia_onion/18_1/18_1.urdf'),
        CONF_KEYS.DENSITY: 10000,
        CONF_KEYS.SCALE: np.array([2.0, 2.0, 2.0]) / 1.15,
        CONF_KEYS.MODEL_PATH: os.path.dirname(os.path.join(ig_dataset_path, 'objects/vidalia_onion/18_1/18_1.urdf')),
        CONF_KEYS.CATEGORY: OBJECT_KEYS.VIDALIA_ONION,
        CONF_KEYS.FIXED_BASE: False,
    },
    OBJECT_KEYS.STEAK: {
        CONF_KEYS.FILENAME: os.path.join(ig_dataset_path, 'objects/steak/steak_000/steak_000.urdf'),
        CONF_KEYS.DENSITY: 10000,
        CONF_KEYS.SCALE: np.array([2.0, 2.0, 2.0]) / 1.15,
        CONF_KEYS.MODEL_PATH: os.path.dirname(os.path.join(ig_dataset_path, 'objects/steak/steak_000/steak_000.urdf')),
        CONF_KEYS.CATEGORY: OBJECT_KEYS.STEAK,
        CONF_KEYS.FIXED_BASE: False,
    },
    OBJECT_KEYS.PLATE: {
        CONF_KEYS.FILENAME: os.path.join(
            ig_dataset_path,
            'objects/bowl/a1393437aac09108d627bfab5d10d45d/a1393437aac09108d627bfab5d10d45d.urdf'
        ),
        CONF_KEYS.SCALE: np.array([2.0, 2.0, 2.0]) / 1.15,
        CONF_KEYS.MODEL_PATH: os.path.dirname(os.path.join(
            ig_dataset_path,
            'objects/bowl/a1393437aac09108d627bfab5d10d45d/a1393437aac09108d627bfab5d10d45d.urdf'
        )),
        CONF_KEYS.CATEGORY: OBJECT_KEYS.PLATE,
        CONF_KEYS.FIXED_BASE: False,
    },
})
