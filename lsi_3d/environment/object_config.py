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
    ONION='onion',
    EMPTY='empty'
)

CONF_KEYS = SimpleNamespace(
    FILENAME='filename',
    DENSITY='density',
    SCALE='scale',
    SCALE_FACTOR='scale_factor',
    MODEL_PATH='model_path',
    CATEGORY='category',
    FIXED_BASE='fixed_base',
    ABILITIES='abilities',
    NAME='name'
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
    },
    OBJECT_KEYS.STEAK: {
        CONF_KEYS.FILENAME: os.path.join(ig_dataset_path, 'objects/steak/steak_000/steak_000.urdf'),
        CONF_KEYS.DENSITY: 10000,
        CONF_KEYS.SCALE: np.array([2.0, 2.0, 2.0]) / 1.15,
        CONF_KEYS.MODEL_PATH: os.path.dirname(os.path.join(ig_dataset_path, 'objects/steak/steak_000/steak_000.urdf')),
        CONF_KEYS.CATEGORY: OBJECT_KEYS.STEAK,
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
        CONF_KEYS.NAME: OBJECT_KEYS.PLATE
    },
    OBJECT_KEYS.STOVE: {
        CONF_KEYS.FILENAME: os.path.join(ig_dataset_path, 'objects/stove/101940/101940.urdf'),
        CONF_KEYS.SCALE: np.array([1.1, 1, 1]) / 1.15,
        CONF_KEYS.MODEL_PATH: os.path.dirname(os.path.join(ig_dataset_path, 'objects/stove/101940/101940.urdf')),
        CONF_KEYS.CATEGORY: OBJECT_KEYS.STOVE,
    },
    OBJECT_KEYS.PAN: {
        CONF_KEYS.FILENAME: os.path.join(ig_dataset_path, 'objects/frying_pan/36_0/36_0.urdf'),
        CONF_KEYS.SCALE: np.array([1.3, 1.3, 1]) / 1.15,
        CONF_KEYS.MODEL_PATH: os.path.dirname(os.path.join(ig_dataset_path, 'objects/frying_pan/36_0/36_0.urdf')),
        CONF_KEYS.CATEGORY: OBJECT_KEYS.PAN,
        CONF_KEYS.NAME: OBJECT_KEYS.PAN,
    },
    OBJECT_KEYS.GREEN_ONION: {
        CONF_KEYS.FILENAME: os.path.join(ig_dataset_path, 'objects/green_onion/green_onion_000/green_onion_000.urdf'),
        CONF_KEYS.SCALE: np.array([0.1, 0.1, 0.1]) / 1.15,
        CONF_KEYS.MODEL_PATH: os.path.dirname(os.path.join(
            ig_dataset_path,
            'objects/green_onion/green_onion_000/green_onion_000.urdf'
        )),
        CONF_KEYS.CATEGORY: OBJECT_KEYS.GREEN_ONION,
        CONF_KEYS.NAME: OBJECT_KEYS.GREEN_ONION,
        CONF_KEYS.ABILITIES: {
            'burnable': {},
            'freezable': {},
            'cookable': {},
            'sliceable': {
                "slice_force": .0
            }
        }
    },
    OBJECT_KEYS.BOWL: {
        CONF_KEYS.FILENAME: os.path.join(
            ig_dataset_path,
            'objects/bowl/a1393437aac09108d627bfab5d10d45d/a1393437aac09108d627bfab5d10d45d.urdf'
        ),
        CONF_KEYS.SCALE: np.array([0.8, 0.8, 0.8]) / 1.15,
        CONF_KEYS.MODEL_PATH: os.path.dirname(os.path.join(
            ig_dataset_path,
            'objects/bowl/a1393437aac09108d627bfab5d10d45d/a1393437aac09108d627bfab5d10d45d.urdf'
        )),
        CONF_KEYS.CATEGORY: OBJECT_KEYS.BOWL,
        CONF_KEYS.NAME: OBJECT_KEYS.BOWL,
        CONF_KEYS.ABILITIES: {
                'dustyable': {},
                'stainable': {}
            }
    },
    OBJECT_KEYS.SINK: {
        CONF_KEYS.FILENAME: os.path.join(ig_dataset_path, 'objects/sink/kitchen_sink/kitchen_sink.urdf'),
        CONF_KEYS.SCALE: np.array([1.2, 1.25, 1.25]) / 1.15,
        CONF_KEYS.MODEL_PATH: os.path.dirname(os.path.join(
            ig_dataset_path,
            'objects/sink/kitchen_sink/kitchen_sink.urdf'
        )),
        CONF_KEYS.CATEGORY: OBJECT_KEYS.SINK,
        CONF_KEYS.NAME: OBJECT_KEYS.SINK,
    },
    OBJECT_KEYS.CHOPPING_BOARD: {
        CONF_KEYS.FILENAME: os.path.join(ig_dataset_path, 'objects/chopping_board/10_0/10_0.urdf'),
        CONF_KEYS.SCALE: np.array([1.2, 1.2, 1.2]) / 1.15,
        CONF_KEYS.MODEL_PATH: os.path.dirname(os.path.join(ig_dataset_path, 'objects/chopping_board/10_0/10_0.urdf')),
        CONF_KEYS.CATEGORY: OBJECT_KEYS.CHOPPING_BOARD,
        CONF_KEYS.NAME: OBJECT_KEYS.CHOPPING_BOARD,
    },
    OBJECT_KEYS.KNIFE: {
        CONF_KEYS.FILENAME: os.path.join(ig_dataset_path, 'objects/carving_knife/14_1/14_1.urdf'),
        CONF_KEYS.SCALE: np.array([1, 1, 1]) / 1.15,
        CONF_KEYS.MODEL_PATH: os.path.dirname(os.path.join(ig_dataset_path, 'objects/carving_knife/14_1/14_1.urdf')),
        CONF_KEYS.CATEGORY: OBJECT_KEYS.KNIFE,
        CONF_KEYS.NAME: OBJECT_KEYS.KNIFE,
        CONF_KEYS.ABILITIES: {
                'slicer': {}
            }
    },
    OBJECT_KEYS.LARGE_BOWL: {
        CONF_KEYS.FILENAME: os.path.join(
            ig_dataset_path,
            'objects/bowl/5aad71b5e6cb3967674684c50f1db165/5aad71b5e6cb3967674684c50f1db165.urdf'
        ),
        CONF_KEYS.SCALE: np.array([0.5, 0.5, 0.5]) / 1.15,
        CONF_KEYS.MODEL_PATH: os.path.dirname(os.path.join(
            ig_dataset_path,
            'objects/bowl/5aad71b5e6cb3967674684c50f1db165/5aad71b5e6cb3967674684c50f1db165.urdf'
        )),
        CONF_KEYS.CATEGORY: OBJECT_KEYS.LARGE_BOWL,
        CONF_KEYS.NAME: OBJECT_KEYS.LARGE_BOWL,
    },
    OBJECT_KEYS.STEAK_BOWL: {
        CONF_KEYS.FILENAME: os.path.join(
            ig_dataset_path,
            'objects/bowl/7d7bdea515818eb844638317e9e4ff18/7d7bdea515818eb844638317e9e4ff18.urdf'
        ),
        CONF_KEYS.SCALE: np.array([0.5, 0.5, 0.5]) / 1.15,
        CONF_KEYS.MODEL_PATH: os.path.dirname(os.path.join(
            ig_dataset_path,
            'objects/bowl/7d7bdea515818eb844638317e9e4ff18/7d7bdea515818eb844638317e9e4ff18.urdf'
        )),
        CONF_KEYS.CATEGORY: OBJECT_KEYS.STEAK_BOWL,
        CONF_KEYS.NAME: OBJECT_KEYS.STEAK_BOWL,
    },
})
