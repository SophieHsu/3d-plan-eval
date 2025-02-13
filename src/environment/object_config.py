import os
import numpy as np
from types import SimpleNamespace
from igibson import ig_dataset_path


OBJECT_KEYS = SimpleNamespace(
    COUNTER='counter',
    TABLE='table',
    TABLE_V='table_v',
    TABLE_H='table_h',
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
    CHOPPING_BOARD='chopping_board',
    KNIFE='knife',
    ONION='onion',
    EMPTY='empty'
)

CONF_KEYS = SimpleNamespace(
    FILENAME='filename',
    DENSITY='density',
    SCALE='scale',
    MODEL_PATH='model_path',
    CATEGORY='category',
    FIXED_BASE='fixed_base',
    ABILITIES='abilities',
    NAME='name'
)

OBJECT_ABBRS = {
    OBJECT_KEYS.COUNTER: 'C',
    OBJECT_KEYS.TABLE_V: 'T',
    OBJECT_KEYS.TABLE_H: 'T',
    OBJECT_KEYS.STOVE: 'H',
    OBJECT_KEYS.BOWL: 'B',
    OBJECT_KEYS.PAN: 'P',
    OBJECT_KEYS.SCRUB_BRUSH: 'W',
    OBJECT_KEYS.SINK: 'W',
    OBJECT_KEYS.BROCCOLI: 'F',
    OBJECT_KEYS.STEAK: 'F',
    OBJECT_KEYS.GREEN_ONION: 'G',
    OBJECT_KEYS.TRAY: 'F',
    OBJECT_KEYS.APPLE: 'F',
    OBJECT_KEYS.FRIDGE: 'F',
    OBJECT_KEYS.PLATE: 'D',
    OBJECT_KEYS.KNIFE: 'K',
    OBJECT_KEYS.CHOPPING_BOARD: 'K',
    OBJECT_KEYS.EMPTY: 'X',
}

OBJECT_TRANSLATIONS = {
    OBJECT_KEYS.COUNTER: (0, 0, 0),
    OBJECT_KEYS.TABLE_V: (0.5, 0, 0),
    OBJECT_KEYS.TABLE_H: (0, 0.5, 0),
    OBJECT_KEYS.STOVE: (0, 0, 0),
    OBJECT_KEYS.BOWL: (0, 0, 1.1),
    OBJECT_KEYS.PAN: (0.23, 0.24, 1.24),  # shift height
    OBJECT_KEYS.SINK: (0, 0, 0.1),
    OBJECT_KEYS.FRIDGE: (0, 0, 0.2),
    OBJECT_KEYS.VIDALIA_ONION: (0.15, -0.1, 1.3),
    OBJECT_KEYS.STEAK: (0, 0, 0.9),  # (0.23, -0.1, 1.25),
    OBJECT_KEYS.TRAY: (0, 0, 0.9),
    OBJECT_KEYS.APPLE: (0, 0.2, 1.0),
    OBJECT_KEYS.BROCCOLI: (0, 0.2, 0.6),
    OBJECT_KEYS.GREEN_ONION: (0, 0, 1.25),
    OBJECT_KEYS.PLATE: (0, 0, 1.15),
    OBJECT_KEYS.SCRUB_BRUSH: (0, 0, 1.3),
    OBJECT_KEYS.CHOPPING_BOARD: (0, 0, 1.2),
    OBJECT_KEYS.KNIFE: (0, 0.3, 1.4),
    OBJECT_KEYS.ONION: (0.15, -0.1, 0),
    OBJECT_KEYS.LARGE_BOWL: (0, 0, 0),
    OBJECT_KEYS.STEAK_BOWL: (0, 0, 0)
}

OBJECT_AWAY_POSES_OFFSETS = {
    OBJECT_KEYS.GREEN_ONION: [200, 100, 1],
    OBJECT_KEYS.LARGE_BOWL: [300, 200, 1],
    OBJECT_KEYS.STEAK_BOWL: [275, 275, 1],
}

OBJECT_ABBR_MAP = {value: key for key, value in OBJECT_ABBRS.items()}

OBJECT_CONFIG = {
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
        CONF_KEYS.SCALE: np.array([0.1, 0.1, 0.1]) / 1.15,
        CONF_KEYS.MODEL_PATH: os.path.dirname(os.path.join(ig_dataset_path, 'objects/steak/steak_000/steak_000.urdf')),
        CONF_KEYS.CATEGORY: OBJECT_KEYS.STEAK,
    },
    OBJECT_KEYS.PLATE: {
        CONF_KEYS.FILENAME: os.path.join(
            ig_dataset_path,
            'objects/bowl/a1393437aac09108d627bfab5d10d45d/a1393437aac09108d627bfab5d10d45d.urdf'
        ),
        CONF_KEYS.SCALE: np.array([0.8, 0.8, 0.8]) / 1.15,
        CONF_KEYS.MODEL_PATH: os.path.dirname(os.path.join(
            ig_dataset_path,
            'objects/bowl/a1393437aac09108d627bfab5d10d45d/a1393437aac09108d627bfab5d10d45d.urdf'
        )),
        CONF_KEYS.CATEGORY: OBJECT_KEYS.PLATE,
        CONF_KEYS.NAME: OBJECT_KEYS.PLATE,
        CONF_KEYS.ABILITIES: {
                "dustyable": {},
                "stainable": {}
            },
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
    OBJECT_KEYS.TABLE_H: {
        CONF_KEYS.FILENAME: os.path.join(ig_dataset_path, 'objects/breakfast_table/26670/26670.urdf'),
        CONF_KEYS.SCALE: np.array([1.2, 1.2, 1.3]) / 1.15,
        CONF_KEYS.MODEL_PATH: os.path.dirname(os.path.join(
            ig_dataset_path, 'objects/breakfast_table/26670/26670.urdf'
        )),
        CONF_KEYS.CATEGORY: OBJECT_KEYS.TABLE_H,
        CONF_KEYS.NAME: OBJECT_KEYS.TABLE_H,
    },
    OBJECT_KEYS.TABLE_V: {
        CONF_KEYS.FILENAME: os.path.join(ig_dataset_path, 'objects/breakfast_table/26670/26670.urdf'),
        CONF_KEYS.SCALE: np.array([1.2, 1.2, 1.3]) / 1.15,
        CONF_KEYS.MODEL_PATH: os.path.dirname(os.path.join(
            ig_dataset_path, 'objects/breakfast_table/26670/26670.urdf'
        )),
        CONF_KEYS.CATEGORY: OBJECT_KEYS.TABLE_V,
        CONF_KEYS.NAME: OBJECT_KEYS.TABLE_V,
    },
    OBJECT_KEYS.SCRUB_BRUSH: {
        CONF_KEYS.FILENAME: os.path.join(ig_dataset_path, 'objects/scrub_brush/scrub_brush_000/scrub_brush_000.urdf'),
        CONF_KEYS.SCALE: np.array([0.01, 0.01, 0.01]) / 1.15,
        CONF_KEYS.MODEL_PATH: os.path.dirname(os.path.join(
            ig_dataset_path, 'objects/scrub_brush/scrub_brush_000/scrub_brush_000.urdf'
        )),
        CONF_KEYS.CATEGORY: OBJECT_KEYS.SCRUB_BRUSH,
        CONF_KEYS.NAME: OBJECT_KEYS.SCRUB_BRUSH,
        CONF_KEYS.ABILITIES: {"soakable": {}, "cleaningTool": {}}
    },
    OBJECT_KEYS.APPLE: {
        CONF_KEYS.FILENAME: os.path.join(ig_dataset_path, 'objects/apple/00_0/00_0.urdf'),
        CONF_KEYS.SCALE: np.array([0.1, 0.1, 0.1]) / 1.15,
        CONF_KEYS.MODEL_PATH: os.path.dirname(os.path.join(
            ig_dataset_path, 'objects/apple/00_0/00_0.urdf'
        )),
        CONF_KEYS.CATEGORY: OBJECT_KEYS.APPLE,
        CONF_KEYS.NAME: OBJECT_KEYS.APPLE,
    },
}
