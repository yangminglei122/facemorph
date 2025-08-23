# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files, collect_all

block_cipher = None

# 收集mediapipe中必要的数据文件
mediapipe_datas = []
mediapipe_datas += collect_data_files('mediapipe', include_py_files=False, subdir='modules/face_landmark')
mediapipe_datas += collect_data_files('mediapipe', include_py_files=False, subdir='modules/face_detection')
mediapipe_datas += collect_data_files('mediapipe', include_py_files=False, subdir='modules/pose_detection')
mediapipe_datas += collect_data_files('mediapipe', include_py_files=False, subdir='modules/pose_landmark')
mediapipe_datas += collect_data_files('mediapipe', include_py_files=False, subdir='models', includes=['*.tflite', '*.pb', '*.txt'])

# 统一版本的文件包含列表
datas = [
    ('utils.py', '.'),
    ('detect.py', '.'),
    ('align.py', '.'),
    ('morph.py', '.'),
    ('exporter.py', '.'),
    ('main.py', '.'),
    ('app_entry.py', '.'),
    ('src', 'src'),
    ('scripts/logging_hook.py', 'scripts'),
    ('ref', 'ref'),
] + mediapipe_datas

hiddenimports = [
    'cv2',
    'numpy',
    'PIL',
    'mediapipe',
    'imageio',
    'imageio_ffmpeg',
    'tqdm',
    'flask',
    'src.pipeline',
    'src.webapp',
    'utils',
    'detect',
    'align',
    'morph',
    'exporter',
    'scripts.logging_hook',
]

a = Analysis(
    ['app_entry.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['scripts/logging_hook.py'],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='FaceGrowthApp',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='NONE',
)
