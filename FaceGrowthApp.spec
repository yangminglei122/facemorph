# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_all

# 包含项目根目录下的必要文件
datas = [
    ('src', 'src'),
    ('utils.py', '.'),
    ('detect.py', '.'),
    ('align.py', '.'),
    ('morph.py', '.'),
    ('exporter.py', '.'),
]
binaries = []
hiddenimports = [
    'flask',
    'imageio',
    'imageio_ffmpeg',
    'cv2',
    'numpy',
    'PIL',
    'mediapipe',
    'scipy',
    'tqdm'
]
datas += collect_data_files('imageio_ffmpeg')
datas += collect_data_files('imageio')
tmp_ret = collect_all('mediapipe')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['src/__main__.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='FaceGrowthApp',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='NONE',
)
