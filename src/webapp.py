import os
import time
import json
import zipfile
import io
import threading
import secrets
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from flask import Flask, request, render_template_string, send_file, jsonify, make_response
from werkzeug.utils import secure_filename

from src.pipeline import run_pipeline

# 配置日志
def setup_logging():
    """配置日志系统"""
    # 创建logs目录
    logs_dir = os.path.abspath(os.path.join(os.getcwd(), "logs"))
    os.makedirs(logs_dir, exist_ok=True)
    
    # 生成日志文件名（按日期）
    log_filename = f"webapp_{datetime.now().strftime('%Y%m%d')}.log"
    log_path = os.path.join(logs_dir, log_filename)
    
    # 配置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, encoding='utf-8'),
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )
    
    return logging.getLogger(__name__)

# 初始化日志
logger = setup_logging()

app = Flask(__name__)

JOBS_DIR = os.path.abspath(os.path.join(os.getcwd(), "jobs"))
os.makedirs(JOBS_DIR, exist_ok=True)


def _job_path(job_id: str) -> str:
    return os.path.join(JOBS_DIR, f"{job_id}.json")


def _job_write(job_id: str, data: Dict[str, Any]) -> None:
    path = _job_path(job_id)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp, path)


def _job_read(job_id: str) -> Optional[Dict[str, Any]]:
    path = _job_path(job_id)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Upload helpers
UPLOADS_DIR = os.path.abspath(os.path.join(os.getcwd(), "uploads"))
os.makedirs(UPLOADS_DIR, exist_ok=True)


def _prepare_upload_dir() -> str:
    d = os.path.join(UPLOADS_DIR, str(int(time.time() * 1000)))
    os.makedirs(d, exist_ok=True)
    return d


def _handle_uploads(form, files) -> str:
    """Save uploaded images or zip into a local folder; return folder path or empty string if none."""
    upload_dir = ""
    up_zip = files.get("images_zip")
    up_imgs: List = files.getlist("images[]") if "images[]" in files else []
    
    if up_zip and up_zip.filename:
        upload_dir = _prepare_upload_dir()
        zname = secure_filename(up_zip.filename)
        zpath = os.path.join(upload_dir, zname)
        up_zip.save(zpath)
        logger.info(f"处理ZIP上传: {zname} -> {zpath}")
        try:
            with zipfile.ZipFile(zpath, 'r') as zf:
                zf.extractall(upload_dir)
            logger.info(f"ZIP解压完成: {zpath}")
        except Exception as e:
            logger.error(f"ZIP解压失败: {zpath} - {str(e)}")
            raise
        finally:
            os.remove(zpath)
            logger.info(f"清理临时ZIP文件: {zpath}")
    elif up_imgs:
        valid = [f for f in up_imgs if f and f.filename]
        if valid:
            upload_dir = _prepare_upload_dir()
            logger.info(f"处理图片上传: {len(valid)} 个文件 -> {upload_dir}")
            for f in valid:
                fname = secure_filename(f.filename)
                fpath = os.path.join(upload_dir, fname)
                os.makedirs(os.path.dirname(fpath), exist_ok=True)
                f.save(fpath)
                logger.debug(f"保存图片: {fname} -> {fpath}")
    
    if upload_dir:
        logger.info(f"上传处理完成，目录: {upload_dir}")
    else:
        logger.debug("无上传文件")
    
    return upload_dir


def _start_job_file(cfg: Dict[str, Any], job_id: str):
    def cb(p: float, msg: str):
        st = _job_read(job_id) or {}
        st.update({"p": float(max(0.0, min(1.0, p))), "msg": str(msg)})
        _job_write(job_id, st)
        # 记录进度日志
        if p > 0 and p < 1:
            logger.info(f"任务 {job_id}: 进度 {p*100:.1f}% - {msg}")
    
    try:
        logger.info(f"任务 {job_id}: 开始执行pipeline")
        res = run_pipeline(cfg, progress_cb=cb)
        st = _job_read(job_id) or {}
        st.update({"busy": False, "result": res, "error": None, "p": 1.0, "msg": "完成"})
        _job_write(job_id, st)
        logger.info(f"任务 {job_id}: 执行成功完成")
    except Exception as e:
        error_msg = str(e)
        st = _job_read(job_id) or {}
        st.update({"busy": False, "error": error_msg})
        _job_write(job_id, st)
        logger.error(f"任务 {job_id}: 执行失败 - {error_msg}", exc_info=True)


TEMPLATE = """
<!doctype html>
<title>Face Growth Morph</title>
<style>
.bar{width:100%;background:#eee;border-radius:6px;height:12px;margin:6px 0}
.fill{background:#4caf50;height:12px;border-radius:6px;width:0%}
.small{color:#555;font-size:12px}
fieldset{border:1px solid #ccc;margin:10px 0;padding:8px}
legend{font-size:14px;color:#444}
.err{color:#c00}
</style>
<h2>Face Growth Morph</h2>
<form id="cfg" method="post" action="/run" enctype="multipart/form-data">
  <fieldset>
    <legend>输入路径（服务器可访问的本地路径）</legend>
    <label>输入目录: <input type="text" name="input_dir" value="{{input_dir}}" size="60" placeholder="例如 /home/user/photos"></label>
  </fieldset>
  <fieldset>
    <legend>或上传图片（多文件）/压缩包（zip）</legend>
    <label>多文件: <input type="file" name="images[]" multiple accept="image/*"></label><br/>
    <label>ZIP: <input type="file" name="images_zip" accept=".zip"></label><br/>
    <small class="small">若同时提供上传与路径，优先使用上传内容</small>
  </fieldset>
  <fieldset>
    <legend>参数</legend>
    <label>输出目录: <input type="text" name="output_dir" value="{{output_dir}}" size="60"></label><br/>
    <label>排序: 
      <select name="sort">
        {% for s in sorts %}
        <option value="{{s}}" {% if sort==s %}selected{% endif %}>{{s}}</option>
        {% endfor %}
      </select>
    </label><br/>
    <label>宽: <input type="number" name="width" value="{{width}}"></label>
    <label>高: <input type="number" name="height" value="{{height}}"></label><br/>
    <label>主体缩放: <input type="number" step="0.01" name="subject_scale" value="{{subject_scale}}"></label><br/>
    <label>渐变方式: 
      <select name="morph">
        {% for m in morphs %}
        <option value="{{m}}" {% if morph==m %}selected{% endif %}>{{m}}</option>
        {% endfor %}
      </select>
    </label><br/>
    <label>过渡时长(s): <input type="number" step="0.1" name="transition_seconds" value="{{transition_seconds}}"></label>
    <label>停留时长(s): <input type="number" step="0.1" name="hold_seconds" value="{{hold_seconds}}"></label><br/>
    <label>视频FPS: <input type="number" name="video_fps" value="{{video_fps}}"></label>
    <label>GIF FPS: <input type="number" name="gif_fps" value="{{gif_fps}}"></label><br/>
    <label>光流强度: <input type="number" step="0.1" name="flow_strength" value="{{flow_strength}}"></label>
    <label>面部保护: <input type="number" step="0.1" name="face_protect" value="{{face_protect}}"></label>
    <label>锐化: <input type="number" step="0.05" name="sharpen" value="{{sharpen}}"></label><br/>
    <label>权重曲线: 
      <select name="easing">
        {% for e in easings %}
        <option value="{{e}}" {% if easing==e %}selected{% endif %}>{{e}}</option>
        {% endfor %}
      </select>
    </label>
    <label>ease_a: <input type="number" step="0.05" name="ease_a" value="{{ease_a}}"></label>
    <label>ease_b: <input type="number" step="0.05" name="ease_b" value="{{ease_b}}"></label>
    <label>ease_p1: <input type="number" step="0.1" name="ease_p1" value="{{ease_p1}}"></label>
    <label>ease_p3: <input type="number" step="0.1" name="ease_p3" value="{{ease_p3}}"></label><br/>
    <label>使用GPU: <input type="checkbox" name="use_gpu" {% if use_gpu %}checked{% endif %}></label>
    <label>保存对齐帧: <input type="checkbox" name="save_aligned" {% if save_aligned %}checked{% endif %}></label><br/>
    <fieldset>
      <legend>参考图选择</legend>
      <label><input type="radio" name="ref_selection" value="auto" {% if ref_selection == "auto" %}checked{% endif %}> 自动选择参考帧（基于质量评分）</label><br/>
      <label><input type="radio" name="ref_selection" value="manual" {% if ref_selection == "manual" %}checked{% endif %}> 手动指定帧号</label><br/>
      <label><input type="radio" name="ref_selection" value="upload" {% if ref_selection == "upload" %}checked{% endif %}> 上传自定义参考图像</label><br/>
      <label><input type="radio" name="ref_selection" value="default" {% if ref_selection == "default" %}checked{% endif %}> 使用系统默认参考图像</label><br/>
      
      <!-- 手动指定帧号输入框 -->
      <div id="manual_frame_div" style="margin-left: 20px; {% if ref_selection != 'manual' %}display: none;{% endif %}">
        <label>参考帧索引: <input type="number" name="ref_index" value="{{ref_index}}" placeholder="0, 1, 2..." min="0"></label><br/>
        <small class="small">输入要作为参考的帧的索引号（从0开始）</small>
      </div>
      
      <!-- 上传参考图输入框 -->
      <div id="upload_ref_div" style="margin-left: 20px; {% if ref_selection != 'upload' %}display: none;{% endif %}">
        <input type="file" name="ref_image_file" accept="image/*"><br/>
        <small class="small">上传自定义参考图像，不参与最终效果生成</small>
      </div>
      
      <!-- 默认参考图信息 -->
      <div id="default_ref_div" style="margin-left: 20px; {% if ref_selection != 'default' %}display: none;{% endif %}">
        <small class="small">使用系统默认参考图像: ./ref/ref_image.jpg</small>
      </div>
    </fieldset>
  </fieldset>
  <button type="submit">开始生成</button>
</form>
<div id="prog" style="display:none;">
  <div class="bar"><div class="fill" id="fill"></div></div>
  <div class="small" id="msg">准备中...</div>
  <div class="small" id="eta"></div>
</div>
<div id="done" style="display:none;">
  <h3>结果</h3>
  <div id="links"></div>
  <div id="preview"></div>
</div>
<div id="err" class="small" style="color:#c00"></div>
<script>
function fmtETA(sec){
  if(sec == null) return '';
  sec = Math.max(0, Math.round(sec));
  const m = Math.floor(sec/60); const s = sec%60;
  return '预计剩余: ' + (m>0? (m+'分') : '') + s + '秒';
}

// 处理参考图选择方式切换
function handleRefSelectionChange() {
  const refSelection = document.querySelector('input[name="ref_selection"]:checked').value;
  
  // 隐藏所有相关div
  document.getElementById('manual_frame_div').style.display = 'none';
  document.getElementById('upload_ref_div').style.display = 'none';
  document.getElementById('default_ref_div').style.display = 'none';
  
  // 根据选择显示对应的div
  if (refSelection === 'manual') {
    document.getElementById('manual_frame_div').style.display = 'block';
  } else if (refSelection === 'upload') {
    document.getElementById('upload_ref_div').style.display = 'block';
  } else if (refSelection === 'default') {
    document.getElementById('default_ref_div').style.display = 'block';
  }
}

// 页面加载完成后绑定事件
document.addEventListener('DOMContentLoaded', function() {
  // 绑定参考图选择方式切换事件
  const refSelectionRadios = document.querySelectorAll('input[name="ref_selection"]');
  refSelectionRadios.forEach(radio => {
    radio.addEventListener('change', handleRefSelectionChange);
  });
  
  // 初始化显示状态
  handleRefSelectionChange();
});

let job = null;
const form = document.getElementById('cfg');
const prog = document.getElementById('prog');
const fill = document.getElementById('fill');
const msg = document.getElementById('msg');
const eta = document.getElementById('eta');
const done = document.getElementById('done');
const links = document.getElementById('links');
const preview = document.getElementById('preview');
const err = document.getElementById('err');

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  prog.style.display = 'block';
  done.style.display = 'none';
  links.innerHTML = '';
  preview.innerHTML = '';
  eta.textContent = '';
  err.textContent = '';
  const data = new FormData(form);
  const resp = await fetch('/run', { method: 'POST', body: data });
  if(resp.status !== 200){
    try{ const j = await resp.json(); err.textContent = j.err || '提交失败'; }catch(ex){ err.textContent = '提交失败'; }
    prog.style.display = 'none';
    return;
  }
  const info = await resp.json();
  job = info.job;
  const poll = setInterval(async ()=>{
    const r = await fetch('/progress?job='+encodeURIComponent(job), {cache:'no-store'});
    const j = await r.json();
    fill.style.width = Math.round((j.p||0)*100)+'%';
    msg.textContent = j.msg||'';
    eta.textContent = fmtETA(j.eta_sec);
    if(j.error){ err.textContent = j.error; }
    if(j.busy === false){
      clearInterval(poll);
      eta.textContent = '';
      if(j.result){
        const hasMP4 = !!j.result.mp4_path;
        const mp4Link = hasMP4 ? ` | <a href="/file?path=${j.result.mp4_path}" target="_blank">下载 MP4</a>` : ' | <span class="small">(MP4 未生成)</span>';
        links.innerHTML = `<a href="/file?path=${j.result.gif_path}" target="_blank">下载 GIF</a>${mp4Link} | <a href="/download?job=${encodeURIComponent(job)}">下载全部ZIP</a>`;
        preview.innerHTML = `<img src="/file?path=${j.result.gif_path}" style="max-width: 600px; border:1px solid #ccc;"/>`;
      }
      done.style.display = 'block';
    }
  }, 800);
});
</script>
"""

DEFAULTS = {
    "input_dir": "",
    "output_dir": "./output",
    "sort": "name",
    "width": 1080,
    "height": 1350,
    "subject_scale": 0.55,
    "transition_seconds": 0.4,
    "hold_seconds": 0.7,
    "gif_fps": 15,
    "video_fps": 30,
    "morph": "flow",
    "flow_strength": 0.9,
    "face_protect": 0.7,
    "sharpen": 0.2,
    "easing": "compressed_mid",
    "ease_a": 0.4,
    "ease_b": 0.6,
    "ease_p1": 2.5,
    "ease_p3": 0.6,
    "use_gpu": False,
    "save_aligned": False,
    "ref_selection": "auto",
    "ref_index": "",
}

@app.route("/", methods=["GET"])
def index():
    return render_template_string(
        TEMPLATE,
        sorts=["name", "exif", "filename_date", "name_numeric"],
        morphs=["crossfade", "flow"],
        easings=["linear", "compressed_mid"],
        **DEFAULTS,
    )

@app.route("/run", methods=["POST"])
def run():
    # prepare job id and initial state
    job_id = secrets.token_hex(8)
    logger.info(f"开始新任务: {job_id}")
    
    _job_write(job_id, {"p": 0.0, "msg": "开始", "busy": True, "result": None, "error": None, "start": time.time()})
    
    # uploads
    upload_dir = _handle_uploads(request.form, request.files)
    if upload_dir:
        logger.info(f"任务 {job_id}: 处理上传文件，目录: {upload_dir}")
    
    # 处理参考图选择
    ref_selection = request.form.get("ref_selection", "auto")
    ref_image_path = None
    ref_index = None
    
    logger.info(f"任务 {job_id}: 用户选择的参考图方式: {ref_selection}")
    
    if ref_selection == "auto":
        # 自动选择参考帧
        logger.info(f"任务 {job_id}: 使用自动选择参考帧模式")
        ref_index = None  # 让pipeline自动选择
        
    elif ref_selection == "manual":
        # 手动指定帧号
        manual_ref_index = request.form.get("ref_index")
        if manual_ref_index and manual_ref_index.strip():
            try:
                ref_index = int(manual_ref_index)
                logger.info(f"任务 {job_id}: 手动指定参考帧索引: {ref_index}")
            except ValueError:
                logger.warning(f"任务 {job_id}: 手动指定的帧号无效，回退到自动选择")
                ref_index = None
        else:
            logger.warning(f"任务 {job_id}: 手动指定模式但未提供帧号，回退到自动选择")
            ref_index = None
            
    elif ref_selection == "upload":
        # 上传自定义参考图像
        ref_file = request.files.get("ref_image_file")
        if ref_file and ref_file.filename:
            # 创建专门的参考图目录
            ref_dir = os.path.abspath(os.path.join(os.getcwd(), "uploads", "ref"))
            os.makedirs(ref_dir, exist_ok=True)
            
            # 保存参考图文件
            ref_filename = secure_filename(ref_file.filename)
            ref_path = os.path.join(ref_dir, ref_filename)
            ref_file.save(ref_path)
            ref_image_path = ref_path
            ref_index = -1  # 使用外部参考图
            logger.info(f"任务 {job_id}: 上传参考图 {ref_filename} -> {ref_path}")
        else:
            logger.warning(f"任务 {job_id}: 选择上传参考图但未提供文件，回退到自动选择")
            ref_index = None
            
    elif ref_selection == "default":
        # 使用系统默认参考图像
        ref_image_path = "./ref/ref_image.jpg"
        ref_index = -1  # 使用外部参考图
        logger.info(f"任务 {job_id}: 使用系统默认参考图 {ref_image_path}")
    
    # 如果ref_index为None，说明需要自动选择
    if ref_index is None:
        logger.info(f"任务 {job_id}: 将使用pipeline自动选择参考帧")
    
    cfg: Dict[str, Any] = {}
    for k in DEFAULTS.keys():
        v = request.form.get(k)
        if v is None:
            if k in ("use_gpu", "save_aligned"):
                cfg[k] = False
            elif k == "ref_index":
                # 处理ref_index参数
                if v.strip() == "":
                    cfg[k] = None
                else:
                    try:
                        cfg[k] = int(v)
                    except ValueError:
                        cfg[k] = None
            else:
                cfg[k] = DEFAULTS[k]
        else:
            if k in ("use_gpu", "save_aligned"):
                cfg[k] = True
            elif k == "ref_index":
                # 处理ref_index参数
                if v.strip() == "":
                    cfg[k] = None
                else:
                    try:
                        cfg[k] = int(v)
                    except ValueError:
                        cfg[k] = None
            else:
                cfg[k] = v
    
    if upload_dir:
        cfg["input_dir"] = upload_dir
    elif not cfg.get("input_dir"):
        error_msg = "未提供输入目录或上传文件"
        logger.error(f"任务 {job_id}: {error_msg}")
        _job_write(job_id, {"p": 0.0, "msg": "", "busy": False, "result": None, "error": error_msg})
        return jsonify({"err": error_msg}), 400
    
    # 记录配置信息
    logger.info(f"任务 {job_id}: 配置参数 - 输入目录: {cfg.get('input_dir')}, 输出目录: {cfg.get('output_dir')}, 参考图选择方式: {ref_selection}, 参考图索引: {ref_index}, 参考图路径: {ref_image_path}")
    
    # 将参考图相关参数添加到配置中
    if ref_index is not None:
        cfg["ref_index"] = ref_index
    if ref_image_path is not None:
        cfg["ref_image"] = ref_image_path
    
    t = threading.Thread(target=_start_job_file, args=(cfg, job_id), daemon=True)
    t.start()
    logger.info(f"任务 {job_id}: 已启动后台处理线程")
    return jsonify({"job": job_id}), 200

@app.get("/progress")
def progress():
    job_id = request.args.get("job")
    if not job_id:
        logger.warning("进度查询: 缺少job参数")
        return jsonify({"err": "missing job"}), 400
    
    st = _job_read(job_id) or {"p": 0.0, "msg": "就绪", "busy": False}
    p = st.get("p", 0.0)
    start = st.get("start")
    eta = None
    if st.get("busy") and start and p > 0.01 and p < 1.0:
        elapsed = time.time() - start
        eta = max(0.0, elapsed * (1.0 - p) / max(1e-6, p))
    st["eta_sec"] = eta
    
    # 记录进度查询日志（避免过于频繁）
    if p > 0.9:  # 只在接近完成时记录
        logger.debug(f"任务 {job_id}: 进度查询 - {p*100:.1f}%")
    
    resp = make_response(jsonify(st))
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return resp

@app.get("/file")
def file():
    path = request.args.get("path")
    if not path or not os.path.exists(path):
        logger.warning(f"文件下载失败: 路径不存在 - {path}")
        return "Not found", 404
    
    logger.info(f"文件下载: {path}")
    return send_file(path)

@app.get("/download")
def download_zip():
    job_id = request.args.get("job")
    if not job_id:
        logger.warning("ZIP下载: 缺少job参数")
        return "missing job", 400
    
    st = _job_read(job_id)
    if not st or not st.get("result"):
        logger.warning(f"ZIP下载失败: 任务 {job_id} 结果未就绪")
        return "Not ready", 404
    
    res = st["result"]
    gif_path = res.get("gif_path")
    mp4_path = res.get("mp4_path")
    
    logger.info(f"ZIP下载: 任务 {job_id} - GIF: {gif_path}, MP4: {mp4_path}")
    
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        if gif_path and os.path.exists(gif_path):
            zf.write(gif_path, arcname=os.path.basename(gif_path))
        if mp4_path and os.path.exists(mp4_path):
            zf.write(mp4_path, arcname=os.path.basename(mp4_path))
    mem.seek(0)
    return send_file(mem, as_attachment=True, download_name='morph_output.zip', mimetype='application/zip')


def start(host: str = "127.0.0.1", port: int = 5000):
    logger.info(f"启动Face Growth Morph Web服务 - 地址: {host}:{port}")
    logger.info(f"日志文件位置: logs/webapp_{datetime.now().strftime('%Y%m%d')}.log")
    app.run(host=host, port=port, debug=False)
