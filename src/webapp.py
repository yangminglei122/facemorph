import os
import time
import json
import zipfile
import io
import threading
import secrets
from typing import Any, Dict, List, Optional
from flask import Flask, request, render_template_string, send_file, jsonify, make_response
from werkzeug.utils import secure_filename

from src.pipeline import run_pipeline

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


def _handle_uploads(form, files) -> tuple[str, str]:
    """Save uploaded images or zip into a local folder; return (upload_dir, ref_image_path) tuple."""
    upload_dir = ""
    ref_image_path = ""
    
    # Handle main images upload
    up_zip = files.get("images_zip")
    up_imgs: List = files.getlist("images[]") if "images[]" in files else []
    if up_zip and up_zip.filename:
        upload_dir = _prepare_upload_dir()
        zname = secure_filename(up_zip.filename)
        zpath = os.path.join(upload_dir, zname)
        up_zip.save(zpath)
        try:
            with zipfile.ZipFile(zpath, 'r') as zf:
                zf.extractall(upload_dir)
        finally:
            os.remove(zpath)
    elif up_imgs:
        valid = [f for f in up_imgs if f and f.filename]
        if valid:
            upload_dir = _prepare_upload_dir()
            for f in valid:
                fname = secure_filename(f.filename)
                fpath = os.path.join(upload_dir, fname)
                os.makedirs(os.path.dirname(fpath), exist_ok=True)
                f.save(fpath)
    
    # Handle reference image upload
    ref_img = files.get("ref_image")
    if ref_img and ref_img.filename:
        ref_dir = _prepare_upload_dir()
        ref_fname = secure_filename(ref_img.filename)
        ref_image_path = os.path.join(ref_dir, ref_fname)
        ref_img.save(ref_image_path)
    
    return upload_dir, ref_image_path


def _start_job_file(cfg: Dict[str, Any], job_id: str):
    def cb(p: float, msg: str):
        st = _job_read(job_id) or {}
        st.update({"p": float(max(0.0, min(1.0, p))), "msg": str(msg)})
        _job_write(job_id, st)
    try:
        res = run_pipeline(cfg, progress_cb=cb)
        st = _job_read(job_id) or {}
        st.update({"busy": False, "result": res, "error": None, "p": 1.0, "msg": "完成"})
        _job_write(job_id, st)
    except Exception as e:
        st = _job_read(job_id) or {}
        st.update({"busy": False, "error": str(e)})
        _job_write(job_id, st)


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
    <legend>参考图选项</legend>
    <label><input type="radio" name="ref_option" value="auto" checked> 自动选择</label><br/>
    <label><input type="radio" name="ref_option" value="index"> 指定索引号: <input type="number" name="ref_index" value="{{ref_index}}" placeholder="输入索引号"></label><br/>
    <label><input type="radio" name="ref_option" value="internal"> 使用系统内部预定参考图 (./ref/ref_image.jpg)</label><br/>
    <label><input type="radio" name="ref_option" value="upload"> 上传指定参考图: <input type="file" name="ref_image" accept="image/*"></label><br/>
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
    <label>保存对齐帧: <input type="checkbox" name="save_aligned" {% if save_aligned %}checked{% endif %}></label>
    <label>生成GIF: <input type="checkbox" name="generate_gif" {% if generate_gif %}checked{% endif %}></label>
    <label>流式保存(减少内存): <input type="checkbox" name="streaming_save" {% if streaming_save %}checked{% endif %}></label><br/>
    <label>批处理大小: <input type="number" name="batch_size" value="{{batch_size}}"></label><br/>
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
        const hasGIF = !!j.result.gif_path;
        const mp4Link = hasMP4 ? ` | <a href="/file?path=${j.result.mp4_path}" target="_blank">下载 MP4</a>` : ' | <span class="small">(MP4 未生成)</span>';
        const gifLink = hasGIF ? `<a href="/file?path=${j.result.gif_path}" target="_blank">下载 GIF</a>` : '<span class="small">(GIF 未生成)</span>';
        links.innerHTML = `${gifLink}${mp4Link} | <a href="/download?job=${encodeURIComponent(job)}">下载全部ZIP</a>`;
        if (hasGIF) {
          preview.innerHTML = `<img src="/file?path=${j.result.gif_path}" style="max-width: 600px; border:1px solid #ccc;"/>`;
        } else if (hasMP4) {
          preview.innerHTML = `<video controls style="max-width: 600px; border:1px solid #ccc;"><source src="/file?path=${j.result.mp4_path}" type="video/mp4">您的浏览器不支持视频播放</video>`;
        }
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
    "subject_scale": 0.75,
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
    "use_gpu": True,
    "save_aligned": False,
    "generate_gif": False,
    "streaming_save": True,
    "ref_index": "",
    "batch_size": 50,
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
    _job_write(job_id, {"p": 0.0, "msg": "开始", "busy": True, "result": None, "error": None, "start": time.time()})
    # uploads
    upload_dir, ref_image_path = _handle_uploads(request.form, request.files)
    cfg: Dict[str, Any] = {}
    for k in DEFAULTS.keys():
        v = request.form.get(k)
        if v is None:
            if k in ("use_gpu", "save_aligned", "generate_gif", "streaming_save"):
                cfg[k] = False
                cfg[k] = None  # 空字符串转换为None
            else:
                cfg[k] = DEFAULTS[k]
        else:
            if k in ("use_gpu", "save_aligned", "generate_gif", "streaming_save"):
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
            elif k == "batch_size":
                # 处理batch_size参数
                try:
                    cfg[k] = int(v)
                except ValueError:
                    cfg[k] = DEFAULTS[k]
            else:
                cfg[k] = v
    
    # 处理参考图选项
    ref_option = request.form.get("ref_option", "auto")
    if ref_option == "internal":
        # 使用系统内部预定的参考图
        internal_ref_path = os.path.abspath(os.path.join(os.getcwd(), "ref", "ref_image.jpg"))
        if os.path.exists(internal_ref_path):
            cfg["ref_image"] = internal_ref_path
        else:
            _job_write(job_id, {"p": 0.0, "msg": "", "busy": False, "result": None, "error": "系统内部预定参考图不存在"})
            return jsonify({"err": "系统内部预定参考图不存在"}), 400
    elif ref_option == "upload" and ref_image_path:
        # 使用上传的参考图
        cfg["ref_image"] = ref_image_path
    elif ref_option == "index":
        # 使用指定索引号
        # ref_index参数已经在上面处理过了
        pass
    else:
        # 自动选择或其他情况
        cfg["ref_index"] = None
    
    if upload_dir:
        cfg["input_dir"] = upload_dir
    elif not cfg.get("input_dir"):
        _job_write(job_id, {"p": 0.0, "msg": "", "busy": False, "result": None, "error": "未提供输入目录或上传文件"})
        return jsonify({"err": "未提供输入目录或上传文件"}), 400
    t = threading.Thread(target=_start_job_file, args=(cfg, job_id), daemon=True)
    t.start()
    return jsonify({"job": job_id}), 200

@app.get("/progress")
def progress():
    job_id = request.args.get("job")
    if not job_id:
        return jsonify({"err": "missing job"}), 400
    st = _job_read(job_id) or {"p": 0.0, "msg": "就绪", "busy": False}
    p = st.get("p", 0.0)
    start = st.get("start")
    eta = None
    if st.get("busy") and start and p > 0.01 and p < 1.0:
        elapsed = time.time() - start
        eta = max(0.0, elapsed * (1.0 - p) / max(1e-6, p))
    st["eta_sec"] = eta
    resp = make_response(jsonify(st))
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return resp

@app.get("/file")
def file():
    path = request.args.get("path")
    if not path or not os.path.exists(path):
        return "Not found", 404
    return send_file(path)

@app.get("/download")
def download_zip():
    job_id = request.args.get("job")
    if not job_id:
        return "missing job", 400
    st = _job_read(job_id)
    if not st or not st.get("result"):
        return "Not ready", 404
    res = st["result"]
    gif_path = res.get("gif_path")
    mp4_path = res.get("mp4_path")
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        if gif_path and os.path.exists(gif_path):
            zf.write(gif_path, arcname=os.path.basename(gif_path))
        if mp4_path and os.path.exists(mp4_path):
            zf.write(mp4_path, arcname=os.path.basename(mp4_path))
    mem.seek(0)
    return send_file(mem, as_attachment=True, download_name='morph_output.zip', mimetype='application/zip')


def start(host: str = "127.0.0.1", port: int = 5000):
    app.run(host=host, port=port, debug=False)
