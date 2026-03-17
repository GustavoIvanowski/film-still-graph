import os
import uuid
import shutil
import threading
import time
import json
from flask import Flask, request, render_template, Response, stream_with_context, jsonify

from pipeline import run_pipeline

app = Flask(__name__)

UPLOAD_FOLDER  = "uploads"
SESSION_FOLDER = "sessions"
SESSION_TTL    = 60 * 60 * 2  # 2 hours

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SESSION_FOLDER, exist_ok=True)

# ── session state ─────────────────────────────────────────────────────────────
# { session_id: { "progress": 0, "message": "", "done": False, "error": None } }

sessions = {}

# ── cleanup thread ────────────────────────────────────────────────────────────

def cleanup_loop():
    while True:
        time.sleep(60 * 10)  # check every 10 mins
        now = time.time()
        for name in os.listdir(SESSION_FOLDER):
            path = os.path.join(SESSION_FOLDER, name)
            if os.path.isdir(path):
                age = now - os.path.getmtime(path)
                if age > SESSION_TTL:
                    shutil.rmtree(path, ignore_errors=True)
                    sessions.pop(name, None)
                    print(f"Cleaned up session {name}")

threading.Thread(target=cleanup_loop, daemon=True).start()

# ── routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "zip" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f = request.files["zip"]
    if not f.filename.endswith(".zip"):
        return jsonify({"error": "Please upload a .zip file"}), 400

    session_id  = str(uuid.uuid4())
    session_dir = os.path.join(SESSION_FOLDER, session_id)
    os.makedirs(session_dir, exist_ok=True)

    zip_path = os.path.join(UPLOAD_FOLDER, f"{session_id}.zip")
    f.save(zip_path)

    sessions[session_id] = {
        "progress": 0,
        "message":  "Starting...",
        "done":     False,
        "error":    None
    }

    def run():
        def progress(pct, msg):
            sessions[session_id]["progress"] = pct
            sessions[session_id]["message"]  = msg
            print(f"[{session_id[:8]}] {pct}% - {msg}")

        try:
            run_pipeline(zip_path, session_dir, progress)
            sessions[session_id]["done"] = True
        except Exception as e:
            sessions[session_id]["error"] = str(e)
            print(f"Pipeline error: {e}")
        finally:
            os.remove(zip_path)  # clean up uploaded zip

    threading.Thread(target=run, daemon=True).start()

    return jsonify({"session_id": session_id})


@app.route("/progress/<session_id>")
def progress(session_id):
    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 404

    def stream():
        while True:
            state = sessions.get(session_id)
            if not state:
                yield f"data: {json.dumps({'error': 'Session expired'})}\n\n"
                break

            payload = {
                "progress": state["progress"],
                "message":  state["message"],
                "done":     state["done"],
                "error":    state["error"]
            }
            yield f"data: {json.dumps(payload)}\n\n"

            if state["done"] or state["error"]:
                break

            time.sleep(0.5)

    return Response(
        stream_with_context(stream()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"  # important for nginx
        }
    )


@app.route("/graph/<session_id>")
def graph(session_id):
    state = sessions.get(session_id)
    if not state or not state["done"]:
        return "Session not found or not ready", 404
    return render_template("graph.html", session_id=session_id)


@app.route("/sessions/<session_id>/force_graph.json")
def serve_graph_json(session_id):
    from flask import send_from_directory
    return send_from_directory(
        os.path.join(SESSION_FOLDER, session_id),
        "force_graph.json"
    )


@app.route("/sessions/<session_id>/cached_images/<path:filename>")
def serve_image(session_id, filename):
    from flask import send_from_directory
    return send_from_directory(
        os.path.join(SESSION_FOLDER, session_id, "cached_images"),
        filename
    )


if __name__ == "__main__":
    app.run(debug=True)