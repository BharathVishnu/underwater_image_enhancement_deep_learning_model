from flask import Flask, request, jsonify
from enhancer import enhance_image
import time
import logging
import os

app = Flask(__name__)
logging.basicConfig(filename="enhancer.log", level=logging.INFO)

@app.route("/enhance", methods=["POST"])
def enhance():
    start = time.time()

    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    result_path = enhance_image(file)
    duration = round(time.time() - start, 2)

    logging.info(f"{file.filename} processed in {duration}s -> {result_path}")
    return jsonify({
        "message": "Image enhanced successfully",
        "output_path": result_path,
        "processing_time": duration
    })

@app.route("/metrics", methods=["GET"])
def metrics():
    return jsonify({
        "status": "running",
        "uptime": round(time.time() - app.start_time, 2)
    })

app.start_time = time.time()

if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    app.run(debug=True, host="0.0.0.0", port=5000)
