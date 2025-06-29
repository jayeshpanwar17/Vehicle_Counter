from flask import Flask, render_template, request, jsonify, url_for, redirect, Response
import sqlite3
import socket
import webbrowser
import threading
import datetime
import os # Import os for file operations
import cv2
import sys

# === NEW: Import backend functions ===
try:
    # Import the frame streaming functions from backend
    sys.path.append('.')
    import backend
except ImportError:
    print("⚠️ Warning: Could not import backend.py. Live streaming will not work.")
    backend = None

print("✅ Running the correct app.py from vehicle_counter")

app = Flask(__name__)

# === Define available locations ===
LOCATIONS = {
    "Basni Crossing": {"name": "Basni Crossing, Jodhpur"},
    "Bhagat ki kothi crossing": {"name": "Bhagat ki kothi crossing, Jodhpur"},
    "Rai ka bagh crossing": {"name": "Rai ka bagh crossing, Jodhpur"}
}

# Define the path for the file that holds the current active camera location ID
LOCATION_CONFIG_FILE = "current_camera_location.txt"

def init_database():
    """Initialize database with the new schema."""
    conn = sqlite3.connect("vehicle_data.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS vehicles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            vehicle_type TEXT NOT NULL,
            vehicle_id INTEGER,
            location_id TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()

# Helper function to write the active location to a file for backend to read
def write_current_location_to_file(location_id):
    try:
        with open(LOCATION_CONFIG_FILE, 'w') as f:
            f.write(location_id)
        print(f"📝 Frontend updated active location in '{LOCATION_CONFIG_FILE}' to: {location_id}")
    except Exception as e:
        print(f"❌ Error writing location config file: {e}")

# === NEW: Live streaming functions ===
def generate_frames():
    """Generate frames for MJPEG streaming."""
    if backend is None:
        # Return a placeholder frame if backend is not available
        placeholder = "Backend not available"
        while True:
            # Create a simple text frame
            frame = cv2.imread('placeholder.jpg') if os.path.exists('placeholder.jpg') else None
            if frame is None:
                # Create a black frame with text
                frame = cv2.zeros((480, 640, 3), dtype=cv2.uint8)
                cv2.putText(frame, placeholder, (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    else:
        while True:
            try:
                # Get the latest processed frame from backend
                frame = backend.get_latest_frame()
                
                if frame is not None:
                    # Encode frame as JPEG
                    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ret:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                else:
                    # If no frame available, create a "No Signal" frame
                    no_signal_frame = cv2.zeros((480, 640, 3), dtype=cv2.uint8)
                    cv2.putText(no_signal_frame, "No Signal - Camera Connecting...", 
                              (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    ret, buffer = cv2.imencode('.jpg', no_signal_frame)
                    if ret:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                
                # Small delay to prevent overwhelming the system
                threading.Event().wait(0.033)  # ~30 FPS
                
            except Exception as e:
                print(f"❌ Error in frame generation: {e}")
                # Create error frame
                error_frame = cv2.zeros((480, 640, 3), dtype=cv2.uint8)
                cv2.putText(error_frame, f"Error: {str(e)[:50]}", 
                          (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                ret, buffer = cv2.imencode('.jpg', error_frame)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                threading.Event().wait(1)  # Wait longer on error

# === Root route now shows the location selection page ===
@app.route("/")
def select_location():
    """Renders the page for selecting a location."""
    return render_template("selection.html", locations=LOCATIONS)

# === New route to set the active location and redirect to dashboard ===
@app.route("/set_location/<location_id>")
def set_active_location(location_id):
    if location_id not in LOCATIONS:
        return "Location not found", 404
    
    # Write the selected location to the config file
    write_current_location_to_file(location_id)

    # Redirect to the dashboard for the selected location
    return redirect(url_for('dashboard', location_id=location_id))

# === Dashboard is now location-specific ===
@app.route("/dashboard/<location_id>")
def dashboard(location_id):
    """Renders the dashboard for a specific location."""
    if location_id not in LOCATIONS:
        return "Location not found", 404

    location_name = LOCATIONS[location_id]["name"]
    return render_template("dashboard.html", location_id=location_id, location_name=location_name)

# === NEW: Live streaming routes ===
@app.route("/live-feed")
def live_feed():
    """MJPEG streaming route."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/video")
def video_page():
    """Simple HTML page to display the live video feed."""
    camera_status = "Active" if backend and backend.is_camera_active() else "Inactive"
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Live Vehicle Detection Feed</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #1a1a1a;
                color: white;
                text-align: center;
            }}
            .container {{
                max-width: 900px;
                margin: 0 auto;
            }}
            .video-container {{
                margin: 20px 0;
                border: 3px solid #333;
                border-radius: 10px;
                overflow: hidden;
                background-color: #000;
            }}
            .video-feed {{
                width: 100%;
                max-width: 640px;
                height: auto;
                display: block;
            }}
            .info-panel {{
                background-color: #2a2a2a;
                padding: 15px;
                border-radius: 8px;
                margin: 20px 0;
                text-align: left;
            }}
            .status {{
                display: inline-block;
                padding: 5px 15px;
                border-radius: 20px;
                font-weight: bold;
                margin-left: 10px;
            }}
            .status.active {{
                background-color: #28a745;
                color: white;
            }}
            .status.inactive {{
                background-color: #dc3545;
                color: white;
            }}
            .nav-links {{
                margin: 20px 0;
            }}
            .nav-links a {{
                color: #007bff;
                text-decoration: none;
                margin: 0 15px;
                padding: 10px 20px;
                border: 1px solid #007bff;
                border-radius: 5px;
                transition: all 0.3s;
            }}
            .nav-links a:hover {{
                background-color: #007bff;
                color: white;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚗 Live Vehicle Detection Feed</h1>
            
            <div class="info-panel">
                <p><strong>Camera Status:</strong> 
                   <span class="status {'active' if camera_status == 'Active' else 'inactive'}">
                       {camera_status}
                   </span>
                </p>
                <p><strong>Features:</strong></p>
                <ul>
                    <li>Real-time YOLOv8 object detection</li>
                    <li>Vehicle counting (cars, motorcycles, trucks)</li>
                    <li>Live bounding boxes and tracking IDs</li>
                    <li>Location-based processing</li>
                </ul>
            </div>

            <div class="video-container">
                <img src="/live-feed" alt="Live Vehicle Detection Feed" class="video-feed" id="videoFeed">
            </div>

            <div class="nav-links">
                <a href="/">🏠 Home</a>
                <a href="javascript:location.reload()">🔄 Refresh</a>
                <a href="javascript:toggleFullscreen()">📺 Fullscreen</a>
            </div>

            <div class="info-panel">
                <p><em>The feed shows live vehicle detection with bounding boxes, vehicle counts, and the counting line. 
                Detected vehicles crossing the red line are automatically logged to the database.</em></p>
            </div>
        </div>

        <script>
            function toggleFullscreen() {{
                const video = document.getElementById('videoFeed');
                if (video.requestFullscreen) {{
                    video.requestFullscreen();
                }} else if (video.webkitRequestFullscreen) {{
                    video.webkitRequestFullscreen();
                }} else if (video.msRequestFullscreen) {{
                    video.msRequestFullscreen();
                }}
            }}

            // Auto-refresh page if video fails to load
            document.getElementById('videoFeed').onerror = function() {{
                console.log('Video feed error, refreshing in 5 seconds...');
                setTimeout(() => location.reload(), 5000);
            }};
        </script>
    </body>
    </html>
    """
    return html_content

# === NEW: Camera status API endpoint ===
@app.route("/api/camera/status")
def camera_status_api():
    """API endpoint to check camera status."""
    if backend:
        status = {
            "active": backend.is_camera_active(),
            "message": "Camera is active" if backend.is_camera_active() else "Camera is inactive"
        }
    else:
        status = {
            "active": False,
            "message": "Backend not available"
        }
    return jsonify(status)

# === ALL EXISTING API ENDPOINTS REMAIN THE SAME ===

@app.route("/api/<location_id>/traffic/summary")
def summary_data(location_id):
    now = datetime.datetime.now()
    conn = sqlite3.connect("vehicle_data.db")
    cursor = conn.cursor()
    today = now.strftime("%Y-%m-%d")
    week_start = (now - datetime.timedelta(days=now.weekday())).strftime("%Y-%m-%d")

    summary = {"total_today": 0, "total_week": 0, "peak_hour": "00:00", "current_hour": 0}

    # Today's total
    cursor.execute("SELECT COUNT(*) FROM vehicles WHERE DATE(timestamp) = ? AND location_id = ? AND vehicle_type != 'bus'", (today, location_id))
    result = cursor.fetchone()
    summary["total_today"] = result[0] if result else 0

    # Week's total
    cursor.execute("SELECT COUNT(*) FROM vehicles WHERE DATE(timestamp) >= ? AND location_id = ? AND vehicle_type != 'bus'", (week_start, location_id))
    result = cursor.fetchone()
    summary["total_week"] = result[0] if result else 0

    # Peak hour
    cursor.execute("""
        SELECT strftime('%H', timestamp), COUNT(*) FROM vehicles
        WHERE DATE(timestamp) = ? AND location_id = ? AND vehicle_type != 'bus'
        GROUP BY strftime('%H', timestamp)
        ORDER BY COUNT(*) DESC LIMIT 1
    """, (today, location_id))
    row = cursor.fetchone()
    if row:
        summary["peak_hour"] = f"{int(row[0]):02d}:00"

    # Current hour count
    current_hour = now.strftime("%H")
    cursor.execute("""
        SELECT COUNT(*) FROM vehicles
        WHERE strftime('%H', timestamp) = ? AND DATE(timestamp) = ? AND location_id = ? AND vehicle_type != 'bus'
    """, (current_hour, today, location_id))
    result = cursor.fetchone()
    summary["current_hour"] = result[0] if result else 0

    conn.close()
    return jsonify(summary)

@app.route("/api/<location_id>/traffic/vehicle-types")
def vehicle_types_data(location_id):
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    conn = sqlite3.connect("vehicle_data.db")
    cursor = conn.cursor()

    vehicle_counts = {"car": 0, "truck": 0, "motorcycle": 0}

    cursor.execute("""
        SELECT vehicle_type, COUNT(*) FROM vehicles
        WHERE DATE(timestamp) = ? AND location_id = ? AND vehicle_type != 'bus'
        GROUP BY vehicle_type
    """, (today, location_id))

    results = cursor.fetchall()
    for vehicle_type, count in results:
        if vehicle_type in vehicle_counts:
            vehicle_counts[vehicle_type] = count

    conn.close()
    return jsonify(vehicle_counts)

@app.route("/api/<location_id>/traffic/hourly")
def hourly(location_id):
    date = request.args.get("date", datetime.datetime.now().strftime("%Y-%m-%d"))
    conn = sqlite3.connect("vehicle_data.db")
    cursor = conn.cursor()

    hourly_data = {f"{hour:02d}:00": 0 for hour in range(24)}

    cursor.execute("""
        SELECT strftime('%H', timestamp), COUNT(*) FROM vehicles
        WHERE DATE(timestamp) = ? AND location_id = ? AND vehicle_type != 'bus'
        GROUP BY strftime('%H', timestamp)
    """, (date, location_id))

    for hour, count in cursor.fetchall():
        hourly_data[f"{int(hour):02d}:00"] = count

    conn.close()
    return jsonify(hourly_data)

@app.route("/api/<location_id>/traffic/daily")
def daily(location_id):
    conn = sqlite3.connect("vehicle_data.db")
    cursor = conn.cursor()

    daily_data = {}
    for i in range(7):
        date = (datetime.datetime.now() - datetime.timedelta(days=i)).strftime("%Y-%m-%d")
        daily_data[date] = 0

    cursor.execute("""
        SELECT DATE(timestamp), COUNT(*) FROM vehicles
        WHERE DATE(timestamp) >= date('now', '-7 days') AND location_id = ? AND vehicle_type != 'bus'
        GROUP BY DATE(timestamp)
    """, (location_id,))

    for date, count in cursor.fetchall():
        if date in daily_data:
            daily_data[date] = count

    # Convert to day names for display
    day_names = {}
    sorted_dates = sorted(daily_data.keys())
    for date in sorted_dates:
        day_obj = datetime.datetime.strptime(date, "%Y-%m-%d")
        day_name = day_obj.strftime("%a") # Mon, Tue, etc.
        day_names[day_name] = daily_data[date]

    conn.close()
    return jsonify(day_names)

# Helper function to find a free port
def find_free_port():
    s = socket.socket()
    s.bind(('', 0))
    port = s.getsockname()[1]
    s.close()
    return port

# Helper function to open the browser
def open_browser(url):
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()

if __name__ == "__main__":
    init_database()
    port = find_free_port()
    url = f"http://127.0.0.1:{port}"
    print(f"\n🚀 Running Dashboard Frontend on {url}")
    print("   Please open the link above to select a location.")
    print(f"🎥 Live video feed available at: {url}/video")
    print(f"📡 Live MJPEG stream at: {url}/live-feed")

    # Initialize the location config file with a default or clear it
    if not os.path.exists(LOCATION_CONFIG_FILE):
        # Fix: Use a valid location key from LOCATIONS
        write_current_location_to_file("Basni Crossing")
    else:
        # Optionally clear or set to a known state on app startup
        pass

    open_browser(url)
    app.run(debug=True, use_reloader=False, port=port)
