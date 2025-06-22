from flask import Flask, render_template, request, jsonify, url_for, redirect
import sqlite3
import socket
import webbrowser
import threading
import datetime
import os # Import os for file operations

print("✅ Running the correct app.py from vehicle_counter")

app = Flask(__name__)

# === Define available locations ===
LOCATIONS = {
    "Kudi": {"name": "Kudi, Jodhpur"},
    "Shastri Nagar": {"name": "Shastri Nagar, Jodhpur"},
    "Ratanada": {"name": "Ratanada, Jodhpur"}
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

# === ALL API ENDPOINTS REMAIN THE SAME, using location_id from URL ===

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

    # Initialize the location config file with a default or clear it
    if not os.path.exists(LOCATION_CONFIG_FILE):
        write_current_location_to_file(LOCATIONS["Ratanada"]["name"]) # Or any default
    else:
        # Optionally clear or set to a known state on app startup
        pass

    open_browser(url)
    app.run(debug=True, use_reloader=False, port=port)
