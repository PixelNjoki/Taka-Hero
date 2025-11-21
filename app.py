import os
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_mail import Mail, Message
#from waste_predictor import create_predictor
from model_components import WastePriorityPredictor

# --- SETUP APP ---
app = Flask(__name__)
app.secret_key = "supersecretkey"

# --- DATABASE CONFIG ---
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///taka_hero.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# --- UPLOAD CONFIG ---
app.config["UPLOAD_FOLDER"] = "static/uploads"

# --- EMAIL CONFIG (set these later if needed) ---
app.config["MAIL_SERVER"] = "smtp.gmail.com"
app.config["MAIL_PORT"] = 587
app.config["MAIL_USE_TLS"] = True
app.config["MAIL_USERNAME"] = "whitneynjoki8@gmail.com"  
app.config["MAIL_PASSWORD"] = "Mm15Life2"          
app.config["MAIL_DEFAULT_SENDER"] = "your_email@gmail.com"

db = SQLAlchemy(app)
mail = Mail(app)

# --- AI PRIORITY PREDICTOR ---
try:
    predictor = WastePriorityPredictor(model_path='models/waste_priority_model.pth')
    #predictor = create_predictor('models/waste_priority_model.pth')
    print("✅ AI Priority Classifier loaded successfully!")
except Exception as e:
    print(f"⚠️ Warning: Could not load AI model: {e}")
    predictor = None

# --- DATABASE MODEL ---
class WasteReport(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    email = db.Column(db.String(100))
    phone = db.Column(db.String(20))
    location = db.Column(db.String(200))
    description = db.Column(db.Text)
    image_filename = db.Column(db.String(200))
    image_path = db.Column(db.String(300))
    status = db.Column(db.String(20), default="Pending")
    date_reported = db.Column(db.DateTime, default=datetime.utcnow)
    
    # AI-generated fields
    priority = db.Column(db.String(20), default="Secondary")  # Critical, Urgent, Important, Secondary
    priority_color = db.Column(db.String(20), default="blue")
    waste_type = db.Column(db.String(50), default="general")
    confidence = db.Column(db.Float, default=0.0)
    ai_solutions = db.Column(db.Text)  # JSON string of solution suggestions

# --- HOME PAGE ---
@app.route("/")
def index():
    return render_template("index.html")

# --- REPORT PAGE ---
@app.route("/report", methods=["GET", "POST"])
def report():
    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        phone = request.form["phone"]
        location = request.form["location"]
        description = request.form["description"]
        image = request.files["image"]

        if image:
            image_filename = image.filename
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], image_filename)
            image.save(image_path)
        else:
            image_filename = None
            image_path = None

        # Create initial report
        report = WasteReport(
            name=name, email=email, phone=phone,
            location=location, description=description,
            image_filename=image_filename, image_path=image_path
        )
        db.session.add(report)
        db.session.commit()
        
        # --- AI PRIORITY CLASSIFICATION ---
        if predictor and image_path:
            try:
                prediction = predictor.predict(image_path, description)
                report.priority = prediction['priority']
                report.priority_color = prediction['color']
                report.waste_type = prediction['waste_type']
                report.confidence = prediction['confidence']
                report.ai_solutions = ','.join(prediction['solutions'])
                report.status = 'Pending'  # Auto-set to Pending after AI analysis
                db.session.commit()
                print(f"✅ AI classified report as {prediction['priority']} priority")
            except Exception as e:
                print(f"⚠️ AI classification failed: {e}")
                # Continue without AI classification

        # --- EMAIL ALERT ---
        try:
            msg = Message(
                "Waste Report Submitted",
                recipients=[email],
                body=f"Hello {name}, your waste report has been received! We'll review it soon."
            )
            mail.send(msg)
        except Exception as e:
            print("Email failed:", e)

        flash("✅ Report submitted successfully!", "success")
        return redirect(url_for("success"))
    return render_template("report.html")

# --- SUCCESS PAGE ---
@app.route("/success")
def success():
    return render_template("success.html")

# --- ADMIN DASHBOARD ---
@app.route("/admin")
def admin():
    status_filter = request.args.get('status', 'all')

    priority_order = {
        'Critical': 1,
        'Urgent': 2,
        'Important': 3,
        'Secondary': 4
    }

    # Always fetch all reports for the chart
    all_reports = WasteReport.query.all()

    # Filter list based on dropdown
    if status_filter == 'Collected':
        reports = WasteReport.query.filter_by(status='Collected').order_by(WasteReport.date_reported.desc()).all()
    elif status_filter == 'all':
        reports = WasteReport.query.filter(WasteReport.status != 'Collected').all()
        reports = sorted(reports, key=lambda r: (priority_order.get(r.priority, 5), -r.date_reported.timestamp()))
    else:
        reports = WasteReport.query.filter_by(status=status_filter).all()
        reports = sorted(reports, key=lambda r: (priority_order.get(r.priority, 5), -r.date_reported.timestamp()))

    return render_template("admin.html",
                           reports=reports,
                           all_reports=all_reports,
                           selected_status=status_filter)

# --- UPDATE STATUS ---
@app.route("/update_status/<int:report_id>/<status>")
def update_status(report_id, status):
    report = WasteReport.query.get_or_404(report_id)
    report.status = status
    
    # If marked as resolved, remove from active list (or you can delete)
    if status == "Resolved":
        # Option 1: Delete the report
        # db.session.delete(report)
        # Option 2: Keep it but mark as resolved (current implementation)
        db.session.commit()
        flash(f"Report #{report_id} marked as resolved!", "success")
    else:
        db.session.commit()
        flash(f"Status updated to {status}!", "info")
    
    return redirect(url_for("admin"))

# --- GET REPORT DETAILS (API for AJAX) ---
@app.route("/api/report/<int:report_id>")
def get_report_details(report_id):
    """API endpoint to get full report details including AI suggestions"""
    report = WasteReport.query.get_or_404(report_id)
    
    solutions = []
    if report.ai_solutions:
        solutions = report.ai_solutions.split(',')
    
    return jsonify({
        'id': report.id,
        'name': report.name,
        'email': report.email,
        'phone': report.phone,
        'location': report.location,
        'description': report.description,
        'priority': report.priority,
        'priority_color': report.priority_color,
        'waste_type': report.waste_type,
        'confidence': report.confidence,
        'solutions': solutions,
        'status': report.status,
        'date_reported': report.date_reported.strftime('%Y-%m-%d %H:%M:%S'),
        'image_url': url_for('static', filename=f'uploads/{report.image_filename}') if report.image_filename else None
    })

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
