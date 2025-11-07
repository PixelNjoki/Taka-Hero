import os
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_mail import Mail, Message

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

        report = WasteReport(
            name=name, email=email, phone=phone,
            location=location, description=description,
            image_filename=image_filename, image_path=image_path
        )
        db.session.add(report)
        db.session.commit()

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
    reports = WasteReport.query.order_by(WasteReport.date_reported.desc()).all()
    return render_template("admin.html", reports=reports)

# --- UPDATE STATUS ---
@app.route("/update_status/<int:report_id>/<status>")
def update_status(report_id, status):
    report = WasteReport.query.get_or_404(report_id)
    report.status = status
    db.session.commit()
    flash(f"Status updated to {status}!", "info")
    return redirect(url_for("admin"))

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
