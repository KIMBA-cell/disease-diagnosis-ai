import os
from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from dotenv import load_dotenv
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

from models import db, User

load_dotenv()

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY")
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"
login_manager.login_message = "Please log in to access this page."

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# =========================
# DISEASE DESCRIPTIONS
# =========================
disease_info = {
    "Malaria": "Malaria is a life-threatening disease caused by parasites transmitted through infected mosquito bites.",
    "Dengue": "Dengue is a viral infection spread by mosquitoes causing high fever, rash and severe joint pain.",
    "Tuberculosis": "Tuberculosis is a bacterial infection that mainly affects the lungs and spreads through air.",
    "Diabetes": "Diabetes is a chronic condition where the body cannot properly regulate blood sugar levels.",
    "Hypertension": "Hypertension is high blood pressure that can lead to heart disease and stroke if untreated.",
    "Jaundice": "Jaundice causes yellowing of skin and eyes due to excess bilirubin in the blood.",
    "Hepatitis A": "Hepatitis A is a liver infection caused by the Hepatitis A virus spread through contaminated food or water.",
    "Hepatitis B": "Hepatitis B is a serious liver infection caused by the Hepatitis B virus spread through blood or body fluids.",
    "Hepatitis C": "Hepatitis C is a viral infection causing liver inflammation spread through contaminated blood.",
    "Hepatitis D": "Hepatitis D is a liver infection that only affects people already infected with Hepatitis B.",
    "Hepatitis E": "Hepatitis E is a liver disease caused by the Hepatitis E virus spread through contaminated water.",
    "Typhoid": "Typhoid is a bacterial infection spread through contaminated food and water causing high fever.",
    "Pneumonia": "Pneumonia is an infection that inflames air sacs in the lungs which may fill with fluid.",
    "Chicken pox": "Chickenpox is a highly contagious viral infection causing itchy blister-like rash on the skin.",
    "Acne": "Acne is a skin condition that occurs when hair follicles become plugged with oil and dead skin cells.",
    "Arthritis": "Arthritis is inflammation of joints causing pain and stiffness that worsens with age.",
    "Bronchial Asthma": "Asthma is a condition in which airways narrow and swell causing breathing difficulty.",
    "Urinary tract infection": "UTI is an infection in any part of the urinary system causing burning sensation during urination.",
    "Psoriasis": "Psoriasis is a skin disease that causes red itchy scaly patches on the skin.",
    "Impetigo": "Impetigo is a highly contagious skin infection causing red sores that burst and form honey-colored crusts.",
    "Fungal infection": "A fungal infection is caused by fungi affecting skin, nails or internal organs.",
    "Allergy": "An allergy is an immune system reaction to a foreign substance that is not harmful to most people.",
    "GERD": "GERD is a digestive disorder where stomach acid flows back into the esophagus causing heartburn.",
    "Chronic cholestasis": "Chronic cholestasis is a liver condition where bile flow from the liver is reduced or blocked.",
    "Drug Reaction": "A drug reaction is an unwanted side effect caused by taking a medication.",
    "Peptic ulcer disease": "Peptic ulcers are open sores that develop on the inner lining of the stomach.",
    "AIDS": "AIDS is a chronic condition caused by HIV that damages the immune system.",
    "Dimorphic hemmorhoids(piles)": "Piles are swollen veins in the rectum or anus causing discomfort and bleeding.",
    "Heart attack": "A heart attack occurs when blood flow to the heart is blocked causing heart muscle damage.",
    "Varicose veins": "Varicose veins are twisted enlarged veins usually appearing in the legs.",
    "Hypothyroidism": "Hypothyroidism is a condition where the thyroid gland does not produce enough thyroid hormone.",
    "Hyperthyroidism": "Hyperthyroidism is a condition where the thyroid gland produces too much thyroid hormone.",
    "Hypoglycemia": "Hypoglycemia is abnormally low blood sugar levels causing shakiness dizziness and confusion.",
    "Osteoarthritis": "Osteoarthritis is a degenerative joint disease causing breakdown of cartilage in joints.",
    "Paralysis (brain hemorrhage)": "Brain hemorrhage causes bleeding in the brain leading to paralysis or stroke symptoms.",
    "Cervical spondylosis": "Cervical spondylosis is age-related wear of spinal disks in the neck causing pain and stiffness.",
    "Migraine": "Migraine is a headache disorder causing severe throbbing pain usually on one side of the head.",
    "Alcoholic hepatitis": "Alcoholic hepatitis is liver inflammation caused by excessive alcohol consumption.",
    "Common Cold": "The common cold is a viral infection of the upper respiratory tract causing runny nose and cough.",
    "Gastroenteritis": "Gastroenteritis is inflammation of the stomach and intestines causing vomiting and diarrhea.",
    "(vertigo) Paroymsal  Positional Vertigo": "Vertigo is a sensation of spinning or dizziness often caused by inner ear problems affecting balance.",
}

# =========================
# DISEASE SEVERITY
# =========================
disease_severity = {
    "Malaria": "Moderate",
    "Dengue": "Moderate",
    "Tuberculosis": "High",
    "Diabetes": "Moderate",
    "Hypertension": "Moderate",
    "Jaundice": "Moderate",
    "Hepatitis A": "Moderate",
    "Hepatitis B": "High",
    "Hepatitis C": "High",
    "Hepatitis D": "High",
    "Hepatitis E": "Moderate",
    "Typhoid": "Moderate",
    "Pneumonia": "High",
    "Chicken pox": "Mild",
    "Acne": "Mild",
    "Arthritis": "Moderate",
    "Bronchial Asthma": "Moderate",
    "Urinary tract infection": "Mild",
    "Psoriasis": "Mild",
    "Impetigo": "Mild",
    "Fungal infection": "Mild",
    "Allergy": "Mild",
    "GERD": "Mild",
    "Chronic cholestasis": "Moderate",
    "Drug Reaction": "Moderate",
    "Peptic ulcer disease": "Moderate",
    "AIDS": "High",
    "Dimorphic hemmorhoids(piles)": "Mild",
    "Heart attack": "High",
    "Varicose veins": "Mild",
    "Hypothyroidism": "Moderate",
    "Hyperthyroidism": "Moderate",
    "Hypoglycemia": "Moderate",
    "Osteoarthritis": "Mild",
    "Paralysis (brain hemorrhage)": "High",
    "Cervical spondylosis": "Mild",
    "Migraine": "Mild",
    "Alcoholic hepatitis": "High",
    "Common Cold": "Mild",
    "Gastroenteritis": "Mild",
    "(vertigo) Paroymsal  Positional Vertigo": "Mild",
}

# =========================
# PRECAUTION TIPS
# =========================
disease_precautions = {
    "Malaria": "Sleep under an insecticide-treated net, use mosquito repellent, and seek treatment promptly if fever persists.",
    "Dengue": "Avoid mosquito bites, stay hydrated, rest, and monitor for warning signs like severe abdominal pain or bleeding.",
    "Tuberculosis": "Complete the full course of prescribed medication, cover your mouth when coughing, and ensure good room ventilation.",
    "Diabetes": "Monitor blood sugar regularly, maintain a balanced diet low in refined sugar, and stay physically active.",
    "Hypertension": "Reduce salt intake, manage stress, exercise regularly, and monitor blood pressure routinely.",
    "Jaundice": "Rest adequately, stay hydrated, avoid alcohol, and follow up with a doctor to identify the underlying cause.",
    "Hepatitis A": "Practice good hand hygiene, avoid contaminated food or water, and rest until symptoms resolve.",
    "Hepatitis B": "Avoid sharing needles or personal items that may carry blood, and seek antiviral treatment as advised by a doctor.",
    "Hepatitis C": "Avoid alcohol, do not share sharp personal items, and seek antiviral treatment from a specialist.",
    "Hepatitis D": "Seek immediate care as it worsens Hepatitis B; avoid alcohol and follow your doctor's treatment plan closely.",
    "Hepatitis E": "Drink only clean, boiled or treated water, maintain hygiene, and rest until fully recovered.",
    "Typhoid": "Drink only clean water, avoid street food during outbreaks, and complete the full antibiotic course if prescribed.",
    "Pneumonia": "Rest, stay hydrated, avoid smoking or smoky environments, and seek prompt medical care for breathing difficulty.",
    "Chicken pox": "Avoid scratching the rash, keep skin clean, isolate from others until fully healed, and rest.",
    "Acne": "Keep skin clean, avoid touching or picking at the affected area, and avoid heavy oil-based cosmetics.",
    "Arthritis": "Stay physically active with low-impact exercise, maintain a healthy weight, and apply heat or cold as needed for pain relief.",
    "Bronchial Asthma": "Avoid known triggers such as dust and smoke, keep prescribed inhalers accessible, and seek care during severe attacks.",
    "Urinary tract infection": "Drink plenty of water, urinate frequently, maintain good hygiene, and complete any prescribed antibiotics.",
    "Psoriasis": "Moisturize skin regularly, avoid known triggers such as stress, and avoid excessive scratching.",
    "Impetigo": "Keep the affected area clean and covered, avoid sharing towels, and wash hands frequently to prevent spreading it.",
    "Fungal infection": "Keep the affected area clean and dry, avoid sharing personal items, and use antifungal treatment as advised.",
    "Allergy": "Identify and avoid known allergy triggers, and keep antihistamines available if prescribed.",
    "GERD": "Avoid large meals and lying down right after eating, reduce spicy or fatty foods, and elevate the head while sleeping.",
    "Chronic cholestasis": "Follow up with a liver specialist, avoid alcohol, and monitor for worsening jaundice or itching.",
    "Drug Reaction": "Stop the suspected medication and seek medical attention promptly; always inform doctors of known drug allergies.",
    "Peptic ulcer disease": "Avoid spicy foods, alcohol, and NSAIDs; eat smaller frequent meals and manage stress.",
    "AIDS": "Follow antiretroviral therapy consistently, practice safe sex, and attend regular medical check-ups.",
    "Dimorphic hemmorhoids(piles)": "Increase fiber and water intake, avoid straining during bowel movements, and stay physically active.",
    "Heart attack": "Seek emergency medical care immediately; do not delay, as prompt treatment greatly improves outcomes.",
    "Varicose veins": "Avoid standing for long periods, elevate the legs when resting, and consider compression stockings.",
    "Hypothyroidism": "Take prescribed thyroid medication consistently and attend regular follow-up blood tests.",
    "Hyperthyroidism": "Follow prescribed treatment closely, avoid excess iodine intake, and attend regular monitoring.",
    "Hypoglycemia": "Carry a fast-acting sugar source at all times and eat regular, balanced meals to prevent drops in blood sugar.",
    "Osteoarthritis": "Maintain a healthy weight, engage in gentle regular exercise, and avoid repetitive joint strain.",
    "Paralysis (brain hemorrhage)": "Seek emergency medical care immediately, as rapid treatment significantly affects recovery outcomes.",
    "Cervical spondylosis": "Maintain good posture, avoid prolonged neck strain, and do gentle neck stretching exercises.",
    "Migraine": "Identify and avoid personal triggers, rest in a dark quiet room during attacks, and stay hydrated.",
    "Alcoholic hepatitis": "Stop alcohol consumption completely and seek medical supervision for liver recovery.",
    "Common Cold": "Rest, stay hydrated, and allow the illness to run its course; seek care if symptoms worsen or persist.",
    "Gastroenteritis": "Stay hydrated with oral rehydration solutions, eat bland foods, and practice good hand hygiene.",
    "(vertigo) Paroymsal  Positional Vertigo": "Avoid sudden head movements, sit or lie down when dizzy, and consult a doctor for balance exercises.",
}

# =========================
# LABEL CORRECTIONS
# =========================
label_corrections = {
    "hepatitis A": "Hepatitis A",
    "Osteoarthristis": "Osteoarthritis",
    "Peptic ulcer diseae": "Peptic ulcer disease",
}

# =========================
# TRAIN MODEL ON STARTUP
# =========================
print("Loading dataset and training model...")

train_data = pd.read_csv("final_clean_dataset.csv")
test_data = pd.read_csv("test_dataset.csv")

X_train = train_data.drop("Disease", axis=1)
y_train = train_data["Disease"]

symptom_names = list(X_train.columns)

X_test = test_data[symptom_names]
y_test = test_data["Disease"]

dt_model = DecisionTreeClassifier(random_state=42)
rf_model = RandomForestClassifier(random_state=42)
nb_model = GaussianNB()

dt_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
nb_model.fit(X_train, y_train)

model = VotingClassifier(
    estimators=[
        ('dt', dt_model),
        ('rf', rf_model),
        ('nb', nb_model)
    ],
    voting='soft'
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)

print(f"✅ Model trained! Accuracy: {accuracy}%")

# =========================
# CREATE DATABASE TABLES
# =========================
with app.app_context():
    db.create_all()

# =========================
# HOMEPAGE (PUBLIC)
# =========================
@app.route("/")
def home():
    return render_template("home.html")

# =========================
# REGISTER ROUTE
# =========================
@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("predict"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")
        consent = request.form.get("consent")

        if not username or not email or not password:
            flash("All fields are required.", "error")
            return redirect(url_for("register"))

        if password != confirm_password:
            flash("Passwords do not match.", "error")
            return redirect(url_for("register"))

        if not consent:
            flash("You must accept the consent statement to register.", "error")
            return redirect(url_for("register"))

        if User.query.filter_by(username=username).first():
            flash("Username already taken.", "error")
            return redirect(url_for("register"))

        if User.query.filter_by(email=email).first():
            flash("Email or phone number already registered.", "error")
            return redirect(url_for("register"))

        new_user = User(username=username, email=email)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()

        flash("Registration successful! Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")

# =========================
# LOGIN ROUTE
# =========================
@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("predict"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        user = User.query.filter_by(username=username).first()

        if user and user.check_password(password):
            login_user(user)
            flash(f"Welcome back, {user.username}!", "success")
            return redirect(url_for("predict"))

        flash("Invalid username or password.", "error")
        return redirect(url_for("login"))

    return render_template("login.html")

# =========================
# LOGOUT ROUTE
# =========================
@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "success")
    return redirect(url_for("home"))

# =========================
# PREDICTION ROUTE (PROTECTED) — SYMPTOM FORM ONLY
# =========================
@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    if request.method == "POST":
        user_input = []
        for symptom in symptom_names:
            value = request.form.get(symptom, "No")
            user_input.append(1 if value == "Yes" else 0)

        if sum(user_input) == 0:
            flash("Please select at least one symptom before predicting.", "error")
            return redirect(url_for("predict"))

        input_data = pd.DataFrame([user_input], columns=symptom_names)

        raw_prediction = str(model.predict(input_data)[0]).strip()
        prediction = label_corrections.get(raw_prediction, raw_prediction)

        probabilities = model.predict_proba(input_data)[0]
        confidence = round(max(probabilities) * 100, 1)

        description = disease_info.get(prediction,
            "This disease was predicted by the AI model. Please consult a doctor for more information.")
        severity = disease_severity.get(prediction, "Moderate")
        precaution = disease_precautions.get(prediction,
            "Please consult a qualified medical doctor for appropriate guidance.")

        # Store the result in the session so it can be displayed on a separate page
        session["last_result"] = {
            "prediction": prediction,
            "description": description,
            "severity": severity,
            "confidence": confidence,
            "precaution": precaution,
        }

        return redirect(url_for("result"))

    return render_template("predict.html", symptoms=symptom_names, accuracy=accuracy)

# =========================
# RESULT ROUTE (PROTECTED) — SEPARATE RESULT PAGE
# =========================
@app.route("/result")
@login_required
def result():
    result_data = session.get("last_result")

    if not result_data:
        flash("No prediction found. Please select your symptoms first.", "error")
        return redirect(url_for("predict"))

    return render_template("result.html",
                           prediction=result_data["prediction"],
                           description=result_data["description"],
                           severity=result_data["severity"],
                           confidence=result_data["confidence"],
                           precaution=result_data["precaution"],
                           accuracy=accuracy)

# =========================
# ABOUT ROUTE
# =========================
@app.route("/about")
def about():
    return render_template("about.html", accuracy=accuracy)

# =========================
# RUN APP
# =========================
if __name__ == "__main__":
    app.run(debug=True)