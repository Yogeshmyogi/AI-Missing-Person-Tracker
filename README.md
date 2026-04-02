AI Missing Person Tracker

Project Overview

AI Missing Person Tracker is a web-based application that uses face recognition technology to identify and match missing persons.
It allows users to upload images, report missing individuals, and verify matches using AI.

---

 Features

  Upload missing person images
  Face recognition using Deep Learning
  AI-powered matching system (DeepFace)
  Report & verify missing persons
  Database integration for storing records
  User-friendly web interface

---

Tech Stack

Frontend:

Streamlit / HTML / CSS

Backend:

* Python
* Django (or Flask)

AI & ML:

* DeepFace
* OpenCV

Database:

* SQLite / MongoDB (based on setup)

---

Project Structure

```id="s9k3x2"
AI-Missing-Person-Tracker/
├── find face project/
├── models/
├── database/
├── app.py / manage.py
├── README.md
```

---

Installation & Setup

1️⃣ Clone the repository

```bash id="x1p9k3"
git clone https://github.com/Yogeshmyogi/AI-Missing-Person-Tracker.git
cd AI-Missing-Person-Tracker
```


2️⃣ Install dependencies

```bash id="y8k2p4"
pip install -r requirements.txt
```


3️⃣ Run the application

For Django:

```bash id="p4k9x1"
python manage.py runserver
```

OR for Streamlit:

```bash id="m2x8k3"
streamlit run app.py
```

How It Works

1. Upload an image of a missing person
2. System extracts facial features using DeepFace
3. Compares with stored database images
4. Returns matching results

---

Known Issues

Requires good quality images for accurate result
Performance depends on dataset size
Face recognition may fail for blurred images

---

 Future Improvements

 Real-time CCTV integration 
 Mobile app version 
 Advanced AI models 
 Government database integration 

---

Author

Yogesh M
