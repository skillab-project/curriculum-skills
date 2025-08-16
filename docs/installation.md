# 🧰 Installation Options
You can run SkillCrawl in two ways:

- 🐳 With Docker Compose – fast, portable, and no local Python setup needed.

- ⚙️ Manually with Python – useful for development or testing without containers.

The sections below walk you through both options.

## 🐳 Install & Run with Docker Compose

### 📁 Prerequisites
- Docker & Docker Compose

### ⚙️ Step-by-Step
1. Clone the repo:
```bash
git clone https://github.com/skillab-project/curriculum-skills.git
cd curriculum-skills
```

2. Create a `.env` file:
```env
APP_NAME=skillcrawl-api
DOCKER_REG=
DOCKER_REPO=
DOCKER_TAG=latest

MYSQL_ROOT_PASSWORD=root
MYSQL_USER=skilluser
MYSQL_PASSWORD=skillpass
MYSQL_DATABASE=SkillCrawl
```

3. Run the containers:
```bash
docker-compose --env-file .env up --build
```

4. Open your browser:
```
http://localhost:5002/docs
```
You'll see the Swagger UI with access to all endpoints.

---

## 🧪 Using Without Docker (Manual Mode)

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Launch the API:
```bash
uvicorn main:app --reload
```

Visit:
```
http://127.0.0.1:8000/docs
```

---

## 📄 Database Setup

The MySQL schema is automatically loaded from `skillcrawl.sql` when the database container is initialized.

It contains tables for:
- `University`
- `Lessons`
- `Skills`

DB runs on:
```
localhost:5003
```

Use credentials from `.env` to connect.

---

## 📄 Database Setup

### 🗃️ Default MySQL Setup (Docker/Production)

If you're running the app with Docker, the MySQL schema is **automatically loaded** from `skillcrawl.sql` during initialization. This creates the required tables:

- `University`
- `Lessons`
- `Skills`

Your app is configured to connect to MySQL running at:

```
localhost:5003
```

Make sure to use the credentials defined in your `.env` file to connect (e.g., host, port, username, password, and database name).

---

### 🖥️ Alternative: Setup Using XAMPP (Local Windows Installation)

If you're not using Docker and prefer a GUI-based setup using **XAMPP**, follow these steps:

#### ✅ Step-by-Step: Importing `skillcrawl.sql` via XAMPP

1. **Download and Install XAMPP**
   - Go to [https://www.apachefriends.org/index.html](https://www.apachefriends.org/index.html) and download XAMPP for your OS.
   - Install it and launch the **XAMPP Control Panel**.

2. **Start MySQL and Apache**
   - In the XAMPP Control Panel, click **Start** next to:
     - **Apache** (to enable phpMyAdmin via browser)
     - **MySQL** (your database engine)

3. **Open phpMyAdmin**
   - Click the **"Admin"** button next to MySQL.
   - This opens **phpMyAdmin** in your browser: [http://localhost/phpmyadmin](http://localhost/phpmyadmin)

4. **Create a New Database**
   - In phpMyAdmin, go to the **"Databases"** tab.
   - Enter a name for your database (e.g., `skillcrawl`) and click **Create**.

5. **Import the `skillcrawl.sql` File**
   - With your new database selected (left sidebar), go to the **"Import"** tab.
   - Click **"Choose File"** and select `skillcrawl.sql` from your project directory.
   - Click **"Go"** to run the import.
   - This will create the following tables:
     - `University`
     - `Lessons`
     - `Skills`

6. **(Optional) View & Query the Tables**
   - Click on each table from the left panel to browse the data.
   - Use the **"SQL"** tab to run queries like:
     ```sql
        -- List all universities
        SELECT * FROM University;

        -- Get all lessons in semester 1
        SELECT * FROM Lessons WHERE semester = 'Semester 1';

        -- Find all skills related to "Machine Learning"
        SELECT * FROM Skills WHERE skill_name LIKE '%Machine Learning%';
     ```


---

## 🧠 API Endpoints (via Swagger UI)

Key endpoints:
- `/process_pdf`: Upload and process a PDF
- `/search_skill`: Search for courses that teach a skill
- `/filter_skillnames`: Get skill names for a lesson
- `/calculate_skillnames`: Enrich or calculate from scratch the corresponding skill names to the URLs.
- `/get_top_skills`: Get top N skills per university
- `/get_top_skills_all`: Get top skills across all universities
- `/get_universities_by_skills`: Returns university names and their corresponding courses, for universities that contain every skill in the provided list.
