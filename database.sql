CREATE DATABASE IF NOT EXISTS SkillCrawl;
USE SkillCrawl;


CREATE TABLE IF NOT EXISTS University (
    university_id INT AUTO_INCREMENT PRIMARY KEY,
    university_name VARCHAR(255) NOT NULL,
    country VARCHAR(100) NOT NULL,
    number_of_semesters INT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS Lessons (
    lesson_id INT AUTO_INCREMENT PRIMARY KEY,
    lesson_name VARCHAR(255) NOT NULL,
    semester VARCHAR(255) NOT NULL,
    description TEXT,
    msc_bsc VARCHAR(10),
    degree VARCHAR(255),
    university_id INT,
    FOREIGN KEY (university_id) REFERENCES University(university_id) ON DELETE CASCADE
);


CREATE TABLE IF NOT EXISTS Skills (
    skill_id INT AUTO_INCREMENT PRIMARY KEY,
    skill_name VARCHAR(255),
    skill_url TEXT NOT NULL,
    lesson_id INT,
    FOREIGN KEY (lesson_id) REFERENCES Lessons(lesson_id) ON DELETE CASCADE
);
