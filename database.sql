CREATE DATABASE IF NOT EXISTS skillcrawl;
USE skillcrawl;

CREATE TABLE IF NOT EXISTS University (
  university_id INT AUTO_INCREMENT PRIMARY KEY,
  university_name VARCHAR(255) NOT NULL,
  country VARCHAR(100) NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  UNIQUE KEY uq_university (university_name, country)
);

CREATE TABLE IF NOT EXISTS DegreeProgram (
  program_id INT AUTO_INCREMENT PRIMARY KEY,
  university_id INT NOT NULL,
  degree_type ENUM('BSc','MSc','PhD','Other') NOT NULL,
  degree_titles JSON NULL,
  language VARCHAR(100),
  duration_semesters LONGTEXT,
  total_ects LONGTEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  KEY idx_program_univ_type (university_id, degree_type),
  FOREIGN KEY (university_id) REFERENCES University(university_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS Course (
  course_id INT AUTO_INCREMENT PRIMARY KEY,
  university_id INT NOT NULL,
  program_id INT NULL,
  lesson_name VARCHAR(255) NOT NULL,
  language LONGTEXT,
  website TEXT,
  semester_number LONGTEXT,
  semester_label LONGTEXT,
  ects_list JSON NULL,
  mand_opt_list JSON NULL,
  msc_bsc_list JSON NULL,
  fee_list JSON NULL,
  hours LONGTEXT,
  description LONGTEXT,
  objectives LONGTEXT,
  learning_outcomes LONGTEXT,
  course_content LONGTEXT,
  assessment LONGTEXT,
  exam LONGTEXT,
  prerequisites LONGTEXT,
  general_competences LONGTEXT,
  educational_material LONGTEXT,
  attendance_type LONGTEXT,
  professors JSON NULL,
  extras JSON NULL,
  degree_titles JSON NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP NULL DEFAULT NULL ON UPDATE CURRENT_TIMESTAMP,
  FOREIGN KEY (university_id) REFERENCES University(university_id) ON DELETE CASCADE,
  FOREIGN KEY (program_id) REFERENCES DegreeProgram(program_id) ON DELETE SET NULL,
  FULLTEXT KEY ft_course_text (description, objectives, learning_outcomes, course_content),
  KEY idx_course_univ (university_id),
  KEY idx_course_program (program_id),
  KEY idx_course_name (lesson_name)
);

CREATE TABLE IF NOT EXISTS Skill (
  skill_id INT AUTO_INCREMENT PRIMARY KEY,
  skill_name VARCHAR(255) NOT NULL,
  skill_url TEXT,
  esco_id VARCHAR(64),
  esco_level VARCHAR(32),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  UNIQUE KEY uq_skill (skill_name(191), skill_url(191))
);

CREATE TABLE IF NOT EXISTS Occupation (
    occupation_id VARCHAR(64) PRIMARY KEY,
    label         VARCHAR(255),
    parent_label  VARCHAR(255),
    top_sector    VARCHAR(255)
);

CREATE TABLE IF NOT EXISTS SkillOccupation (
    skill_id      INT NOT NULL,
    occupation_id VARCHAR(64) NOT NULL,
    PRIMARY KEY (skill_id, occupation_id),
    FOREIGN KEY (skill_id) REFERENCES Skill(skill_id),
    FOREIGN KEY (occupation_id) REFERENCES Occupation(occupation_id)
);


CREATE TABLE IF NOT EXISTS CourseSkill (
  course_id INT NOT NULL,
  skill_id INT NOT NULL,
  categories JSON NOT NULL,
  PRIMARY KEY (course_id, skill_id),
  FOREIGN KEY (course_id) REFERENCES Course(course_id) ON DELETE CASCADE,
  FOREIGN KEY (skill_id) REFERENCES Skill(skill_id) ON DELETE CASCADE,
  CHECK (JSON_VALID(categories)),
  CHECK (JSON_TYPE(categories) = 'ARRAY')
);
