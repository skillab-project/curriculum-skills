-- -- connection/session defaults (optional but handy) ---------------------------
SET NAMES utf8mb4 COLLATE utf8mb4_unicode_ci;
SET SESSION sql_mode = 'STRICT_TRANS_TABLES,ERROR_FOR_DIVISION_BY_ZERO,NO_ENGINE_SUBSTITUTION';

-- -- database ------------------------------------------------------------------
CREATE DATABASE IF NOT EXISTS skillcrawl
  /*!40100 DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci */;
USE skillcrawl;

-- -- University ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS University (
  university_id   INT AUTO_INCREMENT PRIMARY KEY,
  university_name VARCHAR(255) NOT NULL,
  country         VARCHAR(100) NOT NULL,
  created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  UNIQUE KEY uq_university (university_name, country),
  KEY idx_university_name (university_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- -- DegreeProgram -------------------------------------------------------------
-- NOTE: duration_semesters & total_ects are TEXTUAL now.
CREATE TABLE IF NOT EXISTS DegreeProgram (
  program_id         INT AUTO_INCREMENT PRIMARY KEY,
  university_id      INT NOT NULL,
  degree_type        ENUM('BSc','MSc','PhD','Other') NOT NULL,
  degree_titles      JSON NULL,
  language           LONGTEXT,
  duration_semesters LONGTEXT,  -- was INT
  total_ects         LONGTEXT,  -- was INT
  created_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  KEY idx_program_univ_type (university_id, degree_type),
  CONSTRAINT fk_program_univ
    FOREIGN KEY (university_id) REFERENCES University(university_id) ON DELETE CASCADE,
  CHECK (degree_titles IS NULL OR JSON_VALID(degree_titles))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- -- Course --------------------------------------------------------------------
-- NOTE: hours already VARCHAR; semester_* textual; ects stored as JSON list
CREATE TABLE IF NOT EXISTS Course (
  course_id            INT AUTO_INCREMENT PRIMARY KEY,
  university_id        INT NOT NULL,
  program_id           INT NULL,
  lesson_name          VARCHAR(255) NOT NULL,
  language             LONGTEXT,
  website              TEXT,
  semester_number      LONGTEXT,
  semester_label       LONGTEXT,
  ects_list            JSON NULL,
  mand_opt_list        JSON NULL,
  msc_bsc_list         JSON NULL,
  fee_list             JSON NULL,
  hours                LONGTEXT,
  description          LONGTEXT,
  objectives           LONGTEXT,
  learning_outcomes    LONGTEXT,
  course_content       LONGTEXT,
  assessment           LONGTEXT,
  exam                 LONGTEXT,
  prerequisites        LONGTEXT,
  general_competences  LONGTEXT,
  educational_material LONGTEXT,
  attendance_type      LONGTEXT,
  professors           JSON NULL,
  extras               JSON NULL,
  degree_titles        JSON NULL,
  created_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at           TIMESTAMP NULL DEFAULT NULL ON UPDATE CURRENT_TIMESTAMP,
  CONSTRAINT fk_course_univ
    FOREIGN KEY (university_id) REFERENCES University(university_id) ON DELETE CASCADE,
  CONSTRAINT fk_course_program
    FOREIGN KEY (program_id) REFERENCES DegreeProgram(program_id) ON DELETE SET NULL,
  FULLTEXT KEY ft_course_text (description, objectives, learning_outcomes, course_content),
  KEY idx_course_univ (university_id),
  KEY idx_course_program (program_id),
  KEY idx_course_name (lesson_name),
  CHECK (ects_list IS NULL OR JSON_VALID(ects_list)),
  CHECK (mand_opt_list IS NULL OR JSON_VALID(mand_opt_list)),
  CHECK (msc_bsc_list IS NULL OR JSON_VALID(msc_bsc_list)),
  CHECK (fee_list IS NULL OR JSON_VALID(fee_list)),
  CHECK (professors IS NULL OR JSON_VALID(professors)),
  CHECK (extras IS NULL OR JSON_VALID(extras)),
  CHECK (degree_titles IS NULL OR JSON_VALID(degree_titles))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- -- Skill ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS Skill (
  skill_id    INT AUTO_INCREMENT PRIMARY KEY,
  skill_name  VARCHAR(255) NOT NULL,
  skill_url   TEXT,
  esco_id     VARCHAR(64),
  esco_level  VARCHAR(32),
  created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  UNIQUE KEY uq_skill (skill_name(191), skill_url(191))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- -- CourseSkill ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS CourseSkill (
  course_id  INT NOT NULL,
  skill_id   INT NOT NULL,
  categories JSON NOT NULL,
  PRIMARY KEY (course_id, skill_id),
  CONSTRAINT fk_cs_course FOREIGN KEY (course_id) REFERENCES Course(course_id) ON DELETE CASCADE,
  CONSTRAINT fk_cs_skill  FOREIGN KEY (skill_id)  REFERENCES Skill(skill_id)  ON DELETE CASCADE,
  CHECK (JSON_VALID(categories)),
  CHECK (JSON_TYPE(categories) = 'ARRAY')
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
