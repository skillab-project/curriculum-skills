-- db/init/10_big_seed.sql
-- Large realistic seed for recommender_test1
-- Safe to run only on a fresh volume (mounted to /docker-entrypoint-initdb.d).
-- Assumes tables are created by SQLAlchemy on app startup.

CREATE DATABASE IF NOT EXISTS recommender_test1;
USE recommender_test1;

-- =========================
-- Universities
-- =========================
INSERT INTO University (university_id, university_name, country) VALUES
  (1, 'Aristotle University of Thessaloniki', 'Greece'),
  (2, 'National and Kapodistrian University of Athens', 'Greece'),
  (3, 'University of Patras', 'Greece'),
  (4, 'Technical University of Munich', 'Germany'),
  (5, 'Politecnico di Milano', 'Italy')
ON DUPLICATE KEY UPDATE university_name=VALUES(university_name), country=VALUES(country);

-- =========================
-- Degree Programs
-- degree_titles uses JSON array
-- =========================
INSERT INTO DegreeProgram (program_id, university_id, degree_type, degree_titles, language, duration_semesters, total_ects) VALUES
  (11, 1, 'BSc', JSON_ARRAY('Computer Science'), 'English', '8', '240'),
  (12, 1, 'MSc', JSON_ARRAY('Data Science'), 'English', '4', '120'),
  (21, 2, 'BSc', JSON_ARRAY('Informatics'), 'Greek', '8', '240'),
  (22, 2, 'MSc', JSON_ARRAY('Artificial Intelligence'), 'English', '4', '120'),
  (31, 3, 'BSc', JSON_ARRAY('Electrical & Computer Engineering'), 'Greek', '10', '300'),
  (32, 3, 'MSc', JSON_ARRAY('Software Engineering'), 'English', '4', '120'),
  (41, 4, 'BSc', JSON_ARRAY('Informatics'), 'English', '6', '180'),
  (42, 4, 'MSc', JSON_ARRAY('Robotics, Cognition, Intelligence'), 'English', '4', '120'),
  (51, 5, 'BSc', JSON_ARRAY('Computer Engineering'), 'English', '6', '180'),
  (52, 5, 'MSc', JSON_ARRAY('Machine Learning & AI'), 'English', '4', '120')
ON DUPLICATE KEY UPDATE degree_type=VALUES(degree_type), degree_titles=VALUES(degree_titles);

-- =========================
-- Courses
-- =========================
INSERT INTO Course (course_id, university_id, program_id, lesson_name, language, description, hours) VALUES
  -- AUT BSc
  (1001, 1, 11, 'Programming I', 'English', 'Intro to programming in Python', '4'),
  (1002, 1, 11, 'Programming II', 'English', 'OOP, data structures in Java', '4'),
  (1003, 1, 11, 'Data Structures', 'English', 'Arrays, linked lists, trees, graphs', '5'),
  (1004, 1, 11, 'Databases', 'English', 'Relational models, SQL, normalization', '5'),
  (1005, 1, 11, 'Operating Systems', 'English', 'Processes, threads, memory, file systems', '5'),
  (1006, 1, 11, 'Computer Networks', 'English', 'TCP/IP, routing, congestion control', '5'),
  -- AUT MSc
  (1011, 1, 12, 'Machine Learning', 'English', 'Supervised & unsupervised methods', '6'),
  (1012, 1, 12, 'Deep Learning', 'English', 'CNNs, RNNs, transformers', '6'),
  (1013, 1, 12, 'Big Data Systems', 'English', 'Spark, Hadoop, data lakes', '6'),
  (1014, 1, 12, 'Data Mining', 'English', 'Pattern discovery, clustering', '5'),
  (1015, 1, 12, 'MLOps', 'English', 'Deployment, monitoring, CI/CD for ML', '5'),

  -- NKUA BSc
  (2001, 2, 21, 'Discrete Mathematics', 'Greek', 'Logic, sets, combinatorics', '4'),
  (2002, 2, 21, 'Algorithms', 'Greek', 'Design & analysis of algorithms', '5'),
  (2003, 2, 21, 'Computer Architecture', 'Greek', 'CPU, memory, pipelines', '4'),
  (2004, 2, 21, 'Software Engineering', 'Greek', 'Requirements, design patterns, testing', '5'),
  (2005, 2, 21, 'Web Technologies', 'Greek', 'HTML5, CSS, JS, REST', '4'),
  (2006, 2, 21, 'Information Retrieval', 'Greek', 'Indexing, ranking, IR models', '4'),
  -- NKUA MSc
  (2011, 2, 22, 'Natural Language Processing', 'English', 'Text processing, embeddings, transformers', '6'),
  (2012, 2, 22, 'Reinforcement Learning', 'English', 'MDPs, Q-Learning, Policy Gradients', '6'),
  (2013, 2, 22, 'Computer Vision', 'English', 'Image processing, CNNs', '6'),
  (2014, 2, 22, 'Cloud Computing', 'English', 'AWS/GCP, containers, orchestration', '5'),
  (2015, 2, 22, 'Graph Mining', 'English', 'Graph algorithms, GNN basics', '5'),

  -- UoP BSc
  (3001, 3, 31, 'Signals & Systems', 'Greek', 'LTI systems, Fourier, sampling', '5'),
  (3002, 3, 31, 'Digital Logic', 'Greek', 'Combinational & sequential logic', '4'),
  (3003, 3, 31, 'Embedded Systems', 'Greek', 'MCUs, RTOS, peripherals', '5'),
  (3004, 3, 31, 'Control Systems', 'Greek', 'Feedback, stability, PID', '5'),
  (3005, 3, 31, 'Communication Systems', 'Greek', 'Modulation, coding, channels', '5'),
  (3006, 3, 31, 'Numerical Methods', 'Greek', 'Linear algebra, ODE/PDE approximations', '4'),
  -- UoP MSc
  (3011, 3, 32, 'Agile Software Development', 'English', 'Scrum, XP, CI/CD', '5'),
  (3012, 3, 32, 'Advanced Databases', 'English', 'NoSQL, NewSQL, transactions', '5'),
  (3013, 3, 32, 'Microservices Architecture', 'English', 'Design, resilience, observability', '6'),
  (3014, 3, 32, 'Distributed Systems', 'English', 'Consensus, consistency, CAP', '6'),
  (3015, 3, 32, 'Software Quality & Testing', 'English', 'Unit/integration testing, TDD', '5'),

  -- TUM BSc
  (4001, 4, 41, 'Linear Algebra', 'English', 'Vectors, matrices, eigenvalues', '5'),
  (4002, 4, 41, 'Probability & Statistics', 'English', 'Distributions, estimation, inference', '5'),
  (4003, 4, 41, 'Functional Programming', 'English', 'Haskell, immutability, recursion', '5'),
  (4004, 4, 41, 'Compilers', 'English', 'Parsing, IR, optimization', '6'),
  (4005, 4, 41, 'Parallel Computing', 'English', 'Threads, MPI, GPU', '6'),
  (4006, 4, 41, 'Human-Computer Interaction', 'English', 'UX, usability, prototyping', '5'),
  -- TUM MSc
  (4011, 4, 42, 'Robotics I', 'English', 'Kinematics, dynamics, control', '6'),
  (4012, 4, 42, 'Robotics II', 'English', 'Perception, SLAM, planning', '6'),
  (4013, 4, 42, 'Perception for Robotics', 'English', 'Vision, LiDAR, sensor fusion', '6'),
  (4014, 4, 42, 'Autonomous Systems', 'English', 'Decision making, safety', '6'),
  (4015, 4, 42, 'AI for Robotics', 'English', 'Deep RL, policy learning', '6'),

  -- PoliMi BSc
  (5001, 5, 51, 'Computer Systems', 'English', 'OS + architecture overview', '5'),
  (5002, 5, 51, 'Database Systems', 'English', 'SQL, design, transactions', '5'),
  (5003, 5, 51, 'Networks & Security', 'English', 'Network layers, security basics', '5'),
  (5004, 5, 51, 'Programming Paradigms', 'English', 'Procedural, object-oriented, functional', '5'),
  (5005, 5, 51, 'Software Project', 'English', 'Team project, DevOps basics', '6'),
  (5006, 5, 51, 'Numerical Analysis', 'English', 'Approximation, optimization', '5'),
  -- PoliMi MSc
  (5011, 5, 52, 'Deep Learning Systems', 'English', 'Training at scale, inference', '6'),
  (5012, 5, 52, 'ML Security', 'English', 'Adversarial ML, robustness', '5'),
  (5013, 5, 52, 'Recommender Systems', 'English', 'CF, content-based, hybrid', '6'),
  (5014, 5, 52, 'Information Retrieval Systems', 'English', 'Indexing at scale, ranking', '6'),
  (5015, 5, 52, 'Data Engineering', 'English', 'ETL, warehousing, pipelines', '6')
ON DUPLICATE KEY UPDATE lesson_name=VALUES(lesson_name);

-- =========================
-- Skills
-- =========================
INSERT INTO Skill (skill_id, skill_name, skill_url, esco_id, esco_level) VALUES
  (1, 'Python', 'https://esco.example/python', 'ESCO:PY', '4'),
  (2, 'Java', 'https://esco.example/java', 'ESCO:JAVA', '4'),
  (3, 'C++', 'https://esco.example/cpp', 'ESCO:CPP', '4'),
  (4, 'Data Structures', 'https://esco.example/data-structures', 'ESCO:DS', '5'),
  (5, 'Algorithms', 'https://esco.example/algorithms', 'ESCO:ALG', '5'),
  (6, 'SQL', 'https://esco.example/sql', 'ESCO:SQL', '4'),
  (7, 'NoSQL', 'https://esco.example/nosql', 'ESCO:NOSQL', '4'),
  (8, 'Operating Systems', 'https://esco.example/os', 'ESCO:OS', '5'),
  (9, 'Computer Networks', 'https://esco.example/networks', 'ESCO:NET', '4'),
  (10, 'Machine Learning', 'https://esco.example/ml', 'ESCO:ML', '6'),
  (11, 'Deep Learning', 'https://esco.example/dl', 'ESCO:DL', '6'),
  (12, 'NLP', 'https://esco.example/nlp', 'ESCO:NLP', '6'),
  (13, 'Computer Vision', 'https://esco.example/cv', 'ESCO:CV', '6'),
  (14, 'Distributed Systems', 'https://esco.example/distributed', 'ESCO:DSYS', '5'),
  (15, 'Cloud Computing', 'https://esco.example/cloud', 'ESCO:CLOUD', '5'),
  (16, 'MLOps', 'https://esco.example/mlops', 'ESCO:MLOPS', '5'),
  (17, 'Reinforcement Learning', 'https://esco.example/rl', 'ESCO:RL', '7'),
  (18, 'Robotics', 'https://esco.example/robotics', 'ESCO:ROB', '6'),
  (19, 'Information Retrieval', 'https://esco.example/ir', 'ESCO:IR', '5'),
  (20, 'Security', 'https://esco.example/security', 'ESCO:SEC', '5')
ON DUPLICATE KEY UPDATE skill_name=VALUES(skill_name);

-- =========================
-- Occupations (optional, for completeness)
-- =========================
INSERT INTO Occupation (occupation_id, occupation_name, occupation_url, esco_code) VALUES
  ('O-001', 'Data Scientist', 'https://esco.example/occupations/data-scientist', '1234.1'),
  ('O-002', 'Software Engineer', 'https://esco.example/occupations/software-engineer', '2512.0'),
  ('O-003', 'ML Engineer', 'https://esco.example/occupations/ml-engineer', '2511.5'),
  ('O-004', 'Network Engineer', 'https://esco.example/occupations/network-engineer', '2522.3'),
  ('O-005', 'Robotics Engineer', 'https://esco.example/occupations/robotics-engineer', '2144.7')
ON DUPLICATE KEY UPDATE occupation_name=VALUES(occupation_name);

-- =========================
-- Course ↔ Skill links (JSON categories must be valid arrays)
-- =========================
INSERT INTO CourseSkill (course_id, skill_id, categories) VALUES
  -- AUT BSc
  (1001, 1, JSON_ARRAY('programming', 'python basics')),
  (1002, 2, JSON_ARRAY('programming', 'oop')),
  (1003, 4, JSON_ARRAY('data structures', 'theory')),
  (1003, 5, JSON_ARRAY('algorithms', 'theory')),
  (1004, 6, JSON_ARRAY('sql', 'databases')),
  (1005, 8, JSON_ARRAY('os', 'systems')),
  (1006, 9, JSON_ARRAY('networks', 'systems')),
  -- AUT MSc
  (1011, 10, JSON_ARRAY('ml', 'supervised')),
  (1012, 11, JSON_ARRAY('dl', 'neural networks')),
  (1013, 14, JSON_ARRAY('big data', 'distributed')),
  (1013, 15, JSON_ARRAY('cloud', 'infra')),
  (1014, 19, JSON_ARRAY('mining', 'unsupervised')),
  (1015, 16, JSON_ARRAY('mlops', 'devops')),

  -- NKUA BSc
  (2001, 5, JSON_ARRAY('math', 'discrete')),
  (2002, 5, JSON_ARRAY('algorithms', 'analysis')),
  (2003, 3, JSON_ARRAY('architecture', 'low-level')),
  (2004, 14, JSON_ARRAY('dev process', 'systems')),
  (2005, 19, JSON_ARRAY('web', 'ir')),
  (2006, 19, JSON_ARRAY('ir', 'search')),
  -- NKUA MSc
  (2011, 12, JSON_ARRAY('nlp', 'transformers')),
  (2012, 17, JSON_ARRAY('rl', 'control')),
  (2013, 13, JSON_ARRAY('cv', 'vision')),
  (2014, 15, JSON_ARRAY('cloud', 'infra')),
  (2015, 14, JSON_ARRAY('graphs', 'mining')),

  -- UoP BSc
  (3001, 14, JSON_ARRAY('signals', 'systems')),
  (3002, 3, JSON_ARRAY('logic', 'hardware')),
  (3003, 3, JSON_ARRAY('embedded', 'rtos')),
  (3004, 14, JSON_ARRAY('control', 'systems')),
  (3005, 9, JSON_ARRAY('communications', 'networks')),
  (3006, 10, JSON_ARRAY('numerical', 'ml basics')),
  -- UoP MSc
  (3011, 14, JSON_ARRAY('agile', 'process')),
  (3012, 7, JSON_ARRAY('nosql', 'advanced db')),
  (3013, 14, JSON_ARRAY('microservices', 'distributed')),
  (3014, 14, JSON_ARRAY('distributed', 'consensus')),
  (3015, 14, JSON_ARRAY('quality', 'testing')),

  -- TUM BSc
  (4001, 10, JSON_ARRAY('math', 'linear algebra')),
  (4002, 10, JSON_ARRAY('math', 'statistics')),
  (4003, 3, JSON_ARRAY('fp', 'haskell')),
  (4004, 5, JSON_ARRAY('compilers', 'optimization')),
  (4005, 14, JSON_ARRAY('parallel', 'gpu')),
  (4006, 20, JSON_ARRAY('hci', 'ux')),
  -- TUM MSc
  (4011, 18, JSON_ARRAY('robotics', 'kinematics')),
  (4012, 18, JSON_ARRAY('robotics', 'perception')),
  (4013, 13, JSON_ARRAY('perception', 'sensor fusion')),
  (4014, 18, JSON_ARRAY('autonomy', 'planning')),
  (4015, 10, JSON_ARRAY('ai', 'rl')),

  -- PoliMi BSc
  (5001, 8, JSON_ARRAY('systems', 'os')),
  (5002, 6, JSON_ARRAY('databases', 'sql')),
  (5003, 20, JSON_ARRAY('security', 'networks')),
  (5004, 1, JSON_ARRAY('paradigms', 'programming')),
  (5005, 14, JSON_ARRAY('project', 'devops')),
  (5006, 10, JSON_ARRAY('numerical', 'ml basics')),
  -- PoliMi MSc
  (5011, 11, JSON_ARRAY('deep learning', 'systems')),
  (5012, 20, JSON_ARRAY('security', 'adversarial ml')),
  (5013, 19, JSON_ARRAY('recommenders', 'ir')),
  (5014, 19, JSON_ARRAY('ir', 'retrieval')),
  (5015, 15, JSON_ARRAY('data eng', 'pipelines'))
ON DUPLICATE KEY UPDATE categories=VALUES(categories);

-- =========================
-- Skill ↔ Occupation (coarser mapping)
-- =========================
INSERT INTO SkillOccupation (skill_id, occupation_id) VALUES
  (10, 'O-001'), (11, 'O-001'), (12, 'O-001'), (19, 'O-001'),
  (1, 'O-002'), (2, 'O-002'), (4, 'O-002'), (5, 'O-002'), (14, 'O-002'),
  (10, 'O-003'), (11, 'O-003'), (16, 'O-003'),
  (9, 'O-004'), (20, 'O-004'),
  (18, 'O-005'), (13, 'O-005')
ON DUPLICATE KEY UPDATE occupation_id=VALUES(occupation_id);