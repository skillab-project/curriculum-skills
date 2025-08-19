rebuild-fastapi:
	docker compose build --no-cache fastapi-service
	docker compose up -d fastapi-service

rebuild-mysql:
	docker compose down -v
	docker compose build mysql-curriculum-skill
	docker compose up -d mysql-curriculum-skill

rebuild-all:
	docker compose down -v
	docker compose build --no-cache
	docker compose up -d
