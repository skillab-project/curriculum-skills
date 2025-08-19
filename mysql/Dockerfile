FROM mysql:8.0


COPY sql_init/backup.tar.gz /tmp/backup.tar.gz

RUN mkdir -p /docker-entrypoint-initdb.d \
 && tar -xzf /tmp/backup.tar.gz -C /docker-entrypoint-initdb.d --strip-components=1 \
 && rm /tmp/backup.tar.gz