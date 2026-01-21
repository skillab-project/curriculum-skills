# Dockerfile για το Frontend (Nginx Server)
FROM nginx:alpine

# Ο φάκελος εργασίας του Nginx
WORKDIR /usr/share/nginx/html

# Αντιγράφουμε τα static αρχεία (HTML, CSS, JS)
COPY . /usr/share/nginx/html

# Ρύθμιση Nginx (προαιρετικά — μόνο αν έχεις default.conf)
COPY default.conf /etc/nginx/conf.d/default.conf

EXPOSE 80