# remove all containers that could have been used
docker rm -f homepage wikipedia shopping shopping_admin simple-web reddit gitlab bash flask-playwright
# remove multistep web tasks networks
docker network prune
# remove generated iptables rules
sudo iptables -F DOCKER-USER
